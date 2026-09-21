"""The reference backend: plain torch, no e3nn, and the correctness oracle.

Mandatory and CPU-capable. Every other backend is checked against this one, so
it is written for clarity over speed: the symmetric contraction expands the
outer power rather than fusing anything, and the linear map builds a dense
matrix from its flat weights on every call. Both are wasteful and both are
obviously right, which is the trade a reference is for.

It holds the canonical weight layout directly, so ``to_canonical`` and
``load_canonical`` are views rather than conversions. That is the property that
makes one checkpoint loadable by any backend: the reference defines the format
by holding it.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    full_symmetric_tensor_product_basis,
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.canonical import (
    KERNEL_SPEC_VERSION,
    contraction_path_labels,
)
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    RadialBasisDescriptor,
    SegmentReduceDescriptor,
    SphericalHarmonicsDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.protocol import DISPATCHED_OPS, REFERENCE_ONLY_OPS
from torch import Tensor, nn

from mace_torch.backends.reference.spherical_harmonics import spherical_harmonics
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
    equivariant_linear,
    segment_sum,
    symmetric_contraction,
)
from mace_torch.nn.radial import (
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
)

__all__ = ["ReferenceBackend"]

_TORCH_DTYPE = {"float64": torch.float64, "float32": torch.float32}


def _linear_plan(descriptor: LinearDescriptor):
    """Which weight writes to which entry of the dense matrix.

    One weight per matching multiplicity pair, repeated over the ``2l+1``
    components of its irrep. That repetition is the equivariance: a free matrix
    would mix components of one irrep into another.
    """
    source_irreps = Irreps.parse(descriptor.irreps_in)
    target_irreps = Irreps.parse(descriptor.irreps_out)
    rows, columns, sources = [], [], []
    weight = 0
    for out_slice, out_ir in target_irreps.slices():
        for in_slice, in_ir in source_irreps.slices():
            if in_ir != out_ir:
                continue
            for component in range(in_ir.dimension):
                rows.append(out_slice.start + component)
                columns.append(in_slice.start + component)
                sources.append(weight)
            weight += 1
    bias_rows = []
    if descriptor.has_bias:
        for out_slice, out_ir in target_irreps.slices():
            if out_ir.degree == 0 and out_ir.parity == 1:
                bias_rows.append(out_slice.start)
    return rows, columns, sources, weight, bias_rows


class ReferenceLinear(nn.Module):
    """An equivariant linear map with first-class bias."""

    def __init__(self, descriptor: LinearDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        rows, columns, sources, count, bias_rows = _linear_plan(descriptor)
        dtype = _TORCH_DTYPE[descriptor.precision]
        self.dim_out = Irreps.parse(descriptor.irreps_out).dimension
        self.register_buffer("row", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("column", torch.tensor(columns, dtype=torch.long))
        self.register_buffer("source", torch.tensor(sources, dtype=torch.long))
        self.register_buffer("bias_row", torch.tensor(bias_rows, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(count, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(len(bias_rows), dtype=dtype))

    def forward(self, features: Tensor) -> Tensor:
        return equivariant_linear(
            features,
            self.weight,
            self.row,
            self.column,
            self.source,
            self.bias,
            self.bias_row,
            self.dim_out,
        )

    def to_canonical(self) -> dict[str, Tensor]:
        """A view. The reference holds the canonical layout already."""
        return {"weight": self.weight.detach(), "bias": self.bias.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])
            self.bias.copy_(state["bias"])


class _ConstantTensors(nn.Module):
    """A device-following list of constant tables.

    Buffers move with the module and parameters do not fit, since these carry
    no gradient. A list of plain tensors would go stale the first time the
    model is moved to a device, so they are held as buffers and iterated in
    the order they were given.
    """

    def __init__(self, tensors: list[Tensor]) -> None:
        super().__init__()
        self.count = len(tensors)
        for position, tensor in enumerate(tensors):
            self.register_buffer(f"table_{position}", tensor, persistent=False)

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(getattr(self, f"table_{p}") for p in range(self.count))


class ReferenceSymmetricContraction(nn.Module):
    """The many-body contraction, over the basis the descriptor records.

    One contraction per output irrep, concatenated on the component axis. They
    cannot share a stacked basis: each output irrep has its own component count,
    so stacking them would be joining arrays whose second axis differs. The
    frozen tree reaches the same shape by holding one `Contraction` per output
    irrep, and this is that, with the loop kept explicit.
    """

    def __init__(self, descriptor: SymmetricContractionDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = _TORCH_DTYPE[descriptor.precision]
        build = (
            reduced_symmetric_tensor_product_basis
            if descriptor.basis == "reduced"
            else full_symmetric_tensor_product_basis
        )
        self.orders = descriptor.correlation
        self.targets = [str(ir) for _, ir in Irreps.parse(descriptor.irreps_out)]
        weights, bases = [], []
        for target in self.targets:
            group, tables = [], []
            for order in range(1, descriptor.correlation + 1):
                array = build(descriptor.irreps_in, order, target)[target]
                flat = array.reshape(array.shape[0], array.shape[1], -1)
                tables.append(torch.tensor(flat, dtype=dtype))
                group.append(
                    nn.Parameter(
                        torch.zeros(
                            descriptor.num_elements,
                            flat.shape[0],
                            descriptor.num_features,
                            dtype=dtype,
                        )
                    )
                )
            weights.extend(group)
            bases.append(_ConstantTensors(tables))
        self.weights = nn.ParameterList(weights)
        self.bases = nn.ModuleList(bases)

    def _group(self, position: int) -> list[Tensor]:
        """One output irrep's weights, by integer index.

        Flat storage with integer indexing rather than a slice of the
        `ParameterList`: slicing one goes through `slice.indices`, a C builtin
        that `torch.compile` cannot trace, and the break lands in the middle of
        the backbone rather than here.
        """
        base = position * self.orders
        return [self.weights[base + order] for order in range(self.orders)]

    def forward(self, features: Tensor, element: Tensor) -> Tensor:
        pieces = [
            symmetric_contraction(
                features, self._group(position), list(tables), element
            )
            for position, tables in enumerate(self.bases)
        ]
        return torch.cat(pieces, dim=-1)

    def canonical_metadata(self) -> dict[str, object]:
        """What each weight on the path axis multiplies, by name.

        The path count alone does not pin the layout: two implementations can
        agree on how many paths there are and keep a different subset of the
        linearly dependent coupling trees. Writing the trees is what turns that
        into a refused load rather than a model that runs on the wrong basis.
        """
        return {
            "spec_version": KERNEL_SPEC_VERSION,
            "basis": self.descriptor.basis,
            "paths": list(
                contraction_path_labels(
                    self.descriptor.irreps_in,
                    self.descriptor.irreps_out,
                    self.descriptor.correlation,
                    basis=self.descriptor.basis,
                )
            ),
        }

    def to_canonical(self) -> dict[str, Tensor]:
        """The flat ``[Z, A, mul]`` array, joined over irreps and body orders.

        The per-piece tensors are contiguous slices of it in the pinned order,
        so this is a concatenate rather than a conversion.
        """
        return {
            "weight": torch.cat([w.detach() for w in self.weights], dim=1),
            "path_counts": torch.tensor([w.shape[1] for w in self.weights]),
        }

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        counts = [int(n) for n in state["path_counts"]]
        pieces = torch.split(state["weight"], counts, dim=1)
        with torch.no_grad():
            for parameter, piece in zip(self.weights, pieces, strict=True):
                parameter.copy_(piece)


def _coupling_coefficients(descriptor: ChannelwiseTPConvDescriptor) -> np.ndarray:
    """The Clebsch-Gordan coefficients of the message-passing product.

    One block per path, in the order
    :func:`~mace_core.kernels.paths.channelwise_paths` pins, written into a
    dense ``[paths, dim_paths, dim_node, dim_edge]`` array.

    The output axis spans the **paths**, not the target irreps. Two couplings
    landing on the same irrep keep separate slices, because the linear that
    follows mixes them and can send one copy where the other does not go.
    Summing them here would be a smaller model wearing the same shape, and it
    would not match a trained artifact.

    Dense because this is the reference; a backend with a real kernel keeps
    them sparse.
    """
    node = Irreps.parse(descriptor.irreps_node)
    edge = Irreps.parse(descriptor.irreps_edge)
    paths = channelwise_paths(
        descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
    )
    node_slices = list(node.slices())
    edge_slices = list(edge.slices())
    width = sum(path.irrep.dimension for path in paths)
    if not paths:
        return np.zeros((0, 0, node.dimension, edge.dimension))

    blocks = []
    offset = 0
    for path in paths:
        block = np.zeros((width, node.dimension, edge.dimension))
        in_slice, in_ir = node_slices[path.node_term]
        edge_slice, edge_ir = edge_slices[path.edge_term]
        # Scaled so each path's output has unit variance when its inputs do.
        # The coupling table has unit norm, so the factor is the square root of
        # the output's dimension: a coupling spreading over five components
        # would otherwise contribute a fifth of the variance of one landing on a
        # single scalar, and the linear after the convolution would see paths
        # whose sizes differ by more than a factor of two.
        block[offset : offset + path.irrep.dimension, in_slice, edge_slice] = (
            wigner_3j_real(path.irrep.degree, in_ir.degree, edge_ir.degree)
            * np.sqrt(path.irrep.dimension)
        )
        blocks.append(block)
        offset += path.irrep.dimension
    return np.stack(blocks)


class ReferenceChannelwiseTPConv(nn.Module):
    """The message-passing tensor product. Node-level, always."""

    def __init__(self, descriptor: ChannelwiseTPConvDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        dtype = _TORCH_DTYPE[descriptor.precision]
        coefficients = _coupling_coefficients(descriptor)
        self.register_buffer(
            "coefficients", torch.tensor(coefficients, dtype=dtype), persistent=False
        )

    @property
    def num_paths(self) -> int:
        """How many weights the external radial MLP has to produce."""
        return int(self.coefficients.shape[0])

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        return channelwise_tp_conv(
            node_features,
            edge_attributes,
            radial_weights,
            self.coefficients,
            sender,
            receiver,
            num_nodes,
        )


class ReferenceFullyConnectedTP(nn.Module):
    """The skip connection's tensor product against the element attributes.

    Only the case the models use is built: the second input is scalars, the
    element one-hots, so the product is a per-element linear map. The general
    case raises rather than returning something plausible, because a wrong skip
    connection is a wrong model that still trains.
    """

    def __init__(self, descriptor: FullyConnectedTPDescriptor) -> None:
        super().__init__()
        second = Irreps.parse(descriptor.irreps_in2)
        if any(ir.degree != 0 or ir.parity != 1 for _, ir in second):
            raise NotImplementedError(
                f"the reference backend builds the fully connected tensor "
                f"product only against scalars, and {descriptor.irreps_in2!r} "
                f"is not. That is the case the models use, for the element "
                f"attributes; a general second input needs a backend that "
                f"declares it."
            )
        self.descriptor = descriptor
        dtype = _TORCH_DTYPE[descriptor.precision]
        rows, columns, sources, count, _ = _linear_plan(
            LinearDescriptor(
                irreps_in=descriptor.irreps_in1,
                irreps_out=descriptor.irreps_out,
                precision=descriptor.precision,
            )
        )
        self.num_scalars = second.dimension
        self.dim_out = Irreps.parse(descriptor.irreps_out).dimension
        self.register_buffer("row", torch.tensor(rows, dtype=torch.long))
        self.register_buffer("column", torch.tensor(columns, dtype=torch.long))
        self.register_buffer("source", torch.tensor(sources, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(self.num_scalars, count, dtype=dtype))

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        empty = features.new_zeros(0)
        empty_rows = self.row.new_zeros(0)
        total = None
        for scalar in range(self.num_scalars):
            mapped = equivariant_linear(
                features,
                self.weight[scalar],
                self.row,
                self.column,
                self.source,
                empty,
                empty_rows,
                self.dim_out,
            )
            scaled = mapped * attributes[:, scalar : scalar + 1]
            total = scaled if total is None else total + scaled
        return total

    def to_canonical(self) -> dict[str, Tensor]:
        return {"weight": self.weight.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.weight.copy_(state["weight"])


class ReferenceSegmentReduce(nn.Module):
    """A reduction into segments. Sum only, which is what the models use."""

    def __init__(self, descriptor: SegmentReduceDescriptor) -> None:
        super().__init__()
        if descriptor.reduction != "sum":
            raise NotImplementedError(
                f"the reference backend reduces by sum, not by "
                f"{descriptor.reduction!r}."
            )
        self.descriptor = descriptor

    def forward(self, values: Tensor, index: Tensor, num_segments: int) -> Tensor:
        return segment_sum(values, index, num_segments)


class ReferenceSphericalHarmonics(nn.Module):
    """Real spherical harmonics in this project's convention."""

    def __init__(self, descriptor: SphericalHarmonicsDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    def forward(self, directions: Tensor) -> Tensor:
        return spherical_harmonics(
            directions, self.descriptor.lmax, self.descriptor.normalize
        )


#: The basis each declared kind builds. ARCH-1 owns these classes; this is only
#: the lookup from the descriptor's name to one of them, so there is no second
#: implementation of a basis anywhere.
_BASES = {
    "bessel": BesselBasis,
    "chebyshev": ChebyshevBasis,
    "gaussian": GaussianBasis,
}


class ReferenceRadialBasis(nn.Module):
    """The radial embedding, with the cutoff envelope already applied."""

    def __init__(self, descriptor: RadialBasisDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor
        if descriptor.kind not in _BASES:
            raise ValueError(
                f"{descriptor.kind!r} is not a radial basis this backend "
                f"builds. The kinds are {sorted(_BASES)}."
            )
        # ARCH-1's Chebyshev takes no r_max, and rightly: the frozen tree
        # stored one and never used it, so the polynomials run past the unit
        # interval into the divergent branch. Passing it would be inventing a
        # parameter the basis does not have.
        if descriptor.kind == "chebyshev":
            self.basis: nn.Module = ChebyshevBasis(num_basis=descriptor.num_basis)
        elif descriptor.kind == "gaussian":
            self.basis = GaussianBasis(
                r_max=descriptor.cutoff, num_basis=descriptor.num_basis
            )
        else:
            self.basis = BesselBasis(
                r_max=descriptor.cutoff, num_basis=descriptor.num_basis
            )
        self.cutoff = PolynomialCutoff(
            r_max=descriptor.cutoff, polynomial_order=descriptor.cutoff_order
        )

    def forward(self, lengths: Tensor) -> Tensor:
        return self.basis(lengths) * self.cutoff(lengths)


class ReferenceBackend:
    """Plain torch, every op, no e3nn. The oracle the others are checked against."""

    name = "reference"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            ops=DISPATCHED_OPS | REFERENCE_ONLY_OPS,
            devices=frozenset({"cpu", "cuda"}),
            dtypes=frozenset({"float64", "float32"}),
            max_lmax=0,
            layouts=frozenset({"mul_ir"}),
            bases=frozenset({"reduced", "full"}),
            supports_double_backward=True,
        )

    def _check(self, descriptor) -> None:
        self.capabilities().require(descriptor, self.name)

    def make_linear(self, descriptor: LinearDescriptor) -> ReferenceLinear:
        self._check(descriptor)
        return ReferenceLinear(descriptor)

    def make_channelwise_tp_conv(
        self, descriptor: ChannelwiseTPConvDescriptor
    ) -> ReferenceChannelwiseTPConv:
        self._check(descriptor)
        return ReferenceChannelwiseTPConv(descriptor)

    def make_symmetric_contraction(
        self, descriptor: SymmetricContractionDescriptor
    ) -> ReferenceSymmetricContraction:
        self._check(descriptor)
        return ReferenceSymmetricContraction(descriptor)

    def make_fully_connected_tp(
        self, descriptor: FullyConnectedTPDescriptor
    ) -> ReferenceFullyConnectedTP:
        self._check(descriptor)
        return ReferenceFullyConnectedTP(descriptor)

    def make_segment_reduce(
        self, descriptor: SegmentReduceDescriptor
    ) -> ReferenceSegmentReduce:
        self._check(descriptor)
        return ReferenceSegmentReduce(descriptor)

    def make_spherical_harmonics(
        self, descriptor: SphericalHarmonicsDescriptor
    ) -> ReferenceSphericalHarmonics:
        self._check(descriptor)
        return ReferenceSphericalHarmonics(descriptor)

    def make_radial_basis(
        self, descriptor: RadialBasisDescriptor
    ) -> ReferenceRadialBasis:
        self._check(descriptor)
        return ReferenceRadialBasis(descriptor)

    def make_interaction_layer(self, descriptors) -> None:
        """The reference fuses nothing. ``None`` means op by op, which is the
        normal answer and not a failure."""
        return None
