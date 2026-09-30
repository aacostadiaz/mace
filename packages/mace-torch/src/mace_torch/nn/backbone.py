"""The backbone: node features in, node features out.

The first of the two layers the architecture is split into, and it is a block
rather than a model. A model is the thing that carries observables, is built
from one config object and returns the typed output; this returns the learned
descriptor those are computed from, so it lives beside the other blocks and
not in ``models/``. The model that holds one arrives with the output layer.

It is a learned descriptor and nothing else: **no readouts, no heads, no scale
shift, and no gradient calls**. Forces and stress are taken by a derivative
engine *around* this, never inside it, which is what keeps a compiled graph
whole.

Four things this does not do, each replacing something the frozen tree does:

* It does not resolve a backend in ``forward``. Every op is built once from a
  descriptor and held.
* It does not branch on whether a fusion attribute happens to be set.
* It does not carry LAMMPS partitioning through its blocks. The locality seam
  is one explicit hook, absent by default, rather than an attribute smuggled
  into the block signatures behind a scripting probe.
* It does not write into the graph it is given. The dict it receives is read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    LinearDescriptor,
    RadialBasisDescriptor,
    RadialKind,
    SphericalHarmonicsDescriptor,
)
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec
from torch import Tensor, nn

from mace_torch.backends.layout import layout_of
from mace_torch.nn.interaction import (
    DEFAULT_RADIAL_HIDDEN,
    InteractionBlock,
    NonLinearInteractionBlock,
    ResidualInteractionBlock,
)
from mace_torch.nn.layout import expanded_irreps
from mace_torch.nn.node_inputs import NodeInputEmbedding
from mace_torch.nn.product_basis import EquivariantProductBasisBlock
from mace_torch.nn.radial import AgnesiTransform, PolynomialCutoff, SoftTransform

__all__ = ["MACEBackbone"]


def _scalars_of(irreps: str) -> str:
    """The even scalars of a whole declaration, as one term."""
    count = sum(
        mul
        for mul, ir in Irreps.parse(irreps).terms
        if ir.degree == 0 and ir.parity == 1
    )
    return f"{count}x0e"


def _per_channel(irreps: str) -> tuple[str, int]:
    """A whole declaration as one channel's and the channel count.

    Raises:
        ValueError: If its terms do not share one multiplicity, which is what
            a convolution with one radial weight per channel and path needs.
    """
    terms = Irreps.parse(irreps).terms
    multiplicities = {mul for mul, _ in terms}
    if len(multiplicities) != 1:
        raise ValueError(
            f"convolution_irreps is {irreps!r}; every term needs the same "
            f"multiplicity, such as '128x0e+128x1o'."
        )
    return "+".join(str(ir) for _, ir in terms), multiplicities.pop()


class MACEBackbone(nn.Module):
    """The equivariant message-passing backbone.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending. A node's element index is
            its position here.
        num_layers: How many interaction and product pairs.
        num_features: The channel width.
        lmax: The highest degree of the edge attributes.
        hidden_irreps: One channel's node-feature irreps.
        num_radial: Width of the radial embedding.
        cutoff: In Angstrom.
        correlation: The body order of the product basis.
        avg_num_neighbors: The density normalization.
        radial_kind: Which radial basis.
        cutoff_order: The order of the envelope that takes it to zero at the
            cutoff. It comes from the model's cutoff setting, not from the
            basis.
        distance_transform: ``"agnesi"`` or ``"soft"`` evaluates the radial
            basis on that transform of each length, scaled by the pair's
            covalent radii, and the cutoff envelope on the length itself.
            ``"none"`` by default.
        apply_cutoff: Multiply the cutoff envelope into the radial basis, the
            default. Off, every interaction multiplies it into the output of
            its radial networks instead.
        radial_hidden: The hidden widths of every interaction's radial
            network.
        precision: The dtype name every op is built at.
        node_inputs: Declared per-node input streams to mix into the features
            before the first layer. Nothing about them is special-cased: each
            is read by name and brought in by an equivariant map from its own
            declared irreps.
        locality: An optional hook taking the node features and the graph and
            returning the features to carry into the next layer. This is where
            a domain-decomposed run slices off its ghost nodes. Absent by
            default, and when absent the forward is the hook-free path exactly.
        residual_first_layer: Build the first layer as every later one is,
            with the skip taken from its input, the element embedding, and
            added by the product basis. The frozen tree does so when its first
            interaction is ``RealAgnosticResidualInteractionBlock``, its default
            and what MACE-MP-0 was trained with; otherwise the first layer's
            skip is applied to the message and replaces it.
        learned_density_first_layer: Normalize the first layer's messages by a
            density learned per atom rather than by the average neighbour
            count, as the frozen tree's density blocks do.
        learned_density: The same for every later layer.
        nonlinear_first_layer: Build the first layer as a
            :class:`NonLinearInteractionBlock`: a convolution conditioned on
            both elements, normalized by a learned density and gated. It
            carries its skip to the product basis.
        nonlinear: The same for every later layer.
        convolution_irreps: What a nonlinear layer convolves, as a whole
            declaration with one multiplicity for every term, such as
            ``"128x0e+128x1o"``. ``None`` convolves the node features at their
            own width.
        narrow_first_convolution: Convolve only the scalars of
            ``convolution_irreps`` in the first layer, at its multiplicity.
        last_layer_irreps: One channel's irreps in the last layer's features.
            Only its scalars by default, since an energy reads nothing else
            off it. A model that reads more keeps more: a charge-aware model
            keeps every irrep, and a dipole-only model keeps only the vectors.
        element_agnostic_product: Share the product basis's weights across
            elements rather than holding one set per element.
        edge_axes: The order the edge vector's components are handed to the
            spherical harmonics. The identity by default. A charge-aware model
            hands them as ``(y, z, x)``, ``(1, 2, 0)``, which puts its degree
            one features in the long-range solver's component order.
    """

    def __init__(
        self,
        backend,
        atomic_numbers: Sequence[int],
        num_layers: int = 2,
        num_features: int = 8,
        lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        cutoff: float = 5.0,
        correlation: int = 3,
        avg_num_neighbors: float = 1.0,
        radial_kind: RadialKind = "bessel",
        cutoff_order: int = 6,
        distance_transform: Literal["none", "agnesi", "soft"] = "none",
        apply_cutoff: bool = True,
        radial_hidden: Sequence[int] = DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        locality: Callable[[Tensor, Mapping[str, Any]], Tensor] | None = None,
        node_inputs: Sequence[InputSpec] = (),
        last_layer_irreps: str = "0e",
        residual_first_layer: bool = False,
        learned_density_first_layer: bool = False,
        learned_density: bool = False,
        nonlinear_first_layer: bool = False,
        nonlinear: bool = False,
        convolution_irreps: str | None = None,
        narrow_first_convolution: bool = False,
        element_agnostic_product: bool = False,
        edge_axes: tuple[int, int, int] = (0, 1, 2),
    ) -> None:
        super().__init__()
        if sorted(edge_axes) != [0, 1, 2]:
            raise ValueError(
                f"edge_axes is {edge_axes}; it is an order of the three axes."
            )
        self.atomic_numbers = list(atomic_numbers)
        self.element_agnostic_product = element_agnostic_product
        self.edge_axes = None if tuple(edge_axes) == (0, 1, 2) else list(edge_axes)
        self.num_features = num_features
        self.layout = layout_of(backend)
        self.lmax = lmax
        self.cutoff = cutoff
        self.locality = locality
        self.hidden_irreps = hidden_irreps

        edge_irreps = "+".join(
            f"{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in range(lmax + 1)
        )
        self.edge_irreps = edge_irreps
        self.edge_attributes = backend.make_spherical_harmonics(
            SphericalHarmonicsDescriptor(lmax=lmax, precision=precision)
        )
        self.radial = backend.make_radial_basis(
            RadialBasisDescriptor(
                kind=radial_kind,
                num_basis=num_radial,
                cutoff=cutoff,
                cutoff_order=cutoff_order,
                apply_cutoff=apply_cutoff,
                precision=precision,
            )
        )
        # Without the envelope in the basis, the interactions multiply it into
        # what their radial networks produce.
        self.envelope = (
            None
            if apply_cutoff
            else PolynomialCutoff(r_max=cutoff, polynomial_order=cutoff_order)
        )
        transforms = {"agnesi": AgnesiTransform, "soft": SoftTransform}
        if distance_transform != "none" and distance_transform not in transforms:
            raise ValueError(
                f"distance_transform is {distance_transform!r}; it is 'none', "
                f"'agnesi' or 'soft'."
            )
        self.distance_transform = (
            transforms[distance_transform]() if distance_transform != "none" else None
        )
        # The embedding produces scalars only, one per channel. The higher
        # irreps appear for the first time out of the first convolution, which
        # is why that layer's node declaration is narrower than the rest.
        # Scalars, plus whatever the declared node inputs carry, since that is
        # where they are mixed in and an equivariant map cannot create an irrep
        # its input lacks.
        embedding_terms = ["0e"]
        for spec in node_inputs:
            for _, irrep in Irreps.parse(spec.irreps).terms:
                if str(irrep) not in embedding_terms:
                    embedding_terms.append(str(irrep))
        self.embedding_irreps = "+".join(embedding_terms)
        self.node_embedding = backend.make_linear(
            LinearDescriptor(
                irreps_in=f"{len(self.atomic_numbers)}x0e",
                irreps_out=expanded_irreps(self.embedding_irreps, num_features),
                precision=precision,
            )
        )

        interactions, products = [], []
        for layer in range(num_layers):
            # The first layer reads the embedding, which is scalars. The last
            # one produces only what is read off it, scalars for an energy.
            node_per_channel = self.embedding_irreps if layer == 0 else hidden_irreps
            last = layer == num_layers - 1
            product_per_channel = last_layer_irreps if last else hidden_irreps
            density = learned_density_first_layer if layer == 0 else learned_density
            if nonlinear_first_layer if layer == 0 else nonlinear:
                convolved = convolution_irreps
                if layer == 0:
                    convolved = (
                        _scalars_of(convolution_irreps)
                        if narrow_first_convolution and convolution_irreps
                        else None
                    )
                irreps_up, num_up_features = (
                    _per_channel(convolved)
                    if convolved
                    else (node_per_channel, num_features)
                )
                interactions.append(
                    NonLinearInteractionBlock(
                        backend,
                        irreps_node=node_per_channel,
                        irreps_up=irreps_up,
                        irreps_edge=edge_irreps,
                        irreps_target=edge_irreps,
                        irreps_skip_out=product_per_channel,
                        num_radial=num_radial,
                        num_features=num_features,
                        num_up_features=num_up_features,
                        num_elements=len(self.atomic_numbers),
                        radial_hidden=tuple(radial_hidden),
                        precision=precision,
                    )
                )
            elif layer == 0 and not residual_first_layer:
                interactions.append(
                    InteractionBlock(
                        backend,
                        irreps_node=node_per_channel,
                        irreps_edge=edge_irreps,
                        irreps_target=edge_irreps,
                        num_radial=num_radial,
                        num_features=num_features,
                        num_elements=len(self.atomic_numbers),
                        avg_num_neighbors=avg_num_neighbors,
                        radial_hidden=tuple(radial_hidden),
                        precision=precision,
                        learned_density=density,
                    )
                )
            else:
                interactions.append(
                    ResidualInteractionBlock(
                        backend,
                        irreps_node=node_per_channel,
                        irreps_edge=edge_irreps,
                        irreps_target=edge_irreps,
                        irreps_skip_out=product_per_channel,
                        num_radial=num_radial,
                        num_features=num_features,
                        num_elements=len(self.atomic_numbers),
                        avg_num_neighbors=avg_num_neighbors,
                        radial_hidden=tuple(radial_hidden),
                        precision=precision,
                        learned_density=density,
                    )
                )
            products.append(
                EquivariantProductBasisBlock(
                    backend,
                    irreps_in=edge_irreps,
                    irreps_out=product_per_channel,
                    correlation=correlation,
                    num_elements=(
                        1 if element_agnostic_product else len(self.atomic_numbers)
                    ),
                    num_features=num_features,
                    precision=precision,
                )
            )
        self.node_inputs = (
            NodeInputEmbedding(
                backend,
                list(node_inputs),
                hidden_irreps=self.embedding_irreps,
                num_features=num_features,
                precision=precision,
            )
            if node_inputs
            else None
        )

        # The message declaration is the edge attributes' own, which is what the
        # trained artifacts carry: the convolution couples the node features
        # with the harmonics and keeps what lands on those irreps.
        self.layer_irreps = [
            last_layer_irreps if layer == num_layers - 1 else hidden_irreps
            for layer in range(num_layers)
        ]
        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)
        self.width = Irreps.parse(hidden_irreps).dimension

    def element_index(self, atomic_numbers: Tensor) -> Tensor:
        """Each node's position in the element table.

        Built from the table rather than read off a one-hot with an argmax,
        which is what the frozen tree does inside every block.
        """
        table = torch.as_tensor(
            self.atomic_numbers,
            dtype=atomic_numbers.dtype,
            device=atomic_numbers.device,
        )
        return (
            (atomic_numbers[:, None] == table[None, :]).to(torch.int64).argmax(dim=-1)
        )

    def forward(self, graph: Mapping[str, Any]) -> list[Tensor]:
        """Every layer's node features, in order.

        Args:
            graph: The flat dict. Read and never written to.

        Returns:
            One ``[n_nodes, num_features * width]`` tensor per layer, grouped
            by irrep in the backend's layout. The list rather than only the
            last, because the readouts above this read every layer and the
            descriptor API slices them.
        """
        positions = graph["positions"]
        edge_index = graph["edge_index"]
        sender, receiver = edge_index[0], edge_index[1]
        num_nodes = int(positions.shape[0])

        # The derivative engine forms these in its own phase, because a strain
        # has to reach the positions before the vectors are built and because
        # edge forces need the vectors themselves as the graph leaf. When they
        # are already there they are used as given; otherwise they are built
        # here, which is what a plain forward with no derivatives does.
        if "vectors" in graph:
            vectors = graph["vectors"]
        else:
            vectors = positions[receiver] - positions[sender] + graph["shifts"]
        lengths = vectors.norm(dim=-1, keepdim=True)
        edge_attributes = self.edge_attributes(
            vectors if self.edge_axes is None else vectors[:, self.edge_axes]
        )
        if self.distance_transform is None:
            radial = self.radial(lengths)
        else:
            radial = self.radial(
                lengths,
                self.distance_transform(lengths, graph["atomic_numbers"], edge_index),
            )

        edge_envelope = None if self.envelope is None else self.envelope(lengths)

        element = self.element_index(graph["atomic_numbers"])
        one_hot = torch.zeros(
            num_nodes,
            len(self.atomic_numbers),
            dtype=positions.dtype,
            device=positions.device,
        )
        one_hot[torch.arange(num_nodes, device=positions.device), element] = 1.0

        # One set of product weights for every element is the same contraction
        # with every node reading the first.
        product_element = (
            torch.zeros_like(element) if self.element_agnostic_product else element
        )

        # Scalars only, so the grouped layout is already what comes out.
        features = self.node_embedding(one_hot)
        if self.node_inputs is not None:
            features = self.node_inputs(graph, features)

        outputs = []
        for interaction, product in zip(self.interactions, self.products, strict=True):
            message, carried = interaction(
                features,
                edge_attributes,
                radial,
                one_hot,
                sender,
                receiver,
                num_nodes,
                edge_envelope,
            )
            features = product(message, product_element, carried)
            if self.locality is not None:
                features = self.locality(features, graph)
            outputs.append(features)
        return outputs

    def descriptors(
        self,
        graph: Mapping[str, Any],
        num_layers: int | None = None,
        invariants_only: bool = True,
        aggregation: str | None = None,
    ) -> Tensor:
        """The backbone's raison d'etre, as a first-class call.

        Args:
            graph: The flat dict.
            num_layers: How many layers to keep, from the first. ``None`` keeps
                all of them.
            invariants_only: Keep only the scalar channels. The last layer of a
                legacy model is invariant-only already, which is why its width
                differs from the others.
            aggregation: ``None`` for per-node, ``"mean"`` for the structure
                mean, ``"per_element_mean"`` for one row per element.

        Returns:
            ``[n_nodes, features]``, or the aggregated form.
        """
        layers = self.forward(graph)
        if num_layers is not None:
            layers = layers[:num_layers]
        # Grouped by irrep, so the scalars are the leading block: one per
        # channel, contiguous.
        pieces = []
        for index, layer in enumerate(layers):
            if not invariants_only:
                # Handed to a user, so canonical whatever the backend's layout.
                grouped = expanded_irreps(self.layer_irreps[index], self.num_features)
                terms = self.layout.terms(grouped)
                pieces.append(self.layout.to_canonical(layer, terms))
                continue
            multiplicity, irrep = Irreps.parse(self.layer_irreps[index]).terms[0]
            if irrep.degree != 0 or irrep.parity != 1:
                # A layer that keeps no scalars has no invariants to offer.
                continue
            pieces.append(
                layer[..., : self.num_features * multiplicity * irrep.dimension]
            )
        stacked = torch.cat(pieces, dim=-1)
        if aggregation is None:
            return stacked
        if aggregation == "mean":
            return stacked.mean(dim=0, keepdim=True)
        if aggregation == "per_element_mean":
            element = self.element_index(graph["atomic_numbers"])
            rows = []
            for index in range(len(self.atomic_numbers)):
                mask = element == index
                rows.append(
                    stacked[mask].mean(dim=0)
                    if bool(mask.any())
                    else torch.zeros_like(stacked[0])
                )
            return torch.stack(rows)
        raise ValueError(
            f"{aggregation!r} is not an aggregation this computes. The choices "
            f"are None, 'mean' and 'per_element_mean'."
        )
