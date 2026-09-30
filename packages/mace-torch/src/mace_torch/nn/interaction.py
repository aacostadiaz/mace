"""Message passing: one convolution, one linear, one skip.

Two variants, because the anchors have two and they differ in where the skip
goes. In the first layer the skip is applied **to the message** and replaces
it. In every later layer it is taken from the **input** node features and
handed onward untouched, for the product basis to add after its contraction.
That is not a refactor away: the two compute different functions and a trained
model depends on which one it has. A third block, the nonlinear one, is its
own convolution rather than a variant of theirs; its docstring says how.

Three things here are the format rather than a choice, and each was read off a
trained artifact:

**The linears mix channels.** They are maps between declarations that carry
multiplicity, so a ``16x0e -> 16x0e`` linear has 256 weights and not one. A
per-channel version is a different, much smaller model.

**The convolution keeps its paths apart.** Several couplings land on the same
output irrep, and the linear after the convolution mixes them with independent
weights. Summing them inside the convolution would fold two weights into one:
measured against the anchor, that is 768 weights where the trained model has
1792.

**The division by the neighbour count comes after the linear**, and it is the
average itself, not its square root. A block with a learned density divides
by ``1 + sum over its edges of tanh(f(r)^2)`` instead, where ``f`` is one
linear map of the radial embedding: each atom's own count, softened, in place
of the dataset's average.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.kernels import segment_sum
from mace_torch.nn.layout import (
    expanded_irreps,
)
from mace_torch.nn.radial_mlp import RadialMLP

__all__ = ["InteractionBlock", "NonLinearInteractionBlock", "ResidualInteractionBlock"]

#: The widths of the radial network's hidden layers, as the anchors carry them.
DEFAULT_RADIAL_HIDDEN = (64, 64, 64)


class _Convolution(nn.Module):
    """What both variants share: up, convolve, down, normalize.

    Held apart from the two blocks so that the difference between them is
    visible as the one thing it is, rather than as a flag.
    """

    neighbours: Tensor

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        num_radial: int,
        num_features: int,
        avg_num_neighbors: float,
        radial_hidden=DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        learned_density: bool = False,
    ) -> None:
        super().__init__()
        paths = channelwise_paths(irreps_node, irreps_edge, irreps_target)
        node_flat = expanded_irreps(irreps_node, num_features)
        target_flat = expanded_irreps(irreps_target, num_features)
        path_flat = "+".join(f"{num_features}x{path.irrep}" for path in paths)

        self.num_features = num_features
        self.num_paths = len(paths)
        self.target_width = Irreps.parse(irreps_target).dimension
        self.irreps_out = target_flat

        self.linear_up = backend.make_linear(
            LinearDescriptor(
                irreps_in=node_flat, irreps_out=node_flat, precision=precision
            )
        )
        self.convolution = backend.make_channelwise_tp_conv(
            ChannelwiseTPConvDescriptor(
                irreps_node=irreps_node,
                irreps_edge=irreps_edge,
                irreps_out=irreps_target,
                num_radial=num_radial,
                num_features=num_features,
                precision=precision,
            )
        )
        self.radial = RadialMLP(
            num_radial, radial_hidden, self.num_paths * num_features, precision
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=path_flat, irreps_out=target_flat, precision=precision
            )
        )
        self.register_buffer(
            "neighbours", torch.tensor(float(avg_num_neighbors)), persistent=False
        )
        # One linear map of the radial embedding to one number per edge, with
        # no hidden layer: what the frozen tree's density blocks fit.
        self.density = (
            RadialMLP(num_radial, (), 1, precision) if learned_density else None
        )

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
        edge_envelope: Tensor | None = None,
    ) -> Tensor:
        """Flat node features in, flat message out, both grouped by irrep.

        ``edge_envelope``, ``[n_edges, 1]``, is the cutoff envelope when the
        radial embedding does not carry it, multiplied into the radial
        network's output and the learned density.
        """
        mapped = self.linear_up(node_features)
        weights = self.radial(edge_radial)
        if edge_envelope is not None:
            weights = weights * edge_envelope
        weights = weights.reshape(-1, self.num_paths, self.num_features)
        message = self.convolution(
            mapped, edge_attributes, weights, sender, receiver, num_nodes
        )
        if self.density is None:
            return self.linear(message) / self.neighbours
        edge_density = torch.tanh(self.density(edge_radial) ** 2)
        if edge_envelope is not None:
            edge_density = edge_density * edge_envelope
        density = segment_sum(edge_density, receiver, num_nodes)
        return self.linear(message) / (density + 1.0)

    def to_canonical(self) -> dict[str, Tensor]:
        """The neighbour normalization, which the checkpoint has to carry.

        It is a constant of the trained model rather than a weight: the
        messages are divided by it, and a model rebuilt with another value
        loads without complaint and computes a different function. Measured
        on a water molecule, a model trained at six and reloaded at the
        default of one is off by 0.49 eV. So it travels with the weights, like
        the isolated-atom energies do, and a rebuild can start from any value.
        """
        return {"neighbours": self.neighbours.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        """Put the normalization back, in the model's own dtype and place."""
        with torch.no_grad():
            self.neighbours.copy_(state["neighbours"])


class InteractionBlock(nn.Module):
    """The first layer. The skip is applied to the message and replaces it.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_node: One channel's node-feature declaration.
        irreps_edge: The edge attributes.
        irreps_target: One channel's message declaration.
        num_radial: Width of the radial embedding.
        num_features: The channel count.
        num_elements: How many species, which is the skip's second input.
        avg_num_neighbors: The density normalization.
        radial_hidden: The radial network's hidden widths.
        precision: The dtype every op is built at.
        learned_density: Normalize by a density learned per atom rather than
            by ``avg_num_neighbors``.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        num_radial: int,
        num_features: int,
        num_elements: int,
        avg_num_neighbors: float = 1.0,
        radial_hidden=DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        learned_density: bool = False,
    ) -> None:
        super().__init__()
        self.body = _Convolution(
            backend,
            irreps_node,
            irreps_edge,
            irreps_target,
            num_radial,
            num_features,
            avg_num_neighbors,
            radial_hidden,
            precision,
            learned_density,
        )
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=self.body.irreps_out,
                irreps_in2=f"{num_elements}x0e",
                irreps_out=self.body.irreps_out,
                precision=precision,
            )
        )

    @property
    def num_paths(self) -> int:
        return self.body.num_paths

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        element_attributes: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
        edge_envelope: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """The message, flat and grouped by irrep, and no carried skip."""
        message = self.body(
            node_features,
            edge_attributes,
            edge_radial,
            sender,
            receiver,
            num_nodes,
            edge_envelope,
        )
        return self.skip(message, element_attributes), None


class ResidualInteractionBlock(nn.Module):
    """Every later layer. The skip is taken from the input and carried onward.

    It is not added here. The product basis adds it after its contraction,
    which is where a trained model puts it, and the skip's output declaration
    is the product's rather than this block's.

    Args:
        irreps_skip_out: What the skip produces, which is the product basis's
            output rather than this block's. The remaining arguments are
            :class:`InteractionBlock`'s.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_target: str,
        irreps_skip_out: str,
        num_radial: int,
        num_features: int,
        num_elements: int,
        avg_num_neighbors: float = 1.0,
        radial_hidden=DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
        learned_density: bool = False,
    ) -> None:
        super().__init__()
        self.body = _Convolution(
            backend,
            irreps_node,
            irreps_edge,
            irreps_target,
            num_radial,
            num_features,
            avg_num_neighbors,
            radial_hidden,
            precision,
            learned_density,
        )
        self.skip = backend.make_fully_connected_tp(
            FullyConnectedTPDescriptor(
                irreps_in1=expanded_irreps(irreps_node, num_features),
                irreps_in2=f"{num_elements}x0e",
                irreps_out=expanded_irreps(irreps_skip_out, num_features),
                precision=precision,
            )
        )

    @property
    def num_paths(self) -> int:
        return self.body.num_paths

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        element_attributes: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
        edge_envelope: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The message and the carried skip, both flat and grouped."""
        carried = self.skip(node_features, element_attributes)
        message = self.body(
            node_features,
            edge_attributes,
            edge_radial,
            sender,
            receiver,
            num_nodes,
            edge_envelope,
        )
        return message, carried


def _even_scalars(irreps: str) -> int:
    return sum(
        mul
        for mul, ir in Irreps.parse(irreps).terms
        if ir.degree == 0 and ir.parity == 1
    )


class NonLinearInteractionBlock(nn.Module):
    """A convolution conditioned on both elements, with a gated output.

    Four things set it apart from the standard blocks, and each is a weight a
    trained model carries:

    * **The convolution runs at its own width.** The node features are mapped
      up to ``irreps_up`` channels of ``num_up_features``, which may be fewer
      than the node features', convolved there, and mapped back.
    * **The radial network sees both elements.** Its input is the radial
      embedding with a learned embedding of the sender's element and one of
      the receiver's appended, and it is a layer-normalized network rather
      than the standard one.
    * **The normalization is learned.** The message is divided by ``alpha +
      beta * density``, with the density learned per atom as in the density
      blocks and ``alpha``, ``beta`` two trained scalars.
    * **It ends in a nonlinearity.** A residual from the up-projection is
      added, an equivariant gate applied and a last linear map taken.

    The skip is a linear map of the input, carried to the product basis as
    the residual blocks carry theirs, in the first layer too.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_node: One channel's node-feature declaration.
        irreps_up: One channel's declaration of what is convolved.
        irreps_edge: The edge attributes.
        irreps_target: One channel's message declaration.
        irreps_skip_out: What the skip produces, the product basis's output.
        num_radial: Width of the radial embedding.
        num_features: The node features' channel count.
        num_up_features: The channel count of what is convolved.
        num_elements: How many species.
        radial_hidden: The radial network's hidden widths.
        precision: The dtype every op is built at.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_up: str,
        irreps_edge: str,
        irreps_target: str,
        irreps_skip_out: str,
        num_radial: int,
        num_features: int,
        num_up_features: int,
        num_elements: int,
        radial_hidden=DEFAULT_RADIAL_HIDDEN,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        from mace_torch.backends.layout import layout_of
        from mace_torch.nn.field import LayerNormMLP, _GatedActivation

        paths = channelwise_paths(irreps_up, irreps_edge, irreps_target)
        node_flat = expanded_irreps(irreps_node, num_features)
        up_flat = expanded_irreps(irreps_up, num_up_features)
        target_flat = expanded_irreps(irreps_target, num_features)
        path_flat = "+".join(f"{num_up_features}x{path.irrep}" for path in paths)
        target = Irreps.parse(target_flat)
        scalars = sum(mul for mul, ir in target.terms if ir.degree == 0)
        gated = "+".join(f"{mul}x{ir}" for mul, ir in target.terms if ir.degree > 0)
        gates = sum(mul for mul, ir in target.terms if ir.degree > 0)
        # The gate's input: the scalars and the gates together, then what they
        # gate, which is the order the frozen tree's sorted declaration has.
        nonlinear_flat = "+".join(
            term for term in (f"{scalars + gates}x0e", gated) if term
        )
        element_scalars = _even_scalars(node_flat)

        self.num_up_features = num_up_features
        self.num_paths = len(paths)
        self.up_width = Irreps.parse(irreps_up).dimension
        self.irreps_out = target_flat

        def linear(irreps_in: str, irreps_out: str):
            return backend.make_linear(
                LinearDescriptor(
                    irreps_in=irreps_in, irreps_out=irreps_out, precision=precision
                )
            )

        self.linear_up = linear(node_flat, up_flat)
        self.linear_res = linear(up_flat, nonlinear_flat)
        self.source = linear(f"{num_elements}x0e", f"{element_scalars}x0e")
        self.target = linear(f"{num_elements}x0e", f"{element_scalars}x0e")
        self.convolution = backend.make_channelwise_tp_conv(
            ChannelwiseTPConvDescriptor(
                irreps_node=irreps_up,
                irreps_edge=irreps_edge,
                irreps_out=irreps_target,
                num_radial=num_radial,
                num_features=num_up_features,
                precision=precision,
            )
        )
        conditioned = num_radial + 2 * element_scalars
        self.radial = LayerNormMLP(
            [conditioned, *radial_hidden, self.num_paths * num_up_features]
        )
        self.density = LayerNormMLP([conditioned, 64, 1])
        dtype = torch.float64 if precision == "float64" else torch.float32
        self.alpha = nn.Parameter(torch.tensor(20.0, dtype=dtype))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.linear_mid = linear(path_flat, nonlinear_flat)
        self.gate = _GatedActivation(scalars, gated, layout_of(backend))
        self.linear_out = linear(target_flat, target_flat)
        self.skip = linear(node_flat, expanded_irreps(irreps_skip_out, num_features))

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        element_attributes: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
        edge_envelope: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """The message and the carried skip, both flat and grouped."""
        carried = self.skip(node_features)
        up = self.linear_up(node_features)
        residual = self.linear_res(up)
        conditioned = torch.cat(
            [
                edge_radial,
                self.source(element_attributes)[sender],
                self.target(element_attributes)[receiver],
            ],
            dim=-1,
        )
        weights = self.radial(conditioned)
        edge_density = torch.tanh(self.density(conditioned) ** 2)
        if edge_envelope is not None:
            weights = weights * edge_envelope
            edge_density = edge_density * edge_envelope
        weights = weights.reshape(-1, self.num_paths, self.num_up_features)
        density = segment_sum(edge_density, receiver, num_nodes)
        message = self.convolution(
            up, edge_attributes, weights, sender, receiver, num_nodes
        )
        message = self.linear_mid(message) / (density * self.beta + self.alpha)
        message = self.gate(message + residual)
        return self.linear_out(message), carried

    def initialize_weights(self, seed: int) -> None:
        """The frozen tree's start: alpha at 20, beta at 0."""
        with torch.no_grad():
            self.alpha.fill_(20.0)
            self.beta.zero_()

    def to_canonical(self) -> dict[str, Tensor]:
        return {"alpha": self.alpha.detach(), "beta": self.beta.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            self.alpha.copy_(state["alpha"])
            self.beta.copy_(state["beta"])
