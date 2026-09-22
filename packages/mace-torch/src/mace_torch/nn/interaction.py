"""Message passing: one convolution, one linear, one skip.

Two variants, because the anchors have two and they differ in where the skip
goes. In the first layer the skip is applied **to the message** and replaces
it. In every later layer it is taken from the **input** node features and
handed onward untouched, for the product basis to add after its contraction.
That is not a refactor away: the two compute different functions and a trained
model depends on which one it has.

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
average itself, not its square root.
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

from mace_torch.nn.layout import (
    expanded_irreps,
    inverse_layout_index,
    path_layout_index,
)
from mace_torch.nn.radial_mlp import RadialMLP

__all__ = ["InteractionBlock", "ResidualInteractionBlock"]

#: The widths of the radial network's hidden layers, as the anchors carry them.
DEFAULT_RADIAL_HIDDEN = (64, 64, 64)


class _Convolution(nn.Module):
    """What both variants share: up, convolve, down, normalize.

    Held apart from the two blocks so that the difference between them is
    visible as the one thing it is, rather than as a flag.
    """

    to_channels: Tensor
    from_paths: Tensor

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
    ) -> None:
        super().__init__()
        paths = channelwise_paths(irreps_node, irreps_edge, irreps_target)
        node_flat = expanded_irreps(irreps_node, num_features)
        target_flat = expanded_irreps(irreps_target, num_features)
        path_flat = "+".join(f"{num_features}x{path.irrep}" for path in paths)

        self.num_features = num_features
        self.num_paths = len(paths)
        self.node_width = Irreps.parse(irreps_node).dimension
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
            "to_channels",
            torch.tensor(inverse_layout_index(irreps_node, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "from_paths",
            torch.tensor(path_layout_index(paths, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "neighbours", torch.tensor(float(avg_num_neighbors)), persistent=False
        )

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        edge_radial: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Flat node features in, flat message out, both grouped by irrep."""
        mapped = self.linear_up(node_features)
        mapped = mapped[..., self.to_channels].reshape(
            -1, self.num_features, self.node_width
        )
        weights = self.radial(edge_radial).reshape(
            -1, self.num_paths, self.num_features
        )
        message = self.convolution(
            mapped, edge_attributes, weights, sender, receiver, num_nodes
        )
        message = message.reshape(num_nodes, -1)[..., self.from_paths]
        return self.linear(message) / self.neighbours


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
    ) -> tuple[Tensor, Tensor | None]:
        """The message, flat and grouped by irrep, and no carried skip."""
        message = self.body(
            node_features, edge_attributes, edge_radial, sender, receiver, num_nodes
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
    ) -> tuple[Tensor, Tensor]:
        """The message and the carried skip, both flat and grouped."""
        carried = self.skip(node_features, element_attributes)
        message = self.body(
            node_features, edge_attributes, edge_radial, sender, receiver, num_nodes
        )
        return message, carried
