"""The message-passing interaction, on the kernel backend's factories.

Two variants, which is what the frozen tree spreads over eight block classes:
the first layer, which has no residual path because there is nothing yet to add
to, and the residual one for every layer after it.

Three things this deliberately does not have, each replacing something in the
frozen tree that a v1 block must never grow:

* **No fusion branch.** The frozen tree's blocks test
  ``hasattr(self, "conv_fusion")`` at six sites and take a different path. Here
  the convolution is always node-level and whether the reduction is fused is
  the backend's business.
* **No LAMMPS attributes.** ``lammps_class`` and ``lammps_natoms`` are smuggled
  through the blocks in the frozen tree, with a guard that also probes
  ``torch.jit.is_scripting()``. The locality seam here is an explicit hook on
  the model, default absent.
* **Nothing resolved in forward.** Every op is built once from a descriptor.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

__all__ = ["InteractionBlock"]


class InteractionBlock(nn.Module):
    """One message-passing layer.

    Args:
        backend: The kernel backend, consulted at construction only.
        irreps_node: The node features entering, one channel's worth.
        irreps_edge: The edge attributes, normally spherical harmonics.
        irreps_out: What the layer produces.
        num_radial: Width of the radial embedding.
        num_features: The channel width.
        num_elements: How many elements, for the skip connection.
        avg_num_neighbors: The density normalization. Messages are divided by
            its square root, so a model trained on dense structures does not
            see a different message scale on sparse ones.
        residual: Whether the layer carries the skip connection. False for the
            first layer, where there is nothing to skip from.
        precision: The dtype name.
    """

    def __init__(
        self,
        backend,
        irreps_node: str,
        irreps_edge: str,
        irreps_out: str,
        num_radial: int,
        num_features: int,
        num_elements: int,
        avg_num_neighbors: float = 1.0,
        residual: bool = True,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        self.linear_up = backend.make_linear(
            LinearDescriptor(
                irreps_in=irreps_node, irreps_out=irreps_node, precision=precision
            )
        )
        self.convolution = backend.make_channelwise_tp_conv(
            ChannelwiseTPConvDescriptor(
                irreps_node=irreps_node,
                irreps_edge=irreps_edge,
                irreps_out=irreps_out,
                num_radial=num_radial,
                precision=precision,
            )
        )
        self.linear_down = backend.make_linear(
            LinearDescriptor(
                irreps_in=irreps_out, irreps_out=irreps_out, precision=precision
            )
        )
        self.skip = None
        if residual:
            self.skip = backend.make_fully_connected_tp(
                FullyConnectedTPDescriptor(
                    irreps_in1=irreps_node,
                    irreps_in2=f"{num_elements}x0e",
                    irreps_out=irreps_out,
                    precision=precision,
                )
            )
        self.register_buffer("density", torch.tensor(float(avg_num_neighbors) ** 0.5))
        self.num_paths = self.convolution.num_paths
        self.width_out = Irreps.parse(irreps_out).dimension

    def forward(
        self,
        node_features: Tensor,
        edge_attributes: Tensor,
        radial_weights: Tensor,
        element_attributes: Tensor,
        sender: Tensor,
        receiver: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """``[n_nodes, num_features, dim_out]``, always node-level.

        Args:
            node_features: ``[n_nodes, num_features, dim_node]``.
            edge_attributes: ``[n_edges, dim_edge]``.
            radial_weights: ``[n_edges, num_paths, num_features]``.
            element_attributes: ``[n_nodes, num_elements]`` one-hot, for the
                skip connection.
            sender, receiver: ``[n_edges]`` int64.
            num_nodes: A plain int, so it stays symbolic under compile.
        """
        mapped = self.linear_up(node_features)
        messages = self.convolution(
            mapped, edge_attributes, radial_weights, sender, receiver, num_nodes
        )
        messages = self.linear_down(messages / self.density)
        if self.skip is not None:
            flat = node_features.reshape(node_features.shape[0], -1)
            attributes = element_attributes
            skipped = self.skip(
                node_features.reshape(-1, node_features.shape[-1]),
                attributes.repeat_interleave(node_features.shape[1], dim=0),
            ).reshape(messages.shape)
            del flat
            return messages + skipped
        return messages
