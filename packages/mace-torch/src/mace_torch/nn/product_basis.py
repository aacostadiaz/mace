"""The many-body product basis, on the kernel backend's contraction.

No e3nn, no fx codegen, no TorchScript. The frozen tree builds its symmetric
contraction with `CodeGenMixin` and `torch.fx.symbolic_trace`, which is what a
model has to carry so that TorchScript can see through it; here the contraction
is one backend op and the block around it is ordinary torch.

Nothing resolves in ``forward``. The backend is asked for the op once, at
construction, and the op is held.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.nn.layout import (
    channel_layout_index,
    expanded_irreps,
    inverse_layout_index,
)

__all__ = ["EquivariantProductBasisBlock"]


class EquivariantProductBasisBlock(nn.Module):
    """The many-body contraction, then a linear that mixes channels.

    Takes and returns flat node features grouped by irrep, which is the layout
    the blocks pass between them; the ``[nodes, channels, components]`` view
    exists only inside, for the contraction.

    The skip a residual interaction carries is added **here**, after the linear,
    because that is where a trained model puts it. Its declaration is this
    block's output rather than the interaction's.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_in: One channel's input declaration.
        irreps_out: One channel's output declaration.
        correlation: The body order.
        num_elements: How many species, which the contraction weights index.
        num_features: The channel count.
        precision: The dtype every op is built at.
    """

    to_channels: Tensor
    from_channels: Tensor

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        correlation: int,
        num_elements: int,
        num_features: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.width_in = Irreps.parse(irreps_in).dimension
        self.width_out = Irreps.parse(irreps_out).dimension
        out_flat = expanded_irreps(irreps_out, num_features)

        self.contraction = backend.make_symmetric_contraction(
            SymmetricContractionDescriptor(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                num_features=num_features,
                precision=precision,
            )
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=out_flat, irreps_out=out_flat, precision=precision
            )
        )
        self.register_buffer(
            "to_channels",
            torch.tensor(inverse_layout_index(irreps_in, num_features)),
            persistent=False,
        )
        self.register_buffer(
            "from_channels",
            torch.tensor(channel_layout_index(irreps_out, num_features)),
            persistent=False,
        )

    def forward(
        self, message: Tensor, element: Tensor, skip: Tensor | None = None
    ) -> Tensor:
        """Flat grouped features in, flat grouped features out.

        Args:
            message: ``[n_nodes, dim_in]``, grouped by irrep.
            element: ``[n_nodes]`` int64, which element each node is.
            skip: The residual interaction's carried skip, already at this
                block's output declaration, or ``None`` in the first layer.
        """
        nodes = message.shape[0]
        features = message[..., self.to_channels].reshape(
            nodes, self.num_features, self.width_in
        )
        contracted = self.contraction(features, element)
        contracted = contracted.reshape(nodes, -1)[..., self.from_channels]
        mapped = self.linear(contracted)
        if skip is not None:
            return mapped + skip
        return mapped
