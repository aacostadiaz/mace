"""The many-body product basis, on the kernel backend's contraction.

No e3nn, no fx codegen, no TorchScript. The frozen tree builds its symmetric
contraction with `CodeGenMixin` and `torch.fx.symbolic_trace`, which is what a
model has to carry so that TorchScript can see through it; here the contraction
is one backend op and the block around it is ordinary torch.

Nothing resolves in ``forward``. The backend is asked for the op once, at
construction, and the op is held.
"""

from __future__ import annotations

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

__all__ = ["EquivariantProductBasisBlock"]


class EquivariantProductBasisBlock(nn.Module):
    """A symmetric contraction followed by a linear map, with an optional skip.

    Args:
        backend: The kernel backend. Consulted here and never again.
        irreps_in: The node features entering, as one channel's irreps.
        irreps_out: What the block produces.
        correlation: The body order of the contraction.
        num_elements: How many elements carry their own contraction weights.
        num_features: The channel width.
        residual: Whether to add the input through. The frozen tree expresses
            this as a separate block class per variant.
        precision: The dtype name the ops are built at.
    """

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        correlation: int,
        num_elements: int,
        num_features: int,
        residual: bool = False,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
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
                irreps_in=irreps_out, irreps_out=irreps_out, precision=precision
            )
        )
        self.residual = residual
        self.width = Irreps.parse(irreps_out).dimension

    def forward(self, node_features: Tensor, element: Tensor) -> Tensor:
        """``[n_nodes, num_features, dim_in]`` in, the same shape out at ``dim_out``.

        Args:
            node_features: The features to contract.
            element: ``[n_nodes]`` int64, which element each node is.
        """
        contracted = self.contraction(node_features, element)
        mapped = self.linear(contracted)
        if self.residual and mapped.shape == node_features.shape:
            return mapped + node_features
        return mapped
