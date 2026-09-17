"""Real spherical harmonics, native, in e3nn's convention.

Not a table and not a library. Each degree is the previous one coupled with the
l = 1 block and projected with
:func:`mace_core.clebsch_gordan.real_basis.wigner_3j_real`, the same function
the symmetric contraction is built from. Deriving one from the other is what
makes a convention disagreement between them impossible rather than merely
testable, and a disagreement there is wrong numbers rather than an error.

The convention is the one ARCH-1 specifies and the frozen tree uses: components
ordered so that the l = 1 block is ``(x, y, z)``, and ``"component"``
normalization, meaning the squared norm of degree ``l`` is ``2l + 1``.

Verified against ``o3.spherical_harmonics`` at fp64 up to l = 4. e3nn is a
test-only oracle; nothing here imports it.
"""

from __future__ import annotations

import math

import torch
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from torch import Tensor

__all__ = ["spherical_harmonics"]


def _coupling(degree: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(
        wigner_3j_real(degree, degree - 1, 1), dtype=dtype, device=device
    )


def spherical_harmonics(
    directions: Tensor, lmax: int, normalize: bool = True
) -> Tensor:
    """Real spherical harmonics of ``directions``, concatenated over degrees.

    Args:
        directions: ``[n, 3]`` Cartesian. Normalized first unless ``normalize``
            is false, in which case they are taken as already unit.
        lmax: The highest degree.
        normalize: Whether to normalize the input directions.

    Returns:
        ``[n, (lmax + 1) ** 2]``, degree 0 first. Degree ``l`` has squared norm
        ``2l + 1`` for a unit direction, which is what ``"component"`` means.

    The per-degree scale is fixed by that normalization rather than carried
    through the recursion, so an error in one degree cannot propagate into the
    next as a silent factor.
    """
    if normalize:
        directions = directions / directions.norm(dim=-1, keepdim=True)
    dtype, device = directions.dtype, directions.device

    blocks = [directions.new_ones((directions.shape[0], 1))]
    if lmax >= 1:
        first = directions * math.sqrt(3.0)
        blocks.append(first)
        for degree in range(2, lmax + 1):
            raw = torch.einsum(
                "oab,na,nb->no", _coupling(degree, dtype, device), blocks[-1], first
            )
            scale = math.sqrt(2 * degree + 1) / raw.norm(dim=-1).mean()
            blocks.append(raw * scale)
    return torch.cat(blocks, dim=-1)
