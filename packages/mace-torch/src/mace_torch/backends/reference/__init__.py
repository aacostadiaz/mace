"""The plain-torch reference backend: the correctness oracle, CPU-capable.

Reference-only ops (the spherical harmonics) live here. They are closed-form
and cheap, are never dispatched to an accelerated kernel today, and are written
for clarity rather than speed.
"""

from mace_torch.backends.reference.backend import ReferenceBackend
from mace_torch.backends.reference.spherical_harmonics import (
    SphericalHarmonics,
    spherical_harmonics,
)

__all__ = ["ReferenceBackend", "SphericalHarmonics", "spherical_harmonics"]
