"""Radial bases and the cutoff, ported from the frozen tree.

Plain torch, no e3nn. These are closed forms and they port nearly verbatim, so
the numbers are pinned against the frozen implementation bit for bit rather
than to a tolerance.

Two deliberate differences from what was ported, both of which a reader would
otherwise take for drift:

**The dtype is an argument, not the global default.** Every frozen class reads
``torch.get_default_dtype()`` while it is being constructed, which makes a
module's precision depend on whatever the process last set. Here it is passed
in and defaults to float64. That is the seam the precision configuration
replaces, and taking it now costs nothing.

**One Chebyshev class instead of two.** The frozen tree has ``ChebychevBasis``
and ``ChebyshevBasisGeneral`` computing the same polynomials, differing in
whether the constant term is included. That is a flag, not a class.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "BesselBasis",
    "ChebyshevBasis",
    "GaussianBasis",
    "PolynomialCutoff",
]

_DTYPES = {"float64": torch.float64, "float32": torch.float32}


class BesselBasis(nn.Module):
    """Sine-over-r radial functions, equation (7) of the MACE paper.

    Args:
        r_max: The cutoff, in Angstrom.
        num_basis: How many functions.
        trainable: Whether the frequencies are learned. Off by default, as in
            the frozen tree.
        dtype: Precision name.
    """

    #: Annotated because `register_buffer` alone leaves the attribute
    #: typed as a `Module`, and then arithmetic on it has no operators.
    bessel_weights: Tensor
    r_max: Tensor
    prefactor: Tensor

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        trainable: bool = False,
        dtype: str = "float64",
    ) -> None:
        super().__init__()
        torch_dtype = _DTYPES[dtype]
        frequencies = (
            math.pi
            / r_max
            * torch.linspace(1.0, num_basis, num_basis, dtype=torch_dtype)
        )
        if trainable:
            self.bessel_weights = nn.Parameter(frequencies)
        else:
            self.register_buffer("bessel_weights", frequencies)
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch_dtype))
        self.register_buffer(
            "prefactor", torch.tensor(math.sqrt(2.0 / r_max), dtype=torch_dtype)
        )

    def forward(self, lengths: Tensor) -> Tensor:
        """``[n_edges, 1]`` in, ``[n_edges, num_basis]`` out."""
        return self.prefactor * (torch.sin(self.bessel_weights * lengths) / lengths)


class ChebyshevBasis(nn.Module):
    """Chebyshev polynomials of the first kind.

    Args:
        r_max: The cutoff, in Angstrom. Carried for the repr and for callers;
            the polynomials themselves are evaluated on the length directly,
            as in the frozen tree.
        num_basis: How many polynomials.
        include_constant: Whether to start at ``T_0`` rather than ``T_1``. The
            frozen tree expresses this as two classes computing the same thing.
        dtype: Precision name.
    """

    #: Annotated because `register_buffer` alone leaves the attribute
    #: typed as a `Module`, and then arithmetic on it has no operators.
    orders: Tensor

    def __init__(
        self,
        r_max: float,
        num_basis: int = 8,
        include_constant: bool = False,
        dtype: str = "float64",
    ) -> None:
        super().__init__()
        first = 0 if include_constant else 1
        self.register_buffer(
            "orders",
            torch.arange(first, first + num_basis, dtype=_DTYPES[dtype]).unsqueeze(0),
        )
        self.num_basis = num_basis
        self.r_max = r_max

    def forward(self, lengths: Tensor) -> Tensor:
        repeated = lengths.repeat(1, self.num_basis)
        orders = self.orders.repeat(len(repeated), 1)
        return torch.special.chebyshev_polynomial_t(repeated, orders)


class GaussianBasis(nn.Module):
    """Gaussians evenly spaced from 0 to the cutoff."""

    #: Annotated because `register_buffer` alone leaves the attribute
    #: typed as a `Module`, and then arithmetic on it has no operators.
    gaussian_weights: Tensor

    def __init__(
        self,
        r_max: float,
        num_basis: int = 128,
        trainable: bool = False,
        dtype: str = "float64",
    ) -> None:
        super().__init__()
        centres = torch.linspace(0.0, r_max, num_basis, dtype=_DTYPES[dtype])
        if trainable:
            self.gaussian_weights = nn.Parameter(centres)
        else:
            self.register_buffer("gaussian_weights", centres)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, lengths: Tensor) -> Tensor:
        return torch.exp(self.coeff * torch.pow(lengths - self.gaussian_weights, 2))


class PolynomialCutoff(nn.Module):
    """A smooth envelope from 1 at zero to 0 at the cutoff.

    Zero value *and* zero derivative at the cutoff, which is what keeps a force
    continuous as an atom crosses the radius. Without it the discontinuity
    shows up as energy drift in a simulation rather than as an error.
    """

    #: Annotated because `register_buffer` alone leaves the attribute
    #: typed as a `Module`, and then arithmetic on it has no operators.
    p: Tensor
    r_max: Tensor

    def __init__(self, r_max: float, p: int = 6, dtype: str = "float64") -> None:
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=_DTYPES[dtype]))

    def forward(self, lengths: Tensor) -> Tensor:
        power = self.p
        scaled = lengths / self.r_max
        envelope = (
            1.0
            - ((power + 1.0) * (power + 2.0) / 2.0) * torch.pow(scaled, power)
            + power * (power + 2.0) * torch.pow(scaled, power + 1)
            - (power * (power + 1.0) / 2.0) * torch.pow(scaled, power + 2)
        )
        return envelope * (lengths < self.r_max)
