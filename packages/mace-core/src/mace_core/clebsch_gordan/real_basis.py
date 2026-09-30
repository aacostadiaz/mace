"""The Wigner 3j table in the real basis the models use.

The coefficients in :mod:`mace_core.clebsch_gordan.coefficients` are in the
complex spherical basis. Everything downstream works over real features, so
they have to be carried across, and which real basis that is is a convention
two tickets have to agree on: the spherical harmonics and the contraction are
built from it, and if they disagree the model is wrong in a way that shows up
as a number rather than as an error.

**The convention is e3nn's**, which is what the frozen tree's harmonics use and
what ARCH-1 specifies. It is not the textbook one, and the difference is a
single fact: e3nn orders the l = 1 components as ``(x, y, z)`` while the
textbook ``m = -l .. +l`` ordering gives ``(y, z, x)``. That relabelling is a
cyclic permutation of the axes, which is a rotation of space, so at l >= 2 it
induces a genuine orthogonal mixing rather than another permutation. Measured
against ``o3.spherical_harmonics`` up to l = 4, the two agree to 3.6e-15 once
the coordinates are permuted and each degree is scaled by sqrt(2l+1), which is
what ``"component"`` normalization means.

None of that needs e3nn to compute. The induced rotation at each degree is
solved for from the textbook harmonics alone, by evaluating them on directions
and on permuted directions, which is what :func:`induced_rotation` does.

**One difference from e3nn survives and is gauge.** For 17 of the 75 triples up
to l = 4 the 3j comes out with the opposite overall sign. That sign is the
arbitrary part of any construction: making the carried tensor real admits both
``i**L`` and ``(-i)**L``, which differ by ``(-1)**L``, so for odd degree sums
the sign is a choice. e3nn's comes from its own recursion rather than from a
rule, and no canonical condition reproduces it: of four tried, the best matched
60 of 75. It does not matter, because the sign of a basis path is absorbed by
the weight that multiplies it, and a legacy checkpoint is converted by solving
for the map rather than assuming it.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np

from mace_core.clebsch_gordan.coefficients import wigner_3j_complex

__all__ = [
    "AXIS_PERMUTATION",
    "induced_rotation",
    "real_basis_change",
    "symmetric_matrix_basis",
    "textbook_harmonics",
    "wigner_3j_real",
    "wigner_d_real",
]

#: Taking a direction to the coordinates the textbook ordering calls its
#: own. e3nn reads l = 1 as (x, y, z) and the textbook as (y, z, x), so
#: this is the relabelling between them, and it is a rotation of space.
AXIS_PERMUTATION = (2, 0, 1)


def real_basis_change(degree: int) -> np.ndarray:
    """The unitary carrying the complex spherical basis to the real one.

    Args:
        degree: The rotation order ``l``.

    Returns:
        Shape ``(2l+1, 2l+1)``, complex. Row ``i`` holds the complex
        coefficients of real component ``m = i - l``.
    """
    matrix = np.zeros((2 * degree + 1, 2 * degree + 1), dtype=np.complex128)
    half = 1 / math.sqrt(2)
    for index, m in enumerate(range(-degree, degree + 1)):
        if m < 0:
            matrix[index, degree + m] = 1j * half
            matrix[index, degree - m] = -1j * half * (-1) ** m
        elif m == 0:
            matrix[index, degree] = 1.0
        else:
            matrix[index, degree - m] = half
            matrix[index, degree + m] = half * (-1) ** m
    return matrix


def _textbook_3j(l1: int, l2: int, l3: int) -> np.ndarray:
    """The 3j table in the textbook real basis, before the convention change.

    Args:
        l1, l2, l3: The three degrees.

    Returns:
        Shape ``(2*l1+1, 2*l2+1, 2*l3+1)``, real, with unit sum of squares
        whenever the triangle inequality holds. Zero otherwise.

    The factor of ``i**(l1+l2+l3)`` is what makes the result real. The raw
    change of basis leaves a tensor that is either real or purely imaginary
    depending on the parity of the degree sum, and that single factor covers
    both cases; the assertion below states it rather than trusting it, because
    a silently complex basis would surface much later as a wrong gradient.
    """
    complex_table = wigner_3j_complex(l1, l2, l3).astype(np.complex128)
    carried = np.einsum(
        "ai,bj,ck,ijk->abc",
        real_basis_change(l1),
        real_basis_change(l2),
        real_basis_change(l3),
        complex_table,
    )
    carried = (1j) ** (l1 + l2 + l3) * carried
    residue = float(np.abs(carried.imag).max())
    if residue > 1e-12:
        raise AssertionError(
            f"the real 3j table for degrees ({l1}, {l2}, {l3}) came out "
            f"complex, with a largest imaginary part of {residue:.3e}. The "
            f"phase convention in this module is wrong for this triple."
        )
    return np.ascontiguousarray(carried.real)


@cache
def textbook_harmonics(degree: int, directions: tuple) -> np.ndarray:
    """Real spherical harmonics in the textbook basis, on given directions.

    Built by the same recursion the model's harmonics use: each degree is the
    previous one coupled with the l = 1 block through the 3j of this module.
    It exists here, rather than only in the framework packages, because
    :func:`induced_rotation` needs it and has to stay framework-free.
    """
    points = np.asarray(directions, dtype=float).reshape(-1, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    blocks = [np.ones((points.shape[0], 1))]
    if degree == 0:
        return blocks[0]
    first = points[:, [1, 2, 0]]
    blocks.append(first)
    for order in range(2, degree + 1):
        coupling = _textbook_3j(order, order - 1, 1)
        raw = np.einsum("oab,na,nb->no", coupling, blocks[-1], first)
        blocks.append(raw / np.linalg.norm(raw, axis=1, keepdims=True).mean())
    return blocks[degree]


@cache
def induced_rotation(degree: int) -> np.ndarray:
    """The orthogonal map from the textbook basis to e3nn's, at one degree.

    Solved for from the textbook harmonics alone: evaluating them on a set of
    directions and on the same directions with the axes permuted gives
    ``Y(Pr) = Y(r) @ Q``, and ``Q`` is that matrix. No e3nn is consulted, and
    none can be: this module is imported by a package that must not have it.

    A fixed set of directions rather than a random one, so the result is the
    same on every machine and every run. The system is heavily overdetermined,
    so the particular directions do not matter as long as they are generic.
    """
    if degree == 0:
        return np.ones((1, 1))
    angles = np.linspace(0.13, 3.01, 4 * (2 * degree + 1))
    points = np.stack(
        [np.cos(angles) * 0.9, np.sin(angles) * 0.8, np.cos(2.0 * angles) * 0.7], axis=1
    )
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    permuted = points[:, list(AXIS_PERMUTATION)]
    here = textbook_harmonics(degree, tuple(map(tuple, points)))
    there = textbook_harmonics(degree, tuple(map(tuple, permuted)))
    rotation, *_ = np.linalg.lstsq(here, there, rcond=None)
    return np.ascontiguousarray(rotation)


def wigner_3j_real(l1: int, l2: int, l3: int) -> np.ndarray:
    """The 3j table in e3nn's real basis, which is the project's convention.

    The textbook table conjugated by :func:`induced_rotation` on each index.
    Same shape, same unit sum of squares, and the same span; what changes is
    which basis of the irrep the components are written against, which is the
    thing the spherical harmonics have to agree with.
    """
    table = _textbook_3j(l1, l2, l3)
    if not table.any():
        return table
    return np.ascontiguousarray(
        np.einsum(
            "ia,jb,kc,ijk->abc",
            induced_rotation(l1),
            induced_rotation(l2),
            induced_rotation(l3),
            table,
        )
    )


def wigner_d_real(degree: int, rotation: np.ndarray) -> np.ndarray:
    """How one irrep's components move when space is rotated.

    The representation matrix ``D`` of the rotation at this degree, in the
    project's own basis, so that a feature row ``f`` carrying this irrep goes
    to ``f @ D`` when the positions go to ``r @ rotation.T``.

    Solved from the project's own harmonics rather than from a closed form in
    Euler angles: the basis is the one this module defines, so deriving ``D``
    from anything else would be asserting a convention instead of reading it.
    The system is overdetermined by a factor of four, which is what makes the
    least squares an identification rather than a fit.

    Args:
        degree: The irrep's degree.
        rotation: A ``(3, 3)`` proper rotation.

    Returns:
        A ``(2 * degree + 1, 2 * degree + 1)`` orthogonal matrix.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"a rotation is a (3, 3) matrix, and this one has shape {rotation.shape}."
        )
    if degree == 0:
        return np.ones((1, 1))

    angles = np.linspace(0.13, 3.01, 4 * (2 * degree + 1))
    points = np.stack(
        [np.cos(angles) * 0.9, np.sin(angles) * 0.8, np.cos(2.0 * angles) * 0.7], axis=1
    )
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    turned = points @ rotation.T

    change = induced_rotation(degree)
    here = textbook_harmonics(degree, tuple(map(tuple, points))) @ change
    there = textbook_harmonics(degree, tuple(map(tuple, turned))) @ change
    wigner, *_ = np.linalg.lstsq(here, there, rcond=None)

    residue = np.abs(wigner @ wigner.T - np.eye(2 * degree + 1)).max()
    if residue > 1e-9:
        raise AssertionError(
            f"the Wigner matrix for degree {degree} came out non-orthogonal by "
            f"{residue:.3e}, which means the argument was not a proper "
            f"rotation. Its determinant is {np.linalg.det(rotation):.6f}."
        )
    return np.ascontiguousarray(wigner)


@cache
def symmetric_matrix_basis() -> np.ndarray:
    """A symmetric 3x3 matrix as ``0e+2e``, and back.

    Row ``m`` is the matrix component ``m`` of the six spherical components
    stands for, so a spherical row ``t`` is the matrix
    ``einsum("mij,m->ij", basis, t)``, and since the rows are orthonormal the
    same table read the other way takes a symmetric matrix to its six
    components. The first row is the trace part, ``I / sqrt(3)``.

    These are the 3j tables coupling two vectors to degrees 0 and 2, scaled to
    unit norm per row: the matrix ``v w^T`` of two vectors is exactly that
    coupling. The components are therefore in the project's basis, the one the
    harmonics and every readout use, which is also the frozen tree's.

    Returns:
        ``(6, 3, 3)``, read only.
    """
    rows = [
        np.moveaxis(wigner_3j_real(1, 1, degree), -1, 0) * math.sqrt(2 * degree + 1)
        for degree in (0, 2)
    ]
    basis = np.concatenate(rows, axis=0)
    basis.setflags(write=False)
    return basis
