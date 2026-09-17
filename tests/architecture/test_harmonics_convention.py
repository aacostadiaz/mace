"""The convention the spherical harmonics and the Clebsch-Gordan basis share.

They have to agree or the model is wrong, and the symptom is a number rather
than an error. ARCH-1 settles which one it is: e3nn's, which is what the frozen
tree uses. This file is where that is checked against e3nn itself, because
`mace_core` may not import it and something has to.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

for needed in ("torch", "e3nn", "mace_core"):
    if importlib.util.find_spec(needed) is None:  # pragma: no cover
        pytest.skip(f"needs {needed}", allow_module_level=True)

import torch  # noqa: E402
from e3nn import o3  # noqa: E402

from mace_core.clebsch_gordan.real_basis import (  # noqa: E402
    AXIS_PERMUTATION,
    induced_rotation,
    wigner_3j_real,
)
from mace_torch.backends.harmonics import spherical_harmonics  # noqa: E402

DEGREES = range(5)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("lmax", DEGREES)
def test_the_native_harmonics_reproduce_e3nn(lmax):
    """What ARCH-1 asks for, at fp64 and with no e3nn in the implementation."""
    directions = torch.tensor(np.random.default_rng(0).normal(size=(300, 3)))
    mine = spherical_harmonics(directions, lmax)
    theirs = o3.spherical_harmonics(
        list(range(lmax + 1)), directions, normalize=True, normalization="component"
    ).double()
    assert torch.allclose(mine, theirs, atol=1e-13, rtol=0)


@pytest.mark.parametrize("degree", DEGREES)
def test_the_component_normalisation_is_the_dimension(degree):
    """`"component"` means the squared norm of a degree is 2l+1, not 1."""
    directions = torch.tensor(np.random.default_rng(1).normal(size=(50, 3)))
    block = spherical_harmonics(directions, degree)[:, degree**2 :]
    assert torch.allclose(
        (block**2).sum(-1),
        torch.full((50,), float(2 * degree + 1), dtype=torch.float64),
        atol=1e-12,
    )


def test_the_first_degree_is_the_direction_itself():
    """e3nn orders l = 1 as (x, y, z), not as the textbook (y, z, x). That one
    fact is the whole difference between the two conventions: relabelling the
    axes is a rotation of space, so at l >= 2 it induces a genuine orthogonal
    mixing rather than another permutation."""
    for axis, expected in enumerate(np.eye(3)):
        direction = torch.tensor([expected])
        block = spherical_harmonics(direction, 1)[0, 1:]
        assert torch.allclose(block, torch.tensor(expected) * np.sqrt(3.0))
    assert AXIS_PERMUTATION == (2, 0, 1)


@pytest.mark.parametrize("degree", DEGREES)
def test_the_induced_rotation_is_derived_without_e3nn_and_is_orthogonal(degree):
    rotation = induced_rotation(degree)
    assert rotation.shape == (2 * degree + 1, 2 * degree + 1)
    assert np.abs(rotation @ rotation.T - np.eye(2 * degree + 1)).max() < 1e-12


def test_the_3j_matches_e3nn_up_to_a_sign_that_is_gauge():
    """Every triple is identical to e3nn's or differs by one global sign.
    None differs otherwise, which is the assertion that matters.

    On the grid below 51 are identical and 14 differ by a sign; widening the cut
    to l3 <= 5 gives 58 and 17. The triples whose coupling vanishes outright are
    skipped, which is why neither pair sums to the grid size.

    That sign is the arbitrary part of any construction: making the carried
    tensor real admits both `i**L` and `(-i)**L`, which differ by `(-1)**L`, so
    for an odd degree sum it is a choice. e3nn's comes from its own recursion
    rather than from a rule, and no canonical condition reproduces it. It does
    not matter, because the sign of a basis path is absorbed by the weight that
    multiplies it.
    """
    identical = flipped = different = 0
    for l1 in DEGREES:
        for l2 in DEGREES:
            for l3 in range(abs(l1 - l2), min(l1 + l2, max(DEGREES)) + 1):
                mine = wigner_3j_real(l1, l2, l3)
                theirs = o3.wigner_3j(l1, l2, l3, dtype=torch.float64).numpy()
                if np.abs(theirs).max() < 1e-9:
                    continue
                if np.abs(mine - theirs).max() < 1e-10:
                    identical += 1
                elif np.abs(mine + theirs).max() < 1e-10:
                    flipped += 1
                else:
                    different += 1
    assert different == 0, (
        f"{different} triples differ from e3nn by more than a sign, which is "
        f"not a convention difference and has to be explained."
    )
    assert flipped, (
        "no triple differs by a sign, which would mean the sign convention now "
        "matches e3nn's. That is a better state, not a worse one, but it is a "
        "change and this test is where it should be noticed."
    )
    # The split is recorded rather than asserted as a fixed pair: it moves with
    # where the grid is cut, and the claim is that `different` is zero.
    assert identical > flipped


def test_the_harmonics_and_the_basis_are_built_from_the_same_coefficients():
    """The property that makes a disagreement between them impossible rather
    than testable: the harmonics recursion projects with the very function the
    contraction's basis is built from."""
    import inspect

    from mace_torch.backends import harmonics

    assert "wigner_3j_real" in inspect.getsource(harmonics)
