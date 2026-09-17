"""The radial bases, against the frozen tree and against their own derivatives.

Values are pinned bit for bit, because these are closed forms evaluated the
same way; a tolerance would be admitting a difference nobody had looked at.

Derivatives are pinned separately, and that is not redundant. A basis whose
values are right and whose derivative is silently zero produces forces that are
wrong by exactly the term it dropped, and nothing about the values would show
it. One of the four does exactly that; see the test that records it.
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("needs torch", allow_module_level=True)

import torch
from mace_torch.nn.radial import (
    BesselBasis,
    ChebyshevBasis,
    GaussianBasis,
    PolynomialCutoff,
)


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture
def lengths():
    return torch.rand(50, 1) * 4.5 + 0.2


DIFFERENTIABLE = [
    ("bessel", lambda: BesselBasis(5.0, 8)),
    ("gaussian", lambda: GaussianBasis(5.0, 16)),
    ("cutoff", lambda: PolynomialCutoff(5.0, 6)),
]


@pytest.mark.parametrize(
    ("name", "build"), DIFFERENTIABLE, ids=[n for n, _ in DIFFERENTIABLE]
)
def test_the_values_match_the_frozen_tree_bit_for_bit(name, build, lengths):
    legacy = pytest.importorskip("mace.modules.radial")
    frozen = {
        "bessel": lambda: legacy.BesselBasis(5.0, 8),
        "gaussian": lambda: legacy.GaussianBasis(5.0, 16),
        "cutoff": lambda: legacy.PolynomialCutoff(5.0, 6),
    }[name]()
    assert torch.equal(build()(lengths), frozen(lengths))


def test_the_chebyshev_values_match_the_frozen_tree_too(lengths):
    legacy = pytest.importorskip("mace.modules.radial")
    assert torch.equal(
        ChebyshevBasis(5.0, 8)(lengths), legacy.ChebychevBasis(5.0, 8)(lengths)
    )


@pytest.mark.parametrize(
    ("name", "build"), DIFFERENTIABLE, ids=[n for n, _ in DIFFERENTIABLE]
)
def test_they_differentiate_twice(name, build, lengths):
    """Force training differentiates the energy, and training on forces
    differentiates that again."""
    module = build()
    argument = lengths.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(module, (argument,))
    assert torch.autograd.gradgradcheck(module, (argument,))


def test_the_chebyshev_basis_carries_no_derivative_and_the_frozen_one_does_not_either():
    """A defect of the frozen tree, reproduced here only so it is written down.

    `torch.special.chebyshev_polynomial_t` returns a tensor with
    `requires_grad` false, so the derivative of the polynomial with respect to
    the distance is silently zero. `--radial_type chebyshev` is reachable from
    the command line and the inventory carries it as KEEP, pinned by a values
    test alone.

    The chain downstream is not fully broken, which is what makes it quiet: the
    cutoff envelope is differentiable, so the embedding still carries a
    gradient. What is missing is the product-rule term through the basis
    itself, and a force is wrong by exactly that.

    This test asserts the current behaviour rather than the desired one, so that
    fixing it is a visible change here rather than a surprise.
    """
    legacy = pytest.importorskip("mace.modules.radial")
    argument = (torch.rand(10, 1) * 4 + 0.3).requires_grad_(True)

    assert not ChebyshevBasis(5.0, 8)(argument).requires_grad
    assert not legacy.ChebychevBasis(5.0, 8)(argument).requires_grad
    assert not torch.special.chebyshev_polynomial_t(
        argument, torch.full_like(argument, 3.0)
    ).requires_grad


def test_one_chebyshev_class_covers_what_the_frozen_tree_splits_in_two():
    """The frozen tree has ChebychevBasis and ChebyshevBasisGeneral computing
    the same polynomials and differing in whether the constant term is there."""
    lengths = torch.rand(7, 1)
    without = ChebyshevBasis(5.0, 4, include_constant=False)(lengths)
    with_constant = ChebyshevBasis(5.0, 4, include_constant=True)(lengths)
    assert torch.allclose(with_constant[:, 0], torch.ones_like(with_constant[:, 0]))
    assert torch.allclose(with_constant[:, 1:], without[:, :3])


def test_the_cutoff_is_zero_with_zero_slope_at_the_radius():
    """The property that keeps a force continuous as an atom crosses the
    radius. Without it the discontinuity shows up as energy drift in a
    simulation rather than as an error."""
    cutoff = PolynomialCutoff(5.0, 6)
    at_radius = torch.tensor([[5.0]], requires_grad=True)
    value = cutoff(at_radius)
    value.backward()
    assert float(value) == 0.0
    assert float(at_radius.grad) == 0.0


def test_the_precision_is_an_argument_rather_than_the_process_default():
    """Every frozen class reads torch.get_default_dtype() while it is being
    constructed, so a module's precision depends on what the process last set.
    That is the seam the precision configuration replaces."""
    torch.set_default_dtype(torch.float32)
    assert BesselBasis(5.0, 8, dtype="float64").bessel_weights.dtype == torch.float64
    assert (
        GaussianBasis(5.0, 8, dtype="float32").gaussian_weights.dtype == torch.float32
    )


def test_a_trainable_basis_exposes_its_frequencies_as_parameters():
    assert isinstance(
        BesselBasis(5.0, 8, trainable=True).bessel_weights, torch.nn.Parameter
    )
    assert not isinstance(
        BesselBasis(5.0, 8, trainable=False).bessel_weights, torch.nn.Parameter
    )
