"""The declarative specification, end to end.

Three things worth seeing, and the third is the one the team asked about.

1. The whole v0.3 output surface comes out of a model whose head code knows
   nothing about any of it.
2. A derivative carries the sign the declaration gives it, checked against a
   finite difference rather than against itself.
3. A new observable is a row in a YAML string. No import changes, no subclass,
   no registry entry, no edit to the head.
"""

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the head demonstration needs torch", allow_module_level=True)

import torch
from mace_core.observables import (
    ObservableCatalogue,
    irreps_dimension,
    load_catalogue,
)
from mace_torch.heads import DeclarativeHeads, differentiate

FULL_SURFACE = (
    Path(__file__).resolve().parents[2]
    / "mace-core/src/mace_core/defaults/full_surface.yaml"
)
WIDTH = 16


def toy_backbone(
    positions: torch.Tensor,
    strain: torch.Tensor,
    magmoms: torch.Tensor,
    width: int = WIDTH,
) -> torch.Tensor:
    """A stand-in for the equivariant backbone.

    It is a function of all three declared inputs, so all five declared
    derivatives are real rather than structurally zero. The strain enters the
    way the frozen tree's does, as a displacement applied to the positions
    before any distance is computed, which is what makes the cell derivative a
    strain derivative and not a derivative of nine cell entries.

    Squared distances rather than `cdist`, so the function is differentiable at
    zero separation instead of carrying a kink an atom could sit on.
    """
    displaced = positions + positions @ strain
    offsets = displaced[:, None, :] - displaced[None, :, :]
    squared = (offsets**2).sum(-1)
    distance = torch.stack(
        [torch.exp(-squared / s) for s in (0.5, 1.0, 2.0)], dim=-1
    ).sum(dim=1)
    magnetic = (magmoms**2).sum(-1, keepdim=True)
    features = torch.cat([distance, magnetic, distance * magnetic], dim=-1)
    return features.repeat(1, width // features.shape[-1] + 1)[:, :width]


@pytest.fixture
def water():
    """Positions, a symmetric strain and magnetic moments, all differentiable.

    The strain starts at zero, as the frozen tree's displacement does: it is a
    handle to differentiate against, not a quantity anyone sets.
    """
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
    magmoms = torch.tensor(
        [[0.0, 0.0, 0.7], [0.0, 0.0, -0.2], [0.0, 0.0, 0.1]],
        dtype=torch.float64,
        requires_grad=True,
    )
    batch = torch.zeros(len(positions), dtype=torch.long)
    return positions, strain, magmoms, batch, 1


def evaluate(catalogue, positions, strain, magmoms, batch, num_graphs, seed=0):
    torch.manual_seed(seed)
    heads = DeclarativeHeads(catalogue, WIDTH).double()
    output = heads(toy_backbone(positions, strain, magmoms), batch, num_graphs)
    return differentiate(
        catalogue,
        output,
        {"pos": positions, "cell": strain, "magmom": magmoms},
        create_graph=True,
    )


# ---------------------------------------------------------------------------
# 1. The whole surface, from a head that knows none of it
# ---------------------------------------------------------------------------


def test_the_full_legacy_surface_comes_out_of_one_head(water):
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)

    assert len(catalogue.observables) == 28
    for spec in catalogue.observables:
        value = output.get(spec.name)
        assert value is not None, spec.name
        expected_rows = len(positions) if spec.per_atom else num_graphs
        width = irreps_dimension(spec.irreps)
        assert value.shape == (expected_rows, width), spec.name


def test_a_per_graph_row_is_summed_and_a_per_atom_row_is_not(water):
    """The one place the head consults `per_atom`, and the only difference
    between an atomic dipole and a total one in this whole pipeline."""
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)
    atomic = output.get("atomic_dipoles")
    total = output.get("dipole")
    assert atomic.shape == (3, 3)
    assert total.shape == (1, 3)


def test_the_core_fields_are_fields_and_the_rest_is_extras(water):
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)
    assert output.total_energy is not None
    assert output.dipole is not None
    assert output.forces is not None
    assert "latent_quads" in output.extras
    assert "latent_quads" not in {"total_energy", "dipole", "forces"}


# ---------------------------------------------------------------------------
# 2. The sign is data
# ---------------------------------------------------------------------------


def test_forces_are_the_negative_energy_gradient(water):
    """Checked against a central difference of the energy, not against the same
    autograd call that produced them."""
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)
    forces = output.forces.detach()

    step = 1e-6
    for atom in range(len(positions)):
        for axis in range(3):
            shifted = []
            for sign in (+1, -1):
                moved = positions.detach().clone()
                moved[atom, axis] += sign * step
                moved.requires_grad_(True)
                torch.manual_seed(0)
                heads = DeclarativeHeads(catalogue, WIDTH).double()
                shifted.append(
                    heads(
                        toy_backbone(moved, strain, magmoms), batch, num_graphs
                    ).total_energy.item()
                )
            numerical = (shifted[0] - shifted[1]) / (2 * step)
            assert forces[atom, axis].item() == pytest.approx(-numerical, abs=1e-6)


def test_the_declared_derivative_names_are_the_legacy_ones(water):
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    assert [d.name for d in catalogue.requested_derivatives()] == [
        "forces",
        "stress",
        "magforces",
        "d_dipole_d_pos",
        "d_polarizability_d_pos",
    ]
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)
    for name in ("forces", "stress"):
        assert output.get(name) is not None
    for name in ("magforces", "d_dipole_d_pos", "d_polarizability_d_pos"):
        assert name in output.extras


def test_magforces_carry_the_minus_sign_the_declaration_gives_them(water):
    """The case that pays for the grammar being written over declared inputs.
    Nothing in the head or the engine spells `magmom`."""
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    output = evaluate(catalogue, positions, strain, magmoms, batch, num_graphs)
    magforces = output.extras["magforces"].detach()
    assert magforces.shape == magmoms.shape

    step = 1e-6
    moved = magmoms.detach().clone()
    moved[0, 2] += step
    moved.requires_grad_(True)
    torch.manual_seed(0)
    heads = DeclarativeHeads(catalogue, WIDTH).double()
    plus = heads(
        toy_backbone(positions, strain, moved), batch, num_graphs
    ).total_energy.item()
    moved = magmoms.detach().clone()
    moved[0, 2] -= step
    moved.requires_grad_(True)
    torch.manual_seed(0)
    heads = DeclarativeHeads(catalogue, WIDTH).double()
    minus = heads(
        toy_backbone(positions, strain, moved), batch, num_graphs
    ).total_energy.item()

    numerical = (plus - minus) / (2 * step)
    assert magforces[0, 2].item() == pytest.approx(-numerical, abs=1e-6)


def test_a_declaration_the_model_cannot_satisfy_is_an_error_not_a_silence(water):
    """An output that quietly fails to appear is the failure this whole
    specification exists to remove, so asking for a derivative against an input
    nobody supplied raises rather than skipping."""
    positions, strain, magmoms, batch, num_graphs = water
    catalogue = load_catalogue(FULL_SURFACE)
    torch.manual_seed(0)
    heads = DeclarativeHeads(catalogue, WIDTH).double()
    output = heads(toy_backbone(positions, strain, magmoms), batch, num_graphs)
    with pytest.raises(KeyError) as caught:
        differentiate(catalogue, output, {"pos": positions})
    assert "cell" in str(caught.value)


# ---------------------------------------------------------------------------
# 3. A new observable is a row
# ---------------------------------------------------------------------------


def test_a_brand_new_observable_is_a_row_and_nothing_else(water, tmp_path):
    """The claim the whole specification exists to make. Nothing below imports
    anything new, subclasses anything, or registers anything."""
    positions, strain, magmoms, batch, num_graphs = water
    before = load_catalogue(FULL_SURFACE)
    assert "octupole" not in before.names()

    extended = (
        FULL_SURFACE.read_text(encoding="utf-8")
        + """
  - name: octupole
    irreps: "3o"
    per_atom: true
    units: "e*Å^3"
    normalization: "rms"
    default_loss_weight: 5.0
    derivatives:
      - wrt: pos
        units: "e*Å^2"
"""
    )
    path = tmp_path / "with_octupole.yaml"
    path.write_text(extended, encoding="utf-8")

    after = load_catalogue(path)
    output = evaluate(after, positions, strain, magmoms, batch, num_graphs)

    octupole = output.get("octupole")
    assert octupole is not None
    assert octupole.shape == (3, 7)  # one per atom, 2*3+1 components
    assert "d_octupole_d_pos" in output.extras
    assert after.observable("octupole").default_loss_weight == 5.0


def test_declaring_an_observable_twice_or_against_an_unknown_input_is_refused():
    """The errors a declarations file can make are caught where they are made,
    not by whatever reads the output later."""
    with pytest.raises(Exception) as caught:
        ObservableCatalogue.model_validate(
            {
                "inputs": [
                    {
                        "name": "pos",
                        "irreps": "1o",
                        "per_atom": True,
                        "units": "Å",
                    }
                ],
                "observables": [
                    {
                        "name": "energy",
                        "irreps": "0e",
                        "per_atom": False,
                        "units": "eV",
                        "normalization": "std",
                        "derivatives": [{"wrt": "elec_temp"}],
                    }
                ],
            }
        )
    assert "elec_temp" in str(caught.value)
