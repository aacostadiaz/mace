"""The v1 graph against the frozen tree's, bit for bit.

This is the data exit gate. The frozen tree builds its neighbour list with
matscipy and batches through a vendored `torch_geometric`, whose `ptr` and
`batch` every forward reads. v1 does neither, so the only way to know it agrees
is to build both and compare.

Bit-exact, not to a tolerance. These are integer indices and exactly
representable shifts; a tolerance here would be admitting a disagreement
nobody had looked at.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

for needed in ("torch", "ase", "matscipy", "mace_core"):
    if importlib.util.find_spec(needed) is None:  # pragma: no cover
        pytest.skip(f"needs {needed}", allow_module_level=True)

import torch  # noqa: E402
from ase import Atoms  # noqa: E402

from mace_core.neighbors import get_neighborhood as v1_neighborhood  # noqa: E402

CUTOFF = 3.0


def frames():
    rng = np.random.default_rng(0)
    return [
        ("periodic", Atoms(numbers=[1, 1, 8], positions=rng.random((3, 3)) * 3,
                           cell=np.eye(3) * 6.0, pbc=True)),
        ("slab", Atoms(numbers=[1, 8, 8, 1], positions=rng.random((4, 3)) * 3,
                       cell=np.diag([6.0, 6.0, 20.0]), pbc=[True, True, False])),
        ("molecule", Atoms(numbers=[1, 1, 8], positions=rng.random((3, 3)) * 2,
                           pbc=False)),
    ]


@pytest.mark.parametrize(("label", "atoms"), frames(), ids=[f[0] for f in frames()])
def test_the_neighbour_list_is_identical_to_the_frozen_one(label, atoms):
    from mace.data.neighborhood import get_neighborhood as legacy

    cell = np.array(atoms.get_cell())
    mine = v1_neighborhood(
        atoms.get_positions(), CUTOFF, tuple(atoms.get_pbc()), cell.copy()
    )
    theirs = legacy(atoms.get_positions(), CUTOFF, tuple(atoms.get_pbc()), cell.copy())
    names = ("edge_index", "shifts", "unit_shifts", "cell")
    for name, ours, legacy_value in zip(names, mine, theirs, strict=True):
        assert np.array_equal(ours, legacy_value), (
            f"{name} differs from the frozen tree on the {label} case"
        )


def test_the_batch_is_identical_to_the_vendored_collater():
    """`ptr`, `batch`, `edge_index` and `shifts` against
    `Batch.from_data_list`, which is what v1 removes."""
    from mace import data as legacy_data
    from mace.data.utils import config_from_atoms
    from mace.tools import AtomicNumberTable, torch_geometric
    from mace_torch.graph import collate as v1_collate

    torch.set_default_dtype(torch.float64)
    z_table = AtomicNumberTable([1, 8])
    atoms_list = [atoms for _, atoms in frames() if all(atoms.get_pbc())]
    rng = np.random.default_rng(1)
    for count in (4, 5):
        atoms_list.append(
            Atoms(
                numbers=[1] * (count - 1) + [8],
                positions=rng.random((count, 3)) * 3,
                cell=np.eye(3) * 6.0,
                pbc=True,
            )
        )

    legacy_batch = torch_geometric.batch.Batch.from_data_list(
        [
            legacy_data.AtomicData.from_config(
                config_from_atoms(atoms), z_table=z_table, cutoff=CUTOFF
            )
            for atoms in atoms_list
        ]
    )

    graphs = []
    for atoms in atoms_list:
        neighbourhood = v1_neighborhood(
            atoms.get_positions(),
            CUTOFF,
            tuple(atoms.get_pbc()),
            np.array(atoms.get_cell()),
        )
        graphs.append(
            {
                "positions": atoms.get_positions(),
                "atomic_numbers": atoms.get_atomic_numbers().astype(np.int64),
                "edge_index": neighbourhood.edge_index,
                "shifts": neighbourhood.shifts,
                "unit_shifts": neighbourhood.unit_shifts,
                "cell": neighbourhood.cell,
                "pbc": np.array(atoms.get_pbc()),
            }
        )
    mine = v1_collate(graphs)

    assert torch.equal(mine["ptr"], legacy_batch.ptr)
    assert torch.equal(mine["batch"], legacy_batch.batch)
    assert torch.equal(mine["edge_index"], legacy_batch.edge_index)
    assert torch.equal(
        mine["shifts"], legacy_batch.shifts.reshape(mine["shifts"].shape)
    )


def test_no_file_under_packages_imports_torch_geometric():
    """The vendored library is what this ticket removes, and an import that
    crept back would make the parity above meaningless.

    Checked over the parsed syntax tree rather than the file's text. A textual
    search matches the docstrings that explain what was removed and why, which
    is exactly what the first version of this test did.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "packages"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("torch_geometric" in name for name in names):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders
