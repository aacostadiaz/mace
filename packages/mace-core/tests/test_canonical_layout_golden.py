"""The canonical contraction layout against its committed golden.

What each position on the path axis of the ``[Z, A, mul]`` weights means is
the file format. This fails when a change to the basis, the enumeration or the
normalization would make a checkpoint on disk mean something else, which is
exactly the change that has to be a deliberate schema-version bump rather than
an accident.
"""

import json

import pytest
from mace_core.kernels.canonical import KERNEL_SPEC_VERSION
from make_canonical_layout_golden import GOLDEN, GRID, layout

DOCUMENT = json.loads(GOLDEN.read_text())
CASES = {
    (case["irreps_in"], case["irreps_out"], case["correlation"]): case["paths"]
    for case in DOCUMENT["cases"]
}


def test_the_golden_covers_the_grid_and_the_current_spec():
    assert set(CASES) == set(GRID)
    assert DOCUMENT["spec_version"] == KERNEL_SPEC_VERSION


@pytest.mark.parametrize("case", GRID, ids=lambda case: "-".join(map(str, case)))
def test_the_path_order_is_the_committed_one(case):
    assert [path["label"] for path in layout(*case)] == [
        path["label"] for path in CASES[case]
    ]


@pytest.mark.parametrize("case", GRID, ids=lambda case: "-".join(map(str, case)))
def test_each_path_s_scale_is_the_committed_one(case):
    computed = [path["scale"] for path in layout(*case)]
    committed = [path["scale"] for path in CASES[case]]
    assert computed == pytest.approx(committed, rel=1e-12, abs=1e-14)
