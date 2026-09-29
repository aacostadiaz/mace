"""Write the canonical-layout golden: path order and per-path scale.

The flat ``[Z, A, mul]`` contraction weights are the file format, so what each
position on the path axis means is committed rather than recomputed and
trusted. For every point of the grid this records, in canonical order, each
path's coupling-tree label and its scale, the Frobenius norm of its basis
tensor, which is the per-path normalization the layout pins.

Run from the repository root, only when the convention is deliberately changed,
which is a schema-version bump:

    python packages/mace-core/tests/make_canonical_layout_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_core.kernels.canonical import (
    KERNEL_SPEC_VERSION,
    contraction_path_labels,
    contraction_path_order,
)

GRID = [
    ("0e+1o", "0e+1o", 3),
    ("0e+1o+2e", "0e+1o", 3),
    ("0e+1o+2e+3o", "0e+1o", 3),
    ("0e+1o+2e+3o", "0e+1o+2e", 3),
    ("0e+1o+2e", "0e+1o", 4),
]

GOLDEN = Path(__file__).with_name("canonical_layout.json")


def layout(irreps_in: str, irreps_out: str, correlation: int) -> list[dict]:
    labels = contraction_path_labels(irreps_in, irreps_out, correlation)
    scales = []
    for target, order in contraction_path_order(irreps_out, correlation):
        basis = reduced_symmetric_tensor_product_basis(irreps_in, order, target)[target]
        scales.extend(
            float(np.linalg.norm(basis[path])) for path in range(basis.shape[0])
        )
    return [
        {"label": label, "scale": scale}
        for label, scale in zip(labels, scales, strict=True)
    ]


def main() -> None:
    document = {
        "spec_version": KERNEL_SPEC_VERSION,
        "cases": [
            {
                "irreps_in": irreps_in,
                "irreps_out": irreps_out,
                "correlation": correlation,
                "paths": layout(irreps_in, irreps_out, correlation),
            }
            for irreps_in, irreps_out, correlation in GRID
        ],
    }
    GOLDEN.write_text(json.dumps(document, indent=1) + "\n")


if __name__ == "__main__":
    main()
