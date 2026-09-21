"""The canonical weight layout, which is what makes one checkpoint portable.

One layout means one checkpoint loads into any backend. The frozen tree has the
opposite: each backend stores weights its own way, and five conversion command
line tools exist to move between them.

The canonical form is stated here once:

* **Layout** is ``mul_ir``: the multiplicity index varies slowest.
* **Symmetric-contraction weights** are one flat ``[Z, A, mul]`` array over the
  reduced Clebsch-Gordan basis, in the path order
  :mod:`mace_core.clebsch_gordan` pins. ``Z`` is the element, ``A`` the path,
  ``mul`` the channel.
* **Conversion happens only at the checkpoint boundary.** ``to_canonical`` and
  ``load_canonical`` are called when saving and loading, never in ``forward``.
  A backend that wants another layout permutes once at build time and keeps the
  permuted copy.

The reference backend holds the canonical form directly, so for it both
functions are views.
"""

from __future__ import annotations

__all__ = ["CANONICAL_LAYOUT", "KERNEL_SPEC_VERSION", "canonical_weight_shape"]

#: The version of this contract. A backend records it, and a checkpoint carries
#: it, so a format change is a loud mismatch rather than a silent misread.
KERNEL_SPEC_VERSION = "1.0"

#: The one layout a checkpoint is ever written in.
CANONICAL_LAYOUT = "mul_ir"


def canonical_weight_shape(
    num_elements: int, path_count: int, num_features: int
) -> tuple[int, int, int]:
    """The shape of a symmetric-contraction weight array, ``[Z, A, mul]``.

    The legacy nested per-``(irrep_out, body order)`` tensors are contiguous
    slices of this one along the path axis, in the pinned order, so splitting
    and joining them is free.
    """
    return (num_elements, path_count, num_features)


def contraction_path_order(
    irreps_out: str, correlation: int
) -> tuple[tuple[str, int], ...]:
    """The order the per-``(output irrep, body order)`` pieces are joined in.

    Output irreps in the order the declaration writes them, and body orders
    ascending within each. It is stated here rather than left to whichever loop
    happens to build the pieces, because it is the layout of the flat ``[Z, A,
    mul]`` array and therefore the file format.

    Args:
        irreps_out: The kept output irreps, as a declaration string.
        correlation: The body order the model builds up to.

    Returns:
        One ``(irrep, body order)`` pair per piece, in the joined order.
    """
    from mace_core.clebsch_gordan.irreps import Irreps

    return tuple(
        (str(ir), order)
        for _, ir in Irreps.parse(irreps_out)
        for order in range(1, correlation + 1)
    )


def contraction_path_labels(
    irreps_in: str,
    irreps_out: str,
    correlation: int,
    basis: str = "reduced",
) -> tuple[str, ...]:
    """The coupling-tree label of every path in the flat contraction weights.

    Aligned element for element with the path axis of the canonical ``[Z, A,
    mul]`` array, so ``labels[a]`` names the path whose weights sit at ``A =
    a``. This is what a checkpoint records beside the weights: a reader that
    enumerates paths differently detects it here instead of misreading the
    numbers.

    Args:
        irreps_in: The node features being contracted.
        irreps_out: The kept output irreps.
        correlation: The body order.
        basis: ``"reduced"`` or ``"full"``.

    Returns:
        One written coupling tree per path, in the canonical order.

    Raises:
        ValueError: If ``basis`` is neither name.
    """
    from mace_core.clebsch_gordan.reduced_basis import full_path_labels, path_labels

    if basis not in ("reduced", "full"):
        raise ValueError(f"basis must be 'reduced' or 'full', got {basis!r}")
    read = path_labels if basis == "reduced" else full_path_labels
    return tuple(
        str(tree)
        for target, order in contraction_path_order(irreps_out, correlation)
        for tree in read(irreps_in, order, target)[target]
    )
