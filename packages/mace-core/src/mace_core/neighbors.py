"""The neighbour list, as a pure function, with the cell regimes stated.

Numpy and matscipy. No torch, no jax: the edge set is a property of the
structure, not of the framework that will consume it.

**The cell this returns is not always the cell it searched with**, and that is
deliberate rather than an implementation detail. matscipy cannot bin atoms
along a non-periodic axis, so the search needs a box there. But the returned
cell is read downstream as a volume: the stress divides by its determinant, and
electrostatic models use it as their k-space box and divide their slab and
molecule corrections by the same determinant. Returning the inflated search box
for a partially periodic system would silently rescale both.

So there are three regimes, and each is a decision:

* **fully periodic** -- search and return the physical cell.
* **partially periodic** -- search with the inflated box, return the physical
  cell, except that a non-periodic axis whose physical row is all zeros keeps
  the inflated row. A slab built with no vacuum would otherwise have a zero
  determinant, which turns the stress into a division by zero.
* **fully aperiodic** -- search and return the inflated box. There is no
  physical cell to return, stress is meaningless and masked away by the model,
  and long-range models need a non-degenerate box to work in at all.

The inflated box is sized from the atoms' **extent** plus the cutoff, not from
the largest absolute coordinate. Sizing it from the coordinates makes it depend
on where the origin happens to be and produces boxes large enough to run an
electrostatic model out of GPU memory, for identical neighbour lists.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NeighborList", "get_neighborhood"]


class NeighborList(tuple):
    """What a neighbour search returns.

    Attributes:
        edge_index: ``[2, n_edges]``, sender then receiver.
        shifts: ``[n_edges, 3]``, the Cartesian offset of the receiver's image,
            already multiplied into the returned cell.
        unit_shifts: ``[n_edges, 3]``, the same offsets in lattice units.
        cell: ``[3, 3]`` the cell downstream should use, per the regimes above.
    """

    __slots__ = ()

    def __new__(cls, edge_index, shifts, unit_shifts, cell):
        return super().__new__(cls, (edge_index, shifts, unit_shifts, cell))

    edge_index = property(lambda self: self[0])
    shifts = property(lambda self: self[1])
    unit_shifts = property(lambda self: self[2])
    cell = property(lambda self: self[3])


def _search_box(
    positions: np.ndarray, cell: np.ndarray, pbc: tuple[bool, bool, bool], cutoff: float
) -> np.ndarray:
    """The box to bin in: the physical cell, inflated along non-periodic axes."""
    box = np.array(cell, dtype=float, copy=True)
    identity = np.identity(3, dtype=float)
    for axis in range(3):
        if not pbc[axis]:
            extent = float(positions[:, axis].max() - positions[:, axis].min())
            box[axis, :] = (extent + 2.0 * cutoff + 1.0) * identity[axis, :]
    return box


def get_neighborhood(
    positions: np.ndarray,
    cutoff: float,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: np.ndarray | None = None,
    true_self_interaction: bool = False,
) -> NeighborList:
    """Every edge within ``cutoff``, and the cell to interpret them in.

    Args:
        positions: ``[n_atoms, 3]`` in Angstrom.
        cutoff: In Angstrom.
        pbc: Which axes are periodic. ``None`` means none.
        cell: ``[3, 3]`` lattice vectors as rows. ``None`` or all zeros means
            there is no cell, and the identity stands in before inflation.
        true_self_interaction: Keep the zero-shift self edge. Off by default,
            since an atom is not its own neighbour; a self edge across a
            periodic boundary is a real neighbour and is kept either way.

    Returns:
        A :class:`NeighborList`.

    The caller's cell is never written to. The frozen tree defends against that
    with a deepcopy at the call site, which has been vestigial since the
    function stopped mutating, and is not reproduced here.
    """
    # New names rather than writing over the parameters: `pbc` arrives as any
    # sequence and is used as a three-tuple, and `cell` arrives possibly absent
    # and is used as a 3x3 array, so reassigning them makes each name mean two
    # things in one function.
    given = (False, False, False) if pbc is None else [bool(flag) for flag in pbc]
    if len(given) != 3:
        raise ValueError(f"pbc must have three entries, got {len(given)}")
    first, second, third = given
    flags: tuple[bool, bool, bool] = (first, second, third)

    positions = np.asarray(positions, dtype=float)
    # An absent or all-zero cell means there is no cell. The frozen tree writes
    # this as `cell.any() == np.zeros((3, 3)).any()`, which is true exactly
    # when the cell is all zeros and reads as if it compared two cells.
    if cell is None or not np.asarray(cell).any():
        lattice = np.identity(3, dtype=float)
    else:
        lattice = np.asarray(cell, dtype=float).reshape(3, 3)

    search_box = _search_box(positions, lattice, flags, cutoff)

    if any(flags):
        returned = np.array(lattice, dtype=float, copy=True)
        for axis in range(3):
            if not flags[axis] and not returned[axis].any():
                returned[axis] = search_box[axis]
    else:
        returned = search_box

    from matscipy.neighbours import neighbour_list

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS",
        pbc=flags,
        cell=search_box,
        positions=positions,
        cutoff=cutoff,
    )

    if not true_self_interaction:
        self_edge = (sender == receiver) & np.all(unit_shifts == 0, axis=1)
        keep = ~self_edge
        sender, receiver, unit_shifts = sender[keep], receiver[keep], unit_shifts[keep]

    return NeighborList(
        np.stack((sender, receiver)),
        unit_shifts @ returned,
        unit_shifts,
        returned,
    )
