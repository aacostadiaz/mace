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

**The canonical form carries its normalization folded into the weight.** The
frozen tree keeps a weight drawn from a standard normal and multiplies it by a
per-path factor inside the operation; here the factor is already in the number,
so an operation is a plain contraction and a checkpoint means the same thing
whatever reads it. That makes the factor part of the format rather than part of
an implementation, which is why the scales are stated here and used by every
backend that draws a fresh set of weights.

The factors are what the frozen tree's ``e3nn`` operations apply:

* a **linear** map divides by the square root of the total input multiplicity
  feeding the output irrep;
* the **skip connection's** tensor product against the element attributes
  divides by the square root of the product of the two input multiplicities,
  because ``e3nn``'s per-instruction factor and the coupling of an irrep with a
  scalar cancel the output dimension between them;
* the **symmetric contraction** applies none, so a fresh weight there is a
  standard normal.
"""

from __future__ import annotations

from mace_core.clebsch_gordan.irreps import Irrep, Irreps

__all__ = [
    "CANONICAL_LAYOUT",
    "KERNEL_SPEC_VERSION",
    "canonical_weight_shape",
    "fully_connected_tp_weight_scale",
    "linear_bias_table",
    "linear_weight_scale",
    "linear_weight_table",
    "symmetric_contraction_weight_scale",
]

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


def linear_weight_scale(irreps_in: str, irrep_out: Irrep) -> float:
    """The factor a fresh linear weight writing to ``irrep_out`` carries.

    An equivariant linear map connects a term only to a term of the same irrep,
    so the fan-in of an output copy is the total multiplicity of the inputs
    that share its irrep.

    Returns:
        ``1 / sqrt(fan_in)``, or ``1.0`` when nothing feeds the irrep, which is
        an output the map cannot produce and whose weights do not exist.
    """
    fan_in = sum(
        multiplicity
        for multiplicity, irrep in Irreps.parse(irreps_in)
        if irrep == irrep_out
    )
    return 1.0 if fan_in == 0 else float(fan_in) ** -0.5


def fully_connected_tp_weight_scale(
    multiplicity_in1: int, multiplicity_in2: int
) -> float:
    """The factor a fresh skip-connection weight carries.

    Args:
        multiplicity_in1: The node features' multiplicity for this path.
        multiplicity_in2: How many element attributes, the second input's
            width.
    """
    product = multiplicity_in1 * multiplicity_in2
    return 1.0 if product == 0 else float(product) ** -0.5


def symmetric_contraction_weight_scale() -> float:
    """One. Stated as a function so that a backend reads a scale for every op
    rather than remembering which of the three is the exception."""
    return 1.0


def linear_weight_table(irreps_in: str, irreps_out: str) -> dict[tuple[int, int], int]:
    """Which canonical weight joins each output copy to each input copy.

    A copy is one multiplicity of one term, counted in
    :meth:`~mace_core.clebsch_gordan.irreps.Irreps.slices` order. An
    equivariant linear map joins a copy only to copies of the same irrep, with
    one weight per pair shared by the ``2l + 1`` components, so the table is
    the whole of the map's structure. The order is the layout: output copies
    outermost, and within one of them the matching input copies in
    declaration order.

    Returns:
        ``{(output copy, input copy): weight index}``, for every pair the map
        connects and no other.
    """
    table: dict[tuple[int, int], int] = {}
    weight = 0
    sources = list(Irreps.parse(irreps_in).slices())
    for out_copy, (_, out_irrep) in enumerate(Irreps.parse(irreps_out).slices()):
        for in_copy, (_, in_irrep) in enumerate(sources):
            if in_irrep != out_irrep:
                continue
            table[(out_copy, in_copy)] = weight
            weight += 1
    return table


def linear_bias_table(irreps_out: str) -> dict[int, int]:
    """Which canonical bias entry belongs to each scalar output copy.

    Only an even scalar can carry a bias: adding a constant to anything else
    would pick out a direction. So the bias vector has one entry per ``0e``
    output copy, in output order.

    Returns:
        ``{output copy: bias index}``, for the ``0e`` copies only.
    """
    table: dict[int, int] = {}
    for out_copy, (_, irrep) in enumerate(Irreps.parse(irreps_out).slices()):
        if irrep.degree == 0 and irrep.parity == 1:
            table[out_copy] = len(table)
    return table
