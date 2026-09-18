"""How a derivative of a declared quantity is named and signed.

The rule is one line: the derivative of a declared quantity ``q`` with respect
to a declared input ``x`` is called ``d_<q>_d_<x>``. Three pairs have a name of
their own, and they are data in the table below rather than branches spread
through the consumers that need them.

The third special case is the reason this grammar is written over declared
inputs rather than over positions and the cell. ``magforces`` is
``-dE/d(magmom)``, computed in the same autograd call as the forces, trained
with its own loss term, and used by the magnetic self-consistent model to drive
its fixed point. A grammar that knew only ``d_<q>_d_pos`` and ``d_<q>_d_cell``
could not express it, and the magnetic work would have had to go around the
abstraction that exists to prevent exactly that.

The sign is the one the reported quantity carries, so that
``reported = sign * d(quantity)/d(input)``. The volume division that turns the
strain derivative into a stress is not a sign and is not here: it belongs to
whatever computes the stress.
"""

from __future__ import annotations

__all__ = [
    "DERIVATION_MODES",
    "SPECIAL_CASES",
    "derivation_mode",
    "derivative_name",
    "derivative_sign",
]

#: ``(quantity, input) -> (name, sign)`` for the three pairs whose name is not
#: the ``d_<q>_d_<x>`` default. Everything else follows the rule.
SPECIAL_CASES: dict[tuple[str, str], tuple[str, int]] = {
    ("energy", "pos"): ("forces", -1),
    ("energy", "cell"): ("stress", +1),
    ("energy", "magmom"): ("magforces", -1),
}


def derivative_name(quantity: str, wrt: str) -> str:
    """The canonical name of ``d(quantity)/d(wrt)``.

    Args:
        quantity: The name of the differentiated observable.
        wrt: The name of the declared input it is differentiated against.

    Returns:
        The special-cased name if the pair has one, otherwise
        ``f"d_{quantity}_d_{wrt}"``.
    """
    special = SPECIAL_CASES.get((quantity, wrt))
    if special is not None:
        return special[0]
    return f"d_{quantity}_d_{wrt}"


def derivative_sign(quantity: str, wrt: str) -> int:
    """The sign the reported derivative carries: ``reported = sign * dq/dx``.

    ``+1`` unless the pair is one of the two negated special cases, forces and
    magnetic forces, which are both the negative gradient of the energy.
    """
    special = SPECIAL_CASES.get((quantity, wrt))
    if special is not None:
        return special[1]
    return 1


#: How a derivative is taken, per input. The mode is **derived from the target**
#: rather than declared beside it: a declaration carrying both could say
#: ``wrt: pos`` and ``derivation: grad_input``, and nothing would catch it. What
#: the target is already determines how the derivative can be taken.
#:
#: * ``autograd`` for the positions, the ordinary force path.
#: * ``grad_strain`` for the cell, which is differentiated through an injected
#:   strain rather than against the cell entries themselves.
#: * ``grad_input`` for every other declared input, which becomes a graph leaf
#:   of its own.
DERIVATION_MODES: dict[str, str] = {"pos": "autograd", "cell": "grad_strain"}


def derivation_mode(wrt: str) -> str:
    """How ``d(anything)/d(wrt)`` is taken.

    Args:
        wrt: The name of the declared input.

    Returns:
        ``autograd``, ``grad_strain`` or ``grad_input``.
    """
    return DERIVATION_MODES.get(wrt, "grad_input")
