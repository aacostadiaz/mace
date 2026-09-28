"""How features sit inside each irrep term, as one object the model is handed.

The chain of ops runs in one layout, chosen once when the backend is resolved:
the accelerated backend's native one, which the reference can follow for free.
The model never branches on which it is. The few places outside the ops that
have to look inside a term, a gate per channel, a norm per copy, an input
written into the features, a value handed to a user, ask this object for views
and it answers in the chain's layout. This module and each backend's ops are
the only places the two layouts differ.

Within one term of multiplicity ``mul`` and dimension ``d``:

* ``mul_ir`` holds ``[mul, d]``, copies outermost. It is the canonical layout:
  checkpoints and everything handed to a user are in it.
* ``ir_mul`` holds ``[d, mul]``, components outermost. It is what
  cuEquivariance computes in.

For scalars and for terms with one copy the two are the same array.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import ActivationLayout
from torch import Tensor

__all__ = ["CANONICAL", "Layout", "Terms", "layout_of"]

#: ``(mul, d)`` of every term of a declaration, in order.
Terms = tuple[tuple[int, int], ...]


class Layout:
    """One feature layout, and the views the model needs in it.

    Args:
        name: ``"mul_ir"`` or ``"ir_mul"``.
    """

    def __init__(self, name: ActivationLayout) -> None:
        if name not in ("mul_ir", "ir_mul"):
            raise ValueError(
                f"{name!r} is not a feature layout. The layouts are 'mul_ir' "
                f"and 'ir_mul'."
            )
        self.name: ActivationLayout = name
        self._components_outermost = name == "ir_mul"

    def __repr__(self) -> str:
        return f"Layout({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Layout) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @property
    def is_canonical(self) -> bool:
        return not self._components_outermost

    @staticmethod
    def terms(irreps: str) -> Terms:
        """``(mul, d)`` of every term of ``irreps``, for the methods below.

        Called once, when a module is built. The methods take this rather than
        the declaration so that nothing is parsed in ``forward``, which a
        compiled graph could not trace and which would cost every call.
        """
        return _terms(irreps)

    def blocks(self, features: Tensor, terms: Terms) -> list[Tensor]:
        """One ``[n, mul, d]`` view per term.

        ``features`` is ``[n, dim]`` in this layout. The views share its
        storage; in ``ir_mul`` they are transposed views, so a reshape of one
        copies.
        """
        nodes = features.shape[0]
        pieces, offset = [], 0
        for mul, dimension in terms:
            block = features[:, offset : offset + mul * dimension]
            if self._components_outermost:
                pieces.append(block.reshape(nodes, dimension, mul).transpose(1, 2))
            else:
                pieces.append(block.reshape(nodes, mul, dimension))
            offset += mul * dimension
        return pieces

    def join(self, blocks: list[Tensor]) -> Tensor:
        """The inverse of :meth:`blocks`: ``[n, mul, d]`` pieces to ``[n, dim]``."""
        nodes = blocks[0].shape[0]
        if self._components_outermost:
            blocks = [block.transpose(1, 2) for block in blocks]
        return torch.cat(
            [block.reshape(nodes, block.shape[1] * block.shape[2]) for block in blocks],
            dim=-1,
        )

    def channel_major(
        self, features: Tensor, terms: Terms, num_features: int
    ) -> Tensor:
        """``[n, num_features * dim]`` over ``num_features`` copies of each of
        one channel's ``terms``, as ``[n, num_features, dim]``.

        What a kernel that works channel by channel wants. Each term's copies
        are channel outermost, so every block splits into its channels by a
        view, and only the join copies. Its backward is a split, rather than
        the accumulating scatter a gather by index differentiates into.
        """
        nodes = features.shape[0]
        expanded = tuple((num_features * mul, dimension) for mul, dimension in terms)
        blocks = self.blocks(features, expanded)
        return torch.cat(
            [
                block.reshape(nodes, num_features, mul * dimension)
                for block, (mul, dimension) in zip(blocks, terms, strict=True)
            ],
            dim=-1,
        )

    def grouped(self, channels: Tensor, terms: Terms) -> Tensor:
        """The inverse of :meth:`channel_major`: ``[n, num_features, dim]``
        over one channel's ``terms`` back to ``[n, num_features * dim]``."""
        nodes, num_features = channels.shape[0], channels.shape[1]
        widths = [mul * dimension for mul, dimension in terms]
        pieces = torch.split(channels, widths, dim=-1)
        return self.join(
            [
                piece.reshape(nodes, num_features * mul, dimension)
                for piece, (mul, dimension) in zip(pieces, terms, strict=True)
            ]
        )

    def positions(self, irreps: str) -> np.ndarray:
        """Where each canonical entry of ``irreps`` sits in this layout.

        ``out[k]`` is the index in this layout of entry ``k`` of the ``mul_ir``
        vector. A backend that holds an op as index tables, the reference's
        linear maps, relabels them with this once at build time, so the layout
        costs nothing when it runs.
        """
        return _positions(irreps, self.name)

    def from_canonical(self, features: Tensor, terms: Terms) -> Tensor:
        """A ``mul_ir`` vector, such as an input stream, in this layout."""
        if self.is_canonical or _identical(terms):
            return features
        return self.join(CANONICAL.blocks(features, terms))

    def to_canonical(self, features: Tensor, terms: Terms) -> Tensor:
        """This layout's vector in ``mul_ir``, for anything handed out."""
        if self.is_canonical or _identical(terms):
            return features
        return CANONICAL.join(self.blocks(features, terms))


@cache
def _terms(irreps: str) -> Terms:
    return tuple((mul, ir.dimension) for mul, ir in Irreps.parse(irreps).terms)


@cache
def _positions(irreps: str, name: str) -> np.ndarray:
    out, offset = [], 0
    for mul, dimension in _terms(irreps):
        # Entry (copy u, component m) of the canonical block sits at
        # u * d + m in mul_ir and at m * mul + u in ir_mul.
        if name == "ir_mul":
            span = np.arange(dimension * mul).reshape(dimension, mul).T
        else:
            span = np.arange(mul * dimension).reshape(mul, dimension)
        out.append(offset + span.reshape(-1))
        offset += mul * dimension
    positions = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    return positions.astype(np.int64)


def _identical(terms: Terms) -> bool:
    """Whether the two layouts are the same array for these terms."""
    return all(mul == 1 or dimension == 1 for mul, dimension in terms)


#: The canonical layout.
CANONICAL = Layout("mul_ir")


def layout_of(backend: Any) -> Layout:
    """The layout a model built with ``backend`` runs in.

    A backend that resolved a chain says which; any other builds canonical.
    """
    return getattr(backend, "layout", CANONICAL)
