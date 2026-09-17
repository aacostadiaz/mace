"""Tensorizing the graph. The schema stays in `mace_core`.

This module binds dtype **names** to torch dtypes and turns the numpy
collation into tensors. It does not redefine the schema, and it does not own
one: two definitions of the same table is how the torch and the jax sides end
up disagreeing about what a field means.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from mace_core.graph import GRAPH_SCHEMA, GraphInfo
from mace_core.graph import collate as collate_numpy
from torch import Tensor

__all__ = ["DTYPES", "collate", "to_tensors"]

#: The one place a dtype name becomes a torch dtype.
DTYPES: dict[str, torch.dtype] = {
    "float64": torch.float64,
    "float32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def to_tensors(
    graph: Mapping[str, Any], *, float_dtype: str = "float64"
) -> dict[str, Tensor]:
    """Bind a numpy graph to torch, dtype by dtype from the schema.

    Args:
        graph: The fields, as numpy.
        float_dtype: Which precision the floating fields take. Named rather
            than read from ``torch.get_default_dtype()``, because a module that
            reads the global default at construction is the seam the precision
            configuration exists to remove.
    """
    out: dict[str, Tensor] = {}
    for name, value in graph.items():
        declared = GRAPH_SCHEMA[name].dtype
        dtype = DTYPES[float_dtype if declared.startswith("float") else declared]
        out[name] = torch.as_tensor(np.asarray(value), dtype=dtype)
    return out


def collate(
    graphs: Iterable[Mapping[str, Any]], *, float_dtype: str = "float64"
) -> dict[str, Tensor]:
    """Join graphs into a batch of tensors.

    The joining is the reference numpy one; this only binds the result. Keeping
    one implementation is what lets the torch batch be checked against the
    frozen tree's collation bit for bit.
    """
    return to_tensors(collate_numpy(graphs), float_dtype=float_dtype)


def graph_info(graph: Mapping[str, Tensor]) -> GraphInfo:
    """The counts, from ``ptr`` and the shapes. Never from ``batch.max()``."""
    return GraphInfo(
        num_graphs=int(graph["ptr"].numel() - 1),
        num_nodes=int(graph["positions"].shape[0]),
        num_edges=int(graph["edge_index"].shape[1]),
    )
