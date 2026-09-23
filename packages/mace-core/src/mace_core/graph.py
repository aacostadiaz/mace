"""The graph the model consumes, declared as a table rather than as a class.

The boundary between the data layer and the model is a **flat dictionary**, not
a typed container. That is a decision with two consequences worth stating.

It removes the vendored `torch_geometric` entirely. The frozen tree batches
through `Batch.from_data_list`, and that collation's artifacts, `ptr` and
`batch`, are de facto model inputs: every forward reads them. Keeping the
artifacts and dropping the library means the artifacts have to be specified,
which is what this table does.

And it removes the heuristic promotion at the end of `AtomicData.from_config`,
where anything left over in the properties is coerced with `as_tensor`, cast to
the default dtype if it is floating, and reshaped if it is one-dimensional. A
field is in the schema or it is not.

Shapes are symbolic dimensions and dtypes are names, so this module binds to
neither framework. `mace_torch.graph` turns the names into torch dtypes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = [
    "GRAPH_INPUT_DEFAULTS",
    "GRAPH_SCHEMA",
    "FieldSpec",
    "GraphInfo",
    "GraphValidationError",
    "collate",
    "validate_graph",
]

SymDim = Literal["n_nodes", "n_edges", "n_graphs", "n_elements"]
PadValue = Literal["zero", "one", "self_loop", "fake_graph_tail", "n/a"]


@dataclass(frozen=True)
class FieldSpec:
    """One field of the graph dictionary.

    Attributes:
        shape: Symbolic dimensions and fixed extents, in order.
        dtype: The dtype's **name**. Bound per framework, never a torch dtype.
        required: Whether a graph without it is invalid.
        concatenate: How a batch joins this field. ``"nodes"`` and ``"edges"``
            concatenate along the first axis, ``"graphs"`` stacks one row per
            graph, ``"offset"`` concatenates and shifts the values by the
            running node count, which is what an edge index needs.
        pad_value: What a padding slot carries, for the compiled static-shape
            path.
        doc: What the field is.
    """

    shape: tuple[SymDim | int, ...]
    dtype: str
    required: bool
    concatenate: Literal["nodes", "edges", "graphs", "offset"]
    pad_value: PadValue
    doc: str


GRAPH_SCHEMA: Mapping[str, FieldSpec] = {
    "positions": FieldSpec(
        ("n_nodes", 3),
        "float64",
        True,
        "nodes",
        "zero",
        "Cartesian positions in Angstrom.",
    ),
    "atomic_numbers": FieldSpec(
        ("n_nodes",),
        "int64",
        True,
        "nodes",
        "zero",
        "Element of each node. Carried as the number, not as a one-hot: the "
        "one-hot is a model detail and deriving it once per batch is cheaper "
        "than an argmax in every block, which is what the frozen tree does.",
    ),
    "edge_index": FieldSpec(
        (2, "n_edges"),
        "int64",
        True,
        "offset",
        "self_loop",
        "Sender then receiver. Offset by the running node count when batched, "
        "which is the only field that is.",
    ),
    "shifts": FieldSpec(
        ("n_edges", 3),
        "float64",
        True,
        "edges",
        "zero",
        "Cartesian offset of the receiver's periodic image.",
    ),
    "unit_shifts": FieldSpec(
        ("n_edges", 3),
        "float64",
        False,
        "edges",
        "zero",
        "The same offsets in lattice units.",
    ),
    "cell": FieldSpec(
        ("n_graphs", 3, 3),
        "float64",
        False,
        "graphs",
        "one",
        "One cell per graph, as returned by the neighbour search, which is not "
        "always the physical cell. See mace_core.neighbors.",
    ),
    "pbc": FieldSpec(
        ("n_graphs", 3),
        "bool",
        False,
        "graphs",
        "zero",
        "Which axes are periodic. Reaches the model so that the stress of an "
        "aperiodic graph can be masked rather than divided by an invented "
        "volume.",
    ),
    "batch": FieldSpec(
        ("n_nodes",),
        "int64",
        True,
        "nodes",
        "fake_graph_tail",
        "Which graph each node belongs to. A collation artifact that every "
        "forward reads, so it is a specified field rather than a by-product.",
    ),
    "ptr": FieldSpec(
        ("n_graphs",),
        "int64",
        True,
        "graphs",
        "n/a",
        "Where each graph starts in the node axis, with a final entry at the "
        "total. One longer than the graph count, and the only sanctioned way "
        "to derive that count.",
    ),
    "weight": FieldSpec(
        ("n_graphs",),
        "float64",
        False,
        "graphs",
        "zero",
        "Weight of each graph in the loss. Zero on a padding graph, which is "
        "what makes padding cost nothing in the gradient.",
    ),
    "head": FieldSpec(
        ("n_graphs",),
        "int64",
        False,
        "graphs",
        "zero",
        "Which head each graph belongs to, as a position in the model's head "
        "list. It is data rather than something derived: two structures with "
        "the same elements and the same geometry belong to different heads "
        "when they came from different levels of theory, and the isolated-atom "
        "energies the model subtracts are indexed by it.",
    ),
    "total_charge": FieldSpec(
        ("n_graphs",),
        "float64",
        False,
        "graphs",
        "zero",
        "Total charge of each structure, in units of the elementary charge. "
        "Present only for a model that reads it.",
    ),
    "total_spin": FieldSpec(
        ("n_graphs",),
        "float64",
        False,
        "graphs",
        "one",
        "Spin multiplicity of each structure, 2S + 1, so a singlet is one. "
        "Present only for a model that reads it.",
    ),
    "external_field": FieldSpec(
        ("n_graphs", 3),
        "float64",
        False,
        "graphs",
        "zero",
        "The applied electric field on each structure, in V/Angstrom. Present "
        "only for a model that reads it.",
    ),
}

#: What a structure that does not say has, for each per-structure input a model
#: may read: neutral, a singlet, and no applied field. The value a file leaves
#: out, not a padding value.
GRAPH_INPUT_DEFAULTS: Mapping[str, tuple[float, ...]] = {
    "total_charge": (0.0,),
    "total_spin": (1.0,),
    "external_field": (0.0, 0.0, 0.0),
}


class GraphValidationError(ValueError):
    """A graph dictionary that does not satisfy the schema."""


@dataclass(frozen=True)
class GraphInfo:
    """The three counts, each derived exactly one way.

    Attributes:
        num_graphs: ``len(ptr) - 1``. The only sanctioned derivation: reading
            it from ``batch.max()`` is a data-dependent host read, and it is
            wrong for a batch whose last graph has no nodes.
        num_nodes: ``positions.shape[0]``, padding slots included.
        num_edges: ``edge_index.shape[1]``, padding slots included.
    """

    num_graphs: int
    num_nodes: int
    num_edges: int

    @classmethod
    def of(cls, graph: Mapping[str, Any]) -> GraphInfo:
        return cls(
            num_graphs=int(len(graph["ptr"]) - 1),
            num_nodes=int(np.shape(graph["positions"])[0]),
            num_edges=int(np.shape(graph["edge_index"])[1]),
        )


def validate_graph(graph: Mapping[str, Any]) -> GraphInfo:
    """Check a graph against the schema and return its counts.

    Raises:
        GraphValidationError: Naming the field and what was wrong with it. A
            missing required field, an unknown field, or a shape that
            contradicts the counts. An unknown field is an error rather than
            something carried along, which is what the frozen tree's
            auto-promotion does.
    """
    missing = [
        name
        for name, spec in GRAPH_SCHEMA.items()
        if spec.required and name not in graph
    ]
    if missing:
        raise GraphValidationError(
            f"the graph is missing the required fields {missing}. The schema is "
            f"{sorted(GRAPH_SCHEMA)}."
        )
    unknown = sorted(set(graph) - set(GRAPH_SCHEMA))
    if unknown:
        raise GraphValidationError(
            f"{unknown} are not fields of the graph schema. A field is declared "
            f"or it is absent; nothing is promoted by guessing at its type, "
            f"which is what the frozen tree does with whatever is left over."
        )

    info = GraphInfo.of(graph)
    extents = {
        "n_nodes": info.num_nodes,
        "n_edges": info.num_edges,
        "n_graphs": info.num_graphs,
    }
    for name, value in graph.items():
        spec = GRAPH_SCHEMA[name]
        shape = tuple(np.shape(value))
        if name == "ptr":
            if shape != (info.num_graphs + 1,):
                raise GraphValidationError(
                    f"'ptr' has shape {shape}; it must be one longer than the "
                    f"graph count, so {(info.num_graphs + 1,)}."
                )
            continue
        expected = tuple(
            extents[dim] if isinstance(dim, str) else dim for dim in spec.shape
        )
        if shape != expected:
            raise GraphValidationError(
                f"{name!r} has shape {shape} and the schema says {spec.shape}, "
                f"which with {info} is {expected}."
            )
    return info


def collate(graphs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Join single graphs into a batch, in numpy.

    The reference implementation of the collation, so that the torch one has
    something to be checked against and the jax one has something to copy.

    ``edge_index`` is the only field offset by the running node count. ``batch``
    and ``ptr`` are built here rather than inherited, which is what lets the
    vendored library go.
    """
    graphs = list(graphs)
    if not graphs:
        raise GraphValidationError("there are no graphs to collate")

    out: dict[str, Any] = {}
    present = [name for name in GRAPH_SCHEMA if any(name in g for g in graphs)]
    node_offset = 0
    offsets = []
    for graph in graphs:
        offsets.append(node_offset)
        node_offset += int(np.shape(graph["positions"])[0])

    for name in present:
        if name in ("batch", "ptr"):
            continue
        spec = GRAPH_SCHEMA[name]
        pieces = [g[name] for g in graphs if name in g]
        if len(pieces) != len(graphs):
            raise GraphValidationError(
                f"{name!r} is present in some graphs and not others. A batch "
                f"cannot be half-labelled: give every graph the field or none."
            )
        if spec.concatenate == "offset":
            out[name] = np.concatenate(
                [
                    np.asarray(p) + offset
                    for p, offset in zip(pieces, offsets, strict=True)
                ],
                axis=-1,
            )
        elif spec.concatenate == "graphs":
            out[name] = np.stack([np.asarray(p) for p in pieces])
        else:
            out[name] = np.concatenate([np.asarray(p) for p in pieces], axis=0)

    counts = [int(np.shape(g["positions"])[0]) for g in graphs]
    out["batch"] = np.repeat(np.arange(len(graphs)), counts)
    out["ptr"] = np.concatenate([[0], np.cumsum(counts)])
    return out
