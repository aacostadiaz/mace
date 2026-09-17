"""The graph schema, the collation, and the neighbour list's cell regimes.

The cell the neighbour search returns is not always the cell it searched with,
and the three regimes are decisions rather than implementation details. Each
gets a test, because getting one wrong is silent: a stress divided by an
invented volume, or an electrostatic correction scaled away.
"""

import subprocess
import sys

import numpy as np
import pytest
from mace_core.graph import (
    GRAPH_SCHEMA,
    GraphInfo,
    GraphValidationError,
    collate,
    validate_graph,
)
from mace_core.neighbors import get_neighborhood


def single(nodes=3, edges=4, cell=None):
    return {
        "positions": np.arange(nodes * 3, dtype=float).reshape(nodes, 3),
        "atomic_numbers": np.ones(nodes, dtype=np.int64),
        "edge_index": np.zeros((2, edges), dtype=np.int64),
        "shifts": np.zeros((edges, 3)),
        "cell": np.eye(3) if cell is None else cell,
        "pbc": np.array([True, True, True]),
    }


# ---------------------------------------------------------------------------
# The neighbour list
# ---------------------------------------------------------------------------


def test_a_fully_periodic_cell_comes_back_as_it_went_in():
    cell = np.diag([4.0, 5.0, 6.0])
    result = get_neighborhood(
        np.random.default_rng(0).random((4, 3)) * 3, 2.0, (True, True, True), cell
    )
    assert np.array_equal(result.cell, cell)


def test_a_slab_keeps_its_physical_cell_rather_than_the_search_box():
    """The stress divides by this determinant, and an electrostatic model uses
    it as its box. Returning the inflated search box would silently rescale
    both."""
    cell = np.diag([4.0, 4.0, 20.0])
    result = get_neighborhood(
        np.random.default_rng(0).random((4, 3)) * 3, 2.0, (True, True, False), cell
    )
    assert np.array_equal(result.cell, cell)


def test_a_slab_with_no_vacuum_keeps_the_inflated_row_instead_of_a_zero_volume():
    """A zero row would leave det(cell) = 0, which turns the stress into a
    division by zero rather than into an error."""
    cell = np.array([[4.0, 0, 0], [0, 4.0, 0], [0, 0, 0.0]])
    result = get_neighborhood(
        np.random.default_rng(0).random((4, 3)) * 3, 2.0, (True, True, False), cell
    )
    assert abs(float(np.linalg.det(result.cell))) > 0.0


def test_a_fully_aperiodic_structure_gets_the_inflated_box():
    """There is no physical cell to return, the stress is masked away by the
    model, and a long-range model needs a non-degenerate box to work in."""
    positions = np.random.default_rng(0).random((5, 3)) * 2
    result = get_neighborhood(positions, 2.0, (False, False, False), None)
    assert abs(float(np.linalg.det(result.cell))) > 0.0
    assert not np.array_equal(result.cell, np.eye(3))


def test_the_box_is_sized_from_the_extent_and_not_from_the_coordinates():
    """Translating a molecule must not change the box. Sizing it from the
    largest absolute coordinate makes it depend on where the origin is, and
    produces boxes large enough to run an electrostatic model out of memory."""
    positions = np.random.default_rng(0).random((5, 3)) * 2
    here = get_neighborhood(positions, 2.0, (False, False, False), None)
    far = get_neighborhood(positions + 1000.0, 2.0, (False, False, False), None)
    assert np.allclose(here.cell, far.cell)
    assert here.edge_index.shape == far.edge_index.shape


def test_a_self_edge_across_a_boundary_is_a_real_neighbour():
    """One atom in a cell smaller than the cutoff is its own neighbour through
    the images, and dropping those would empty the graph."""
    result = get_neighborhood(
        np.zeros((1, 3)), 2.1, (True, True, True), np.eye(3) * 2.0
    )
    assert result.edge_index.shape[1] == 6


def test_the_callers_cell_is_never_written_to():
    cell = np.diag([4.0, 4.0, 20.0])
    before = cell.copy()
    get_neighborhood(np.zeros((2, 3)), 2.0, (True, True, False), cell)
    assert np.array_equal(cell, before)


def test_pbc_must_have_three_entries():
    with pytest.raises(ValueError, match="three entries"):
        get_neighborhood(np.zeros((1, 3)), 2.0, (True, True), np.eye(3))


# ---------------------------------------------------------------------------
# The schema
# ---------------------------------------------------------------------------


def test_the_collation_artifacts_are_declared_fields():
    """`ptr` and `batch` are by-products of the vendored library's batching in
    the frozen tree, and every forward reads them. Dropping the library means
    specifying them."""
    assert GRAPH_SCHEMA["ptr"].required
    assert GRAPH_SCHEMA["batch"].required


def test_the_edge_index_is_the_only_offset_field():
    offset = [
        name for name, spec in GRAPH_SCHEMA.items() if spec.concatenate == "offset"
    ]
    assert offset == ["edge_index"]


def test_the_graph_count_comes_from_ptr_and_not_from_batch():
    """`batch.max()` is a data-dependent read and is wrong for a batch whose
    last graph has no nodes."""
    batch = collate([single(nodes=2), single(nodes=3)])
    assert GraphInfo.of(batch).num_graphs == 2
    assert len(batch["ptr"]) == 3


def test_an_unknown_field_is_refused_rather_than_carried_along():
    """The frozen tree coerces whatever is left over with as_tensor, casts it
    if it is floating and reshapes it if it is one-dimensional."""
    graph = collate([single()])
    graph["something_else"] = np.zeros(3)
    with pytest.raises(GraphValidationError, match="not fields of the graph schema"):
        validate_graph(graph)


def test_a_missing_required_field_names_what_is_missing():
    graph = collate([single()])
    del graph["shifts"]
    with pytest.raises(GraphValidationError, match=r"\['shifts'\]"):
        validate_graph(graph)


def test_a_shape_that_contradicts_the_counts_is_refused():
    graph = collate([single()])
    graph["shifts"] = np.zeros((99, 3))
    with pytest.raises(GraphValidationError, match="'shifts' has shape"):
        validate_graph(graph)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def test_ptr_and_batch_are_built_rather_than_inherited():
    batch = collate([single(nodes=2), single(nodes=3), single(nodes=1)])
    assert batch["ptr"].tolist() == [0, 2, 5, 6]
    assert batch["batch"].tolist() == [0, 0, 1, 1, 1, 2]


def test_the_edge_index_is_shifted_by_the_running_node_count():
    first = single(nodes=2, edges=1)
    second = single(nodes=3, edges=1)
    first["edge_index"] = np.array([[0], [1]])
    second["edge_index"] = np.array([[0], [2]])
    batch = collate([first, second])
    assert batch["edge_index"].tolist() == [[0, 2], [1, 4]]


def test_a_half_labelled_batch_is_refused():
    with pytest.raises(GraphValidationError, match="half-labelled"):
        collate([single(), {**single(), "unit_shifts": np.zeros((4, 3))}])


def test_collating_nothing_says_so():
    with pytest.raises(GraphValidationError, match="no graphs"):
        collate([])


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_the_graph_layer_imports_no_framework(framework):
    probe = (
        "import sys, mace_core.graph, mace_core.neighbors\n"
        f"assert {framework!r} not in sys.modules, "
        f"'mace_core pulled in {framework}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
