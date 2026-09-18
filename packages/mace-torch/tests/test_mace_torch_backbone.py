"""The backbone's symmetries, at both precisions.

Equivariance is the property the architecture exists for, so it is asserted
directly rather than inferred from a parity run against the frozen tree. A
rotation is applied to the *inputs* and the outputs are compared against the
same rotation applied to the outputs, one Wigner-D block per irrep.

The claims that are about arithmetic run once per dtype, through the suite's
shared fixture. The ones that are about structure carry ``fp64_only``, because
running them twice would report coverage the second run does not add.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_d_real
from mace_core.neighbors import get_neighborhood
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.backbone import MACEBackbone

ATOMIC_NUMBERS = [1, 8]
CUTOFF = 5.0
HIDDEN = "0e+1o"


def make_model(**overrides):
    """A small backbone at the dtype the fixture has made default."""
    torch.manual_seed(0)
    settings = dict(
        atomic_numbers=ATOMIC_NUMBERS,
        num_layers=2,
        num_features=4,
        lmax=2,
        hidden_irreps=HIDDEN,
        correlation=2,
        cutoff=CUTOFF,
        avg_num_neighbors=6.0,
        precision=str(torch.get_default_dtype()).removeprefix("torch."),
    )
    settings.update(overrides)
    return MACEBackbone(ReferenceBackend(), **settings)


def make_graph(positions, numbers, cell=None, pbc=(False, False, False)):
    positions = np.asarray(positions, dtype=float)
    neighborhood = get_neighborhood(positions, CUTOFF, pbc, cell)
    return {
        "positions": torch.tensor(positions, dtype=torch.get_default_dtype()),
        "atomic_numbers": torch.tensor(list(numbers)),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts, dtype=torch.get_default_dtype()),
    }


def water_dimer():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.95, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
            [2.8, 0.3, 0.1],
            [3.4, 1.0, 0.4],
            [3.1, -0.5, 0.6],
        ]
    )
    return positions, [8, 1, 1, 8, 1, 1]


def random_rotation(seed):
    generator = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def rotate_features(features, irreps, rotation):
    """The rotation applied to each irrep block of the last axis."""
    blocks, offset = [], 0
    for multiplicity, irrep in Irreps.parse(irreps).terms:
        wigner = torch.tensor(
            wigner_d_real(irrep.degree, rotation), dtype=features.dtype
        )
        for _ in range(multiplicity):
            width = irrep.dimension
            blocks.append(features[..., offset : offset + width] @ wigner)
            offset += width
    return torch.cat(blocks, dim=-1)


def test_rotation_equivariance():
    """Rotating the positions rotates every layer's features."""
    model = make_model()
    positions, numbers = water_dimer()
    rotation = random_rotation(3)

    plain = model(make_graph(positions, numbers))
    rotated = model(make_graph(positions @ rotation.T, numbers))

    for layer, (before, after) in enumerate(zip(plain, rotated, strict=True)):
        assert_close(
            after,
            rotate_features(before, HIDDEN, rotation).detach().numpy(),
            f"layer {layer} under a rotation",
        )


def test_inversion_parity():
    """Negating the positions flips the odd irreps and leaves the even ones."""
    model = make_model()
    positions, numbers = water_dimer()

    plain = model(make_graph(positions, numbers))
    inverted = model(make_graph(-positions, numbers))

    signs = []
    for multiplicity, irrep in Irreps.parse(HIDDEN).terms:
        sign = 1.0 if irrep.parity == 1 else -1.0
        signs.extend([sign] * (multiplicity * irrep.dimension))
    signs = np.array(signs)

    for layer, (before, after) in enumerate(zip(plain, inverted, strict=True)):
        assert_close(
            after, before.detach().numpy() * signs, f"layer {layer} under inversion"
        )


def test_translation_invariance():
    """A rigid shift changes nothing. The forward takes differences only."""
    model = make_model()
    positions, numbers = water_dimer()

    plain = model(make_graph(positions, numbers))
    shifted = model(make_graph(positions + np.array([17.0, -4.5, 2.25]), numbers))

    for layer, (before, after) in enumerate(zip(plain, shifted, strict=True)):
        assert_close(
            after, before.detach().numpy(), f"layer {layer} under a translation"
        )


def test_node_permutation_equivariance():
    """Relabelling the nodes permutes the rows and nothing else."""
    model = make_model()
    positions, numbers = water_dimer()
    order = np.array([3, 0, 5, 2, 1, 4])

    plain = model(make_graph(positions, numbers))
    permuted = model(make_graph(positions[order], [numbers[i] for i in order]))

    for layer, (before, after) in enumerate(zip(plain, permuted, strict=True)):
        assert_close(
            after,
            before.detach().numpy()[order],
            f"layer {layer} under a relabelling of the nodes",
        )


def test_padded_and_unpadded_agree():
    """Padding nodes onto the end leaves the real nodes' features alone.

    A padded batch is how a compiled graph keeps one shape across steps, so
    the identity has to hold, not nearly hold.
    """
    model = make_model()
    positions, numbers = water_dimer()
    graph = make_graph(positions, numbers)

    padding = 4
    padded = dict(graph)
    padded["positions"] = torch.cat(
        [
            graph["positions"],
            torch.full((padding, 3), 40.0, dtype=graph["positions"].dtype),
        ]
    )
    padded["atomic_numbers"] = torch.cat(
        [graph["atomic_numbers"], torch.ones(padding, dtype=torch.long)]
    )

    plain = model(graph)
    with_padding = model(padded)
    real = len(numbers)

    for layer, (before, after) in enumerate(zip(plain, with_padding, strict=True)):
        assert_close(
            after[:real], before.detach().numpy(), f"layer {layer} with padding added"
        )


def test_periodic_image_consistency():
    """A supercell of a periodic crystal repeats the primitive cell's features."""
    model = make_model(avg_num_neighbors=12.0)
    cell = np.eye(3) * 3.2
    positions = np.array([[0.0, 0.0, 0.0], [1.6, 1.6, 1.6]])
    numbers = [8, 8]

    primitive = model(make_graph(positions, numbers, cell, (True, True, True)))
    supercell = np.concatenate([positions, positions + np.array([3.2, 0.0, 0.0])])
    doubled = model(
        make_graph(supercell, numbers * 2, np.diag([6.4, 3.2, 3.2]), (True, True, True))
    )

    for layer, (small, big) in enumerate(zip(primitive, doubled, strict=True)):
        assert_close(
            big[:2], small.detach().numpy(), f"layer {layer} in a doubled cell"
        )


def test_descriptors_are_rotation_invariant():
    """The scalar channels do not move under a rotation. That is the point."""
    model = make_model()
    positions, numbers = water_dimer()
    rotation = random_rotation(11)

    plain = model.descriptors(make_graph(positions, numbers))
    rotated = model.descriptors(make_graph(positions @ rotation.T, numbers))

    assert_close(rotated, plain.detach().numpy(), "the invariant descriptors")


@fp64_only
def test_descriptor_aggregations():
    model = make_model()
    positions, numbers = water_dimer()
    graph = make_graph(positions, numbers)

    per_node = model.descriptors(graph)
    assert_close(
        model.descriptors(graph, aggregation="mean"),
        per_node.mean(0, keepdim=True).detach().numpy(),
        "the structure mean",
    )

    per_element = model.descriptors(graph, aggregation="per_element_mean")
    assert per_element.shape == (len(ATOMIC_NUMBERS), per_node.shape[1])
    hydrogen = [index for index, z in enumerate(numbers) if z == 1]
    assert_close(
        per_element[0],
        per_node[hydrogen].mean(0).detach().numpy(),
        "the hydrogen mean",
    )


@fp64_only
def test_forward_does_not_write_to_the_graph():
    """The backbone reads its input. The rewrite's read-only rule, asserted."""
    model = make_model()
    positions, numbers = water_dimer()
    graph = make_graph(positions, numbers)
    before = {key: value.clone() for key, value in graph.items()}

    model(graph)

    assert set(graph) == set(before), (
        f"the forward added keys to the graph: {set(graph) - set(before)}"
    )
    for key, value in before.items():
        assert torch.equal(graph[key], value), f"the forward overwrote {key!r}"


@fp64_only
def test_the_locality_hook_is_absent_by_default():
    """No hook means the plain path, with nothing to configure away."""
    assert make_model().locality is None


@fp64_only
def test_the_locality_hook_runs_once_per_layer():
    """The one seam a domain-decomposed run uses.

    It replaces threading a real-atom count through every block signature,
    which is how the frozen tree carries the same information.
    """
    seen = []

    def watch(features, graph):
        seen.append(int(features.shape[0]))
        return features

    positions, numbers = water_dimer()
    plain = make_model()(make_graph(positions, numbers))
    hooked = make_model(locality=watch)(make_graph(positions, numbers))

    assert seen == [len(numbers)] * 2, f"the hook did not run once per layer: {seen}"
    for before, after in zip(plain, hooked, strict=True):
        assert torch.equal(before, after), "an identity hook changed the result"


@fp64_only
def test_descriptors_keep_only_the_invariants_by_default():
    model = make_model()
    positions, numbers = water_dimer()
    graph = make_graph(positions, numbers)

    assert model.descriptors(graph).shape == (len(numbers), 2 * model.num_features)
    assert model.descriptors(graph, invariants_only=False).shape == (
        len(numbers),
        2 * model.num_features * Irreps.parse(HIDDEN).dimension,
    )


@fp64_only
def test_descriptors_can_be_cut_to_fewer_layers():
    model = make_model()
    positions, numbers = water_dimer()
    graph = make_graph(positions, numbers)

    assert model.descriptors(graph, num_layers=1).shape[1] == model.num_features
    assert model.descriptors(graph, num_layers=2).shape[1] == 2 * model.num_features


@fp64_only
def test_an_unknown_aggregation_says_what_the_choices_are():
    model = make_model()
    positions, numbers = water_dimer()
    with pytest.raises(ValueError, match="per_element_mean"):
        model.descriptors(make_graph(positions, numbers), aggregation="sum")


@fp64_only
def test_the_backbone_takes_no_gradients_of_its_own():
    """Derivatives are taken above the backbone, never inside it."""
    from mace_torch.nn import backbone as base
    from mace_torch.nn import interaction, product_basis

    for module in (base, interaction, product_basis):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"grad", "backward"}:
                pytest.fail(
                    f"{module.__name__} calls autograd directly, which belongs "
                    f"to the derivative engine above the backbone"
                )


@pytest.mark.slow
@fp64_only
def test_the_backbone_compiles_whole_and_takes_any_size():
    """``fullgraph=True`` with no escape hatch, and one graph for every size.

    Whole rather than merely working: a graph break is the thing the rewrite
    is meant to remove, so a run that silently falls back would report the
    same numbers and none of the point. ``dynamic=True`` then says the node
    and edge counts are not baked in, which is what lets one compilation serve
    a trajectory whose neighbour list changes every step.
    """
    model = make_model()
    positions, numbers = water_dimer()
    compiled = torch.compile(model, fullgraph=True, dynamic=True)

    graph = make_graph(positions, numbers)
    for eager, traced in zip(model(graph), compiled(graph), strict=True):
        assert torch.equal(eager, traced), "compiling changed the result"

    larger = np.concatenate([positions, positions + np.array([9.0, 0.0, 0.0])])
    graph = make_graph(larger, numbers * 2)
    for eager, traced in zip(model(graph), compiled(graph), strict=True):
        assert torch.equal(eager, traced), "a second size changed the result"
