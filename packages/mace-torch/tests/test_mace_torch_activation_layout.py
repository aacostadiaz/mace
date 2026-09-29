"""One feature layout per chain of ops, chosen by the backend, never by the model.

The claims, in order. The layout object moves values between the two layouts
and hands out views, exactly. The composite builds every op in the chosen
backend's native layout when the reference can follow, and in the canonical one
otherwise. The reference computes the same thing in either layout, op by op.
And whole models built in ``ir_mul`` give the canonical model's outputs, for
every family whose glue looks inside a term: the gated readouts, several heads,
the dipole and dielectric responses, the charge-aware fields and the magnetic
moments. The accelerated backend that wants ``ir_mul`` needs a GPU, so the
chain is exercised here with the reference declaring it as its native layout.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.capabilities import BackendCapabilities
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends import CompositeBackend, ReferenceBackend
from mace_torch.backends.conformance import run_backend_conformance
from mace_torch.backends.layout import CANONICAL, Layout, layout_of
from mace_torch.serialization import canonical_state, load_canonical_state
from mace_torch.train.model_stage import build_model

IR_MUL = Layout("ir_mul")


class NativeIrMul:
    """The reference, declaring ``ir_mul`` as the layout its kernels want.

    What an accelerated backend such as cueq declares, on a backend that runs
    anywhere, so the whole chain can be built in ``ir_mul`` on the CPU.
    """

    name = "ir-mul-reference"

    def __init__(self) -> None:
        self.reference = ReferenceBackend()

    def capabilities(self) -> BackendCapabilities:
        return replace(self.reference.capabilities(), native_layout="ir_mul")

    def __getattr__(self, name: str):
        return getattr(self.reference, name)


class CanonicalOnly:
    """A fallback that cannot follow any layout but the canonical one."""

    name = "canonical-only"

    def __init__(self) -> None:
        self.reference = ReferenceBackend()

    def capabilities(self) -> BackendCapabilities:
        return replace(
            self.reference.capabilities(), activation_layouts=frozenset({"mul_ir"})
        )

    def __getattr__(self, name: str):
        return getattr(self.reference, name)


def ir_mul_backend() -> CompositeBackend:
    return CompositeBackend(NativeIrMul(), ReferenceBackend())


# ---------------------------------------------------------------------------
# The layout object
# ---------------------------------------------------------------------------

IRREPS = "3x0e+2x1o+2x2e+1x3o"


def test_the_positions_are_where_each_canonical_entry_lands():
    features = torch.randn(4, 3 + 6 + 10 + 7)
    terms = Layout.terms(IRREPS)
    moved = IR_MUL.from_canonical(features, terms)
    assert torch.equal(moved[:, IR_MUL.positions(IRREPS)], features)
    assert torch.equal(IR_MUL.to_canonical(moved, terms), features)
    assert not torch.equal(moved, features), "the layouts differ for 2x1o"
    assert np.array_equal(CANONICAL.positions(IRREPS), np.arange(26))


def test_the_blocks_are_the_same_values_in_either_layout():
    features = torch.randn(5, 26)
    terms = Layout.terms(IRREPS)
    moved = IR_MUL.from_canonical(features, terms)
    for mine, theirs in zip(
        IR_MUL.blocks(moved, terms), CANONICAL.blocks(features, terms), strict=True
    ):
        assert torch.equal(mine, theirs)
        assert mine.untyped_storage().data_ptr() == moved.untyped_storage().data_ptr()
    assert torch.equal(IR_MUL.join(IR_MUL.blocks(moved, terms)), moved)


def test_scalars_and_single_copies_are_not_moved():
    features = torch.randn(3, 1 + 3 + 5)
    terms = Layout.terms("0e+1o+2e")
    assert IR_MUL.from_canonical(features, terms) is features


def test_channel_major_splits_each_channel_out_in_either_layout():
    channels, per_channel = 3, Layout.terms("0e+2x1o")
    features = torch.randn(5, channels * 7)
    expanded = Layout.terms("3x0e+6x1o")
    moved = IR_MUL.from_canonical(features, expanded)
    split = CANONICAL.channel_major(features, per_channel, channels)
    assert torch.equal(IR_MUL.channel_major(moved, per_channel, channels), split)
    assert torch.equal(IR_MUL.grouped(split, per_channel), moved)
    assert torch.equal(CANONICAL.grouped(split, per_channel), features)


def test_an_unknown_layout_is_refused():
    with pytest.raises(ValueError, match="not a feature layout"):
        Layout("mul_mul")  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Where the layout is decided
# ---------------------------------------------------------------------------


def test_the_chain_takes_the_chosen_backend_s_native_layout():
    composite = ir_mul_backend()
    assert layout_of(composite) == IR_MUL
    composite.make_linear(LinearDescriptor(irreps_in="4x1o", irreps_out="4x1o"))
    assert {decision.descriptor.layout for decision in composite.decisions} == {
        "ir_mul"
    }
    assert "every op in ir_mul" in composite.report()


def test_a_fallback_that_cannot_follow_keeps_the_chain_canonical():
    composite = CompositeBackend(NativeIrMul(), CanonicalOnly())
    assert layout_of(composite) == CANONICAL


def test_the_reference_alone_is_canonical():
    assert layout_of(ReferenceBackend()) == CANONICAL


def test_a_backend_declines_a_layout_it_does_not_declare():
    capabilities = BackendCapabilities(
        ops=frozenset({"linear"}), dtypes=frozenset({"float64"})
    )
    descriptor = LinearDescriptor(irreps_in="2x1o", irreps_out="2x1o", layout="ir_mul")
    assert not capabilities.supports(descriptor)


def test_the_reference_passes_its_conformance_in_ir_mul(dtype):
    precision = "float64" if str(dtype).endswith("float64") else "float32"
    results = run_backend_conformance(
        ReferenceBackend(), precision=precision, layout="ir_mul"
    )
    assert all(result.built for result in results)
    checked = {check for result in results for check in result.checks}
    assert {"weights", "values", "gradients", "equivariance"} <= checked


# ---------------------------------------------------------------------------
# Whole models
# ---------------------------------------------------------------------------


def assert_same_outputs(actual, expected) -> None:
    compared = 0
    for name, theirs in vars(expected).items():
        mine = getattr(actual, name)
        if isinstance(theirs, torch.Tensor):
            torch.testing.assert_close(mine, theirs, rtol=1e-10, atol=1e-10)
            compared += 1
        elif isinstance(theirs, dict):
            for key, value in theirs.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(
                        mine[key], value, rtol=1e-10, atol=1e-10, msg=key
                    )
                    compared += 1
    assert compared, "nothing was compared"


def built_in(model) -> set[str]:
    """The layouts the model's ops were built in."""
    return {
        module.descriptor.layout
        for module in model.modules()
        if isinstance(getattr(module, "descriptor", None), LinearDescriptor)
    }


def twin(build):
    """The same model built canonical and in ir_mul, with the same weights."""
    canonical = build(ReferenceBackend())
    in_ir_mul = build(ir_mul_backend())
    load_canonical_state(in_ir_mul, canonical_state(canonical))
    return canonical, in_ir_mul


@pytest.fixture
def ir_mul_by_name(monkeypatch):
    """A config naming the backend resolves to the ir_mul composite."""
    from mace_torch.backends import composite as module

    real = module.get_backend
    monkeypatch.setattr(
        module,
        "get_backend",
        lambda name: NativeIrMul() if name == NativeIrMul.name else real(name),
    )


@fp64_only
@pytest.mark.parametrize(
    "variant",
    [
        {},
        {"heads": ("a", "b")},
        {"readout": {"mlp_irreps": "4x0e+4x1o"}},
        {"heads": ("a", "b"), "readout": {"mlp_irreps": "4x0e+4x1o"}},
        {
            "heads": ("a", "b"),
            "readout": {"mlp_irreps": "4x0e+4x1o"},
            "observables": ("energy", "forces", "stress", "dipole"),
        },
    ],
    ids=["energy", "two-heads", "gated-readout", "two-heads-gated", "two-heads-dipole"],
)
def test_an_energy_model_in_ir_mul_is_the_canonical_model(ir_mul_by_name, variant):
    from mace_torch_backend_models import engine, graph

    canonical = engine("reference", **variant)
    in_ir_mul = engine(NativeIrMul.name, **variant)
    assert built_in(in_ir_mul) == {"ir_mul"} and built_in(canonical) == {"mul_ir"}
    load_canonical_state(in_ir_mul, canonical_state(canonical))
    structure = graph()
    if len(variant.get("heads", ())) > 1:
        structure["head"] = torch.tensor([1])
    expected = canonical(structure, compute=("forces", "stress"), training=False)
    actual = in_ir_mul(dict(structure), compute=("forces", "stress"), training=False)
    assert_same_outputs(actual, expected)


RESPONSE = {"dipole": ["dipole"], "dielectric": ["dipole", "polarizability"]}


@fp64_only
@pytest.mark.parametrize("family", ["dipole", "dielectric"])
def test_a_response_model_in_ir_mul_is_the_canonical_model(ir_mul_by_name, family):
    from test_mace_torch_calculator_families import METHANOL, charged

    numbers = [1, 6, 8]

    def build(backend_name):
        config = ResolvedConfig.model_validate(
            {
                "model": {
                    "model": family,
                    "backend": backend_name,
                    "observables": RESPONSE[family],
                    "r_max": 4.0,
                    "num_channels": 4,
                    "max_ell": 2,
                    "hidden_irreps": "0e+1o+2e",
                    "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"},
                },
                "runtime": {"seed": 0},
            }
        )
        engine, _ = build_model(
            config,
            DEFAULT_CATALOGUE,
            z_table=AtomicNumberTable(numbers),
            heads=("default",),
            e0s=ResolvedE0s({"default": dict.fromkeys(numbers, 0.0)}),
            statistics=DatasetStatistics(avg_num_neighbors=3.0),
        )
        return engine

    canonical = build("reference")
    in_ir_mul = build(NativeIrMul.name)
    load_canonical_state(in_ir_mul, canonical_state(canonical))
    from mace_torch.calculators import MACECalculator
    from test_mace_torch_calculator_families import deployed

    calculator = MACECalculator(models=deployed(family))
    structure, _ = calculator._graph(charged(METHANOL), padded=False)
    expected = canonical(dict(structure), compute=(), training=False)
    actual = in_ir_mul(dict(structure), compute=(), training=False)
    assert_same_outputs(actual, expected)


@fp64_only
def test_a_charge_aware_model_in_ir_mul_is_the_canonical_model():
    from mace_torch_engine_fixtures import molecule
    from test_mace_torch_polar_model import build_model as polar
    from test_mace_torch_polar_model import polar_graph

    canonical, in_ir_mul = twin(lambda backend: polar(backend=backend))
    assert built_in(in_ir_mul) == {"ir_mul"}
    positions, numbers = molecule()
    structure = polar_graph(positions, numbers, charge=-1.0, spin=2.0)
    assert_same_outputs(in_ir_mul(dict(structure)), canonical(dict(structure)))


@fp64_only
def test_a_magnetic_model_in_ir_mul_is_the_canonical_model():
    from test_mace_torch_magnetic_model import engine, graph, moments

    canonical, in_ir_mul = twin(lambda backend: engine(backend=backend))
    assert built_in(in_ir_mul) == {"ir_mul"}
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.9, 0.1, 0.0], [0.0, 2.0, 0.2], [1.8, 1.9, 0.1]]
    )
    structure = graph(positions, moments())
    expected = canonical(dict(structure), compute=("forces",), training=False)
    actual = in_ir_mul(dict(structure), compute=("forces",), training=False)
    assert_same_outputs(actual, expected)


@fp64_only
def test_the_descriptors_are_canonical_whatever_the_layout(ir_mul_by_name):
    from mace_torch_backend_models import engine, graph

    canonical = engine("reference")
    in_ir_mul = engine(NativeIrMul.name)
    load_canonical_state(in_ir_mul, canonical_state(canonical))

    def descriptors(model):
        backbone = next(
            module for module in model.modules() if hasattr(module, "descriptors")
        )
        return backbone.descriptors(graph(), invariants_only=False)

    torch.testing.assert_close(
        descriptors(in_ir_mul), descriptors(canonical), rtol=1e-12, atol=1e-12
    )


def test_the_head_mask_is_the_canonical_one_at_the_layout_s_positions():
    """Not observable through a forward, since the readout's weights are
    block diagonal by head; so the table itself is checked."""
    from mace_torch.models.heads import _GatedReadout, per_head_irreps

    def readout(backend):
        return _GatedReadout(
            backend, "4x0e+4x1o", "1o", "4x0e+4x1o", "float64", num_heads=2
        )

    canonical = readout(ReferenceBackend()).hidden_heads
    in_ir_mul = readout(ir_mul_backend()).hidden_heads
    positions = IR_MUL.positions(per_head_irreps("4x0e+4x1o", 2))
    assert torch.equal(in_ir_mul[positions], canonical)
    assert not torch.equal(in_ir_mul, canonical)


def test_an_input_stream_is_read_in_the_layout_the_maps_expect():
    from mace_core.observables import InputSpec
    from mace_torch.nn.node_inputs import NodeInputEmbedding

    spec = InputSpec(name="vectors", irreps="2x1o", per_atom=True, units="1")

    def embedding(backend):
        return NodeInputEmbedding(backend, [spec], "0e+1o", num_features=3)

    from mace_torch.kernels.initialization import initialize_model_weights

    canonical, in_ir_mul = embedding(ReferenceBackend()), embedding(ir_mul_backend())
    initialize_model_weights(canonical, seed=1)
    load_canonical_state(in_ir_mul, canonical_state(canonical))
    stream = torch.randn(5, 6, dtype=torch.float64)
    features = torch.zeros(5, 12, dtype=torch.float64)
    expected = canonical({"vectors": stream}, features)
    actual = in_ir_mul({"vectors": stream}, features)
    terms = Layout.terms("3x0e+3x1o")
    assert expected.abs().max() > 1e-3, "the maps map to nothing"
    torch.testing.assert_close(IR_MUL.to_canonical(actual, terms), expected)
