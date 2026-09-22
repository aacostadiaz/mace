"""The architecture, and what the model is asked to read out.

Legacy decides what a model computes from the *class name* it was given: a
ladder in `mace/cli/run_train.py:588-625` branches on `AtomicDipolesMACE`,
`AtomicDielectricMACE`, `EnergyDipolesMACE` and `PolarMACE` and sets six
`compute_*` booleans per branch. So the model name and the output set are one
choice wearing two hats, and adding a property means editing the ladder.

Here they are two fields. ``model`` is a registry name and nothing else;
``observables`` is the list of declared properties, by their names in the
observable catalogue, and it is what later creates the heads. Naming a model
never rewrites the observable list.

**The Clebsch-Gordan basis is model state.** On the frozen tree the reduced
basis is reachable only when `cuequivariance` is installed, so the same
hyperparameters train a differently parametrized network on a CPU host. It is
a recorded field here, serialized with the checkpoint, and a backend that
cannot serve the recorded value fails the build rather than quietly switching.

Two things named here are owned elsewhere and say so: ``backend`` is the single
backend-name field the three legacy acceleration flags merge into, and
``scaling`` is the method the output layer's scale-shift spec reads. This
section records the choice; the semantics are those tickets'.
"""

from __future__ import annotations

from typing import Literal

from mace_core.config.section import FrozenSection

__all__ = ["ClebschGordanBasis", "ModelConfig", "ReadoutConfig", "ScalingMethod"]

#: Which Clebsch-Gordan basis the weights are written against. ``"reduced"`` is
#: the only value for new training; ``"full"`` exists to read an artifact
#: trained before the choice was recorded.
ClebschGordanBasis = Literal["reduced", "full"]

#: How a head's output is scaled. The names are the output layer's; a value
#: here is a request it has to be able to serve.
ScalingMethod = Literal["none", "std", "rms_forces", "rms_dipoles"]


class ReadoutConfig(FrozenSection):
    """The shape of the readouts, which is a head property and not a top knob.

    ``MLP_irreps`` is the hidden width of the last readout's own MLP, so it
    belongs to the readout rather than beside the interaction widths where
    legacy keeps it.

    Args:
        mlp_irreps: Hidden irreps of the final non-linear readout.
        gate: The gate activation the non-linear readout uses.
        last_only: Read out only from the last interaction layer, instead of
            summing a readout per layer.
        from_embedding: Add a readout on the node embedding, before any
            interaction.
    """

    mlp_irreps: str = "16x0e"
    gate: str = "silu"
    last_only: bool = False
    from_embedding: bool = False


class ModelConfig(FrozenSection):
    """What to build, and what to make it produce.

    Args:
        model: The registered model name. Unknown values fail here, through
            ordinary config validation, rather than in a deprecation raise
            inside model construction.
        observables: The declared properties, by observable-catalogue name.
            This is the list that creates heads and loss terms; nothing infers
            it from ``model``.
        r_max: Cutoff radius, in Angstrom.
        num_interactions: Message-passing layers.
        num_channels: Channel width, the shorthand half of ``hidden_irreps``.
        max_L: Highest degree kept in the node features, the other half.
        hidden_irreps: The node-feature irreps written out. Given, it is
            authoritative and the two shorthands are ignored.
        edge_irreps: Irreps of the edge features, when they differ.
        use_edge_irreps_first: Whether the first layer uses them too.
        max_ell: Highest degree of the spherical harmonics on an edge.
        correlation: Body order of the symmetric contraction.
        interaction: The interaction block of the later layers.
        interaction_first: The interaction block of the first layer, which
            has no incoming features to skip from.
        use_agnostic_product: Whether the product basis is element-agnostic.
        radial_type: The radial basis.
        num_radial_basis: How many radial functions.
        num_cutoff_basis: Polynomial order of the cutoff envelope. It is also
            the repulsion's envelope order; defaulting it where a trained
            model set something else moves the energy.
        distance_transform: A transform applied to the distance before the
            radial basis.
        apply_cutoff: Whether the envelope multiplies the radial basis.
        radial_mlp: Hidden widths of the radial MLP.
        pair_repulsion: Add the short-range ZBL pair term.
        clebsch_gordan_basis: Model state, not a host property.
        scaling: How the output layer scales its head.
        readout: The readout shape.
        backend: The kernel backend by name. One field, where legacy has
            three flags whose third only chose between building in a layout
            and converting after, and canonical weights remove the conversion.
    """

    model: str = "ScaleShiftMACE"
    observables: tuple[str, ...] = ("energy", "forces")
    r_max: float = 5.0
    num_interactions: int = 2
    num_channels: int = 128
    max_L: int = 1
    hidden_irreps: str | None = None
    edge_irreps: str | None = None
    use_edge_irreps_first: bool = False
    max_ell: int = 3
    correlation: int = 3
    interaction: str = "RealAgnosticResidualInteractionBlock"
    interaction_first: str = "RealAgnosticResidualInteractionBlock"
    use_agnostic_product: bool = False
    radial_type: str = "bessel"
    num_radial_basis: int = 8
    num_cutoff_basis: int = 5
    distance_transform: str = "None"
    apply_cutoff: bool = True
    radial_mlp: tuple[int, ...] = (64, 64, 64)
    pair_repulsion: bool = False
    clebsch_gordan_basis: ClebschGordanBasis = "reduced"
    scaling: ScalingMethod = "rms_forces"
    readout: ReadoutConfig = ReadoutConfig()
    backend: str = "reference"
