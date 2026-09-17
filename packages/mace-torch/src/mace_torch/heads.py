"""Heads built from declarations, and the derivative engine around them.

A DEMONSTRATION of what the declarative observable specification buys, written
to answer a question about how large the rewrite makes the repository. It is
not the design of the head layer: that is ARCH-3's, and it is not settled. Read
this as a worked example and expect it to be replaced.

What it shows is the whole point of the specification. There is no code here
that knows what a dipole is, or a polarizability, or a Born effective charge.
There is one readout per declared row, sized from the row's irreps string, and
one branch on the row's `per_atom` flag deciding whether the values are summed
over a graph. Declaring the entire v0.3 output surface, all four model
families, needs no line in this file.

The derivatives are computed *around* the model call rather than inside a
module's forward. That is a design commitment rather than a convenience:
`torch.autograd.grad` inside a forward is a graph break for `torch.compile`,
and the two-phase shape is what keeps the compiled path whole. It is also what
lets the sign be data: `forces` is the negative position gradient of the energy
because the specification says so, not because this file spells a minus.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from mace_core.observables import ObservableCatalogue, irreps_dimension
from mace_core.outputs import CORE_FIELD_NAMES, FIELD_BY_OBSERVABLE, MACEOutput

__all__ = ["DeclarativeHeads", "build_output", "differentiate"]


def build_output(values: Mapping[str, torch.Tensor]) -> MACEOutput:
    """Route values by name into core fields and the `extras` hatch.

    The split is the type's, not this function's: a name that resolves to one
    of the six core fields becomes that field, everything else becomes an entry
    of `extras`. Nothing here enumerates which is which.
    """
    core: dict[str, torch.Tensor] = {}
    extras: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        field = FIELD_BY_OBSERVABLE.get(name, name)
        if field in CORE_FIELD_NAMES:
            core[field] = value
        else:
            extras[name] = value
    return MACEOutput(**core, extras=extras)


def _sum_over_graphs(
    values: torch.Tensor, batch: torch.Tensor, num_graphs: int
) -> torch.Tensor:
    """Sum per-atom values into their graph. Shape `(n_atoms, d)` -> `(n_graphs, d)`."""
    totals = values.new_zeros((num_graphs, *values.shape[1:]))
    return totals.index_add(0, batch, values)


class DeclarativeHeads(torch.nn.Module):
    """One linear readout per declared observable, sized by its irreps.

    Args:
        catalogue: The declarations. Every observable in it gets a readout.
        node_feature_dim: Width of the node features the backbone produces.

    The readouts are a `ModuleDict` keyed by observable name, so a row added to
    the declarations file adds a readout and a key in the output with no edit
    here. A linear map is of course not an equivariant readout; a real head
    reads the declared irreps to build one. What the demonstration needs is
    that the *shape* comes from the declaration, and it does.
    """

    def __init__(self, catalogue: ObservableCatalogue, node_feature_dim: int) -> None:
        super().__init__()
        self.catalogue = catalogue
        self.readouts = torch.nn.ModuleDict(
            {
                spec.name: torch.nn.Linear(
                    node_feature_dim,
                    irreps_dimension(spec.irreps, observable=spec.name),
                )
                for spec in catalogue.observables
            }
        )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> MACEOutput:
        """Read every declared observable out of the node features.

        Args:
            node_features: `(n_atoms, node_feature_dim)`.
            batch: `(n_atoms,)`, the graph each atom belongs to.
            num_graphs: How many graphs the batch holds.

        Returns:
            A `MACEOutput` carrying one entry per declared observable. A
            per-atom row keeps its atom axis; a per-graph row is summed into
            its graph, which is the only place this file consults the flag.
        """
        values: dict[str, torch.Tensor] = {}
        for spec in self.catalogue.observables:
            per_atom = self.readouts[spec.name](node_features)
            values[spec.name] = (
                per_atom
                if spec.per_atom
                else _sum_over_graphs(per_atom, batch, num_graphs)
            )
        return build_output(values)


def differentiate(
    catalogue: ObservableCatalogue,
    output: MACEOutput,
    inputs: Mapping[str, torch.Tensor],
    *,
    create_graph: bool = False,
) -> MACEOutput:
    """Fill in every derivative the declarations asked for.

    Args:
        catalogue: The declarations. Only the derivatives a row actually
            requested are computed; naming works for any pair, requesting is
            what says "compute it".
        output: What the model produced. Mutated in place and returned, because
            the two-phase shape means these fields are filled after the model
            call rather than by it.
        inputs: The declared inputs, by name, each requiring grad.
        create_graph: Keep the graph, so the result is itself differentiable.
            Needed to train on forces.

    Raises:
        KeyError: If a declaration asks to differentiate a quantity the model
            did not produce, or against an input that was not supplied.
        ValueError: If the quantity turns out not to depend on the input at
            all. Every one of these is a declaration that is wrong about the
            model, and none of them is skipped: an output that quietly fails to
            appear is the failure this whole specification exists to remove.
    """
    for derivative in catalogue.requested_derivatives():
        quantity = output.get(derivative.of)
        if quantity is None:
            raise KeyError(
                f"{derivative.name!r} is the derivative of {derivative.of!r}, "
                f"which this model did not produce. The names it produced were "
                f"{sorted(output.names())}."
            )
        if derivative.wrt not in inputs:
            raise KeyError(
                f"{derivative.name!r} is the derivative of {derivative.of!r} "
                f"with respect to {derivative.wrt!r}, which was not supplied. "
                f"The inputs given were {sorted(inputs)}."
            )
        gradient = torch.autograd.grad(
            outputs=[quantity.sum()],
            inputs=[inputs[derivative.wrt]],
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            raise ValueError(
                f"{derivative.of!r} does not depend on {derivative.wrt!r}, so "
                f"{derivative.name!r} is identically zero and the declaration "
                f"asking for it is wrong about this model."
            )
        value = derivative.sign * gradient
        field = FIELD_BY_OBSERVABLE.get(derivative.name, derivative.name)
        if field in CORE_FIELD_NAMES:
            setattr(output, field, value)
        else:
            output.extras[derivative.name] = value
    return output
