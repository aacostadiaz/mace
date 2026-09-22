"""The backbone: node features in, node features out.

The first of the two layers the architecture is split into, and it is a block
rather than a model. A model is the thing that carries observables, is built
from one config object and returns the typed output; this returns the learned
descriptor those are computed from, so it lives beside the other blocks and
not in ``models/``. The model that holds one arrives with the output layer.

It is a learned descriptor and nothing else: **no readouts, no heads, no scale
shift, and no gradient calls**. Forces and stress are taken by a derivative
engine *around* this, never inside it, which is what keeps a compiled graph
whole.

Four things this does not do, each replacing something the frozen tree does:

* It does not resolve a backend in ``forward``. Every op is built once from a
  descriptor and held.
* It does not branch on whether a fusion attribute happens to be set.
* It does not carry LAMMPS partitioning through its blocks. The locality seam
  is one explicit hook, absent by default, rather than an attribute smuggled
  into the block signatures behind a scripting probe.
* It does not write into the graph it is given. The dict it receives is read.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import (
    LinearDescriptor,
    RadialBasisDescriptor,
    RadialKind,
    SphericalHarmonicsDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.nn.interaction import InteractionBlock
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

__all__ = ["MACEBackbone"]


class MACEBackbone(nn.Module):
    """The equivariant message-passing backbone.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending. A node's element index is
            its position here.
        num_layers: How many interaction and product pairs.
        num_features: The channel width.
        lmax: The highest degree of the edge attributes.
        hidden_irreps: One channel's node-feature irreps.
        num_radial: Width of the radial embedding.
        cutoff: In Angstrom.
        correlation: The body order of the product basis.
        avg_num_neighbors: The density normalization.
        radial_kind: Which radial basis.
        precision: The dtype name every op is built at.
        locality: An optional hook taking the node features and the graph and
            returning the features to carry into the next layer. This is where
            a domain-decomposed run slices off its ghost nodes. Absent by
            default, and when absent the forward is the hook-free path exactly.
    """

    def __init__(
        self,
        backend,
        atomic_numbers: Sequence[int],
        num_layers: int = 2,
        num_features: int = 8,
        lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        cutoff: float = 5.0,
        correlation: int = 3,
        avg_num_neighbors: float = 1.0,
        radial_kind: RadialKind = "bessel",
        precision: Precision = "float64",
        locality: Callable[[Tensor, Mapping[str, Any]], Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.atomic_numbers = list(atomic_numbers)
        self.num_features = num_features
        self.lmax = lmax
        self.cutoff = cutoff
        self.locality = locality
        self.hidden_irreps = hidden_irreps

        edge_irreps = "+".join(
            f"{degree}{'e' if degree % 2 == 0 else 'o'}" for degree in range(lmax + 1)
        )
        self.edge_irreps = edge_irreps
        self.edge_attributes = backend.make_spherical_harmonics(
            SphericalHarmonicsDescriptor(lmax=lmax, precision=precision)
        )
        self.radial = backend.make_radial_basis(
            RadialBasisDescriptor(
                kind=radial_kind,
                num_basis=num_radial,
                cutoff=cutoff,
                precision=precision,
            )
        )
        self.node_embedding = backend.make_linear(
            LinearDescriptor(
                irreps_in=f"{len(self.atomic_numbers)}x0e",
                irreps_out=hidden_irreps,
                precision=precision,
            )
        )

        interactions, products, radial_maps = [], [], []
        for layer in range(num_layers):
            interaction = InteractionBlock(
                backend,
                irreps_node=hidden_irreps,
                irreps_edge=edge_irreps,
                irreps_out=hidden_irreps,
                num_radial=num_radial,
                num_features=num_features,
                num_elements=len(self.atomic_numbers),
                avg_num_neighbors=avg_num_neighbors,
                residual=layer > 0,
                precision=precision,
            )
            interactions.append(interaction)
            radial_maps.append(
                nn.Linear(
                    num_radial,
                    interaction.num_paths * num_features,
                    dtype=torch.float64 if precision == "float64" else torch.float32,
                )
            )
            products.append(
                EquivariantProductBasisBlock(
                    backend,
                    irreps_in=hidden_irreps,
                    irreps_out=hidden_irreps,
                    correlation=correlation,
                    num_elements=len(self.atomic_numbers),
                    num_features=num_features,
                    residual=layer > 0,
                    precision=precision,
                )
            )
        self.interactions = nn.ModuleList(interactions)
        self.products = nn.ModuleList(products)
        self.radial_maps = nn.ModuleList(radial_maps)
        self.width = Irreps.parse(hidden_irreps).dimension

    def element_index(self, atomic_numbers: Tensor) -> Tensor:
        """Each node's position in the element table.

        Built from the table rather than read off a one-hot with an argmax,
        which is what the frozen tree does inside every block.
        """
        table = torch.as_tensor(
            self.atomic_numbers,
            dtype=atomic_numbers.dtype,
            device=atomic_numbers.device,
        )
        return (atomic_numbers[:, None] == table[None, :]).float().argmax(dim=-1)

    def forward(self, graph: Mapping[str, Any]) -> list[Tensor]:
        """Every layer's node features, in order.

        Args:
            graph: The flat dict. Read and never written to.

        Returns:
            One ``[n_nodes, num_features, width]`` tensor per layer. The list
            rather than only the last, because the readouts above this read
            every layer and the descriptor API slices them.
        """
        positions = graph["positions"]
        edge_index = graph["edge_index"]
        sender, receiver = edge_index[0], edge_index[1]
        num_nodes = int(positions.shape[0])

        vectors = positions[receiver] - positions[sender] + graph["shifts"]
        lengths = vectors.norm(dim=-1, keepdim=True)
        edge_attributes = self.edge_attributes(vectors)
        radial = self.radial(lengths)

        element = self.element_index(graph["atomic_numbers"])
        one_hot = torch.zeros(
            num_nodes,
            len(self.atomic_numbers),
            dtype=positions.dtype,
            device=positions.device,
        )
        one_hot[torch.arange(num_nodes, device=positions.device), element] = 1.0

        features = self.node_embedding(one_hot)
        features = features.unsqueeze(1).expand(-1, self.num_features, -1).contiguous()

        outputs = []
        for interaction, product, radial_map in zip(
            self.interactions, self.products, self.radial_maps, strict=True
        ):
            weights = radial_map(radial).reshape(
                radial.shape[0], interaction.num_paths, self.num_features
            )
            messages = interaction(
                features,
                edge_attributes,
                weights,
                one_hot,
                sender,
                receiver,
                num_nodes,
            )
            features = product(messages, element)
            if self.locality is not None:
                features = self.locality(features, graph)
            outputs.append(features)
        return outputs

    def descriptors(
        self,
        graph: Mapping[str, Any],
        num_layers: int | None = None,
        invariants_only: bool = True,
        aggregation: str | None = None,
    ) -> Tensor:
        """The backbone's raison d'etre, as a first-class call.

        Args:
            graph: The flat dict.
            num_layers: How many layers to keep, from the first. ``None`` keeps
                all of them.
            invariants_only: Keep only the scalar channels. The last layer of a
                legacy model is invariant-only already, which is why its width
                differs from the others.
            aggregation: ``None`` for per-node, ``"mean"`` for the structure
                mean, ``"per_element_mean"`` for one row per element.

        Returns:
            ``[n_nodes, features]``, or the aggregated form.
        """
        layers = self.forward(graph)
        if num_layers is not None:
            layers = layers[:num_layers]
        scalars = Irreps.parse(self.hidden_irreps).terms[0][1].dimension
        pieces = [
            layer[..., :scalars] if invariants_only else layer for layer in layers
        ]
        stacked = torch.cat([piece.reshape(piece.shape[0], -1) for piece in pieces], -1)
        if aggregation is None:
            return stacked
        if aggregation == "mean":
            return stacked.mean(dim=0, keepdim=True)
        if aggregation == "per_element_mean":
            element = self.element_index(graph["atomic_numbers"])
            rows = []
            for index in range(len(self.atomic_numbers)):
                mask = element == index
                rows.append(
                    stacked[mask].mean(dim=0)
                    if bool(mask.any())
                    else torch.zeros_like(stacked[0])
                )
            return torch.stack(rows)
        raise ValueError(
            f"{aggregation!r} is not an aggregation this computes. The choices "
            f"are None, 'mean' and 'per_element_mean'."
        )
