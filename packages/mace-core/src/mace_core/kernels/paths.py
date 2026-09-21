"""Which tensor-product paths a message-passing convolution has, and in what order.

A channelwise convolution couples every node irrep with every edge irrep and
keeps whatever the selection rules allow that the layer asked for. Several of
those couplings land on the **same** output irrep, and they are kept apart
rather than summed: the linear map after the convolution mixes them, and it can
send one copy somewhere the other does not go. Summing them first would be a
smaller model wearing the same shape.

The order is part of the weight format. One radial weight is produced per path
per channel, and the linear that follows reads the paths in this order, so a
different order is a different checkpoint. The rule is: generate in
``(node term, edge term, output irrep)`` order, keep what the target declares,
then sort by irrep with a **stable** sort so that couplings landing on the same
irrep stay in generation order.
"""

from __future__ import annotations

from dataclasses import dataclass

from mace_core.clebsch_gordan.irreps import Irrep, Irreps

__all__ = ["ConvPath", "channelwise_paths", "path_irreps"]


@dataclass(frozen=True)
class ConvPath:
    """One coupling kept by a channelwise convolution.

    Attributes:
        node_term: Index of the node irrep term it reads.
        edge_term: Index of the edge irrep term it reads.
        irrep: The output irrep it writes.
        multiplicity: The node term's multiplicity, which the path carries
            through unchanged. That is what ``channelwise`` means.
    """

    node_term: int
    edge_term: int
    irrep: Irrep
    multiplicity: int


def channelwise_paths(
    irreps_node: str, irreps_edge: str, irreps_target: str
) -> tuple[ConvPath, ...]:
    """Every path the convolution keeps, in the order its weights are stored.

    Args:
        irreps_node: The sender node features.
        irreps_edge: The edge attributes, normally spherical harmonics.
        irreps_target: Which output irreps the layer wants. A coupling landing
            outside this is dropped, not truncated later.
    """
    node = Irreps.parse(irreps_node)
    edge = Irreps.parse(irreps_edge)
    wanted = {irrep for _, irrep in Irreps.parse(irreps_target).terms}

    found = []
    for node_index, (multiplicity, node_irrep) in enumerate(node.terms):
        for edge_index, (_, edge_irrep) in enumerate(edge.terms):
            for coupled in node_irrep.couple(edge_irrep):
                if coupled in wanted:
                    found.append(
                        ConvPath(node_index, edge_index, coupled, multiplicity)
                    )
    # Stable, so paths onto the same irrep keep the order they were generated
    # in. That order is what the stored weights are written against.
    return tuple(sorted(found, key=lambda path: path.irrep))


def path_irreps(paths: tuple[ConvPath, ...]) -> str:
    """The convolution's output declaration, one term per path.

    Duplicates are left as separate terms on purpose: merging them would lose
    which weight belongs to which coupling.
    """
    return "+".join(f"{path.multiplicity}x{path.irrep}" for path in paths)
