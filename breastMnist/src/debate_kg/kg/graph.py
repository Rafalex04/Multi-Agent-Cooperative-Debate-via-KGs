"""NetworkX-backed knowledge graph with stable triple UUIDs.

This is the canonical KG class. Do not re-implement it elsewhere.
"""
from __future__ import annotations

import logging

import networkx as nx

from breastMnist.src.debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """MultiDiGraph wrapper where every edge is one Triple with a stable UUID."""

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._triples: dict[str, Triple] = {}

    def add_triple(self, triple: Triple) -> None:
        """Add a triple; overwrites if the UUID already exists."""
        self._triples[triple.uuid] = triple
        self._graph.add_edge(
            triple.subject,
            triple.object,
            key=triple.uuid,
            predicate=triple.predicate,
        )

    def get_triple(self, triple_uuid: str) -> Triple | None:
        """Return the triple for the given UUID, or None."""
        return self._triples.get(triple_uuid)

    def all_triples(self) -> list[Triple]:
        """Return all triples in insertion order."""
        return list(self._triples.values())

    def subgraph_around(self, entity: str, k_hops: int) -> KnowledgeGraph:
        """Return a new KG containing all nodes within k hops of entity.

        Uses undirected BFS (follows both in- and out-edges). Returns the induced
        subgraph: only triples where BOTH endpoints are in the reachable set.

        Args:
            entity: Node label to expand from (must match a triple subject/object string).
            k_hops: Number of hops to expand.

        Returns:
            New KnowledgeGraph with the induced k-hop subgraph, or an empty KG if
            entity is not present in the graph.
        """
        if entity not in self._graph:
            logger.debug("subgraph_around: entity %r not in graph", entity)
            return KnowledgeGraph()

        reachable: set[str] = {entity}
        frontier: set[str] = {entity}

        for _ in range(k_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, nbr in self._graph.out_edges(node):
                    if nbr not in reachable:
                        next_frontier.add(nbr)
                for pred, _ in self._graph.in_edges(node):
                    if pred not in reachable:
                        next_frontier.add(pred)
            if not next_frontier:
                break
            reachable |= next_frontier
            frontier = next_frontier

        sub_kg = KnowledgeGraph()
        for triple in self._triples.values():
            if triple.subject in reachable and triple.object in reachable:
                sub_kg.add_triple(triple)

        logger.debug(
            "subgraph_around(%r, k=%d): %d → %d triples",
            entity, k_hops, len(self._triples), len(sub_kg),
        )
        return sub_kg

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict keyed by UUID."""
        return {uid: t.model_dump() for uid, t in self._triples.items()}

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        """Deserialize from a dict produced by to_dict()."""
        kg = cls()
        for t_data in data.values():
            kg.add_triple(Triple(**t_data))
        return kg

    def __len__(self) -> int:
        return len(self._triples)

    def __repr__(self) -> str:
        return f"KnowledgeGraph(triples={len(self._triples)})"
