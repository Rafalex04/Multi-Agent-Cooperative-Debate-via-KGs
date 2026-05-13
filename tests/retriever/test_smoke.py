"""Unit tests for the GraphRAG retriever (Module 2).

All tests mock _encode to return deterministic numpy arrays — no model download,
no network, fully offline.

Test matrix
-----------
  Helpers:
    test_triple_text_format
    test_cosine_sim_identical
    test_cosine_sim_orthogonal

  retrieve():
    test_retrieve_empty_kg_returns_empty
    test_retrieve_returns_list_of_triples
    test_retrieve_single_triple_kg
    test_retrieve_respects_max_triples_cap
    test_retrieve_deduplicates_triples
    test_retrieve_ranks_relevant_triple_first
    test_retrieve_fallback_when_no_subgraph
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import Triple
from debate_kg.retriever.retriever import (
    _cosine_sim,
    _triple_text,
    retrieve,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
DIM = 4  # embedding dimension used in all fake encoders


def _cfg(k_hops: int = 2, max_triples: int = 50, max_anchors: int = 5) -> object:
    return OmegaConf.create({
        "run": {
            "retriever_k_hops": k_hops,
            "retriever_max_triples": max_triples,
        },
        "model": {"retriever_model": _MODEL_NAME},
        "data": {"max_entity_mentions": max_anchors},
    })


def _triple(subject: str, predicate: str = "p", obj: str = "o") -> Triple:
    return Triple(subject=subject, predicate=predicate, object=obj)


def _chain_kg() -> KnowledgeGraph:
    """A → B → C (three triples)."""
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="r", object="B"))
    kg.add_triple(Triple(subject="B", predicate="r", object="C"))
    kg.add_triple(Triple(subject="C", predicate="r", object="D"))
    return kg


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_triple_text_format() -> None:
    t = Triple(subject="Barack Obama", predicate="born in", object="Hawaii")
    assert _triple_text(t) == "Barack Obama | born in | Hawaii"


def test_cosine_sim_identical() -> None:
    v = np.array([1.0, 0.0, 0.0, 0.0])
    result = _cosine_sim(v, v.reshape(1, -1))
    assert abs(result[0] - 1.0) < 1e-5


def test_cosine_sim_orthogonal() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    m = np.array([[0.0, 1.0, 0.0, 0.0]])
    result = _cosine_sim(q, m)
    assert abs(result[0]) < 1e-5


# ---------------------------------------------------------------------------
# retrieve() — structural / edge-case tests (all mock _encode)
# ---------------------------------------------------------------------------

def test_retrieve_empty_kg_returns_empty() -> None:
    cfg = _cfg()
    result = retrieve(KnowledgeGraph(), "some context", cfg)
    assert result == []


def test_retrieve_returns_list_of_triples() -> None:
    kg = _chain_kg()
    cfg = _cfg()

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        return np.ones((len(texts), DIM))

    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "context", cfg)

    assert isinstance(result, list)
    assert all(isinstance(t, Triple) for t in result)


def test_retrieve_single_triple_kg() -> None:
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="X", predicate="is", object="Y"))
    cfg = _cfg()

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        return np.ones((len(texts), DIM))

    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "X is Y", cfg)

    assert len(result) == 1


def test_retrieve_respects_max_triples_cap() -> None:
    kg = KnowledgeGraph()
    for i in range(20):
        kg.add_triple(Triple(subject=f"s{i}", predicate="p", object=f"o{i}"))

    cfg = _cfg(max_triples=5)

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        return np.random.default_rng(0).random((len(texts), DIM))

    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "context", cfg)

    assert len(result) <= 5


def test_retrieve_deduplicates_triples() -> None:
    """Even if a triple is reachable from multiple anchors, it appears once."""
    # Build a KG where "B" is reachable from both "A" and "C"
    kg = KnowledgeGraph()
    t_ab = Triple(subject="A", predicate="r", object="B")
    t_cb = Triple(subject="C", predicate="r", object="B")
    t_bd = Triple(subject="B", predicate="r", object="D")
    kg.add_triple(t_ab)
    kg.add_triple(t_cb)
    kg.add_triple(t_bd)

    cfg = _cfg(max_triples=50)

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        return np.ones((len(texts), DIM))

    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "A and C go to B", cfg)

    uuids = [t.uuid for t in result]
    assert len(uuids) == len(set(uuids)), "Duplicate UUIDs found in result"


# ---------------------------------------------------------------------------
# retrieve() — semantic ranking tests
# ---------------------------------------------------------------------------

def test_retrieve_ranks_relevant_triple_first() -> None:
    """Triple whose text is most similar to the query should rank first."""
    kg = KnowledgeGraph()
    t_relevant = Triple(subject="Jimi Hendrix", predicate="plays", object="guitar")
    t_irrelevant = Triple(subject="Antarctica", predicate="is", object="cold")
    kg.add_triple(t_relevant)
    kg.add_triple(t_irrelevant)

    # Embeddings: dim=4
    # query: [1, 0, 0, 0]
    # entity "Jimi Hendrix":  [0.9, 0.1, 0, 0]   — high sim to query
    # entity "guitar":        [0.8, 0.2, 0, 0]
    # entity "Antarctica":    [0.0, 0.0, 1, 0]   — low sim to query
    # entity "cold":          [0.0, 0.0, 0, 1]
    # triple texts:
    #   "Jimi Hendrix | plays | guitar":       → high sim
    #   "Antarctica | is | cold":              → low sim

    entity_lookup = {
        "Antarctica": np.array([0.0, 0.0, 1.0, 0.0]),
        "Jimi Hendrix": np.array([0.9, 0.1, 0.0, 0.0]),
        "cold": np.array([0.0, 0.0, 0.0, 1.0]),
        "guitar": np.array([0.8, 0.2, 0.0, 0.0]),
    }
    triple_text_lookup = {
        "Jimi Hendrix | plays | guitar": np.array([0.9, 0.1, 0.0, 0.0]),
        "Antarctica | is | cold": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    query_emb = np.array([1.0, 0.0, 0.0, 0.0])

    call_count = [0]

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: query
            return query_emb.reshape(1, -1)
        elif call_count[0] == 2:
            # Second call: entity labels (sorted)
            return np.stack([entity_lookup[t] for t in texts])
        else:
            # Third call: triple texts
            return np.stack([triple_text_lookup[t] for t in texts])

    cfg = _cfg(k_hops=1, max_triples=10)
    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "guitarist", cfg)

    assert len(result) >= 1
    assert result[0].uuid == t_relevant.uuid, (
        f"Expected {t_relevant.uuid} first, got {result[0].uuid}"
    )


def test_retrieve_fallback_when_no_subgraph() -> None:
    """If k-hop BFS from all anchors yields nothing, fall back to full-KG scan."""
    # KG with a single isolated node pair that BFS with k=0 can't expand
    kg = KnowledgeGraph()
    t = Triple(subject="Isolated", predicate="edge", object="Node")
    kg.add_triple(t)

    # All entity similarities are 0 — BFS from any anchor will still return
    # triples (since subgraph_around with k>=1 from a valid entity works).
    # To force the fallback: use k_hops=0 so subgraph_around returns nothing
    # (no hops, but the entity itself has no self-loop).
    # Actually subgraph_around with k=0 returns only self-loops; for a non-self-loop
    # triple, candidates will be empty → fallback kicks in.
    cfg = _cfg(k_hops=0, max_triples=10)

    def fake_encode(texts: list[str], model_name: str) -> np.ndarray:
        return np.ones((len(texts), DIM))

    with patch("debate_kg.retriever.retriever._encode", side_effect=fake_encode):
        result = retrieve(kg, "something unrelated", cfg)

    # Fallback should have returned the one triple in the KG
    assert len(result) == 1
    assert result[0].uuid == t.uuid
