"""Smoke and unit tests for the kg module.

Non-network tests cover:
  - KnowledgeGraph CRUD and serialisation
  - subgraph_around k-hop BFS
  - _extract_mentions regex + stopword filter
  - _split_triples random split + overlap + determinism
  - build_kg_pair with mocked Wikidata (no network)

Network tests (skipped with -m 'not network'):
  - entity_search against the live Wikidata API
  - fetch_entity_triples against the live SPARQL endpoint
  - build_kg_pair full integration on a real FEVER-style claim
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from debate_kg.kg.construct import (
    _extract_mentions,
    _split_triples,
    build_kg_pair,
)
from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import Triple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(overlap: float = 0.0, seed: int = 42, max_triples_per_kg: int = 500) -> object:
    return OmegaConf.create({
        "data": {
            "kg_split_overlap": overlap,
            "seed": seed,
            "max_triples_per_entity": 50,
            "max_triples_per_kg": max_triples_per_kg,
            "max_entity_mentions": 5,
            "wikidata_cache_dir": "/tmp/debate_kg_test_cache",
        },
    })


def _triples(n: int) -> list[Triple]:
    return [Triple(subject=f"s{i}", predicate="p", object=f"o{i}") for i in range(n)]


# ---------------------------------------------------------------------------
# KnowledgeGraph — CRUD
# ---------------------------------------------------------------------------

def test_triple_uuid_is_stable() -> None:
    t = Triple(subject="A", predicate="rel", object="B")
    assert isinstance(t.uuid, str) and len(t.uuid) > 0
    t2 = Triple(uuid=t.uuid, subject="A", predicate="rel", object="B")
    assert t2.uuid == t.uuid


def test_knowledge_graph_add_and_get() -> None:
    kg = KnowledgeGraph()
    t = Triple(subject="foo", predicate="is", object="bar")
    kg.add_triple(t)
    retrieved = kg.get_triple(t.uuid)
    assert retrieved is not None
    assert retrieved.subject == "foo"
    assert len(kg.all_triples()) == 1


def test_knowledge_graph_missing_uuid_returns_none() -> None:
    kg = KnowledgeGraph()
    assert kg.get_triple("nonexistent") is None


def test_knowledge_graph_serialization_round_trip() -> None:
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="a", predicate="b", object="c"))
    kg.add_triple(Triple(subject="x", predicate="y", object="z"))
    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)
    assert len(kg2.all_triples()) == 2


# ---------------------------------------------------------------------------
# subgraph_around — k-hop BFS
# ---------------------------------------------------------------------------

def _chain_kg() -> KnowledgeGraph:
    """Returns: A → B → C, and X → Y (disconnected)."""
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="A", predicate="r", object="B"))
    kg.add_triple(Triple(subject="B", predicate="r", object="C"))
    kg.add_triple(Triple(subject="X", predicate="r", object="Y"))
    return kg


def test_subgraph_around_k1_includes_direct_edge() -> None:
    sub = _chain_kg().subgraph_around("A", k_hops=1)
    pairs = {(t.subject, t.object) for t in sub.all_triples()}
    assert ("A", "B") in pairs


def test_subgraph_around_k1_excludes_two_hop_node() -> None:
    sub = _chain_kg().subgraph_around("A", k_hops=1)
    pairs = {(t.subject, t.object) for t in sub.all_triples()}
    # C is 2 hops from A; (B, C) requires both B and C reachable at k=1
    assert ("B", "C") not in pairs


def test_subgraph_around_k2_includes_two_hop_edge() -> None:
    sub = _chain_kg().subgraph_around("A", k_hops=2)
    pairs = {(t.subject, t.object) for t in sub.all_triples()}
    assert ("A", "B") in pairs
    assert ("B", "C") in pairs


def test_subgraph_around_excludes_disconnected() -> None:
    sub = _chain_kg().subgraph_around("A", k_hops=5)
    pairs = {(t.subject, t.object) for t in sub.all_triples()}
    assert ("X", "Y") not in pairs


def test_subgraph_around_missing_entity_returns_empty() -> None:
    sub = _chain_kg().subgraph_around("NONEXISTENT", k_hops=2)
    assert len(sub.all_triples()) == 0


def test_subgraph_around_follows_in_edges() -> None:
    """Starting from B should reach A (via in-edge) with k=1."""
    kg = _chain_kg()
    sub = kg.subgraph_around("B", k_hops=1)
    pairs = {(t.subject, t.object) for t in sub.all_triples()}
    # Both (A,B) and (B,C) should be reachable from B at k=1
    assert ("A", "B") in pairs
    assert ("B", "C") in pairs


# ---------------------------------------------------------------------------
# _extract_mentions
# ---------------------------------------------------------------------------

def test_extract_mentions_typical_claim() -> None:
    mentions = _extract_mentions("Jimi Hendrix was born in Seattle.")
    assert "Jimi Hendrix" in mentions
    assert "Seattle" in mentions


def test_extract_mentions_filters_stopwords() -> None:
    mentions = _extract_mentions("The sky is blue.")
    assert "The" not in mentions


def test_extract_mentions_deduplicates() -> None:
    mentions = _extract_mentions("Barack Obama spoke to Barack Obama.")
    assert mentions.count("Barack Obama") == 1


def test_extract_mentions_empty_string() -> None:
    assert _extract_mentions("") == []


def test_extract_mentions_no_entities() -> None:
    # All lowercase — nothing title-cased
    mentions = _extract_mentions("the quick brown fox jumps.")
    assert mentions == []


def test_extract_mentions_multi_word_not_filtered() -> None:
    # Multi-word phrases skip the stopword filter even if first word is a stopword
    mentions = _extract_mentions("The Beatles played Liverpool.")
    # "Liverpool" (single word, not stopword) and "Beatles" should appear
    assert "Liverpool" in mentions


# ---------------------------------------------------------------------------
# _split_triples
# ---------------------------------------------------------------------------

def test_split_triples_even_count() -> None:
    ts = _triples(4)
    a, b = _split_triples(ts, 0.0, np.random.default_rng(42))
    assert len(a) + len(b) == 4
    assert len(a) == 2 and len(b) == 2


def test_split_triples_odd_count_a_gets_extra() -> None:
    ts = _triples(5)
    a, b = _split_triples(ts, 0.0, np.random.default_rng(42))
    assert len(a) + len(b) == 5
    assert len(a) == 3 and len(b) == 2


def test_split_triples_no_overlap_is_disjoint() -> None:
    ts = _triples(8)
    a, b = _split_triples(ts, 0.0, np.random.default_rng(42))
    assert len({t.uuid for t in a} & {t.uuid for t in b}) == 0


def test_split_triples_with_overlap_grows_both_sides() -> None:
    ts = _triples(10)
    a, b = _split_triples(ts, 0.5, np.random.default_rng(42))
    # Each base half is 5; overlap of 0.5 * 5 = 2 triples added to the other side
    assert len(a) > 5
    assert len(b) > 5


def test_split_triples_deterministic() -> None:
    ts = _triples(10)
    a1, b1 = _split_triples(ts, 0.0, np.random.default_rng(99))
    a2, b2 = _split_triples(ts, 0.0, np.random.default_rng(99))
    assert [t.uuid for t in a1] == [t.uuid for t in a2]
    assert [t.uuid for t in b1] == [t.uuid for t in b2]


def test_split_triples_single_triple() -> None:
    ts = _triples(1)
    a, b = _split_triples(ts, 0.0, np.random.default_rng(42))
    assert len(a) == 1 and len(b) == 0


def test_split_triples_empty() -> None:
    a, b = _split_triples([], 0.0, np.random.default_rng(42))
    assert a == [] and b == []


# ---------------------------------------------------------------------------
# build_kg_pair — no-network unit test (mocked Wikidata)
# ---------------------------------------------------------------------------

def test_build_kg_pair_returns_two_knowledge_graphs() -> None:
    cfg = _cfg()
    with patch("debate_kg.kg.wikidata.entity_search", return_value=[]):
        kg_a, kg_b = build_kg_pair("Some Claim Text Here", cfg)
    assert isinstance(kg_a, KnowledgeGraph)
    assert isinstance(kg_b, KnowledgeGraph)


def test_build_kg_pair_no_qids_returns_empty() -> None:
    cfg = _cfg()
    with patch("debate_kg.kg.wikidata.entity_search", return_value=[]):
        kg_a, kg_b = build_kg_pair("Claim with Named Entity", cfg)
    assert len(kg_a) == 0
    assert len(kg_b) == 0


def test_build_kg_pair_with_triples_splits_them() -> None:
    """Mock Wikidata to return 10 triples and verify they end up in the KGs."""
    cfg = _cfg(overlap=0.0)
    fake_triples = _triples(10)
    with (
        patch("debate_kg.kg.wikidata.entity_search", return_value=["Q999"]),
        patch("debate_kg.kg.wikidata.fetch_entity_triples", return_value=fake_triples),
    ):
        kg_a, kg_b = build_kg_pair("Jimi Hendrix played guitar.", cfg)
    assert len(kg_a) + len(kg_b) == 10
    # Disjoint with no overlap
    uuids_a = {t.uuid for t in kg_a.all_triples()}
    uuids_b = {t.uuid for t in kg_b.all_triples()}
    assert len(uuids_a & uuids_b) == 0


def test_build_kg_pair_applies_global_cap() -> None:
    cfg = _cfg(max_triples_per_kg=5)
    # Return 20 triples from Wikidata but cap should truncate to 5
    fake_triples = _triples(20)
    with (
        patch("debate_kg.kg.wikidata.entity_search", return_value=["Q1"]),
        patch("debate_kg.kg.wikidata.fetch_entity_triples", return_value=fake_triples),
    ):
        kg_a, kg_b = build_kg_pair("Some Entity Claim", cfg)
    assert len(kg_a) + len(kg_b) == 5


# ---------------------------------------------------------------------------
# Wikidata integration tests — require live network
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_entity_search_returns_qids() -> None:
    from debate_kg.kg import wikidata as wd
    qids = wd.entity_search("Douglas Adams", limit=3)
    assert len(qids) > 0
    assert all(q.startswith("Q") for q in qids)


@pytest.mark.network
def test_entity_search_top_result_for_famous_entity() -> None:
    from debate_kg.kg import wikidata as wd
    qids = wd.entity_search("Douglas Adams", limit=1)
    # Q42 is the well-known Wikidata item for Douglas Adams
    assert qids[0] == "Q42"


@pytest.mark.network
def test_fetch_entity_triples_returns_labelled_triples() -> None:
    from debate_kg.kg import wikidata as wd
    triples = wd.fetch_entity_triples("Q42", limit=10)
    assert len(triples) > 0
    for t in triples:
        assert isinstance(t, Triple)
        assert t.subject and t.predicate and t.object


@pytest.mark.network
def test_fetch_entity_triples_caches_result(tmp_path: pytest.TempPathFactory) -> None:
    from debate_kg.kg import wikidata as wd
    wd.configure_cache(tmp_path)
    triples1 = wd.fetch_entity_triples("Q42", limit=5)
    # Second call must be a cache hit (no network)
    triples2 = wd.fetch_entity_triples("Q42", limit=5)
    assert [t.uuid for t in triples1] == [t.uuid for t in triples2]


@pytest.mark.network
def test_build_kg_pair_real_claim() -> None:
    cfg = OmegaConf.create({
        "data": {
            "kg_split_overlap": 0.0,
            "seed": 42,
            "max_triples_per_entity": 30,
            "max_triples_per_kg": 100,
            "max_entity_mentions": 3,
            "wikidata_cache_dir": "/tmp/debate_kg_integration_cache",
        },
    })
    kg_a, kg_b = build_kg_pair("Jimi Hendrix was a guitarist.", cfg)
    total = len(kg_a) + len(kg_b)
    assert total > 0, "Expected at least some triples from Wikidata for Jimi Hendrix"
    # UUIDs should be stable within the same run
    uuids_a = {t.uuid for t in kg_a.all_triples()}
    uuids_b = {t.uuid for t in kg_b.all_triples()}
    assert len(uuids_a & uuids_b) == 0, "Disjoint split violated"
