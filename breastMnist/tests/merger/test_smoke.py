"""Tests for the KG merger: rule-based (Module 6) and baselines.

Test matrix
-----------
  Score helpers:
    test_expert_overall_normalizes
    test_expert_overall_unknown_defaults_half
    test_utterance_quality_mean_of_both_dims
    test_utterance_quality_empty_defaults_half

  _triple_score:
    test_triple_score_cited
    test_triple_score_rescue_floor
    test_triple_score_uncited_returns_none
    test_triple_score_ignores_wrong_expert

  merge_kgs — floor logic:
    test_merge_kgs_keeps_cited_above_floor
    test_merge_kgs_drops_cited_below_floor
    test_merge_kgs_keeps_uncited_always
    test_merge_kgs_both_kgs_same_content

  merge_kgs — conflict resolution:
    test_merge_kgs_resolves_conflict_higher_wins
    test_merge_kgs_conflict_winner_in_both_kgs
    test_merge_kgs_conflict_both_below_floor_drops

  merge_kgs — MergeResult accounting:
    test_merge_result_counts_correct

  Consistency checks (warn-only):
    test_consistency_warns_functional_violation
    test_consistency_warns_symmetry_missing

  Legacy smoke:
    test_merge_kgs_returns_two_knowledge_graphs
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore, Triple
from breastMnist.src.debate_kg.merger.baselines import naive_union_merge, random_merge
from breastMnist.src.debate_kg.merger.rule_based import (
    _consistency_checks,
    _expert_overall,
    _triple_score,
    _utterance_quality,
    merge_kgs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(floor: float = 0.3) -> object:
    cfg = MagicMock()
    cfg.run.merger_score_floor = floor
    return cfg


_node_counter = [0]

def _node(expert_id: str, provenance: list[str]) -> DebateNode:
    _node_counter[0] += 1
    return DebateNode(
        short_id=f"c{_node_counter[0]}",
        expert_id=expert_id,
        round_idx=0,
        text="t",
        provenance=provenance,
    )


def _jscore(gnd: float, fact: float) -> JudgeScore:
    return JudgeScore(groundedness=gnd, factuality=fact)


def _escore(overall: float) -> ExpertScore:
    return ExpertScore(overall=overall)


def _triple(subj: str = "A", pred: str = "p", obj: str = "X") -> Triple:
    return Triple(subject=subj, predicate=pred, object=obj)


# ---------------------------------------------------------------------------
# _expert_overall
# ---------------------------------------------------------------------------

def test_expert_overall_normalizes() -> None:
    escores = {"expert_a": [_escore(60.0), _escore(80.0), _escore(70.0)]}
    result = _expert_overall(escores, "expert_a")
    assert result == pytest.approx(70.0 / 100.0)


def test_expert_overall_unknown_defaults_half() -> None:
    assert _expert_overall({}, "expert_a") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _utterance_quality
# ---------------------------------------------------------------------------

def test_utterance_quality_mean_of_both_dims() -> None:
    scores = {"n1": [_jscore(80.0, 60.0)]}
    assert _utterance_quality("n1", scores) == pytest.approx(0.70)


def test_utterance_quality_empty_defaults_half() -> None:
    assert _utterance_quality("missing", {}) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _triple_score
# ---------------------------------------------------------------------------

def test_triple_score_cited() -> None:
    t = _triple()
    node = _node("expert_a", [t.uuid])
    scores = {node.id: [_jscore(90.0, 90.0)]}
    expert_overalls = {"expert_a": 0.85}
    # utterance_quality = (90+90)/2/100 = 0.90; max(0.85, 0.5) = 0.85
    result = _triple_score(t.uuid, "expert_a", [node], scores, expert_overalls)
    assert result == pytest.approx(0.85 * 0.90)


def test_triple_score_rescue_floor() -> None:
    t = _triple()
    node = _node("expert_a", [t.uuid])
    scores = {node.id: [_jscore(90.0, 90.0)]}
    expert_overalls = {"expert_a": 0.25}
    # max(0.25, 0.5) = 0.5; score = 0.5 * 0.90 = 0.45
    result = _triple_score(t.uuid, "expert_a", [node], scores, expert_overalls)
    assert result == pytest.approx(0.5 * 0.90)


def test_triple_score_uncited_returns_none() -> None:
    t = _triple()
    result = _triple_score(t.uuid, "expert_a", [], {}, {"expert_a": 0.85})
    assert result is None


def test_triple_score_ignores_wrong_expert() -> None:
    t = _triple()
    node_b = _node("expert_b", [t.uuid])
    scores = {node_b.id: [_jscore(90.0, 90.0)]}
    # expert_a has no citing nodes → uncited → None
    result = _triple_score(t.uuid, "expert_a", [node_b], scores, {"expert_a": 0.85})
    assert result is None


# ---------------------------------------------------------------------------
# merge_kgs — floor logic
# ---------------------------------------------------------------------------

def _make_score_for(node: DebateNode, gnd: float, fact: float) -> dict[str, list[JudgeScore]]:
    return {node.id: [_jscore(gnd, fact)]}


def test_merge_kgs_keeps_cited_above_floor() -> None:
    t = _triple()
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t)

    node = _node("expert_a", [t.uuid])
    # e=0.85, u=0.90 → 0.765 > 0.3
    expert_sc = {"expert_a": [_escore(85.0)]}
    scores = _make_score_for(node, 90.0, 90.0)

    new_a, _ = merge_kgs(kg_a, KnowledgeGraph(), [node], scores, expert_sc, _cfg(0.3))
    assert any(tr.uuid == t.uuid for tr in new_a.all_triples())


def test_merge_kgs_drops_cited_below_floor() -> None:
    t = _triple()
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t)

    node = _node("expert_a", [t.uuid])
    # e=0.1 → max(0.1, 0.5)=0.5; u=0.10 → 0.5*0.10=0.05 < 0.3
    expert_sc = {"expert_a": [_escore(10.0)]}
    scores = _make_score_for(node, 10.0, 10.0)

    new_a, _ = merge_kgs(kg_a, KnowledgeGraph(), [node], scores, expert_sc, _cfg(0.3))
    assert not any(tr.uuid == t.uuid for tr in new_a.all_triples())


def test_merge_kgs_keeps_uncited_always() -> None:
    t = _triple()
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t)

    # No nodes, no scores — triple is uncited → always kept
    new_a, _ = merge_kgs(kg_a, KnowledgeGraph(), [], {}, {}, _cfg(0.9))
    assert any(tr.uuid == t.uuid for tr in new_a.all_triples())


def test_merge_kgs_both_kgs_same_content() -> None:
    t = _triple()
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t)

    new_a, new_b = merge_kgs(kg_a, KnowledgeGraph(), [], {}, {}, _cfg(0.3))
    uuids_a = {tr.uuid for tr in new_a.all_triples()}
    uuids_b = {tr.uuid for tr in new_b.all_triples()}
    assert uuids_a == uuids_b


# ---------------------------------------------------------------------------
# merge_kgs — conflict resolution
# ---------------------------------------------------------------------------

def test_merge_kgs_resolves_conflict_higher_wins() -> None:
    """A's triple (higher score) wins over B's conflicting triple on same (s, p)."""
    t_a = Triple(subject="S", predicate="p", object="A-val")
    t_b = Triple(subject="S", predicate="p", object="B-val")
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t_a)
    kg_b = KnowledgeGraph()
    kg_b.add_triple(t_b)

    node_a = _node("expert_a", [t_a.uuid])
    node_b = _node("expert_b", [t_b.uuid])
    expert_sc = {"expert_a": [_escore(85.0)], "expert_b": [_escore(50.0)]}
    scores = {
        node_a.id: [_jscore(90.0, 90.0)],  # u=0.90; score≈0.765
        node_b.id: [_jscore(50.0, 50.0)],  # u=0.50; score=0.25
    }

    new_a, _ = merge_kgs(kg_a, kg_b, [node_a, node_b], scores, expert_sc, _cfg(0.3))
    objects_in_merged = {tr.object for tr in new_a.all_triples()}
    assert "A-val" in objects_in_merged
    assert "B-val" not in objects_in_merged


def test_merge_kgs_conflict_winner_in_both_kgs() -> None:
    """The winning triple from conflict resolution appears in both returned KGs."""
    t_a = Triple(subject="S", predicate="p", object="A-val")
    t_b = Triple(subject="S", predicate="p", object="B-val")
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t_a)
    kg_b = KnowledgeGraph()
    kg_b.add_triple(t_b)

    node_a = _node("expert_a", [t_a.uuid])
    node_b = _node("expert_b", [t_b.uuid])
    expert_sc = {"expert_a": [_escore(85.0)], "expert_b": [_escore(50.0)]}
    scores = {
        node_a.id: [_jscore(90.0, 90.0)],
        node_b.id: [_jscore(50.0, 50.0)],
    }

    new_a, new_b = merge_kgs(kg_a, kg_b, [node_a, node_b], scores, expert_sc, _cfg(0.3))
    uuids_a = {tr.uuid for tr in new_a.all_triples()}
    uuids_b = {tr.uuid for tr in new_b.all_triples()}
    assert uuids_a == uuids_b
    assert t_a.uuid in uuids_a  # A won


def test_merge_kgs_conflict_both_below_floor_drops() -> None:
    """Both conflicting triples are dropped when the winner still falls below floor."""
    t_a = Triple(subject="S", predicate="p", object="A-val")
    t_b = Triple(subject="S", predicate="p", object="B-val")
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t_a)
    kg_b = KnowledgeGraph()
    kg_b.add_triple(t_b)

    node_a = _node("expert_a", [t_a.uuid])
    node_b = _node("expert_b", [t_b.uuid])
    # Both score ~0.05 (u=0.10, e→max=0.5 → 0.05), floor=0.3
    expert_sc = {"expert_a": [_escore(10.0)], "expert_b": [_escore(10.0)]}
    scores = {
        node_a.id: [_jscore(10.0, 10.0)],
        node_b.id: [_jscore(10.0, 10.0)],
    }

    new_a, _ = merge_kgs(kg_a, kg_b, [node_a, node_b], scores, expert_sc, _cfg(0.3))
    assert len(new_a.all_triples()) == 0


# ---------------------------------------------------------------------------
# merge_kgs — MergeResult accounting
# ---------------------------------------------------------------------------

def test_merge_result_counts_correct(caplog) -> None:
    """The INFO log from merge_kgs reports the right counts."""
    t_keep = _triple("A", "p1", "X")    # uncited → kept
    t_drop = _triple("B", "p2", "Y")    # cited, low score → dropped

    kg_a = KnowledgeGraph()
    kg_a.add_triple(t_keep)
    kg_a.add_triple(t_drop)

    node = _node("expert_a", [t_drop.uuid])
    expert_sc = {"expert_a": [_escore(10.0)]}
    scores = {node.id: [_jscore(10.0, 10.0)]}

    with caplog.at_level(logging.INFO, logger="debate_kg.merger.rule_based"):
        merge_kgs(kg_a, KnowledgeGraph(), [node], scores, expert_sc, _cfg(0.3))

    log_text = caplog.text
    assert "kept=1" in log_text
    assert "dropped=1" in log_text


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def test_consistency_warns_functional_violation(caplog) -> None:
    """Two triples with the same functional predicate and different objects → warning."""
    t1 = Triple(subject="Einstein", predicate="birthDate", object="1879-03-14")
    t2 = Triple(subject="Einstein", predicate="birthDate", object="1879-03-15")

    with caplog.at_level(logging.WARNING, logger="debate_kg.merger.rule_based"):
        _consistency_checks([t1, t2])

    assert "birthDate" in caplog.text


def test_consistency_warns_symmetry_missing(caplog) -> None:
    """(A marriedTo B) without (B marriedTo A) → warning."""
    t = Triple(subject="Alice", predicate="marriedTo", object="Bob")

    with caplog.at_level(logging.WARNING, logger="debate_kg.merger.rule_based"):
        _consistency_checks([t])

    assert "marriedTo" in caplog.text


# ---------------------------------------------------------------------------
# Legacy smoke tests (updated for new signature)
# ---------------------------------------------------------------------------

def test_merge_kgs_returns_two_knowledge_graphs() -> None:
    cfg = _cfg()
    kg_a = KnowledgeGraph()
    kg_b = KnowledgeGraph()
    new_a, new_b = merge_kgs(kg_a, kg_b, [], {}, {}, cfg)
    assert isinstance(new_a, KnowledgeGraph)
    assert isinstance(new_b, KnowledgeGraph)


# ---------------------------------------------------------------------------
# Baseline mergers
# ---------------------------------------------------------------------------

def _two_kgs() -> tuple[KnowledgeGraph, KnowledgeGraph]:
    t_a = _triple("A", "p", "X")
    t_b = _triple("B", "q", "Y")
    kg_a = KnowledgeGraph()
    kg_a.add_triple(t_a)
    kg_b = KnowledgeGraph()
    kg_b.add_triple(t_b)
    return kg_a, kg_b


def _baseline_cfg(keep_prob: float = 0.5) -> object:
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.run.random_merge_keep_prob = keep_prob
    cfg.data.seed = 42
    return cfg


def test_naive_union_merge_contains_all_triples() -> None:
    kg_a, kg_b = _two_kgs()
    all_uuids = {t.uuid for t in kg_a.all_triples()} | {t.uuid for t in kg_b.all_triples()}
    new_a, _ = naive_union_merge(kg_a, kg_b, [], {}, {}, _baseline_cfg())
    merged_uuids = {t.uuid for t in new_a.all_triples()}
    assert merged_uuids == all_uuids


def test_naive_union_merge_both_kgs_same() -> None:
    kg_a, kg_b = _two_kgs()
    new_a, new_b = naive_union_merge(kg_a, kg_b, [], {}, {}, _baseline_cfg())
    assert {t.uuid for t in new_a.all_triples()} == {t.uuid for t in new_b.all_triples()}


def test_random_merge_seeded_deterministic() -> None:
    kg_a, kg_b = _two_kgs()
    cfg = _baseline_cfg(keep_prob=0.5)
    new_1, _ = random_merge(kg_a, kg_b, [], {}, {}, cfg)
    new_2, _ = random_merge(kg_a, kg_b, [], {}, {}, cfg)
    assert {t.uuid for t in new_1.all_triples()} == {t.uuid for t in new_2.all_triples()}


def test_random_merge_keep_prob_zero_empty() -> None:
    kg_a, kg_b = _two_kgs()
    new_a, _ = random_merge(kg_a, kg_b, [], {}, {}, _baseline_cfg(keep_prob=0.0))
    assert len(new_a.all_triples()) == 0


def test_random_merge_keep_prob_one_full() -> None:
    kg_a, kg_b = _two_kgs()
    all_uuids = {t.uuid for t in kg_a.all_triples()} | {t.uuid for t in kg_b.all_triples()}
    new_a, _ = random_merge(kg_a, kg_b, [], {}, {}, _baseline_cfg(keep_prob=1.0))
    assert {t.uuid for t in new_a.all_triples()} == all_uuids
