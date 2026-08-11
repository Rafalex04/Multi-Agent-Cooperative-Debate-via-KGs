"""Tests for the eval module: metrics and verdict classification.

Test matrix
-----------
  label_accuracy:
    test_label_accuracy_perfect
    test_label_accuracy_zero
    test_label_accuracy_empty

  macro_f1:
    test_macro_f1_returns_float          (legacy)
    test_macro_f1_perfect
    test_macro_f1_all_wrong
    test_macro_f1_empty
    test_macro_f1_partial

  convergence_rate / kg_drift:
    test_convergence_rate_all_converged
    test_convergence_rate_none_converged
    test_convergence_rate_empty
    test_kg_drift_identical_kgs
    test_kg_drift_completely_different

  classify_verdict (weighted vote, no LLM):
    test_classify_verdict_supports_weighted
    test_classify_verdict_refutes_weighted
    test_classify_verdict_nei_threshold
    test_classify_verdict_empty_nodes_nei
    test_classify_verdict_no_scores_fallback_majority
    test_classify_verdict_nei_label_pulls_to_zero
"""
from __future__ import annotations

import pytest

from breastMnist.src.debate_kg.eval.metrics import convergence_rate, kg_drift, label_accuracy, macro_f1
from breastMnist.src.debate_kg.eval.verdict import classify_verdict
from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateNode, JudgeScore, Triple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_node_counter = [0]

def _node(expert_id: str, round_idx: int = 0) -> DebateNode:
    _node_counter[0] += 1
    return DebateNode(
        short_id=f"c{_node_counter[0]}",
        expert_id=expert_id,
        round_idx=round_idx,
        text="t",
        provenance=[],
    )


def _jscore(g: float, f: float) -> JudgeScore:
    return JudgeScore(groundedness=g, factuality=f)


# ---------------------------------------------------------------------------
# label_accuracy
# ---------------------------------------------------------------------------

def test_label_accuracy_perfect() -> None:
    assert label_accuracy(["SUPPORTS", "REFUTES"], ["SUPPORTS", "REFUTES"]) == 1.0


def test_label_accuracy_zero() -> None:
    assert label_accuracy(["SUPPORTS"], ["REFUTES"]) == 0.0


def test_label_accuracy_empty() -> None:
    assert label_accuracy([], []) == 0.0


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------

def test_macro_f1_returns_float() -> None:
    assert isinstance(macro_f1(["SUPPORTS"], ["SUPPORTS"]), float)


def test_macro_f1_perfect() -> None:
    preds = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert macro_f1(preds, labels) == pytest.approx(1.0)


def test_macro_f1_all_wrong() -> None:
    # All predicted as SUPPORTS but actually REFUTES → F1 = 0 for all classes
    preds = ["SUPPORTS", "SUPPORTS"]
    labels = ["REFUTES", "REFUTES"]
    assert macro_f1(preds, labels) == pytest.approx(0.0)


def test_macro_f1_empty() -> None:
    assert macro_f1([], []) == pytest.approx(0.0)


def test_macro_f1_partial() -> None:
    # 2 correct out of 3; macro-F1 should be between 0 and 1
    preds =  ["SUPPORTS", "SUPPORTS", "REFUTES"]
    labels = ["SUPPORTS", "REFUTES",  "REFUTES"]
    result = macro_f1(preds, labels)
    assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# convergence_rate / kg_drift
# ---------------------------------------------------------------------------

def test_convergence_rate_all_converged() -> None:
    assert convergence_rate([1, 2, 3], max_rounds=5) == 1.0


def test_convergence_rate_none_converged() -> None:
    assert convergence_rate([5, 5], max_rounds=5) == 0.0


def test_convergence_rate_empty() -> None:
    assert convergence_rate([], max_rounds=5) == 0.0


def test_kg_drift_identical_kgs() -> None:
    kg = KnowledgeGraph()
    kg.add_triple(Triple(subject="a", predicate="b", object="c"))
    assert kg_drift(kg, kg) == 0.0


def test_kg_drift_completely_different() -> None:
    kg_a = KnowledgeGraph()
    kg_a.add_triple(Triple(subject="a", predicate="b", object="c"))
    kg_b = KnowledgeGraph()
    kg_b.add_triple(Triple(subject="x", predicate="y", object="z"))
    assert 0.0 < kg_drift(kg_a, kg_b) <= 1.0


# ---------------------------------------------------------------------------
# classify_verdict (weighted vote, no LLM)
# ---------------------------------------------------------------------------

def test_classify_verdict_empty_nodes_nei() -> None:
    assert classify_verdict([], node_scores={}, consensus_reached=False) == "NOT ENOUGH INFO"


def test_classify_verdict_supports_weighted() -> None:
    n1, n2 = _node("expert_a"), _node("expert_a")
    n1.label = "SUPPORTS"
    n2.label = "SUPPORTS"
    scores = {n1.id: _jscore(80, 80), n2.id: _jscore(70, 70)}
    assert classify_verdict([n1, n2], node_scores=scores, consensus_reached=True) == "SUPPORTS"


def test_classify_verdict_refutes_weighted() -> None:
    n = _node("expert_a")
    n.label = "REFUTES"
    scores = {n.id: _jscore(90, 90)}
    assert classify_verdict([n], node_scores=scores, consensus_reached=False) == "REFUTES"


def test_classify_verdict_nei_threshold() -> None:
    # SUPPORTS weight=80, REFUTES weight=60 → avg=(80-60)/140 ≈ 0.14 < 0.5 → NEI
    n1, n2 = _node("expert_a"), _node("expert_b")
    n1.label = "SUPPORTS"
    n2.label = "REFUTES"
    scores = {n1.id: _jscore(80, 80), n2.id: _jscore(60, 60)}
    assert classify_verdict([n1, n2], node_scores=scores, consensus_reached=False) == "NOT ENOUGH INFO"


def test_classify_verdict_no_scores_fallback_majority() -> None:
    # No judge scores → unweighted majority vote
    n1, n2, n3 = _node("expert_a"), _node("expert_a"), _node("expert_b")
    n1.label = "SUPPORTS"
    n2.label = "SUPPORTS"
    n3.label = "REFUTES"
    assert classify_verdict([n1, n2, n3], node_scores={}, consensus_reached=True) == "SUPPORTS"


def test_classify_verdict_nei_label_neutral() -> None:
    # SUPPORTS weight=80, NEI weight=80 → avg=0.5, which is on the threshold → NEI
    n1, n2 = _node("expert_a"), _node("expert_b")
    n1.label = "SUPPORTS"
    n2.label = "NOT ENOUGH INFO"
    scores = {n1.id: _jscore(80, 80), n2.id: _jscore(80, 80)}
    # avg = (80*1 + 80*0) / 160 = 0.5, threshold is 0.5 (strict >), so NEI
    assert classify_verdict([n1, n2], node_scores=scores, consensus_reached=True) == "NOT ENOUGH INFO"
