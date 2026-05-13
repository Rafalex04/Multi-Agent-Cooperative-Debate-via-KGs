"""Evaluation metrics reported in SPEC §Benchmark and §Four reported metrics.

All functions are pure (no I/O, no LLM calls). See SPEC §Benchmark.
"""
from __future__ import annotations

import logging

from debate_kg.kg.graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def label_accuracy(predictions: list[str], labels: list[str]) -> float:
    """Fraction of predictions that exactly match the gold label.

    Args:
        predictions: Predicted label strings (SUPPORTS / REFUTES / NOT ENOUGH INFO).
        labels: Gold label strings, same length as predictions.

    Returns:
        Accuracy in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not predictions:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, labels))
    return correct / len(predictions)


def macro_f1(predictions: list[str], labels: list[str]) -> float:
    """Macro-averaged F1 across the three FEVER label classes.

    Args:
        predictions: Predicted label strings.
        labels: Gold label strings.

    Returns:
        Macro F1 in [0.0, 1.0]. Returns 0.0 for empty input.
    """
    if not predictions:
        return 0.0
    f1s: list[float] = []
    for cls in _LABELS:
        tp = sum(p == cls and g == cls for p, g in zip(predictions, labels))
        fp = sum(p == cls and g != cls for p, g in zip(predictions, labels))
        fn = sum(p != cls and g == cls for p, g in zip(predictions, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s)


def convergence_rate(rounds_list: list[int], max_rounds: int) -> float:
    """Fraction of claims that reached consensus strictly before max_rounds.

    Args:
        rounds_list: Rounds-to-consensus for each claim (1-indexed). Claims that
                     hit the hard cap should be recorded as max_rounds.
        max_rounds: The hard round cap from cfg.run.max_rounds.

    Returns:
        Convergence rate in [0.0, 1.0].
    """
    if not rounds_list:
        return 0.0
    converged = sum(r < max_rounds for r in rounds_list)
    return converged / len(rounds_list)


def kg_drift(kg_before: KnowledgeGraph, kg_after: KnowledgeGraph) -> float:
    """Symmetric set-difference of triple UUIDs between consecutive rounds.

    Args:
        kg_before: KG state before the merge.
        kg_after: KG state after the merge.

    Returns:
        |A Δ B| / max(|A|, |B|, 1) — fraction of triples that changed.
    """
    uuids_before = {t.uuid for t in kg_before.all_triples()}
    uuids_after = {t.uuid for t in kg_after.all_triples()}
    union = uuids_before | uuids_after
    if not union:
        return 0.0
    sym_diff = uuids_before.symmetric_difference(uuids_after)
    return len(sym_diff) / len(union)  # Jaccard distance, always in [0, 1]
