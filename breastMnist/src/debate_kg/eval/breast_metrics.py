"""Binary classification metrics for the BreastMNIST ablation pipeline.

All functions operate on string labels: "BENIGN" | "MALIGNANT".
MALIGNANT is treated as the positive class.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_POS = "MALIGNANT"
_NEG = "BENIGN"


def _to_binary(labels: list[str]) -> list[int]:
    """Convert string labels to int (MALIGNANT=1, BENIGN=0)."""
    return [1 if lbl == _POS else 0 for lbl in labels]


def accuracy(predictions: list[str], labels: list[str]) -> float:
    """Fraction of correctly classified samples."""
    if not labels:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, labels))
    return correct / len(labels)


def sensitivity(predictions: list[str], labels: list[str]) -> float:
    """True positive rate: TP / (TP + FN). Recall for MALIGNANT class."""
    tp = sum(p == _POS and g == _POS for p, g in zip(predictions, labels))
    fn = sum(p == _NEG and g == _POS for p, g in zip(predictions, labels))
    denom = tp + fn
    if denom == 0:
        logger.warning("sensitivity: no positive samples — returning 0.0")
        return 0.0
    return tp / denom


def specificity(predictions: list[str], labels: list[str]) -> float:
    """True negative rate: TN / (TN + FP). Recall for BENIGN class."""
    tn = sum(p == _NEG and g == _NEG for p, g in zip(predictions, labels))
    fp = sum(p == _POS and g == _NEG for p, g in zip(predictions, labels))
    denom = tn + fp
    if denom == 0:
        logger.warning("specificity: no negative samples — returning 0.0")
        return 0.0
    return tn / denom


def auc_roc(scores: list[float], labels: list[str]) -> float:
    """Area under the ROC curve using sklearn.

    Args:
        scores: Continuous malignancy scores (higher = more likely MALIGNANT).
                For the weighted-vote verdict this is the raw weighted sum before
                thresholding; for single/opinion verdicts use 1.0 (MALIGNANT) or 0.0.
        labels: Ground-truth string labels.

    Returns:
        AUC-ROC float in [0, 1], or 0.5 if only one class is present.
    """
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    y_true = _to_binary(labels)
    if len(set(y_true)) < 2:
        logger.warning("auc_roc: only one class present — returning 0.5")
        return 0.5
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception as exc:
        logger.warning("auc_roc: sklearn error %s — returning 0.5", exc)
        return 0.5


def convergence_rate(rounds_list: list[int], max_rounds: int) -> float:
    """Fraction of samples that reached consensus before the hard cap."""
    if not rounds_list:
        return 0.0
    return sum(r < max_rounds for r in rounds_list) / len(rounds_list)
