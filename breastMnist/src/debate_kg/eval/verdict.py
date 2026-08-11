"""Verdict classification: SUPPORTS / REFUTES / NOT ENOUGH INFO.

Weighted-vote approach: each DebateNode contributes its label
(SUPPORTS=+1, NOT ENOUGH INFO=0, REFUTES=-1) weighted by its mean
judge score (groundedness + factuality) / 2. The weighted average is
thresholded at ±0.5 to decide the final verdict.

No LLM call — fully deterministic given the judge scores.
"""
from __future__ import annotations

import logging
from collections import Counter

from breastMnist.src.debate_kg.kg.schema import DebateNode, JudgeScore

logger = logging.getLogger(__name__)

_LABEL_SCORE: dict[str, float] = {
    "SUPPORTS": 1.0,
    "NOT ENOUGH INFO": 0.0,
    "REFUTES": -1.0,
}
_NEI_THRESHOLD = 0.5


def classify_verdict(
    nodes: list[DebateNode],
    node_scores: dict[str, JudgeScore],
    consensus_reached: bool,
) -> str:
    """Classify the claim verdict via weighted vote over all debate nodes.

    Each node's label is mapped to +1 (SUPPORTS), 0 (NOT ENOUGH INFO),
    or -1 (REFUTES) and weighted by its mean judge score. The weighted
    average is thresholded at ±0.5. Falls back to unweighted majority
    when all weights are zero (e.g. no judge scores available).

    Returns one of "SUPPORTS", "REFUTES", "NOT ENOUGH INFO".
    """
    if not nodes:
        logger.debug("classify_verdict: no debate nodes — returning NOT ENOUGH INFO")
        return "NOT ENOUGH INFO"

    total_weight = 0.0
    weighted_sum = 0.0
    for node in nodes:
        score = node_scores.get(node.id)
        weight = (score.groundedness + score.factuality) / 2.0 if score else 0.0
        label_val = _LABEL_SCORE.get(node.label or "NOT ENOUGH INFO", 0.0)
        weighted_sum += weight * label_val
        total_weight += weight

    if total_weight == 0.0:
        counts = Counter(node.label or "NOT ENOUGH INFO" for node in nodes)
        verdict = counts.most_common(1)[0][0]
        verdict = verdict if verdict in _LABEL_SCORE else "NOT ENOUGH INFO"
    else:
        avg = weighted_sum / total_weight
        if avg > _NEI_THRESHOLD:
            verdict = "SUPPORTS"
        elif avg < -_NEI_THRESHOLD:
            verdict = "REFUTES"
        else:
            verdict = "NOT ENOUGH INFO"

    logger.info(
        "classify_verdict: weighted_avg=%.3f → %s (nodes=%d, consensus=%s)",
        weighted_sum / total_weight if total_weight > 0 else 0.0,
        verdict, len(nodes), consensus_reached,
    )
    return verdict
