"""Rule-based KG merger (v1 baseline).

Operates over the union of both KGs. Each triple gets a decision:
keep / drop / resolve in favour of the higher-scoring side.

Score formula for cited triples:
  triple_score = utterance_quality × max(expert_overall, 0.5)

Uncited triples (no debate node cited them) are always kept — they are
uncontested and cannot be scored. The floor only applies to cited triples.

Both output KGs receive the same merged content so experts converge toward
shared knowledge. Three minimal consistency checks warn (but do not drop).

See SPEC §Module6. The GNN merger is v2 — do not add it here.
"""
from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig

from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore, MergeResult, Triple

logger = logging.getLogger(__name__)

# Predicates whose domain is clearly symmetric — warn if inverse is absent.
_SYMMETRIC_PREDICATES: frozenset[str] = frozenset(
    {"marriedTo", "spouseOf", "siblingOf", "sameAs"}
)

# Predicates that are functional (at most one object per subject) — warn if violated.
_FUNCTIONAL_PREDICATES: frozenset[str] = frozenset(
    {"birthDate", "birthPlace", "deathDate", "deathPlace", "capital"}
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _expert_overall(
    expert_scores: dict[str, list[ExpertScore]],
    expert_id: str,
) -> float:
    """Mean normalized overall score for one expert. Returns 0.5 if unknown."""
    scores = expert_scores.get(expert_id, [])
    if not scores:
        return 0.5
    return sum(s.overall for s in scores) / (len(scores) * 100.0)


def _utterance_quality(
    node_id: str,
    scores: dict[str, list[JudgeScore]],
) -> float:
    """Mean (groundedness + factuality)/2 / 100 across all judges for one node.

    Returns 0.5 if no judge scores exist.
    """
    judge_scores = scores.get(node_id, [])
    if not judge_scores:
        return 0.5
    return sum((s.groundedness + s.factuality) / 2.0 for s in judge_scores) / (
        len(judge_scores) * 100.0
    )


def _triple_score(
    triple_uuid: str,
    expert_id: str,
    nodes: list[DebateNode],
    scores: dict[str, list[JudgeScore]],
    expert_overalls: dict[str, float],
) -> Optional[float]:
    """Composite score for a triple cited by the given expert's debate nodes.

    Returns None if no node from this expert cited the triple (uncited = always keep).

    Formula: utterance_quality × max(expert_overall, 0.5)
    The rescue floor max(e, 0.5) ensures a well-cited triple survives even from
    a weak expert rather than being dropped purely due to low overall performance.
    """
    citing = [n for n in nodes if triple_uuid in n.provenance and n.expert_id == expert_id]
    if not citing:
        return None
    u = sum(_utterance_quality(n.id, scores) for n in citing) / len(citing)
    e = max(expert_overalls.get(expert_id, 0.5), 0.5)
    return u * e


def _consistency_checks(triples: list[Triple]) -> None:
    """Warn-only checks: type, symmetry, functional predicate violations."""
    triple_index: dict[tuple[str, str], list[str]] = {}
    for t in triples:
        triple_index.setdefault((t.subject, t.predicate), []).append(t.object)

    existing_pairs: set[tuple[str, str, str]] = {
        (t.subject, t.predicate, t.object) for t in triples
    }

    for t in triples:
        pred_lower = t.predicate.lower()

        # Type check: numeric-named predicates should have numeric-looking objects
        if any(pred_lower.endswith(suffix) for suffix in ("date", "year", "count", "number")):
            try:
                float(t.object.replace("-", "", 1))
            except ValueError:
                logger.warning(
                    "consistency: numeric predicate %r has non-numeric object %r "
                    "(subject=%r)",
                    t.predicate, t.object, t.subject,
                )

        # Symmetry check: known symmetric predicates should have their inverse
        if t.predicate in _SYMMETRIC_PREDICATES:
            if (t.object, t.predicate, t.subject) not in existing_pairs:
                logger.warning(
                    "consistency: symmetric predicate %r missing inverse "
                    "(%r → %r exists but not %r → %r)",
                    t.predicate, t.subject, t.object, t.object, t.subject,
                )

    # Functional predicate check: at most one object per (subject, predicate)
    for (subject, predicate), objects in triple_index.items():
        if predicate in _FUNCTIONAL_PREDICATES and len(objects) > 1:
            logger.warning(
                "consistency: functional predicate %r has multiple objects for %r: %s",
                predicate, subject, objects,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_kgs(
    kg_a: KnowledgeGraph,
    kg_b: KnowledgeGraph,
    nodes: list[DebateNode],
    scores: dict[str, list[JudgeScore]],
    expert_scores: dict[str, list[ExpertScore]],
    cfg: DictConfig,
) -> tuple[KnowledgeGraph, KnowledgeGraph]:
    """Merge kg_a and kg_b based on judge scores from the current round.

    Args:
        kg_a: Expert A's current KG.
        kg_b: Expert B's current KG.
        nodes: All DebateNodes from the current round (carry provenance pointers).
        scores: Output of JudgePanel.score_debate() — {node_id: [JudgeScore, ...]}.
        expert_scores: Output of JudgePanel.score_experts_overall() — {expert_id: [ExpertScore]}.
        cfg: Hydra config. Reads cfg.run.merger_score_floor.

    Returns:
        (new_kg_a, new_kg_b) — both experts receive the same merged KG so they
        converge toward shared knowledge (see SPEC §Module6).
    """
    floor: float = cfg.run.merger_score_floor

    # Step 1 — Normalized expert overall scores
    expert_overalls = {
        eid: _expert_overall(expert_scores, eid)
        for eid in ("expert_a", "expert_b")
    }
    logger.debug(
        "merge_kgs: expert_overalls=%s floor=%.2f nodes=%d",
        expert_overalls, floor, len(nodes),
    )

    # Step 2 — Per-triple composite scores (None = uncited = always keep)
    a_scores: dict[str, Optional[float]] = {
        t.uuid: _triple_score(t.uuid, "expert_a", nodes, scores, expert_overalls)
        for t in kg_a.all_triples()
    }
    b_scores: dict[str, Optional[float]] = {
        t.uuid: _triple_score(t.uuid, "expert_b", nodes, scores, expert_overalls)
        for t in kg_b.all_triples()
    }

    # Step 3 — Detect conflicts: same (subject, predicate), different object, different UUID
    a_by_sp: dict[tuple[str, str], Triple] = {}
    for t in kg_a.all_triples():
        a_by_sp.setdefault((t.subject, t.predicate), t)  # first triple wins for key

    b_by_sp: dict[tuple[str, str], Triple] = {}
    for t in kg_b.all_triples():
        b_by_sp.setdefault((t.subject, t.predicate), t)

    conflicting_sps: set[tuple[str, str]] = {
        sp
        for sp in a_by_sp.keys() & b_by_sp.keys()
        if a_by_sp[sp].uuid != b_by_sp[sp].uuid
        and a_by_sp[sp].object != b_by_sp[sp].object
    }

    conflicting_a_uuids = {a_by_sp[sp].uuid for sp in conflicting_sps}
    conflicting_b_uuids = {b_by_sp[sp].uuid for sp in conflicting_sps}

    # Step 4 — Build merged triple list
    kept: list[str] = []
    dropped: list[str] = []
    resolved: list[str] = []
    merged_triples: list[Triple] = []
    merged_uuids: set[str] = set()

    # Resolve conflicts — take the higher-scoring triple; uncited counts as +inf
    for sp in conflicting_sps:
        t_a = a_by_sp[sp]
        t_b = b_by_sp[sp]
        score_a = a_scores[t_a.uuid]
        score_b = b_scores[t_b.uuid]
        eff_a = score_a if score_a is not None else float("inf")
        eff_b = score_b if score_b is not None else float("inf")
        winner, winning_eff = (t_a, eff_a) if eff_a >= eff_b else (t_b, eff_b)
        loser = t_b if winner is t_a else t_a

        if winning_eff == float("inf") or winning_eff >= floor:
            merged_triples.append(winner)
            merged_uuids.add(winner.uuid)
            resolved.append(winner.uuid)
        else:
            dropped.append(winner.uuid)
            dropped.append(loser.uuid)

    # Non-conflicting triples from kg_a
    for t in kg_a.all_triples():
        if t.uuid in conflicting_a_uuids:
            continue
        score = a_scores[t.uuid]
        if score is None or score >= floor:
            merged_triples.append(t)
            merged_uuids.add(t.uuid)
            kept.append(t.uuid)
        else:
            dropped.append(t.uuid)

    # Non-conflicting triples from kg_b (skip any UUID already merged)
    for t in kg_b.all_triples():
        if t.uuid in conflicting_b_uuids or t.uuid in merged_uuids:
            continue
        score = b_scores[t.uuid]
        if score is None or score >= floor:
            merged_triples.append(t)
            merged_uuids.add(t.uuid)
            kept.append(t.uuid)
        else:
            dropped.append(t.uuid)

    # Step 5 — Consistency checks (warn-only)
    _consistency_checks(merged_triples)

    # Step 6 — Log and return
    result = MergeResult(kept=kept, dropped=dropped, resolved=resolved)
    logger.info(
        "merge_kgs: kept=%d dropped=%d resolved=%d → merged=%d",
        len(result.kept), len(result.dropped), len(result.resolved), len(merged_triples),
    )

    new_kg = KnowledgeGraph()
    for t in merged_triples:
        new_kg.add_triple(t)

    return new_kg, new_kg
