"""Baseline merger strategies for ablation comparison.

All three functions share the same signature as merge_kgs so they are
drop-in replacements selected via cfg.run.merger:

  rule_based    → merger.rule_based.merge_kgs          (the real merger)
  naive_union   → baselines.naive_union_merge           (no scoring, no drops)
  random        → baselines.random_merge                (random keep/drop, seeded)

See SPEC §Baselines.
"""
from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig

from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore

logger = logging.getLogger(__name__)


def naive_union_merge(
    kg_a: KnowledgeGraph,
    kg_b: KnowledgeGraph,
    nodes: list[DebateNode],
    scores: dict[str, list[JudgeScore]],
    expert_scores: dict[str, list[ExpertScore]],
    cfg: DictConfig,
) -> tuple[KnowledgeGraph, KnowledgeGraph]:
    """Merge by taking the union of both KGs with no scoring or conflict resolution.

    Every triple from both KGs survives. Both experts receive the same result.
    Baseline: tests whether score-weighted merging adds value over naive union.
    """
    seen: set[str] = set()
    new_kg = KnowledgeGraph()
    for t in kg_a.all_triples():
        if t.uuid not in seen:
            new_kg.add_triple(t)
            seen.add(t.uuid)
    for t in kg_b.all_triples():
        if t.uuid not in seen:
            new_kg.add_triple(t)
            seen.add(t.uuid)
    logger.info("naive_union_merge: %d + %d → %d triples", len(kg_a), len(kg_b), len(new_kg))
    return new_kg, new_kg


def random_merge(
    kg_a: KnowledgeGraph,
    kg_b: KnowledgeGraph,
    nodes: list[DebateNode],
    scores: dict[str, list[JudgeScore]],
    expert_scores: dict[str, list[ExpertScore]],
    cfg: DictConfig,
) -> tuple[KnowledgeGraph, KnowledgeGraph]:
    """Merge by randomly keeping each triple from the union with fixed probability.

    Keep probability comes from cfg.run.random_merge_keep_prob (default 0.5).
    Uses cfg.data.seed for reproducibility.
    Baseline: sanity-checks whether the rule-based merger's scoring signal matters.
    """
    keep_prob: float = getattr(cfg.run, "random_merge_keep_prob", 0.5)
    rng = np.random.default_rng(cfg.data.seed)

    seen: set[str] = set()
    all_triples = []
    for t in kg_a.all_triples():
        if t.uuid not in seen:
            all_triples.append(t)
            seen.add(t.uuid)
    for t in kg_b.all_triples():
        if t.uuid not in seen:
            all_triples.append(t)
            seen.add(t.uuid)

    new_kg = KnowledgeGraph()
    for t in all_triples:
        if rng.random() < keep_prob:
            new_kg.add_triple(t)

    logger.info(
        "random_merge: %d union triples, kept %d (prob=%.2f, seed=%s)",
        len(all_triples), len(new_kg), keep_prob, cfg.data.seed,
    )
    return new_kg, new_kg
