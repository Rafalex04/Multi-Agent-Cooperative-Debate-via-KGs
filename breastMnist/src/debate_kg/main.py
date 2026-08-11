"""Hydra entrypoint: round loop, logging setup, orchestration of all modules.

Reproducibility note: KG construction, retrieval ranking, merge decisions, and
random-merger are all seeded via cfg.data.seed. LLM output is deterministic only
when cfg.model.expert_temperature=0 on the same hardware/Ollama build.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from breastMnist.src.debate_kg.consensus.check import is_consensus
from breastMnist.src.debate_kg.data.fever import load_claims
from breastMnist.src.debate_kg.debate.orchestrator import run_debate, run_single_expert
from breastMnist.src.debate_kg.eval.metrics import convergence_rate, kg_drift, label_accuracy, macro_f1
from breastMnist.src.debate_kg.eval.verdict import classify_verdict
from breastMnist.src.debate_kg.judges.panel import build_judge_panel
from breastMnist.src.debate_kg.kg.construct import build_kg_pair
from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateEdge, DebateNode, ExpertScore, JudgeScore  # ExpertScore used by MergerFn
from breastMnist.src.debate_kg.merger.baselines import naive_union_merge, random_merge
from breastMnist.src.debate_kg.merger.rule_based import merge_kgs
from breastMnist.src.debate_kg.retriever.retriever import retrieve

logger = logging.getLogger(__name__)

MergerFn = Callable[
    [KnowledgeGraph, KnowledgeGraph, list[DebateNode], dict[str, list[JudgeScore]], dict[str, list[ExpertScore]], DictConfig],
    tuple[KnowledgeGraph, KnowledgeGraph],
]

_MERGERS: dict[str, MergerFn] = {
    "rule_based": merge_kgs,
    "naive_union": naive_union_merge,
    "random": random_merge,
}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _claim_slug(text: str, max_len: int = 45) -> str:
    """Turn claim text into a filesystem-safe slug, e.g. 'beethoven-was-born-in-1770'."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:max_len].rstrip('-')


def _claim_prefix(claim_id: str, claim_text: str) -> str:
    """Combine numeric ID with slug: '75193_beethoven-was-born-in-1770'."""
    return f"{claim_id}_{_claim_slug(claim_text)}"


def _write_kg_snapshot(
    output_dir: Path,
    claim_prefix: str,
    round_idx: int,
    kg_a: KnowledgeGraph,
    kg_b: KnowledgeGraph,
) -> None:
    base = output_dir / "kgs" / claim_prefix / f"round_{round_idx}"
    _write_json(base / "a.json", kg_a.to_dict())
    _write_json(base / "b.json", kg_b.to_dict())


def _write_debate_artifact(
    output_dir: Path,
    claim_prefix: str,
    nodes: list[DebateNode],
    edges: list[DebateEdge],
) -> None:
    data = {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }
    _write_json(output_dir / f"debate_{claim_prefix}.json", data)


def _write_judge_scores_artifact(
    output_dir: Path,
    claim_prefix: str,
    utterance_scores: dict[str, list[JudgeScore]],
    expert_scores: dict[str, list[ExpertScore]],
    alpha: float,
) -> None:
    data = {
        "utterances": {
            nid: [s.model_dump() for s in jscores]
            for nid, jscores in utterance_scores.items()
        },
        "experts": {
            eid: [s.model_dump() for s in escores]
            for eid, escores in expert_scores.items()
        },
        "inter_judge_alpha": alpha,
    }
    _write_json(output_dir / f"judge_scores_{claim_prefix}.json", data)


# ---------------------------------------------------------------------------
# Accumulation helpers
# ---------------------------------------------------------------------------

def _merge_expert_scores(
    accum: dict[str, list[ExpertScore]],
    new: dict[str, list[ExpertScore]],
) -> None:
    """Append new expert scores into accum in-place."""
    for eid, scores in new.items():
        accum.setdefault(eid, []).extend(scores)


def _merge_utterance_scores(
    accum: dict[str, list[JudgeScore]],
    new: dict[str, list[JudgeScore]],
) -> None:
    for nid, scores in new.items():
        accum[nid] = scores


# ---------------------------------------------------------------------------
# Single-expert mode
# ---------------------------------------------------------------------------

def _run_single_expert_claim(
    claim_text: str,
    claim_id: str,
    kg_a: KnowledgeGraph,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[str, list[DebateNode]]:
    """Run one expert turn (no debate), classify verdict, write artifacts.

    Returns (verdict, nodes).
    """
    prefix = _claim_prefix(claim_id, claim_text)
    triples_a = retrieve(kg_a, claim_text, cfg)
    nodes, edges = run_single_expert(claim_text, triples_a, cfg)

    _write_debate_artifact(output_dir, prefix, nodes, edges)
    _write_json(output_dir / f"judge_scores_{prefix}.json", {"utterances": {}, "experts": {}, "inter_judge_alpha": 0.0})
    _write_kg_snapshot(output_dir, prefix, 0, kg_a, KnowledgeGraph())

    verdict = classify_verdict(nodes, node_scores={}, consensus_reached=False)
    return verdict, nodes


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info(
        "Starting run: mode=%s merger=%s claims=%d max_rounds=%d",
        cfg.run.mode, cfg.run.merger, cfg.run.num_claims, cfg.run.max_rounds,
    )

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    merger_fn = _MERGERS.get(cfg.run.merger)
    if merger_fn is None:
        raise ValueError(f"Unknown merger: {cfg.run.merger!r}. Choose from {list(_MERGERS)}")

    claims = load_claims(cfg)
    panel = build_judge_panel(cfg)

    predictions: list[str] = []
    labels: list[str] = []
    rounds_list: list[int] = []
    per_claim_drift: list[float] = []
    per_claim_records: list[dict] = []

    for claim_row in claims:
        claim_id = str(claim_row["id"])
        claim_text = claim_row["claim"]
        gold_label = claim_row["label"]
        prefix = _claim_prefix(claim_id, claim_text)
        logger.info("Claim %s: %s", claim_id, claim_text)

        kg_a, kg_b = build_kg_pair(claim_text, cfg)

        # ---- single-expert baseline ----
        if cfg.run.mode == "single_expert":
            verdict, _ = _run_single_expert_claim(claim_text, claim_id, kg_a, cfg, output_dir)
            predictions.append(verdict)
            labels.append(gold_label)
            rounds_list.append(1)
            per_claim_records.append({
                "id": claim_id, "claim_text": claim_text, "slug": _claim_slug(claim_text),
                "verdict": verdict, "label": gold_label, "rounds": 1,
            })
            logger.info("Verdict for %s: %s", claim_id, verdict)
            continue

        # ---- full debate pipeline ----
        all_nodes: list[DebateNode] = []
        all_edges: list[DebateEdge] = []
        all_utterance_scores: dict[str, list[JudgeScore]] = {}
        all_expert_scores: dict[str, list[ExpertScore]] = {}
        rounds_to_consensus = cfg.run.max_rounds
        consensus_reached = False
        last_alpha = 0.0
        round_drifts: list[float] = []
        prev_kg_a = kg_a

        for round_idx in range(cfg.run.max_rounds):
            logger.debug("Round %d", round_idx)

            triples_a = retrieve(kg_a, claim_text, cfg)
            triples_b = retrieve(kg_b, claim_text, cfg)
            nodes, edges = run_debate(claim_text, triples_a, triples_b, cfg, round_idx, history=all_nodes)
            scores = panel.score_debate(nodes, kg_a, kg_b, claim=claim_text)
            expert_scores_round = panel.score_experts_overall(nodes)
            last_alpha = panel.inter_judge_agreement(scores)
            logger.info("Inter-judge α (round %d): %.4f", round_idx, last_alpha)

            all_nodes.extend(nodes)
            all_edges.extend(edges)
            _merge_utterance_scores(all_utterance_scores, scores)
            _merge_expert_scores(all_expert_scores, expert_scores_round)

            _write_kg_snapshot(output_dir, prefix, round_idx, kg_a, kg_b)

            if is_consensus(edges, round_idx, cfg):
                consensus_reached = True
                rounds_to_consensus = round_idx + 1
                logger.info("Consensus at round %d for claim %s", round_idx, claim_id)
                break

            new_kg_a, new_kg_b = merger_fn(kg_a, kg_b, nodes, scores, expert_scores_round, cfg)
            drift = kg_drift(prev_kg_a, new_kg_a)
            round_drifts.append(drift)
            logger.info("KG drift (round %d): %.4f", round_idx, drift)
            prev_kg_a = new_kg_a
            kg_a, kg_b = new_kg_a, new_kg_b

        _write_debate_artifact(output_dir, prefix, all_nodes, all_edges)
        _write_judge_scores_artifact(output_dir, prefix, all_utterance_scores, all_expert_scores, last_alpha)

        agg_scores = panel.aggregate_scores(all_utterance_scores)
        verdict = classify_verdict(all_nodes, node_scores=agg_scores, consensus_reached=consensus_reached)
        logger.info("Verdict for %s: %s", claim_id, verdict)

        predictions.append(verdict)
        labels.append(gold_label)
        rounds_list.append(rounds_to_consensus)
        mean_drift = float(np.mean(round_drifts)) if round_drifts else 0.0
        per_claim_drift.append(mean_drift)
        per_claim_records.append({
            "id": claim_id,
            "claim_text": claim_text,
            "slug": _claim_slug(claim_text),
            "verdict": verdict,
            "label": gold_label,
            "rounds": rounds_to_consensus,
            "consensus_reached": consensus_reached,
            "mean_drift": mean_drift,
            "inter_judge_alpha": last_alpha,
        })

    # ---- aggregate metrics ----
    acc = label_accuracy(predictions, labels)
    mf1 = macro_f1(predictions, labels)
    conv = convergence_rate(rounds_list, cfg.run.max_rounds)
    mean_drift_all = float(np.mean(per_claim_drift)) if per_claim_drift else 0.0
    mean_rounds = float(np.mean(rounds_list)) if rounds_list else 0.0

    metrics = {
        "label_accuracy": acc,
        "macro_f1": mf1,
        "convergence_rate": conv,
        "mean_kg_drift": mean_drift_all,
        "mean_rounds_to_consensus": mean_rounds,
        "num_claims": len(predictions),
        "mode": cfg.run.mode,
        "merger": cfg.run.merger,
        "per_claim": per_claim_records,
    }
    _write_json(output_dir / "metrics.json", metrics)

    claims_index = {"claims": [
        {"id": r["id"], "text": r["claim_text"], "slug": r["slug"], "verdict": r["verdict"]}
        for r in per_claim_records
    ]}
    _write_json(output_dir / "claims_index.json", claims_index)

    logger.info(
        "Done. accuracy=%.4f macro_f1=%.4f convergence=%.4f mean_drift=%.4f mean_rounds=%.2f",
        acc, mf1, conv, mean_drift_all, mean_rounds,
    )


if __name__ == "__main__":
    main()
