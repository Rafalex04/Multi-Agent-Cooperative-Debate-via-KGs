"""Hydra entrypoint for the BreastMNIST VLM ablation pipeline.

Six modes are controlled by cfg.run.mode:
  single_agent          — one VLM call, label + confidence.
  debate_opinion        — free-text debate, judge picks winning expert.
  debate_graph          — structured debate, weighted node-score vote, no KG.
  debate_kg             — structured debate + shared domain KG.
  debate_kg_adversarial — structured debate + KG + fixed opposite stances.
  adaptive_adversarial  — single_agent first; escalates only when confidence
                          < cfg.run.confidence_threshold.

Run via:
  python -m debate_kg.breast_main run=breast_single
  python -m debate_kg.breast_main run=breast_adversarial data.num_samples=50
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from breastMnist.src.debate_kg.consensus.check import is_consensus
from breastMnist.src.debate_kg.data.breastmnist import load_samples
from breastMnist.src.debate_kg.debate.breast_orchestrator import (
    run_opinion_debate,
    run_single_agent,
    run_structured_debate,
)
from breastMnist.src.debate_kg.eval.breast_metrics import (
    accuracy,
    auc_roc,
    convergence_rate,
    sensitivity,
    specificity,
)
from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.loader import (
    load_definitions,
    load_kg_from_json,
    serialize_kg_for_prompt,
    serialize_triples_for_prompt,
)
from breastMnist.src.debate_kg.kg.schema import DebateEdge, DebateNode, JudgeScore, SingleAgentResult
from breastMnist.src.debate_kg.retriever.tfidf_retriever import TFIDFRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_DENSE_RANKER = None  # cached across samples; embedding the KG is the expensive part


def _observation_conditioned_kg_text(
    sample: dict,
    image_b64: str,
    cfg: DictConfig,
    kg: KnowledgeGraph,
    triple_by_uuid: dict,
    fallback_kg_text: str,
) -> str | None:
    """Stage A + Stage B: observe the image, retrieve triples conditioned on it.

    Returns the serialized KG block to use for every round of this sample, or
    None to fall back to the legacy path. Retrieval happens once — the same
    triples are shown in every round.
    """
    global _DENSE_RANKER
    from breastMnist.src.debate_kg.debate.observation import (
        load_observation, load_schema, observe, save_observation,
    )
    from breastMnist.src.debate_kg.models.ollama_client import OllamaClient
    from breastMnist.src.debate_kg.retriever.kg_retrieval import DenseRanker, retrieve

    kg_cfg = cfg.kg_retrieval
    schema_path = Path(str(getattr(kg_cfg, "schema_path", "data/breast/schema.json")))
    if not schema_path.exists():
        logger.error("schema.json not found at %s — run debate_kg.kg.kg_schema first", schema_path)
        return None

    schema = load_schema(schema_path)
    obs_dir = Path(str(getattr(kg_cfg, "observation_dir", "data/breast/observations")))

    obs = load_observation(sample["id"], obs_dir)
    if obs is None:
        client = OllamaClient(
            model=cfg.model.expert_model,
            base_url=cfg.model.ollama_base_url,
            num_ctx=int(cfg.model.num_ctx),
        )
        obs = observe(sample["id"], image_b64, schema, client,
                      max_attempts=int(getattr(kg_cfg, "observation_max_attempts", 3)))
        save_observation(obs, obs_dir)
    else:
        logger.info("sample %s: reusing cached observation", sample["id"])

    observed = {c: v for c, v in obs.values.items() if v != "uncertain"}
    logger.info("sample %s observation: %s", sample["id"], observed or "(all uncertain)")

    triples = [
        {"id": uid, "subject": t.subject, "relation": t.predicate, "object": t.object}
        for uid, t in triple_by_uuid.items()
    ]
    if _DENSE_RANKER is None and bool(getattr(kg_cfg, "use_dense", True)):
        _DENSE_RANKER = DenseRanker(triples, str(getattr(kg_cfg, "dense_model", "all-MiniLM-L6-v2")))

    result = retrieve(
        obs, triples, schema,
        stance_ids=set(schema.get("stance_triple_ids", [])),
        config={
            "anchor_conf_threshold": float(getattr(kg_cfg, "anchor_conf_threshold", 0.4)),
            "max_per_category": int(getattr(kg_cfg, "max_per_category", 3)),
            "max_stance_triples": int(getattr(kg_cfg, "max_stance_triples", 4)),
            "lambda_mmr": float(getattr(kg_cfg, "lambda_mmr", 0.6)),
            "hub_penalty_alpha": float(getattr(kg_cfg, "hub_penalty_alpha", 0.5)),
            "kg_token_budget": int(getattr(kg_cfg, "kg_token_budget", 200)),
            "use_dense": bool(getattr(kg_cfg, "use_dense", True)),
        },
        dense_ranker=_DENSE_RANKER,
    )

    if not result.triples:
        logger.warning("sample %s: retrieval returned nothing — using full KG", sample["id"])
        return fallback_kg_text

    selected = [triple_by_uuid[i] for i in result.ids() if i in triple_by_uuid]
    logger.info(
        "sample %s: frozen KG = %d/%d triples (~%d tok) held for all rounds",
        sample["id"], len(selected), len(triple_by_uuid), result.token_estimate,
    )
    return serialize_triples_for_prompt(selected)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _sample_prefix(sample_id: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', f"sample-{sample_id}".lower()).strip('-')


def _resume_path(cfg: DictConfig) -> Path:
    """Path for incremental per-sample results, used to resume after a crash."""
    return Path("outputs/resume") / f"{cfg.run.mode}_{cfg.data.num_samples}_{cfg.data.seed}.jsonl"


def _ollama_unload(model: str, base_url: str) -> None:
    """Evict a model from Ollama GPU memory (keep_alive=0)."""
    try:
        import requests as _req
        _req.post(f"{base_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
        logger.debug("Unloaded %s from GPU", model)
    except Exception as exc:
        logger.warning("Could not unload %s: %s", model, exc)


def _unload_expert_model(cfg: DictConfig) -> None:
    """Evict the expert model before the judge call to free VRAM."""
    if getattr(cfg.run, "use_stub_data", False):
        return
    base_url: str = getattr(cfg.model, "ollama_base_url", "http://localhost:11434")
    _ollama_unload(cfg.model.expert_model, base_url)


def _unload_judge_model(cfg: DictConfig) -> None:
    """Evict the judge model after its call so the expert model can reload."""
    if getattr(cfg.run, "use_stub_data", False):
        return
    backend: str = getattr(cfg.judges, "backend", "vllm")
    if backend != "ollama":
        return
    base_url: str = getattr(cfg.judges, "ollama_base_url", "http://localhost:11434")
    judge_model: str = getattr(cfg.judges, "ollama_model", "medgemma:4b")
    _ollama_unload(judge_model, base_url)


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------

def _weighted_vote(
    nodes: list[DebateNode],
    scores: dict[str, JudgeScore],
) -> tuple[str, float]:
    """Weighted majority vote over DebateNodes.

    Args:
        nodes: All debate nodes with label = BENIGN | MALIGNANT | None.
        scores: Node UUID → mean JudgeScore (groundedness, factuality).

    Returns:
        (verdict, raw_score) where raw_score > 0 → MALIGNANT, < 0 → BENIGN.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    benign_count = 0
    malignant_count = 0

    for node in nodes:
        if node.label not in ("BENIGN", "MALIGNANT"):
            continue
        vote = 1.0 if node.label == "MALIGNANT" else -1.0
        js = scores.get(node.id)
        weight = ((js.groundedness + js.factuality) / 2.0) / 100.0 if js else 0.5
        weighted_sum += weight * vote
        total_weight += weight
        if node.label == "MALIGNANT":
            malignant_count += 1
        else:
            benign_count += 1

    if total_weight == 0.0:
        raw = 0.0
        verdict = "MALIGNANT" if malignant_count >= benign_count else "BENIGN"
    else:
        raw = weighted_sum / total_weight
        if raw > 0:
            verdict = "MALIGNANT"
        elif raw < 0:
            verdict = "BENIGN"
        else:
            verdict = "MALIGNANT" if malignant_count >= benign_count else "BENIGN"

    return verdict, raw


def _aggregate_scores(
    utterance_scores: dict[str, list[JudgeScore]],
) -> dict[str, JudgeScore]:
    result: dict[str, JudgeScore] = {}
    for nid, jscores in utterance_scores.items():
        if not jscores:
            result[nid] = JudgeScore(groundedness=0.0, factuality=0.0)
        else:
            result[nid] = JudgeScore(
                groundedness=sum(s.groundedness for s in jscores) / len(jscores),
                factuality=sum(s.factuality for s in jscores) / len(jscores),
            )
    return result


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _run_mode_single(
    sample: dict,
    cfg: DictConfig,
    output_dir: Path,
) -> tuple[str, float, int]:
    """Returns (verdict, malignant_score, rounds_used)."""
    result, nodes, edges = run_single_agent(sample["image_b64"], cfg)
    prefix = _sample_prefix(sample["id"])
    _write_json(output_dir / f"debate_{prefix}.json", {
        "mode": "single_agent",
        "nodes": [n.model_dump() for n in nodes],
        "edges": [],
        "single_result": result.model_dump(),
    })
    # malignant_score: probability-like score in [0,1] for MALIGNANT class.
    # Used to compute dataset-level AUC-ROC across all samples.
    conf_0_1 = result.confidence / 100.0
    malignant_score = conf_0_1 if result.label == "MALIGNANT" else 1.0 - conf_0_1
    return result.label, malignant_score, 1


def _run_mode_opinion(
    sample: dict,
    cfg: DictConfig,
    output_dir: Path,
    judge,
) -> tuple[str, float, int]:
    """Run multi-round free-text debate; judge picks winner.

    Early stop when: both experts write [FINISH], or both agree on the same
    verdict for 2 consecutive rounds (consensus stop).
    """
    image_b64 = sample["image_b64"]
    all_nodes: list[DebateNode] = []
    rounds_used = cfg.run.max_rounds
    consecutive_agreement = 0

    for round_idx in range(cfg.run.max_rounds):
        new_nodes, _, finished = run_opinion_debate(image_b64, cfg, round_idx, history=all_nodes)
        all_nodes.extend(new_nodes)

        round_labels = [n.label for n in new_nodes if n.label]
        both_agree = len(set(round_labels)) == 1 and len(round_labels) == 2
        consecutive_agreement = consecutive_agreement + 1 if both_agree else 0

        if finished:
            rounds_used = round_idx + 1
            logger.info("opinion_debate: [FINISH] at round %d — stopping early for sample %s", round_idx, sample["id"])
            break
        if consecutive_agreement >= 2:
            rounds_used = round_idx + 1
            logger.info("opinion_debate: consensus for 2 rounds — stopping at round %d for sample %s", round_idx, sample["id"])
            break

    _unload_expert_model(cfg)
    judgment: object = judge.pick_winner(all_nodes, image_b64=image_b64)
    _unload_judge_model(cfg)
    winner_id = judgment.winner  # type: ignore[union-attr]

    # The winner's most recent stated label is the verdict.
    winner_nodes = [n for n in reversed(all_nodes) if n.expert_id == winner_id and n.label]
    verdict = winner_nodes[0].label if winner_nodes else "BENIGN"

    prefix = _sample_prefix(sample["id"])
    _write_json(output_dir / f"debate_{prefix}.json", {
        "mode": "debate_opinion",
        "nodes": [n.model_dump() for n in all_nodes],
        "judgment": judgment.model_dump(),  # type: ignore[union-attr]
        "verdict": verdict,
    })
    # Opinion mode has no confidence score — binary 0/1 malignant score.
    malignant_score = 1.0 if verdict == "MALIGNANT" else 0.0
    return verdict, malignant_score, rounds_used


def _run_mode_structured(
    sample: dict,
    cfg: DictConfig,
    output_dir: Path,
    judge,
    kg_text: str,
    adversarial: bool,
    mode_name: str,
    kg: KnowledgeGraph | None = None,
    retriever: "TFIDFRetriever | None" = None,
) -> tuple[str, float, int]:
    """Modes 3-5: structured [CLAIM] debate with node scoring.

    When retriever is provided (KG modes 4 and 5):
    - After each round, auto-retrieve the top-K most relevant KG triples for every new claim.
      These are stored in node.provenance (replacing LLM self-citation).
    - From round 1 onward, inject only the union of all accumulated provenance triples into
      the prompt instead of the full KG, focusing the model's context on what has been relevant.
    """
    image_b64 = sample["image_b64"]
    all_nodes: list[DebateNode] = []
    all_edges: list[DebateEdge] = []
    all_scores: dict[str, list[JudgeScore]] = {}
    rounds_used = cfg.run.max_rounds
    k_retrieve: int = int(getattr(cfg.run, "kg_retrieve_k", 5))

    # uuid → Triple map for efficient focused-KG construction
    triple_by_uuid = {t.uuid: t for t in kg.all_triples()} if kg is not None else {}

    # Seed query for round-0 KG pre-filtering (keeps prompt within context window).
    _KG_SEED_QUERY = (
        "breast ultrasound mass lesion shape margin orientation echogenicity "
        "posterior acoustic feature BI-RADS assessment category"
    )
    k_round0: int = int(getattr(cfg.run, "kg_round0_k", 30))

    kg_cfg = getattr(cfg, "kg_retrieval", None)
    kg_mode = str(getattr(kg_cfg, "mode", "constant_seed")) if kg_cfg else "constant_seed"

    # Observation-conditioned mode: retrieve ONCE from the Stage A observation and
    # hold that triple set fixed for every round, so the KG cannot shrink mid-debate
    # (the old cited_only path funnelled 30 -> 16 -> 21) and the only thing evolving
    # across rounds is the debate itself.
    frozen_kg_text: str | None = None
    if kg_mode == "observation_conditioned" and retriever is not None:
        frozen_kg_text = _observation_conditioned_kg_text(
            sample, image_b64, cfg, kg, triple_by_uuid, kg_text
        )

    for round_idx in range(cfg.run.max_rounds):
        if frozen_kg_text is not None:
            # Same triples in every round — retrieved once, before round 0.
            current_kg_text = frozen_kg_text
        elif retriever is not None and round_idx == 0:
            seed_triples = retriever.retrieve(_KG_SEED_QUERY, k=k_round0)
            current_kg_text = serialize_triples_for_prompt(seed_triples)
            logger.info(
                "round 0 sample %s: seed KG = %d/%d triples",
                sample["id"], len(seed_triples), len(triple_by_uuid),
            )
        elif retriever is not None and round_idx > 0 and all_nodes:
            prior_uuids = {uid for n in all_nodes for uid in n.provenance}
            focused_triples = [triple_by_uuid[uid] for uid in prior_uuids if uid in triple_by_uuid]
            current_kg_text = serialize_triples_for_prompt(focused_triples) if focused_triples else kg_text
            logger.info(
                "round %d sample %s: focused KG = %d/%d triples",
                round_idx, sample["id"], len(focused_triples), len(triple_by_uuid),
            )
        else:
            current_kg_text = kg_text

        new_nodes, new_edges = run_structured_debate(
            image_b64, cfg, round_idx, history=all_nodes,
            kg_text=current_kg_text, adversarial=adversarial,
        )

        # Auto-retrieve top-K KG triples for each new claim and store as provenance.
        if retriever is not None:
            for node in new_nodes:
                retrieved = retriever.retrieve(node.text, k=k_retrieve)
                node.provenance = [t.uuid for t in retrieved]
                logger.debug(
                    "  %s %s: auto-retrieved provenance = %s",
                    node.expert_id, node.short_id, node.provenance,
                )

        if judge is not None:
            # Unload expert model before judge to avoid GPU OOM on shared hardware.
            _unload_expert_model(cfg)
            for node in new_nodes:
                js = judge.score_utterance(node, KnowledgeGraph(), image_b64=image_b64)
                all_scores.setdefault(node.id, []).append(js)
            _unload_judge_model(cfg)

        all_nodes.extend(new_nodes)
        all_edges.extend(new_edges)

        if is_consensus(new_edges, round_idx, cfg):
            rounds_used = round_idx + 1
            logger.info("Consensus at round %d for sample %s", round_idx, sample["id"])
            break

    agg = _aggregate_scores(all_scores)
    verdict, raw_score = _weighted_vote(all_nodes, agg)

    prefix = _sample_prefix(sample["id"])
    _write_json(output_dir / f"debate_{prefix}.json", {
        "mode": mode_name,
        "nodes": [n.model_dump() for n in all_nodes],
        "edges": [e.model_dump() for e in all_edges],
        "scores": {nid: [s.model_dump() for s in jscores] for nid, jscores in all_scores.items()},
        "verdict": verdict,
        "raw_score": raw_score,
    })
    # malignant_score: map raw_score from [-1,1] to [0,1].
    malignant_score = float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))
    return verdict, malignant_score, rounds_used


def _run_mode_adaptive(
    sample: dict,
    cfg: DictConfig,
    output_dir: Path,
    judge,
    kg_text: str,
    kg: KnowledgeGraph | None = None,
    retriever: "TFIDFRetriever | None" = None,
) -> tuple[str, float, int]:
    """Mode 6: single agent first; escalate to adversarial debate if confidence low."""
    threshold: int = int(getattr(cfg.run, "confidence_threshold", 70))
    result, nodes, _ = run_single_agent(sample["image_b64"], cfg)

    if result.confidence >= threshold:
        logger.info(
            "adaptive: sample %s confidence %d >= %d → using single-agent verdict %s",
            sample["id"], result.confidence, threshold, result.label,
        )
        prefix = _sample_prefix(sample["id"])
        _write_json(output_dir / f"debate_{prefix}.json", {
            "mode": "adaptive_adversarial",
            "escalated": False,
            "single_result": result.model_dump(),
            "nodes": [n.model_dump() for n in nodes],
        })
        conf_0_1 = result.confidence / 100.0
        malignant_score = conf_0_1 if result.label == "MALIGNANT" else 1.0 - conf_0_1
        return result.label, malignant_score, 1

    logger.info(
        "adaptive: sample %s confidence %d < %d → escalating to adversarial debate",
        sample["id"], result.confidence, threshold,
    )
    verdict, malignant_score, rounds_used = _run_mode_structured(
        sample, cfg, output_dir, judge, kg_text,
        adversarial=True, mode_name="adaptive_adversarial_escalated",
        kg=kg, retriever=retriever,
    )
    return verdict, malignant_score, rounds_used


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf", config_name="config_breast")
def main(cfg: DictConfig) -> None:
    load_dotenv()
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Starting BreastMNIST run: mode=%s samples=%d", cfg.run.mode, cfg.data.num_samples)

    from hydra.core.hydra_config import HydraConfig
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    # Load KG (used in modes 4, 5, 6).
    needs_kg = cfg.run.mode in ("debate_kg", "debate_kg_adversarial", "adaptive_adversarial")
    kg_text = ""
    kg: KnowledgeGraph | None = None
    retriever: TFIDFRetriever | None = None
    if needs_kg:
        kg_path = Path(getattr(cfg.data, "kg_path", "data/breast/knowledge_graph.json"))
        defs_path = Path(getattr(cfg.data, "definitions_path", "data/breast/definitions.json"))
        kg = load_kg_from_json(kg_path)
        definitions = load_definitions(defs_path)
        kg_text = serialize_kg_for_prompt(kg, definitions)
        logger.info("Loaded KG (%d triples) for mode %s", len(kg), cfg.run.mode)
        # Build TF-IDF retriever for automatic KG grounding (modes 4 and 5).
        if cfg.run.mode in ("debate_kg", "debate_kg_adversarial"):
            retriever = TFIDFRetriever(kg.all_triples())
            logger.info("TF-IDF retriever built (k=%d)", getattr(cfg.run, "kg_retrieve_k", 5))

    # Build judge (not needed for single_agent, or when skip_judge=true).
    judge = None
    skip_judge = bool(getattr(cfg.run, "skip_judge", False))
    if cfg.run.mode != "single_agent" and not skip_judge:
        from breastMnist.src.debate_kg.judges.medgemma_judge import build_medgemma_judge
        judge = build_medgemma_judge(cfg)
    if skip_judge:
        logger.info("skip_judge=true: judge disabled; verdict will use equal-weight majority vote")

    samples = load_samples(cfg)
    predictions: list[str] = []
    gold_labels: list[str] = []
    auc_scores: list[float] = []
    rounds_list: list[int] = []
    per_sample_records: list[dict] = []

    sample_times: list[float] = []

    # Resume support: reload any per-sample results from a prior crashed run
    # of this exact (mode, num_samples, seed) combination and skip them.
    resume_path = _resume_path(cfg)
    done_ids: set[str] = set()
    if resume_path.exists():
        for line in resume_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            done_ids.add(record["id"])
            predictions.append(record["verdict"])
            gold_labels.append(record["label"])
            auc_scores.append(record["malignant_score"])
            rounds_list.append(record["rounds"])
            sample_times.append(record["time_s"])
            per_sample_records.append(record)
        logger.info("Resuming %s: %d sample(s) already completed, skipping", cfg.run.mode, len(done_ids))

    for sample in samples:
        if sample["id"] in done_ids:
            logger.info("Sample %s — already completed, skipping", sample["id"])
            continue
        logger.info("Sample %s — gold=%s", sample["id"], sample["label"])
        t0 = time.perf_counter()

        mode = cfg.run.mode
        if mode == "single_agent":
            verdict, malignant_score, rounds = _run_mode_single(sample, cfg, output_dir)
        elif mode == "debate_opinion":
            verdict, malignant_score, rounds = _run_mode_opinion(sample, cfg, output_dir, judge)
        elif mode == "debate_graph":
            verdict, malignant_score, rounds = _run_mode_structured(
                sample, cfg, output_dir, judge, kg_text="", adversarial=False, mode_name=mode,
            )
        elif mode == "debate_kg":
            verdict, malignant_score, rounds = _run_mode_structured(
                sample, cfg, output_dir, judge, kg_text=kg_text, adversarial=False, mode_name=mode,
                kg=kg, retriever=retriever,
            )
        elif mode == "debate_kg_adversarial":
            verdict, malignant_score, rounds = _run_mode_structured(
                sample, cfg, output_dir, judge, kg_text=kg_text, adversarial=True, mode_name=mode,
                kg=kg, retriever=retriever,
            )
        elif mode == "adaptive_adversarial":
            verdict, malignant_score, rounds = _run_mode_adaptive(
                sample, cfg, output_dir, judge, kg_text, kg=kg, retriever=retriever,
            )
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        elapsed = time.perf_counter() - t0
        logger.info(
            "Sample %s → %s (gold=%s, rounds=%d, time=%.1fs)",
            sample["id"], verdict, sample["label"], rounds, elapsed,
        )
        predictions.append(verdict)
        gold_labels.append(sample["label"])
        auc_scores.append(malignant_score)
        rounds_list.append(rounds)
        sample_times.append(elapsed)
        record = {
            "id": sample["id"],
            "verdict": verdict,
            "label": sample["label"],
            "malignant_score": malignant_score,
            "rounds": rounds,
            "time_s": round(elapsed, 2),
        }
        per_sample_records.append(record)
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        with resume_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # Aggregate metrics.
    acc = accuracy(predictions, gold_labels)
    sens = sensitivity(predictions, gold_labels)
    spec = specificity(predictions, gold_labels)
    auc = auc_roc(auc_scores, gold_labels)
    conv = convergence_rate(rounds_list, cfg.run.max_rounds)
    mean_rounds = float(np.mean(rounds_list)) if rounds_list else 0.0
    mean_time = float(np.mean(sample_times)) if sample_times else 0.0
    total_time = float(np.sum(sample_times)) if sample_times else 0.0

    metrics = {
        "mode": cfg.run.mode,
        "num_samples": len(predictions),
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "auc_roc": auc,
        "convergence_rate": conv,
        "mean_rounds": mean_rounds,
        "mean_time_s": round(mean_time, 2),
        "total_time_s": round(total_time, 2),
        "per_sample": per_sample_records,
    }
    _write_json(output_dir / "metrics.json", metrics)
    resume_path.unlink(missing_ok=True)

    logger.info(
        "Done. acc=%.4f sens=%.4f spec=%.4f auc=%.4f conv=%.4f mean_rounds=%.2f mean_time=%.1fs total=%.0fs",
        acc, sens, spec, auc, conv, mean_rounds, mean_time, total_time,
    )


if __name__ == "__main__":
    main()
