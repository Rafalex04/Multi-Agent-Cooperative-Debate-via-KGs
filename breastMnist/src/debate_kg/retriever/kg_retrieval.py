"""Stage B — observation-conditioned KG retrieval.

Replaces the constant-seed TF-IDF retrieval that gave every sample the same 30
triples. Selection is driven by the Stage A observation, so two different
images retrieve different triples.

Ranking combines three signals:

  anchors    triples mentioning an observed value (confidence >= threshold).
             This is what makes retrieval vary per sample.
  expansion  1-hop KG neighbours of anchored entities. This is the mechanism
             that reaches the pathology-appearance triples the old retrieval
             never surfaced (56 of 113 never cited).
  dense      sentence-transformer cosine similarity against the observation
             notes, for everything else. Replaces TF-IDF, which on short
             descriptor strings produced a generic-triple monoculture.

Diversity controls: per-category quota, MMR, hub penalty, stance cap, and a
hard token budget.

Nothing here is ontology-specific — relation roles come from conf/kg_schema.yaml
and categories from schema.json.

IMPORTANT — no label leakage: ranking uses only semantic/structural relevance to
the observation. No signal is fitted against ground-truth labels. Introducing
label-correlated ranking would leak supervision into a zero-shot pipeline.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "anchor_conf_threshold": 0.4,
    "max_per_category": 3,
    "max_stance_triples": 4,
    "lambda_mmr": 0.6,
    "hub_penalty_alpha": 0.5,
    "kg_token_budget": 200,
    "dense_model": "all-MiniLM-L6-v2",
    "use_dense": True,
}

# Rough token estimate; avoids a tokeniser dependency in the hot path.
_CHARS_PER_TOKEN = 4


@dataclass
class RetrievalResult:
    """Triples selected for one sample, with provenance for logging."""

    triples: list = field(default_factory=list)
    source: dict[str, str] = field(default_factory=dict)   # triple id -> anchor|expansion|dense
    dropped: int = 0
    token_estimate: int = 0

    def ids(self) -> list[str]:
        return [t.uuid if hasattr(t, "uuid") else t["id"] for t in self.triples]


def _tid(triple) -> str:
    return triple.uuid if hasattr(triple, "uuid") else triple["id"]


def _fields(triple) -> tuple[str, str, str]:
    if hasattr(triple, "subject"):
        return triple.subject, triple.relation if hasattr(triple, "relation") else triple.predicate, triple.object
    return triple["subject"], triple["relation"], triple["object"]


def verbalise(triple) -> str:
    """Human-readable form of a triple, used for embedding and prompts."""
    s, r, o = _fields(triple)
    return f"{s.replace('_', ' ')} {r.replace('_', ' ')} {o.replace('_', ' ')}"


def _token_len(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class DenseRanker:
    """Sentence-transformer cosine ranker over verbalised triples.

    Loaded lazily and cached — the model download happens once.
    """

    _model = None

    def __init__(self, triples: list, model_name: str):
        self.triples = triples
        self.model_name = model_name
        self._emb = None
        self._index = {_tid(t): i for i, t in enumerate(triples)}

    def _ensure(self):
        if DenseRanker._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer %s", self.model_name)
            DenseRanker._model = SentenceTransformer(self.model_name)
        if self._emb is None:
            self._emb = DenseRanker._model.encode(
                [verbalise(t) for t in self.triples], normalize_embeddings=True
            )
        return DenseRanker._model, self._emb

    def scores(self, query: str) -> dict[str, float]:
        """Cosine similarity of every triple against `query`."""
        if not query.strip():
            return {}
        model, emb = self._ensure()
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = emb @ q
        return {_tid(t): float(s) for t, s in zip(self.triples, sims)}

    def pair_similarity(self, a, b) -> float:
        _, emb = self._ensure()
        ia, ib = self._index.get(_tid(a)), self._index.get(_tid(b))
        if ia is None or ib is None:
            return 0.0
        return float(emb[ia] @ emb[ib])


def _lexical_scores(query: str, triples: list) -> dict[str, float]:
    """Token-overlap fallback when dense ranking is unavailable."""
    q = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
    out = {}
    for t in triples:
        toks = set(re.sub(r"[^a-z0-9 ]", " ", verbalise(t).lower()).split())
        out[_tid(t)] = len(q & toks) / len(q | toks) if (q and toks) else 0.0
    return out


def _entity_category(schema: dict) -> dict[str, str]:
    """value entity -> its category."""
    return {
        v["value"]: c["category"]
        for c in schema["categories"]
        for v in c["values"]
    }


def retrieve(
    observation,
    triples: list,
    schema: dict,
    stance_ids: set[str] | None = None,
    citation_counts: dict[str, int] | None = None,
    config: dict | None = None,
    dense_ranker: "DenseRanker | None" = None,
) -> RetrievalResult:
    """Select triples for one sample, conditioned on its observation.

    Args:
        observation: Stage A Observation (needs .anchors() and .query_text()).
        triples: all KG triples.
        schema: schema.json contents.
        stance_ids: ids of stance-bearing triples, capped separately.
        citation_counts: prior-run citation frequency for the hub penalty;
            falls back to KG degree when absent.
        config: overrides for _DEFAULTS.
        dense_ranker: reuse across samples to avoid re-embedding the KG.

    Returns:
        RetrievalResult with the chosen triples in priority order.
    """
    cfg = {**_DEFAULTS, **(config or {})}
    stance_ids = stance_ids or set()
    by_id = {_tid(t): t for t in triples}

    anchors = set(observation.anchors(cfg["anchor_conf_threshold"]))
    ent_cat = _entity_category(schema)

    # --- signal 1: anchor match -------------------------------------------
    anchor_hits: list = []
    for t in triples:
        s, _, o = _fields(t)
        if s in anchors or o in anchors:
            anchor_hits.append(t)

    # --- signal 2: 1-hop expansion ----------------------------------------
    anchored_entities = set(anchors)
    for t in anchor_hits:
        s, _, o = _fields(t)
        anchored_entities.add(s)
        anchored_entities.add(o)

    anchor_ids = {_tid(t) for t in anchor_hits}
    expansion: list = []
    for t in triples:
        if _tid(t) in anchor_ids:
            continue
        s, _, o = _fields(t)
        if s in anchored_entities or o in anchored_entities:
            expansion.append(t)

    # --- signal 3: dense similarity ---------------------------------------
    query = observation.query_text()
    chosen_ids = anchor_ids | {_tid(t) for t in expansion}
    remainder = [t for t in triples if _tid(t) not in chosen_ids]

    if cfg["use_dense"] and remainder:
        ranker = dense_ranker or DenseRanker(triples, cfg["dense_model"])
        try:
            sims = ranker.scores(query)
        except Exception as exc:  # model download/load failure — degrade, don't crash
            logger.warning("dense ranking unavailable (%s); falling back to lexical", exc)
            ranker = None
            sims = _lexical_scores(query, remainder)
    else:
        ranker = None
        sims = _lexical_scores(query, remainder)

    # --- hub penalty -------------------------------------------------------
    if citation_counts:
        hub = citation_counts
    else:
        hub = {}
        for t in triples:
            s, _, o = _fields(t)
            for e in (s, o):
                hub[e] = hub.get(e, 0) + 1
        hub = {_tid(t): hub.get(_fields(t)[0], 0) + hub.get(_fields(t)[2], 0) for t in triples}

    max_hub = max(hub.values()) if hub else 1
    alpha = cfg["hub_penalty_alpha"]

    def penalised(tid: str, base: float) -> float:
        return base - alpha * (hub.get(tid, 0) / max_hub if max_hub else 0.0)

    dense_sorted = sorted(
        remainder,
        key=lambda t: penalised(_tid(t), sims.get(_tid(t), 0.0)),
        reverse=True,
    )

    # --- assemble under quotas and budget ---------------------------------
    result = RetrievalResult()
    per_category: dict[str, int] = {}
    n_stance = 0
    tokens = 0
    budget = cfg["kg_token_budget"]
    seen: set[str] = set()
    considered = 0

    def try_add(t, source: str) -> bool:
        nonlocal tokens, n_stance, considered
        considered += 1
        tid = _tid(t)
        if tid in seen:
            return False
        if tid in stance_ids:
            if n_stance >= cfg["max_stance_triples"]:
                return False
        s, _, o = _fields(t)
        cat = ent_cat.get(s) or ent_cat.get(o)
        if cat is not None and per_category.get(cat, 0) >= cfg["max_per_category"]:
            return False
        cost = _token_len(verbalise(t)) + 6  # + id prefix and newline
        if tokens + cost > budget:
            return False
        seen.add(tid)
        result.triples.append(t)
        result.source[tid] = source
        tokens += cost
        if cat is not None:
            per_category[cat] = per_category.get(cat, 0) + 1
        if tid in stance_ids:
            n_stance += 1
        return True

    # MMR over dense candidates: balance relevance against redundancy.
    def mmr_order(cands: list) -> list:
        if not ranker or len(cands) < 2:
            return cands
        lam = cfg["lambda_mmr"]
        selected: list = []
        pool = list(cands)
        while pool:
            best, best_score = None, -math.inf
            for t in pool[:40]:  # cap the pool scan; candidates are pre-sorted
                rel = penalised(_tid(t), sims.get(_tid(t), 0.0))
                red = max((ranker.pair_similarity(t, s) for s in selected), default=0.0)
                score = lam * rel - (1 - lam) * red
                if score > best_score:
                    best, best_score = t, score
            if best is None:
                break
            selected.append(best)
            pool.remove(best)
            if len(selected) >= 60:
                break
        return selected

    for t in anchor_hits:
        try_add(t, "anchor")
    for t in expansion:
        try_add(t, "expansion")
    for t in mmr_order(dense_sorted):
        try_add(t, "dense")

    result.dropped = considered - len(result.triples)
    result.token_estimate = tokens

    logger.info(
        "  retrieval: %d triples (%d anchor, %d expansion, %d dense) | "
        "%d stance | ~%d tok | %d candidates dropped",
        len(result.triples),
        sum(1 for v in result.source.values() if v == "anchor"),
        sum(1 for v in result.source.values() if v == "expansion"),
        sum(1 for v in result.source.values() if v == "dense"),
        n_stance, tokens, result.dropped,
    )
    return result
