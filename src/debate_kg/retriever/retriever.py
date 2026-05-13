"""GraphRAG retriever: entity-link the debate context, k-hop expand, return triples.

Module 2 in the pipeline. Called every turn inside the round loop.
See SPEC §Module2.

Algorithm
---------
1. Encode the context string with a sentence-transformer.
2. Find the top-K KG entities most similar to the context (cosine sim).
3. k-hop BFS expand from each anchor entity; union the resulting subgraphs.
4. If no anchor yields any triples (vocabulary mismatch), fall back to all KG triples.
5. Re-rank candidates by cosine similarity to the context; return top retriever_max_triples.
"""
from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.debug("Loading sentence-transformer model: %s", model_name)
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def _encode(texts: list[str], model_name: str) -> np.ndarray:
    return _get_model(model_name).encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    )


def _triple_text(triple: Triple) -> str:
    return f"{triple.subject} | {triple.predicate} | {triple.object}"


def _cosine_sim(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and each row of matrix.

    Args:
        query: shape (D,)
        matrix: shape (N, D)

    Returns:
        shape (N,) — similarity in [-1, 1]; higher is more similar.
    """
    q = query / (np.linalg.norm(query) + 1e-8)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    m = matrix / norms
    return m @ q


def retrieve(kg: KnowledgeGraph, context: str, cfg: DictConfig) -> list[Triple]:
    """Retrieve relevant triples from kg given the current debate context.

    Args:
        kg: The expert's current knowledge graph.
        context: Current debate context (claim text + debate history so far).
                 The caller controls what to include; the retriever treats it as
                 a plain string and encodes it as-is.
        cfg: Hydra config. Reads:
            cfg.run.retriever_k_hops, cfg.run.retriever_max_triples,
            cfg.model.retriever_model, cfg.data.max_entity_mentions.

    Returns:
        At most cfg.run.retriever_max_triples triples, ranked by cosine
        similarity to context. Empty list if kg is empty.
    """
    if len(kg) == 0:
        logger.debug("retrieve: KG is empty, returning []")
        return []

    model_name: str = cfg.model.retriever_model
    k_hops: int = cfg.run.retriever_k_hops
    max_triples: int = cfg.run.retriever_max_triples
    max_anchors: int = cfg.data.max_entity_mentions

    # --- Step 1: encode context ---
    query_emb: np.ndarray = _encode([context], model_name)[0]  # (D,)

    # --- Step 2: find entity anchors ---
    all_triples = kg.all_triples()
    entity_labels: list[str] = sorted(
        {t.subject for t in all_triples} | {t.object for t in all_triples}
    )
    entity_embs: np.ndarray = _encode(entity_labels, model_name)  # (N, D)
    entity_sims: np.ndarray = _cosine_sim(query_emb, entity_embs)  # (N,)

    k = min(max_anchors, len(entity_labels))
    top_k_idx = np.argsort(entity_sims)[::-1][:k]
    anchors = [entity_labels[int(i)] for i in top_k_idx]
    logger.debug("retrieve: top anchors = %s", anchors)

    # --- Step 3: k-hop BFS from each anchor ---
    seen_uuids: set[str] = set()
    candidates: list[Triple] = []
    for anchor in anchors:
        sub = kg.subgraph_around(anchor, k_hops)
        for t in sub.all_triples():
            if t.uuid not in seen_uuids:
                seen_uuids.add(t.uuid)
                candidates.append(t)

    # --- Step 4: fallback to all triples if anchors produced nothing ---
    if not candidates:
        logger.debug("retrieve: no anchor subgraph — falling back to full-KG scan")
        candidates = all_triples

    # --- Step 5: re-rank candidates by cosine similarity ---
    triple_texts = [_triple_text(t) for t in candidates]
    triple_embs: np.ndarray = _encode(triple_texts, model_name)  # (M, D)
    triple_sims: np.ndarray = _cosine_sim(query_emb, triple_embs)  # (M,)

    order = np.argsort(triple_sims)[::-1]
    ranked = [candidates[int(i)] for i in order]

    result = ranked[:max_triples]
    logger.debug(
        "retrieve: %d candidates → %d returned (max=%d)",
        len(candidates), len(result), max_triples,
    )
    return result
