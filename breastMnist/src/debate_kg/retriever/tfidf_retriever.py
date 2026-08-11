"""TF-IDF cosine retriever: maps a free-text claim to the most relevant KG triples.

Replaces LLM self-citation ([CITED:t_NNN]) with automatic relevance scoring.
Built once at startup over the full KG; cheap to query at inference time.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from breastMnist.src.debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)


def _triple_text(t: Triple) -> str:
    return f"{t.subject.replace('_', ' ')} {t.predicate.replace('_', ' ')} {t.object.replace('_', ' ')}"


class TFIDFRetriever:
    """Sparse TF-IDF cosine retriever over a fixed set of KG triples.

    Args:
        triples: All triples in the domain KG. Built once, reused every turn.
    """

    def __init__(self, triples: list[Triple]) -> None:
        self._triples = triples
        self._texts = [_triple_text(t) for t in triples]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(self._texts)
        logger.info("TFIDFRetriever: indexed %d triples", len(triples))

    def retrieve(self, query: str, k: int = 5) -> list[Triple]:
        """Return the top-k most relevant triples for a free-text query.

        Args:
            query: Claim text produced by an expert.
            k: Number of triples to return.

        Returns:
            Up to k Triple objects, ordered by descending cosine similarity.
        """
        if not self._triples:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        top_k = int(min(k, len(self._triples)))
        indices = np.argsort(sims)[::-1][:top_k]
        result = [self._triples[int(i)] for i in indices]
        logger.debug(
            "TFIDFRetriever: query=%r → top-%d triples: %s",
            query[:60], k, [t.uuid for t in result],
        )
        return result
