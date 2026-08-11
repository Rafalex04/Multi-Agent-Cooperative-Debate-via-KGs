"""Build a disjoint pair of KGs from Wikidata, scoped to one FEVER claim.

Module 1 in the pipeline. Offline preprocessing — not called inside the round loop.
See SPEC §Module1 and the Module 1 implementation plan in plans/.

Pipeline inside build_kg_pair
------------------------------
1. Extract candidate entity mentions via capitalized-phrase regex.
2. Link each mention → Wikidata QID via wbsearchentities (top-1 per mention).
3. Fetch direct-property triples per QID via SPARQL (cached).
4. Apply global triple cap (hub-entity guard).
5. Randomly split triples 50/50 with optional overlap injection.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from breastMnist.src.debate_kg.kg import wikidata
from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)

# Words that appear title-cased at sentence start but are not named entities.
_STOPWORDS: frozenset[str] = frozenset({
    "A", "An", "The", "This", "That", "These", "Those",
    "He", "She", "It", "They", "We", "You", "I", "Me", "Him", "Her", "Us", "Them",
    "His", "Its", "Their", "Our", "Your", "My",
    "Who", "What", "Where", "When", "Why", "How",
    "Is", "Are", "Was", "Were", "Be", "Been", "Being",
    "Have", "Has", "Had", "Do", "Does", "Did",
    "Will", "Would", "Could", "Should", "May", "Might", "Must", "Shall", "Can",
    "Not", "No", "And", "Or", "But", "If", "So", "As", "At", "By", "For",
    "In", "Of", "On", "To", "Up", "With", "From", "Into", "Through",
    "Also", "Both", "Only", "Even", "Still", "Yet", "Then", "Than",
    "After", "Before", "During", "Between", "Among", "Against",
    "Although", "Because", "Since", "Unless", "While", "Though",
    "However", "Despite", "Throughout", "Including", "Towards", "Upon",
})

# Regex: one or more title-cased words.  Handles "Barack Obama", "United States", etc.
_MENTION_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_mentions(text: str) -> list[str]:
    """Extract candidate named-entity spans from claim text via regex.

    Returns deduplicated spans, filtering out stopwords and very short tokens.
    """
    seen: set[str] = set()
    result: list[str] = []
    for span in _MENTION_RE.findall(text):
        # Keep multi-word phrases unconditionally; single words must pass stopword filter.
        if " " not in span and span in _STOPWORDS:
            continue
        if span not in seen:
            seen.add(span)
            result.append(span)
    return result


def _link_mentions(mentions: list[str], per_mention_limit: int) -> list[str]:
    """Map entity mention strings to Wikidata QIDs (top-1 per mention, deduped).

    Args:
        mentions: Surface-form entity spans extracted from the claim.
        per_mention_limit: How many Wikidata search results to request per mention.

    Returns:
        Deduplicated list of QID strings in mention order.
    """
    seen_qids: set[str] = set()
    qids: list[str] = []
    for mention in mentions:
        results = wikidata.entity_search(mention, limit=per_mention_limit)
        if results:
            qid = results[0]
            if qid not in seen_qids:
                seen_qids.add(qid)
                qids.append(qid)
                logger.debug("Linked %r → %s", mention, qid)
        else:
            logger.debug("No QID found for %r", mention)
    return qids


def _cap_triples(triples: list[Triple], max_count: int) -> list[Triple]:
    """Truncate to max_count, keeping whole entities (by subject) where possible."""
    by_subject: dict[str, list[Triple]] = {}
    for t in triples:
        by_subject.setdefault(t.subject, []).append(t)

    result: list[Triple] = []
    for subj_triples in by_subject.values():
        remaining = max_count - len(result)
        if remaining <= 0:
            break
        result.extend(subj_triples[:remaining])

    return result


def _split_triples(
    triples: list[Triple],
    kg_split_overlap: float,
    rng: np.random.Generator,
) -> tuple[list[Triple], list[Triple]]:
    """Randomly split triples into two halves with optional overlap injection.

    Args:
        triples: All triples to split (modified in-place order is irrelevant).
        kg_split_overlap: Fraction of each base half to add to the other side.
                          0.0 → strictly disjoint.
        rng: Seeded NumPy generator for reproducibility.

    Returns:
        (triples_a, triples_b) — each at least as large as len(triples) // 2;
        with overlap they may be larger.
    """
    # Stable pre-shuffle order so the same seed always produces the same split.
    sorted_triples = sorted(triples, key=lambda t: t.uuid)

    # Shuffle deterministically.
    idx = rng.permutation(len(sorted_triples))
    shuffled = [sorted_triples[int(i)] for i in idx]

    mid = (len(shuffled) + 1) // 2  # A gets the extra triple when odd
    base_a = shuffled[:mid]
    base_b = shuffled[mid:]

    triples_a: list[Triple] = list(base_a)
    triples_b: list[Triple] = list(base_b)

    if kg_split_overlap > 0:
        if base_b:
            n = max(1, int(kg_split_overlap * len(base_b)))
            n = min(n, len(base_b))
            chosen = rng.choice(len(base_b), size=n, replace=False)
            triples_a.extend(base_b[int(i)] for i in chosen)
        if base_a:
            n = max(1, int(kg_split_overlap * len(base_a)))
            n = min(n, len(base_a))
            chosen = rng.choice(len(base_a), size=n, replace=False)
            triples_b.extend(base_a[int(i)] for i in chosen)

    return triples_a, triples_b


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_kg_pair(claim: str, cfg: DictConfig) -> tuple[KnowledgeGraph, KnowledgeGraph]:
    """Build two KG subgraphs for the given claim via Wikidata.

    Both KGs share the same schema (Triple objects with stable UUIDs) but
    contain different random subsets of the claim-relevant triples.

    Args:
        claim: Raw FEVER claim text.
        cfg: Hydra config. Reads:
            cfg.data.wikidata_cache_dir, cfg.data.max_entity_mentions,
            cfg.data.max_triples_per_entity, cfg.data.max_triples_per_kg,
            cfg.data.kg_split_overlap, cfg.data.seed.

    Returns:
        (kg_a, kg_b) — randomly split subgraphs. Both are empty if no Wikidata
        entities are found (logged as WARNING).
    """
    cache_dir = Path(str(cfg.data.wikidata_cache_dir)).expanduser()
    wikidata.configure_cache(cache_dir)

    # Step 1 — entity mention extraction.
    mentions = _extract_mentions(claim)[: cfg.data.max_entity_mentions]
    logger.info("Extracted %d mention(s) from claim: %s", len(mentions), mentions)

    # Step 2 — Wikidata entity linking.
    qids = _link_mentions(mentions, per_mention_limit=1)
    logger.info("Linked %d QID(s): %s", len(qids), qids)

    if not qids:
        logger.warning("No Wikidata QIDs resolved for claim: %r — returning empty KGs", claim)
        return KnowledgeGraph(), KnowledgeGraph()

    # Step 3 — triple fetch.
    all_triples: list[Triple] = []
    for qid in qids:
        try:
            triples = wikidata.fetch_entity_triples(qid, limit=cfg.data.max_triples_per_entity)
        except RuntimeError as exc:
            logger.warning("Skipping QID %s: fetch failed (%s)", qid, exc)
            continue
        logger.info("Fetched %d triple(s) for %s", len(triples), qid)
        all_triples.extend(triples)

    # Step 4 — global cap (hub-entity guard).
    if len(all_triples) > cfg.data.max_triples_per_kg:
        logger.info(
            "Capping %d → %d triples (max_triples_per_kg)",
            len(all_triples), cfg.data.max_triples_per_kg,
        )
        all_triples = _cap_triples(all_triples, cfg.data.max_triples_per_kg)

    if not all_triples:
        logger.warning("No triples fetched for claim: %r — returning empty KGs", claim)
        return KnowledgeGraph(), KnowledgeGraph()

    logger.info("Total triples before split: %d", len(all_triples))

    # Step 5 — random split with optional overlap.
    rng = np.random.default_rng(cfg.data.seed)
    triples_a, triples_b = _split_triples(all_triples, cfg.data.kg_split_overlap, rng)

    kg_a = KnowledgeGraph()
    for t in triples_a:
        kg_a.add_triple(t)

    kg_b = KnowledgeGraph()
    for t in triples_b:
        kg_b.add_triple(t)

    logger.info(
        "Split complete: KG-A=%d triple(s), KG-B=%d triple(s) (overlap=%.2f)",
        len(kg_a), len(kg_b), cfg.data.kg_split_overlap,
    )
    return kg_a, kg_b
