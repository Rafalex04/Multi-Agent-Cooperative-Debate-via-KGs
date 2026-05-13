"""Wikidata helpers for fetching entity triples, with JSON disk cache.

Uses the Wikidata REST API (wbgetentities) rather than SPARQL so we are not
dependent on the query.wikidata.org SPARQL endpoint, which has historically
been unreliable under load.

Public API
----------
configure_cache(cache_dir)     — call once from construct.build_kg_pair
fetch_entity_triples(qid)      — returns List[Triple]; cached; network-marked in tests
entity_search(label)           — returns List[str] QIDs; cached; network-marked in tests

Design notes
------------
- Triples are built from Wikidata "mainsnak" statements only (no qualifiers).
- Property IDs (P-numbers) and entity-valued objects (Q-numbers) are resolved
  to English labels via a single batch wbgetentities call per entity.
- Cache is per-(entity_id, limit) JSON file; human-readable, easy to invalidate.
- entity_search cache key uses SHA-1(label) prefix to handle special characters safely.
- Both functions log WARNING and propagate on failure; callers decide how to handle.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests

from debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)

_API_URL = "https://www.wikidata.org/w/api.php"
_USER_AGENT = "debate_kg/0.1 (MSc thesis research; https://github.com)"

# Set by configure_cache(); None means no disk persistence.
_cache_dir: Path | None = None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def configure_cache(cache_dir: str | Path) -> None:
    """Enable persistent JSON disk cache. Call once per process from build_kg_pair."""
    global _cache_dir
    _cache_dir = Path(cache_dir).expanduser()
    _cache_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Wikidata cache dir: %s", _cache_dir)


def _triple_cache_path(entity_id: str, limit: int) -> Path | None:
    if _cache_dir is None:
        return None
    return _cache_dir / f"{entity_id}_limit{limit}.json"


def _search_cache_path(label: str, limit: int) -> Path | None:
    if _cache_dir is None:
        return None
    h = hashlib.sha1(label.encode()).hexdigest()[:16]
    return _cache_dir / f"search_{h}_limit{limit}.json"


def _load_triples(path: Path) -> list[Triple]:
    return [Triple(**t) for t in json.loads(path.read_text())]


def _save_triples(path: Path, triples: list[Triple]) -> None:
    path.write_text(json.dumps([t.model_dump() for t in triples]))


# ---------------------------------------------------------------------------
# REST API helpers
# ---------------------------------------------------------------------------

def _api_get(params: dict, timeout: int = 30, max_retries: int = 3) -> dict:
    """GET the Wikidata API with retry/back-off. Returns parsed JSON."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                _API_URL,
                params={**params, "format": "json"},
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(
                    "Wikidata API attempt %d/%d failed (%s), retrying in %ds",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
    raise RuntimeError(f"Wikidata API failed after {max_retries} attempts") from last_exc


def _batch_labels(ids: list[str]) -> dict[str, str]:
    """Return {id: english_label} for up to 50 QIDs/PIDs in one API call."""
    if not ids:
        return {}
    data = _api_get({
        "action": "wbgetentities",
        "ids": "|".join(ids[:50]),
        "props": "labels",
        "languages": "en",
    })
    result: dict[str, str] = {}
    for eid, edata in data.get("entities", {}).items():
        label = edata.get("labels", {}).get("en", {}).get("value")
        if label:
            result[eid] = label
    return result


def _snak_to_str(snak: dict) -> str | None:
    """Extract a string value from a Wikidata mainsnak.

    Returns a QID string for entity-valued snaks (caller resolves label),
    or a plain string for literals. Returns None for no-value/some-value snaks.
    """
    if snak.get("snaktype") != "value":
        return None
    dv = snak.get("datavalue", {})
    dtype = dv.get("type")
    val = dv.get("value")
    if dtype == "wikibase-entityid":
        return val.get("id")          # QID — resolved to label below
    if dtype == "string":
        return str(val)
    if dtype == "monolingualtext":
        return val.get("text")
    if dtype == "time":
        return val.get("time", "")[:11].lstrip("+")   # e.g. "1770-12-17"
    if dtype == "quantity":
        return str(val.get("amount", "")).lstrip("+")
    if dtype == "globe-coordinate":
        return f"{val.get('latitude')},{val.get('longitude')}"
    return None


# ---------------------------------------------------------------------------
# Triple fetch
# ---------------------------------------------------------------------------

def _fetch_triples_impl(entity_id: str, limit: int) -> list[Triple]:
    data = _api_get({
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels|claims",
        "languages": "en",
    })
    entity_data = data.get("entities", {}).get(entity_id, {})
    if not entity_data or entity_data.get("missing") == "":
        return []

    entity_label = entity_data.get("labels", {}).get("en", {}).get("value", entity_id)
    claims = entity_data.get("claims", {})

    # Collect raw (prop_id, raw_value) pairs up to limit
    raw_pairs: list[tuple[str, str]] = []
    entity_valued: set[str] = set()

    for prop_id, statements in claims.items():
        for stmt in statements:
            val = _snak_to_str(stmt.get("mainsnak", {}))
            if val is not None:
                raw_pairs.append((prop_id, val))
                if val.startswith("Q") or val.startswith("P"):
                    entity_valued.add(val)
            if len(raw_pairs) >= limit:
                break
        if len(raw_pairs) >= limit:
            break

    # Batch-resolve labels for all property IDs and entity-valued objects
    all_ids = list({p for p, _ in raw_pairs} | entity_valued)
    labels = _batch_labels(all_ids)

    triples: list[Triple] = []
    for prop_id, raw_val in raw_pairs:
        pred = labels.get(prop_id, prop_id)
        obj = labels.get(raw_val, raw_val) if raw_val in entity_valued else raw_val
        triples.append(Triple(subject=entity_label, predicate=pred, object=obj))

    logger.debug("Fetched %d triples for %s", len(triples), entity_id)
    return triples


def fetch_entity_triples(entity_id: str, limit: int = 100) -> list[Triple]:
    """Fetch direct-property triples for a Wikidata QID via REST API.

    Results are cached to disk if configure_cache() has been called.
    Raises RuntimeError after exhausting retries; caller must handle.

    Args:
        entity_id: Wikidata QID, e.g. "Q42".
        limit: Maximum triples to return.

    Returns:
        List of Triple objects. Subject is the English label of the entity;
        predicate and object are English labels where available, falling back
        to QID/PID strings.
    """
    path = _triple_cache_path(entity_id, limit)
    if path is not None and path.exists():
        logger.debug("Cache hit: %s", entity_id)
        return _load_triples(path)

    triples = _fetch_triples_impl(entity_id, limit)

    if path is not None:
        _save_triples(path, triples)

    return triples


# ---------------------------------------------------------------------------
# Entity search
# ---------------------------------------------------------------------------

def _search_impl(label: str, limit: int) -> list[str]:
    try:
        data = _api_get({
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "limit": limit,
            "type": "item",
        })
        return [item["id"] for item in data.get("search", [])]
    except Exception as exc:  # noqa: BLE001
        logger.warning("entity_search(%r) failed: %s", label, exc)
        return []


def entity_search(label: str, limit: int = 3) -> list[str]:
    """Return Wikidata QIDs whose label best matches label.

    Results are cached to disk if configure_cache() has been called.
    Returns [] on any network failure (logged as WARNING).

    Args:
        label: Entity surface form, e.g. "Barack Obama".
        limit: Maximum QIDs to return.

    Returns:
        List of QID strings in descending relevance order.
    """
    path = _search_cache_path(label, limit)
    if path is not None and path.exists():
        logger.debug("Search cache hit: %r", label)
        return json.loads(path.read_text())

    qids = _search_impl(label, limit)

    if path is not None:
        path.write_text(json.dumps(qids))

    return qids
