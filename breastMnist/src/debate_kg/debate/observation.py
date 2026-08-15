"""Stage A — free-stance observation of an image against a KG-derived schema.

One VLM call per sample, before any debate round. The model describes the image
using only the descriptor categories derived by `kg.kg_schema`, with no
diagnostic vocabulary and no stance-bearing triples in context. The resulting
observation is what conditions KG retrieval in Stage B, replacing the constant
seed query that gave all 780 samples the same 30 triples.

Nothing here is ontology-specific: categories and permitted values come from
schema.json, and the prompt is a Jinja2 template.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

UNCERTAIN = "uncertain"
_PROMPT_DIR = Path(__file__).parent / "prompts" / "breast"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Observation:
    """One sample's structured description of an image."""

    sample_id: str
    values: dict[str, str] = field(default_factory=dict)        # category -> value
    confidence: dict[str, float] = field(default_factory=dict)  # category -> 0..1
    notes: dict[str, str] = field(default_factory=dict)         # category -> free text
    global_note: str = ""
    attempts: int = 0
    failed_categories: list[str] = field(default_factory=list)

    def query_text(self) -> str:
        """Verbalise the observation for use as a retrieval query.

        Uncertain categories contribute nothing — an unassessable feature should
        not steer retrieval.
        """
        parts = [
            v.replace("_", " ")
            for c, v in self.values.items()
            if v != UNCERTAIN
        ]
        parts.extend(n for n in self.notes.values() if n)
        if self.global_note:
            parts.append(self.global_note)
        return " ".join(parts)

    def anchors(self, threshold: float) -> list[str]:
        """Observed entity ids whose confidence clears `threshold`."""
        return [
            v for c, v in self.values.items()
            if v != UNCERTAIN and self.confidence.get(c, 0.0) >= threshold
        ]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "values": self.values,
            "confidence": self.confidence,
            "notes": self.notes,
            "global_note": self.global_note,
            "attempts": self.attempts,
            "failed_categories": self.failed_categories,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Observation":
        return cls(
            sample_id=d["sample_id"],
            values=d.get("values", {}),
            confidence=d.get("confidence", {}),
            notes=d.get("notes", {}),
            global_note=d.get("global_note", ""),
            attempts=d.get("attempts", 0),
            failed_categories=d.get("failed_categories", []),
        )


def load_schema(path: Path) -> dict:
    """Load schema.json produced by debate_kg.kg.kg_schema."""
    schema = json.loads(path.read_text())
    if not schema.get("categories"):
        raise ValueError(f"{path} contains no categories — re-run kg_schema")
    return schema


def render_prompt(schema: dict) -> str:
    """Render the Stage A prompt from the schema."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(
        loader=FileSystemLoader(str(_PROMPT_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
    )
    return env.get_template("observation.j2").render(categories=schema["categories"])


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in (cleaned, (_JSON_RE.search(cleaned) or _JSON_RE.search(text) or None)):
        if candidate is None:
            continue
        raw = candidate if isinstance(candidate, str) else candidate.group(0)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_observation(
    text: str, schema: dict, sample_id: str
) -> tuple[Observation | None, list[str]]:
    """Parse and validate a model response.

    Returns (observation, errors). A non-None observation may still have
    categories recorded as `uncertain` where the model omitted them.
    """
    permitted = {
        c["category"]: {v["value"] for v in c["values"]} | {UNCERTAIN}
        for c in schema["categories"]
    }

    parsed = _extract_json(text)
    if parsed is None:
        return None, ["response contained no parseable JSON object"]

    rows = parsed.get("observations")
    if not isinstance(rows, list):
        return None, ["missing 'observations' list"]

    obs = Observation(sample_id=sample_id)
    errors: list[str] = []
    seen: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue
        cat = str(row.get("category", "")).strip()
        if cat not in permitted:
            errors.append(f"unknown category {cat!r}")
            continue
        val = str(row.get("value", "")).strip().lower().replace(" ", "_")
        if val not in permitted[cat]:
            errors.append(f"{cat}: value {val!r} not permitted")
            continue
        try:
            conf = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        obs.values[cat] = val
        obs.confidence[cat] = max(0.0, min(1.0, conf))
        obs.notes[cat] = str(row.get("note", "")).strip()
        seen.add(cat)

    missing = [c for c in permitted if c not in seen]
    if missing:
        errors.append(f"missing categories: {missing}")

    obs.global_note = str(parsed.get("global_note", "")).strip()
    return obs, errors


def observe(
    sample_id: str,
    image_b64: str,
    schema: dict,
    client,
    max_attempts: int = 3,
) -> Observation:
    """Run Stage A for one sample, retrying on parse failure.

    On repeated failure the unresolved categories are recorded as `uncertain`
    and logged. They are never silently defaulted to a real value — a default
    would reintroduce the constant-query problem this stage exists to fix.
    """
    prompt = render_prompt(schema)
    all_categories = [c["category"] for c in schema["categories"]]
    last: Observation | None = None

    for attempt in range(1, max_attempts + 1):
        # temperature=0: the observation should be a deterministic reading of
        # the image, not a sample from the model's descriptive distribution.
        text = client.complete(prompt=prompt, system="", temperature=0.0, image=image_b64)
        obs, errors = parse_observation(text, schema, sample_id)

        if obs is not None:
            obs.attempts = attempt
            if not errors:
                return obs
            last = obs
            logger.warning(
                "sample %s attempt %d/%d: %s", sample_id, attempt, max_attempts, "; ".join(errors)
            )
        else:
            logger.warning(
                "sample %s attempt %d/%d: %s", sample_id, attempt, max_attempts,
                "; ".join(errors) or "unparseable",
            )

    obs = last or Observation(sample_id=sample_id, attempts=max_attempts)
    for cat in all_categories:
        if cat not in obs.values:
            obs.values[cat] = UNCERTAIN
            obs.confidence[cat] = 0.0
            obs.notes[cat] = ""
            obs.failed_categories.append(cat)
    if obs.failed_categories:
        logger.error(
            "sample %s: recorded uncertain for %d category(ies) after %d attempts: %s",
            sample_id, len(obs.failed_categories), max_attempts, obs.failed_categories,
        )
    return obs


# ---------------------------------------------------------------------------
# Persistence and instrumentation
# ---------------------------------------------------------------------------

def save_observation(obs: Observation, out_dir: Path) -> Path:
    """Write one observation to out_dir/observation_{sample_id}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"observation_{obs.sample_id}.json"
    path.write_text(json.dumps(obs.to_dict(), indent=2, ensure_ascii=False))
    return path


def load_observation(sample_id: str, out_dir: Path) -> Observation | None:
    """Load a cached observation, or None."""
    path = out_dir / f"observation_{sample_id}.json"
    if not path.exists():
        return None
    return Observation.from_dict(json.loads(path.read_text()))


def summarise(observations: list[Observation], dominance_threshold: float = 0.85) -> dict:
    """Value distribution per category, flagging categories that never vary.

    A category whose modal value covers more than `dominance_threshold` of
    samples contributes nothing to per-sample retrieval variation and is
    flagged for the run report.
    """
    import collections

    n = len(observations)
    report: dict[str, dict] = {}
    degenerate: list[str] = []

    categories = sorted({c for o in observations for c in o.values})
    for cat in categories:
        counts = collections.Counter(o.values.get(cat, UNCERTAIN) for o in observations)
        top_value, top_n = counts.most_common(1)[0]
        share = top_n / n if n else 0.0
        report[cat] = {
            "distribution": dict(counts),
            "modal_value": top_value,
            "modal_share": round(share, 4),
            "n_uncertain": counts.get(UNCERTAIN, 0),
            "distinct_values": len(counts),
        }
        if share > dominance_threshold:
            degenerate.append(cat)

    return {
        "n_samples": n,
        "categories": report,
        "degenerate_categories": degenerate,
        "dominance_threshold": dominance_threshold,
    }
