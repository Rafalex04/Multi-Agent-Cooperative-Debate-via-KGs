"""Derive an observation schema from a knowledge graph (offline, run once).

Stage A of the observation-conditioned retrieval pipeline needs to know which
descriptor categories exist and what values each may take — without any
hardcoded ontology vocabulary. This module reads those facts out of the KG
itself, driven entirely by relation names in conf/kg_schema.yaml.

Derivation
----------
1. Observation categories are the SUBJECTS of `observation_category_relations`
   (in BI-RADS: `shape_descriptor category_descriptor mass_finding`). Using the
   KG's own marker keeps diagnosis groupings such as `benign_breast_lesion` out
   of the observation schema — Stage A must not see diagnostic language.
2. Values of a category are the SUBJECTS of `category_relations` triples whose
   OBJECT is that category (`oval is_a shape_descriptor`).
3. Each value carries its definition text, when one exists, for use as
   few-shot grounding in the Stage A prompt.
4. Triples using a `stance_relations` relation are collected as stance-bearing
   so callers can exclude them from Stage A.

Usage
-----
  python -m debate_kg.kg.kg_schema \\
      --kg-path data/breast/knowledge_graph.json \\
      --definitions-path data/breast/definitions.json \\
      --config conf/kg_schema.yaml \\
      --out data/breast/schema.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    """Load the ontology relation-name config."""
    import yaml

    cfg = yaml.safe_load(path.read_text())
    required = ["observation_category_relations", "category_relations", "stance_relations"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return cfg


def _norm(name: str) -> str:
    """Normalise a relation name for case-insensitive matching."""
    return name.strip().lower()


def _pretty(entity: str) -> str:
    """Human-readable form of a KG entity id."""
    return entity.replace("_", " ")


def derive_schema(
    triples: list[dict],
    definitions: dict[str, tuple[str, str]] | None,
    config: dict,
) -> dict:
    """Return the observation schema derived from `triples`.

    Args:
        triples: KG records with keys id/subject/relation/object.
        definitions: entity -> (def_id, text), or None.
        config: parsed conf/kg_schema.yaml.

    Returns:
        dict with keys `categories` and `stance_triple_ids`.
    """
    obs_rels = {_norm(r) for r in config["observation_category_relations"]}
    cat_rels = {_norm(r) for r in config["category_relations"]}
    stance_rels = {_norm(r) for r in config["stance_relations"]}
    min_values = int(config.get("min_values_per_category", 2))

    # 1. Observation categories: subjects of an observation_category_relation.
    category_names: dict[str, str] = {}  # category -> meta-category
    for t in triples:
        if _norm(t["relation"]) in obs_rels:
            category_names[t["subject"]] = t["object"]

    # 2. Values per category: subjects of a category_relation pointing at it.
    values: dict[str, list[str]] = {c: [] for c in category_names}
    for t in triples:
        if _norm(t["relation"]) in cat_rels and t["object"] in values:
            if t["subject"] not in values[t["object"]]:
                values[t["object"]].append(t["subject"])

    # 3. Assemble, attaching definitions.
    defs = definitions or {}
    categories = []
    for cat, meta in sorted(category_names.items()):
        vals = values.get(cat, [])
        if len(vals) < min_values:
            logger.warning(
                "category %s has %d value(s) (< %d) — dropped", cat, len(vals), min_values
            )
            continue
        categories.append({
            "category": cat,
            "display": _pretty(cat),
            "meta_category": meta,
            "values": [
                {
                    "value": v,
                    "display": _pretty(v),
                    "definition": (defs.get(v) or ("", ""))[1],
                }
                for v in sorted(vals)
            ],
        })

    # 4. Stance-bearing triples, excluded from Stage A.
    stance_ids = [t["id"] for t in triples if _norm(t["relation"]) in stance_rels]

    return {"categories": categories, "stance_triple_ids": stance_ids}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Derive an observation schema from a KG")
    parser.add_argument("--kg-path", default="data/breast/knowledge_graph.json")
    parser.add_argument("--definitions-path", default="data/breast/definitions.json")
    parser.add_argument("--config", default="conf/kg_schema.yaml")
    parser.add_argument("--out", default="data/breast/schema.json")
    args = parser.parse_args(argv)

    config = _load_config(Path(args.config))

    raw = json.loads(Path(args.kg_path).read_text())
    triples = raw["triples"] if isinstance(raw, dict) else raw
    logger.info("Loaded %d triples from %s", len(triples), args.kg_path)

    definitions = None
    defs_path = Path(args.definitions_path)
    if defs_path.exists():
        from debate_kg.kg.loader import load_definitions

        definitions = load_definitions(defs_path)

    schema = derive_schema(triples, definitions, config)
    categories = schema["categories"]

    min_categories = int(config.get("min_categories", 3))
    if len(categories) < min_categories:
        raise SystemExit(
            f"Derived only {len(categories)} categories (need >= {min_categories}). "
            f"The relation names in {args.config} probably do not match this KG. "
            f"Relations present: {sorted({t['relation'] for t in triples})[:12]} ..."
        )

    n_values = sum(len(c["values"]) for c in categories)
    n_defined = sum(1 for c in categories for v in c["values"] if v["definition"])
    logger.info(
        "Derived %d categories / %d values (%d with definitions) | %d stance-bearing triples",
        len(categories), n_values, n_defined, len(schema["stance_triple_ids"]),
    )
    for c in categories:
        logger.info(
            "  %-30s (%s) %d values: %s",
            c["category"], c["meta_category"], len(c["values"]),
            ", ".join(v["value"] for v in c["values"]),
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", out.resolve())


if __name__ == "__main__":
    main()
