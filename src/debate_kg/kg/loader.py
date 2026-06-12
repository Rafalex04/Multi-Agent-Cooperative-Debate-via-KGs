"""Load a pre-built domain KG from JSON and serialize it for prompt injection.

The JSON schema expected is:
    {
      "triples": [
        {"id": "t_001", "subject": "...", "relation": "...", "object": "...",
         "notes": "...",         # optional
         "specificity": "...",  # optional
         "confidence": "..."}   # optional
      ]
    }

The definitions file schema is:
    {
      "entities": [
        {"term": "...", "definition": "..."}
      ]
    }

Field mapping to Triple:
    id       → uuid
    relation → predicate
    (all other fields are discarded — only subject/predicate/object are stored)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import Triple

logger = logging.getLogger(__name__)


def load_kg_from_json(kg_path: str | Path) -> KnowledgeGraph:
    """Load a pre-built KG from a JSON file.

    Args:
        kg_path: Path to the JSON file containing {"triples": [...]}.

    Returns:
        Populated KnowledgeGraph instance with stable UUIDs from the ``id`` field.
    """
    path = Path(kg_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    triples_raw = data["triples"]

    kg = KnowledgeGraph()
    for raw in triples_raw:
        triple = Triple(
            uuid=raw["id"],
            subject=raw["subject"],
            predicate=raw["relation"],
            object=raw["object"],
        )
        kg.add_triple(triple)

    logger.info("Loaded KG from %s: %d triples", path, len(kg))
    return kg


def load_definitions(definitions_path: str | Path) -> dict[str, tuple[str, str]]:
    """Load the entity definitions file.

    Args:
        definitions_path: Path to the JSON file containing {"entities": [...]}.
            Each entity must have "term", "definition", and "id" fields.

    Returns:
        Dict mapping term → (def_id, definition_text), e.g.
        {"homogeneous_background_echotexture_fat": ("d_001", "Fat lobules ...")}
    """
    path = Path(definitions_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {
        entry["term"]: (entry["id"], entry["definition"])
        for entry in data["entities"]
    }
    logger.info("Loaded %d entity definitions from %s", len(result), path)
    return result


def serialize_kg_for_prompt(
    kg: KnowledgeGraph,
    definitions: dict[str, tuple[str, str]],
    include_definitions: bool = False,
) -> str:
    """Serialize the full KG into a compact text block for prompt injection.

    Format:
        KNOWLEDGE GRAPH (ACR BI-RADS Ultrasound Lexicon)
        ------------------------------------------------
        [t_001] homogeneous background echotexture fat  IS A  tissue pattern

    Experts can cite triples with [CITED:t_NNN] and definitions with [CITED:d_NNN].

    Args:
        kg: The domain KG to serialize.
        definitions: Term → (def_id, definition_text) from load_definitions().
        include_definitions: If True, append Def lines under each triple (~25K tokens).
            Defaults to False (triples-only, ~6.5K tokens) to stay within model context.

    Returns:
        Multi-line string ready to be embedded in a prompt.
    """
    lines: list[str] = [
        "KNOWLEDGE GRAPH (ACR BI-RADS Ultrasound Lexicon)",
        "-" * 50,
    ]
    for triple in kg.all_triples():
        subj = triple.subject.replace("_", " ")
        pred = triple.predicate.replace("_", " ").upper()
        obj = triple.object.replace("_", " ")
        lines.append(f"[{triple.uuid}] {subj}  {pred}  {obj}")
        if include_definitions:
            defn_entry = definitions.get(triple.subject)
            if defn_entry:
                def_id, def_text = defn_entry
                lines.append(f"  Def [{def_id}]: {def_text}")

    return "\n".join(lines)
