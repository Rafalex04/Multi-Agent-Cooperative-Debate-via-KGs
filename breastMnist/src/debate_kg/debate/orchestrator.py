"""Debate orchestrator: cooperative turn-taking between two KG-grounded experts.

Module 3 in the pipeline. Prompts live in debate/prompts/ as Jinja2 templates —
never inline prompt strings here.
See SPEC §Module3.

Round structure
---------------
Each call to run_debate executes exactly one round:
  1. First expert opens — grounded in their triples, sees all prior history.
  2. Second expert responds — sees prior history + first expert's new claims.
  3. Edges are derived by parsing [ADDRESSED:cN][AGREE|DISAGREE] tags in the
     structured response; no separate edge-classifier LLM call is made.
  Opening expert alternates: A opens on even rounds, B opens on odd rounds.

Node model
----------
Each DebateNode IS one factual claim. A single LLM call may produce N nodes
(one per [CLAIM] block). Short IDs (c1, c2, …) are globally sequential across
all rounds of the debate.

Provenance
----------
Inline [N] citations in each claim's text are mapped to Triple UUIDs for
DebateNode.provenance. If a claim contains no [N] citations, provenance=[] is
stored and a WARNING is logged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from breastMnist.src.debate_kg.kg.schema import DebateAddress, DebateEdge, DebateNode, Triple
from breastMnist.src.debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render(name: str, **kwargs: object) -> str:
    return _JINJA_ENV.get_template(name).render(**kwargs)


def _make_client(cfg: DictConfig) -> LLMClient:
    if getattr(cfg.run, "use_stub_data", False):
        from breastMnist.src.debate_kg.models.stub_client import StubClient
        return StubClient(model="stub", base_url="")
    from breastMnist.src.debate_kg.models.ollama_client import OllamaClient
    base_url: str = getattr(cfg.model, "ollama_base_url", "http://localhost:11434")
    return OllamaClient(model=cfg.model.expert_model, base_url=base_url)


def _parse_provenance(text: str, uuid_map: dict[int, str]) -> list[str]:
    """Extract [N] citation indices from text and map to Triple UUIDs.

    Deduplicates while preserving first-mention order.
    Returns [] if no valid [N] references are found.
    """
    seen: set[str] = set()
    result: list[str] = []
    for idx_str in re.findall(r'\[(\d+)\]', text):
        uid = uuid_map.get(int(idx_str))
        if uid and uid not in seen:
            seen.add(uid)
            result.append(uid)
    return result


@dataclass
class _ParsedClaim:
    text: str
    addressed: list[tuple[str, str]] = field(default_factory=list)  # (short_id, stance)
    label: str | None = None


_CLAIM_SPLIT_RE = re.compile(r'\[CLAIM\]')
_ADDR_RE = re.compile(r'\[ADDRESSED:([^\]]+)\]\[(AGREE|DISAGREE)\]')
_LABEL_RE = re.compile(r'\[LABEL:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)\]')


def _parse_structured_response(text: str) -> list[_ParsedClaim]:
    """Split an expert response on [CLAIM] boundaries and extract structured fields.

    Strict uppercase matching only. Returns empty list when no [CLAIM] tags are found.
    """
    parts = _CLAIM_SPLIT_RE.split(text)
    result = []
    for part in parts[1:]:  # skip any preamble before the first [CLAIM]
        claim_text = re.split(r'\[(?:ADDRESSED:|LABEL:)', part, maxsplit=1)[0].strip()
        if len(claim_text.split()) < 2:  # skip artefacts like "blocks:" or "here:"
            continue
        addressed = [
            (m.group(1).strip(), m.group(2))
            for m in _ADDR_RE.finditer(part)
        ]
        label_m = _LABEL_RE.search(part)
        label = label_m.group(1) if label_m else None
        result.append(_ParsedClaim(text=claim_text, addressed=addressed, label=label))
    return result


def _expert_turn_multi(
    expert_id: str,
    round_idx: int,
    triples: list[Triple],
    history: list[DebateNode],
    system: str,
    client: LLMClient,
    cfg: DictConfig,
) -> list[DebateNode]:
    """Generate one expert turn and return one DebateNode per [CLAIM] block.

    Short IDs are assigned as c{len(history)+1}, c{len(history)+2}, … so they
    are globally unique across the whole debate when history grows monotonically.
    """
    uuid_map: dict[int, str] = {i + 1: t.uuid for i, t in enumerate(triples)}
    short_id_map: dict[str, str] = {n.short_id: n.id for n in history}

    prompt = _render("expert_turn.j2", triples=triples, history=history)
    logger.debug("%s round %d — user prompt:\n%s", expert_id, round_idx, prompt)

    text = client.complete(prompt, system=system, temperature=cfg.model.expert_temperature)
    logger.debug("%s round %d — raw LLM response:\n%s", expert_id, round_idx, text)

    parsed = _parse_structured_response(text)
    for i, pc in enumerate(parsed):
        logger.debug(
            "%s round %d parsed claim %d/%d: label=%s addressed=%s text=%r",
            expert_id, round_idx, i + 1, len(parsed), pc.label, pc.addressed, pc.text,
        )

    if not parsed:
        logger.warning(
            "%s round %d: no [CLAIM] blocks found in response — creating fallback node",
            expert_id, round_idx,
        )
        fallback_prov = _parse_provenance(text, uuid_map)
        return [DebateNode(
            short_id=f"c{len(history) + 1}",
            expert_id=expert_id,
            round_idx=round_idx,
            text=text,
            provenance=fallback_prov,
        )]

    nodes: list[DebateNode] = []
    for i, pc in enumerate(parsed):
        short_id = f"c{len(history) + i + 1}"
        addressed = [
            DebateAddress(node_id=short_id_map[sid], stance=stance)
            for sid, stance in pc.addressed
            if sid in short_id_map
        ]
        prov = _parse_provenance(pc.text, uuid_map)
        if not prov and triples:
            logger.warning(
                "%s round %d claim %s: no [N] citations — storing provenance=[]",
                expert_id, round_idx, short_id,
            )
        nodes.append(DebateNode(
            short_id=short_id,
            expert_id=expert_id,
            round_idx=round_idx,
            text=pc.text,
            label=pc.label,
            addressed=addressed,
            provenance=prov,
        ))
    logger.info(
        "%s round %d: %d claim(s), %d citation(s) total",
        expert_id, round_idx, len(nodes), sum(len(n.provenance) for n in nodes),
    )
    return nodes


def _build_edges(
    new_nodes: list[DebateNode],
    all_context_ids: set[str],
) -> list[DebateEdge]:
    """Build edges from [ADDRESSED] entries across all new nodes.

    Edge direction: source=responding claim → target=addressed claim.
    Deduplicates (source, target) pairs by preferring "-" over "+".
    """
    seen: dict[tuple[str, str], str] = {}
    for node in new_nodes:
        for addr in node.addressed:
            if addr.node_id in all_context_ids:
                sign = "+" if addr.stance == "AGREE" else "-"
                pair = (node.id, addr.node_id)
                if pair not in seen or seen[pair] == "+":
                    seen[pair] = sign
    return [DebateEdge(source_id=src, target_id=tgt, sign=sg) for (src, tgt), sg in seen.items()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_expert(
    claim: str,
    triples_a: list[Triple],
    cfg: DictConfig,
) -> tuple[list[DebateNode], list[DebateEdge]]:
    """Run one expert turn with no opponent — used for the single_expert baseline.

    Returns one or more DebateNodes (one per [CLAIM] block) and an empty edge list.
    """
    client = _make_client(cfg)
    system_prompt = _render("system.j2", claim=claim)
    nodes = _expert_turn_multi(
        expert_id="expert_a",
        round_idx=0,
        triples=triples_a,
        history=[],
        system=system_prompt,
        client=client,
        cfg=cfg,
    )
    logger.info("Single-expert turn: expert_a produced %d claim(s)", len(nodes))
    return nodes, []


def run_debate(
    claim: str,
    triples_a: list[Triple],
    triples_b: list[Triple],
    cfg: DictConfig,
    round_idx: int = 0,
    history: list[DebateNode] | None = None,
) -> tuple[list[DebateNode], list[DebateEdge]]:
    """Run one round of cooperative debate between expert A and expert B.

    Args:
        claim: The FEVER claim text both experts are reasoning about.
        triples_a: Triples retrieved for expert A this round.
        triples_b: Triples retrieved for expert B this round.
        cfg: Hydra config. Reads cfg.model.expert_model,
             cfg.model.expert_temperature, and (optionally)
             cfg.model.ollama_base_url.
        round_idx: Global round counter passed from the main loop.
        history: All DebateNodes accumulated across all prior rounds, so experts
                 can address any previous claim by its short ID.

    Returns:
        (nodes, edges) — all new claim nodes and edges derived from [ADDRESSED]
        tags. Edge direction: responding claim → addressed claim.
    """
    history = list(history or [])
    client = _make_client(cfg)
    system_prompt = _render("system.j2", claim=claim)

    # Alternate which expert opens each round.
    if round_idx % 2 == 0:
        first_id, second_id = "expert_a", "expert_b"
        first_triples, second_triples = triples_a, triples_b
    else:
        first_id, second_id = "expert_b", "expert_a"
        first_triples, second_triples = triples_b, triples_a

    # First expert sees all prior history.
    nodes_first = _expert_turn_multi(
        expert_id=first_id,
        round_idx=round_idx,
        triples=first_triples,
        history=history,
        system=system_prompt,
        client=client,
        cfg=cfg,
    )

    # Second expert sees prior history + first expert's new claims.
    nodes_second = _expert_turn_multi(
        expert_id=second_id,
        round_idx=round_idx,
        triples=second_triples,
        history=history + nodes_first,
        system=system_prompt,
        client=client,
        cfg=cfg,
    )

    all_new = nodes_first + nodes_second
    all_ctx_ids = {n.id for n in history + all_new}
    edges = _build_edges(all_new, all_ctx_ids)

    logger.info(
        "Debate round %d: %s produced %d claim(s), %s produced %d claim(s), %d edge(s)",
        round_idx,
        first_id, len(nodes_first),
        second_id, len(nodes_second),
        len(edges),
    )
    return all_new, edges
