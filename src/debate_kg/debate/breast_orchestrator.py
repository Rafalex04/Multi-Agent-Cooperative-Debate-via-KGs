"""Debate orchestrator for the BreastMNIST VLM ablation pipeline.

Six ablation modes, all operating on a base64-encoded breast ultrasound image:

  single_agent          — one Qwen2.5-VL call, returns label + confidence.
  debate_opinion        — free-text multi-round debate; MedGemma picks the winner.
  debate_graph          — structured [CLAIM] debate; MedGemma scores each node.
  debate_kg             — debate_graph + full domain KG injected into every prompt.
  debate_kg_adversarial — debate_kg with agents locked to opposite stances.
  adaptive_adversarial  — single_agent first; escalates to debate_kg_adversarial
                          only when confidence < cfg.run.confidence_threshold.

Node / edge model is identical to the FEVER pipeline: one DebateNode per [CLAIM]
block, edges from [ADDRESSED:cN][AGREE|DISAGREE] tags. Labels are BENIGN / MALIGNANT.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import (
    DebateAddress,
    DebateEdge,
    DebateNode,
    SingleAgentResult,
    WinnerJudgment,
)
from debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts" / "breast"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


# ---------------------------------------------------------------------------
# Internal helpers (shared with original orchestrator pattern)
# ---------------------------------------------------------------------------

def _render(name: str, **kwargs: object) -> str:
    return _JINJA_ENV.get_template(name).render(**kwargs)


def _make_expert_client(cfg: DictConfig) -> LLMClient:
    if getattr(cfg.run, "use_stub_data", False):
        from debate_kg.models.stub_client import StubClient
        return StubClient(model="stub", base_url="")
    from debate_kg.models.ollama_client import OllamaClient
    base_url: str = getattr(cfg.model, "ollama_base_url", "http://localhost:11434")
    return OllamaClient(model=cfg.model.expert_model, base_url=base_url)


@dataclass
class _ParsedClaim:
    text: str
    addressed: list[tuple[str, str]] = field(default_factory=list)
    label: str | None = None
    provenance: list[str] = field(default_factory=list)


_CLAIM_SPLIT_RE = re.compile(r'\[CLAIM\]')
_ADDR_RE = re.compile(r'\[ADDRESSED:([^\]]+)\]\[(AGREE|DISAGREE)\]')
_LABEL_RE = re.compile(r'\[LABEL:\s*(BENIGN|MALIGNANT)\]')
_CITED_RE = re.compile(r'\[CITED:([^\]]+)\]')
_OPINION_VERDICT_RE = re.compile(r'VERDICT:\s*(BENIGN|MALIGNANT)', re.IGNORECASE)

# Ordered from strictest to most permissive — first match wins.
_LABEL_PATTERNS = [
    re.compile(r'\[?LABEL:\s*(BENIGN|MALIGNANT)\]?'),   # [LABEL: BENIGN] or LABEL: BENIGN
    re.compile(r'\[(BENIGN|MALIGNANT)[|\]]'),             # [BENIGN|75] or [BENIGN]
    re.compile(r'\b(BENIGN|MALIGNANT)\b'),                # bare word anywhere
]
_CONF_PATTERNS = [
    re.compile(r'\[?CONFIDENCE:\s*(\d+)\]?'),            # [CONFIDENCE: 75] or CONFIDENCE: 75
    re.compile(r'\[(BENIGN|MALIGNANT)\|(\d+)\]'),        # [BENIGN|75] — confidence in group 2
]


def _parse_single_response(text: str) -> tuple[str | None, int | None]:
    """Extract (label, confidence) from single-agent response using cascading fallbacks."""
    label: str | None = None
    for pat in _LABEL_PATTERNS:
        m = pat.search(text)
        if m:
            label = m.group(1)
            break

    confidence: int | None = None
    # Try CONFIDENCE: keyword first.
    m = _CONF_PATTERNS[0].search(text)
    if m:
        confidence = int(m.group(1))
    else:
        # Fallback: [LABEL|N] pipe format — confidence is group(2).
        m = _CONF_PATTERNS[1].search(text)
        if m:
            confidence = int(m.group(2))

    return label, confidence


def _parse_structured_response(text: str) -> list[_ParsedClaim]:
    parts = _CLAIM_SPLIT_RE.split(text)
    result = []
    for part in parts[1:]:
        claim_text = re.split(r'\[(?:ADDRESSED:|LABEL:|CITED:)', part, maxsplit=1)[0].strip()
        if len(claim_text.split()) < 2:
            continue
        addressed = [(m.group(1).strip(), m.group(2)) for m in _ADDR_RE.finditer(part)]
        label_m = _LABEL_RE.search(part)
        label = label_m.group(1) if label_m else None
        provenance = [m.group(1).strip() for m in _CITED_RE.finditer(part)]
        result.append(_ParsedClaim(text=claim_text, addressed=addressed, label=label, provenance=provenance))
    return result


def _build_edges(new_nodes: list[DebateNode], all_ctx_ids: set[str]) -> list[DebateEdge]:
    seen: dict[tuple[str, str], str] = {}
    for node in new_nodes:
        for addr in node.addressed:
            if addr.node_id in all_ctx_ids:
                sign = "+" if addr.stance == "AGREE" else "-"
                pair = (node.id, addr.node_id)
                if pair not in seen or seen[pair] == "+":
                    seen[pair] = sign
    return [DebateEdge(source_id=src, target_id=tgt, sign=sg) for (src, tgt), sg in seen.items()]


def _expert_turn(
    expert_id: str,
    round_idx: int,
    image_b64: str,
    history: list[DebateNode],
    system: str,
    client: LLMClient,
    kg_text: str,
    cfg: DictConfig,
) -> list[DebateNode]:
    """One structured expert turn → list of DebateNodes."""
    short_id_map: dict[str, str] = {n.short_id: n.id for n in history}
    prompt = _render("expert_turn.j2", kg_text=kg_text, history=history)

    text = client.complete(
        prompt=prompt,
        system=system,
        temperature=cfg.model.expert_temperature,
        image=image_b64,
    )
    logger.debug("%s round %d — response:\n%s", expert_id, round_idx, text)

    parsed = _parse_structured_response(text)
    if not parsed:
        logger.warning("%s round %d: no [CLAIM] blocks — creating fallback node", expert_id, round_idx)
        return [DebateNode(
            short_id=f"c{len(history) + 1}",
            expert_id=expert_id,
            round_idx=round_idx,
            text=text,
            provenance=[],
        )]

    nodes: list[DebateNode] = []
    for i, pc in enumerate(parsed):
        short_id = f"c{len(history) + i + 1}"
        addressed = [
            DebateAddress(node_id=short_id_map[sid], stance=stance)
            for sid, stance in pc.addressed
            if sid in short_id_map
        ]
        nodes.append(DebateNode(
            short_id=short_id,
            expert_id=expert_id,
            round_idx=round_idx,
            text=pc.text,
            label=pc.label,
            addressed=addressed,
            provenance=pc.provenance,
        ))

    for node in nodes:
        logger.info(
            "  %s round %d %s: LABEL=%s",
            expert_id, round_idx, node.short_id, node.label or "?",
        )
    return nodes


def _free_text_turn(
    expert_id: str,
    expert_letter: str,
    round_idx: int,
    image_b64: str,
    history: list[DebateNode],
    system: str,
    client: LLMClient,
    cfg: DictConfig,
) -> DebateNode:
    """One free-text expert turn (opinion mode) → single DebateNode."""
    prompt = _render("expert_opinion_turn.j2", history=history, expert_letter=expert_letter)

    text = client.complete(
        prompt=prompt,
        system=system,
        temperature=cfg.model.expert_temperature,
        image=image_b64,
    )
    logger.debug("%s round %d — response:\n%s", expert_id, round_idx, text)

    verdict_m = _OPINION_VERDICT_RE.search(text)
    label = verdict_m.group(1).upper() if verdict_m else None
    first_sentence = text.split('.')[0][:120].strip() if text else ""
    logger.info(
        "  Expert %s round %d: VERDICT=%s — %s",
        expert_letter, round_idx, label or "?", first_sentence,
    )

    short_id = f"c{len(history) + 1}"
    return DebateNode(
        short_id=short_id,
        expert_id=expert_id,
        round_idx=round_idx,
        text=text,
        label=label,
        provenance=[],
    )


# ---------------------------------------------------------------------------
# Public API — one function per mode
# ---------------------------------------------------------------------------

def run_single_agent(
    image_b64: str,
    cfg: DictConfig,
) -> tuple[SingleAgentResult, list[DebateNode], list[DebateEdge]]:
    """Mode 1: single Qwen2.5-VL call. Returns label + confidence (no debate).

    Returns:
        (result, [node], []) — node carries label and raw response text.
    """
    client = _make_expert_client(cfg)
    system = _render("system_single.j2")
    text = client.complete(
        prompt="Classify this breast ultrasound image.",
        system=system,
        temperature=cfg.model.expert_temperature,
        image=image_b64,
    )
    logger.debug("single_agent response: %r", text)

    raw_label, raw_conf = _parse_single_response(text)
    label: str = raw_label if raw_label else "BENIGN"
    confidence: int = raw_conf if raw_conf is not None else 50

    if raw_label is None or raw_conf is None:
        logger.warning(
            "single_agent: partial parse (label=%s conf=%s) — got %r",
            raw_label, raw_conf, text[:200],
        )

    result = SingleAgentResult(label=label, confidence=confidence)  # type: ignore[arg-type]
    node = DebateNode(
        short_id="c1",
        expert_id="expert_a",
        round_idx=0,
        text=text,
        label=label,
        provenance=[],
    )
    return result, [node], []


def run_opinion_debate(
    image_b64: str,
    cfg: DictConfig,
    round_idx: int = 0,
    history: list[DebateNode] | None = None,
) -> tuple[list[DebateNode], list[DebateEdge], bool]:
    """Mode 2: one round of free-text debate (no [CLAIM] structure, no KG).

    Both experts emit plain prose. Verdict is decided by the judge later.

    Returns:
        (new_nodes, [], finished) — finished=True if any expert signals [FINISH].
    """
    history = list(history or [])
    client = _make_expert_client(cfg)

    first_id, second_id = "expert_a", "expert_b"
    first_letter, second_letter = "A", "B"

    system_first = _render("system_opinion.j2", expert_letter="A", other_letter="B")
    system_second = _render("system_opinion.j2", expert_letter="B", other_letter="A")

    node_first = _free_text_turn(first_id, first_letter, round_idx, image_b64, history, system_first, client, cfg)
    node_second = _free_text_turn(second_id, second_letter, round_idx, image_b64, history + [node_first], system_second, client, cfg)

    all_new = [node_first, node_second]

    # Accept [FINISH] or bare FINISH (model sometimes omits brackets).
    # Ignore if the same response also has NEW FINDING: (mixed format).
    # Not valid on round 0 — both experts must have spoken at least once first.
    def _signals_finish(node: DebateNode) -> bool:
        t = node.text.upper()
        if "NEW FINDING:" in t:
            return False
        return "[FINISH]" in t or re.search(r'(?<!\w)FINISH(?!\w)', t) is not None

    finished = round_idx >= 1 and all(_signals_finish(n) for n in all_new)

    logger.info(
        "opinion_debate round %d: Expert A=%s Expert B=%s%s",
        round_idx,
        node_first.label or "?",
        node_second.label or "?",
        " [FINISH — debate closed]" if finished else "",
    )
    return all_new, [], finished


def run_structured_debate(
    image_b64: str,
    cfg: DictConfig,
    round_idx: int = 0,
    history: list[DebateNode] | None = None,
    kg_text: str = "",
    adversarial: bool = False,
) -> tuple[list[DebateNode], list[DebateEdge]]:
    """Modes 3-5: one round of structured [CLAIM] debate.

    Args:
        image_b64: Base64-encoded image.
        cfg: Hydra config.
        round_idx: Current round index (determines opener).
        history: All prior DebateNodes across all rounds.
        kg_text: Serialized KG text for injection (empty in mode 3).
        adversarial: If True, agent A is locked to BENIGN, agent B to MALIGNANT.
    """
    history = list(history or [])
    client = _make_expert_client(cfg)

    if adversarial:
        # Fixed stance assignment regardless of round: A=BENIGN, B=MALIGNANT.
        system_a = _render("system_benign.j2", expert_letter="A", other_letter="B")
        system_b = _render("system_malignant.j2", expert_letter="B", other_letter="A")
    else:
        system_a = _render("system_debate.j2", expert_letter="A", other_letter="B")
        system_b = _render("system_debate.j2", expert_letter="B", other_letter="A")

    if round_idx % 2 == 0:
        first_id, second_id = "expert_a", "expert_b"
        system_first, system_second = system_a, system_b
    else:
        first_id, second_id = "expert_b", "expert_a"
        system_first, system_second = system_b, system_a

    nodes_first = _expert_turn(first_id, round_idx, image_b64, history, system_first, client, kg_text, cfg)
    nodes_second = _expert_turn(second_id, round_idx, image_b64, history + nodes_first, system_second, client, kg_text, cfg)

    all_new = nodes_first + nodes_second
    all_ctx_ids = {n.id for n in history + all_new}
    edges = _build_edges(all_new, all_ctx_ids)

    logger.info(
        "structured_debate round %d (adversarial=%s): %d node(s), %d edge(s)",
        round_idx, adversarial, len(all_new), len(edges),
    )
    return all_new, edges
