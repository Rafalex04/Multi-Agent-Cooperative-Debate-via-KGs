"""Concrete LLM judge backed by OllamaClient.

Scores one utterance on KG-groundedness and factuality (0–100 each),
and scores an expert's overall debate performance (0–100).

Prompt templates live in judges/prompts/ as Jinja2 files.
See SPEC §Module4.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from debate_kg.judges.base import LLMJudge
from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore
from debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


def _render(name: str, **kwargs: object) -> str:
    return _JINJA_ENV.get_template(name).render(**kwargs)


def _make_client(model_name: str, cfg: DictConfig) -> LLMClient:
    from debate_kg.models.ollama_client import OllamaClient
    base_url: str = getattr(cfg.model, "ollama_base_url", "http://localhost:11434")
    return OllamaClient(model=model_name, base_url=base_url)


def _parse_score(text: str, key: str, fallback: float) -> float:
    """Extract KEY: VALUE from text; clamp to [0, 100]; fall back with WARNING.

    Tolerates extra words between the key and the number (e.g. "FACTUALITY score: 85",
    "**FACTUALITY**: 85", "FACTUALITY (0-100): 85").
    """
    m = re.search(rf'{key}\D{{0,40}}?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return max(0.0, min(100.0, float(m.group(1))))
    logger.warning("Could not parse %s score from judge response — using fallback %.1f", key, fallback)
    logger.debug("Raw judge response for failed %s parse:\n%s", key, text)
    return fallback


class OllamaJudge(LLMJudge):
    """LLM judge backed by a locally-served Ollama model."""

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        super().__init__(model_name)
        self._cfg = cfg
        self._client = _make_client(model_name, cfg)

    def score_utterance(
        self,
        node: DebateNode,
        kg: KnowledgeGraph,
        claim: str = "",
    ) -> JudgeScore:
        """Score one utterance on groundedness and factuality (0–100 each).

        Resolves node.provenance UUIDs to triples via kg.get_triple() and
        includes them in the prompt. If provenance is empty, the judge sees
        only the utterance text.
        """
        cited = [t for uid in node.provenance if (t := kg.get_triple(uid)) is not None]
        prompt = _render("utterance_rubric.j2", claim=claim, node=node, cited_triples=cited)
        raw = self._client.complete(
            prompt, system="", temperature=self._cfg.model.judge_temperature
        )
        return JudgeScore(
            groundedness=_parse_score(raw, "GROUNDEDNESS", 50.0),
            factuality=_parse_score(raw, "FACTUALITY", 50.0),
        )

    def score_expert_overall(self, nodes: list[DebateNode]) -> ExpertScore:
        """Score an expert's overall performance across all their utterances (0–100)."""
        expert_id = nodes[0].expert_id if nodes else "unknown"
        prompt = _render("expert_overall.j2", expert_id=expert_id, nodes=nodes)
        raw = self._client.complete(
            prompt, system="", temperature=self._cfg.model.judge_temperature
        )
        return ExpertScore(overall=_parse_score(raw, "OVERALL", 50.0))
