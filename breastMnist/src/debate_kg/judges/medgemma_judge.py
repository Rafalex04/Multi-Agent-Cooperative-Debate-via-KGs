"""MedGemma-4B-IT judge for the BreastMNIST pipeline.

Backed by VLLMClient. Implements two scoring modes:
  - score_utterance: rates a single DebateNode on image grounding and medical accuracy.
  - pick_winner:     given the full debate transcript + image, picks the stronger expert.

JudgeScore fields are repurposed for this domain:
  groundedness → IMAGE_GROUNDING  (how well the claim cites observable image features)
  factuality   → MEDICAL_ACCURACY (how ACR BI-RADS correct the claim is)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from breastMnist.src.debate_kg.judges.base import LLMJudge
from breastMnist.src.debate_kg.kg.graph import KnowledgeGraph
from breastMnist.src.debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore, WinnerJudgment
from breastMnist.src.debate_kg.models.base import LLMClient
from breastMnist.src.debate_kg.models.vllm_client import VLLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "debate" / "prompts" / "breast"
_JINJA_ENV = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)


def _render(name: str, **kwargs: object) -> str:
    return _JINJA_ENV.get_template(name).render(**kwargs)


class MedGemmaJudge(LLMJudge):
    """Single MedGemma-4B-IT judge for breast ultrasound DebateNodes."""

    def __init__(self, cfg: DictConfig, client: LLMClient) -> None:
        model_name: str = cfg.judges.model
        super().__init__(model_name=model_name)
        self._client = client
        self._temperature: float = float(cfg.judges.temperature)

    def score_utterance(
        self,
        node: DebateNode,
        kg: KnowledgeGraph,
        claim: str = "",
        image_b64: str | None = None,
    ) -> JudgeScore:
        """Score one DebateNode on image grounding and medical accuracy.

        Args:
            node: The claim node to evaluate.
            kg: Unused here (KG is domain-wide, not per-expert). Kept for
                interface compatibility with LLMJudge.
            claim: Unused (no text claim in image pipeline). Kept for interface.
            image_b64: Base64-encoded JPEG of the breast ultrasound image.

        Returns:
            JudgeScore where groundedness = IMAGE_GROUNDING, factuality = MEDICAL_ACCURACY.
        """
        logger.debug(
            "MedGemmaJudge scoring %s (%s): %s",
            node.short_id, node.expert_id, node.text,
        )
        prompt = _render("judge_node.j2", node=node)
        text = self._client.complete(
            prompt=prompt,
            temperature=self._temperature,
            image=image_b64,
        )
        logger.debug("MedGemmaJudge node %s raw response:\n%s", node.short_id, text)
        js = _parse_node_scores(text, node.short_id)
        logger.info(
            "  scored %s (%s): IMAGE_GROUNDING=%d  MEDICAL_ACCURACY=%d",
            node.short_id, node.expert_id, int(js.groundedness), int(js.factuality),
        )
        return js

    def score_expert_overall(self, nodes: list[DebateNode]) -> ExpertScore:
        """Not used in the breast pipeline — opinion mode uses pick_winner instead.

        Returns a neutral score (50) to satisfy the interface.
        """
        return ExpertScore(overall=50.0)

    def pick_winner(
        self,
        all_nodes: list[DebateNode],
        image_b64: str | None = None,
    ) -> WinnerJudgment:
        """Decide which expert made the stronger argument (opinion-mode verdict).

        Args:
            all_nodes: All DebateNodes from the completed debate.
            image_b64: Base64-encoded JPEG of the breast ultrasound image.

        Returns:
            WinnerJudgment with winner ("expert_a" | "expert_b") and reasoning.
        """
        prompt = _render("judge_opinion.j2", transcript=all_nodes)
        text = self._client.complete(
            prompt=prompt,
            temperature=self._temperature,
            image=image_b64,
        )
        logger.debug("MedGemmaJudge pick_winner response:\n%s", text)
        return _parse_winner(text)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_node_scores(text: str, short_id: str) -> JudgeScore:
    """Extract IMAGE_GROUNDING and MEDICAL_ACCURACY from judge response."""
    grounding_m = re.search(r'IMAGE_GROUNDING\s*:\s*(\d+)', text, re.IGNORECASE)
    accuracy_m = re.search(r'MEDICAL_ACCURACY\s*:\s*(\d+)', text, re.IGNORECASE)

    if not grounding_m or not accuracy_m:
        logger.warning(
            "MedGemmaJudge: could not parse scores for node %s — defaulting to 50/50. Response: %r",
            short_id, text[:200],
        )
        return JudgeScore(groundedness=50.0, factuality=50.0)

    return JudgeScore(
        groundedness=float(min(max(int(grounding_m.group(1)), 0), 100)),
        factuality=float(min(max(int(accuracy_m.group(1)), 0), 100)),
    )


def _parse_winner(text: str) -> WinnerJudgment:
    """Extract WINNER and REASONING from judge response."""
    winner_m = re.search(r'WINNER\s*:\s*(expert_a|expert_b)', text, re.IGNORECASE)
    reasoning_m = re.search(r'REASONING\s*:\s*(.+)', text, re.IGNORECASE)

    if not winner_m:
        logger.warning(
            "MedGemmaJudge: could not parse winner — defaulting to expert_a. Response: %r",
            text[:200],
        )
        return WinnerJudgment(winner="expert_a", reasoning="parse failure")

    winner = winner_m.group(1).lower()
    reasoning = reasoning_m.group(1).strip() if reasoning_m else ""
    return WinnerJudgment(winner=winner, reasoning=reasoning)  # type: ignore[arg-type]


def build_medgemma_judge(cfg: DictConfig) -> MedGemmaJudge:
    """Construct MedGemmaJudge from Hydra config.

    Selects backend based on cfg.judges.backend:
      "ollama" — OllamaClient (local CPU, slow, for testing)
      "vllm"   — VLLMClient  (cluster GPU, fast, for real runs)
    """
    backend: str = getattr(cfg.judges, "backend", "vllm")
    if backend == "ollama":
        from breastMnist.src.debate_kg.models.ollama_client import OllamaClient
        ollama_model: str = getattr(cfg.judges, "ollama_model", "medgemma:latest")
        ollama_url: str = getattr(cfg.judges, "ollama_base_url", "http://localhost:11434")
        # Judge prompts are single claims or short transcripts (a few hundred
        # tokens) — a small num_ctx avoids the huge compute-graph buffer that
        # 32k forces, which can crowd the expert model off the GPU.
        client: LLMClient = OllamaClient(model=ollama_model, base_url=ollama_url, num_ctx=4096)
        logger.info("Built MedGemmaJudge: backend=ollama model=%s", ollama_model)
    else:
        client = VLLMClient(model=cfg.judges.model, base_url=cfg.judges.base_url)
        logger.info("Built MedGemmaJudge: backend=vllm model=%s", cfg.judges.model)

    return MedGemmaJudge(cfg=cfg, client=client)
