"""Judge panel: three judges from different model families, aggregated via mean.

See SPEC §Module4.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import krippendorff
import numpy as np
from omegaconf import DictConfig

from debate_kg.judges.base import LLMJudge
from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore

logger = logging.getLogger(__name__)


class _StubJudge(LLMJudge):
    """Placeholder judge that returns fixed mid-range scores. Used in smoke runs."""

    def score_utterance(
        self,
        node: DebateNode,
        kg: KnowledgeGraph,
        claim: str = "",
    ) -> JudgeScore:
        return JudgeScore(groundedness=50.0, factuality=50.0)

    def score_expert_overall(self, nodes: list[DebateNode]) -> ExpertScore:
        return ExpertScore(overall=50.0)


class JudgePanel:
    """Aggregate scores from a panel of judges. Requires ≥1 judge."""

    def __init__(self, judges: list[LLMJudge]) -> None:
        if not judges:
            raise ValueError("JudgePanel requires at least one judge")
        self.judges = judges

    def score_debate(
        self,
        nodes: list[DebateNode],
        kg_a: KnowledgeGraph,
        kg_b: KnowledgeGraph,
        claim: str = "",
    ) -> dict[str, list[JudgeScore]]:
        """Score all debate nodes with each judge.

        Args:
            nodes: All DebateNodes from the current round.
            kg_a: Expert A's current KG (used to verify citations for A's nodes).
            kg_b: Expert B's current KG.
            claim: The FEVER claim text, threaded to each scoring prompt.

        Returns:
            Dict mapping node.id → list of JudgeScore (one per judge).
        """
        scores: dict[str, list[JudgeScore]] = {}
        for node in nodes:
            kg = kg_a if node.expert_id == "expert_a" else kg_b
            scores[node.id] = [
                judge.score_utterance(node, kg, claim) for judge in self.judges
            ]
            logger.debug(
                "Scored node %s (expert=%s): %s",
                node.id[:8],
                node.expert_id,
                scores[node.id],
            )
        return scores

    def score_experts_overall(
        self,
        nodes: list[DebateNode],
    ) -> dict[str, list[ExpertScore]]:
        """Score each expert's overall performance.

        Args:
            nodes: All DebateNodes from the debate.

        Returns:
            Dict mapping expert_id → list of ExpertScore (one per judge).
        """
        by_expert: dict[str, list[DebateNode]] = defaultdict(list)
        for node in nodes:
            by_expert[node.expert_id].append(node)

        result: dict[str, list[ExpertScore]] = {}
        for expert_id, expert_nodes in by_expert.items():
            result[expert_id] = [
                judge.score_expert_overall(expert_nodes) for judge in self.judges
            ]
        return result

    def aggregate_scores(
        self,
        scores: dict[str, list[JudgeScore]],
    ) -> dict[str, JudgeScore]:
        """Mean JudgeScore across judges for each node.

        Returns JudgeScore(0.0, 0.0) for nodes with no judge scores.
        """
        result: dict[str, JudgeScore] = {}
        for nid, jscores in scores.items():
            if not jscores:
                result[nid] = JudgeScore(groundedness=0.0, factuality=0.0)
            else:
                result[nid] = JudgeScore(
                    groundedness=sum(s.groundedness for s in jscores) / len(jscores),
                    factuality=sum(s.factuality for s in jscores) / len(jscores),
                )
        return result

    def inter_judge_agreement(self, scores: dict[str, list[JudgeScore]]) -> float:
        """Compute Krippendorff's α across judges for groundedness scores.

        Uses level_of_measurement="interval" since 0–100 scores are continuous
        floats with equal-interval semantics. Returns 0.0 when α is undefined
        (fewer than 2 judges or fewer than 2 scored items).

        Args:
            scores: Output of score_debate().

        Returns:
            Krippendorff's α (float in [-1, 1]).
        """
        n_judges = len(self.judges)
        n_items = len(scores)
        if n_judges < 2 or n_items < 2:
            logger.debug(
                "inter_judge_agreement: α undefined (judges=%d, items=%d) — returning 0.0",
                n_judges,
                n_items,
            )
            return 0.0

        node_ids = list(scores.keys())
        data = np.array(
            [[scores[nid][j].groundedness for nid in node_ids] for j in range(n_judges)],
            dtype=float,
        )
        try:
            alpha = krippendorff.alpha(data, level_of_measurement="interval")
        except ValueError:
            # α is undefined when all scores are identical (0/0 form)
            logger.debug("inter_judge_agreement: all scores identical — α undefined, returning 0.0")
            return 0.0
        logger.info("Inter-judge Krippendorff's α (groundedness): %.4f", alpha)
        return float(alpha)


def build_judge_panel(cfg: DictConfig) -> JudgePanel:
    """Build a JudgePanel from Hydra config (cfg.judges.models).

    Each model name maps to one OllamaJudge. Models should come from different
    families (Llama, Qwen, Mistral) to reduce correlated bias — see SPEC §Module4.
    """
    from debate_kg.judges.llm_judge import OllamaJudge
    judges: list[LLMJudge] = [OllamaJudge(model_name=m, cfg=cfg) for m in cfg.judges.models]
    logger.info("Built JudgePanel with %d judges: %s", len(judges), list(cfg.judges.models))
    return JudgePanel(judges=judges)
