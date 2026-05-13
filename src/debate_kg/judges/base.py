"""Abstract base class for LLM judges.

Each concrete judge wraps one model from a different family (Llama, Qwen, Mistral)
to reduce correlated bias. See SPEC §Module4.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import DebateNode, ExpertScore, JudgeScore


class LLMJudge(ABC):
    """Abstract judge: scores utterances and experts on a rubric. See SPEC §Module4."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def score_utterance(
        self,
        node: DebateNode,
        kg: KnowledgeGraph,
        claim: str = "",
    ) -> JudgeScore:
        """Score one utterance on KG-groundedness and factuality (0–100 each).

        Args:
            node: The debate utterance to score.
            kg: The KG that grounded this expert's utterances (for checking citations).
            claim: The FEVER claim text (used in the scoring prompt).

        Returns:
            JudgeScore with groundedness and factuality fields.
        """
        ...

    @abstractmethod
    def score_expert_overall(self, nodes: list[DebateNode]) -> ExpertScore:
        """Score an expert's overall performance across all their utterances (0–100).

        Args:
            nodes: All utterances by one expert in this debate.

        Returns:
            ExpertScore with overall field.
        """
        ...
