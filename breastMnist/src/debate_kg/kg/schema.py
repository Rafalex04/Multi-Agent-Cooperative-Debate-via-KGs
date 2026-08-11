"""Pydantic data structures shared across all modules.

Anything that crosses a module boundary lives here. Keep it flat — no business
logic, just validated data containers.
"""
from __future__ import annotations

import uuid as _uuid
from typing import Literal

from pydantic import BaseModel, Field


class Triple(BaseModel):
    """One RDF-style triple with a stable UUID for provenance tracking."""

    uuid: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    subject: str
    predicate: str
    object: str


class DebateAddress(BaseModel):
    """A stance taken by one claim node toward a previously addressed claim node."""

    node_id: str                            # UUID of the addressed claim node
    stance: Literal["AGREE", "DISAGREE"]


class DebateNode(BaseModel):
    """One factual claim made by an expert, with provenance to KG triples.

    A single expert turn (one LLM call) may produce multiple DebateNodes —
    one per [CLAIM] block in the structured response. Each node IS a claim.
    """

    id: str = Field(default_factory=lambda: str(_uuid.uuid4()))
    short_id: str                           # sequential debate-wide ID: "c1", "c2", …
    expert_id: str
    round_idx: int
    text: str                               # this claim's text content
    label: str | None = None               # SUPPORTS / REFUTES / NOT ENOUGH INFO
    addressed: list[DebateAddress] = Field(default_factory=list)
    provenance: list[str]                   # Triple UUIDs cited in this claim's text


class DebateEdge(BaseModel):
    """Signed relation between two claim nodes (AGREE=+, DISAGREE=-).

    Direction: source → target means "source addresses target".
    The new (responding) claim is the source; the older (addressed) claim is the target.
    """

    source_id: str                          # responding (newer) claim node UUID
    target_id: str                          # addressed (older) claim node UUID
    sign: Literal["+", "-"]               # new claims produce no edge (no "~")


class JudgeScore(BaseModel):
    """Per-utterance scores from one judge (both dimensions 0–100)."""

    groundedness: float
    factuality: float


class ExpertScore(BaseModel):
    """End-of-debate overall score for one expert from one judge (0–100)."""

    overall: float


class MergeResult(BaseModel):
    """What the merger decided for each triple."""

    kept: list[str]      # Triple UUIDs kept in the merged KG
    dropped: list[str]   # Triple UUIDs dropped
    resolved: list[str]  # Triple UUIDs resolved in favour of the higher-scoring side


# ---------------------------------------------------------------------------
# BreastMNIST-specific schemas
# ---------------------------------------------------------------------------

class SingleAgentResult(BaseModel):
    """Parsed output of the single-agent classification call."""

    label: Literal["BENIGN", "MALIGNANT"]
    confidence: int  # 0–100, self-reported by the model


class WinnerJudgment(BaseModel):
    """MedGemma's verdict on which expert won the opinion-mode debate."""

    winner: Literal["expert_a", "expert_b"]
    reasoning: str
