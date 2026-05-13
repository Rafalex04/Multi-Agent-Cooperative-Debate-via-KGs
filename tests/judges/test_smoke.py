"""Unit tests for the Judge Panel (Module 4).

All LLM calls are mocked — no Ollama, no network, fully offline.
The mock patches debate_kg.judges.llm_judge._make_client so every
client.complete() call returns a controlled string.

Test matrix
-----------
  _parse_score:
    test_parse_score_extracts_integer
    test_parse_score_extracts_float
    test_parse_score_clamps_over_100
    test_parse_score_clamps_below_zero
    test_parse_score_fallback_on_missing_key
    test_parse_score_case_insensitive

  OllamaJudge:
    test_ollama_judge_score_utterance_returns_judge_score
    test_ollama_judge_cited_triples_in_prompt
    test_ollama_judge_no_provenance_prompt_says_no_triples
    test_ollama_judge_score_expert_overall

  JudgePanel.score_debate:
    test_score_debate_keys_are_node_ids
    test_score_debate_one_score_per_judge
    test_score_debate_routes_kg_by_expert_a
    test_score_debate_routes_kg_by_expert_b

  JudgePanel.score_experts_overall:
    test_score_experts_overall_keys_are_expert_ids

  JudgePanel.aggregate_scores:
    test_aggregate_scores_computes_mean
    test_aggregate_scores_empty_list_zeros

  JudgePanel.inter_judge_agreement:
    test_inter_judge_agreement_returns_float_in_range
    test_inter_judge_agreement_single_judge_returns_zero
    test_inter_judge_agreement_empty_returns_zero

  build_judge_panel:
    test_build_judge_panel_creates_correct_count

  Legacy smoke tests (still pass):
    test_stub_judge_score_utterance_returns_judge_score
    test_judge_panel_requires_at_least_one_judge
    test_judge_panel_score_debate_returns_dict
    test_inter_judge_agreement_returns_float
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from debate_kg.judges.llm_judge import OllamaJudge, _parse_score
from debate_kg.judges.panel import JudgePanel, _StubJudge, build_judge_panel
from debate_kg.kg.graph import KnowledgeGraph
from debate_kg.kg.schema import DebateNode, JudgeScore, Triple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(models: list[str] | None = None):
    return OmegaConf.create({
        "model": {
            "judge_temperature": 0.0,
            "ollama_base_url": "http://localhost:11434",
        },
        "judges": {
            "models": models or ["llama3.1:8b", "qwen2.5:7b", "mistral:7b-instruct"],
        },
    })


_node_counter = [0]

def _node(expert_id: str = "expert_a", provenance: list[str] | None = None) -> DebateNode:
    _node_counter[0] += 1
    return DebateNode(
        short_id=f"c{_node_counter[0]}",
        expert_id=expert_id,
        round_idx=0,
        text="The sky is blue.",
        provenance=provenance or [],
    )


def _mock_client(response: str) -> MagicMock:
    client = MagicMock()
    client.complete.return_value = response
    return client


def _patched_judge(response: str, cfg=None) -> OllamaJudge:
    cfg = cfg or _cfg()
    with patch("debate_kg.judges.llm_judge._make_client", return_value=_mock_client(response)):
        judge = OllamaJudge(model_name="llama3.1:8b", cfg=cfg)
    judge._client = _mock_client(response)
    return judge


# ---------------------------------------------------------------------------
# _parse_score unit tests
# ---------------------------------------------------------------------------

def test_parse_score_extracts_integer() -> None:
    assert _parse_score("GROUNDEDNESS: 80\nFACTUALITY: 60", "GROUNDEDNESS", 50.0) == 80.0


def test_parse_score_extracts_float() -> None:
    assert _parse_score("GROUNDEDNESS: 73.5", "GROUNDEDNESS", 50.0) == 73.5


def test_parse_score_clamps_over_100() -> None:
    assert _parse_score("GROUNDEDNESS: 150", "GROUNDEDNESS", 50.0) == 100.0


def test_parse_score_clamps_below_zero() -> None:
    assert _parse_score("GROUNDEDNESS: -5", "GROUNDEDNESS", 50.0) == 0.0


def test_parse_score_fallback_on_missing_key() -> None:
    assert _parse_score("nothing useful here", "GROUNDEDNESS", 42.0) == 42.0


def test_parse_score_case_insensitive() -> None:
    assert _parse_score("groundedness: 70", "GROUNDEDNESS", 50.0) == 70.0


# ---------------------------------------------------------------------------
# OllamaJudge tests
# ---------------------------------------------------------------------------

def test_ollama_judge_score_utterance_returns_judge_score() -> None:
    judge = _patched_judge("GROUNDEDNESS: 70\nFACTUALITY: 85")
    node = _node()
    score = judge.score_utterance(node, KnowledgeGraph(), claim="Test claim.")
    assert score.groundedness == 70.0
    assert score.factuality == 85.0


def test_ollama_judge_cited_triples_in_prompt() -> None:
    kg = KnowledgeGraph()
    triple = Triple(subject="Sky", predicate="hasColor", object="Blue")
    kg.add_triple(triple)

    judge = _patched_judge("GROUNDEDNESS: 90\nFACTUALITY: 90")
    node = _node(provenance=[triple.uuid])
    judge.score_utterance(node, kg, claim="The sky is blue.")

    prompt_sent = judge._client.complete.call_args[0][0]
    assert "Sky" in prompt_sent
    assert "hasColor" in prompt_sent
    assert "Blue" in prompt_sent


def test_ollama_judge_no_provenance_prompt_says_no_triples() -> None:
    judge = _patched_judge("GROUNDEDNESS: 50\nFACTUALITY: 50")
    node = _node(provenance=[])
    judge.score_utterance(node, KnowledgeGraph(), claim="Some claim.")

    prompt_sent = judge._client.complete.call_args[0][0]
    assert "No triples were cited" in prompt_sent


def test_ollama_judge_score_expert_overall() -> None:
    judge = _patched_judge("OVERALL: 72")
    nodes = [_node("expert_a"), _node("expert_a")]
    score = judge.score_expert_overall(nodes)
    assert score.overall == 72.0


# ---------------------------------------------------------------------------
# JudgePanel.score_debate tests
# ---------------------------------------------------------------------------

def _stub_panel(n: int = 2) -> JudgePanel:
    return JudgePanel(judges=[_StubJudge(f"s{i}") for i in range(n)])


def test_score_debate_keys_are_node_ids() -> None:
    panel = _stub_panel(2)
    nodes = [_node("expert_a"), _node("expert_b")]
    result = panel.score_debate(nodes, KnowledgeGraph(), KnowledgeGraph())
    assert set(result.keys()) == {n.id for n in nodes}


def test_score_debate_one_score_per_judge() -> None:
    panel = _stub_panel(2)
    nodes = [_node("expert_a")]
    result = panel.score_debate(nodes, KnowledgeGraph(), KnowledgeGraph())
    assert len(result[nodes[0].id]) == 2


def test_score_debate_routes_kg_by_expert_a() -> None:
    """expert_a node should be scored with kg_a."""
    kg_a = KnowledgeGraph()
    triple_a = Triple(subject="A", predicate="p", object="X")
    kg_a.add_triple(triple_a)

    received_kgs: list[KnowledgeGraph] = []

    class _RecordingJudge(_StubJudge):
        def score_utterance(self, node, kg, claim=""):
            received_kgs.append(kg)
            return super().score_utterance(node, kg, claim)

    panel = JudgePanel(judges=[_RecordingJudge("r")])
    node_a = _node("expert_a", provenance=[triple_a.uuid])
    panel.score_debate([node_a], kg_a, KnowledgeGraph())
    assert received_kgs[0] is kg_a


def test_score_debate_routes_kg_by_expert_b() -> None:
    kg_b = KnowledgeGraph()
    received_kgs: list[KnowledgeGraph] = []

    class _RecordingJudge(_StubJudge):
        def score_utterance(self, node, kg, claim=""):
            received_kgs.append(kg)
            return super().score_utterance(node, kg, claim)

    panel = JudgePanel(judges=[_RecordingJudge("r")])
    node_b = _node("expert_b")
    panel.score_debate([node_b], KnowledgeGraph(), kg_b)
    assert received_kgs[0] is kg_b


# ---------------------------------------------------------------------------
# JudgePanel.score_experts_overall tests
# ---------------------------------------------------------------------------

def test_score_experts_overall_keys_are_expert_ids() -> None:
    panel = _stub_panel(2)
    nodes = [_node("expert_a"), _node("expert_b"), _node("expert_a")]
    result = panel.score_experts_overall(nodes)
    assert set(result.keys()) == {"expert_a", "expert_b"}
    assert len(result["expert_a"]) == 2  # one per judge


# ---------------------------------------------------------------------------
# JudgePanel.aggregate_scores tests
# ---------------------------------------------------------------------------

def test_aggregate_scores_computes_mean() -> None:
    panel = _stub_panel(2)
    nid = "node-1"
    scores = {nid: [JudgeScore(groundedness=80.0, factuality=60.0),
                    JudgeScore(groundedness=60.0, factuality=80.0)]}
    result = panel.aggregate_scores(scores)
    assert result[nid].groundedness == pytest.approx(70.0)
    assert result[nid].factuality == pytest.approx(70.0)


def test_aggregate_scores_empty_list_zeros() -> None:
    panel = _stub_panel(1)
    nid = "node-empty"
    result = panel.aggregate_scores({nid: []})
    assert result[nid].groundedness == 0.0
    assert result[nid].factuality == 0.0


# ---------------------------------------------------------------------------
# JudgePanel.inter_judge_agreement tests
# ---------------------------------------------------------------------------

def test_inter_judge_agreement_returns_float_in_range() -> None:
    panel = _stub_panel(3)
    scores = {
        "n1": [JudgeScore(groundedness=80.0, factuality=70.0),
               JudgeScore(groundedness=75.0, factuality=65.0),
               JudgeScore(groundedness=85.0, factuality=75.0)],
        "n2": [JudgeScore(groundedness=40.0, factuality=50.0),
               JudgeScore(groundedness=45.0, factuality=55.0),
               JudgeScore(groundedness=35.0, factuality=45.0)],
    }
    alpha = panel.inter_judge_agreement(scores)
    assert isinstance(alpha, float)
    assert -1.0 <= alpha <= 1.0


def test_inter_judge_agreement_single_judge_returns_zero() -> None:
    panel = _stub_panel(1)
    scores = {"n1": [JudgeScore(groundedness=80.0, factuality=70.0)]}
    assert panel.inter_judge_agreement(scores) == 0.0


def test_inter_judge_agreement_empty_returns_zero() -> None:
    panel = _stub_panel(2)
    assert panel.inter_judge_agreement({}) == 0.0


# ---------------------------------------------------------------------------
# build_judge_panel test
# ---------------------------------------------------------------------------

def test_build_judge_panel_creates_correct_count() -> None:
    cfg = _cfg(models=["llama3.1:8b", "qwen2.5:7b", "mistral:7b-instruct"])
    with patch("debate_kg.judges.llm_judge._make_client", return_value=_mock_client("")):
        panel = build_judge_panel(cfg)
    assert len(panel.judges) == 3


# ---------------------------------------------------------------------------
# Legacy smoke tests (unchanged API, still pass)
# ---------------------------------------------------------------------------

def test_stub_judge_score_utterance_returns_judge_score() -> None:
    judge = _StubJudge("stub")
    score = judge.score_utterance(_node(), KnowledgeGraph())
    assert 0.0 <= score.groundedness <= 100.0
    assert 0.0 <= score.factuality <= 100.0


def test_judge_panel_requires_at_least_one_judge() -> None:
    with pytest.raises(ValueError):
        JudgePanel(judges=[])


def test_judge_panel_score_debate_returns_dict() -> None:
    panel = JudgePanel(judges=[_StubJudge("s1"), _StubJudge("s2")])
    nodes = [_node(expert_id="expert_a")]
    result = panel.score_debate(nodes, KnowledgeGraph(), KnowledgeGraph())
    assert isinstance(result, dict)
    assert nodes[0].id in result


def test_inter_judge_agreement_returns_float() -> None:
    panel = JudgePanel(judges=[_StubJudge("s1")])
    alpha = panel.inter_judge_agreement({})
    assert isinstance(alpha, float)
