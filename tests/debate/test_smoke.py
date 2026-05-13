"""Unit tests for the Debate Orchestrator (Module 3).

All LLM calls are mocked — no Ollama, no network, fully offline.
The mock patches debate_kg.debate.orchestrator._make_client so every
client.complete() call can return a controlled string.

Test matrix
-----------
  Helpers:
    test_parse_provenance_direct
    test_parse_provenance_deduplicates
    test_parse_provenance_unknown_index_ignored
    test_parse_provenance_empty_text

  Structured response parser:
    test_parse_structured_single_claim
    test_parse_structured_multi_claim
    test_parse_structured_no_addressed
    test_parse_structured_nei_label
    test_parse_structured_missing_claims_returns_empty

  run_debate() structure:
    test_run_debate_agree_edge
    test_run_debate_disagree_edge
    test_run_debate_no_addressed_no_edge
    test_run_debate_returns_multiple_claim_nodes
    test_run_debate_alternates_opener_round0
    test_run_debate_alternates_opener_round1
    test_run_debate_conflict_dedup_prefers_disagree
    test_run_debate_cross_round_edge
    test_short_ids_sequential
    test_node_label_populated

  Provenance:
    test_provenance_extracted_from_citations
    test_provenance_empty_when_no_citations

  Prompt templates:
    test_all_templates_load_without_error
    test_system_template_includes_claim
    test_expert_turn_template_renders_history_with_short_ids
    test_expert_turn_template_triple_numbering
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from debate_kg.debate.orchestrator import (
    _JINJA_ENV,
    _parse_provenance,
    _parse_structured_response,
    run_debate,
    run_single_expert,
)
from debate_kg.kg.schema import DebateNode, Triple

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _cfg() -> object:
    return OmegaConf.create({
        "model": {
            "expert_model": "llama3.1:8b",
            "expert_temperature": 0.7,
        },
        "run": {},
    })


def _triples(n: int) -> list[Triple]:
    return [Triple(subject=f"S{i}", predicate="p", object=f"O{i}") for i in range(n)]


def _mock_client(responses: list[str]) -> MagicMock:
    client = MagicMock()
    client.complete.side_effect = responses
    return client


def _node(short_id: str, expert_id: str = "expert_a", round_idx: int = 0) -> DebateNode:
    return DebateNode(
        short_id=short_id,
        expert_id=expert_id,
        round_idx=round_idx,
        text="some claim text",
        provenance=[],
    )


def _patched_run(
    responses: list[str],
    claim: str = "Test claim.",
    triples_a: list[Triple] | None = None,
    triples_b: list[Triple] | None = None,
    round_idx: int = 0,
    history: list[DebateNode] | None = None,
):
    cfg = _cfg()
    with patch("debate_kg.debate.orchestrator._make_client", return_value=_mock_client(responses)):
        return run_debate(
            claim,
            triples_a or [],
            triples_b or [],
            cfg,
            round_idx,
            history=history,
        )


# ---------------------------------------------------------------------------
# _parse_provenance unit tests
# ---------------------------------------------------------------------------

def test_parse_provenance_direct() -> None:
    uuid_map = {1: "uuid-aaa", 2: "uuid-bbb"}
    result = _parse_provenance("Statement one [1] and two [2].", uuid_map)
    assert result == ["uuid-aaa", "uuid-bbb"]


def test_parse_provenance_deduplicates() -> None:
    uuid_map = {1: "uuid-aaa"}
    result = _parse_provenance("First [1] and again [1].", uuid_map)
    assert result == ["uuid-aaa"]


def test_parse_provenance_unknown_index_ignored() -> None:
    uuid_map = {1: "uuid-aaa"}
    result = _parse_provenance("Known [1] and unknown [99].", uuid_map)
    assert result == ["uuid-aaa"]


def test_parse_provenance_empty_text() -> None:
    result = _parse_provenance("", {1: "uuid-aaa"})
    assert result == []


# ---------------------------------------------------------------------------
# _parse_structured_response unit tests
# ---------------------------------------------------------------------------

def test_parse_structured_single_claim() -> None:
    text = "[CLAIM] Beethoven was born in 1770 [1]. [LABEL: REFUTES]"
    parsed = _parse_structured_response(text)
    assert len(parsed) == 1
    assert "Beethoven was born" in parsed[0].text
    assert parsed[0].label == "REFUTES"
    assert parsed[0].addressed == []


def test_parse_structured_multi_claim() -> None:
    text = (
        "[CLAIM] First assertion [1]. [LABEL: SUPPORTS]\n"
        "[CLAIM] Second assertion. [LABEL: REFUTES]"
    )
    parsed = _parse_structured_response(text)
    assert len(parsed) == 2
    assert parsed[0].label == "SUPPORTS"
    assert parsed[1].label == "REFUTES"


def test_parse_structured_no_addressed() -> None:
    text = "[CLAIM] Independent claim. [LABEL: NOT ENOUGH INFO]"
    parsed = _parse_structured_response(text)
    assert len(parsed) == 1
    assert parsed[0].addressed == []
    assert parsed[0].label == "NOT ENOUGH INFO"


def test_parse_structured_nei_label() -> None:
    text = "[CLAIM] No evidence found. [LABEL: NOT ENOUGH INFO]"
    parsed = _parse_structured_response(text)
    assert parsed[0].label == "NOT ENOUGH INFO"


def test_parse_structured_missing_claims_returns_empty() -> None:
    parsed = _parse_structured_response("No structured tags here at all.")
    assert parsed == []


def test_parse_structured_addressed_entries() -> None:
    text = "[CLAIM] I disagree. [ADDRESSED:c1][DISAGREE] You are wrong [1]. [LABEL: REFUTES]"
    parsed = _parse_structured_response(text)
    assert len(parsed) == 1
    assert parsed[0].addressed == [("c1", "DISAGREE")]


def test_parse_structured_agree_entry() -> None:
    text = "[CLAIM] I agree. [ADDRESSED:c2][AGREE] Correct per [1]. [LABEL: SUPPORTS]"
    parsed = _parse_structured_response(text)
    assert parsed[0].addressed == [("c2", "AGREE")]


# ---------------------------------------------------------------------------
# run_debate — edges from structured responses
# ---------------------------------------------------------------------------

_RESP_A_NO_ADDR = "[CLAIM] First claim [1]. [LABEL: SUPPORTS]"
_RESP_B_AGREE = "[CLAIM] I agree. [ADDRESSED:c1][AGREE] Correct [1]. [LABEL: SUPPORTS]"
_RESP_B_DISAGREE = "[CLAIM] I disagree. [ADDRESSED:c1][DISAGREE] Wrong [1]. [LABEL: REFUTES]"
_RESP_B_NO_ADDR = "[CLAIM] New independent point. [LABEL: NOT ENOUGH INFO]"


def test_run_debate_agree_edge() -> None:
    nodes, edges = _patched_run([_RESP_A_NO_ADDR, _RESP_B_AGREE])
    assert len(edges) == 1
    assert edges[0].sign == "+"


def test_run_debate_disagree_edge() -> None:
    nodes, edges = _patched_run([_RESP_A_NO_ADDR, _RESP_B_DISAGREE])
    assert len(edges) == 1
    assert edges[0].sign == "-"


def test_run_debate_no_addressed_no_edge() -> None:
    nodes, edges = _patched_run([_RESP_A_NO_ADDR, _RESP_B_NO_ADDR])
    assert edges == []


def test_run_debate_returns_multiple_claim_nodes() -> None:
    resp_a_two = (
        "[CLAIM] First claim. [LABEL: SUPPORTS]\n"
        "[CLAIM] Second claim. [LABEL: REFUTES]"
    )
    resp_b_two = (
        "[CLAIM] Third claim. [LABEL: NOT ENOUGH INFO]\n"
        "[CLAIM] Fourth claim. [ADDRESSED:c1][AGREE] Yes [1]. [LABEL: SUPPORTS]"
    )
    nodes, _ = _patched_run([resp_a_two, resp_b_two], triples_a=_triples(1), triples_b=_triples(1))
    assert len(nodes) == 4


def test_run_debate_alternates_opener_round0() -> None:
    nodes, _ = _patched_run([_RESP_A_NO_ADDR, _RESP_B_NO_ADDR], round_idx=0)
    assert nodes[0].expert_id == "expert_a"


def test_run_debate_alternates_opener_round1() -> None:
    nodes, _ = _patched_run([_RESP_A_NO_ADDR, _RESP_B_NO_ADDR], round_idx=1)
    assert nodes[0].expert_id == "expert_b"


def test_run_debate_conflict_dedup_prefers_disagree() -> None:
    # One claim agrees with c1, another disagrees with c1 in the same turn.
    resp_b_both = (
        "[CLAIM] I agree. [ADDRESSED:c1][AGREE] Yes. [LABEL: SUPPORTS]\n"
        "[CLAIM] But also disagree. [ADDRESSED:c1][DISAGREE] No. [LABEL: REFUTES]"
    )
    nodes, edges = _patched_run([_RESP_A_NO_ADDR, resp_b_both])
    # Only one edge for the (b_node_1, c1_uuid) pair — should prefer "-"
    c1_target = nodes[0].id  # c1 is the first node from A
    relevant = [e for e in edges if e.target_id == c1_target]
    signs = {e.sign for e in relevant}
    assert "-" in signs


def test_short_ids_sequential() -> None:
    resp_a_two = (
        "[CLAIM] Claim one. [LABEL: SUPPORTS]\n"
        "[CLAIM] Claim two. [LABEL: REFUTES]"
    )
    nodes, _ = _patched_run([resp_a_two, _RESP_B_NO_ADDR])
    short_ids = [n.short_id for n in nodes]
    assert short_ids[0] == "c1"
    assert short_ids[1] == "c2"
    assert short_ids[2] == "c3"  # B's first claim, sees 2 history nodes


def test_node_label_populated() -> None:
    nodes, _ = _patched_run([_RESP_A_NO_ADDR, _RESP_B_DISAGREE])
    a_node = next(n for n in nodes if n.expert_id == "expert_a")
    b_node = next(n for n in nodes if n.expert_id == "expert_b")
    assert a_node.label == "SUPPORTS"
    assert b_node.label == "REFUTES"


def test_run_debate_cross_round_edge() -> None:
    """B in round 1 can address A's round-0 claim via history."""
    # Round 0: A makes c1, B makes c2 (no address)
    r0_a = "[CLAIM] Round zero A claim. [LABEL: SUPPORTS]"
    r0_b = "[CLAIM] Round zero B claim. [LABEL: NOT ENOUGH INFO]"
    cfg = _cfg()
    with patch("debate_kg.debate.orchestrator._make_client", return_value=_mock_client([r0_a, r0_b])):
        nodes_r0, _ = run_debate("Claim.", [], [], cfg, round_idx=0, history=[])

    # Round 1: B opens (odd round), A responds — A addresses c1 from round 0
    c1_id = nodes_r0[0].short_id  # "c1"
    r1_b = "[CLAIM] Round one B new. [LABEL: NOT ENOUGH INFO]"
    r1_a = f"[CLAIM] Addressing round-zero claim. [ADDRESSED:{c1_id}][AGREE] Yes. [LABEL: SUPPORTS]"
    with patch("debate_kg.debate.orchestrator._make_client", return_value=_mock_client([r1_b, r1_a])):
        nodes_r1, edges_r1 = run_debate("Claim.", [], [], cfg, round_idx=1, history=nodes_r0)

    assert len(edges_r1) == 1
    assert edges_r1[0].sign == "+"
    assert edges_r1[0].target_id == nodes_r0[0].id  # points at c1


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------

def test_provenance_extracted_from_citations() -> None:
    ts = _triples(2)
    resp = "[CLAIM] Claim citing [2]. [LABEL: SUPPORTS]"
    nodes, _ = _patched_run([resp, _RESP_B_NO_ADDR], triples_a=ts)
    a_node = next(n for n in nodes if n.expert_id == "expert_a")
    assert ts[1].uuid in a_node.provenance


def test_provenance_empty_when_no_citations() -> None:
    ts = _triples(2)
    resp = "[CLAIM] No citations here. [LABEL: NOT ENOUGH INFO]"
    nodes, _ = _patched_run([resp, _RESP_B_NO_ADDR], triples_a=ts)
    a_node = next(n for n in nodes if n.expert_id == "expert_a")
    assert a_node.provenance == []


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------

def test_all_templates_load_without_error() -> None:
    for name in ("system.j2", "expert_turn.j2"):
        tpl = _JINJA_ENV.get_template(name)
        assert tpl is not None


def test_system_template_includes_claim() -> None:
    rendered = _JINJA_ENV.get_template("system.j2").render(claim="The Moon is large.")
    assert "The Moon is large." in rendered


def test_expert_turn_template_renders_history_with_short_ids() -> None:
    node = _node("c1", expert_id="expert_a", round_idx=0)
    node.text = "hello"
    rendered = _JINJA_ENV.get_template("expert_turn.j2").render(triples=[], history=[node])
    assert "c1" in rendered
    assert "hello" in rendered


def test_expert_turn_template_triple_numbering() -> None:
    ts = _triples(3)
    rendered = _JINJA_ENV.get_template("expert_turn.j2").render(triples=ts, history=[])
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "[3]" in rendered


def test_run_single_expert_returns_nodes() -> None:
    cfg = _cfg()
    resp = "[CLAIM] Single expert claim [1]. [LABEL: SUPPORTS]"
    with patch("debate_kg.debate.orchestrator._make_client", return_value=_mock_client([resp])):
        nodes, edges = run_single_expert("Claim.", _triples(1), cfg)
    assert len(nodes) >= 1
    assert edges == []
    assert nodes[0].expert_id == "expert_a"
