"""Tests for the consensus check (Module 5).

Pure logic — no mocks needed.

Test matrix
-----------
  All-positive consensus:
    test_all_positive_single_edge_is_consensus
    test_all_positive_multiple_edges_is_consensus

  Any negative → no consensus:
    test_any_negative_is_not_consensus
    test_all_negative_is_not_consensus
    test_mixed_edges_is_not_consensus

  Empty edges → no consensus (conservative):
    test_empty_edges_is_not_consensus

  Hard cap:
    test_hard_cap_at_last_round
    test_hard_cap_overrides_negative_edges
    test_before_last_round_negative_continues

  Return type (legacy):
    test_is_consensus_returns_bool
    test_is_consensus_with_edges_returns_bool
"""
from __future__ import annotations

from unittest.mock import MagicMock

from breastMnist.src.debate_kg.consensus.check import is_consensus
from breastMnist.src.debate_kg.kg.schema import DebateEdge


def _cfg(max_rounds: int = 5) -> object:
    cfg = MagicMock()
    cfg.run.max_rounds = max_rounds
    return cfg


def _edge(sign: str) -> DebateEdge:
    return DebateEdge(source_id="n1", target_id="n2", sign=sign)  # type: ignore[arg-type]  # noqa: PYI041


# ---------------------------------------------------------------------------
# All-positive consensus
# ---------------------------------------------------------------------------

def test_all_positive_single_edge_is_consensus() -> None:
    assert is_consensus([_edge("+")], round_idx=0, cfg=_cfg()) is True


def test_all_positive_multiple_edges_is_consensus() -> None:
    edges = [_edge("+"), _edge("+"), _edge("+")]
    assert is_consensus(edges, round_idx=0, cfg=_cfg()) is True


# ---------------------------------------------------------------------------
# Any negative → no consensus
# ---------------------------------------------------------------------------

def test_any_negative_is_not_consensus() -> None:
    edges = [_edge("+"), _edge("-")]
    assert is_consensus(edges, round_idx=0, cfg=_cfg()) is False


def test_all_negative_is_not_consensus() -> None:
    assert is_consensus([_edge("-")], round_idx=0, cfg=_cfg()) is False


def test_mixed_edges_is_not_consensus() -> None:
    edges = [_edge("+"), _edge("-"), _edge("+")]
    assert is_consensus(edges, round_idx=1, cfg=_cfg()) is False


# ---------------------------------------------------------------------------
# Empty edges → no consensus (conservative)
# ---------------------------------------------------------------------------

def test_empty_edges_is_not_consensus() -> None:
    assert is_consensus([], round_idx=0, cfg=_cfg()) is False


# ---------------------------------------------------------------------------
# Hard cap
# ---------------------------------------------------------------------------

def test_hard_cap_at_last_round() -> None:
    # round_idx == max_rounds - 1 → stop regardless
    assert is_consensus([_edge("-")], round_idx=4, cfg=_cfg(max_rounds=5)) is True


def test_hard_cap_overrides_negative_edges() -> None:
    edges = [_edge("-"), _edge("-")]
    assert is_consensus(edges, round_idx=2, cfg=_cfg(max_rounds=3)) is True


def test_before_last_round_negative_continues() -> None:
    # round_idx 3 < max_rounds-1=4 → cap not hit; negative edge → False
    assert is_consensus([_edge("-")], round_idx=3, cfg=_cfg(max_rounds=5)) is False


def test_hard_cap_with_one_round() -> None:
    # max_rounds=1: round_idx=0 == max_rounds-1=0 → always stops
    assert is_consensus([_edge("-")], round_idx=0, cfg=_cfg(max_rounds=1)) is True


# ---------------------------------------------------------------------------
# Legacy smoke tests (unchanged interface, still pass)
# ---------------------------------------------------------------------------

def test_is_consensus_returns_bool() -> None:
    result = is_consensus([], round_idx=0, cfg=_cfg())
    assert isinstance(result, bool)


def test_is_consensus_with_edges_returns_bool() -> None:
    edges = [_edge("+"), _edge("-")]
    result = is_consensus(edges, round_idx=0, cfg=_cfg())
    assert isinstance(result, bool)
