"""Consensus check: decide whether to stop the round loop.

Module 5 in the pipeline. Pure logic — no learning, no LLM calls.
See SPEC §Module5.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

from breastMnist.src.debate_kg.kg.schema import DebateEdge

logger = logging.getLogger(__name__)


def is_consensus(edges: list[DebateEdge], round_idx: int, cfg: DictConfig) -> bool:
    """Return True if the debate has reached consensus or the round cap is hit.

    Consensus = all debate edges are positive (sign == "+").
    Also returns True if round_idx >= cfg.run.max_rounds - 1 (hard cap).

    Args:
        edges: All signed edges in the current debate graph.
        round_idx: Zero-based index of the round just completed.
        cfg: Hydra config. Reads cfg.run.max_rounds.

    Returns:
        True → stop the loop. False → continue to merger and next round.
    """
    if round_idx >= cfg.run.max_rounds - 1:
        logger.info("is_consensus: hard cap reached (round=%d, max_rounds=%d)", round_idx, cfg.run.max_rounds)
        return True
    if round_idx >= 1 and edges and all(e.sign != "-" for e in edges):
        logger.info("is_consensus: no disagreement edges at round %d (%d edge(s))", round_idx, len(edges))
        return True
    logger.debug("is_consensus: continuing (round=%d, edges=%d)", round_idx, len(edges))
    return False
