"""FEVER dataset loader via HuggingFace datasets.

When cfg.run.use_stub_data is True (default for run=smoke), returns a hardcoded
claim so the smoke run needs no network access. Set to False for real runs.

Uses copenlu/fever_gold_evidence (parquet, no custom script) instead of the
original fever/fever repo which requires a legacy loading script no longer
supported by datasets>=3.0.
"""
from __future__ import annotations

import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

_STUB_CLAIM = {
    "id": "stub-0",
    "claim": "The sky is blue.",
    "label": "SUPPORTS",
}


def load_claims(cfg: DictConfig) -> list[dict]:
    """Load FEVER claims from HuggingFace datasets (or stub data for smoke runs).

    Args:
        cfg: Hydra config. Reads cfg.run.use_stub_data and cfg.run.num_claims.

    Returns:
        List of dicts with keys: id (str), claim (str), label (str).
        Labels are SUPPORTS / REFUTES / NOT ENOUGH INFO.
    """
    if cfg.run.use_stub_data:
        logger.info("load_claims: using stub data (1 hardcoded claim)")
        return [_STUB_CLAIM]

    from datasets import load_dataset  # noqa: PLC0415

    logger.info("load_claims: loading %d FEVER claims from HuggingFace", cfg.run.num_claims)
    ds = load_dataset("copenlu/fever_gold_evidence", split="validation")
    rows = ds.select(range(cfg.run.num_claims))
    return [
        {
            "id": str(r["id"]),
            "claim": r["claim"],
            "label": str(r["label"]),  # already a string: SUPPORTS/REFUTES/NOT ENOUGH INFO
        }
        for r in rows
    ]
