"""Smoke tests for the data module."""
import pytest
from omegaconf import OmegaConf

from breastMnist.src.debate_kg.data.fever import load_claims


def test_load_claims_stub_returns_one_claim() -> None:
    cfg = OmegaConf.create({"run": {"use_stub_data": True, "num_claims": 1}})
    claims = load_claims(cfg)
    assert len(claims) == 1
    assert "id" in claims[0]
    assert "claim" in claims[0]
    assert "label" in claims[0]


def test_load_claims_stub_label_is_valid() -> None:
    cfg = OmegaConf.create({"run": {"use_stub_data": True, "num_claims": 1}})
    claims = load_claims(cfg)
    assert claims[0]["label"] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


@pytest.mark.network
def test_load_claims_real_fever_returns_list() -> None:
    cfg = OmegaConf.create({"run": {"use_stub_data": False, "num_claims": 3}})
    claims = load_claims(cfg)
    assert len(claims) == 3
    for c in claims:
        assert "claim" in c and "label" in c
