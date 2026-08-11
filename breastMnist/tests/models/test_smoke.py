"""Smoke tests for the models module."""
import pytest

from breastMnist.src.debate_kg.models.ollama_client import OllamaClient
from breastMnist.src.debate_kg.models.stub_client import StubClient
from breastMnist.src.debate_kg.models.vllm_client import VLLMClient


@pytest.mark.network
def test_ollama_client_complete_returns_string() -> None:
    client = OllamaClient(model="llama3.1:8b", base_url="http://localhost:11434")
    result = client.complete("Hello", system="", temperature=0.7)
    assert isinstance(result, str)


@pytest.mark.network
def test_vllm_client_complete_returns_string() -> None:
    client = VLLMClient(model="llama3.1:8b", base_url="http://localhost:8000")
    result = client.complete("Hello", system="", temperature=0.7)
    assert isinstance(result, str)


def test_ollama_client_inherits_model_and_url() -> None:
    client = OllamaClient(model="qwen2.5:7b", base_url="http://host:11434")
    assert client.model == "qwen2.5:7b"
    assert client.base_url == "http://host:11434"


def test_stub_client_with_system_returns_citation() -> None:
    client = StubClient(model="stub", base_url="")
    result = client.complete("prompt", system="You are an expert.", temperature=0.0)
    assert "[1]" in result


def test_stub_client_without_system_returns_classification() -> None:
    client = StubClient(model="stub", base_url="")
    result = client.complete("classify this", system="", temperature=0.0)
    assert "AGREE" in result or "SUPPORTS" in result
