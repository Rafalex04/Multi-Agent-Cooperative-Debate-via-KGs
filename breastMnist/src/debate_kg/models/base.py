"""Abstract base class for LLM serving backends.

All LLM calls must go through a subclass of LLMClient. Never call Ollama or
vLLM HTTP endpoints directly from module code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract LLM backend. Subclasses wrap Ollama, vLLM, etc."""

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        image: str | None = None,
    ) -> str:
        """Send a completion request and return the response text.

        Args:
            prompt: User message content.
            system: System prompt (empty string → no system message sent).
            temperature: Sampling temperature.
            image: Optional base64-encoded JPEG/PNG image string for vision models.
        """
        ...
