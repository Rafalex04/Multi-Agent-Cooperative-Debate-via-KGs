"""Ollama HTTP client — used for local dev and single-GPU testing."""
from __future__ import annotations

import logging

import requests

from debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Wraps the Ollama /api/chat endpoint."""

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        image: str | None = None,
    ) -> str:
        """Send a chat completion to Ollama and return the assistant text.

        Args:
            prompt: User message content.
            system: System prompt (empty string → no system message sent).
            temperature: Sampling temperature.
            image: Optional base64-encoded image string. Passed in the Ollama
                   ``images`` field for vision-capable models (e.g. Qwen2.5-VL).

        Returns:
            Response text from the model.

        Raises:
            requests.RequestException: If the Ollama server is unreachable or returns an error.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        user_msg: dict = {"role": "user", "content": prompt}
        if image:
            user_msg["images"] = [image]
        messages.append(user_msg)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.debug("OllamaClient → %s (temp=%.2f, system=%s)", self.model, temperature, bool(system))
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
