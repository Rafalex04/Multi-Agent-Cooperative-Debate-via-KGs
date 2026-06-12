"""Ollama HTTP client — used for local dev and single-GPU testing."""
from __future__ import annotations

import logging
import os
import subprocess
import time

import requests

from debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)

# Backoff schedule (seconds) for transient Ollama failures — typically GPU OOM
# from other cluster jobs competing for VRAM, which can crash the Ollama
# server itself. ~30 min cumulative wait before giving up.
_RETRY_DELAYS_S = [30, 60, 120, 240, 300, 300, 300, 300]

# TODO(ollama_client): make this configurable via cfg instead of an env var.
_OLLAMA_RESTART_SCRIPT = os.environ.get("OLLAMA_RESTART_SCRIPT", "/data/rm2125/start_ollama.sh")


class OllamaClient(LLMClient):
    """Wraps the Ollama /api/chat endpoint."""

    def __init__(self, model: str, base_url: str, num_ctx: int = 32768) -> None:
        super().__init__(model=model, base_url=base_url)
        self.num_ctx = num_ctx

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
            requests.RequestException: If Ollama still fails after retrying
                through ``_RETRY_DELAYS_S`` (restarting the server on
                connection failures).
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
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        logger.debug("OllamaClient → %s (temp=%.2f, system=%s)", self.model, temperature, bool(system))

        last_exc: requests.exceptions.RequestException | None = None
        response: requests.Response | None = None
        for attempt, delay in enumerate([0, *_RETRY_DELAYS_S]):
            if delay:
                time.sleep(delay)
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=1800,
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "OllamaClient: request failed (attempt %d/%d): %s — restarting Ollama",
                    attempt + 1, len(_RETRY_DELAYS_S) + 1, exc,
                )
                subprocess.run(["bash", _OLLAMA_RESTART_SCRIPT], check=False)
                continue
            if response.status_code >= 500:
                logger.warning(
                    "OllamaClient: %d error from Ollama (attempt %d/%d): %s",
                    response.status_code, attempt + 1, len(_RETRY_DELAYS_S) + 1, response.text[:300],
                )
                continue
            response.raise_for_status()
            return response.json()["message"]["content"]

        if response is None:
            raise last_exc
        response.raise_for_status()
        return response.json()["message"]["content"]
