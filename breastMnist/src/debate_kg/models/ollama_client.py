"""Ollama HTTP client — used for local dev and single-GPU testing."""
from __future__ import annotations

import logging
import os
import subprocess
import time

import requests

from breastMnist.src.debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)

# Backoff schedule for transient Ollama failures (connection errors, 5xx).
_RETRY_DELAYS_S = [30, 60, 120, 240, 300, 300, 300, 300]

# For RAM-OOM errors ("model requires more system memory"), retry every
# 10 minutes for up to 12 hours — waits for system RAM to free up.
_OOM_RETRY_DELAY_S = 600
_OOM_MAX_RETRIES = 72

# TODO(ollama_client): make this configurable via cfg instead of an env var.
_OLLAMA_RESTART_SCRIPT = os.environ.get("OLLAMA_RESTART_SCRIPT")


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

        def _post() -> requests.Response:
            return requests.post(f"{self.base_url}/api/chat", json=payload, timeout=1800)

        # OOM loop: wait up to 12 h for system RAM to free up.
        for oom_attempt in range(_OOM_MAX_RETRIES + 1):
            try:
                response = _post()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                response = None
                break  # fall through to standard retry loop
            if response.status_code >= 500 and "model requires more system memory" in response.text:
                if oom_attempt >= _OOM_MAX_RETRIES:
                    logger.error("OllamaClient: OOM retries exhausted (%d attempts)", oom_attempt + 1)
                    response.raise_for_status()
                logger.warning(
                    "OllamaClient: OOM (attempt %d/%d, retrying in %ds): %s",
                    oom_attempt + 1, _OOM_MAX_RETRIES, _OOM_RETRY_DELAY_S, response.text[:200],
                )
                time.sleep(_OOM_RETRY_DELAY_S)
                continue
            # Non-OOM response (success or other error) — exit OOM loop
            break

        # If we got a successful response from the OOM loop, return it.
        if response is not None and response.status_code < 500:
            response.raise_for_status()
            return response.json()["message"]["content"]

        # Standard retry loop for connection errors and non-OOM 5xx.
        for attempt, delay in enumerate([0, *_RETRY_DELAYS_S]):
            if delay:
                time.sleep(delay)
            try:
                response = _post()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    "OllamaClient: request failed (attempt %d/%d): %s — restarting Ollama",
                    attempt + 1, len(_RETRY_DELAYS_S) + 1, exc,
                )
                if _OLLAMA_RESTART_SCRIPT:
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
