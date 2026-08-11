"""vLLM OpenAI-compatible client — used on the university cluster."""
from __future__ import annotations

import logging

import requests

from breastMnist.src.debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """Wraps vLLM's OpenAI-compatible /v1/chat/completions endpoint."""

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        image: str | None = None,
    ) -> str:
        """Send a chat completion to vLLM and return the assistant text.

        Args:
            prompt: User message content.
            system: System prompt (empty string → no system message).
            temperature: Sampling temperature.
            image: Optional base64-encoded image string. When provided, the user
                   message uses the OpenAI vision content format so vLLM passes
                   the image to vision-capable models (e.g. MedGemma-4B-IT).

        Returns:
            Response text from the model.

        Raises:
            requests.RequestException: On non-2xx response or connection failure.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        if image:
            user_content: list[dict] | str = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
            ]
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }
        logger.debug("VLLMClient → %s (temp=%.2f, image=%s)", self.model, temperature, image is not None)
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
