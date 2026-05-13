"""vLLM OpenAI-compatible client — used on the university cluster."""
from __future__ import annotations

import logging

from debate_kg.models.base import LLMClient

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """Wraps vLLM's OpenAI-compatible /v1/chat/completions endpoint."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """Send a chat completion to vLLM and return the assistant text.

        Args:
            prompt: User message content.
            system: System prompt (empty string → no system message).
            temperature: Sampling temperature.

        Returns:
            Response text from the model.
        """
        # TODO(models, ref=SPEC §Stack): POST to {self.base_url}/v1/chat/completions
        # with model=self.model, messages, temperature; parse choices[0].message.content.
        # Use requests.post; raise on non-2xx.
        logger.debug("VLLMClient.complete — stub")
        return "[stub response]"
