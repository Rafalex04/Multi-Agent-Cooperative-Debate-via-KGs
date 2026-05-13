"""Stub LLM client — returned when cfg.run.use_stub_data=True.

Returns canned responses that keep the smoke pipeline functional without
requiring a live Ollama instance. Not used in any real run.
"""
from __future__ import annotations

from debate_kg.models.base import LLMClient


class StubClient(LLMClient):
    """Returns predictable canned responses for smoke/unit-test runs."""

    def complete(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        if system:
            # Expert turns: return two [CLAIM] blocks in the required structured format.
            # [1] citation ensures provenance parsing finds at least one triple reference.
            return (
                "[CLAIM] Based on the available evidence, the claim is supported [1]. [LABEL: SUPPORTS]\n"
                "[CLAIM] I lack sufficient evidence to assess this aspect. [LABEL: NOT ENOUGH INFO]"
            )
        # Verdict classifier and judge scoring calls (no system prompt):
        # "SUPPORTS" → verdict parsed as "SUPPORTS"; judge rubric looks for GROUNDEDNESS/FACTUALITY.
        return "SUPPORTS\nGROUNDEDNESS: 50\nFACTUALITY: 50\nOVERALL: 50"
