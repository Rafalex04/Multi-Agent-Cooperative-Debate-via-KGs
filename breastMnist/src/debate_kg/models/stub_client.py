"""Stub LLM client — returned when cfg.run.use_stub_data=True.

Returns canned responses that keep the smoke pipeline functional without
requiring a live Ollama instance. Not used in any real run.
"""
from __future__ import annotations

from breastMnist.src.debate_kg.models.base import LLMClient


class StubClient(LLMClient):
    """Returns predictable canned responses for smoke/unit-test runs."""

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        image: str | None = None,
    ) -> str:
        # BreastMNIST single-agent mode: system asks for [LABEL][CONFIDENCE] only.
        if system and "CONFIDENCE" in system:
            return "[LABEL: BENIGN][CONFIDENCE: 75]"

        if system and image is not None:
            # BreastMNIST structured debate expert turn.
            return (
                "[CLAIM] The mass has a circumscribed margin suggesting a benign lesion. [LABEL: BENIGN]\n"
                "[CLAIM] Posterior enhancement is present indicating acoustic transmission. [LABEL: BENIGN]"
            )

        if system:
            # FEVER expert turns: two [CLAIM] blocks with provenance citations.
            return (
                "[CLAIM] Based on the available evidence, the claim is supported [1]. [LABEL: SUPPORTS]\n"
                "[CLAIM] I lack sufficient evidence to assess this aspect. [LABEL: NOT ENOUGH INFO]"
            )

        # Judge / verdict calls (no system prompt). "SUPPORTS" prefix keeps FEVER test assertions passing.
        return (
            "SUPPORTS\n"
            "IMAGE_GROUNDING: 50\nMEDICAL_ACCURACY: 50\n"
            "GROUNDEDNESS: 50\nFACTUALITY: 50\nOVERALL: 50"
        )
