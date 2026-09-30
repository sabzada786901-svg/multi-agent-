"""Output guardrail: enforce grounding result, block prompt leaks, redact secrets."""
from __future__ import annotations

from app.guardrails.messages import UNSAFE_MESSAGE, UNSUPPORTED_MESSAGE
from app.utils.security import redact_secrets

SYSTEM_PROMPT_MARKERS = (
    "ABSOLUTE RULES",
    "<document_data>",
    "You are the strict PDF question-answering component",
    "You are the input guardrail/router",
)


def apply_output_guardrail(text: str, grounding_result: dict | None = None) -> str:
    text = text or ""
    # 1. A failed grounding check can never be overridden by later composition.
    if grounding_result and not (grounding_result.get("grounded") and grounding_result.get("supported")):
        if UNSUPPORTED_MESSAGE not in text and "couldn't find" not in text.lower():
            return UNSUPPORTED_MESSAGE
    # 2. Never leak the system prompt.
    if any(m in text for m in SYSTEM_PROMPT_MARKERS):
        return UNSAFE_MESSAGE
    # 3. Never leak credentials.
    return redact_secrets(text)
