"""
Guardrails — Phase 10b: PII redaction + prompt-injection detection.

Every piece of human-supplied text (JD, profile, interview answers) is
untrusted input, exactly like a request body hitting your service. Two
deterministic, regex-based checks sit at the boundary before that text is
stored in state or sent to a model:

  redact_pii(text)      -> text with emails/phones/long digit runs masked out
  detect_injection(text) -> True if the text looks like it's trying to
                            hijack the system prompt ("ignore previous
                            instructions", "you are now", ...)

Both are pure functions — no model call, so they're free and instant, and
fully covered by tests/test_guardrails.py without touching the network.
"""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Formatted phone numbers / long digit-with-separators runs, e.g.
# "415-555-0142", "(415) 555 0142" — at least 10 characters total so it
# doesn't accidentally catch short numbers like "8/10" or "2026".
_PHONE_RE = re.compile(r"\+?\d[\d\-.\s()]{8,}\d")

# Bare digit runs of 9+ (SSNs, card numbers) not already caught by _PHONE_RE.
_LONG_DIGIT_RE = re.compile(r"\b\d{9,}\b")

# Heuristic prompt-injection patterns — phrases that try to override the
# system prompt or extract it. Not exhaustive; a seam for an LLM-based check
# (e.g. "ask a model whether this looks like an injection attempt") if the
# regex approach proves too easy to evade.
_INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.I),
    re.compile(r"disregard (all |any )?(previous|prior|above)", re.I),
    re.compile(r"you are now\b", re.I),
    re.compile(r"new system prompt", re.I),
    re.compile(r"reveal your (system prompt|instructions)", re.I),
    re.compile(r"act as (?:if you (?:are|were)|a) (?:different|another)", re.I),
]


def redact_pii(text: str) -> str:
    """Mask emails, phone numbers, and long digit runs in `text`.

    Order matters: email first (has its own delimiters), then phone-shaped
    runs (may include bare 10+ digit numbers), then any remaining bare 9+
    digit run (e.g. a 9-digit SSN with no separators).
    """
    if not text:
        return text
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = _LONG_DIGIT_RE.sub("[REDACTED_NUMBER]", text)
    return text


def detect_injection(text: str) -> bool:
    """Return True if `text` matches a known prompt-injection pattern."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)
