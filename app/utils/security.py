"""Input validation, prompt-injection detection, secret redaction, friendly errors."""
from __future__ import annotations

import re

from app.config import ConfigError, get_settings

MAX_INPUT_CHARS = 4000
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_user_input(text: str | None) -> str:
    """Strip control chars and enforce length. Raises ValueError with a user-safe message."""
    text = _CTRL.sub("", text or "").strip()
    if not text:
        raise ValueError("Please type a message.")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Your message is too long (max {MAX_INPUT_CHARS} characters).")
    return text


_INJECTION = [
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|your)\s+(instructions?|prompts?|rules)",
    r"(reveal|show|print|display|repeat|leak|tell me)\s+(me\s+)?(your|the)\s+(system|hidden|initial)\s+(prompt|instructions?|message)",
    r"(reveal|show|print|display|leak|give me|tell me|what is|what's)\s+(me\s+)?your\s+(api[\s_-]?keys?|secret|tokens?|passwords?|credentials)",
    r"\breveal\b.{0,40}\b(api[\s_-]?keys?|secrets?|passwords?|\.env)\b",
    r"you\s+are\s+now\s+(in\s+)?(dan|developer mode|jailbroken)",
    r"(cat|type|print|read)\s+.{0,20}\.env\b",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text or "") for p in _INJECTION_RE)


_SECRET_RE = [
    re.compile(p)
    for p in (
        r"sk-or-v1-[A-Za-z0-9]{16,}",
        r"sk-[A-Za-z0-9_\-]{20,}",
        r"gsk_[A-Za-z0-9]{20,}",
        r"gh[pousr]_[A-Za-z0-9]{20,}",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"AIza[0-9A-Za-z_\-]{30,}",
        r"ya29\.[0-9A-Za-z_\-]{20,}",
        r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}",
    )
]


def contains_secret(text: str) -> bool:
    if any(p.search(text or "") for p in _SECRET_RE):
        return True
    try:
        return any(v in (text or "") for v in get_settings().secret_values())
    except ConfigError:
        return False


def redact_secrets(text: str) -> str:
    out = text or ""
    for p in _SECRET_RE:
        out = p.sub("[REDACTED]", out)
    try:
        for v in get_settings().secret_values():
            out = out.replace(v, "[REDACTED]")
    except ConfigError:
        pass
    return out


_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}$")


def is_valid_email(addr: str) -> bool:
    return bool(_EMAIL_RE.match((addr or "").strip()))


def invalid_addresses(args: dict) -> list[str]:
    """Return invalid e-mail addresses found in recipient-like tool arguments."""
    bad: list[str] = []
    for key in ("to", "cc", "bcc", "recipient", "recipients"):
        val = args.get(key)
        if val is None:
            continue
        items = val if isinstance(val, list) else re.split(r"[;,]", str(val))
        bad += [str(a).strip() for a in items if str(a).strip() and not is_valid_email(str(a))]
    return bad


def friendly_error(exc: BaseException) -> str:
    """Map any exception to a message that is safe to show (no secrets, no traces)."""
    if isinstance(exc, ConfigError):
        return str(exc)
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "429" in msg or "rate limit" in msg or "rate-limit" in msg:
        return ("The AI service is rate-limiting requests (free tiers have strict limits). "
                "Wait a minute and retry, or switch LLM_PROVIDER in .env.")
    if "authentication" in name or "401" in msg or "invalid api key" in msg or "unauthorized" in msg:
        return "A service rejected the credentials. Check the API key / token in your .env file."
    if "403" in msg or "permission" in name or "forbidden" in msg or "permission" in msg:
        return "Permission denied. Check that your token/OAuth scopes allow this action."
    if any(k in name or k in msg for k in ("timeout", "connect", "network", "dns", "unreachable")):
        return "Network problem while contacting a service. Check your internet connection and retry."
    return "Something went wrong while handling that request. Details were written to logs/app.log."
