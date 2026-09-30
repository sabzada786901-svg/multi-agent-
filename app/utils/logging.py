"""File logging with automatic secret redaction. Console stays quiet (warnings only)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from app.config import ROOT
from app.utils.security import redact_secrets

_configured = False


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return redact_secrets(super().format(record))


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    (ROOT / "logs").mkdir(exist_ok=True)
    handler = RotatingFileHandler(ROOT / "logs" / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("app")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"app.{name}")
