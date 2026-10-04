"""Structured logging with loguru, with secret redaction.

Every log message passes through a filter that replaces any known secret value
(API keys loaded from the environment) with ``***``, so a key accidentally
interpolated into a message never reaches the log sink.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | {message}{extra[extra_str]}"
)


class SecretRedactor:
    """loguru filter that masks secret values in messages and bound extras."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets: list[str] = []
        self.update(secrets)

    def update(self, secrets: Iterable[str]) -> None:
        merged = {*self._secrets, *(s for s in secrets if s and len(s) >= 6)}
        # Longest first so overlapping secrets are fully masked.
        self._secrets = sorted(merged, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, "***")
        return text

    def __call__(self, record: Record) -> bool:
        record["message"] = self.redact(record["message"])
        extra = record["extra"]
        for key, value in list(extra.items()):
            if isinstance(value, str):
                extra[key] = self.redact(value)
        extras = {k: v for k, v in extra.items() if not k.startswith("_")}
        extra["extra_str"] = (
            " | " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        )
        return True


_redactor = SecretRedactor()


def configure_logging(level: str | None = None, secrets: Iterable[str] = ()) -> SecretRedactor:
    """(Re)configure the global loguru logger. Safe to call more than once."""
    _redactor.update(secrets)
    level = (level or os.environ.get("TRADESENSE_LOG_LEVEL") or "INFO").upper()
    # Windows consoles redirected to a file may use a legacy code page.
    reconfigure: Any = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(OSError, ValueError):
            reconfigure(errors="backslashreplace")
    logger.remove()
    logger.add(sys.stderr, level=level, format=_FORMAT, filter=_redactor, backtrace=False)
    # Third-party request logs would include URLs with the account id; keep them quiet.
    for noisy in ("httpx", "httpcore", "anthropic", "openai", "google_genai", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return _redactor
