"""Structured JSON logging configuration — melhoria #43.

Usage
-----
    from hpe.core.logging_config import configure_logging
    configure_logging(level="INFO", json_format=True)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """Formatter que emite cada log record como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Extra context fields (added via logger.info(..., extra={...}))
        skip_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "asctime", "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip_keys and not k.startswith("_"):
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)

        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Formatter texto colorido para dev (mantém compat com logs anteriores)."""

    COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.utcfromtimestamp(record.created).strftime("%H:%M:%S")
        return f"{color}[{ts}] {record.levelname:8s} {record.name}{self.RESET} — {record.getMessage()}"


def configure_logging(
    level: str = "INFO",
    json_format: bool | None = None,
) -> None:
    """Configurar logging global.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    json_format : bool | None
        True para JSON, False para texto.  Default: detecta env HPE_LOG_JSON.
    """
    if json_format is None:
        json_format = os.environ.get("HPE_LOG_JSON", "0") == "1"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())
    root.addHandler(handler)

    # Reduce noise from chatty libraries
    for noisy in ("urllib3", "fsspec", "asyncio", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_with_context(logger: logging.Logger, level: str, msg: str, **context) -> None:
    """Log com campos extras estruturados."""
    getattr(logger, level.lower())(msg, extra=context)
