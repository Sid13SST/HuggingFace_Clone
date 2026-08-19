from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure(level: str | None = None) -> None:
    """Idempotent structlog setup. Console renderer on a tty, JSON otherwise."""
    global _configured
    if _configured:
        return

    from shared.config import get_settings

    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=lvl)

    renderer = (
        structlog.dev.ConsoleRenderer()
        if sys.stdout.isatty()
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, lvl, logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure()
    return structlog.get_logger(name)
