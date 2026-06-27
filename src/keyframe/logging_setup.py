"""Structured logging using rich. Falls back to plain stderr if rich is missing."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Initialise a rich-backed logger. Call once at process start."""
    log = logging.getLogger("keyframe")
    if log.handlers:
        return log
    log.setLevel(level)
    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
            omit_repeated_times=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    except Exception:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
    log.addHandler(handler)
    log.propagate = False
    return log


def get_logger(name: str | None = None) -> logging.Logger:
    parent = logging.getLogger("keyframe")
    if not parent.handlers:
        setup_logging()
    return parent.getChild(name) if name else parent
