"""Rich-backed logging for the CLI and server."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def get_logger(name: str = "mini_vllm") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
        )
        _CONFIGURED = True
    return logging.getLogger(name)
