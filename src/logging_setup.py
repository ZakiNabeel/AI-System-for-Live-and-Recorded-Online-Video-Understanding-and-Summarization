"""Per-run logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_file: Path, level: int = logging.INFO) -> None:
    """Configure root logging to stderr and the run log file."""

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
