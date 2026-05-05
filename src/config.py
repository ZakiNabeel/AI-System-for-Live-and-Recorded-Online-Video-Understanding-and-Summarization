"""Project configuration loading."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "ingest": {"max_height": 720, "chunk_seconds": 10},
    "audio": {"sample_rate": 16000, "mono": True},
    "speech": {"engine": "whisper", "model": "small.en"},
    "frames": {"ssim_threshold": 0.92, "min_gap_sec": 1.5},
    "llm": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
    "vision": {"captioner": "gemini"},
}


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    """Load config.yaml, falling back to defaults when it is missing."""

    _load_dotenv_if_available()
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(path)

    if path.exists():
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "pyyaml is required to load config.yaml. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        with path.open("r", encoding="utf-8") as file:
            file_config = yaml.safe_load(file) or {}
        if not isinstance(file_config, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {path}")
        _deep_merge(config, file_config)
    else:
        LOGGER.warning("Config file %s not found; using defaults.", path)

    config["env"] = {
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "youtube_cookies_file": os.getenv("YOUTUBE_COOKIES_FILE")
        or os.getenv("YT_DLP_COOKIES_FILE"),
    }
    return config


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
