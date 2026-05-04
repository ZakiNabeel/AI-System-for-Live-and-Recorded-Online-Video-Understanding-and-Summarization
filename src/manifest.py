"""Manifest creation and atomic updates."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .paths import RunPaths


_MANIFEST_LOCK = threading.Lock()
STAGE_NAMES = ["ingest", "audio", "stt", "frames", "ocr", "fusion", "summary"]


def write_manifest(
    paths: RunPaths,
    run_id: str,
    mode: Literal["recorded", "live"],
    url: str,
    started_at: datetime,
    ingest_result: Any,
    ingest_status: str = "complete",
    error: str | None = None,
) -> Path:
    """Write a run manifest and return its path."""

    manifest_path = paths.intermediate / "manifest.json"
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "url": url,
        "started_at": _format_datetime(started_at),
        "ingest": _serialize_ingest(mode, ingest_result, error),
        "stages": _initial_stages(ingest_status, error),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def update_manifest(manifest_path: Path, **fields: Any) -> None:
    """Atomically merge fields into an existing manifest."""

    manifest_path = Path(manifest_path)
    with _MANIFEST_LOCK:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        _deep_merge(manifest, fields)
        _atomic_write_json(manifest_path, manifest)


def _serialize_ingest(
    mode: Literal["recorded", "live"],
    ingest_result: Any,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return {"error": error}
    if ingest_result is None:
        return {}

    if mode == "recorded":
        return {
            "video_path": str(getattr(ingest_result, "video_path", "")),
            "title": getattr(ingest_result, "title", ""),
            "duration_sec": getattr(ingest_result, "duration_sec", 0.0),
            "has_subtitles": getattr(ingest_result, "has_subtitles", False),
            "available_subtitle_langs": getattr(
                ingest_result, "available_subtitle_langs", []
            ),
        }

    return {
        "chunk_dir": str(getattr(ingest_result, "output_dir", "")),
        "chunks": [],
    }


def _initial_stages(ingest_status: str, error: str | None) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    completed_at = _format_datetime(datetime.now(timezone.utc))
    for stage_name in STAGE_NAMES:
        stages[stage_name] = {"status": "pending"}
    stages["ingest"] = {"status": ingest_status}
    if ingest_status == "complete":
        stages["ingest"]["completed_at"] = completed_at
    if error:
        stages["ingest"]["error"] = error
    return stages


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(data), file, indent=2)
        file.write("\n")
    os.replace(tmp_path, path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _format_datetime(value)
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
