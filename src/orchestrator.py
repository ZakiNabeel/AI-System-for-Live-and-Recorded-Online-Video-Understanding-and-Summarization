"""Top-level router for recorded and live ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .config import load_config
from .ingest.errors import IngestError, ModeDetectionError, RunIdConflictError
from .ingest.live_chunker import LiveCaptureHandle, start_live_capture
from .ingest.youtube_downloader import VideoDownloadResult, download_youtube_video
from .logging_setup import configure_logging
from .manifest import update_manifest, write_manifest
from .paths import RunPaths, create_run_paths


LOGGER = logging.getLogger(__name__)
Mode = Literal["recorded", "live"]
RequestedMode = Literal["auto", "recorded", "live"]


@dataclass
class RunContext:
    run_id: str
    mode: Mode
    url: str
    started_at: datetime
    paths: RunPaths
    config: dict[str, Any]
    ingest_result: VideoDownloadResult | LiveCaptureHandle | None
    manifest_path: Path


class _QuietLogger:
    def debug(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None


def run(
    url: str,
    mode: RequestedMode = "auto",
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
) -> RunContext:
    """Create a run context and dispatch ingestion for the given URL."""

    if mode not in {"auto", "recorded", "live"}:
        raise ValueError(mode)

    selected_run_id = run_id or str(uuid.uuid4())
    if _run_id_exists(selected_run_id):
        raise RunIdConflictError(f"run_id already exists: {selected_run_id}")

    config = load_config(config_path)
    paths = create_run_paths(selected_run_id)
    configure_logging(paths.log_file)
    started_at = datetime.now(timezone.utc)

    resolved_mode: Mode = detect_mode(url) if mode == "auto" else mode
    ingest_result: VideoDownloadResult | LiveCaptureHandle | None = None
    manifest_path = paths.intermediate / "manifest.json"

    try:
        if resolved_mode == "recorded":
            ingest_result = download_youtube_video(
                url=url,
                run_id=selected_run_id,
                output_root=paths.raw.parent,
                max_height=int(config["ingest"]["max_height"]),
            )
            manifest_path = write_manifest(
                paths,
                selected_run_id,
                resolved_mode,
                url,
                started_at,
                ingest_result,
            )
        elif resolved_mode == "live":
            manifest_path = write_manifest(
                paths,
                selected_run_id,
                resolved_mode,
                url,
                started_at,
                ingest_result,
                ingest_status="running",
            )
            ingest_result = start_live_capture(
                url=url,
                run_id=selected_run_id,
                output_root=paths.raw.parent,
                chunk_seconds=int(config["ingest"]["chunk_seconds"]),
                on_chunk_ready=lambda path, idx: _record_live_chunk(
                    manifest_path,
                    path,
                    idx,
                ),
            )
            manifest_path = write_manifest(
                paths,
                selected_run_id,
                resolved_mode,
                url,
                started_at,
                ingest_result,
                ingest_status="running",
            )
        else:
            raise ValueError(resolved_mode)
    except Exception as exc:
        write_manifest(
            paths,
            selected_run_id,
            resolved_mode,
            url,
            started_at,
            ingest_result,
            ingest_status="failed",
            error=str(exc),
        )
        raise

    return RunContext(
        run_id=selected_run_id,
        mode=resolved_mode,
        url=url,
        started_at=started_at,
        paths=paths,
        config=config,
        ingest_result=ingest_result,
        manifest_path=manifest_path,
    )


def detect_mode(url: str) -> Mode:
    """Probe URL metadata and return whether it is live or recorded."""

    try:
        import yt_dlp
    except ImportError as exc:
        raise ModeDetectionError(
            "yt-dlp is required for mode detection. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        with yt_dlp.YoutubeDL(
            {"quiet": True, "skip_download": True, "no_warnings": True, "logger": _QuietLogger()}
        ) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise ModeDetectionError(f"Could not detect URL mode: {exc}") from exc

    return "live" if info.get("is_live") else "recorded"


def _record_live_chunk(manifest_path: Path, path: Path, idx: int) -> None:
    update_manifest(
        manifest_path,
        ingest={
            "last_chunk": {
                "chunk_index": idx,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "wall_time": datetime.now(timezone.utc),
            }
        },
    )


def _run_id_exists(run_id: str) -> bool:
    return any(
        (Path(stage) / run_id).exists()
        for stage in [
            "data/raw",
            "data/audio",
            "data/frames",
            "data/intermediate",
            "data/output",
            "data/runs",
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ingestion orchestrator.")
    parser.add_argument("--url", required=True, help="URL to ingest.")
    parser.add_argument(
        "--mode",
        choices=["auto", "recorded", "live"],
        default="auto",
        help="Route as recorded, live, or auto-detect.",
    )
    parser.add_argument("--run-id", help="Pipeline run id. Defaults to a UUID4.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        context = run(
            url=args.url,
            mode=args.mode,
            run_id=args.run_id,
            config_path=args.config,
        )
    except IngestError as exc:
        LOGGER.exception("orchestrator failed")
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        LOGGER.exception("orchestrator failed")
        print(str(exc), file=sys.stderr)
        return 1

    print(context.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
