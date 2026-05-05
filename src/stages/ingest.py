"""Ingest stage: validate the downloaded video exists."""

from __future__ import annotations

from pathlib import Path


def run_ingest(ctx) -> None:
    """Validate that the ingested video file is present."""
    paths = ctx.run_ctx.paths
    video_path = locate_video(paths.raw)
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video file not found at {video_path}. "
            "Ensure orchestrator.run() downloaded it first."
        )
    ctx.log.info(
        "Ingest validated: %s (%.1f MB)",
        video_path,
        video_path.stat().st_size / 1024 ** 2,
    )


def locate_video(raw_dir: Path) -> Path:
    """Return the canonical video file path inside a raw run directory."""
    raw_dir = Path(raw_dir)
    for ext in ("mp4", "mkv", "webm", "ts"):
        p = raw_dir / f"video.{ext}"
        if p.exists():
            return p
    # Fallback: first video-looking file
    if raw_dir.exists():
        for p in sorted(raw_dir.iterdir()):
            if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".ts"}:
                return p
    return raw_dir / "video.mp4"
