"""Audio extraction stage."""

from __future__ import annotations

from src.audio.extractor import extract_audio
from .ingest import locate_video


def run_audio(ctx) -> None:
    paths = ctx.run_ctx.paths
    video_path = locate_video(paths.raw)
    audio_path = paths.audio / "audio.wav"

    cfg = ctx.config.get("audio", {})
    extract_audio(
        video_path=video_path,
        output_path=audio_path,
        sample_rate=int(cfg.get("sample_rate", 16000)),
        mono=bool(cfg.get("mono", True)),
    )
    ctx.log.info("Audio extracted: %s", audio_path)
