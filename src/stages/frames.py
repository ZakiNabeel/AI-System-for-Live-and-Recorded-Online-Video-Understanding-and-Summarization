"""Key-frame extraction stage."""

from __future__ import annotations

from src.vision.frame_extractor import extract_keyframes
from .ingest import locate_video


def run_frames(ctx) -> None:
    paths = ctx.run_ctx.paths
    video_path = locate_video(paths.raw)
    cfg = ctx.config.get("frames", {})

    result = extract_keyframes(
        video_path=video_path,
        output_dir=paths.frames,
        ssim_threshold=float(cfg.get("ssim_threshold", 0.92)),
        min_gap_sec=float(cfg.get("min_gap_sec", 1.5)),
    )
    ctx.log.info(
        "Frames: %d kept / %d candidates -> %s",
        result.total_kept,
        result.total_candidates,
        paths.frames / "frames.json",
    )
