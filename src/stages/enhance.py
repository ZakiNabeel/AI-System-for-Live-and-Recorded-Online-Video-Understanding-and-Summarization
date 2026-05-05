"""Frame enhancement stage (classical DIP)."""

from __future__ import annotations

from src.vision.enhancer import enhance_frames


def run_enhance(ctx) -> None:
    paths = ctx.run_ctx.paths
    frames_json = paths.frames / "frames.json"
    enhanced_dir = paths.frames / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    result = enhance_frames(
        frames_json_path=frames_json,
        output_dir=enhanced_dir,
    )
    ctx.log.info(
        "Enhanced %d frames -> %s",
        len(result.frames),
        enhanced_dir / "enhancements.json",
    )
