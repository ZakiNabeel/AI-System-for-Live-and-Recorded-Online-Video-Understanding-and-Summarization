"""Data fusion stage."""

from __future__ import annotations

from src.fusion.fuser import fuse


def run_fuse(ctx) -> None:
    paths = ctx.run_ctx.paths
    transcript_path = paths.intermediate / "transcript.aligned.json"
    visual_path = paths.intermediate / "visual.json"
    fused_path = paths.intermediate / "fused.json"

    fuse(
        transcript_path=transcript_path,
        visual_path=visual_path,
        output_path=fused_path,
        run_id=ctx.run_ctx.run_id,
    )
    ctx.log.info("Fused document -> %s", fused_path)
