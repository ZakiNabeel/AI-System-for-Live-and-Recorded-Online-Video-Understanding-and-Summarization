"""OCR and optional captioning stage."""

from __future__ import annotations

from src.vision.extractor import extract_visual_content


def run_ocr(ctx) -> None:
    paths = ctx.run_ctx.paths
    enhanced_manifest = paths.frames / "enhanced" / "enhancements.json"
    frames_manifest = paths.frames / "frames.json"

    # Prefer enhanced manifest when available.
    manifest_path = enhanced_manifest if enhanced_manifest.exists() else frames_manifest

    result = extract_visual_content(
        frames_manifest_path=manifest_path,
        output_dir=paths.intermediate,
        enable_captions=ctx.enable_captions,
    )
    text_frames = sum(1 for f in result.frames if f.has_text)
    ctx.log.info(
        "OCR done: %d frames, %d with text -> %s",
        len(result.frames),
        text_frames,
        paths.intermediate / "visual.json",
    )
