"""OCR and optional captioning stage."""

from __future__ import annotations

from src.vision.extractor import extract_visual_content


def run_ocr(ctx) -> None:
    paths = ctx.run_ctx.paths
    enhanced_manifest = paths.frames / "enhanced" / "enhancements.json"
    frames_manifest = paths.frames / "frames.json"

    # Prefer enhanced manifest when available.
    manifest_path = enhanced_manifest if enhanced_manifest.exists() else frames_manifest

    vision_cfg = ctx.config.get("vision", {})
    captioner = vision_cfg.get("captioner", "gemini")
    ocr_engine = vision_cfg.get("ocr_engine", "easyocr")
    result = extract_visual_content(
        frames_manifest_path=manifest_path,
        output_dir=paths.intermediate,
        run_id=ctx.run_ctx.run_id,
        ocr_engine=ocr_engine,
        enable_captions=ctx.enable_captions,
        captioner=captioner,
    )
    text_frames = sum(1 for f in result.frames if f.has_text)
    ctx.log.info(
        "OCR done: %d frames, %d with text -> %s",
        len(result.frames),
        text_frames,
        paths.intermediate / "visual.json",
    )
