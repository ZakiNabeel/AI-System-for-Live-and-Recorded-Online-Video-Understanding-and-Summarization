"""Extract OCR text and optional captions from visual frame manifests."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from .captioning import CaptionCache, caption_claude, caption_gemini, caption_llava_local, caption_openai
from .errors import OCRBackendError
from .ocr import ocr_easyocr, ocr_tesseract

LOGGER = logging.getLogger(__name__)
OCREngine = Literal["tesseract", "easyocr"]
Captioner = Literal["claude", "openai", "llava-local", "gemini"]


@dataclass(frozen=True)
class TextLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    language: str


@dataclass(frozen=True)
class FrameVisual:
    timestamp: float
    frame_index: int
    frame_path: Path
    enhanced_path: Path | None
    text: str
    lines: list[TextLine]
    caption: str | None
    caption_source: str | None
    has_text: bool
    raw_ocr: dict | None


@dataclass(frozen=True)
class VisualExtractionResult:
    frames: list[FrameVisual]
    ocr_engine: str
    captioner: str | None
    elapsed_sec: float


def extract_visual_content(
    frames_manifest_path: Path,
    output_dir: Path,
    *,
    ocr_engine: OCREngine = "tesseract",
    languages: list[str] | None = None,
    enable_captions: bool = False,
    captioner: Captioner = "claude",
) -> VisualExtractionResult:
    """Run OCR and optional captioning over frames.json or enhancements.json."""

    started = time.perf_counter()
    languages = languages or ["eng"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_frame_records(Path(frames_manifest_path))
    cache = CaptionCache(output_dir / "caption_cache.json")
    visuals: list[FrameVisual] = []

    for record in records:
        original = record["frame_path"]
        enhanced = record.get("enhanced_path")
        ocr_path = enhanced or original
        lines, raw_ocr = _run_ocr(ocr_engine, ocr_path, languages)
        text = "\n".join(line.text for line in lines)
        caption = None
        caption_source = None
        if enable_captions:
            caption = cache.get(original)
            if caption is None:
                caption = _run_captioner(captioner, original)
                if caption:
                    cache.set(original, caption)
            caption_source = captioner if caption else None

        visuals.append(
            FrameVisual(
                timestamp=float(record.get("timestamp") or 0.0),
                frame_index=int(record.get("index") or 0),
                frame_path=original.resolve(),
                enhanced_path=enhanced.resolve() if enhanced else None,
                text=text,
                lines=lines,
                caption=caption,
                caption_source=caption_source,
                has_text=len(text.strip()) > 5,
                raw_ocr=raw_ocr,
            )
        )

    result = VisualExtractionResult(
        frames=visuals,
        ocr_engine=ocr_engine,
        captioner=captioner if enable_captions else None,
        elapsed_sec=time.perf_counter() - started,
    )
    _write_visual_json(output_dir / "visual.json", result)
    return result


def _load_frame_records(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Visual manifest not found: '{manifest_path.resolve()}'")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(f"Unknown visual manifest schema: '{manifest_path}'")

    base_dir = manifest_path.parent
    records: list[dict[str, Any]] = []
    for entry in data["frames"]:
        original_value = entry.get("source_path") or entry.get("frame_path") or entry.get("path")
        if original_value is None:
            raise ValueError(f"Frame entry missing source path: {entry}")
        enhanced_value = entry.get("enhanced_path")
        records.append(
            {
                "index": entry.get("index"),
                "timestamp": entry.get("timestamp"),
                "frame_path": _resolve_path(str(original_value), base_dir),
                "enhanced_path": _resolve_path(str(enhanced_value), base_dir) if enhanced_value else None,
            }
        )
    return records


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _run_ocr(engine: str, image_path: Path, languages: list[str]) -> tuple[list[TextLine], dict | None]:
    if engine == "tesseract":
        raw_lines, raw = ocr_tesseract(image_path, languages)
    elif engine == "easyocr":
        raw_lines, raw = ocr_easyocr(image_path, languages)
    else:
        raise ValueError(f"Unsupported OCR engine: {engine}")
    return [TextLine(**line) for line in raw_lines], raw


def _run_captioner(captioner: str, image_path: Path) -> str | None:
    try:
        if captioner == "claude":
            return caption_claude(image_path)
        if captioner == "openai":
            return caption_openai(image_path)
        if captioner == "llava-local":
            return caption_llava_local(image_path)
        if captioner == "gemini":
            return caption_gemini(image_path)
    except Exception as exc:
        LOGGER.warning("captioner %s failed for %s: %s", captioner, image_path, exc)
        return None
    raise ValueError(f"Unsupported captioner: {captioner}")


def _write_visual_json(path: Path, result: VisualExtractionResult) -> None:
    payload = {
        "version": "1",
        "ocr_engine": result.ocr_engine,
        "captioner": result.captioner,
        "elapsed_sec": result.elapsed_sec,
        "frames": [_visual_to_json(frame, path.parent) for frame in result.frames],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _visual_to_json(frame: FrameVisual, output_dir: Path) -> dict[str, Any]:
    payload = asdict(frame)
    payload["frame_path"] = _portable_path(frame.frame_path, output_dir)
    payload["enhanced_path"] = (
        _portable_path(frame.enhanced_path, output_dir)
        if frame.enhanced_path is not None
        else None
    )
    payload["lines"] = [
        {
            **asdict(line),
            "bbox": list(line.bbox),
        }
        for line in frame.lines
    ]
    return payload


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract OCR text and optional captions.")
    parser.add_argument("--manifest", type=Path, required=True, help="frames.json or enhancements.json.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ocr", choices=["tesseract", "easyocr"], default="tesseract")
    parser.add_argument("--languages", nargs="+", default=["eng"])
    parser.add_argument("--captions", action="store_true")
    parser.add_argument("--captioner", choices=["claude", "openai", "llava-local"], default="claude")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = extract_visual_content(
            frames_manifest_path=args.manifest,
            output_dir=args.out_dir,
            ocr_engine=args.ocr,
            languages=args.languages,
            enable_captions=args.captions,
            captioner=args.captioner,
        )
    except OCRBackendError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    line_count = sum(len(frame.lines) for frame in result.frames)
    caption_count = sum(1 for frame in result.frames if frame.caption)
    print(f"{len(result.frames)} frames; {line_count} lines of text; {caption_count} captioned; {result.elapsed_sec:.1f} s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
