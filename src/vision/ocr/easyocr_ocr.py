"""EasyOCR adapter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..errors import OCRBackendError


def ocr_easyocr(enhanced_path: Path, languages: list[str]) -> tuple[list[dict], dict]:
    """Run EasyOCR and return line-level OCR dictionaries plus raw data."""

    try:
        reader = _reader(tuple(languages or ["en"]))
    except ImportError as exc:
        raise OCRBackendError(
            "EasyOCR is not installed. Install `easyocr` or choose `--ocr tesseract`."
        ) from exc

    try:
        results = reader.readtext(str(enhanced_path))
    except Exception as exc:
        raise OCRBackendError(f"EasyOCR failed for '{enhanced_path}': {exc}") from exc

    lines: list[dict] = []
    for box, text, confidence in results:
        clean_text = str(text).strip()
        scaled_confidence = float(confidence) * 100.0
        if scaled_confidence < 30 or len(clean_text) < 3:
            continue
        xs = [int(point[0]) for point in box]
        ys = [int(point[1]) for point in box]
        x0 = min(xs)
        y0 = min(ys)
        lines.append(
            {
                "text": clean_text,
                "confidence": scaled_confidence,
                "bbox": (x0, y0, max(xs) - x0, max(ys) - y0),
                "language": (languages or ["en"])[0],
            }
        )

    return lines, {"results": results}


@lru_cache(maxsize=4)
def _reader(languages: tuple[str, ...]):
    import easyocr

    return easyocr.Reader(list(languages), gpu=False)
