"""Tesseract OCR adapter."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..errors import OCRBackendError, TesseractNotInstalledError


def ocr_tesseract(enhanced_path: Path, languages: list[str]) -> tuple[list[dict], dict]:
    """Run Tesseract and return line-level OCR dictionaries plus raw data."""

    try:
        import pytesseract
    except ImportError as exc:
        raise TesseractNotInstalledError() from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise TesseractNotInstalledError() from exc

    try:
        with Image.open(enhanced_path) as image:
            data = pytesseract.image_to_data(
                image,
                lang="+".join(languages),
                output_type=pytesseract.Output.DICT,
            )
    except Exception as exc:
        raise OCRBackendError(f"Tesseract failed for '{enhanced_path}': {exc}") from exc

    grouped: dict[tuple[int, int, int], list[int]] = {}
    for idx, text in enumerate(data.get("text", [])):
        if not str(text).strip():
            continue
        key = (
            int(data["page_num"][idx]),
            int(data["block_num"][idx]),
            int(data["line_num"][idx]),
        )
        grouped.setdefault(key, []).append(idx)

    lines: list[dict] = []
    for _, indexes in sorted(grouped.items()):
        words = [str(data["text"][idx]).strip() for idx in indexes if str(data["text"][idx]).strip()]
        confidences = [_parse_conf(data["conf"][idx]) for idx in indexes]
        confidences = [conf for conf in confidences if conf >= 0]
        if not words or not confidences:
            continue
        text = " ".join(words).strip()
        confidence = sum(confidences) / len(confidences)
        if confidence < 30 or len(text) < 3:
            continue

        x0 = min(int(data["left"][idx]) for idx in indexes)
        y0 = min(int(data["top"][idx]) for idx in indexes)
        x1 = max(int(data["left"][idx]) + int(data["width"][idx]) for idx in indexes)
        y1 = max(int(data["top"][idx]) + int(data["height"][idx]) for idx in indexes)
        lines.append(
            {
                "text": text,
                "confidence": float(confidence),
                "bbox": (x0, y0, x1 - x0, y1 - y0),
                "language": languages[0] if languages else "eng",
            }
        )

    return lines, data


def _parse_conf(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0
