"""Vision-stage exceptions."""

from __future__ import annotations

from pathlib import Path


class VisionError(RuntimeError):
    """Base class for visual processing failures."""


class CannotOpenVideoError(VisionError):
    """Raised when OpenCV cannot open a source video."""

    def __init__(self, video_path: Path) -> None:
        super().__init__(f"Cannot open video: '{Path(video_path).resolve()}'")


class FrameExtractionError(VisionError):
    """Raised when key-frame extraction cannot complete."""


class EnhancementError(VisionError):
    """Raised when frame enhancement cannot complete."""


class OCRBackendError(VisionError):
    """Raised when an OCR backend is unavailable or fails."""


class TesseractNotInstalledError(OCRBackendError):
    """Raised when pytesseract or the tesseract binary is unavailable."""

    def __init__(self) -> None:
        super().__init__(
            "Tesseract OCR is not available. Install the tesseract binary and "
            "`pytesseract`, then ensure `tesseract` is on PATH."
        )
