"""OCR backend adapters."""

from .easyocr_ocr import ocr_easyocr
from .tesseract_ocr import ocr_tesseract

__all__ = ["ocr_easyocr", "ocr_tesseract"]
