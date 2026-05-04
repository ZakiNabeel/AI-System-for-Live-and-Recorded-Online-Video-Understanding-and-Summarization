"""Audio extraction utilities for the DIP pipeline."""

from .errors import AudioExtractionError
from .extractor import AudioExtractionResult, extract_audio, extract_audio_for_chunks

__all__ = [
    "AudioExtractionError",
    "AudioExtractionResult",
    "extract_audio",
    "extract_audio_for_chunks",
]

