"""Speech-to-text package."""

from .errors import AudioTooShortError, BackendUnavailableError, NoBackendAvailableError, TranscriptionError
from .schema import Transcript, TranscriptSegment, Word, load_transcript, save_transcript
from .transcriber import transcribe

__all__ = [
    "AudioTooShortError",
    "BackendUnavailableError",
    "NoBackendAvailableError",
    "TranscriptionError",
    "Word",
    "TranscriptSegment",
    "Transcript",
    "save_transcript",
    "load_transcript",
    "transcribe",
]

