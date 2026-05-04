"""Speech-to-text package."""

from .aligner import AlignedTranscriptResult, align_transcript
from .forced_align import ForcedAlignmentNotInstalledError
from .errors import AudioTooShortError, BackendUnavailableError, NoBackendAvailableError, TranscriptionError
from .schema import AlignedTranscript, Sentence, Transcript, TranscriptSegment, Word, load_transcript, save_transcript
from .transcriber import transcribe

__all__ = [
    "AlignedTranscript",
    "Sentence",
    "AlignedTranscriptResult",
    "align_transcript",
    "ForcedAlignmentNotInstalledError",
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

