"""Speech module exceptions."""


class TranscriptionError(RuntimeError):
    """Base class for speech transcription errors."""


class NoBackendAvailableError(TranscriptionError):
    """Raised when no transcription backend can be resolved."""


class BackendUnavailableError(TranscriptionError):
    """Raised when a requested backend dependency is not installed."""


class AudioTooShortError(TranscriptionError):
    """Raised when the input audio is too short for reliable transcription."""

