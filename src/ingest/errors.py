"""Exceptions raised by media ingestion modules."""


class IngestError(Exception):
    """Base class for user-facing ingestion errors."""


class PrivateVideoError(IngestError):
    """Raised when a video requires private access."""


class UnavailableVideoError(IngestError):
    """Raised when a video is unavailable."""


class NetworkError(IngestError):
    """Raised when a network problem prevents ingestion."""


class FFmpegMissingError(IngestError):
    """Raised when ffmpeg is missing or fails during muxing."""


class LiveStreamNotSupportedError(IngestError):
    """Raised when a live stream is sent to the batch downloader."""


class NotALiveStreamError(IngestError):
    """Raised when a URL expected to be live is not a live stream."""


class UnresolvableStreamError(IngestError):
    """Raised when a live stream URL cannot be resolved to playable media."""


class LiveCaptureError(IngestError):
    """Raised when live stream capture cannot be started or controlled."""


class ModeDetectionError(IngestError):
    """Raised when the orchestrator cannot determine recorded vs live mode."""


class RunIdConflictError(IngestError):
    """Raised when a run_id would overwrite an existing run."""
