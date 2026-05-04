"""Speech backend implementations."""

from .local_whisper import LocalWhisperBackend
from .openai_api import OpenAIWhisperBackend
from .youtube_subs import fetch_youtube_transcript

__all__ = ["LocalWhisperBackend", "OpenAIWhisperBackend", "fetch_youtube_transcript"]

