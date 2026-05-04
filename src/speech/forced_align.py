"""Optional forced alignment wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import TranscriptionError
from .schema import Transcript, TranscriptSegment, Word


class ForcedAlignmentNotInstalledError(TranscriptionError):
    """Raised when forced alignment dependencies are unavailable."""


def _get_value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _segment_from_payload(payload: Any, language: str) -> TranscriptSegment:
    words_payload = _get_value(payload, "words", []) or []
    words = [
        Word(
            start=float(_get_value(word, "start", 0.0)),
            end=float(_get_value(word, "end", 0.0)),
            text=str(_get_value(word, "word", _get_value(word, "text", ""))).strip(),
            confidence=float(_get_value(word, "probability", _get_value(word, "confidence", 0.0))),
        )
        for word in words_payload
    ]
    segment_language = str(_get_value(payload, "language", language) or language)
    return TranscriptSegment(
        start=float(_get_value(payload, "start", 0.0)),
        end=float(_get_value(payload, "end", 0.0)),
        text=str(_get_value(payload, "text", "")).strip(),
        words=words,
        language=segment_language,
        speaker=_get_value(payload, "speaker"),
    )


def forced_realign(transcript: Transcript, audio_path: Path) -> Transcript:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: '{audio_path.resolve()}'")

    try:
        import stable_whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ForcedAlignmentNotInstalledError(
            "stable-ts is not installed. Install stable-ts>=2.17.0 to use forced mode."
        ) from exc

    model = stable_whisper.load_faster_whisper("small.en")
    result = model.transcribe_stable(str(audio_path), regroup=True)
    segments_payload = _get_value(result, "segments", []) or []
    language = str(_get_value(result, "language", transcript.language) or transcript.language)

    segments = [_segment_from_payload(segment, language) for segment in segments_payload]
    duration = float(_get_value(result, "duration", transcript.duration_sec) or transcript.duration_sec)
    if duration <= 0 and segments:
        duration = float(segments[-1].end)

    return Transcript(
        segments=segments,
        language=language,
        duration_sec=duration,
        source=transcript.source,
        audio_path=audio_path,
        raw_response=transcript.raw_response,
    )
