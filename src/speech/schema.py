"""Transcript schema and JSON persistence helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

TranscriptSource = Literal["local-whisper", "openai-whisper", "youtube-subtitles"]


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    confidence: float


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word]
    language: str
    speaker: str | None = None


@dataclass(frozen=True)
class Transcript:
    segments: list[TranscriptSegment]
    language: str
    duration_sec: float
    source: TranscriptSource
    audio_path: Path | None
    raw_response: dict[str, Any] | None


@dataclass(frozen=True)
class Sentence:
    start: float
    end: float
    text: str
    word_count: int
    segment_indices: list[int]


@dataclass(frozen=True)
class AlignedTranscript(Transcript):
    sentences: list[Sentence]


def _word_from_dict(payload: dict[str, Any]) -> Word:
    return Word(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        confidence=float(payload["confidence"]),
    )


def _segment_from_dict(payload: dict[str, Any]) -> TranscriptSegment:
    words_payload = payload.get("words", [])
    return TranscriptSegment(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        words=[_word_from_dict(item) for item in words_payload],
        language=str(payload["language"]),
        speaker=payload.get("speaker"),
    )


def _sentence_from_dict(payload: dict[str, Any]) -> Sentence:
    return Sentence(
        start=float(payload["start"]),
        end=float(payload["end"]),
        text=str(payload["text"]),
        word_count=int(payload["word_count"]),
        segment_indices=[int(item) for item in payload.get("segment_indices", [])],
    )


def save_transcript(transcript: Transcript, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(transcript)
    payload["version"] = "1"
    payload["audio_path"] = str(transcript.audio_path) if transcript.audio_path is not None else None

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def load_transcript(path: Path) -> Transcript:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    segments = [_segment_from_dict(item) for item in payload.get("segments", [])]
    audio_path_raw = payload.get("audio_path")
    audio_path = Path(audio_path_raw) if audio_path_raw else None
    base_kwargs = dict(
        segments=segments,
        language=str(payload["language"]),
        duration_sec=float(payload["duration_sec"]),
        source=payload["source"],
        audio_path=audio_path,
        raw_response=payload.get("raw_response"),
    )

    if "sentences" in payload:
        return AlignedTranscript(
            **base_kwargs,
            sentences=[_sentence_from_dict(item) for item in payload.get("sentences", [])],
        )

    return Transcript(**base_kwargs)

