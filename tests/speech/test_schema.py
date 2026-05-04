from __future__ import annotations

import math
from pathlib import Path

from src.speech.schema import (
    AlignedTranscript,
    Sentence,
    Transcript,
    TranscriptSegment,
    Word,
    load_transcript,
    save_transcript,
)


def test_schema_round_trip(tmp_path: Path) -> None:
    transcript = Transcript(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.2,
                text="hello world",
                words=[
                    Word(start=0.0, end=0.5, text="hello", confidence=0.9),
                    Word(start=0.6, end=1.2, text="world", confidence=math.nan),
                ],
                language="en",
                speaker=None,
            )
        ],
        language="en",
        duration_sec=1.2,
        source="local-whisper",
        audio_path=Path("data/audio/run/audio.wav"),
        raw_response={"debug": True},
    )
    out = tmp_path / "transcript.json"
    save_transcript(transcript, out)
    loaded = load_transcript(out)

    assert loaded.language == transcript.language
    assert loaded.duration_sec == transcript.duration_sec
    assert loaded.source == transcript.source
    assert loaded.audio_path == transcript.audio_path
    assert loaded.raw_response == transcript.raw_response
    assert len(loaded.segments) == 1
    assert loaded.segments[0].text == "hello world"
    assert len(loaded.segments[0].words) == 2
    assert math.isnan(loaded.segments[0].words[1].confidence)


def test_aligned_schema_round_trip(tmp_path: Path) -> None:
    transcript = AlignedTranscript(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="hello world",
                words=[
                    Word(start=0.0, end=1.0, text="hello", confidence=0.9),
                    Word(start=1.0, end=2.0, text="world", confidence=0.8),
                ],
                language="en",
                speaker=None,
            )
        ],
        language="en",
        duration_sec=2.0,
        source="local-whisper",
        audio_path=Path("data/audio/run/audio.wav"),
        raw_response={"debug": True},
        sentences=[
            Sentence(
                start=0.0,
                end=2.0,
                text="Hello world",
                word_count=2,
                segment_indices=[0],
            )
        ],
    )
    out = tmp_path / "aligned.json"
    save_transcript(transcript, out)
    loaded = load_transcript(out)

    assert isinstance(loaded, AlignedTranscript)
    assert loaded.sentences[0].text == "Hello world"
    assert loaded.sentences[0].segment_indices == [0]

