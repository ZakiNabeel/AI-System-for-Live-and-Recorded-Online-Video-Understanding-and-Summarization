from __future__ import annotations

import pytest

from src.speech.aligner_steps import (
    build_sentences,
    drop_hallucinations,
    fix_gaps_and_overlaps,
    merge_tiny_segments,
    split_overlong_segments,
)
from src.speech.schema import TranscriptSegment, Word


def _segment(start: float, end: float, text: str, words: list[Word] | None = None) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text, words=words or [], language="en")


def test_drop_hallucinations_removes_low_confidence_phrase() -> None:
    cleaned = drop_hallucinations(
        [
            _segment(0.0, 1.0, "thanks for watching"),
            _segment(1.0, 2.0, "hello", [Word(1.0, 2.0, "hello", 0.9)]),
        ]
    )

    assert len(cleaned) == 1
    assert cleaned[0].text == "hello"


def test_fix_gaps_and_overlaps_shifts_overlap_forward() -> None:
    fixed = fix_gaps_and_overlaps(
        [
            _segment(0.0, 1.0, "one", [Word(0.0, 1.0, "one", 0.9)]),
            _segment(0.8, 2.0, "two", [Word(0.8, 2.0, "two", 0.9)]),
        ]
    )

    assert fixed[1].start == pytest.approx(1.01)
    assert fixed[1].words[0].start == pytest.approx(1.01)


def test_merge_tiny_segments_merges_with_nearest_neighbor() -> None:
    merged = merge_tiny_segments(
        [
            _segment(0.0, 2.0, "first", [Word(0.0, 2.0, "first", 0.9)]),
            _segment(2.0, 2.2, "tiny", [Word(2.0, 2.2, "tiny", 0.9)]),
            _segment(2.2, 4.0, "third", [Word(2.2, 4.0, "third", 0.9)]),
        ],
        min_segment_sec=1.0,
    )

    assert len(merged) == 2
    assert "tiny" in merged[0].text


def test_split_overlong_segments_limits_duration() -> None:
    segment = _segment(
        0.0,
        16.0,
        "one two three four",
        [
            Word(0.0, 4.0, "one", 0.9),
            Word(4.0, 8.0, "two", 0.9),
            Word(8.0, 12.0, "three", 0.9),
            Word(12.0, 16.0, "four", 0.9),
        ],
    )

    split = split_overlong_segments([segment], max_segment_sec=12.0)

    assert len(split) == 2
    assert all(item.end - item.start <= 12.0 for item in split)


def test_build_sentences_assigns_times_and_segments() -> None:
    sentences = build_sentences(
        [
            _segment(
                0.0,
                4.0,
                "hello world. second sentence.",
                [
                    Word(0.0, 1.0, "hello", 0.9),
                    Word(1.0, 2.0, "world", 0.9),
                    Word(2.0, 3.0, "second", 0.9),
                    Word(3.0, 4.0, "sentence", 0.9),
                ],
            )
        ],
        language="en",
    )

    assert len(sentences) == 2
    assert sentences[0].text.startswith("Hello")
    assert sentences[0].start == pytest.approx(0.0)
    assert sentences[0].end <= sentences[1].start
    assert sentences[0].segment_indices == [0]
