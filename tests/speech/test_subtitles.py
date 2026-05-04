from __future__ import annotations

import re
from pathlib import Path

from src.speech.schema import Sentence
from src.speech.subtitles import write_srt, write_vtt


def test_subtitle_writers_produce_valid_headers(tmp_path: Path) -> None:
    sentences = [
        Sentence(start=0.0, end=2.0, text="Hello and welcome.", word_count=3, segment_indices=[0]),
        Sentence(start=2.0, end=4.0, text="Today we cover alignment.", word_count=4, segment_indices=[0]),
    ]

    srt_path = write_srt(sentences, tmp_path / "captions.srt")
    vtt_path = write_vtt(sentences, tmp_path / "captions.vtt")

    srt_text = srt_path.read_text(encoding="utf-8-sig")
    vtt_text = vtt_path.read_text(encoding="utf-8")

    assert "00:00:00,000 --> 00:00:02,000" in srt_text
    assert vtt_text.startswith("WEBVTT")


def test_long_sentence_wraps_with_short_lines(tmp_path: Path) -> None:
    text = " ".join(["alignment"] * 40)
    sentences = [Sentence(start=0.0, end=10.0, text=text, word_count=40, segment_indices=[0])]

    srt_path = write_srt(sentences, tmp_path / "long.srt")
    lines = [line for line in srt_path.read_text(encoding="utf-8-sig").splitlines() if line]
    text_lines = [
        line
        for line in lines
        if not re.fullmatch(r"\d+", line)
        and "-->" not in line
    ]

    assert text_lines
    assert all(len(line) <= 42 for line in text_lines)


def test_empty_subtitles_are_valid(tmp_path: Path) -> None:
    srt_path = write_srt([], tmp_path / "empty.srt")
    vtt_path = write_vtt([], tmp_path / "empty.vtt")

    assert srt_path.read_text(encoding="utf-8-sig") == ""
    assert vtt_path.read_text(encoding="utf-8") == "WEBVTT\n\n"
