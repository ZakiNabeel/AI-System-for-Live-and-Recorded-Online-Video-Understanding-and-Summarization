"""Subtitle writers for aligned transcript sentences."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import textwrap

from .schema import Sentence


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


def _format_timestamp(seconds: float, separator: str) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _wrap_sentence_text(text: str) -> list[str]:
    lines = textwrap.wrap(
        text.strip(),
        width=42,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not lines:
        return []
    chunks: list[str] = []
    for index in range(0, len(lines), 2):
        chunks.append("\n".join(lines[index : index + 2]))
    return chunks


def _sentence_to_cues(sentence: Sentence) -> list[SubtitleCue]:
    wrapped = _wrap_sentence_text(sentence.text)
    if not wrapped:
        return []
    if len(wrapped) == 1:
        start = float(sentence.start)
        end = float(sentence.end)
        if end <= start:
            end = start + 0.01
        return [SubtitleCue(start=start, end=end, text=wrapped[0])]

    total_chars = sum(len(chunk.replace("\n", "")) for chunk in wrapped) or len(wrapped)
    duration = max(0.01, float(sentence.end) - float(sentence.start))
    current = float(sentence.start)
    cues: list[SubtitleCue] = []
    for index, chunk in enumerate(wrapped):
        if index == len(wrapped) - 1:
            end = float(sentence.end)
        else:
            share = len(chunk.replace("\n", "")) / total_chars if total_chars else 1.0 / len(wrapped)
            end = current + duration * share
        if end <= current:
            end = current + 0.01
        cues.append(SubtitleCue(start=current, end=end, text=chunk))
        current = end
    return cues


def _compose_srt(cues: Sequence[SubtitleCue]) -> str:
    lines: list[str] = []
    for index, cue in enumerate(cues, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_timestamp(cue.start, ',')} --> {_format_timestamp(cue.end, ',')}"
        )
        lines.extend(cue.text.splitlines() or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def _compose_vtt(cues: Sequence[SubtitleCue]) -> str:
    if not cues:
        return "WEBVTT\n\n"
    lines: list[str] = ["WEBVTT", ""]
    for cue in cues:
        lines.append(
            f"{_format_timestamp(cue.start, '.')} --> {_format_timestamp(cue.end, '.')}"
        )
        lines.extend(cue.text.splitlines() or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_srt(sentences: Sequence[Sentence], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cues: list[SubtitleCue] = []
    for sentence in sentences:
        cues.extend(_sentence_to_cues(sentence))
    out_path.write_text(_compose_srt(cues), encoding="utf-8-sig")
    return out_path


def write_vtt(sentences: Sequence[Sentence], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cues: list[SubtitleCue] = []
    for sentence in sentences:
        cues.extend(_sentence_to_cues(sentence))
    out_path.write_text(_compose_vtt(cues), encoding="utf-8")
    return out_path
