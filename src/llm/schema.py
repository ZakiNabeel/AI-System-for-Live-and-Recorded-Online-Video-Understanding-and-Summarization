"""Schema and data structures for LLM summarization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class KeyPoint:
    """A timestamped key insight from the content."""

    timestamp: float
    text: str
    confidence: Literal["low", "medium", "high"]
    source_event_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class DetectedEvent:
    """A notable event or state change in the content."""

    timestamp: float
    event_type: str  # "topic-change", "demo-start", "Q&A", etc.
    description: str
    source_event_indices: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Chapter:
    """A chapter/segment of the content."""

    t_start: float
    t_end: float
    title: str


@dataclass(frozen=True)
class QAPair:
    """A question-answer pair extracted from content."""

    question: str
    answer: str
    timestamp: float


@dataclass(frozen=True)
class Summary:
    """Complete structured summary of content."""

    run_id: str
    full_summary: str
    short_summary: str
    key_points: list[KeyPoint] = field(default_factory=list)
    events: list[DetectedEvent] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    qa_pairs: list[QAPair] | None = None
    model: str = ""
    provider: str = ""
    chunked: bool = False
    n_chunks: int = 0
    elapsed_sec: float = 0.0
    token_usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})


def _keypoint_from_dict(payload: dict[str, Any]) -> KeyPoint:
    """Deserialize KeyPoint from JSON dict."""
    return KeyPoint(
        timestamp=float(payload["timestamp"]),
        text=str(payload["text"]),
        confidence=payload.get("confidence", "medium"),
        source_event_indices=payload.get("source_event_indices", []),
    )


def _event_from_dict(payload: dict[str, Any]) -> DetectedEvent:
    """Deserialize DetectedEvent from JSON dict."""
    return DetectedEvent(
        timestamp=float(payload["timestamp"]),
        event_type=str(payload["event_type"]),
        description=str(payload["description"]),
        source_event_indices=payload.get("source_event_indices", []),
    )


def _chapter_from_dict(payload: dict[str, Any]) -> Chapter:
    """Deserialize Chapter from JSON dict."""
    return Chapter(
        t_start=float(payload["t_start"]),
        t_end=float(payload["t_end"]),
        title=str(payload["title"]),
    )


def _qapair_from_dict(payload: dict[str, Any]) -> QAPair:
    """Deserialize QAPair from JSON dict."""
    return QAPair(
        question=str(payload["question"]),
        answer=str(payload["answer"]),
        timestamp=float(payload["timestamp"]),
    )


def save_summary(summary: Summary, path: Path) -> None:
    """Persist summary to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(summary)
    payload["version"] = "1"
    payload["key_points"] = [asdict(kp) for kp in summary.key_points]
    payload["events"] = [asdict(evt) for evt in summary.events]
    payload["chapters"] = [asdict(ch) for ch in summary.chapters]
    if summary.qa_pairs:
        payload["qa_pairs"] = [asdict(qa) for qa in summary.qa_pairs]

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_summary(path: Path) -> Summary:
    """Load summary from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    key_points = [_keypoint_from_dict(kp) for kp in payload.get("key_points", [])]
    events = [_event_from_dict(evt) for evt in payload.get("events", [])]
    chapters = [_chapter_from_dict(ch) for ch in payload.get("chapters", [])]
    qa_pairs = [_qapair_from_dict(qa) for qa in payload.get("qa_pairs", [])] if payload.get("qa_pairs") else None

    return Summary(
        run_id=payload["run_id"],
        full_summary=payload["full_summary"],
        short_summary=payload["short_summary"],
        key_points=key_points,
        events=events,
        chapters=chapters,
        qa_pairs=qa_pairs,
        model=payload.get("model", ""),
        provider=payload.get("provider", ""),
        chunked=payload.get("chunked", False),
        n_chunks=payload.get("n_chunks", 0),
        elapsed_sec=payload.get("elapsed_sec", 0.0),
        token_usage=payload.get("token_usage", {}),
    )
