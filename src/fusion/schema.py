"""Schema and persistence helpers for multimodal event fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class FusedEvent:
    """A single timestamped multimodal event."""

    t_start: float
    t_end: float
    kind: Literal["speech", "visual", "speech+visual"]
    speech_text: str | None = None
    speech_segment_indices: list[int] = field(default_factory=list)
    visual_text: str | None = None
    visual_caption: str | None = None
    frame_index: int | None = None
    frame_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate event structure."""
        if self.t_start > self.t_end:
            raise ValueError(f"Invalid event: t_start ({self.t_start}) > t_end ({self.t_end})")
        if self.kind == "speech" and not self.speech_text:
            raise ValueError("Speech event must have speech_text")
        if self.kind == "visual" and not (self.visual_text or self.visual_caption or self.frame_path):
            raise ValueError("Visual event must have visual_text, visual_caption, or frame_path")


@dataclass(frozen=True)
class FusedDocument:
    """Complete multimodal fused document."""

    run_id: str
    duration_sec: float
    language: str
    events: list[FusedEvent]
    speech_source: str
    ocr_engine: str
    has_captions: bool


def _event_from_dict(payload: dict[str, Any]) -> FusedEvent:
    """Deserialize FusedEvent from JSON dict."""
    return FusedEvent(
        t_start=float(payload["t_start"]),
        t_end=float(payload["t_end"]),
        kind=payload["kind"],
        speech_text=payload.get("speech_text"),
        speech_segment_indices=payload.get("speech_segment_indices", []),
        visual_text=payload.get("visual_text"),
        visual_caption=payload.get("visual_caption"),
        frame_index=payload.get("frame_index"),
        frame_path=payload.get("frame_path"),
        notes=payload.get("notes", []),
    )


def save_fused_document(doc: FusedDocument, path: Path) -> None:
    """Persist fused document to JSON with version info."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(doc)
    payload["version"] = "1"
    payload["events"] = [asdict(event) for event in doc.events]

    from src.json_utils import dump as safe_dump
    with path.open("w", encoding="utf-8") as handle:
        safe_dump(payload, handle, indent=2, ensure_ascii=False)


def load_fused_document(path: Path) -> FusedDocument:
    """Load fused document from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    events = [_event_from_dict(evt) for evt in payload["events"]]
    return FusedDocument(
        run_id=payload["run_id"],
        duration_sec=float(payload["duration_sec"]),
        language=payload["language"],
        events=events,
        speech_source=payload["speech_source"],
        ocr_engine=payload["ocr_engine"],
        has_captions=payload["has_captions"],
    )
