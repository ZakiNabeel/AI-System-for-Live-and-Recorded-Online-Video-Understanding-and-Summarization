"""Schema and persistence helpers for visual extraction outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Frame:
    """A single extracted video frame with OCR/caption metadata."""

    timestamp: float
    path: str
    text: str = ""
    caption: str | None = None
    lines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VisualExtraction:
    """Complete visual extraction artifact."""

    run_id: str
    frames: list[Frame]
    total_frames: int
    sample_rate: int


def _frame_from_dict(payload: dict[str, Any]) -> Frame:
    return Frame(
        timestamp=float(payload["timestamp"]),
        path=str(payload["path"]),
        text=str(payload.get("text", "")),
        caption=payload.get("caption"),
        lines=[str(line) for line in payload.get("lines", [])],
    )


def save_visual_extraction(visual: VisualExtraction, path: Path) -> None:
    """Persist visual extraction to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(visual)
    payload["version"] = "1"
    payload["frames"] = [asdict(frame) for frame in visual.frames]

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_visual_extraction(path: Path) -> VisualExtraction:
    """Load visual extraction from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frames = [_frame_from_dict(frame) for frame in payload.get("frames", [])]
    return VisualExtraction(
        run_id=payload.get("run_id", ""),
        frames=frames,
        total_frames=int(payload.get("total_frames", len(frames))),
        sample_rate=int(payload.get("sample_rate", 1)),
    )
