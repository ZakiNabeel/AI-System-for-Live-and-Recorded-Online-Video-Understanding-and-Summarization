"""Qualitative note tagging for fused events."""

from __future__ import annotations

from typing import NamedTuple


class Window(NamedTuple):
    """A time window with associated data."""

    t_start: float
    t_end: float
    has_speech: bool
    has_visual: bool
    gap_before: float | None  # gap since previous non-empty window


def tag_notes(
    window: Window,
    prev_window_end: float | None,
    next_window_start: float | None,
    is_scene_change: bool = False,
) -> list[str]:
    """
    Determine qualitative tags for a window.

    Args:
        window: Current window data
        prev_window_end: End time of previous non-empty window (if any)
        next_window_start: Start time of next non-empty window (if any)
        is_scene_change: Whether frame has scene-change indicator

    Returns:
        List of tag strings
    """
    notes: list[str] = []

    # Long pause: gap > 8 seconds between non-empty windows
    if prev_window_end is not None and window.t_start - prev_window_end > 8.0:
        notes.append("long-pause")

    # Scene change: visual transition + speech in same window
    if is_scene_change and window.has_speech and window.has_visual:
        notes.append("scene-change")

    # Silent visual: visual content with no speech
    if window.has_visual and not window.has_speech:
        notes.append("silent-visual")

    return notes
