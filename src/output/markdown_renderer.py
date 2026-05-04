"""Markdown report renderer."""

from pathlib import Path
from typing import Any

from src.fusion.schema import FusedDocument
from src.llm.schema import Summary
from src.vision.schema import VisualExtraction

from .helpers import fmt_time, make_relative_path, md_timestamp, escape_html


def render_header(
    summary: Summary,
    fused: FusedDocument,
    youtube_url: str | None = None,
) -> str:
    """Render header with metadata."""
    lines = []

    # Title
    lines.append(f"# Summary: Run {summary.run_id}")
    lines.append("")

    # Metadata
    metadata_parts = [
        f"**Run ID:** `{summary.run_id}`",
    ]

    if youtube_url:
        metadata_parts.append(f"**Source:** [{youtube_url}]({youtube_url})")

    duration_h = int(fused.duration_sec // 3600)
    duration_m = int((fused.duration_sec % 3600) // 60)
    duration_s = int(fused.duration_sec % 60)
    if duration_h > 0:
        duration_str = f"{duration_h}h {duration_m}m {duration_s}s"
    else:
        duration_str = f"{duration_m}m {duration_s}s"
    metadata_parts.append(f"**Duration:** {duration_str}")

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", " UTC")
    metadata_parts.append(f"**Generated:** {now}")

    lines.append(" | ".join(metadata_parts))
    lines.append("")
    lines.append("---")

    return "\n".join(lines)


def render_tldr(summary: Summary) -> str:
    """Render TL;DR section."""
    if not summary.short_summary:
        return ""
    lines = [
        "## TL;DR",
        "",
        escape_html(summary.short_summary),
    ]
    return "\n".join(lines)


def render_summary(summary: Summary) -> str:
    """Render full summary section."""
    if not summary.full_summary:
        return ""
    lines = [
        "## Summary",
        "",
        escape_html(summary.full_summary),
    ]
    return "\n".join(lines)


def render_chapters(summary: Summary, youtube_url: str | None = None) -> str:
    """Render chapters section."""
    if not summary.chapters:
        return ""

    lines = ["## Chapters"]
    lines.append("")

    for i, chapter in enumerate(summary.chapters, 1):
        t_start_str = fmt_time(chapter.t_start)
        t_end_str = fmt_time(chapter.t_end) if chapter.t_end else "—"
        title = escape_html(chapter.title)

        if youtube_url:
            ts_link = md_timestamp(chapter.t_start, youtube_url)
            lines.append(f"{i}. {ts_link} – {t_end_str} **{title}**")
        else:
            lines.append(f"{i}. `{t_start_str}` – `{t_end_str}` **{title}**")

    return "\n".join(lines)


def render_keypoints(summary: Summary, youtube_url: str | None = None) -> str:
    """Render key points section."""
    if not summary.key_points:
        return ""

    lines = ["## Key Points"]
    lines.append("")

    for kp in summary.key_points:
        ts = md_timestamp(kp.timestamp, youtube_url)
        text = escape_html(kp.text)
        confidence = kp.confidence if kp.confidence else "unknown"
        lines.append(f"- {ts} — {text} *(confidence: {confidence})*")

    return "\n".join(lines)


def render_events(summary: Summary, youtube_url: str | None = None) -> str:
    """Render detected events section."""
    if not summary.events:
        return ""

    lines = [
        "## Detected Events",
        "",
        "| Time | Type | Description |",
        "|------|------|-------------|",
    ]

    for event in summary.events:
        ts = md_timestamp(event.timestamp, youtube_url)
        event_type = escape_html(event.event_type)
        description = escape_html(event.description)
        lines.append(f"| {ts} | {event_type} | {description} |")

    return "\n".join(lines)


def render_visuals(
    visual: VisualExtraction,
    summary: Summary,
    fused: FusedDocument,
    youtube_url: str | None = None,
    max_visuals: int = 8,
) -> str:
    """
    Render selected visuals section with captions and OCR.

    Strategy:
    1. Prefer frames matching key point or event timestamps
    2. Then frames with longest OCR
    3. Always include first frame
    """
    if not visual.frames:
        return ""

    # Build set of preferred timestamps from key points and events
    preferred_timestamps = set()
    for kp in summary.key_points or []:
        preferred_timestamps.add(kp.timestamp)
    for event in summary.events or []:
        preferred_timestamps.add(event.timestamp)

    # Score frames: prefer matches to key points/events, then by OCR length
    scored_frames = []
    for frame in visual.frames:
        if frame.timestamp in preferred_timestamps:
            score = (2, len(frame.text) if frame.text else 0, frame.timestamp)
        else:
            score = (1, len(frame.text) if frame.text else 0, frame.timestamp)
        scored_frames.append((score, frame))

    # Sort: descending priority, descending OCR length, ascending timestamp
    scored_frames.sort(key=lambda x: (-x[0][0], -x[0][1], x[0][2]))

    # Select frames: always include first from fused, then top N
    selected_frames = []
    if visual.frames:
        selected_frames.append(visual.frames[0])

    for _score, frame in scored_frames[: max_visuals - 1]:
        if frame not in selected_frames:
            selected_frames.append(frame)

    if len(selected_frames) > max_visuals:
        selected_frames = selected_frames[:max_visuals]

    lines = ["## Selected Visuals"]
    lines.append("")

    for frame in selected_frames:
        ts = md_timestamp(frame.timestamp, youtube_url)
        caption = escape_html(frame.caption) if frame.caption else f"Frame at {fmt_time(frame.timestamp)}"
        rel_path = make_relative_path(Path(frame.path), Path("."))
        lines.append(f"### {ts} — {caption}")
        lines.append("")
        lines.append(f"![frame]({rel_path})")

        if frame.text:
            ocr_text = escape_html(frame.text)
            lines.append("")
            lines.append(f"> **OCR:** {ocr_text}")

        lines.append("")

    return "\n".join(lines)


def render_qa(summary: Summary, youtube_url: str | None = None) -> str:
    """Render Q&A section if present."""
    if not summary.qa_pairs:
        return ""

    lines = ["## Q&A"]
    lines.append("")

    for qa in summary.qa_pairs:
        question = escape_html(qa.question)
        answer = escape_html(qa.answer)
        ts = md_timestamp(qa.timestamp, youtube_url) if qa.timestamp else "(no timestamp)"
        lines.append(f"**Q:** {question}")
        lines.append("")
        lines.append(f"**A:** {answer} *({ts})*")
        lines.append("")

    return "\n".join(lines)


def render_footer() -> str:
    """Render footer."""
    lines = [
        "---",
        "",
        "*Generated by DIP Video Understanding System.*",
    ]
    return "\n".join(lines)


def render_markdown(
    summary: Summary,
    fused: FusedDocument,
    visual: VisualExtraction,
    youtube_url: str | None = None,
) -> str:
    """
    Render complete Markdown report.

    Args:
        summary: Summary object from Plan 4.2
        fused: FusedDocument object from Plan 4.1
        visual: VisualExtraction object from Plan 3.3
        youtube_url: Optional YouTube URL for timestamp links

    Returns:
        Complete Markdown string
    """
    parts = [
        render_header(summary, fused, youtube_url),
        render_tldr(summary),
        render_summary(summary),
        render_chapters(summary, youtube_url),
        render_keypoints(summary, youtube_url),
        render_events(summary, youtube_url),
        render_visuals(visual, summary, fused, youtube_url),
        render_qa(summary, youtube_url),
        render_footer(),
    ]

    # Filter out empty parts and join with double newlines
    result = "\n\n".join(p for p in parts if p)
    return result
