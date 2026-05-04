"""YouTube chapters generator."""

from src.llm.schema import Summary

from .helpers import fmt_time, escape_html


def render_chapters_txt(summary: Summary) -> str:
    """
    Generate YouTube-style chapters.txt.

    Format: MM:SS Title or HH:MM:SS Title
    First chapter must be 00:00.

    Args:
        summary: Summary object with chapters list

    Returns:
        YouTube chapters string (one per line)
    """
    if not summary.chapters:
        lines = ["00:00 Introduction"]
        return "\n".join(lines)

    lines = []

    # Always start with 00:00
    first_chapter_time = summary.chapters[0].t_start if summary.chapters else 0
    if first_chapter_time > 0:
        lines.append("00:00 Introduction")

    # Add all chapters
    for chapter in summary.chapters:
        t_str = fmt_time(chapter.t_start, with_hours_if_long=summary.duration_sec if hasattr(summary, 'duration_sec') else 3600)
        title = escape_html(chapter.title)
        lines.append(f"{t_str} {title}")

    return "\n".join(lines)
