"""Output formatting utilities."""

import base64
import io
import re
from pathlib import Path

from PIL import Image


def fmt_time(seconds: float, with_hours_if_long: float | None = None) -> str:
    """
    Format seconds as MM:SS or HH:MM:SS.

    Args:
        seconds: Time in seconds
        with_hours_if_long: If provided, use HH:MM:SS format if total time >= this threshold

    Returns:
        Formatted time string
    """
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)

    threshold = with_hours_if_long or 0
    if seconds >= 3600 or threshold >= 3600:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def is_youtube_url(url: str | None) -> bool:
    """Check if URL is a YouTube video URL."""
    if not url:
        return False
    return bool(
        re.search(r"(youtube\.com/watch\?v=|youtu\.be/)([\w-]+)", url)
    )


def yt_link(seconds: float, url: str | None) -> str:
    """
    Generate YouTube timestamp link.

    Args:
        seconds: Time in seconds
        url: YouTube URL (if not a YouTube URL, returns empty string)

    Returns:
        Full URL with timestamp query parameter, or empty string
    """
    if not url or not is_youtube_url(url):
        return ""
    t = int(round(seconds))
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={t}s"


def md_timestamp(t: float, url: str | None = None) -> str:
    """
    Generate Markdown timestamp (linked if YouTube URL provided).

    Args:
        t: Time in seconds
        url: Optional YouTube URL

    Returns:
        Markdown formatted timestamp
    """
    label = fmt_time(t)
    if url and is_youtube_url(url):
        link = yt_link(t, url)
        return f"[{label}]({link})"
    return f"`{label}`"


def to_data_uri(path: Path, max_width: int = 480) -> str:
    """
    Convert image to data URI for embedding in HTML.

    Args:
        path: Path to image file
        max_width: Maximum width before resizing

    Returns:
        data:image/png;base64,... URI string
    """
    img = Image.open(path)
    img.thumbnail((max_width, 9999))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def make_relative_path(
    frame_path: Path, output_dir: Path, base_folder: str = "frames"
) -> str:
    """
    Convert frame path to relative path for markdown/html.

    Args:
        frame_path: Absolute path to frame
        output_dir: Output directory
        base_folder: Relative folder to reference from (default: "frames")

    Returns:
        Relative path string like "frames/frame_001.png"
    """
    frame_name = frame_path.name
    return f"{base_folder}/{frame_name}"


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    import html
    return html.escape(text)
