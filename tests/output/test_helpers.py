"""Tests for output formatter helpers."""

import pytest

from src.output.helpers import (
    escape_html,
    fmt_time,
    is_youtube_url,
    make_relative_path,
    md_timestamp,
    yt_link,
)
from pathlib import Path


class TestTimeFormatter:
    """Test fmt_time function."""

    def test_format_zero(self):
        """Test formatting 0 seconds."""
        assert fmt_time(0) == "00:00"

    def test_format_under_hour(self):
        """Test formatting under 1 hour."""
        assert fmt_time(125) == "02:05"

    def test_format_hours(self):
        """Test formatting with hours."""
        assert fmt_time(3661) == "01:01:01"
        assert fmt_time(3600) == "01:00:00"

    def test_format_with_threshold(self):
        """Test format_time with hours threshold."""
        # Should use HH:MM:SS if threshold is >= 3600
        assert fmt_time(125, with_hours_if_long=3600) == "00:02:05"
        # Should not use HH:MM:SS if threshold is < 3600
        assert fmt_time(125, with_hours_if_long=60) == "02:05"

    def test_rounding(self):
        """Test that seconds are properly rounded."""
        assert fmt_time(125.6) == "02:06"
        assert fmt_time(125.4) == "02:05"


class TestYouTubeLink:
    """Test YouTube URL utilities."""

    def test_is_youtube_url(self):
        """Test YouTube URL detection."""
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert not is_youtube_url("https://example.com/video")
        assert not is_youtube_url(None)
        assert not is_youtube_url("")

    def test_yt_link_with_youtube_url(self):
        """Test YouTube timestamp link generation."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        link = yt_link(125.5, url)
        assert "t=126s" in link or "t=125s" in link
        assert "youtube.com" in link

    def test_yt_link_without_youtube_url(self):
        """Test YouTube link with non-YouTube URL."""
        assert yt_link(125, "https://example.com/video") == ""
        assert yt_link(125, None) == ""

    def test_yt_link_already_has_query(self):
        """Test YouTube link when URL already has query params."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        link = yt_link(125, url)
        assert "&t=" in link
        assert "t=125s" in link


class TestMarkdownTimestamp:
    """Test Markdown timestamp generation."""

    def test_md_timestamp_without_url(self):
        """Test Markdown timestamp without YouTube URL."""
        assert md_timestamp(125, None) == "`02:05`"

    def test_md_timestamp_with_url(self):
        """Test Markdown timestamp with YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ts = md_timestamp(125, url)
        assert "[" in ts
        assert "](" in ts
        assert "youtube.com" in ts

    def test_md_timestamp_with_non_youtube_url(self):
        """Test Markdown timestamp with non-YouTube URL (should not link)."""
        url = "https://example.com/video"
        assert md_timestamp(125, url) == "`02:05`"


class TestHTMLEscape:
    """Test HTML escaping."""

    def test_escape_html_basic(self):
        """Test basic HTML escaping."""
        assert escape_html("<script>alert('xss')</script>") == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        assert escape_html("A & B") == "A &amp; B"
        assert escape_html('"quoted"') == "&quot;quoted&quot;"

    def test_escape_html_unicode(self):
        """Test HTML escaping with unicode."""
        text = "Hello 世界 🌍"
        escaped = escape_html(text)
        assert "世界" in escaped
        assert "🌍" in escaped


class TestRelativePath:
    """Test relative path generation."""

    def test_make_relative_path(self):
        """Test relative path generation."""
        frame_path = Path("/data/frames/frame_001.png")
        output_dir = Path("/output")
        rel = make_relative_path(frame_path, output_dir)
        assert rel == "frames/frame_001.png"

    def test_make_relative_path_custom_folder(self):
        """Test relative path with custom folder."""
        frame_path = Path("/data/frames/frame_001.png")
        output_dir = Path("/output")
        rel = make_relative_path(frame_path, output_dir, base_folder="images")
        assert rel == "images/frame_001.png"
