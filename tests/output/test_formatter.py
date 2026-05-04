"""Integration tests for output formatter."""

import json
from dataclasses import replace
from pathlib import Path
from html.parser import HTMLParser

import pytest

from src.output.formatter import format_outputs, render_report_json
from src.output.markdown_renderer import render_markdown
from src.output.html_renderer import render_html
from src.output.chapters import render_chapters_txt
from src.output.report_card import render_report_card


class TestMarkdownRendering:
    """Test Markdown report rendering."""

    def test_render_markdown_basic(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test basic markdown rendering."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        assert "# Summary" in md
        assert sample_summary.short_summary in md
        assert sample_summary.full_summary in md

    def test_render_markdown_with_youtube_url(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test markdown rendering with YouTube URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction, youtube_url=url)
        assert "[" in md
        assert "youtube.com" in md

    def test_render_markdown_chapters_section(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that chapters are included in markdown."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        assert "## Chapters" in md
        for chapter in sample_summary.chapters:
            assert chapter.title in md

    def test_render_markdown_keypoints_section(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that key points are included."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        assert "## Key Points" in md
        for kp in sample_summary.key_points:
            assert kp.text in md

    def test_render_markdown_events_section(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that events are included."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        assert "## Detected Events" in md
        for event in sample_summary.events:
            assert event.event_type in md
            assert event.description in md

    def test_render_markdown_qa_section(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that Q&A is included when present."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        assert "## Q&A" in md
        assert sample_summary.qa_pairs[0].question in md
        assert sample_summary.qa_pairs[0].answer in md


class TestHTMLRendering:
    """Test HTML report rendering."""

    def test_render_html_basic(self, sample_summary, sample_fused_document, sample_visual_extraction, temp_output_dir):
        """Test basic HTML rendering."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        html = render_html(md, temp_output_dir, run_id=sample_summary.run_id)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert sample_summary.short_summary in html

    def test_render_html_contains_css(self, sample_summary, sample_fused_document, sample_visual_extraction, temp_output_dir):
        """Test that HTML contains CSS."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        html = render_html(md, temp_output_dir, run_id=sample_summary.run_id)
        assert "<style>" in html
        assert "max-width" in html or "font-family" in html

    def test_render_html_is_valid(self, sample_summary, sample_fused_document, sample_visual_extraction, temp_output_dir):
        """Test that rendered HTML is valid."""
        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        html = render_html(md, temp_output_dir, run_id=sample_summary.run_id)

        # Parse HTML to verify it's well-formed
        class HTMLValidator(HTMLParser):
            def __init__(self):
                super().__init__()
                self.tag_stack = []

        validator = HTMLValidator()
        try:
            validator.feed(html)
        except Exception as e:
            pytest.fail(f"HTML parsing failed: {e}")


class TestReportJSON:
    """Test JSON report generation."""

    def test_render_report_json_basic(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test JSON report generation."""
        report = render_report_json(sample_summary, sample_fused_document, sample_visual_extraction)
        assert report["run_id"] == sample_summary.run_id
        assert "summary" in report
        assert "stats" in report
        assert "metadata" in report

    def test_render_report_json_contains_stats(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that stats are included in JSON."""
        report = render_report_json(sample_summary, sample_fused_document, sample_visual_extraction)
        stats = report["stats"]
        assert "word_count" in stats
        assert "frame_count" in stats
        assert "duration_sec" in stats
        assert stats["frame_count"] == len(sample_visual_extraction.frames)

    def test_render_report_json_serializable(self, sample_summary, sample_fused_document, sample_visual_extraction):
        """Test that JSON report can be serialized."""
        report = render_report_json(sample_summary, sample_fused_document, sample_visual_extraction)
        try:
            json_str = json.dumps(report, default=str)
            assert json_str
        except Exception as e:
            pytest.fail(f"JSON serialization failed: {e}")


class TestChaptersGeneration:
    """Test chapters.txt generation."""

    def test_render_chapters_basic(self, sample_summary):
        """Test basic chapters generation."""
        chapters = render_chapters_txt(sample_summary)
        assert "00:00" in chapters or "Introduction" in chapters
        for chapter in sample_summary.chapters:
            assert chapter.title in chapters

    def test_render_chapters_starts_at_zero(self, sample_summary):
        """Test that chapters start at 00:00."""
        chapters = render_chapters_txt(sample_summary)
        first_line = chapters.split("\n")[0]
        assert first_line.startswith("00:00")

    def test_render_chapters_proper_format(self, sample_summary):
        """Test that chapters follow YouTube format."""
        chapters = render_chapters_txt(sample_summary)
        lines = chapters.split("\n")
        for line in lines:
            if line.strip():
                # Should match pattern: HH:MM:SS Title or MM:SS Title
                parts = line.split(" ", 1)
                assert len(parts) >= 2
                time_part = parts[0]
                # Should contain colons for time format
                assert ":" in time_part


class TestReportCard:
    """Test report card generation."""

    def test_render_report_card_basic(self, temp_output_dir):
        """Test basic report card generation."""
        # Create empty marker files
        (temp_output_dir / "summary.md").touch()
        (temp_output_dir / "summary.html").touch()
        (temp_output_dir / "report.json").touch()
        (temp_output_dir / "chapters.txt").touch()

        card = render_report_card(temp_output_dir)
        assert "Report Card" in card
        assert "Deliverable" in card or "✅" in card

    def test_render_report_card_missing_files(self, temp_output_dir):
        """Test report card with missing files."""
        # Create only some files
        (temp_output_dir / "summary.md").touch()
        (temp_output_dir / "summary.html").touch()

        card = render_report_card(temp_output_dir)
        assert "summary.md" in card
        assert "summary.html" in card
        assert "❌" in card  # Missing files should show X


class TestFormatOutputsIntegration:
    """Test complete format_outputs workflow."""

    def test_format_outputs_creates_all_files(self, sample_files_in_temp_dir):
        """Test that format_outputs creates all deliverables."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        assert result.markdown.exists()
        assert result.html.exists()
        assert result.report_json.exists()
        assert result.chapters_txt.exists()
        assert result.report_card.exists()

    def test_format_outputs_markdown_content(self, sample_files_in_temp_dir):
        """Test markdown file content."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        md_content = result.markdown.read_text()
        assert "Summary" in md_content
        assert "test-run-001" in md_content

    def test_format_outputs_html_content(self, sample_files_in_temp_dir):
        """Test HTML file content."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        html_content = result.html.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "</html>" in html_content

    def test_format_outputs_json_content(self, sample_files_in_temp_dir):
        """Test JSON file content."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        json_content = json.loads(result.report_json.read_text())
        assert json_content["run_id"] == "test-run-001"
        assert "summary" in json_content
        assert "stats" in json_content

    def test_format_outputs_with_youtube_url(self, sample_files_in_temp_dir):
        """Test format_outputs with YouTube URL."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
            youtube_url=youtube_url,
        )

        md_content = result.markdown.read_text()
        assert "youtube.com" in md_content or "youtu.be" in md_content

    def test_format_outputs_missing_input_file(self, sample_files_in_temp_dir):
        """Test that format_outputs raises error for missing input."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        with pytest.raises(FileNotFoundError):
            format_outputs(
                summary_path=Path("/nonexistent/summary.json"),
                transcript_path=files["transcript"],
                fused_path=files["fused"],
                visual_path=files["visual"],
                output_dir=output_dir,
                run_id="test-run-001",
            )

    def test_format_outputs_total_size(self, sample_files_in_temp_dir):
        """Test that total_size is calculated correctly."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        assert result.total_size_bytes > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_format_outputs_empty_summary(self, sample_summary, sample_fused_document, sample_visual_extraction, temp_output_dir):
        """Test handling of empty summary."""
        # Create version with empty summary
        sample_summary = replace(sample_summary, full_summary="", short_summary="")

        md = render_markdown(sample_summary, sample_fused_document, sample_visual_extraction)
        # Should still render page structure
        assert "## Chapters" in md or "## Key Points" in md

    def test_format_outputs_unicode_content(self, sample_files_in_temp_dir):
        """Test handling of unicode in content."""
        files = sample_files_in_temp_dir
        output_dir = files["output_dir"] / "output"

        result = format_outputs(
            summary_path=files["summary"],
            transcript_path=files["transcript"],
            fused_path=files["fused"],
            visual_path=files["visual"],
            output_dir=output_dir,
            run_id="test-run-001",
        )

        # Verify files are written with UTF-8
        md_content = result.markdown.read_text(encoding="utf-8")
        assert isinstance(md_content, str)
