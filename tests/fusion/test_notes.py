"""Tests for note tagging."""

from __future__ import annotations

import pytest

from src.fusion.notes import Window, tag_notes


class TestNoteTagging:
    def test_no_notes_normal_window(self):
        """Test that normal window gets no notes."""
        window = Window(
            t_start=0.0,
            t_end=5.0,
            has_speech=True,
            has_visual=False,
            gap_before=None,
        )

        notes = tag_notes(window, prev_window_end=None, next_window_start=None)

        assert notes == []

    def test_long_pause_detected(self):
        """Test detection of long pause."""
        window = Window(
            t_start=20.0,
            t_end=25.0,
            has_speech=True,
            has_visual=False,
            gap_before=9.0,  # > 8s
        )

        notes = tag_notes(window, prev_window_end=10.0, next_window_start=None)

        assert "long-pause" in notes

    def test_long_pause_not_triggered_short_gap(self):
        """Test that short gap doesn't trigger long-pause."""
        window = Window(
            t_start=10.0,
            t_end=15.0,
            has_speech=True,
            has_visual=False,
            gap_before=2.0,  # < 8s
        )

        notes = tag_notes(window, prev_window_end=8.0, next_window_start=None)

        assert "long-pause" not in notes

    def test_scene_change_detected(self):
        """Test detection of scene change."""
        window = Window(
            t_start=5.0,
            t_end=10.0,
            has_speech=True,
            has_visual=True,
            gap_before=None,
        )

        notes = tag_notes(window, prev_window_end=None, next_window_start=None, is_scene_change=True)

        assert "scene-change" in notes

    def test_scene_change_requires_speech_and_visual(self):
        """Test that scene-change needs both speech and visual."""
        # Visual only
        window_visual = Window(
            t_start=5.0,
            t_end=10.0,
            has_speech=False,
            has_visual=True,
            gap_before=None,
        )
        notes = tag_notes(window_visual, prev_window_end=None, next_window_start=None, is_scene_change=True)
        assert "scene-change" not in notes

        # Speech only
        window_speech = Window(
            t_start=5.0,
            t_end=10.0,
            has_speech=True,
            has_visual=False,
            gap_before=None,
        )
        notes = tag_notes(window_speech, prev_window_end=None, next_window_start=None, is_scene_change=True)
        assert "scene-change" not in notes

    def test_silent_visual_detected(self):
        """Test detection of silent visual content."""
        window = Window(
            t_start=5.0,
            t_end=10.0,
            has_speech=False,
            has_visual=True,
            gap_before=None,
        )

        notes = tag_notes(window, prev_window_end=None, next_window_start=None)

        assert "silent-visual" in notes

    def test_silent_visual_not_triggered_with_speech(self):
        """Test that silent-visual isn't tagged when there's speech."""
        window = Window(
            t_start=5.0,
            t_end=10.0,
            has_speech=True,
            has_visual=True,
            gap_before=None,
        )

        notes = tag_notes(window, prev_window_end=None, next_window_start=None)

        assert "silent-visual" not in notes

    def test_multiple_notes_possible(self):
        """Test that multiple notes can be tagged simultaneously."""
        window = Window(
            t_start=20.0,
            t_end=25.0,
            has_speech=False,
            has_visual=True,
            gap_before=10.0,  # > 8s, triggers long-pause
        )

        notes = tag_notes(window, prev_window_end=10.0, next_window_start=None)

        assert "long-pause" in notes
        assert "silent-visual" in notes

    def test_edge_case_gap_exactly_8s(self):
        """Test gap exactly at 8s threshold."""
        window = Window(
            t_start=18.0,
            t_end=23.0,
            has_speech=True,
            has_visual=False,
            gap_before=8.0,
        )

        notes = tag_notes(window, prev_window_end=10.0, next_window_start=None)

        # Exactly 8s should not trigger (> 8, not >= 8)
        assert "long-pause" not in notes

    def test_edge_case_gap_just_over_8s(self):
        """Test gap just over 8s threshold."""
        window = Window(
            t_start=18.1,
            t_end=23.0,
            has_speech=True,
            has_visual=False,
            gap_before=8.1,
        )

        notes = tag_notes(window, prev_window_end=10.0, next_window_start=None)

        # Just over 8s should trigger
        assert "long-pause" in notes

    def test_no_previous_window(self):
        """Test tagging when there's no previous window."""
        window = Window(
            t_start=0.0,
            t_end=5.0,
            has_speech=True,
            has_visual=False,
            gap_before=None,
        )

        notes = tag_notes(window, prev_window_end=None, next_window_start=None)

        # No long-pause without previous window
        assert "long-pause" not in notes
