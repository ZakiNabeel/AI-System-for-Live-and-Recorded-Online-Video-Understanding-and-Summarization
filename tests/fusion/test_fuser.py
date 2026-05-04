"""Tests for fusion module."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from dataclasses import asdict

from src.fusion.fuser import (
    fuse,
    _make_windows,
    _bucket_sentences,
    _bucket_frames,
    _best_frame,
    _emit_events,
)
from src.fusion.schema import FusedEvent, FusedDocument, load_fused_document
from src.speech.schema import AlignedTranscript, Sentence, TranscriptSegment, Word
from src.vision.extractor import FrameVisual


# Fixtures
@pytest.fixture
def sample_transcript() -> AlignedTranscript:
    """Create a sample aligned transcript."""
    words = [
        Word(start=0.0, end=1.0, text="Hello", confidence=0.9),
        Word(start=1.0, end=2.0, text="world", confidence=0.95),
    ]
    segment = TranscriptSegment(
        start=0.0,
        end=2.0,
        text="Hello world",
        words=words,
        language="en",
    )
    sentence = Sentence(
        start=0.0,
        end=2.0,
        text="Hello world",
        word_count=2,
        segment_indices=[0],
    )
    return AlignedTranscript(
        segments=[segment],
        language="en",
        duration_sec=10.0,
        source="local-whisper",
        audio_path=None,
        raw_response=None,
        sentences=[sentence],
    )


@pytest.fixture
def sample_frames() -> list[FrameVisual]:
    """Create sample visual frames."""
    return [
        FrameVisual(
            timestamp=0.5,
            frame_index=0,
            frame_path=Path("data/frames/run_id/frame_0000.png"),
            enhanced_path=None,
            text="Hello",
            lines=[],
            caption="Person greeting",
            caption_source="claude",
            has_text=True,
            raw_ocr=None,
        ),
        FrameVisual(
            timestamp=5.0,
            frame_index=5,
            frame_path=Path("data/frames/run_id/frame_0005.png"),
            enhanced_path=None,
            text="",
            lines=[],
            caption=None,
            caption_source=None,
            has_text=False,
            raw_ocr=None,
        ),
    ]


@pytest.fixture
def temp_transcript_json(tmp_path: Path, sample_transcript: AlignedTranscript) -> Path:
    """Save sample transcript to temp JSON."""
    from src.speech.schema import save_transcript

    path = tmp_path / "transcript.json"
    save_transcript(sample_transcript, path)
    return path


@pytest.fixture
def temp_visual_json(tmp_path: Path, sample_frames: list[FrameVisual]) -> Path:
    """Save sample frames to temp JSON."""
    path = tmp_path / "visual.json"
    payload = {
        "version": "1",
        "ocr_engine": "tesseract",
        "has_captions": True,
        "frames": [asdict(f) for f in sample_frames],
    }
    # Convert Path objects to strings
    for frame in payload["frames"]:
        frame["frame_path"] = str(frame["frame_path"])
        if frame["enhanced_path"]:
            frame["enhanced_path"] = str(frame["enhanced_path"])
    
    with path.open("w") as f:
        json.dump(payload, f)
    return path


# Test window building
class TestWindowBuilding:
    def test_make_windows_even_division(self):
        """Test window creation with even division."""
        windows = _make_windows(10.0, 5.0)
        assert len(windows) == 2
        assert windows[0] == (0.0, 5.0)
        assert windows[1] == (5.0, 10.0)

    def test_make_windows_with_remainder(self):
        """Test window creation with time remainder."""
        windows = _make_windows(12.0, 5.0)
        assert len(windows) == 3
        assert windows[0] == (0.0, 5.0)
        assert windows[1] == (5.0, 10.0)
        assert windows[2] == (10.0, 12.0)

    def test_make_windows_small_duration(self):
        """Test window creation for very short duration."""
        windows = _make_windows(1.0, 5.0)
        assert len(windows) == 1
        assert windows[0] == (0.0, 1.0)


# Test bucketing
class TestBucketing:
    def test_bucket_sentences_simple(self, sample_transcript: AlignedTranscript):
        """Test sentence bucketing."""
        buckets = _bucket_sentences(sample_transcript.sentences, 5.0)
        # Single sentence at 0-2s, should be in window 0
        assert 0 in buckets
        assert len(buckets[0]) == 1
        assert buckets[0][0].text == "Hello world"

    def test_bucket_sentences_spanning_windows(self):
        """Test sentence spanning multiple windows."""
        sentence = Sentence(
            start=3.0,
            end=7.0,
            text="spanning",
            word_count=1,
            segment_indices=[0],
        )
        buckets = _bucket_sentences([sentence], 5.0)
        # Should be in windows 0 and 1
        assert 0 in buckets
        assert 1 in buckets
        assert len(buckets[0]) == 1
        assert len(buckets[1]) == 1

    def test_bucket_sentences_invalid_skip(self):
        """Test that invalid sentences are skipped."""
        invalid = Sentence(
            start=5.0,
            end=2.0,  # Invalid: end < start
            text="invalid",
            word_count=1,
            segment_indices=[0],
        )
        buckets = _bucket_sentences([invalid], 5.0)
        assert len(buckets) == 0

    def test_bucket_frames_simple(self, sample_frames: list[FrameVisual]):
        """Test frame bucketing."""
        buckets = _bucket_frames(sample_frames, 5.0)
        # Frame at 0.5s → window 0, frame at 5.0s → window 1
        assert 0 in buckets
        assert 1 in buckets
        assert len(buckets[0]) == 1
        assert len(buckets[1]) == 1


# Test best frame selection
class TestBestFrameSelection:
    def test_best_frame_none(self):
        """Test selection from empty list."""
        assert _best_frame([]) is None

    def test_best_frame_single(self, sample_frames: list[FrameVisual]):
        """Test selection from single frame."""
        best = _best_frame(sample_frames[:1])
        assert best is sample_frames[0]

    def test_best_frame_prefers_text(self):
        """Test that frame with more text is preferred."""
        frame_short = FrameVisual(
            timestamp=1.0,
            frame_index=1,
            frame_path=Path("f1.png"),
            enhanced_path=None,
            text="Hi",
            lines=[],
            caption=None,
            caption_source=None,
            has_text=True,
            raw_ocr=None,
        )
        frame_long = FrameVisual(
            timestamp=1.0,
            frame_index=2,
            frame_path=Path("f2.png"),
            enhanced_path=None,
            text="Hello world this is longer",
            lines=[],
            caption=None,
            caption_source=None,
            has_text=True,
            raw_ocr=None,
        )
        best = _best_frame([frame_short, frame_long])
        assert best is frame_long

    def test_best_frame_prefers_caption(self):
        """Test that frame with caption is preferred."""
        frame_no_caption = FrameVisual(
            timestamp=1.0,
            frame_index=1,
            frame_path=Path("f1.png"),
            enhanced_path=None,
            text="text",
            lines=[],
            caption=None,
            caption_source=None,
            has_text=True,
            raw_ocr=None,
        )
        frame_with_caption = FrameVisual(
            timestamp=1.0,
            frame_index=2,
            frame_path=Path("f2.png"),
            enhanced_path=None,
            text="text",
            lines=[],
            caption="Scene",
            caption_source="llava",
            has_text=True,
            raw_ocr=None,
        )
        best = _best_frame([frame_no_caption, frame_with_caption])
        assert best is frame_with_caption


# Test event emission
class TestEventEmission:
    def test_emit_events_speech_only(self, sample_transcript: AlignedTranscript):
        """Test event emission for speech-only content."""
        sentence_buckets = _bucket_sentences(sample_transcript.sentences, 5.0)
        frame_buckets = {}
        windows = _make_windows(10.0, 5.0)

        events = _emit_events(windows, sentence_buckets, frame_buckets, [])

        assert len(events) == 1
        assert events[0].kind == "speech"
        assert "Hello world" in events[0].speech_text
        assert events[0].visual_text is None

    def test_emit_events_visual_only(self, sample_frames: list[FrameVisual]):
        """Test event emission for visual-only content."""
        sentence_buckets = {}
        frame_buckets = _bucket_frames(sample_frames, 5.0)
        windows = _make_windows(10.0, 5.0)

        events = _emit_events(windows, sentence_buckets, frame_buckets, sample_frames)

        # Should have 2 events (one per window with content)
        assert len(events) >= 1
        visual_events = [e for e in events if e.kind == "visual"]
        assert len(visual_events) >= 1

    def test_emit_events_skips_silent(self):
        """Test that silent+no-visual windows are skipped."""
        windows = _make_windows(10.0, 5.0)
        events = _emit_events(windows, {}, {}, [])
        assert len(events) == 0

    def test_emit_events_sorting(self, sample_transcript: AlignedTranscript, sample_frames: list[FrameVisual]):
        """Test that events are sorted by t_start."""
        sentence_buckets = _bucket_sentences(sample_transcript.sentences, 5.0)
        frame_buckets = _bucket_frames(sample_frames, 5.0)
        windows = _make_windows(10.0, 5.0)

        events = _emit_events(windows, sentence_buckets, frame_buckets, sample_frames)

        for i in range(len(events) - 1):
            assert events[i].t_start <= events[i + 1].t_start


# Integration tests
class TestFusionIntegration:
    def test_fuse_produces_valid_document(
        self,
        tmp_path: Path,
        temp_transcript_json: Path,
        temp_visual_json: Path,
    ):
        """Test that fuse() produces a valid FusedDocument."""
        output_path = tmp_path / "fused.json"
        
        doc = fuse(
            temp_transcript_json,
            temp_visual_json,
            output_path,
            run_id="test-run-001",
            window_sec=5.0,
        )

        assert doc.run_id == "test-run-001"
        assert doc.duration_sec > 0
        assert len(doc.events) > 0
        assert output_path.exists()

    def test_fuse_output_persistence(
        self,
        tmp_path: Path,
        temp_transcript_json: Path,
        temp_visual_json: Path,
    ):
        """Test that fused document can be loaded back from disk."""
        output_path = tmp_path / "fused.json"
        
        doc1 = fuse(
            temp_transcript_json,
            temp_visual_json,
            output_path,
            run_id="test-run-001",
        )

        # Load back
        doc2 = load_fused_document(output_path)

        assert doc1.run_id == doc2.run_id
        assert doc1.duration_sec == doc2.duration_sec
        assert len(doc1.events) == len(doc2.events)

    def test_fuse_determinism(
        self,
        tmp_path: Path,
        temp_transcript_json: Path,
        temp_visual_json: Path,
    ):
        """Test that fuse produces identical output on repeated runs."""
        output1 = tmp_path / "fused1.json"
        output2 = tmp_path / "fused2.json"

        fuse(temp_transcript_json, temp_visual_json, output1, run_id="test")
        fuse(temp_transcript_json, temp_visual_json, output2, run_id="test")

        with output1.open() as f1, output2.open() as f2:
            assert f1.read() == f2.read()

    def test_fuse_missing_transcript(self, tmp_path: Path):
        """Test that fuse raises when transcript missing."""
        with pytest.raises(FileNotFoundError):
            fuse(
                tmp_path / "missing.json",
                tmp_path / "visual.json",
                tmp_path / "fused.json",
                run_id="test",
            )

    def test_fuse_with_chunking(
        self,
        tmp_path: Path,
        temp_transcript_json: Path,
        temp_visual_json: Path,
    ):
        """Test that fuse produces chunked variant when requested."""
        output_path = tmp_path / "fused.json"
        
        fuse(
            temp_transcript_json,
            temp_visual_json,
            output_path,
            run_id="test-run-001",
            max_chunk_tokens=500,  # Small chunks for testing
        )

        chunked_path = tmp_path / "fused_chunked.json"
        assert chunked_path.exists()

        with chunked_path.open() as f:
            payload = json.load(f)
            assert "chunks" in payload
            assert len(payload["chunks"]) > 0


# Edge case tests
class TestEdgeCases:
    def test_transcript_longer_than_visual(self, tmp_path: Path, sample_transcript: AlignedTranscript, sample_frames: list[FrameVisual]):
        """Test handling when transcript is longer than visual duration."""
        from src.speech.schema import save_transcript

        # Extend transcript
        long_transcript = AlignedTranscript(
            segments=sample_transcript.segments,
            language=sample_transcript.language,
            duration_sec=20.0,  # Longer than frames
            source=sample_transcript.source,
            audio_path=sample_transcript.audio_path,
            raw_response=sample_transcript.raw_response,
            sentences=sample_transcript.sentences,
        )
        
        trans_path = tmp_path / "transcript.json"
        save_transcript(long_transcript, trans_path)

        visual_path = tmp_path / "visual.json"
        payload = {
            "version": "1",
            "ocr_engine": "tesseract",
            "has_captions": True,
            "frames": [asdict(f) for f in sample_frames],
        }
        for frame in payload["frames"]:
            frame["frame_path"] = str(frame["frame_path"])
            if frame["enhanced_path"]:
                frame["enhanced_path"] = str(frame["enhanced_path"])
        
        with visual_path.open("w") as f:
            json.dump(payload, f)

        output_path = tmp_path / "fused.json"
        doc = fuse(trans_path, visual_path, output_path, run_id="test")

        # Duration should be max of both
        assert doc.duration_sec == 20.0

    def test_empty_transcript(self, tmp_path: Path):
        """Test handling of empty transcript."""
        from src.speech.schema import save_transcript

        empty_transcript = AlignedTranscript(
            segments=[],
            language="en",
            duration_sec=5.0,
            source="local-whisper",
            audio_path=None,
            raw_response=None,
            sentences=[],
        )

        trans_path = tmp_path / "transcript.json"
        save_transcript(empty_transcript, trans_path)

        visual_path = tmp_path / "visual.json"
        payload = {
            "version": "1",
            "ocr_engine": "tesseract",
            "has_captions": False,
            "frames": [],
        }
        with visual_path.open("w") as f:
            json.dump(payload, f)

        output_path = tmp_path / "fused.json"
        doc = fuse(trans_path, visual_path, output_path, run_id="test")

        # Should produce document with no events
        assert len(doc.events) == 0
