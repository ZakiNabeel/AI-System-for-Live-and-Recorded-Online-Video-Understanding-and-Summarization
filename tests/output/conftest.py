"""Test fixtures for output tests."""

import base64
import json
import tempfile
from pathlib import Path

import pytest

from src.fusion.schema import FusedDocument, FusedEvent
from src.llm.schema import Chapter, DetectedEvent, KeyPoint, QAPair, Summary
from src.vision.schema import Frame, VisualExtraction


@pytest.fixture
def sample_summary():
    """Create a sample Summary object for testing."""
    return Summary(
        run_id="test-run-001",
        full_summary="This is a comprehensive summary of the video content.",
        short_summary="Video overview in one sentence.",
        key_points=[
            KeyPoint(
                timestamp=10.5,
                text="First important point",
                confidence="high",
                source_event_indices=[0],
            ),
            KeyPoint(
                timestamp=65.0,
                text="Second important point",
                confidence="medium",
                source_event_indices=[1],
            ),
        ],
        events=[
            DetectedEvent(
                timestamp=10.5,
                event_type="topic-start",
                description="Introduction begins",
                source_event_indices=[0],
            ),
            DetectedEvent(
                timestamp=65.0,
                event_type="topic-change",
                description="Moved to main content",
                source_event_indices=[1],
            ),
        ],
        chapters=[
            Chapter(t_start=0.0, t_end=30.0, title="Introduction"),
            Chapter(t_start=30.0, t_end=90.0, title="Main Content"),
            Chapter(t_start=90.0, t_end=120.0, title="Conclusion"),
        ],
        qa_pairs=[
            QAPair(
                question="What is the main topic?",
                answer="The main topic is about video understanding.",
                timestamp=50.0,
            ),
        ],
        model="test-model",
        provider="test-provider",
        chunked=False,
        n_chunks=1,
        elapsed_sec=2.5,
        token_usage={"input_tokens": 1000, "output_tokens": 500},
    )


@pytest.fixture
def sample_fused_document(temp_output_dir):
    """Create a sample FusedDocument for testing."""
    frames_dir = temp_output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5XfS8AAAAASUVORK5CYII="
    )
    (frames_dir / "frame_001.png").write_bytes(png_bytes)

    return FusedDocument(
        run_id="test-run-001",
        duration_sec=120.0,
        language="en",
        events=[
            FusedEvent(
                t_start=0.0,
                t_end=10.0,
                kind="speech",
                speech_text="Welcome to this presentation.",
                visual_text=None,
                visual_caption=None,
                frame_path=None,
                notes=[],
            ),
            FusedEvent(
                t_start=10.0,
                t_end=20.0,
                kind="speech+visual",
                speech_text="Here is a slide.",
                visual_text="Title: Main Point",
                visual_caption="Title slide",
                frame_path=str(frames_dir / "frame_001.png"),
                notes=["scene-change"],
            ),
        ],
        speech_source="whisper",
        ocr_engine="tesseract",
        has_captions=False,
    )


@pytest.fixture
def sample_visual_extraction(temp_output_dir):
    """Create a sample VisualExtraction for testing."""
    frames_dir = temp_output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_000 = frames_dir / "frame_000.png"
    frame_001 = frames_dir / "frame_001.png"
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5XfS8AAAAASUVORK5CYII="
    )
    frame_000.write_bytes(png_bytes)
    frame_001.write_bytes(png_bytes)

    return VisualExtraction(
        run_id="test-run-001",
        frames=[
            Frame(
                timestamp=5.0,
                path=str(frame_000),
                text="Welcome",
                caption="Opening slide",
                lines=["Welcome"],
            ),
            Frame(
                timestamp=15.0,
                path=str(frame_001),
                text="Main Point\nDetails here",
                caption="Main content",
                lines=["Main Point", "Details here"],
            ),
        ],
        total_frames=2,
        sample_rate=1,
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_files_in_temp_dir(temp_output_dir, sample_summary, sample_fused_document, sample_visual_extraction):
    """Create sample input files in a temporary directory."""
    summary_path = temp_output_dir / "summary.raw.json"
    summary_path.write_text(json.dumps({
        "run_id": sample_summary.run_id,
        "full_summary": sample_summary.full_summary,
        "short_summary": sample_summary.short_summary,
        "key_points": [
            {
                "timestamp": kp.timestamp,
                "text": kp.text,
                "confidence": kp.confidence,
                "source_event_indices": kp.source_event_indices,
            }
            for kp in sample_summary.key_points
        ],
        "events": [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "description": e.description,
                "source_event_indices": e.source_event_indices,
            }
            for e in sample_summary.events
        ],
        "chapters": [
            {
                "t_start": c.t_start,
                "t_end": c.t_end,
                "title": c.title,
            }
            for c in sample_summary.chapters
        ],
        "qa_pairs": [
            {
                "question": qa.question,
                "answer": qa.answer,
                "timestamp": qa.timestamp,
            }
            for qa in sample_summary.qa_pairs
        ] if sample_summary.qa_pairs else None,
        "model": sample_summary.model,
        "provider": sample_summary.provider,
        "chunked": sample_summary.chunked,
        "n_chunks": sample_summary.n_chunks,
        "elapsed_sec": sample_summary.elapsed_sec,
        "token_usage": sample_summary.token_usage,
    }))

    fused_path = temp_output_dir / "fused.json"
    fused_path.write_text(json.dumps({
        "run_id": sample_fused_document.run_id,
        "duration_sec": sample_fused_document.duration_sec,
        "language": sample_fused_document.language,
        "events": [
            {
                "t_start": e.t_start,
                "t_end": e.t_end,
                "kind": e.kind,
                "speech_text": e.speech_text,
                "visual_text": e.visual_text,
                "visual_caption": e.visual_caption,
                "frame_path": e.frame_path,
                "notes": e.notes,
            }
            for e in sample_fused_document.events
        ],
        "speech_source": sample_fused_document.speech_source,
        "ocr_engine": sample_fused_document.ocr_engine,
        "has_captions": sample_fused_document.has_captions,
    }))

    visual_path = temp_output_dir / "visual.json"
    visual_path.write_text(json.dumps({
        "run_id": sample_visual_extraction.run_id,
        "frames": [
            {
                "timestamp": f.timestamp,
                "path": f.path,
                "text": f.text,
                "caption": f.caption,
                "lines": f.lines,
            }
            for f in sample_visual_extraction.frames
        ],
        "total_frames": sample_visual_extraction.total_frames,
        "sample_rate": sample_visual_extraction.sample_rate,
    }))

    transcript_path = temp_output_dir / "transcript.aligned.json"
    transcript_path.write_text(json.dumps({
        "run_id": "test-run-001",
        "sentences": [],
        "word_count": 0,
        "language": "en",
        "source": "test",
    }))

    return {
        "summary": summary_path,
        "fused": fused_path,
        "visual": visual_path,
        "transcript": transcript_path,
        "output_dir": temp_output_dir,
    }
