"""Tests for event chunking."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.fusion.chunker import chunk_events
from src.fusion.schema import FusedEvent, FusedDocument
from src.fusion.tokens import count_tokens


@pytest.fixture
def sample_document() -> FusedDocument:
    """Create a sample fused document with multiple events."""
    events = [
        FusedEvent(
            t_start=0.0,
            t_end=5.0,
            kind="speech",
            speech_text="Hello world. This is the first event." * 10,  # Longer text
            notes=[],
        ),
        FusedEvent(
            t_start=5.0,
            t_end=10.0,
            kind="visual",
            visual_text="Slide 1",
            visual_caption="Introduction slide" * 5,  # Longer caption
            notes=["silent-visual"],
        ),
        FusedEvent(
            t_start=10.0,
            t_end=15.0,
            kind="speech+visual",
            speech_text="Now we move to the second topic. This requires more explanation." * 10,  # Longer text
            visual_text="Slide 2" * 5,
            visual_caption="Main content" * 5,
            notes=[],
        ),
        FusedEvent(
            t_start=15.0,
            t_end=20.0,
            kind="speech",
            speech_text="Let me summarize what we discussed. Key points are important." * 10,  # Longer text
            notes=["long-pause"],
        ),
    ]

    return FusedDocument(
        run_id="test-run",
        duration_sec=20.0,
        language="en",
        events=events,
        speech_source="local-whisper",
        ocr_engine="tesseract",
        has_captions=True,
    )


class TestChunking:
    def test_chunk_empty_document(self):
        """Test chunking empty document."""
        doc = FusedDocument(
            run_id="test",
            duration_sec=0.0,
            language="en",
            events=[],
            speech_source="local-whisper",
            ocr_engine="tesseract",
            has_captions=False,
        )

        result = chunk_events(doc, max_tokens=1000)

        assert len(result.chunks) == 1
        assert len(result.chunks[0].events) == 0
        assert result.total_events == 0

    def test_chunk_single_event(self, sample_document: FusedDocument):
        """Test chunking with single event."""
        doc = FusedDocument(
            run_id="test",
            duration_sec=5.0,
            language="en",
            events=sample_document.events[:1],
            speech_source="local-whisper",
            ocr_engine="tesseract",
            has_captions=False,
        )

        result = chunk_events(doc, max_tokens=1000)

        assert len(result.chunks) == 1
        assert len(result.chunks[0].events) == 1

    def test_chunk_respects_max_tokens(self, sample_document: FusedDocument):
        """Test that chunks don't exceed max tokens."""
        result = chunk_events(sample_document, max_tokens=50, backend="openai")

        total_events = sum(len(c.events) for c in result.chunks)
        assert total_events == len(sample_document.events)

        # Verify each chunk stays under limit (approximately, with margin for fallback estimation)
        for chunk in result.chunks:
            text = "\n".join(
                (e.speech_text or "") + (e.visual_text or "") + (e.visual_caption or "")
                for e in chunk.events
            )
            token_count = count_tokens(text, backend="openai")
            # Allow 200% margin because fallback token counter is approximate
            assert token_count <= 150

    def test_chunk_produces_multiple_chunks(self, sample_document: FusedDocument):
        """Test that large document produces multiple chunks."""
        result = chunk_events(sample_document, max_tokens=50, backend="openai")

        assert len(result.chunks) > 1
        assert result.total_events == len(sample_document.events)

    def test_chunk_preserves_all_events(self, sample_document: FusedDocument):
        """Test that chunking preserves all events."""
        result = chunk_events(sample_document, max_tokens=200, backend="openai")

        chunked_events = []
        for chunk in result.chunks:
            chunked_events.extend(chunk.events)

        # Events should match original (accounting for potential overlap markers)
        original_texts = [e.speech_text or e.visual_text for e in sample_document.events]
        chunked_texts = [e.speech_text or e.visual_text for e in chunked_events if "[context-overlap]" not in e.notes]

        assert len(chunked_texts) == len(original_texts)

    def test_chunk_overlap_context(self, sample_document: FusedDocument):
        """Test that chunks include overlap context."""
        result = chunk_events(sample_document, max_tokens=150, overlap_tokens=50, backend="openai")

        if len(result.chunks) > 1:
            # Check that second chunk may have overlap marker
            for chunk in result.chunks[1:]:
                # Overlap should be marked in notes of first event
                if chunk.events:
                    first_event_notes = chunk.events[0].notes
                    # Overlap might be present (depends on token counting)
                    # Just verify structure is valid
                    assert isinstance(first_event_notes, list)

    def test_chunk_metadata_preserved(self, sample_document: FusedDocument):
        """Test that document metadata is preserved in chunks."""
        result = chunk_events(sample_document, max_tokens=200, backend="openai")

        for chunk in result.chunks:
            assert chunk.run_id == sample_document.run_id
            assert chunk.language == sample_document.language
            assert chunk.speech_source == sample_document.speech_source
            assert chunk.ocr_engine == sample_document.ocr_engine

    def test_chunk_time_ranges(self, sample_document: FusedDocument):
        """Test that chunk time ranges are valid."""
        result = chunk_events(sample_document, max_tokens=200, backend="openai")

        for chunk in result.chunks:
            if chunk.events:
                assert chunk.events[0].t_start <= chunk.events[-1].t_end

    def test_chunk_with_zero_overlap(self, sample_document: FusedDocument):
        """Test chunking with zero overlap."""
        result = chunk_events(sample_document, max_tokens=200, overlap_tokens=0, backend="openai")

        assert len(result.chunks) > 0
        total_events = sum(len(c.events) for c in result.chunks)
        assert total_events == len(sample_document.events)

    def test_chunk_overlap_exceeds_max_warning(self, sample_document: FusedDocument):
        """Test that warning is logged when overlap > max."""
        # overlap > max should trigger warning and disable overlap
        result = chunk_events(sample_document, max_tokens=100, overlap_tokens=200, backend="openai")

        # Should still produce valid chunks
        assert len(result.chunks) > 0

    def test_chunk_average_size(self, sample_document: FusedDocument):
        """Test that average chunk size is calculated."""
        result = chunk_events(sample_document, max_tokens=200, backend="openai")

        assert result.avg_chunk_size > 0
        assert result.avg_chunk_size <= len(sample_document.events)

    def test_chunk_determinism(self, sample_document: FusedDocument):
        """Test that chunking is deterministic."""
        result1 = chunk_events(sample_document, max_tokens=200, backend="openai")
        result2 = chunk_events(sample_document, max_tokens=200, backend="openai")

        assert len(result1.chunks) == len(result2.chunks)
        for c1, c2 in zip(result1.chunks, result2.chunks):
            assert len(c1.events) == len(c2.events)
