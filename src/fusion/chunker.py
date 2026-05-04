"""Chunk fused events into LLM-context-sized documents."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .schema import FusedDocument, FusedEvent
from .tokens import count_tokens

LOGGER = logging.getLogger(__name__)


@dataclass
class ChunkingResult:
    """Result of chunking operation."""

    chunks: list[FusedDocument] = field(default_factory=list)
    total_events: int = 0
    avg_chunk_size: float = 0.0


def chunk_events(
    doc: FusedDocument,
    max_tokens: int = 8000,
    overlap_tokens: int = 200,
    backend: str = "openai",
) -> ChunkingResult:
    """
    Chunk fused events into LLM-context-sized documents.

    Uses greedy fill: adds events until adding the next would exceed max_tokens.
    When rolling over, includes trailing text from previous chunk for context.

    Args:
        doc: Complete fused document
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Tokens of context to overlap between chunks
        backend: Token counter backend ("openai" or "anthropic")

    Returns:
        ChunkingResult with list of chunked documents
    """
    if not doc.events:
        return ChunkingResult(chunks=[doc], total_events=0)

    if max_tokens <= overlap_tokens:
        LOGGER.warning(
            f"max_tokens ({max_tokens}) <= overlap_tokens ({overlap_tokens}), disabling overlap"
        )
        overlap_tokens = 0

    chunks: list[FusedDocument] = []
    current_chunk_events: list[FusedEvent] = []
    current_chunk_tokens = 0
    overlap_text = ""

    for event_idx, event in enumerate(doc.events):
        # Build event text for token counting
        event_text = _event_to_text(event)
        event_tokens = count_tokens(event_text, backend=backend)

        # Check if adding this event would exceed limit
        test_tokens = current_chunk_tokens + event_tokens
        if current_chunk_events and test_tokens > max_tokens:
            # Flush current chunk and start new one with overlap
            chunk = _make_chunk(
                doc, current_chunk_events, overlap_text, event_idx - len(current_chunk_events)
            )
            chunks.append(chunk)
            LOGGER.debug(
                f"Chunk {len(chunks)}: {len(current_chunk_events)} events, ~{current_chunk_tokens} tokens"
            )

            # Start new chunk with overlap context
            overlap_text = _extract_overlap_text(current_chunk_events, overlap_tokens, backend)
            current_chunk_events = [event]
            current_chunk_tokens = event_tokens

        else:
            current_chunk_events.append(event)
            current_chunk_tokens = test_tokens

    # Flush final chunk
    if current_chunk_events:
        chunk = _make_chunk(doc, current_chunk_events, overlap_text, len(doc.events) - len(current_chunk_events))
        chunks.append(chunk)
        LOGGER.debug(
            f"Chunk {len(chunks)}: {len(current_chunk_events)} events, ~{current_chunk_tokens} tokens"
        )

    avg_chunk_size = sum(len(c.events) for c in chunks) / len(chunks) if chunks else 0

    return ChunkingResult(
        chunks=chunks,
        total_events=len(doc.events),
        avg_chunk_size=avg_chunk_size,
    )


def _event_to_text(event: FusedEvent) -> str:
    """Convert event to text for token counting."""
    parts = []
    if event.speech_text:
        parts.append(event.speech_text)
    if event.visual_text:
        parts.append(event.visual_text)
    if event.visual_caption:
        parts.append(event.visual_caption)
    return " ".join(parts)


def _extract_overlap_text(
    events: list[FusedEvent],
    overlap_tokens: int,
    backend: str,
) -> str:
    """Extract trailing text from events to use as overlap context."""
    if not events or overlap_tokens <= 0:
        return ""

    combined = _event_to_text(events[-1])
    current_tokens = count_tokens(combined, backend=backend)

    # Work backwards to accumulate overlap text
    for event in reversed(events[:-1]):
        event_text = _event_to_text(event)
        event_tokens = count_tokens(event_text, backend=backend)

        if current_tokens + event_tokens > overlap_tokens:
            break

        combined = event_text + " " + combined
        current_tokens += event_tokens

    return combined


def _make_chunk(
    doc: FusedDocument,
    events: list[FusedEvent],
    overlap_text: str,
    start_index: int,
) -> FusedDocument:
    """Create a chunk document from a slice of events."""
    if not events:
        return FusedDocument(
            run_id=doc.run_id,
            duration_sec=0.0,
            language=doc.language,
            events=[],
            speech_source=doc.speech_source,
            ocr_engine=doc.ocr_engine,
            has_captions=doc.has_captions,
        )

    t_start = events[0].t_start
    t_end = events[-1].t_end

    # Add overlap text as metadata in notes of first event if present
    chunk_events = list(events)
    if overlap_text and chunk_events:
        first_event = chunk_events[0]
        new_notes = list(first_event.notes) + ["[context-overlap]"]
        chunk_events[0] = FusedEvent(
            t_start=first_event.t_start,
            t_end=first_event.t_end,
            kind=first_event.kind,
            speech_text=first_event.speech_text,
            speech_segment_indices=first_event.speech_segment_indices,
            visual_text=first_event.visual_text,
            visual_caption=first_event.visual_caption,
            frame_index=first_event.frame_index,
            frame_path=first_event.frame_path,
            notes=new_notes,
        )

    return FusedDocument(
        run_id=doc.run_id,
        duration_sec=t_end - t_start,
        language=doc.language,
        events=chunk_events,
        speech_source=doc.speech_source,
        ocr_engine=doc.ocr_engine,
        has_captions=doc.has_captions,
    )
