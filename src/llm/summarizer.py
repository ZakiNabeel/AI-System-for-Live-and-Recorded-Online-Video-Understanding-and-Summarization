"""Main summarization logic."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from src.fusion.schema import FusedDocument, load_fused_document
from src.fusion.chunker import chunk_events
from src.llm.parsing import validate_timestamps
from src.llm.prompts_system import get_chunk_prompt, get_global_prompt
from src.llm.providers import get_provider
from src.llm.schema import Chapter, DetectedEvent, KeyPoint, QAPair, Summary

LOGGER = logging.getLogger(__name__)


def summarize(
    fused_path: Path,
    output_path: Path,
    *,
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic",
    model: str | None = None,
    style: Literal["concise", "detailed", "bullet-only"] = "detailed",
    enable_qa: bool = False,
    domain: str | None = None,
    rolling: bool = False,
    rolling_state_path: Path | None = None,
) -> Summary:
    """
    Summarize a fused document using an LLM.

    Args:
        fused_path: Path to fused.json
        output_path: Path to write summary.raw.json
        provider: LLM provider ("anthropic", "openai", "ollama")
        model: Override default model
        style: Summary style ("concise", "detailed", "bullet-only")
        enable_qa: Generate Q&A pairs
        domain: Domain-specific hooks (e.g., "education")
        rolling: Use rolling-summary mode (for live streams)
        rolling_state_path: Path to rolling state file

    Returns:
        Summary object

    Raises:
        FileNotFoundError: If fused_path not found
        ValueError: If invalid parameters
    """
    fused_path = Path(fused_path)
    output_path = Path(output_path)

    if not fused_path.exists():
        raise FileNotFoundError(f"Fused document not found: {fused_path}")

    LOGGER.info(f"Loading fused document from {fused_path}")
    fused_doc = load_fused_document(fused_path)

    # Handle empty document
    if not fused_doc.events:
        LOGGER.warning("Empty fused document, producing stub summary")
        summary = Summary(
            run_id=fused_doc.run_id,
            full_summary="(no content detected)",
            short_summary="(no content)",
            key_points=[],
            events=[],
            chapters=[],
            qa_pairs=[] if enable_qa else None,
            model=model or "unknown",
            provider=provider,
            chunked=False,
            n_chunks=0,
            elapsed_sec=0.0,
        )
        _save_summary(summary, output_path)
        return summary

    start_time = time.time()
    llm_provider = get_provider(provider, model)

    # Determine if we need multi-pass
    events_json = json.dumps([vars(e) for e in fused_doc.events], default=str, indent=2)
    events_size_tokens = len(events_json) // 4  # Rough estimate

    # Context window safety limits
    context_limits = {
        "anthropic": 180000,
        "openai": 120000,
        "gemini": 800000,
        "ollama": 4000,
    }
    safe_limit = context_limits.get(provider, 8000) * 0.7  # 70% safety margin

    LOGGER.info(f"Events size estimate: ~{events_size_tokens} tokens, safe limit: ~{safe_limit:.0f}")

    if events_size_tokens > safe_limit:
        LOGGER.info("Document requires multi-pass summarization")
        summary = _summarize_multipass(
            fused_doc,
            llm_provider,
            style=style,
            enable_qa=enable_qa,
            domain=domain,
        )
    else:
        LOGGER.info("Using single-pass summarization")
        summary = _summarize_singlepass(
            fused_doc,
            llm_provider,
            style=style,
            enable_qa=enable_qa,
            domain=domain,
        )

    elapsed = time.time() - start_time
    summary = Summary(
        run_id=summary.run_id,
        full_summary=summary.full_summary,
        short_summary=summary.short_summary,
        key_points=summary.key_points,
        events=summary.events,
        chapters=summary.chapters,
        qa_pairs=summary.qa_pairs,
        model=model or llm_provider.model,
        provider=provider,
        chunked=summary.chunked,
        n_chunks=summary.n_chunks,
        elapsed_sec=elapsed,
        token_usage=summary.token_usage,
    )

    LOGGER.info(f"Summarization complete ({elapsed:.1f}s). Saving to {output_path}")
    _save_summary(summary, output_path)

    return summary


def _summarize_singlepass(
    fused_doc: FusedDocument,
    llm_provider: Any,
    style: str = "detailed",
    enable_qa: bool = False,
    domain: str | None = None,
) -> Summary:
    """Single-pass summarization for small documents."""
    events_json = json.dumps([vars(e) for e in fused_doc.events], default=str, indent=2)

    # Per-chunk pass (on whole doc)
    LOGGER.info("Pass 1: Local summary")
    system_chunk, user_chunk = get_chunk_prompt(events_json)
    chunk_payload, chunk_usage = llm_provider.complete_json(system_chunk, user_chunk)
    chunk_payload = validate_timestamps(chunk_payload, fused_doc.duration_sec)

    # Global synthesis (on single "chunk")
    LOGGER.info("Pass 2: Global synthesis")
    local_summaries = [chunk_payload]
    local_summaries_json = json.dumps(local_summaries, default=str, indent=2)
    system_global, user_global = get_global_prompt(local_summaries_json)
    global_payload, global_usage = llm_provider.complete_json(system_global, user_global)
    global_payload = validate_timestamps(global_payload, fused_doc.duration_sec)

    # Build Summary
    return _payload_to_summary(
        fused_doc,
        global_payload,
        chunked=False,
        n_chunks=1,
        token_usage={
            "input_tokens": chunk_usage.get("input_tokens", 0) + global_usage.get("input_tokens", 0),
            "output_tokens": chunk_usage.get("output_tokens", 0) + global_usage.get("output_tokens", 0),
        },
    )


def _summarize_multipass(
    fused_doc: FusedDocument,
    llm_provider: Any,
    style: str = "detailed",
    enable_qa: bool = False,
    domain: str | None = None,
) -> Summary:
    """Multi-pass summarization for large documents."""
    # Chunk the document
    LOGGER.info("Chunking document for multi-pass processing")
    result = chunk_events(fused_doc, max_tokens=8000, overlap_tokens=200)
    chunks = result.chunks
    LOGGER.info(f"Created {len(chunks)} chunks")

    # Pass 1: Summarize each chunk
    LOGGER.info("Pass 1: Local summaries for each chunk")
    local_summaries = []
    total_tokens = {"input": 0, "output": 0}

    for i, chunk in enumerate(chunks):
        LOGGER.debug(f"  Chunk {i + 1}/{len(chunks)}")
        events_json = json.dumps([vars(e) for e in chunk.events], default=str, indent=2)
        system_chunk, user_chunk = get_chunk_prompt(events_json)

        chunk_payload, chunk_usage = llm_provider.complete_json(system_chunk, user_chunk)
        chunk_payload = validate_timestamps(chunk_payload, chunk.duration_sec)
        local_summaries.append(chunk_payload)

        total_tokens["input"] += chunk_usage.get("input_tokens", 0)
        total_tokens["output"] += chunk_usage.get("output_tokens", 0)

    # Pass 2: Synthesize global summary
    LOGGER.info("Pass 2: Global synthesis")
    local_summaries_json = json.dumps(local_summaries, default=str, indent=2)
    system_global, user_global = get_global_prompt(local_summaries_json)
    global_payload, global_usage = llm_provider.complete_json(system_global, user_global)
    global_payload = validate_timestamps(global_payload, fused_doc.duration_sec)

    total_tokens["input"] += global_usage.get("input_tokens", 0)
    total_tokens["output"] += global_usage.get("output_tokens", 0)

    # Build Summary
    return _payload_to_summary(
        fused_doc,
        global_payload,
        chunked=True,
        n_chunks=len(chunks),
        token_usage=total_tokens,
    )


def _payload_to_summary(
    fused_doc: FusedDocument,
    payload: dict[str, Any],
    chunked: bool = False,
    n_chunks: int = 0,
    token_usage: dict[str, int] | None = None,
) -> Summary:
    """Convert LLM payload to Summary object."""
    # Extract fields with sensible defaults
    full_summary = payload.get("full_summary", "")
    short_summary = payload.get("short_summary", "")

    # Parse key points
    key_points = []
    for kp in payload.get("merged_key_points", payload.get("key_points", [])):
        try:
            key_points.append(
                KeyPoint(
                    timestamp=float(kp.get("timestamp", 0)),
                    text=str(kp.get("text", "")),
                    confidence=kp.get("confidence", "medium"),
                    source_event_indices=kp.get("source_event_indices", []),
                )
            )
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Skipping invalid key point: {e}")

    # Parse events
    events = []
    for evt in payload.get("merged_events", payload.get("events", [])):
        try:
            events.append(
                DetectedEvent(
                    timestamp=float(evt.get("timestamp", 0)),
                    event_type=str(evt.get("event_type", "")),
                    description=str(evt.get("description", "")),
                    source_event_indices=evt.get("source_event_indices", []),
                )
            )
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Skipping invalid event: {e}")

    # Parse chapters
    chapters = []
    for ch in payload.get("chapters", []):
        try:
            chapters.append(
                Chapter(
                    t_start=float(ch.get("t_start", 0)),
                    t_end=float(ch.get("t_end", 0)),
                    title=str(ch.get("title", "")),
                )
            )
        except (ValueError, TypeError) as e:
            LOGGER.warning(f"Skipping invalid chapter: {e}")

    # Sort chapters by t_start
    chapters = sorted(chapters, key=lambda ch: ch.t_start)

    # Parse Q&A pairs
    qa_pairs = None
    if "qa_pairs" in payload:
        qa_pairs = []
        for qa in payload["qa_pairs"]:
            try:
                qa_pairs.append(
                    QAPair(
                        question=str(qa.get("question", "")),
                        answer=str(qa.get("answer", "")),
                        timestamp=float(qa.get("timestamp", 0)),
                    )
                )
            except (ValueError, TypeError) as e:
                LOGGER.warning(f"Skipping invalid Q&A pair: {e}")

    return Summary(
        run_id=fused_doc.run_id,
        full_summary=full_summary,
        short_summary=short_summary,
        key_points=key_points,
        events=events,
        chapters=chapters,
        qa_pairs=qa_pairs,
        model="",  # Will be set by caller
        provider="",  # Will be set by caller
        chunked=chunked,
        n_chunks=n_chunks,
        elapsed_sec=0.0,  # Will be set by caller
        token_usage=token_usage or {},
    )


def _save_summary(summary: Summary, path: Path) -> None:
    """Save summary to JSON."""
    from src.llm.schema import save_summary
    save_summary(summary, path)
