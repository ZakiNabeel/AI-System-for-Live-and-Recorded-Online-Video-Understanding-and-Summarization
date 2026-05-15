"""Main fusion module: merge transcript and visual data into multimodal events."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.speech.schema import AlignedTranscript, Sentence, load_transcript
from src.vision.extractor import FrameVisual

from .notes import Window, tag_notes
from .schema import FusedDocument, FusedEvent, save_fused_document
from .tokens import count_tokens

LOGGER = logging.getLogger(__name__)


def fuse(
    transcript_path: Path,
    visual_path: Path,
    output_path: Path,
    *,
    run_id: str,
    window_sec: float = 5.0,
    max_chunk_tokens: int | None = None,
    chunk_overlap_tokens: int = 200,
) -> FusedDocument:
    """
    Merge aligned transcript and visual extraction into fused multimodal events.

    Args:
        transcript_path: Path to transcript.aligned.json
        visual_path: Path to visual.json
        output_path: Path to write fused.json
        run_id: Unique run identifier
        window_sec: Time window granularity (seconds)
        max_chunk_tokens: If set, also write fused_chunked.json
        chunk_overlap_tokens: Context overlap between chunks

    Returns:
        FusedDocument with merged events

    Raises:
        FileNotFoundError: If input files not found
        ValueError: If inputs invalid
    """
    transcript_path = Path(transcript_path)
    visual_path = Path(visual_path)
    output_path = Path(output_path)

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not visual_path.exists():
        raise FileNotFoundError(f"Visual extraction not found: {visual_path}")

    LOGGER.info(f"Loading transcript from {transcript_path}")
    transcript = load_transcript(transcript_path)
    if not isinstance(transcript, AlignedTranscript):
        raise ValueError("Transcript must be aligned (from Plan 2.3)")

    LOGGER.info(f"Loading visual extraction from {visual_path}")
    frames = _load_visual_frames(visual_path)

    # Determine overall duration
    duration = max(
        transcript.duration_sec,
        max([f.timestamp for f in frames], default=0.0),
    )
    LOGGER.info(f"Duration: {duration:.1f}s, {len(transcript.sentences)} sentences, {len(frames)} frames")

    # Build time-windowed grid
    windows = _make_windows(duration, window_sec)
    LOGGER.info(f"Created {len(windows)} windows ({window_sec}s each)")

    # Bucket sentences and frames
    sentence_buckets = _bucket_sentences(transcript.sentences, window_sec)
    frame_buckets = _bucket_frames(frames, window_sec)

    # Emit events
    events = _emit_events(windows, sentence_buckets, frame_buckets, frames)
    LOGGER.info(f"Emitted {len(events)} events")

    # Create fused document
    doc = FusedDocument(
        run_id=run_id,
        duration_sec=duration,
        language=transcript.language,
        events=events,
        speech_source=transcript.source,
        ocr_engine="unknown",  # Will be populated from visual.json if present
        has_captions=False,  # Will be set based on frame data
    )

    # Parse ocr_engine and has_captions from visual metadata
    try:
        with visual_path.open("r", encoding="utf-8") as f:
            visual_json = json.load(f)
            doc = FusedDocument(
                run_id=run_id,
                duration_sec=duration,
                language=transcript.language,
                events=events,
                speech_source=transcript.source,
                ocr_engine=visual_json.get("ocr_engine", "unknown"),
                has_captions=visual_json.get("has_captions", False),
            )
    except Exception as e:
        LOGGER.warning(f"Failed to read visual metadata: {e}")

    # Save main output
    LOGGER.info(f"Saving fused document to {output_path}")
    save_fused_document(doc, output_path)

    # Optional: produce chunked variant
    if max_chunk_tokens:
        from .chunker import chunk_events

        LOGGER.info(f"Chunking into {max_chunk_tokens}-token chunks")
        result = chunk_events(doc, max_tokens=max_chunk_tokens, overlap_tokens=chunk_overlap_tokens)
        LOGGER.info(f"Created {len(result.chunks)} chunks (avg {result.avg_chunk_size:.1f} events each)")

        # Save chunked variant
        chunked_path = output_path.parent / "fused_chunked.json"
        _save_chunks(result.chunks, chunked_path)
        LOGGER.info(f"Saved chunked variant to {chunked_path}")

    return doc


def _make_windows(duration: float, window_sec: float) -> list[tuple[float, float]]:
    """Build a list of (t_start, t_end) time windows."""
    n = math.ceil(duration / window_sec)
    return [(i * window_sec, min((i + 1) * window_sec, duration)) for i in range(n)]


def _bucket_sentences(sentences: list[Sentence], window_sec: float) -> dict[int, list[Sentence]]:
    """Assign sentences to windows based on overlap."""
    buckets: dict[int, list[Sentence]] = defaultdict(list)
    for s in sentences:
        if s.end < s.start:
            LOGGER.warning(f"Skipping invalid sentence: start={s.start}, end={s.end}")
            continue
        i_start = int(s.start // window_sec)
        i_end = int(s.end // window_sec)
        for i in range(i_start, i_end + 1):
            buckets[i].append(s)
    return buckets


def _bucket_frames(frames: list[FrameVisual], window_sec: float) -> dict[int, list[FrameVisual]]:
    """Assign frames to windows by timestamp."""
    buckets: dict[int, list[FrameVisual]] = defaultdict(list)
    for f in frames:
        i = int(f.timestamp // window_sec)
        buckets[i].append(f)
    return buckets


def _best_frame(frames: list[FrameVisual]) -> FrameVisual | None:
    """Select best frame from a list (prefer more OCR text + captions)."""
    if not frames:
        return None
    return max(frames, key=lambda f: (len(f.text or ""), bool(f.caption)))


def _emit_events(
    windows: list[tuple[float, float]],
    sentence_buckets: dict[int, list[Sentence]],
    frame_buckets: dict[int, list[FrameVisual]],
    all_frames: list[FrameVisual],
) -> list[FusedEvent]:
    """Generate fused events from windows and buckets."""
    events: list[FusedEvent] = []
    prev_nonempty_end: float | None = None

    for window_idx, (t_start, t_end) in enumerate(windows):
        sentences = sentence_buckets.get(window_idx, [])
        frame_list = frame_buckets.get(window_idx, [])
        best_frame = _best_frame(frame_list)

        # Skip silent + no-visual windows
        if not sentences and not best_frame:
            continue

        # Determine event kind
        has_speech = bool(sentences)
        has_visual = bool(best_frame)
        if has_speech and has_visual:
            kind = "speech+visual"
        elif has_speech:
            kind = "speech"
        else:
            kind = "visual"

        # Build event
        speech_text = " ".join(s.text for s in sentences) if sentences else None
        visual_text = best_frame.text if best_frame else None
        visual_caption = best_frame.caption if best_frame else None
        frame_path = str(best_frame.frame_path) if best_frame and best_frame.frame_path else None

        # Detect scene changes: frame with scene-change reason + speech
        is_scene_change = False
        if best_frame and has_speech:
            # Check if frame indicates a scene change (this is a heuristic)
            # In real implementation, could check best_frame.reason or other metadata
            is_scene_change = False

        # Tag notes
        window_obj = Window(
            t_start=t_start,
            t_end=t_end,
            has_speech=has_speech,
            has_visual=has_visual,
            gap_before=prev_nonempty_end and (t_start - prev_nonempty_end) or None,
        )
        notes = tag_notes(window_obj, prev_nonempty_end, None, is_scene_change)

        event = FusedEvent(
            t_start=round(t_start, 3),
            t_end=round(t_end, 3),
            kind=kind,
            speech_text=speech_text,
            speech_segment_indices=[s.segment_indices[0] for s in sentences if s.segment_indices],
            visual_text=visual_text,
            visual_caption=visual_caption,
            frame_index=best_frame.frame_index if best_frame else None,
            frame_path=frame_path,
            notes=notes,
        )

        events.append(event)
        if has_speech or has_visual:
            prev_nonempty_end = t_end

    return events


def _load_visual_frames(visual_path: Path) -> list[FrameVisual]:
    """Load visual extraction frames from JSON."""
    with visual_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    frames: list[FrameVisual] = []
    for frame_data in payload.get("frames", []):
        try:
            frame = FrameVisual(
                timestamp=float(frame_data["timestamp"]),
                frame_index=int(frame_data["frame_index"]),
                frame_path=Path(frame_data["frame_path"]),
                enhanced_path=Path(frame_data["enhanced_path"]) if frame_data.get("enhanced_path") else None,
                text=frame_data.get("text", ""),
                lines=[],  # Simplified; full version would deserialize TextLine
                caption=frame_data.get("caption"),
                caption_source=frame_data.get("caption_source"),
                has_text=bool(frame_data.get("text")),
                raw_ocr=frame_data.get("raw_ocr"),
            )
            frames.append(frame)
        except Exception as e:
            LOGGER.warning(f"Failed to load frame: {e}")
            continue

    return frames


def _save_chunks(chunks: list[FusedDocument], output_path: Path) -> None:
    """Save chunked documents as JSON array."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1",
        "total_chunks": len(chunks),
        "chunks": [asdict(chunk) for chunk in chunks],
    }

    from src.json_utils import dump as safe_dump
    with output_path.open("w", encoding="utf-8") as f:
        safe_dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Fuse aligned transcript and visual extraction into multimodal events"
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to transcript.aligned.json",
    )
    parser.add_argument(
        "--visual",
        type=Path,
        required=True,
        help="Path to visual.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write fused.json",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Unique run identifier",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Time window granularity (seconds)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="If set, produce fused_chunked.json with this token limit per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Token overlap between chunks",
    )

    args = parser.parse_args()

    try:
        doc = fuse(
            args.transcript,
            args.visual,
            args.out,
            run_id=args.run_id,
            window_sec=args.window_sec,
            max_chunk_tokens=args.chunk_tokens,
            chunk_overlap_tokens=args.chunk_overlap,
        )
        print(f"✓ Fused {len(doc.events)} events → {args.out}")
    except Exception as e:
        LOGGER.exception(f"Fusion failed: {e}")
        raise


if __name__ == "__main__":
    main()
