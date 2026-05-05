"""Live-mode pipeline: producer-consumer per-chunk processing with rolling summarization."""

from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .logging_setup import configure_logging
from .manifest import update_manifest
from .orchestrator import RunContext
from .paths import create_run_paths
from .pipeline import (
    STAGE_NAMES,
    StageCtx,
    _ensure_pipeline_stages,
    _load_stages,
    _mark_complete,
    _mark_failed,
    _write_performance_report,
)

LOGGER = logging.getLogger(__name__)

# How often (seconds) rolling summarize+format are triggered while live.
DEFAULT_ROLLING_INTERVAL = 30


@dataclass
class LivePipelineHandle:
    run_ctx: RunContext
    stop_event: threading.Event
    _worker_thread: threading.Thread = field(repr=False)

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the live pipeline to stop and wait for the final pass to finish."""
        LOGGER.info("Stop signal received; finishing live pipeline …")
        self.stop_event.set()
        self._worker_thread.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Per-chunk processing
# ---------------------------------------------------------------------------

def _process_chunk(
    chunk_path: Path,
    chunk_index: int,
    ctx: StageCtx,
    chunk_out: Path,
) -> dict[str, Path]:
    """Run audio → stt → frames → enhance → ocr for a single chunk.

    Returns a dict of canonical output paths for this chunk.
    """
    from .audio.extractor import extract_audio
    from .speech.transcriber import transcribe
    from .speech.schema import save_transcript
    from .speech.aligner import align_transcript
    from .vision.frame_extractor import extract_keyframes
    from .vision.enhancer import enhance_frames
    from .vision.extractor import extract_visual_content

    chunk_out.mkdir(parents=True, exist_ok=True)

    # Audio
    audio_path = chunk_out / "audio.wav"
    cfg_audio = ctx.config.get("audio", {})
    extract_audio(
        video_path=chunk_path,
        output_path=audio_path,
        sample_rate=int(cfg_audio.get("sample_rate", 16000)),
        mono=bool(cfg_audio.get("mono", True)),
    )

    # STT
    transcript_path = chunk_out / "transcript.json"
    cfg_speech = ctx.config.get("speech", {})
    transcript = transcribe(
        audio_path=audio_path,
        model=cfg_speech.get("model", "small.en"),
        use_youtube_subs_if_available=False,
    )
    save_transcript(transcript, transcript_path)

    # Align
    aligned_result = align_transcript(
        transcript_path=transcript_path,
        audio_path=audio_path,
        output_dir=chunk_out,
    )

    # Frames
    frames_dir = chunk_out / "frames"
    cfg_frames = ctx.config.get("frames", {})
    extract_keyframes(
        video_path=chunk_path,
        output_dir=frames_dir,
        ssim_threshold=float(cfg_frames.get("ssim_threshold", 0.92)),
        min_gap_sec=float(cfg_frames.get("min_gap_sec", 1.5)),
    )

    # Enhance
    enhanced_dir = frames_dir / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    enhance_frames(
        frames_json_path=frames_dir / "frames.json",
        output_dir=enhanced_dir,
    )

    # OCR
    manifest_path = enhanced_dir / "enhancements.json"
    if not manifest_path.exists():
        manifest_path = frames_dir / "frames.json"
    extract_visual_content(
        frames_manifest_path=manifest_path,
        output_dir=chunk_out,
        enable_captions=ctx.enable_captions,
    )

    LOGGER.info("Chunk %04d processed -> %s", chunk_index, chunk_out)
    return {
        "aligned": aligned_result.aligned_json,
        "visual": chunk_out / "visual.json",
    }


# ---------------------------------------------------------------------------
# Accumulator: merge per-chunk aligned/visual outputs into running files
# ---------------------------------------------------------------------------

def _merge_chunk_outputs(
    chunk_results: list[dict[str, Path]],
    intermediate: Path,
) -> tuple[Path, Path]:
    """Merge all per-chunk aligned and visual JSONs into combined files."""
    import json

    from .speech.schema import AlignedTranscript, Sentence, TranscriptSegment, Word

    combined_segments: list[dict] = []
    combined_sentences: list[dict] = []
    combined_visuals: list[dict] = []
    time_offset = 0.0
    last_end = 0.0
    source = "local-whisper"

    for cr in chunk_results:
        aligned_path = cr.get("aligned")
        visual_path = cr.get("visual")

        if aligned_path and aligned_path.exists():
            data = json.loads(aligned_path.read_text(encoding="utf-8"))
            chunk_segments = data.get("segments", [])
            chunk_sentences = data.get("sentences", [])
            source = data.get("source", source)

            for seg in chunk_segments:
                seg = dict(seg)
                seg["start"] = seg.get("start", 0.0) + time_offset
                seg["end"] = seg.get("end", 0.0) + time_offset
                if seg["words"]:
                    for w in seg["words"]:
                        w["start"] += time_offset
                        w["end"] += time_offset
                combined_segments.append(seg)
                last_end = max(last_end, seg["end"])

            for sent in chunk_sentences:
                sent = dict(sent)
                sent["start"] = sent.get("start", 0.0) + time_offset
                sent["end"] = sent.get("end", 0.0) + time_offset
                combined_sentences.append(sent)

            if chunk_segments:
                chunk_duration = max(s.get("end", 0.0) for s in chunk_segments)
                time_offset += chunk_duration

        if visual_path and visual_path.exists():
            vdata = json.loads(visual_path.read_text(encoding="utf-8"))
            combined_visuals.extend(vdata.get("frames", []))

    # Write merged aligned transcript
    merged_aligned = intermediate / "transcript.aligned.json"
    aligned_payload = {
        "segments": combined_segments,
        "sentences": combined_sentences,
        "language": "en",
        "duration_sec": last_end,
        "source": source,
        "audio_path": None,
        "raw_response": None,
    }
    merged_aligned.write_text(
        __import__("json").dumps(aligned_payload, indent=2, default=str),
        encoding="utf-8",
    )

    # Write merged visual
    merged_visual = intermediate / "visual.json"
    visual_payload = {
        "frames": combined_visuals,
        "ocr_engine": "tesseract",
        "captioner": None,
        "elapsed_sec": 0.0,
    }
    merged_visual.write_text(
        __import__("json").dumps(visual_payload, indent=2, default=str),
        encoding="utf-8",
    )

    return merged_aligned, merged_visual


# ---------------------------------------------------------------------------
# Rolling summarize + format
# ---------------------------------------------------------------------------

def _rolling_pass(ctx: StageCtx, chunk_results: list[dict[str, Path]]) -> None:
    """Merge all chunk outputs and run fuse → summarize → format."""
    if not chunk_results:
        return

    paths = ctx.run_ctx.paths
    intermediate = paths.intermediate
    intermediate.mkdir(parents=True, exist_ok=True)

    try:
        merged_aligned, merged_visual = _merge_chunk_outputs(chunk_results, intermediate)
    except Exception as exc:
        LOGGER.warning("Merge failed: %s", exc)
        return

    # Fuse
    try:
        from .fusion.fuser import fuse
        fuse(
            transcript_path=merged_aligned,
            visual_path=merged_visual,
            output_path=intermediate / "fused.json",
            run_id=ctx.run_ctx.run_id,
        )
    except Exception as exc:
        LOGGER.warning("Fuse failed during rolling pass: %s", exc)
        return

    # Summarize
    cfg = ctx.config.get("llm", {})
    try:
        from .llm.summarizer import summarize
        summarize(
            fused_path=intermediate / "fused.json",
            output_path=intermediate / "summary.raw.json",
            provider=cfg.get("provider", "anthropic"),
            model=cfg.get("model"),
            enable_qa=ctx.enable_qa,
            domain=ctx.domain,
            rolling=True,
        )
    except Exception as exc:
        LOGGER.warning("Summarize failed during rolling pass: %s", exc)
        return

    # Format
    try:
        from .output.formatter import format_outputs
        format_outputs(
            summary_path=intermediate / "summary.raw.json",
            transcript_path=merged_aligned,
            fused_path=intermediate / "fused.json",
            visual_path=merged_visual,
            output_dir=paths.output,
            run_id=ctx.run_ctx.run_id,
            youtube_url=ctx.run_ctx.url,
        )
        LOGGER.info("Rolling output written to %s", paths.output)
    except Exception as exc:
        LOGGER.warning("Format failed during rolling pass: %s", exc)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _worker(
    chunk_queue: "queue.Queue[tuple[Path, int] | None]",
    ctx: StageCtx,
    stop_event: threading.Event,
    rolling_interval: float,
    timings: dict[str, float],
) -> None:
    chunk_results: list[dict[str, Path]] = []
    last_rolling = time.monotonic()

    while True:
        try:
            item = chunk_queue.get(timeout=1.0)
        except queue.Empty:
            # Check if we should do a rolling pass
            if time.monotonic() - last_rolling >= rolling_interval and chunk_results:
                LOGGER.info("Triggering periodic rolling summarize …")
                _rolling_pass(ctx, chunk_results)
                last_rolling = time.monotonic()
            # Check for stop
            if stop_event.is_set() and chunk_queue.empty():
                break
            continue

        if item is None:  # Sentinel: live capture finished
            break

        chunk_path, chunk_index = item
        chunk_out = ctx.run_ctx.paths.intermediate / f"chunk_{chunk_index:04d}"
        t0 = time.monotonic()
        try:
            result = _process_chunk(chunk_path, chunk_index, ctx, chunk_out)
            elapsed = time.monotonic() - t0
            timings[f"chunk_{chunk_index:04d}"] = elapsed
            chunk_results.append(result)
        except Exception as exc:
            LOGGER.error("Chunk %04d failed: %s", chunk_index, exc)

        chunk_queue.task_done()

    # Final pass on all accumulated chunks
    LOGGER.info("Performing final rolling pass on %d chunks …", len(chunk_results))
    _rolling_pass(ctx, chunk_results)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_live_pipeline(
    url: str,
    *,
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
    rolling_interval_sec: float = DEFAULT_ROLLING_INTERVAL,
    domain: str | None = None,
    enable_captions: bool = False,
    enable_qa: bool = False,
) -> LivePipelineHandle:
    """Start a live pipeline.  Returns a handle; call handle.stop() to finalize."""

    from .orchestrator import run as _orchestrator_run

    LOGGER.info("Starting live pipeline (run_id=%s)", run_id or "auto")
    chunk_queue: "queue.Queue[tuple[Path, int] | None]" = queue.Queue()

    def _on_chunk(path: Path, idx: int) -> None:
        chunk_queue.put((path, idx))

    run_ctx = _orchestrator_run(
        url=url,
        mode="live",
        run_id=run_id,
        config_path=config_path,
    )

    _ensure_pipeline_stages(run_ctx.manifest_path)

    ctx = StageCtx(
        run_ctx=run_ctx,
        config=run_ctx.config,
        log=LOGGER,
        enable_captions=enable_captions,
        enable_qa=enable_qa,
        domain=domain,
    )

    timings: dict[str, float] = {}
    stop_event = threading.Event()

    # Wire on_chunk_ready to enqueue chunks.
    live_handle = run_ctx.ingest_result
    if live_handle is not None and hasattr(live_handle, "on_chunk_ready"):
        live_handle.on_chunk_ready = _on_chunk

    worker = threading.Thread(
        target=_worker,
        args=(chunk_queue, ctx, stop_event, rolling_interval_sec, timings),
        daemon=True,
    )
    worker.start()

    def _stop_all() -> None:
        if live_handle is not None and hasattr(live_handle, "stop"):
            live_handle.stop()
        chunk_queue.put(None)  # Sentinel
        stop_event.set()

    handle = LivePipelineHandle(
        run_ctx=run_ctx,
        stop_event=stop_event,
        _worker_thread=worker,
    )
    # Monkey-patch stop to also tear down the live capture.
    _orig_stop = handle.stop

    def _patched_stop(timeout: float = 60.0) -> None:
        _stop_all()
        _orig_stop(timeout)
        _write_performance_report(run_ctx, timings)

    handle.stop = _patched_stop  # type: ignore[method-assign]

    return handle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python -m src.pipeline_live",
        description="DIP Live Pipeline — stream and summarize a live broadcast.",
    )
    p.add_argument("--url", required=True)
    p.add_argument("--run-id", default=None)
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--rolling-interval", type=float, default=DEFAULT_ROLLING_INTERVAL)
    p.add_argument("--domain", default=None)
    p.add_argument("--captions", action="store_true")
    p.add_argument("--qa", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    handle = run_live_pipeline(
        url=args.url,
        run_id=args.run_id,
        config_path=args.config,
        rolling_interval_sec=args.rolling_interval,
        domain=args.domain,
        enable_captions=args.captions,
        enable_qa=args.qa,
    )

    def _sig_handler(sig, frame):  # noqa: ANN001
        print("\nCtrl+C — finalizing …")
        handle.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"Live pipeline running (run_id={handle.run_ctx.run_id}).  Press Ctrl+C to stop.")
    handle._worker_thread.join()
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
