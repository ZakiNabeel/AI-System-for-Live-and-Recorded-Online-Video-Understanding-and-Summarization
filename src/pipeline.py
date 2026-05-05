"""Master pipeline: orchestrate all stages into a resumable, observable run."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from .config import load_config
from .logging_setup import configure_logging
from .manifest import update_manifest
from .orchestrator import RunContext
from .paths import create_run_paths

LOGGER = logging.getLogger(__name__)

STAGE_NAMES: list[str] = [
    "ingest",
    "audio",
    "stt",
    "align",
    "frames",
    "enhance",
    "ocr",
    "fuse",
    "summarize",
    "format",
]


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class StageError:
    stage: str
    error: str
    tb: str


@dataclass
class PipelineResult:
    run_context: RunContext
    output_dir: Path
    summary_md: Path
    summary_html: Path
    report_json: Path
    timings: dict[str, float]
    errors: list[StageError]


@dataclass
class StageCtx:
    run_ctx: RunContext
    config: dict[str, Any]
    log: logging.Logger
    enable_captions: bool
    enable_qa: bool
    domain: str | None


@dataclass
class Stage:
    name: str
    func: Callable[[StageCtx], None]
    inputs: list[str]
    outputs: list[str]
    optional: bool = False


# ---------------------------------------------------------------------------
# Stage registry  (lazy import avoids circular deps at module load time)
# ---------------------------------------------------------------------------

def _load_stages() -> list[Stage]:
    from .stages.ingest import run_ingest
    from .stages.audio import run_audio
    from .stages.stt import run_stt
    from .stages.align import run_align
    from .stages.frames import run_frames
    from .stages.enhance import run_enhance
    from .stages.ocr import run_ocr
    from .stages.fuse import run_fuse
    from .stages.summarize import run_summarize
    from .stages.format import run_format

    return [
        Stage("ingest",    run_ingest,    [],                                          ["video"]),
        Stage("audio",     run_audio,     ["video"],                                   ["audio"],    optional=True),
        Stage("stt",       run_stt,       ["audio"],                                   ["transcript"]),
        Stage("align",     run_align,     ["transcript", "audio"],                     ["aligned", "srt"]),
        Stage("frames",    run_frames,    ["video"],                                   ["frames"]),
        Stage("enhance",   run_enhance,   ["frames"],                                  ["enhanced"]),
        Stage("ocr",       run_ocr,       ["enhanced"],                                ["visual"]),
        Stage("fuse",      run_fuse,      ["aligned", "visual"],                       ["fused"]),
        Stage("summarize", run_summarize, ["fused"],                                   ["summary"]),
        Stage(
            "format", run_format,
            ["summary", "aligned", "fused", "visual"],
            ["markdown", "html", "report", "chapters", "report_card"],
        ),
    ]


# ---------------------------------------------------------------------------
# Run-context helpers
# ---------------------------------------------------------------------------

def _manifest_path_for(run_id: str) -> Path:
    return Path("data") / "intermediate" / run_id / "manifest.json"


def _load_run_context(manifest_path: Path, config_path: Path) -> RunContext:
    """Reconstruct a RunContext from an existing manifest (resume path)."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = data["run_id"]
    paths = create_run_paths(run_id)
    config = load_config(config_path)

    ts = data.get("started_at", "")
    try:
        started_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        started_at = datetime.now(timezone.utc)

    configure_logging(paths.log_file)

    return RunContext(
        run_id=run_id,
        mode=data.get("mode", "recorded"),
        url=data.get("url", ""),
        started_at=started_at,
        paths=paths,
        config=config,
        ingest_result=None,
        manifest_path=manifest_path,
    )


# ---------------------------------------------------------------------------
# Manifest pipeline-stage tracking
# ---------------------------------------------------------------------------

def _ensure_pipeline_stages(manifest_path: Path) -> None:
    """Add 'pipeline' tracking block to manifest if absent."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "pipeline" in data:
        return

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    pipeline: dict[str, Any] = {name: {"status": "pending"} for name in STAGE_NAMES}

    # Ingest already done by orchestrator.run().
    ingest_ts = data.get("stages", {}).get("ingest", {}).get("completed_at", now)
    pipeline["ingest"] = {"status": "complete", "completed_at": ingest_ts}

    update_manifest(manifest_path, pipeline=pipeline)


def _get_stage_status(manifest_path: Path, stage_name: str) -> str:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data.get("pipeline", {}).get(stage_name, {}).get("status", "pending")


def _mark_complete(manifest_path: Path, stage: str, elapsed: float) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    update_manifest(
        manifest_path,
        pipeline={stage: {"status": "complete", "completed_at": now, "elapsed_sec": elapsed}},
    )


def _mark_failed(manifest_path: Path, stage: str, error: str, tb: str) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    update_manifest(
        manifest_path,
        pipeline={stage: {"status": "failed", "failed_at": now, "error": error, "traceback": tb}},
    )


def _mark_skipped(manifest_path: Path, stage: str, reason: str) -> None:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    update_manifest(
        manifest_path,
        pipeline={stage: {"status": "skipped", "skipped_at": now, "reason": reason}},
    )


def _skip_reason(
    stage_name: str,
    skip: set[str],
    only: set[str] | None,
    resume: bool,
    manifest_path: Path,
) -> str | None:
    """Return a non-empty reason string if this stage should be skipped, else None."""
    if stage_name in skip:
        return "explicit --skip"
    if only is not None and stage_name not in only:
        return "not in --only"
    if resume and _get_stage_status(manifest_path, stage_name) == "complete":
        return "already complete (resume)"
    return None


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

def _write_performance_report(run_ctx: RunContext, timings: dict[str, float]) -> None:
    report = {
        "run_id": run_ctx.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stage_timings_sec": timings,
        "total_sec": sum(timings.values()),
    }
    path = run_ctx.paths.output / "performance_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Performance report -> %s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    url: str,
    *,
    mode: Literal["auto", "recorded", "live"] = "auto",
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
    skip: set[str] = frozenset(),
    only: set[str] | None = None,
    resume: bool = True,
    domain: str | None = None,
    enable_captions: bool = False,
    enable_qa: bool = False,
) -> PipelineResult:
    """Run every pipeline stage in order with manifest-driven resume support."""

    # --- 1. Set up RunContext ---
    if run_id:
        candidate = _manifest_path_for(run_id)
        if candidate.exists():
            LOGGER.info("Resuming run %s from %s", run_id, candidate)
            run_ctx = _load_run_context(candidate, config_path)
        else:
            LOGGER.info("Starting fresh run (run_id=%s)", run_id)
            from .orchestrator import run as _orchestrator_run
            run_ctx = _orchestrator_run(
                url=url, mode=mode, run_id=run_id, config_path=config_path,
            )
    else:
        LOGGER.info("Starting fresh run (auto run_id)")
        from .orchestrator import run as _orchestrator_run
        run_ctx = _orchestrator_run(
            url=url, mode=mode, run_id=None, config_path=config_path,
        )

    _ensure_pipeline_stages(run_ctx.manifest_path)

    # --- 2. Build stage context ---
    stages = _load_stages()
    ctx = StageCtx(
        run_ctx=run_ctx,
        config=run_ctx.config,
        log=LOGGER,
        enable_captions=enable_captions,
        enable_qa=enable_qa,
        domain=domain,
    )

    timings: dict[str, float] = {}
    errors: list[StageError] = []

    # --- 3. Dispatch stages ---
    for stage in stages:
        reason = _skip_reason(stage.name, skip, only, resume, run_ctx.manifest_path)
        if reason:
            LOGGER.info("[%s] SKIP (%s)", stage.name, reason)
            _mark_skipped(run_ctx.manifest_path, stage.name, reason)
            continue

        LOGGER.info("[%s] START", stage.name)
        t0 = time.monotonic()
        try:
            stage.func(ctx)
            elapsed = time.monotonic() - t0
            timings[stage.name] = elapsed
            _mark_complete(run_ctx.manifest_path, stage.name, elapsed)
            LOGGER.info("[%s] DONE (%.1fs)", stage.name, elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            tb = traceback.format_exc()
            _mark_failed(run_ctx.manifest_path, stage.name, str(exc), tb)
            LOGGER.error("[%s] FAILED (%.1fs): %s", stage.name, elapsed, exc)
            errors.append(StageError(stage=stage.name, error=str(exc), tb=tb))
            if not stage.optional:
                raise

    # --- 4. Write performance report ---
    _write_performance_report(run_ctx, timings)

    output_dir = run_ctx.paths.output
    return PipelineResult(
        run_context=run_ctx,
        output_dir=output_dir,
        summary_md=output_dir / "summary.md",
        summary_html=output_dir / "summary.html",
        report_json=output_dir / "report.json",
        timings=timings,
        errors=errors,
    )


def run_pipeline_streaming(
    url: str,
    mode: str = "auto",
    *,
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
    skip: set[str] = frozenset(),
    only: set[str] | None = None,
    resume: bool = True,
    domain: str | None = None,
    enable_captions: bool = False,
    enable_qa: bool = False,
) -> Iterator[tuple[str, float]]:
    """Generator wrapper for Streamlit: yields (stage_name, progress_fraction) after each stage."""

    if run_id:
        candidate = _manifest_path_for(run_id)
        if candidate.exists():
            run_ctx = _load_run_context(candidate, config_path)
        else:
            from .orchestrator import run as _orchestrator_run
            run_ctx = _orchestrator_run(url=url, mode=mode, run_id=run_id, config_path=config_path)
    else:
        from .orchestrator import run as _orchestrator_run
        run_ctx = _orchestrator_run(url=url, mode=mode, run_id=None, config_path=config_path)

    _ensure_pipeline_stages(run_ctx.manifest_path)

    stages = _load_stages()
    ctx = StageCtx(
        run_ctx=run_ctx,
        config=run_ctx.config,
        log=LOGGER,
        enable_captions=enable_captions,
        enable_qa=enable_qa,
        domain=domain,
    )
    total = len(stages)

    run_id = run_ctx.run_id

    for i, stage in enumerate(stages):
        reason = _skip_reason(stage.name, skip, only, resume, run_ctx.manifest_path)
        if reason:
            _mark_skipped(run_ctx.manifest_path, stage.name, reason)
            yield stage.name, (i + 1) / total, run_id
            continue

        yield stage.name, i / total, run_id
        t0 = time.monotonic()
        try:
            stage.func(ctx)
            elapsed = time.monotonic() - t0
            _mark_complete(run_ctx.manifest_path, stage.name, elapsed)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _mark_failed(run_ctx.manifest_path, stage.name, str(exc), traceback.format_exc())
            if not stage.optional:
                raise
        yield stage.name, (i + 1) / total, run_id

    _write_performance_report(run_ctx, {})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="DIP Video Understanding Pipeline",
    )
    p.add_argument("--url", required=True, help="YouTube URL or live-stream URL.")
    p.add_argument(
        "--mode", choices=["auto", "recorded", "live"], default="auto",
        help="Ingestion mode (default: auto-detect).",
    )
    p.add_argument("--run-id", default=None, help="Reuse an existing run ID for resume.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument(
        "--skip", default="",
        help="Comma-separated stage names to skip (e.g. frames,ocr).",
    )
    p.add_argument(
        "--only", default="",
        help="Comma-separated stage names to run (all others skipped).",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Re-run all stages even if marked complete in manifest.",
    )
    p.add_argument("--captions", action="store_true", help="Enable image captioning (OCR stage).")
    p.add_argument("--qa", action="store_true", help="Generate Q&A pairs (summarize stage).")
    p.add_argument("--domain", default=None, help="Domain hint for LLM (e.g. education).")
    p.add_argument("--quiet", action="store_true", help="Suppress info logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    import sys

    args = _build_parser().parse_args(argv)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    skip_set: set[str] = {s.strip() for s in args.skip.split(",") if s.strip()}
    only_set: set[str] | None = {s.strip() for s in args.only.split(",") if s.strip()} or None

    try:
        result = run_pipeline(
            url=args.url,
            mode=args.mode,
            run_id=args.run_id,
            config_path=args.config,
            skip=skip_set,
            only=only_set,
            resume=not args.no_resume,
            domain=args.domain,
            enable_captions=args.captions,
            enable_qa=args.qa,
        )
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    total_sec = sum(result.timings.values())
    print(f"\nPipeline complete! ({total_sec:.1f}s total)")
    print(f"  Summary MD:   {result.summary_md}")
    print(f"  Summary HTML: {result.summary_html}")
    print(f"  Report JSON:  {result.report_json}")
    print(f"  Output dir:   {result.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
