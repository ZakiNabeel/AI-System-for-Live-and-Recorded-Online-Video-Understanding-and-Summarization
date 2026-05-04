"""Tests for manifest-driven resume behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import (
    STAGE_NAMES,
    Stage,
    _ensure_pipeline_stages,
    _get_stage_status,
    _mark_complete,
    _skip_reason,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    mp = tmp_path / "manifest.json"
    mp.write_text(
        json.dumps(
            {
                "run_id": "resume-run",
                "mode": "recorded",
                "url": "https://example.com",
                "started_at": "2026-01-01T00:00:00Z",
                "stages": {"ingest": {"status": "complete", "completed_at": "2026-01-01T00:00:01Z"}},
            }
        ),
        encoding="utf-8",
    )
    return mp


# ---------------------------------------------------------------------------
# Scenario: half-completed run
# ---------------------------------------------------------------------------

def _setup_half_run(manifest: Path) -> None:
    """Simulate ingest → audio → stt → align → frames already complete."""
    _ensure_pipeline_stages(manifest)
    for name in ("audio", "stt", "align", "frames"):
        _mark_complete(manifest, name, 1.0)


def test_completed_stages_skip_on_resume(manifest: Path) -> None:
    _setup_half_run(manifest)
    for name in ("ingest", "audio", "stt", "align", "frames"):
        reason = _skip_reason(name, set(), None, True, manifest)
        assert reason is not None, f"{name} should be skipped on resume"


def test_pending_stages_run_on_resume(manifest: Path) -> None:
    _setup_half_run(manifest)
    for name in ("enhance", "ocr", "fuse", "summarize", "format"):
        reason = _skip_reason(name, set(), None, True, manifest)
        assert reason is None, f"{name} should run on resume"


def test_no_resume_runs_all_stages(manifest: Path) -> None:
    _setup_half_run(manifest)
    for name in STAGE_NAMES:
        reason = _skip_reason(name, set(), None, False, manifest)
        assert reason is None, f"{name} should run when resume=False"


# ---------------------------------------------------------------------------
# Scenario: run_pipeline with resume skips completed stages
# ---------------------------------------------------------------------------

@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_resume_skips_completed_stages(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    manifest: Path,
) -> None:
    from src.pipeline import run_pipeline

    _setup_half_run(manifest)
    mock_mp.return_value = manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "resume-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = manifest
    run_ctx.config = {}
    run_ctx.url = ""
    mock_load_ctx.return_value = run_ctx

    ran: list[str] = []

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            ran.append(n)
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=(n == "audio"))

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    run_pipeline(url="https://example.com", run_id="resume-run", resume=True)

    # Completed stages must NOT have run.
    for name in ("ingest", "audio", "stt", "align", "frames"):
        assert name not in ran, f"{name} ran despite being complete"

    # Pending stages MUST have run.
    for name in ("enhance", "ocr", "fuse", "summarize", "format"):
        assert name in ran, f"{name} did not run"


# ---------------------------------------------------------------------------
# Scenario: failed stage is re-run on resume
# ---------------------------------------------------------------------------

def test_failed_stage_reruns_on_resume(manifest: Path) -> None:
    _ensure_pipeline_stages(manifest)
    _mark_complete(manifest, "audio", 1.0)

    # Simulate a failure at stt
    from src.pipeline import _mark_failed
    _mark_failed(manifest, "stt", "boom", "tb")
    assert _get_stage_status(manifest, "stt") == "failed"

    # On resume, failed stage should NOT be skipped (only "complete" stages skip)
    reason = _skip_reason("stt", set(), None, True, manifest)
    assert reason is None, "Failed stage should be re-run on resume"


# ---------------------------------------------------------------------------
# Scenario: manifest pipeline stages persist across calls
# ---------------------------------------------------------------------------

def test_stage_status_persists(manifest: Path) -> None:
    _ensure_pipeline_stages(manifest)
    _mark_complete(manifest, "fuse", 2.0)
    # Read fresh
    status = _get_stage_status(manifest, "fuse")
    assert status == "complete"
    elapsed = json.loads(manifest.read_text())["pipeline"]["fuse"]["elapsed_sec"]
    assert elapsed == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Scenario: --only with resume
# ---------------------------------------------------------------------------

def test_only_overrides_resume(manifest: Path) -> None:
    """--only always wins over resume status — only listed stages run."""
    _setup_half_run(manifest)
    # Even if "summarize" is pending, it runs when in --only
    reason = _skip_reason("summarize", set(), {"summarize", "format"}, True, manifest)
    assert reason is None

    # "audio" is complete but also in --only → skip reason is from resume
    reason = _skip_reason("audio", set(), {"summarize", "format"}, True, manifest)
    assert reason is not None  # not in --only
