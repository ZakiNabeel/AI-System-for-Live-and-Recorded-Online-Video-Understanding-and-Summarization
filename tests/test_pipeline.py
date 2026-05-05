"""Unit tests for the pipeline dispatcher, stage registry, and manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline import (
    STAGE_NAMES,
    Stage,
    StageCtx,
    StageError,
    _ensure_pipeline_stages,
    _get_stage_status,
    _load_stages,
    _mark_complete,
    _mark_failed,
    _mark_skipped,
    _skip_reason,
    _write_performance_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_manifest(tmp_path: Path) -> Path:
    """Write a minimal manifest and return its path."""
    mp = tmp_path / "manifest.json"
    mp.write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "mode": "recorded",
                "url": "https://example.com",
                "started_at": "2026-01-01T00:00:00Z",
                "stages": {"ingest": {"status": "complete", "completed_at": "2026-01-01T00:00:01Z"}},
            }
        ),
        encoding="utf-8",
    )
    return mp


@pytest.fixture()
def fake_run_ctx(tmp_path: Path, fake_manifest: Path) -> MagicMock:
    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    ctx = MagicMock()
    ctx.run_id = "test-run"
    ctx.paths = paths
    ctx.manifest_path = fake_manifest
    return ctx


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

def test_stage_registry_has_ten_stages() -> None:
    stages = _load_stages()
    assert len(stages) == 10


def test_stage_names_match_constant() -> None:
    stages = _load_stages()
    assert [s.name for s in stages] == STAGE_NAMES


def test_audio_stage_is_optional() -> None:
    stages = _load_stages()
    audio = next(s for s in stages if s.name == "audio")
    assert audio.optional is True


def test_all_other_stages_are_not_optional() -> None:
    stages = _load_stages()
    for stage in stages:
        if stage.name != "audio":
            assert stage.optional is False, f"{stage.name} should not be optional"


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def test_ensure_pipeline_stages_adds_pipeline_key(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    data = json.loads(fake_manifest.read_text())
    assert "pipeline" in data
    for name in STAGE_NAMES:
        assert name in data["pipeline"]


def test_ensure_pipeline_stages_idempotent(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _ensure_pipeline_stages(fake_manifest)  # Should not raise
    data = json.loads(fake_manifest.read_text())
    assert data["pipeline"]["ingest"]["status"] == "complete"


def test_ingest_marked_complete_after_ensure(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    assert _get_stage_status(fake_manifest, "ingest") == "complete"


def test_other_stages_pending_after_ensure(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    for name in STAGE_NAMES:
        if name != "ingest":
            assert _get_stage_status(fake_manifest, name) == "pending"


def test_mark_complete(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _mark_complete(fake_manifest, "audio", 1.23)
    data = json.loads(fake_manifest.read_text())
    assert data["pipeline"]["audio"]["status"] == "complete"
    assert data["pipeline"]["audio"]["elapsed_sec"] == pytest.approx(1.23)


def test_mark_failed(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _mark_failed(fake_manifest, "stt", "boom", "traceback here")
    data = json.loads(fake_manifest.read_text())
    assert data["pipeline"]["stt"]["status"] == "failed"
    assert "boom" in data["pipeline"]["stt"]["error"]


def test_mark_skipped(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _mark_skipped(fake_manifest, "frames", "explicit --skip")
    data = json.loads(fake_manifest.read_text())
    assert data["pipeline"]["frames"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Skip-reason logic
# ---------------------------------------------------------------------------

def test_skip_explicit(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    reason = _skip_reason("audio", {"audio"}, None, True, fake_manifest)
    assert reason is not None
    assert "skip" in reason.lower()


def test_skip_not_in_only(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    reason = _skip_reason("audio", set(), {"summarize"}, True, fake_manifest)
    assert reason is not None


def test_no_skip_when_in_only(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    reason = _skip_reason("summarize", set(), {"summarize"}, True, fake_manifest)
    assert reason is None  # Not complete yet, so should run


def test_skip_resume_complete(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _mark_complete(fake_manifest, "audio", 0.5)
    reason = _skip_reason("audio", set(), None, True, fake_manifest)
    assert "resume" in reason.lower()


def test_no_skip_resume_false_even_if_complete(fake_manifest: Path) -> None:
    _ensure_pipeline_stages(fake_manifest)
    _mark_complete(fake_manifest, "audio", 0.5)
    reason = _skip_reason("audio", set(), None, False, fake_manifest)
    assert reason is None


# ---------------------------------------------------------------------------
# Performance report
# ---------------------------------------------------------------------------

def test_write_performance_report(fake_run_ctx: MagicMock) -> None:
    _write_performance_report(fake_run_ctx, {"audio": 1.5, "stt": 3.2})
    report_path = fake_run_ctx.paths.output / "performance_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["stage_timings_sec"]["audio"] == pytest.approx(1.5)
    assert data["total_sec"] == pytest.approx(4.7)


def test_write_performance_report_empty_timings(fake_run_ctx: MagicMock) -> None:
    _write_performance_report(fake_run_ctx, {})
    report_path = fake_run_ctx.paths.output / "performance_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["total_sec"] == 0.0


# ---------------------------------------------------------------------------
# run_pipeline integration (fully mocked stages)
# ---------------------------------------------------------------------------

def _make_noop_stage(name: str, optional: bool = False) -> Stage:
    return Stage(name=name, func=lambda ctx: None, inputs=[], outputs=[], optional=optional)


@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_run_pipeline_calls_all_stages(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    fake_manifest: Path,
) -> None:
    """With all stages as no-ops, run_pipeline should complete without error."""
    from src.pipeline import run_pipeline, _ensure_pipeline_stages

    # Arrange manifest
    _ensure_pipeline_stages(fake_manifest)
    mock_mp.return_value = fake_manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "test-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = fake_manifest
    run_ctx.config = {}
    run_ctx.url = "https://example.com"
    mock_load_ctx.return_value = run_ctx

    called: list[str] = []

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            called.append(n)
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=(n == "audio"))

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    result = run_pipeline(url="https://example.com", run_id="test-run", resume=False)

    # ingest is already complete so it's skipped by default when resume=False... actually
    # resume=False means don't skip. But ingest stage func will also be called.
    # All stage names should be in called (except ingest which is pre-seeded as complete
    # but resume=False so it runs too).
    assert set(called) == set(STAGE_NAMES)
    assert result.output_dir == paths.output


@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_run_pipeline_skip_flag(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    fake_manifest: Path,
) -> None:
    from src.pipeline import run_pipeline, _ensure_pipeline_stages

    _ensure_pipeline_stages(fake_manifest)
    mock_mp.return_value = fake_manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "test-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = fake_manifest
    run_ctx.config = {}
    run_ctx.url = ""
    mock_load_ctx.return_value = run_ctx

    called: list[str] = []

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            called.append(n)
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=(n == "audio"))

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    run_pipeline(
        url="https://example.com",
        run_id="test-run",
        resume=False,
        skip={"frames", "enhance"},
    )
    assert "frames" not in called
    assert "enhance" not in called
    assert "stt" in called


@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_run_pipeline_only_flag(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    fake_manifest: Path,
) -> None:
    from src.pipeline import run_pipeline, _ensure_pipeline_stages

    _ensure_pipeline_stages(fake_manifest)
    mock_mp.return_value = fake_manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "test-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = fake_manifest
    run_ctx.config = {}
    run_ctx.url = ""
    mock_load_ctx.return_value = run_ctx

    called: list[str] = []

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            called.append(n)
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=(n == "audio"))

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    run_pipeline(
        url="https://example.com",
        run_id="test-run",
        resume=False,
        only={"summarize", "format"},
    )
    assert called == ["summarize", "format"]


@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_optional_stage_failure_does_not_abort(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    fake_manifest: Path,
) -> None:
    from src.pipeline import run_pipeline, _ensure_pipeline_stages

    _ensure_pipeline_stages(fake_manifest)
    mock_mp.return_value = fake_manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "test-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = fake_manifest
    run_ctx.config = {}
    run_ctx.url = ""
    mock_load_ctx.return_value = run_ctx

    called: list[str] = []

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            if n == "audio":
                raise RuntimeError("no audio track")
            called.append(n)
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=(n == "audio"))

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    result = run_pipeline(
        url="https://example.com",
        run_id="test-run",
        resume=False,
    )
    assert len(result.errors) == 1
    assert result.errors[0].stage == "audio"
    assert "stt" in called  # pipeline continued past the optional failure


@patch("src.pipeline._load_stages")
@patch("src.pipeline._load_run_context")
@patch("src.pipeline._manifest_path_for")
def test_non_optional_stage_failure_aborts(
    mock_mp: MagicMock,
    mock_load_ctx: MagicMock,
    mock_load_stages: MagicMock,
    tmp_path: Path,
    fake_manifest: Path,
) -> None:
    from src.pipeline import run_pipeline, _ensure_pipeline_stages

    _ensure_pipeline_stages(fake_manifest)
    mock_mp.return_value = fake_manifest

    paths = MagicMock()
    paths.output = tmp_path / "output"
    paths.output.mkdir()
    run_ctx = MagicMock()
    run_ctx.run_id = "test-run"
    run_ctx.paths = paths
    run_ctx.manifest_path = fake_manifest
    run_ctx.config = {}
    run_ctx.url = ""
    mock_load_ctx.return_value = run_ctx

    def make_stage(n: str) -> Stage:
        def fn(ctx):
            if n == "stt":
                raise RuntimeError("stt kaboom")
        return Stage(name=n, func=fn, inputs=[], outputs=[], optional=False)

    mock_load_stages.return_value = [make_stage(n) for n in STAGE_NAMES]

    with pytest.raises(RuntimeError, match="stt kaboom"):
        run_pipeline(url="https://example.com", run_id="test-run", resume=False)
