"""End-to-end smoke tests.

Marked @pytest.mark.e2e — skipped by default.
Run with:  pytest -m e2e --run-e2e

Requirements:
  - Network access to YouTube
  - ANTHROPIC_API_KEY or OPENAI_API_KEY in environment / .env
  - ffmpeg on PATH
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Short CC-licensed clip used as the canonical canary URL.
CANARY_URL = "https://www.youtube.com/watch?v=BaW_jenozKc"  # 10-second Big Buck Bunny trailer


def pytest_configure(config):  # noqa: ANN001
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring network + API keys")


# ---------------------------------------------------------------------------
# Full recorded-pipeline smoke test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_full_pipeline_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full end-to-end run on the canary URL; asserts all five output files exist."""
    monkeypatch.chdir(tmp_path)
    # Replicate config skeleton
    (tmp_path / "config.yaml").write_text(
        "llm:\n  provider: anthropic\n  model: claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )

    from src.pipeline import run_pipeline

    result = run_pipeline(
        url=CANARY_URL,
        run_id="test-e2e",
        mode="recorded",
        resume=False,
    )

    assert result.summary_md.exists(), "summary.md missing"
    assert result.summary_html.exists(), "summary.html missing"
    assert result.report_json.exists(), "report.json missing"
    assert (result.output_dir / "chapters.txt").exists(), "chapters.txt missing"
    assert (result.output_dir / "report_card.md").exists(), "report_card.md missing"
    assert (result.output_dir / "performance_report.json").exists(), "performance_report.json missing"

    # Timing entries must be non-zero for every executed stage
    import json
    perf = json.loads((result.output_dir / "performance_report.json").read_text())
    for stage, secs in perf["stage_timings_sec"].items():
        assert secs > 0, f"Stage {stage} has zero timing"


# ---------------------------------------------------------------------------
# Resume smoke test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_resume_skips_completed_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run half the pipeline, kill simulate, re-run — verify only remaining stages execute."""
    import json
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "llm:\n  provider: anthropic\n  model: claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )

    from src.pipeline import run_pipeline, _manifest_path_for, _ensure_pipeline_stages, _mark_complete

    # First partial run: only up to enhance (skip everything from ocr onwards)
    run_pipeline(
        url=CANARY_URL,
        run_id="test-resume",
        mode="recorded",
        only={"ingest", "audio", "stt", "align", "frames", "enhance"},
        resume=False,
    )

    manifest_path = _manifest_path_for("test-resume")
    assert manifest_path.exists()

    # Verify those stages are complete
    data = json.loads(manifest_path.read_text())
    for stage in ("ingest", "audio", "stt", "align", "frames", "enhance"):
        assert data["pipeline"][stage]["status"] in ("complete", "skipped"), \
            f"{stage} not done after first pass"

    # Second run: resume=True should skip the already-complete stages
    ran_stages: list[str] = []
    original_run = run_pipeline.__wrapped__ if hasattr(run_pipeline, "__wrapped__") else None

    from src.pipeline import _load_stages, Stage
    import src.pipeline as pipeline_mod

    original_load = pipeline_mod._load_stages

    def patched_load():
        stages = original_load()
        result_stages = []
        for s in stages:
            original_func = s.func

            def make_wrapper(name, fn):
                def wrapper(ctx):
                    ran_stages.append(name)
                    fn(ctx)
                return wrapper

            result_stages.append(
                Stage(name=s.name, func=make_wrapper(s.name, original_func),
                      inputs=s.inputs, outputs=s.outputs, optional=s.optional)
            )
        return result_stages

    pipeline_mod._load_stages = patched_load
    try:
        run_pipeline(url=CANARY_URL, run_id="test-resume", mode="recorded", resume=True)
    finally:
        pipeline_mod._load_stages = original_load

    # The already-complete stages must NOT have run again
    for stage in ("audio", "stt", "align", "frames", "enhance"):
        assert stage not in ran_stages, f"{stage} ran again despite being complete"

    # The remaining stages MUST have run
    for stage in ("ocr", "fuse", "summarize", "format"):
        assert stage in ran_stages, f"{stage} did not run in second pass"


# ---------------------------------------------------------------------------
# --only flag smoke test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_only_flag_runs_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """First do a full run, then use --only summarize,format to regenerate outputs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "llm:\n  provider: anthropic\n  model: claude-haiku-4-5-20251001\n",
        encoding="utf-8",
    )

    from src.pipeline import run_pipeline

    # Full run
    run_pipeline(url=CANARY_URL, run_id="test-only", mode="recorded", resume=False)

    # Now regenerate just summary and format
    run_pipeline(
        url=CANARY_URL,
        run_id="test-only",
        only={"summarize", "format"},
        resume=False,
    )

    output_dir = Path("data") / "output" / "test-only"
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "summary.html").exists()
