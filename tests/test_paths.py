from __future__ import annotations

from pathlib import Path

from src.paths import create_run_paths


def test_create_run_paths_creates_full_skeleton(tmp_path: Path) -> None:
    paths = create_run_paths("run-abc", base=tmp_path)

    assert paths.root.is_dir()
    assert paths.raw.is_dir()
    assert paths.audio.is_dir()
    assert paths.frames.is_dir()
    assert paths.intermediate.is_dir()
    assert paths.output.is_dir()
    assert paths.log_file.parent.is_dir()


def test_create_run_paths_is_idempotent(tmp_path: Path) -> None:
    first = create_run_paths("run-abc", base=tmp_path)
    second = create_run_paths("run-abc", base=tmp_path)

    assert first == second
