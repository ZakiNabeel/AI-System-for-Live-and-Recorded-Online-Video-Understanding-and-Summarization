"""Run directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    raw: Path
    audio: Path
    frames: Path
    intermediate: Path
    output: Path
    log_file: Path


def create_run_paths(run_id: str, base: Path = Path(".")) -> RunPaths:
    """Create and return the canonical per-run folder skeleton."""

    base = Path(base)
    paths = RunPaths(
        root=(base / "data" / "runs" / run_id).resolve(),
        raw=(base / "data" / "raw" / run_id).resolve(),
        audio=(base / "data" / "audio" / run_id).resolve(),
        frames=(base / "data" / "frames" / run_id).resolve(),
        intermediate=(base / "data" / "intermediate" / run_id).resolve(),
        output=(base / "data" / "output" / run_id).resolve(),
        log_file=(base / "logs" / f"{run_id}.log").resolve(),
    )

    for directory in [
        paths.root,
        paths.raw,
        paths.audio,
        paths.frames,
        paths.intermediate,
        paths.output,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    return paths
