from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def require_ffmpeg() -> None:
    if not _ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe are required for audio extraction tests.")


@pytest.fixture
def tone_video(tmp_path: Path, require_ffmpeg: None) -> Path:
    out = tmp_path / "tone.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=3:r=10",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture
def video_without_audio(tmp_path: Path, require_ffmpeg: None) -> Path:
    out = tmp_path / "video_only.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=10:duration=2",
            "-an",
            "-c:v",
            "libx264",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out

