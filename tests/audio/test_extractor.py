from __future__ import annotations

import time
import subprocess
from pathlib import Path

import pytest

from src.audio.errors import AudioExtractionError
from src.audio.extractor import extract_audio, extract_audio_for_chunks


def _touch_gap() -> None:
    # Ensure mtime granularity differences are visible on Windows file systems.
    time.sleep(1.1)


def test_extract_audio_happy_path(tone_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "audio.wav"
    result = extract_audio(tone_video, output)

    assert output.exists()
    assert result.audio_path == output
    assert result.source_video == tone_video
    assert result.sample_rate == 16000
    assert result.channels == 1
    assert result.bits_per_sample == 16
    assert result.file_size_bytes > 0
    assert 2.9 < result.duration_sec < 3.1


def test_extract_audio_idempotent_no_overwrite(tone_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "audio.wav"
    extract_audio(tone_video, output, overwrite=False)
    first_mtime = output.stat().st_mtime_ns

    _touch_gap()
    extract_audio(tone_video, output, overwrite=False)
    second_mtime = output.stat().st_mtime_ns

    assert second_mtime == first_mtime


def test_extract_audio_overwrite_updates_file(tone_video: Path, tmp_path: Path) -> None:
    output = tmp_path / "audio.wav"
    extract_audio(tone_video, output, overwrite=False)
    first_mtime = output.stat().st_mtime_ns

    _touch_gap()
    extract_audio(tone_video, output, overwrite=True)
    second_mtime = output.stat().st_mtime_ns

    assert second_mtime > first_mtime


def test_extract_audio_missing_source_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    out = tmp_path / "audio.wav"
    with pytest.raises(FileNotFoundError):
        extract_audio(missing, out)


def test_extract_audio_no_audio_track(video_without_audio: Path, tmp_path: Path) -> None:
    out = tmp_path / "audio.wav"
    with pytest.raises(AudioExtractionError):
        extract_audio(video_without_audio, out)


def test_extract_audio_for_chunks_order_and_idempotency(
    tmp_path: Path,
    require_ffmpeg: None,
) -> None:
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    output_dir = tmp_path / "audio"

    names = ["chunk_000003.mp4", "chunk_000001.mp4", "chunk_000002.mp4"]
    for name in names:
        target = chunk_dir / name
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
                "sine=frequency=440:duration=1",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=1:r=5",
                "-shortest",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(target),
            ],
            check=True,
            capture_output=True,
        )

    first = list(extract_audio_for_chunks(chunk_dir, output_dir))
    assert [r.source_video.name for r in first] == [
        "chunk_000001.mp4",
        "chunk_000002.mp4",
        "chunk_000003.mp4",
    ]
    first_mtimes = {r.audio_path.name: r.audio_path.stat().st_mtime_ns for r in first}

    _touch_gap()
    second = list(extract_audio_for_chunks(chunk_dir, output_dir))
    second_mtimes = {r.audio_path.name: r.audio_path.stat().st_mtime_ns for r in second}

    assert first_mtimes == second_mtimes

