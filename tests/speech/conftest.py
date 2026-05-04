from __future__ import annotations

import wave
from pathlib import Path

import pytest


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    sample_rate = 16000
    duration_sec = 2
    total_frames = sample_rate * duration_sec

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * total_frames)

    return path

