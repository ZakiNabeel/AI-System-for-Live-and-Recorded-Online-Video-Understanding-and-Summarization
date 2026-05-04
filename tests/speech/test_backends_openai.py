from __future__ import annotations

import math
import wave
from pathlib import Path

from src.speech.backends.openai_api import OpenAIWhisperBackend


def _write_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    total_frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * total_frames)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


class _FakeTranscriptions:
    def __init__(self):
        self.calls: list[str] = []

    def create(self, **kwargs):
        filename = Path(kwargs["file"].name).name
        self.calls.append(filename)
        if filename.startswith("part_000"):
            start = 0.0
        elif filename.startswith("part_001"):
            start = 0.0
        else:
            start = 0.0
        return _FakeResponse(
            {
                "language": "en",
                "segments": [
                    {"start": start, "end": start + 1.0, "text": f"seg-{filename}", "words": []}
                ],
                "words": [{"start": start, "end": start + 0.5, "word": "hello"}],
            }
        )


class _FakeAudio:
    def __init__(self):
        self.transcriptions = _FakeTranscriptions()


class _FakeClient:
    def __init__(self):
        self.audio = _FakeAudio()


def test_openai_backend_transcribes_single_file(wav_file: Path) -> None:
    backend = OpenAIWhisperBackend(client=_FakeClient())
    transcript = backend.transcribe(wav_file, language="en")

    assert transcript.source == "openai-whisper"
    assert transcript.language == "en"
    assert transcript.audio_path == wav_file
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text.startswith("seg-")
    assert len(transcript.segments[0].words) == 1
    assert math.isnan(transcript.segments[0].words[0].confidence)


def test_openai_backend_size_guard_slices_and_offsets(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "large.wav"
    source.write_bytes(b"x" * ((25 * 1024 * 1024) + 1))

    part1 = tmp_path / "part_000.wav"
    part2 = tmp_path / "part_001.wav"
    _write_wav(part1, duration_sec=1.0)
    _write_wav(part2, duration_sec=1.0)

    backend = OpenAIWhisperBackend(client=_FakeClient())
    monkeypatch.setattr(
        backend,
        "_slice_audio_with_ffmpeg",
        lambda _audio, _out_dir: [part1, part2],
    )
    monkeypatch.setattr(
        "src.speech.backends.openai_api._audio_duration_seconds",
        lambda p: 2.0 if p == source else 1.0,
    )

    transcript = backend.transcribe(source, language="en")
    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 1.0
    assert transcript.segments[1].start == 1.0
    assert transcript.segments[1].end == 2.0

