from __future__ import annotations

from pathlib import Path

import pytest

from src.speech.errors import AudioTooShortError, NoBackendAvailableError
from src.speech.schema import Transcript, TranscriptSegment
from src.speech.transcriber import _resolve_backend, main, transcribe


def _build_transcript(source: str) -> Transcript:
    return Transcript(
        segments=[TranscriptSegment(start=0.0, end=2.0, text="hello", words=[], language="en")],
        language="en",
        duration_sec=2.0,
        source=source,  # type: ignore[arg-type]
        audio_path=Path("sample.wav"),
        raw_response=None,
    )


def test_resolve_backend_auto_prefers_local(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    assert _resolve_backend("auto") == "local-whisper"


def test_resolve_backend_auto_falls_back_to_openai(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert _resolve_backend("auto") == "openai-whisper"


def test_resolve_backend_auto_raises_without_any_backend(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(NoBackendAvailableError):
        _resolve_backend("auto")


def test_transcribe_uses_youtube_subs_first(monkeypatch, wav_file: Path) -> None:
    expected = _build_transcript("youtube-subtitles")
    expected = Transcript(
        segments=expected.segments,
        language=expected.language,
        duration_sec=expected.duration_sec,
        source="youtube-subtitles",
        audio_path=None,
        raw_response=None,
    )
    monkeypatch.setattr("src.speech.transcriber.fetch_youtube_transcript", lambda _url: expected)
    result = transcribe(audio_path=wav_file, youtube_url="https://www.youtube.com/watch?v=abc")
    assert result.source == "youtube-subtitles"


def test_transcribe_requires_audio_when_subs_missing(monkeypatch) -> None:
    monkeypatch.setattr("src.speech.transcriber.fetch_youtube_transcript", lambda _url: None)
    with pytest.raises(ValueError):
        transcribe(audio_path=None, youtube_url="https://www.youtube.com/watch?v=abc")


def test_transcribe_audio_too_short(monkeypatch, tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.wav"
    tiny.write_bytes(b"")
    monkeypatch.setattr("src.speech.transcriber.fetch_youtube_transcript", lambda _url: None)
    with pytest.raises(AudioTooShortError):
        transcribe(audio_path=tiny, backend="auto", use_youtube_subs_if_available=False)


def test_transcribe_dispatches_local_backend(monkeypatch, wav_file: Path) -> None:
    class FakeLocal:
        def __init__(self, model_name: str):
            assert model_name == "small.en"

        def transcribe(self, audio_path: Path, language: str | None):
            assert audio_path == wav_file
            assert language == "en"
            return _build_transcript("local-whisper")

    monkeypatch.setattr("src.speech.transcriber.LocalWhisperBackend", FakeLocal)
    result = transcribe(audio_path=wav_file, backend="local-whisper", language="en")
    assert result.source == "local-whisper"


def test_cli_writes_output(monkeypatch, tmp_path: Path, wav_file: Path) -> None:
    out = tmp_path / "out.json"

    monkeypatch.setattr(
        "src.speech.transcriber.transcribe",
        lambda **_kwargs: _build_transcript("local-whisper"),
    )

    code = main(["--audio", str(wav_file), "--out", str(out)])
    assert code == 0
    assert out.exists()

