from __future__ import annotations

from pathlib import Path

from src.speech.backends.local_whisper import LocalWhisperBackend


class _FakeWord:
    def __init__(self, start: float, end: float, word: str, probability: float):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str, words: list[_FakeWord]):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _FakeInfo:
    language = "en"
    duration = 2.0


class _FakeModel:
    def transcribe(self, _path: str, language: str | None, **_kwargs):
        del language
        segments = [
            _FakeSegment(
                start=0.0,
                end=2.0,
                text=" one two ",
                words=[
                    _FakeWord(0.0, 0.8, "one", 0.95),
                    _FakeWord(0.9, 1.8, "two", 0.93),
                ],
            )
        ]
        return iter(segments), _FakeInfo()


def test_local_backend_maps_segments_and_words(wav_file: Path) -> None:
    backend = LocalWhisperBackend.__new__(LocalWhisperBackend)
    backend.model = _FakeModel()

    transcript = backend.transcribe(wav_file, language="en")

    assert transcript.source == "local-whisper"
    assert transcript.audio_path == wav_file
    assert transcript.language == "en"
    assert transcript.duration_sec == 2.0
    assert len(transcript.segments) == 1
    assert transcript.segments[0].text == "one two"
    assert len(transcript.segments[0].words) == 2
    assert transcript.segments[0].words[0].text == "one"

