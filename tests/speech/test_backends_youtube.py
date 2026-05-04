from __future__ import annotations

import types

from src.speech.backends.youtube_subs import _parse_video_id, fetch_youtube_transcript


def test_parse_video_id_variants() -> None:
    assert _parse_video_id("https://www.youtube.com/watch?v=abc123") == "abc123"
    assert _parse_video_id("https://youtu.be/xyz987") == "xyz987"
    assert _parse_video_id("https://www.youtube.com/shorts/short55") == "short55"


def test_fetch_youtube_transcript_success(monkeypatch) -> None:
    fake_module = types.ModuleType("youtube_transcript_api")
    fake_errors = types.ModuleType("youtube_transcript_api._errors")

    class NoTranscriptFound(Exception):
        pass

    class TranscriptsDisabled(Exception):
        pass

    class VideoUnavailable(Exception):
        pass

    fake_errors.NoTranscriptFound = NoTranscriptFound
    fake_errors.TranscriptsDisabled = TranscriptsDisabled
    fake_errors.VideoUnavailable = VideoUnavailable

    class _Snippet:
        def __init__(self, text: str, start: float, duration: float):
            self.text = text
            self.start = start
            self.duration = duration

    class _Result:
        language_code = "en"
        snippets = [_Snippet("hello", 0.0, 1.0), _Snippet("world", 1.0, 1.0)]

    class YouTubeTranscriptApi:
        def fetch(self, video_id: str, languages: list[str]):
            assert video_id == "abc123"
            assert languages == ["en"]
            return _Result()

    fake_module.YouTubeTranscriptApi = YouTubeTranscriptApi

    monkeypatch.setitem(__import__("sys").modules, "youtube_transcript_api", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "youtube_transcript_api._errors", fake_errors)

    transcript = fetch_youtube_transcript("https://www.youtube.com/watch?v=abc123")
    assert transcript is not None
    assert transcript.source == "youtube-subtitles"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].words == []
    assert transcript.duration_sec == 2.0


def test_fetch_youtube_transcript_missing_returns_none(monkeypatch) -> None:
    fake_module = types.ModuleType("youtube_transcript_api")
    fake_errors = types.ModuleType("youtube_transcript_api._errors")

    class NoTranscriptFound(Exception):
        pass

    class TranscriptsDisabled(Exception):
        pass

    class VideoUnavailable(Exception):
        pass

    fake_errors.NoTranscriptFound = NoTranscriptFound
    fake_errors.TranscriptsDisabled = TranscriptsDisabled
    fake_errors.VideoUnavailable = VideoUnavailable

    class YouTubeTranscriptApi:
        def fetch(self, _video_id: str, _languages: list[str]):
            raise NoTranscriptFound("missing")

    fake_module.YouTubeTranscriptApi = YouTubeTranscriptApi

    monkeypatch.setitem(__import__("sys").modules, "youtube_transcript_api", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "youtube_transcript_api._errors", fake_errors)

    transcript = fetch_youtube_transcript("https://www.youtube.com/watch?v=abc123")
    assert transcript is None

