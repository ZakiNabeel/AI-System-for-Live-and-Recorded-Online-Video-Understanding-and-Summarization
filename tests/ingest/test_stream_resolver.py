from __future__ import annotations

import sys
import types

import pytest

from src.ingest.errors import NotALiveStreamError
from src.ingest.stream_resolver import resolve_stream_url


class FakeDownloadError(Exception):
    pass


def install_fake_streamlink(
    monkeypatch: pytest.MonkeyPatch,
    streams: object,
) -> None:
    streamlink_module = types.ModuleType("streamlink")
    streamlink_module.streams = lambda url: streams
    monkeypatch.setitem(sys.modules, "streamlink", streamlink_module)


def install_fake_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
    info: dict[str, object],
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            assert download is False
            return info

    yt_dlp_module = types.ModuleType("yt_dlp")
    yt_dlp_utils_module = types.ModuleType("yt_dlp.utils")
    yt_dlp_module.YoutubeDL = FakeYoutubeDL
    yt_dlp_utils_module.DownloadError = FakeDownloadError
    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_module)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", yt_dlp_utils_module)


def test_direct_hls_url_returns_unchanged() -> None:
    url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"

    assert resolve_stream_url(url) == url


def test_resolve_stream_url_uses_streamlink_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    best_stream = types.SimpleNamespace(url="https://cdn.example/live.m3u8")
    install_fake_streamlink(monkeypatch, {"best": best_stream})

    assert resolve_stream_url("https://youtube.com/live/test") == best_stream.url


def test_resolve_stream_url_falls_back_to_ytdlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_streamlink(monkeypatch, {})
    install_fake_ytdlp(
        monkeypatch,
        {"is_live": True, "url": "https://manifest.example/live.m3u8"},
    )

    assert (
        resolve_stream_url("https://youtube.com/live/test")
        == "https://manifest.example/live.m3u8"
    )


def test_resolve_stream_url_rejects_non_live_ytdlp_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_streamlink(monkeypatch, {})
    install_fake_ytdlp(monkeypatch, {"is_live": False, "url": "https://video"})

    with pytest.raises(NotALiveStreamError):
        resolve_stream_url("https://youtube.com/watch?v=recorded")
