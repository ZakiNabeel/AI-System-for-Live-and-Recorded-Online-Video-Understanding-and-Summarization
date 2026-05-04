"""Resolve live-page URLs into stream URLs ffmpeg can read."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .errors import IngestError, NotALiveStreamError, UnresolvableStreamError


class _QuietLogger:
    def debug(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None


def resolve_stream_url(url: str) -> str:
    """Resolve a live URL to an HLS/DASH/RTMP URL suitable for ffmpeg."""

    if _looks_direct_stream_url(url):
        return url

    streamlink_url = _resolve_with_streamlink(url)
    if streamlink_url:
        return streamlink_url

    return _resolve_with_ytdlp(url)


def _looks_direct_stream_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme in {"rtmp", "rtmps"}:
        return True
    path = parsed.path.lower()
    return path.endswith(".m3u8") or path.endswith(".mpd")


def _resolve_with_streamlink(url: str) -> str | None:
    try:
        import streamlink
    except ImportError:
        return None

    try:
        streams = streamlink.streams(url)
    except Exception:
        return None

    if not streams:
        return None

    stream = streams.get("best") or next(iter(streams.values()))
    return getattr(stream, "url", None)


def _resolve_with_ytdlp(url: str) -> str:
    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise IngestError(
            "yt-dlp is required to resolve this live URL. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from exc

    opts: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
        "logger": _QuietLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise UnresolvableStreamError(str(exc)) from exc

    if not info.get("is_live"):
        raise NotALiveStreamError(f"URL is not a live stream: {url}")

    resolved_url = info.get("url")
    if not resolved_url:
        raise UnresolvableStreamError(f"Could not resolve stream URL: {url}")
    return str(resolved_url)
