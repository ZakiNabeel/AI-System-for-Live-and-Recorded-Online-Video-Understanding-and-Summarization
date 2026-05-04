"""YouTube subtitles backend."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from ..schema import Transcript, TranscriptSegment


def _parse_video_id(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/")
        if video_id:
            return video_id

    if host.endswith("youtube.com"):
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            video_id = query.get("v", [""])[0]
            if video_id:
                return video_id
        for prefix in ("/shorts/", "/live/", "/embed/"):
            if parsed.path.startswith(prefix):
                candidate = parsed.path.split(prefix, 1)[1].split("/", 1)[0]
                if candidate:
                    return candidate

    raise ValueError(f"Unable to parse YouTube video ID from URL: {url}")


def fetch_youtube_transcript(
    url: str,
    languages: tuple[str, ...] = ("en",),
) -> Transcript | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        from youtube_transcript_api._errors import (  # type: ignore[import-not-found]
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
        known_errors = (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)
    except Exception:
        known_errors = None

    video_id = _parse_video_id(url)
    api = YouTubeTranscriptApi()
    try:
        result = api.fetch(video_id, languages=list(languages))
    except Exception as exc:
        if known_errors is not None and isinstance(exc, known_errors):
            return None
        message = str(exc).lower()
        if "transcript" in message or "caption" in message or "disabled" in message:
            return None
        raise

    snippets = getattr(result, "snippets", result)
    language = getattr(result, "language_code", None) or languages[0]

    segments: list[TranscriptSegment] = []
    raw_items: list[dict[str, object]] = []
    for item in snippets:
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
        else:
            text = str(getattr(item, "text", ""))
            start = float(getattr(item, "start", 0.0))
            duration = float(getattr(item, "duration", 0.0))
        end = start + duration
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                words=[],
                language=language,
            )
        )
        raw_items.append({"text": text, "start": start, "duration": duration})

    return Transcript(
        segments=segments,
        language=language,
        duration_sec=segments[-1].end if segments else 0.0,
        source="youtube-subtitles",
        audio_path=None,
        raw_response={"items": raw_items},
    )

