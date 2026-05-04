"""YouTube video downloader for recorded-video ingestion."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError

from .errors import (
    FFmpegMissingError,
    IngestError,
    LiveStreamNotSupportedError,
    NetworkError,
    PrivateVideoError,
    UnavailableVideoError,
)


FFMPEG_INSTALL_HINT = (
    "ffmpeg is required for merging video and audio. Install it and make sure "
    "the ffmpeg binary is available on PATH. On Windows, you can install it "
    "with `winget install Gyan.FFmpeg` and then restart your shell."
)
LOGGER = logging.getLogger(__name__)


class _YtDlpQuietLogger:
    def debug(self, message: str) -> None:
        return None

    def warning(self, message: str) -> None:
        return None

    def error(self, message: str) -> None:
        return None


@dataclass
class VideoDownloadResult:
    run_id: str
    url: str
    video_path: Path
    title: str
    duration_sec: float
    channel: str
    upload_date: str
    width: int
    height: int
    fps: float
    has_subtitles: bool
    available_subtitle_langs: list[str]
    raw_metadata: dict[str, Any]


def download_youtube_video(
    url: str,
    run_id: str,
    output_root: Path = Path("data/raw"),
    max_height: int = 720,
    prefer_codec: str = "mp4",
) -> VideoDownloadResult:
    """Download a recorded YouTube video and return normalized metadata."""

    if not url.strip():
        raise UnavailableVideoError("A YouTube URL is required.")
    if not run_id.strip():
        raise IngestError("A run_id is required.")
    if max_height <= 0:
        raise IngestError("max_height must be a positive integer.")

    output_root = Path(output_root)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _load_dotenv_if_available()

    ydl_opts = _build_ydl_options(output_dir, max_height, prefer_codec)

    try:
        import yt_dlp
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise IngestError(
            "yt-dlp is required. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            preview_info = ydl.extract_info(url, download=False)
            if preview_info.get("is_live") is True:
                raise LiveStreamNotSupportedError(
                    "Live streams are not supported by the batch downloader; "
                    "use the live-stream chunker instead."
                )
            _warn_if_long_video(preview_info)
            _ensure_enough_disk_space(output_root, preview_info)
            info = ydl.extract_info(url, download=True)
    except LiveStreamNotSupportedError:
        raise
    except URLError as exc:
        raise NetworkError(f"Network error while downloading video: {exc}") from exc
    except DownloadError as exc:
        raise _map_download_error(exc) from exc

    video_path = _find_downloaded_video(output_dir, prefer_codec)
    return _result_from_info(
        info=info,
        run_id=run_id,
        url=url,
        video_path=video_path,
    )


def _build_ydl_options(
    output_dir: Path,
    max_height: int,
    prefer_codec: str,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}]"
        ),
        "outtmpl": str(output_dir / "video.%(ext)s"),
        "merge_output_format": prefer_codec,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtDlpQuietLogger(),
        "writesubtitles": False,
        "writeautomaticsub": False,
        "noplaylist": True,
        "retries": 3,
    }

    cookies_file = _cookies_file_from_env()
    if cookies_file:
        opts["cookiefile"] = cookies_file

    return opts


def _cookies_file_from_env() -> str | None:
    from os import getenv

    cookies_file = getenv("YOUTUBE_COOKIES_FILE")
    if cookies_file:
        return cookies_file
    return getenv("YT_DLP_COOKIES_FILE")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def _ensure_enough_disk_space(output_root: Path, info: dict[str, Any]) -> None:
    estimated_size = _estimated_download_size(info)
    if estimated_size is None:
        return

    try:
        free_bytes = shutil.disk_usage(output_root).free
    except OSError:
        return

    if free_bytes <= estimated_size * 2:
        raise IngestError(
            "Not enough free disk space for this video. Free at least twice the "
            "estimated download size and try again."
        )


def _estimated_download_size(info: dict[str, Any]) -> int | None:
    size = info.get("filesize") or info.get("filesize_approx")
    if isinstance(size, int) and size > 0:
        return size

    requested_formats = info.get("requested_formats") or []
    sizes: list[int] = []
    for item in requested_formats:
        item_size = item.get("filesize") or item.get("filesize_approx")
        if isinstance(item_size, int) and item_size > 0:
            sizes.append(item_size)

    if sizes:
        return sum(sizes)
    return None


def _warn_if_long_video(info: dict[str, Any]) -> None:
    duration = info.get("duration")
    if isinstance(duration, int | float) and duration > 7200:
        LOGGER.warning(
            "Video duration is %.0f seconds, which is longer than 2 hours.",
            duration,
        )


def _find_downloaded_video(output_dir: Path, prefer_codec: str) -> Path:
    preferred_path = output_dir / f"video.{prefer_codec}"
    if preferred_path.exists():
        return preferred_path.resolve()

    video_paths = sorted(
        path for path in output_dir.glob("video.*") if path.is_file()
    )
    if not video_paths:
        raise IngestError(f"No downloaded video file was created in {output_dir}.")
    return video_paths[0].resolve()


def _result_from_info(
    info: dict[str, Any],
    run_id: str,
    url: str,
    video_path: Path,
) -> VideoDownloadResult:
    subtitles = info.get("subtitles") or {}
    automatic_captions = info.get("automatic_captions") or {}
    subtitle_langs = sorted({*subtitles.keys(), *automatic_captions.keys()})

    return VideoDownloadResult(
        run_id=run_id,
        url=url,
        video_path=video_path,
        title=str(info.get("title") or ""),
        duration_sec=float(info.get("duration") or 0.0),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        upload_date=_format_upload_date(info.get("upload_date")),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        fps=float(info.get("fps") or 0.0),
        has_subtitles=bool(subtitles or automatic_captions),
        available_subtitle_langs=subtitle_langs,
        raw_metadata=info,
    )


def _format_upload_date(upload_date: Any) -> str:
    if not upload_date:
        return ""
    upload_date_str = str(upload_date)
    if len(upload_date_str) != 8 or not upload_date_str.isdigit():
        return upload_date_str
    return datetime.strptime(upload_date_str, "%Y%m%d").date().isoformat()


def _map_download_error(error: Exception) -> IngestError:
    message = str(error)
    message_lower = message.lower()
    if "private video" in message_lower:
        return PrivateVideoError("This video is private and cannot be downloaded.")
    if "video unavailable" in message_lower:
        return UnavailableVideoError("This video is unavailable.")
    if "not a valid url" in message_lower or "unsupported url" in message_lower:
        return UnavailableVideoError(
            "This URL is not a supported or available YouTube video."
        )
    if "unable to download webpage" in message_lower or "timed out" in message_lower:
        return NetworkError("Network error while downloading video.")
    if "ffmpeg" in message_lower:
        return FFmpegMissingError(FFMPEG_INSTALL_HINT)
    return IngestError(message)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a recorded YouTube video.")
    parser.add_argument("--url", required=True, help="YouTube video URL to download.")
    parser.add_argument("--run-id", help="Pipeline run id. Defaults to a UUID4.")
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum video height to download.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Root folder where raw videos are stored.",
    )
    parser.add_argument(
        "--prefer-codec",
        default="mp4",
        help="Merged output container/codec preference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or str(uuid.uuid4())

    try:
        result = download_youtube_video(
            url=args.url,
            run_id=run_id,
            output_root=args.output_root,
            max_height=args.max_height,
            prefer_codec=args.prefer_codec,
        )
    except IngestError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(asdict(result), default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
