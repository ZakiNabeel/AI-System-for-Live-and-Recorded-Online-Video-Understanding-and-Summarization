from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.ingest.errors import (
    FFmpegMissingError,
    LiveStreamNotSupportedError,
    PrivateVideoError,
    UnavailableVideoError,
)
from src.ingest.youtube_downloader import (
    VideoDownloadResult,
    download_youtube_video,
    main,
)


class FakeDownloadError(Exception):
    pass


def install_fake_yt_dlp(monkeypatch: pytest.MonkeyPatch, youtube_dl_cls: type) -> None:
    yt_dlp_module = types.ModuleType("yt_dlp")
    yt_dlp_utils_module = types.ModuleType("yt_dlp.utils")
    yt_dlp_module.YoutubeDL = youtube_dl_cls
    yt_dlp_utils_module.DownloadError = FakeDownloadError
    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_module)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", yt_dlp_utils_module)


def fake_info(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "title": "A tiny test video",
        "duration": 19,
        "channel": "Test Channel",
        "uploader": "Fallback Uploader",
        "upload_date": "20050423",
        "width": 640,
        "height": 360,
        "fps": 29.97,
        "subtitles": {"en": [{"ext": "vtt"}]},
        "automatic_captions": {"es": [{"ext": "vtt"}]},
    }
    info.update(overrides)
    return info


def video_path_from_opts(opts: dict[str, object], ext: str = "mp4") -> Path:
    outtmpl = str(opts["outtmpl"])
    return Path(outtmpl.replace("%(ext)s", ext))


def test_download_youtube_video_maps_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            assert url == "https://www.youtube.com/watch?v=test"
            if download:
                video_path_from_opts(self.opts).write_bytes(b"video")
            return fake_info()

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    result = download_youtube_video(
        url="https://www.youtube.com/watch?v=test",
        run_id="run-123",
        output_root=tmp_path,
        max_height=480,
    )

    assert isinstance(result, VideoDownloadResult)
    assert result.run_id == "run-123"
    assert result.video_path == (tmp_path / "run-123" / "video.mp4").resolve()
    assert result.video_path.exists()
    assert result.title == "A tiny test video"
    assert result.duration_sec == 19.0
    assert result.channel == "Test Channel"
    assert result.upload_date == "2005-04-23"
    assert result.width == 640
    assert result.height == 360
    assert result.fps == 29.97
    assert result.has_subtitles is True
    assert result.available_subtitle_langs == ["en", "es"]


def test_download_youtube_video_uses_uploader_as_channel_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            if download:
                video_path_from_opts(self.opts).write_bytes(b"video")
            return fake_info(channel=None, subtitles={}, automatic_captions={})

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    result = download_youtube_video(
        url="https://www.youtube.com/watch?v=test",
        run_id="run-456",
        output_root=tmp_path,
    )

    assert result.channel == "Fallback Uploader"
    assert result.has_subtitles is False
    assert result.available_subtitle_langs == []


def test_private_video_error_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            raise FakeDownloadError("Private video")

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    with pytest.raises(PrivateVideoError):
        download_youtube_video(
            url="https://www.youtube.com/watch?v=private",
            run_id="run-private",
            output_root=tmp_path,
        )


def test_live_stream_rejected_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            calls.append(download)
            return fake_info(is_live=True)

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    with pytest.raises(LiveStreamNotSupportedError):
        download_youtube_video(
            url="https://www.youtube.com/live/test",
            run_id="run-live",
            output_root=tmp_path,
        )

    assert calls == [False]


def test_ffmpeg_error_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            raise FakeDownloadError("ffmpeg was not found")

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    with pytest.raises(FFmpegMissingError, match="ffmpeg is required"):
        download_youtube_video(
            url="https://www.youtube.com/watch?v=test",
            run_id="run-ffmpeg",
            output_root=tmp_path,
        )


def test_cli_returns_code_2_for_invalid_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeYoutubeDL:
        def __init__(self, opts: dict[str, object]) -> None:
            self.opts = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, object]:
            raise FakeDownloadError("Video unavailable")

    install_fake_yt_dlp(monkeypatch, FakeYoutubeDL)

    exit_code = main(
        [
            "--url",
            "not-a-real-url",
            "--run-id",
            "run-invalid",
            "--output-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "This video is unavailable.\n"


@pytest.mark.network
def test_download_youtube_video_network_canary(tmp_path: Path) -> None:
    pytest.importorskip("yt_dlp")

    result = download_youtube_video(
        url="https://www.youtube.com/watch?v=jNQXAC9IVRw",
        run_id="network-canary",
        output_root=tmp_path,
        max_height=144,
    )

    assert result.video_path.exists()
    assert result.video_path.parent == tmp_path / "network-canary"
    assert result.duration_sec == pytest.approx(19, abs=4)
