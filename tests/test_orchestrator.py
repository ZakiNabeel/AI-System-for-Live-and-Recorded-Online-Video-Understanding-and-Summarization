from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from src.ingest.errors import RunIdConflictError
from src.ingest.youtube_downloader import VideoDownloadResult
from src.orchestrator import detect_mode, run


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
    yt_dlp_module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", yt_dlp_module)


def test_detect_mode_live_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_ytdlp(monkeypatch, {"is_live": True})
    assert detect_mode("https://youtube.com/live/test") == "live"

    install_fake_ytdlp(monkeypatch, {"is_live": False})
    assert detect_mode("https://youtube.com/watch?v=test") == "recorded"


def test_run_dispatches_recorded_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str, Path, int]] = []

    def fake_download(
        url: str,
        run_id: str,
        output_root: Path,
        max_height: int,
    ) -> VideoDownloadResult:
        calls.append((url, run_id, output_root, max_height))
        return VideoDownloadResult(
            run_id=run_id,
            url=url,
            video_path=output_root / run_id / "video.mp4",
            title="Recorded",
            duration_sec=19.0,
            channel="Channel",
            upload_date="2005-04-23",
            width=320,
            height=240,
            fps=30.0,
            has_subtitles=False,
            available_subtitle_langs=[],
            raw_metadata={},
        )

    monkeypatch.setattr("src.orchestrator.download_youtube_video", fake_download)

    context = run(
        url="https://youtube.com/watch?v=test",
        mode="recorded",
        run_id="recorded-run",
    )

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert context.mode == "recorded"
    assert calls == [
        (
            "https://youtube.com/watch?v=test",
            "recorded-run",
            context.paths.raw.parent,
            720,
        )
    ]
    assert manifest["mode"] == "recorded"
    assert manifest["ingest"]["title"] == "Recorded"


def test_run_dispatches_live_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str, Path, int]] = []

    class FakeLiveHandle:
        def __init__(self, run_id: str, output_dir: Path) -> None:
            self.run_id = run_id
            self.output_dir = output_dir

    def fake_capture(
        url: str,
        run_id: str,
        output_root: Path,
        chunk_seconds: int,
        on_chunk_ready: object,
    ) -> FakeLiveHandle:
        calls.append((url, run_id, output_root, chunk_seconds))
        return FakeLiveHandle(run_id, output_root / run_id)

    monkeypatch.setattr("src.orchestrator.start_live_capture", fake_capture)

    context = run(
        url="https://example.test/live.m3u8",
        mode="live",
        run_id="live-run",
    )

    manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
    assert context.mode == "live"
    assert calls == [
        (
            "https://example.test/live.m3u8",
            "live-run",
            context.paths.raw.parent,
            10,
        )
    ]
    assert manifest["mode"] == "live"
    assert manifest["ingest"]["chunk_dir"] == str(context.paths.raw)
    assert manifest["stages"]["ingest"]["status"] == "running"


def test_run_auto_mode_uses_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.orchestrator.detect_mode", lambda url: "recorded")
    monkeypatch.setattr(
        "src.orchestrator.download_youtube_video",
        lambda url, run_id, output_root, max_height: VideoDownloadResult(
            run_id=run_id,
            url=url,
            video_path=output_root / run_id / "video.mp4",
            title="Auto",
            duration_sec=1.0,
            channel="Channel",
            upload_date="2026-05-04",
            width=1,
            height=1,
            fps=1.0,
            has_subtitles=False,
            available_subtitle_langs=[],
            raw_metadata={},
        ),
    )

    context = run("https://youtube.com/watch?v=auto", run_id="auto-run")

    assert context.mode == "recorded"


def test_bad_mode_raises_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError):
        run("https://example.test", mode="weird")  # type: ignore[arg-type]


def test_run_id_conflict_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw" / "same-run").mkdir(parents=True)

    with pytest.raises(RunIdConflictError):
        run("https://example.test", mode="recorded", run_id="same-run")
