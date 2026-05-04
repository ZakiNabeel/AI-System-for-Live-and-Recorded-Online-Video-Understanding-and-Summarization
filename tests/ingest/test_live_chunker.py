from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from src.ingest.errors import LiveCaptureError
from src.ingest.live_chunker import (
    _build_ffmpeg_args,
    _watch_chunks,
    start_live_capture,
)


def test_build_ffmpeg_args_uses_segment_numbering(tmp_path: Path) -> None:
    args = _build_ffmpeg_args(
        resolved_url="https://example.test/live.m3u8",
        output_dir=tmp_path,
        chunk_seconds=5,
        max_duration_sec=30,
        container="ts",
    )

    assert args[:8] == [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rw_timeout",
        "30000000",
        "-i",
        "https://example.test/live.m3u8",
    ]
    assert "-segment_start_number" in args
    assert "1" == args[args.index("-segment_start_number") + 1]
    assert str(tmp_path / "chunk_%06d.ts") == args[-1]


def test_watch_chunks_emits_only_completed_chunks(tmp_path: Path) -> None:
    (tmp_path / "chunk_000001.ts").write_bytes(b"first")
    (tmp_path / "chunk_000002.ts").write_bytes(b"second")
    emitted: list[tuple[Path, int]] = []
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_watch_chunks,
        args=(tmp_path, lambda path, idx: emitted.append((path, idx)), stop_event),
        daemon=True,
    )

    watcher.start()
    time.sleep(0.7)
    stop_event.set()
    watcher.join(timeout=2)

    assert emitted == [(tmp_path / "chunk_000001.ts", 1)]


def test_start_live_capture_rejects_existing_chunks(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing-run"
    output_dir.mkdir()
    (output_dir / "chunk_000001.ts").write_bytes(b"old")

    with pytest.raises(LiveCaptureError, match="already contains chunks"):
        start_live_capture(
            url="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
            run_id="existing-run",
            output_root=tmp_path,
        )


def test_start_live_capture_launches_ffmpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched_args: list[str] = []

    class FakeProcess:
        pid = 12345
        returncode = 0

        def __init__(self, args: list[str], **kwargs: object) -> None:
            launched_args.extend(args)
            self.kwargs = kwargs
            self._running = True

        def poll(self) -> int | None:
            return None if self._running else self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self._running = False
            return self.returncode

        def kill(self) -> None:
            self._running = False

    monkeypatch.setattr(
        "src.ingest.live_chunker.resolve_stream_url",
        lambda url: "https://cdn.example/live.m3u8",
    )
    monkeypatch.setattr(subprocess, "Popen", FakeProcess)

    handle = start_live_capture(
        url="https://youtube.com/live/test",
        run_id="run-live",
        output_root=tmp_path,
        chunk_seconds=4,
        max_duration_sec=8,
    )
    exit_code = handle.wait()

    assert exit_code == 0
    assert handle.output_dir == (tmp_path / "run-live").resolve()
    assert launched_args[0] == "ffmpeg"
    assert "https://cdn.example/live.m3u8" in launched_args
    assert "4" == launched_args[launched_args.index("-segment_time") + 1]


def test_mp4_requires_bounded_capture(tmp_path: Path) -> None:
    with pytest.raises(LiveCaptureError, match="mp4 chunks require"):
        start_live_capture(
            url="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
            run_id="mp4-run",
            output_root=tmp_path,
            container="mp4",
        )


@pytest.mark.live
def test_live_capture_hls_smoke(tmp_path: Path) -> None:
    chunks: list[Path] = []
    handle = start_live_capture(
        url="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
        run_id="live-smoke",
        output_root=tmp_path,
        chunk_seconds=3,
        max_duration_sec=10,
        on_chunk_ready=lambda path, idx: chunks.append(path),
    )

    exit_code = handle.wait()

    assert exit_code in {0, 255}
    assert len(chunks) >= 1
    assert all(path.exists() and path.stat().st_size > 0 for path in chunks)
