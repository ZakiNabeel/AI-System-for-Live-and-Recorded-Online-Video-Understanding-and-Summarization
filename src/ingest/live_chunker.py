"""Live stream capture into rolling chunk files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .errors import IngestError, LiveCaptureError
from .stream_resolver import resolve_stream_url


ChunkReadyCallback = Callable[[Path, int], None]
LOGGER = logging.getLogger(__name__)
CHUNK_PATTERN = re.compile(r"chunk_(\d{6})\.\w+$")


@dataclass
class LiveCaptureHandle:
    run_id: str
    output_dir: Path
    process: subprocess.Popen[bytes]
    watcher_thread: threading.Thread
    stop_event: threading.Event = field(repr=False)
    on_chunk_ready: ChunkReadyCallback | None = field(default=None, repr=False)
    emitted_chunks: set[Path] = field(default_factory=set, repr=False)
    emitted_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask ffmpeg to finalize the active chunk and stop."""

        self.stop_event.set()
        if self.is_running:
            _send_graceful_stop(self.process)
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                LOGGER.warning("ffmpeg did not exit cleanly; killing process")
                self.process.kill()
                self.process.wait(timeout=timeout)

        _emit_chunks(
            self.output_dir,
            self.on_chunk_ready,
            self.emitted_chunks,
            self.emitted_lock,
            include_last=True,
        )
        self.watcher_thread.join(timeout=2.0)

    def wait(self) -> int:
        """Block until ffmpeg exits, then emit any final chunk."""

        exit_code = self.process.wait()
        self.stop_event.set()
        _emit_chunks(
            self.output_dir,
            self.on_chunk_ready,
            self.emitted_chunks,
            self.emitted_lock,
            include_last=True,
        )
        self.watcher_thread.join(timeout=2.0)
        return int(exit_code)

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None


def start_live_capture(
    url: str,
    run_id: str,
    chunk_seconds: int = 10,
    output_root: Path = Path("data/raw"),
    max_duration_sec: int | None = None,
    on_chunk_ready: ChunkReadyCallback | None = None,
    container: str = "ts",
) -> LiveCaptureHandle:
    """Start capturing a live stream into fixed-duration chunk files."""

    if chunk_seconds <= 0:
        raise LiveCaptureError("chunk_seconds must be a positive integer.")
    if container not in {"ts", "mp4"}:
        raise LiveCaptureError("container must be either 'ts' or 'mp4'.")
    if container == "mp4" and max_duration_sec is None:
        raise LiveCaptureError("mp4 chunks require max_duration_sec to be set.")

    output_root = Path(output_root)
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("chunk_*.*")):
        raise LiveCaptureError(
            f"Output directory already contains chunks for run_id {run_id}."
        )
    resolved_output_dir = output_dir.resolve()

    resolved_url = resolve_stream_url(url)
    args = _build_ffmpeg_args(
        resolved_url=resolved_url,
        output_dir=output_dir,
        chunk_seconds=chunk_seconds,
        max_duration_sec=max_duration_sec,
        container=container,
    )

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_windows_creationflags(),
            start_new_session=os.name != "nt",
        )
    except FileNotFoundError as exc:
        raise LiveCaptureError(
            "ffmpeg is required for live capture and was not found on PATH."
        ) from exc

    stop_event = threading.Event()
    emitted_chunks: set[Path] = set()
    emitted_lock = threading.Lock()
    watcher_thread = threading.Thread(
        target=_watch_chunks,
        args=(
            resolved_output_dir,
            on_chunk_ready,
            stop_event,
            emitted_chunks,
            emitted_lock,
        ),
        name=f"live-chunk-watcher-{run_id}",
        daemon=True,
    )
    handle = LiveCaptureHandle(
        run_id=run_id,
        output_dir=resolved_output_dir,
        process=process,
        watcher_thread=watcher_thread,
        stop_event=stop_event,
        on_chunk_ready=on_chunk_ready,
        emitted_chunks=emitted_chunks,
        emitted_lock=emitted_lock,
    )
    watcher_thread.start()
    return handle


def _build_ffmpeg_args(
    resolved_url: str,
    output_dir: Path,
    chunk_seconds: int,
    max_duration_sec: int | None,
    container: str,
) -> list[str]:
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rw_timeout",
        "30000000",
        "-i",
        resolved_url,
    ]
    if max_duration_sec is not None:
        args.extend(["-t", str(max_duration_sec)])
    args.extend(
        [
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(chunk_seconds),
            "-reset_timestamps",
            "1",
            "-segment_start_number",
            "1",
            "-segment_format",
            container,
            str(output_dir / f"chunk_%06d.{container}"),
        ]
    )
    return args


def _windows_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))


def _send_graceful_stop(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return


def _watch_chunks(
    output_dir: Path,
    callback: ChunkReadyCallback | None,
    stop_event: threading.Event,
    emitted_chunks: set[Path] | None = None,
    emitted_lock: threading.Lock | None = None,
) -> None:
    emitted_chunks = emitted_chunks if emitted_chunks is not None else set()
    emitted_lock = emitted_lock if emitted_lock is not None else threading.Lock()

    while not stop_event.is_set():
        _warn_if_disk_low(output_dir)
        _emit_chunks(
            output_dir,
            callback,
            emitted_chunks,
            emitted_lock,
            include_last=False,
        )
        stop_event.wait(0.5)


def _emit_chunks(
    output_dir: Path,
    callback: ChunkReadyCallback | None,
    emitted_chunks: set[Path],
    emitted_lock: threading.Lock,
    include_last: bool,
) -> None:
    files = _sorted_chunk_files(output_dir)
    if not include_last:
        if len(files) < 2:
            return
        files = files[:-1]

    for chunk_path in files:
        match = CHUNK_PATTERN.search(chunk_path.name)
        if not match:
            continue
        with emitted_lock:
            if chunk_path in emitted_chunks:
                continue
            emitted_chunks.add(chunk_path)

        chunk_index = int(match.group(1))
        if callback is None:
            continue
        try:
            callback(chunk_path, chunk_index)
        except Exception:
            LOGGER.exception("chunk callback failed")


def _sorted_chunk_files(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("chunk_*.*")
        if path.is_file() and CHUNK_PATTERN.search(path.name)
    )


def _warn_if_disk_low(output_dir: Path) -> None:
    try:
        free_bytes = shutil.disk_usage(output_dir).free
    except OSError:
        return
    if free_bytes < 1_000_000_000:
        LOGGER.warning("Free disk space below 1 GB while capturing live stream.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a live stream into chunks.")
    parser.add_argument("--url", required=True, help="Live stream URL to capture.")
    parser.add_argument("--run-id", help="Pipeline run id. Defaults to a UUID4.")
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=10,
        help="Duration of each chunk in seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        help="Maximum capture duration in seconds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Root folder where raw chunks are stored.",
    )
    parser.add_argument(
        "--container",
        choices=["ts", "mp4"],
        default="ts",
        help="Chunk container format.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id or str(uuid.uuid4())
    handle: LiveCaptureHandle | None = None

    def emit(path: Path, idx: int) -> None:
        payload = {
            "chunk_index": idx,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "wall_time": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(payload), flush=True)

    def handle_sigint(signum: int, frame: object) -> None:
        if handle is not None:
            handle.stop()
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        handle = start_live_capture(
            url=args.url,
            run_id=run_id,
            chunk_seconds=args.chunk_seconds,
            output_root=args.output_root,
            max_duration_sec=args.max_duration,
            on_chunk_ready=emit,
            container=args.container,
        )
        return handle.wait()
    except KeyboardInterrupt:
        if handle is not None:
            handle.stop()
        return 130
    except IngestError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
