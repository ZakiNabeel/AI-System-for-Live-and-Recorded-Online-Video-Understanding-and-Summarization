from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.ingest.youtube_downloader import VideoDownloadResult
from src.manifest import update_manifest, write_manifest
from src.paths import create_run_paths


def test_write_manifest_for_recorded_ingest(tmp_path: Path) -> None:
    paths = create_run_paths("run-manifest", base=tmp_path)
    result = VideoDownloadResult(
        run_id="run-manifest",
        url="https://youtube.com/watch?v=test",
        video_path=paths.raw / "video.mp4",
        title="Demo",
        duration_sec=12.5,
        channel="Channel",
        upload_date="2026-05-04",
        width=1280,
        height=720,
        fps=30.0,
        has_subtitles=True,
        available_subtitle_langs=["en"],
        raw_metadata={"id": "test"},
    )

    manifest_path = write_manifest(
        paths,
        "run-manifest",
        "recorded",
        result.url,
        datetime(2026, 5, 4, tzinfo=timezone.utc),
        result,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-manifest"
    assert manifest["mode"] == "recorded"
    assert manifest["ingest"]["video_path"] == str(paths.raw / "video.mp4")
    assert manifest["ingest"]["duration_sec"] == 12.5
    assert manifest["stages"]["ingest"]["status"] == "complete"
    assert manifest["stages"]["audio"]["status"] == "pending"


def test_update_manifest_preserves_existing_fields(tmp_path: Path) -> None:
    paths = create_run_paths("run-update", base=tmp_path)
    manifest_path = write_manifest(
        paths,
        "run-update",
        "live",
        "https://example.test/live.m3u8",
        datetime(2026, 5, 4, tzinfo=timezone.utc),
        ingest_result=None,
        ingest_status="running",
    )

    update_manifest(
        manifest_path,
        ingest={"last_chunk": {"chunk_index": 1, "path": "chunk_000001.ts"}},
        stages={"audio": {"status": "complete"}},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-update"
    assert manifest["ingest"]["last_chunk"]["chunk_index"] == 1
    assert manifest["stages"]["ingest"]["status"] == "running"
    assert manifest["stages"]["audio"]["status"] == "complete"
