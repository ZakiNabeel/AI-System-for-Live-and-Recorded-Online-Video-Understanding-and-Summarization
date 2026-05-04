from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.vision.frame_extractor import extract_keyframes


def test_static_video_keeps_only_first_frame(static_video: Path, tmp_path: Path) -> None:
    result = extract_keyframes(
        static_video,
        tmp_path / "frames",
        sample_fps=2,
        min_gap_sec=0.5,
        resize_width=160,
    )

    assert result.total_candidates >= 5
    assert result.total_kept == 1
    assert result.frames[0].reason == "first"
    assert cv2.imread(str(result.frames[0].path)) is not None


def test_changing_video_keeps_scene_changes(changing_video: Path, tmp_path: Path) -> None:
    result = extract_keyframes(
        changing_video,
        tmp_path / "frames",
        sample_fps=2,
        min_gap_sec=0.5,
        resize_width=160,
    )

    assert result.total_kept >= 5
    assert [frame.index for frame in result.frames] == list(range(result.total_kept))
    assert all(frame.reason in {"first", "ssim-drop"} for frame in result.frames)


def test_frame_manifest_schema(changing_video: Path, tmp_path: Path) -> None:
    result = extract_keyframes(
        changing_video,
        tmp_path / "frames",
        sample_fps=1,
        min_gap_sec=0.5,
        resize_width=160,
    )
    manifest = json.loads((result.output_dir / "frames.json").read_text(encoding="utf-8"))

    assert manifest["version"] == "1"
    assert manifest["total_kept"] == result.total_kept
    assert {"index", "timestamp", "path", "width", "height", "ssim_to_prev", "reason"} <= set(
        manifest["frames"][0]
    )
    assert not Path(manifest["frames"][0]["path"]).is_absolute()
