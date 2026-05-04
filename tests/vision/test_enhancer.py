from __future__ import annotations

import json
from pathlib import Path

import cv2

from src.vision.enhancer import enhance_frames


def test_enhance_frames_writes_outputs_and_manifest(text_frame: Path, tmp_path: Path) -> None:
    frames_json = tmp_path / "frames.json"
    frames_json.write_text(
        json.dumps(
            {
                "version": "1",
                "frames": [
                    {
                        "index": 0,
                        "timestamp": 0.0,
                        "path": text_frame.name,
                        "width": 360,
                        "height": 180,
                        "ssim_to_prev": None,
                        "reason": "first",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = enhance_frames(frames_json, profile="screen", save_intermediates=True)
    manifest = json.loads((tmp_path / "enhancements.json").read_text(encoding="utf-8"))

    assert len(result.frames) == 1
    assert result.frames[0].enhanced_path.exists()
    assert result.frames[0].intermediates["gray"].exists()
    assert manifest["profile"] == "screen"
    assert manifest["frames"][0]["enhanced_path"].endswith(".enhanced.png")
    assert cv2.imread(str(result.frames[0].enhanced_path), cv2.IMREAD_GRAYSCALE) is not None
