from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def static_video(tmp_path: Path) -> Path:
    return _write_video(
        tmp_path / "static.mp4",
        [np.full((120, 160, 3), 127, dtype=np.uint8) for _ in range(50)],
        fps=10,
    )


@pytest.fixture
def changing_video(tmp_path: Path) -> Path:
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 0),
    ]
    frames: list[np.ndarray] = []
    for color in colors:
        frame = np.full((120, 160, 3), color, dtype=np.uint8)
        frames.extend([frame.copy() for _ in range(10)])
    return _write_video(tmp_path / "changing.mp4", frames, fps=10)


@pytest.fixture
def text_frame(tmp_path: Path) -> Path:
    image = np.full((180, 360, 3), 235, dtype=np.uint8)
    cv2.putText(
        image,
        "HELLO WORLD",
        (24, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    path = tmp_path / "frame_000000_t0.00.png"
    assert cv2.imwrite(str(path), image)
    return path


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter cannot create mp4 test fixtures")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path
