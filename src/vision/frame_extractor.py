"""Extract sparse key-frames from recorded videos or live chunks."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import cv2
import numpy as np

from .errors import CannotOpenVideoError, FrameExtractionError

FrameKeepReason = Literal["first", "ssim-drop", "min-gap-forced"]


@dataclass(frozen=True)
class KeyFrame:
    index: int
    timestamp: float
    path: Path
    width: int
    height: int
    ssim_to_prev: float | None
    reason: FrameKeepReason


@dataclass(frozen=True)
class FrameExtractionResult:
    video_path: Path
    output_dir: Path
    frames: list[KeyFrame]
    total_candidates: int
    total_kept: int
    sample_fps: float
    ssim_threshold: float


@dataclass
class _ExtractionState:
    next_index: int = 0
    total_candidates: int = 0
    last_kept_timestamp: float | None = None
    last_kept_gray: np.ndarray | None = None


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    ssim_threshold: float = 0.92,
    min_gap_sec: float = 1.5,
    resize_width: int = 640,
    always_keep_first: bool = True,
) -> FrameExtractionResult:
    """Sample a video and save frames that differ from the last kept frame."""

    state = _ExtractionState()
    frames = _extract_from_single_video(
        video_path=Path(video_path),
        output_dir=Path(output_dir),
        sample_fps=sample_fps,
        ssim_threshold=ssim_threshold,
        min_gap_sec=min_gap_sec,
        resize_width=resize_width,
        always_keep_first=always_keep_first,
        state=state,
        timestamp_offset=0.0,
        refuse_non_empty=True,
    )
    result = FrameExtractionResult(
        video_path=Path(video_path).resolve(),
        output_dir=Path(output_dir).resolve(),
        frames=frames,
        total_candidates=state.total_candidates,
        total_kept=len(frames),
        sample_fps=sample_fps,
        ssim_threshold=ssim_threshold,
    )
    _write_frames_manifest(result, min_gap_sec=min_gap_sec, resize_width=resize_width)
    return result


def extract_keyframes_for_chunks(
    chunk_dir: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    ssim_threshold: float = 0.92,
    min_gap_sec: float = 1.5,
    resize_width: int = 640,
    always_keep_first: bool = True,
) -> list[FrameExtractionResult]:
    """Extract key-frames from live chunks while preserving cross-chunk state."""

    chunk_dir = Path(chunk_dir)
    output_dir = Path(output_dir)
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: '{chunk_dir.resolve()}'")
    if not chunk_dir.is_dir():
        raise NotADirectoryError(f"Chunk path is not a directory: '{chunk_dir.resolve()}'")
    _ensure_output_dir(output_dir, refuse_non_empty=True)

    chunks = sorted(chunk_dir.glob("chunk_*.ts")) + sorted(chunk_dir.glob("chunk_*.mp4"))
    chunks = sorted(chunks)
    state = _ExtractionState()
    results: list[FrameExtractionResult] = []
    timestamp_offset = 0.0
    all_frames: list[KeyFrame] = []

    for chunk in chunks:
        before_candidates = state.total_candidates
        before_kept = state.next_index
        frames = _extract_from_single_video(
            video_path=chunk,
            output_dir=output_dir,
            sample_fps=sample_fps,
            ssim_threshold=ssim_threshold,
            min_gap_sec=min_gap_sec,
            resize_width=resize_width,
            always_keep_first=always_keep_first,
            state=state,
            timestamp_offset=timestamp_offset,
            refuse_non_empty=False,
        )
        all_frames.extend(frames)
        results.append(
            FrameExtractionResult(
                video_path=chunk.resolve(),
                output_dir=output_dir.resolve(),
                frames=frames,
                total_candidates=state.total_candidates - before_candidates,
                total_kept=state.next_index - before_kept,
                sample_fps=sample_fps,
                ssim_threshold=ssim_threshold,
            )
        )
        timestamp_offset += _video_duration_sec(chunk)

    aggregate = FrameExtractionResult(
        video_path=chunk_dir.resolve(),
        output_dir=output_dir.resolve(),
        frames=all_frames,
        total_candidates=state.total_candidates,
        total_kept=len(all_frames),
        sample_fps=sample_fps,
        ssim_threshold=ssim_threshold,
    )
    _write_frames_manifest(aggregate, min_gap_sec=min_gap_sec, resize_width=resize_width)
    return results


def _extract_from_single_video(
    *,
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    ssim_threshold: float,
    min_gap_sec: float,
    resize_width: int,
    always_keep_first: bool,
    state: _ExtractionState,
    timestamp_offset: float,
    refuse_non_empty: bool,
) -> list[KeyFrame]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    if not 0 <= ssim_threshold <= 1:
        raise ValueError("ssim_threshold must be between 0 and 1")
    if min_gap_sec < 0:
        raise ValueError("min_gap_sec must be >= 0")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: '{video_path.resolve()}'")

    _ensure_output_dir(output_dir, refuse_non_empty=refuse_non_empty)
    kept: list[KeyFrame] = []

    for timestamp, frame_bgr in _iter_sampled_frames(video_path, sample_fps):
        timestamp += timestamp_offset
        state.total_candidates += 1
        resized_bgr, gray = _prep_frame(frame_bgr, resize_width)
        score = _safe_ssim(state.last_kept_gray, gray)
        absdiff = _safe_absdiff(state.last_kept_gray, gray)
        reason = _keep_reason(
            timestamp=timestamp,
            last_kept_timestamp=state.last_kept_timestamp,
            ssim_score=score,
            absdiff=absdiff,
            ssim_threshold=ssim_threshold,
            min_gap_sec=min_gap_sec,
            always_keep_first=always_keep_first,
        )
        if reason is None:
            continue

        image_path = _save_frame(resized_bgr, state.next_index, timestamp, output_dir)
        height, width = resized_bgr.shape[:2]
        record = KeyFrame(
            index=state.next_index,
            timestamp=round(float(timestamp), 3),
            path=image_path.resolve(),
            width=int(width),
            height=int(height),
            ssim_to_prev=score,
            reason=reason,
        )
        kept.append(record)
        state.next_index += 1
        state.last_kept_timestamp = timestamp
        state.last_kept_gray = gray

    return kept


def _iter_sampled_frames(video_path: Path, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CannotOpenVideoError(video_path)

    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, round(src_fps / sample_fps))
        frame_idx = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_idx / src_fps, frame
            frame_idx += step
            if total > 0 and frame_idx >= total:
                break
    finally:
        cap.release()


def _prep_frame(frame_bgr: np.ndarray, target_width: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_bgr.ndim != 3:
        raise FrameExtractionError("Expected a BGR color frame from OpenCV.")
    height, width = frame_bgr.shape[:2]
    if target_width > 0 and width != target_width:
        scale = target_width / float(width)
        target_height = max(1, int(round(height * scale)))
        frame_bgr = cv2.resize(
            frame_bgr,
            (int(target_width), target_height),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return frame_bgr, gray


def _safe_ssim(prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> float | None:
    if prev_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(
            curr_gray,
            (prev_gray.shape[1], prev_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return _ssim(prev_gray, curr_gray)


def _safe_absdiff(prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> float | None:
    if prev_gray is None:
        return None
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(
            curr_gray,
            (prev_gray.shape[1], prev_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return float(np.mean(cv2.absdiff(prev_gray, curr_gray)) / 255.0)


def _ssim(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(prev_gray, curr_gray, data_range=255))
    except ImportError:
        return _fast_ssim(prev_gray, curr_gray)


def _fast_ssim(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    prev = prev_gray.astype(np.float64)
    curr = curr_gray.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_prev = cv2.GaussianBlur(prev, (11, 11), 1.5)
    mu_curr = cv2.GaussianBlur(curr, (11, 11), 1.5)
    mu_prev_sq = mu_prev * mu_prev
    mu_curr_sq = mu_curr * mu_curr
    mu_prev_curr = mu_prev * mu_curr
    sigma_prev_sq = cv2.GaussianBlur(prev * prev, (11, 11), 1.5) - mu_prev_sq
    sigma_curr_sq = cv2.GaussianBlur(curr * curr, (11, 11), 1.5) - mu_curr_sq
    sigma_prev_curr = cv2.GaussianBlur(prev * curr, (11, 11), 1.5) - mu_prev_curr
    numerator = (2 * mu_prev_curr + c1) * (2 * sigma_prev_curr + c2)
    denominator = (mu_prev_sq + mu_curr_sq + c1) * (sigma_prev_sq + sigma_curr_sq + c2)
    score = np.mean(numerator / np.maximum(denominator, 1e-12))
    return float(np.clip(score, -1.0, 1.0))


def _keep_reason(
    *,
    timestamp: float,
    last_kept_timestamp: float | None,
    ssim_score: float | None,
    absdiff: float | None,
    ssim_threshold: float,
    min_gap_sec: float,
    always_keep_first: bool,
) -> FrameKeepReason | None:
    if last_kept_timestamp is None:
        return "first" if always_keep_first else None

    gap = timestamp - last_kept_timestamp
    if gap < min_gap_sec:
        return None
    if ssim_score is not None and ssim_score < ssim_threshold:
        return "ssim-drop"
    if absdiff is not None and absdiff > 0.08:
        return "ssim-drop"
    if gap >= 30.0:
        return "min-gap-forced"
    return None


def _save_frame(frame_bgr: np.ndarray, index: int, timestamp: float, output_dir: Path) -> Path:
    path = output_dir / f"frame_{index:06d}_t{timestamp:.2f}.png"
    ok = cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise FrameExtractionError(f"Failed to write frame image: '{path}'")
    return path


def _ensure_output_dir(output_dir: Path, *, refuse_non_empty: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if refuse_non_empty:
        existing = [p for p in output_dir.iterdir() if p.name != ".gitkeep"]
        if existing:
            raise FrameExtractionError(
                f"Output directory is not empty: '{output_dir.resolve()}'. "
                "Choose a new run directory."
            )


def _write_frames_manifest(
    result: FrameExtractionResult,
    *,
    min_gap_sec: float,
    resize_width: int,
) -> Path:
    payload = {
        "version": "1",
        "video_path": str(result.video_path),
        "output_dir": str(result.output_dir),
        "sample_fps": result.sample_fps,
        "ssim_threshold": result.ssim_threshold,
        "min_gap_sec": min_gap_sec,
        "resize_width": resize_width,
        "frames": [_frame_to_manifest(frame, result.output_dir) for frame in result.frames],
        "total_candidates": result.total_candidates,
        "total_kept": result.total_kept,
    }
    manifest_path = result.output_dir / "frames.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path


def _frame_to_manifest(frame: KeyFrame, output_dir: Path) -> dict[str, object]:
    payload = asdict(frame)
    payload["path"] = _portable_path(frame.path, output_dir)
    return payload


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CannotOpenVideoError(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0:
            return 0.0
        return frames / fps
    finally:
        cap.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract visually distinct key-frames.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video file.")
    group.add_argument("--chunk-dir", type=Path, help="Directory of live chunks.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output frame directory.")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--ssim", type=float, default=0.92, help="Keep when SSIM is below this value.")
    parser.add_argument("--min-gap-sec", type=float, default=1.5)
    parser.add_argument("--resize-width", type=int, default=640)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.video is not None:
            result = extract_keyframes(
                video_path=args.video,
                output_dir=args.out_dir,
                sample_fps=args.sample_fps,
                ssim_threshold=args.ssim,
                min_gap_sec=args.min_gap_sec,
                resize_width=args.resize_width,
            )
        else:
            extract_keyframes_for_chunks(
                chunk_dir=args.chunk_dir,
                output_dir=args.out_dir,
                sample_fps=args.sample_fps,
                ssim_threshold=args.ssim,
                min_gap_sec=args.min_gap_sec,
                resize_width=args.resize_width,
            )
            result = None
        count = result.total_kept if result is not None else len(json.loads((args.out_dir / "frames.json").read_text())["frames"])
        print(f"{args.out_dir / 'frames.json'} ({count} frames)")
        return 0
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
