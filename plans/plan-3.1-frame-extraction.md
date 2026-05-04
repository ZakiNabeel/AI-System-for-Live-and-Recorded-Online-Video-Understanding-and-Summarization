# Plan 3.1 — Frame Extraction & Differencing

> **Self-contained scope.** Read a video file (or live chunk) and emit a sparse set of "interesting" frames — frames that are visually different from their neighbors. Down-stream OCR and summarization run only on these key-frames, so the rest of the pipeline gets a 50–500× cost reduction relative to processing every frame. No knowledge of any other module is required.

---

## 1. Objective

Build `src/vision/frame_extractor.py` that:

1. Accepts a video path.
2. Samples frames at a configurable rate (default: 1 fps).
3. For each candidate frame, computes a similarity metric vs. the last *kept* frame.
4. Keeps the candidate **only if** it is different enough (SSIM below a threshold, plus a minimum time gap).
5. Saves kept frames as PNG files: `frame_NNNNNN_t<seconds>.png`.
6. Writes a `frames.json` manifest with timestamps and metadata.

This is the **DIP heart** of the project — everything downstream visual depends on it.

---

## 2. Contract

### Function signature
```python
def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 1.0,
    ssim_threshold: float = 0.92,    # below = keep
    min_gap_sec: float = 1.5,
    resize_width: int = 640,
    always_keep_first: bool = True,
) -> FrameExtractionResult: ...
```

### Result + per-frame record
```python
@dataclass
class KeyFrame:
    index: int          # ascending, zero-based
    timestamp: float    # seconds from video start
    path: Path          # absolute
    width: int
    height: int
    ssim_to_prev: float | None    # None for first frame
    reason: Literal["first", "ssim-drop", "min-gap-forced"]

@dataclass
class FrameExtractionResult:
    video_path: Path
    output_dir: Path
    frames: list[KeyFrame]
    total_candidates: int    # how many frames were sampled
    total_kept: int          # len(frames)
    sample_fps: float
    ssim_threshold: float
```

### CLI
```
python -m src.vision.frame_extractor --video <mp4> --out-dir data/frames/<id> [--sample-fps 1] [--ssim 0.92]
```
Writes `output_dir/frames.json` and one PNG per kept frame.

---

## 3. Why SSIM (not absolute pixel diff)

| Metric | Pros | Cons |
|---|---|---|
| Absolute frame difference | Cheap, simple | Sensitive to noise, lighting, encoder dithering — produces lots of false positives |
| MSE | Cheap | Same problems as abs diff |
| **SSIM** (Structural Similarity) | Robust to small noise, matches human perception of "are these the same?" | Slightly more compute |
| pHash / dHash | Very cheap, robust | Coarse — may miss small but important changes (a code line) |

We use **SSIM** as primary, with an absolute-diff sanity backup so a fully-black/all-white frame doesn't slip through.

---

## 4. Phased Implementation

### Phase A — Skeleton + dataclasses (~30 min)
1. Create `src/vision/__init__.py`, `src/vision/frame_extractor.py`, `tests/vision/`.
2. Define `KeyFrame`, `FrameExtractionResult`.
3. Stub `extract_keyframes` raising `NotImplementedError`.

### Phase B — Frame iterator (~1 hr)
Use **OpenCV** for reading because it handles every container ffmpeg can read and is faster than ffmpeg→PNG→re-read.

```python
def _iter_sampled_frames(video_path, sample_fps):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise CannotOpenVideoError(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / sample_fps))
    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / src_fps
        yield ts, frame
        idx += step
        if idx >= total:
            break
    cap.release()
```

> **Note:** `cap.set(POS_FRAMES)` can be slow/inaccurate on some codecs. If you need exact timestamps, alternative: read sequentially and use `idx % step == 0`. Default to the seek approach — it's much faster on long videos.

### Phase C — Resize + grayscale prep (~20 min)
Both inputs to SSIM should be the same size and ideally smaller than the source (faster).

```python
def _prep(frame, target_width):
    h, w = frame.shape[:2]
    if w != target_width:
        scale = target_width / w
        frame = cv2.resize(frame, (target_width, int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray
```

### Phase D — SSIM comparison (~30 min)
```python
from skimage.metrics import structural_similarity as ssim_fn

def _ssim(prev_gray, curr_gray):
    return float(ssim_fn(prev_gray, curr_gray, data_range=255))
```
Add `scikit-image` to requirements.

### Phase E — Decision logic (~45 min)
For each sampled frame at time `t`:

```
if first frame and always_keep_first: keep, reason="first"
elif (t - last_kept_t) >= min_gap_sec:
    if ssim < threshold:        keep, reason="ssim-drop"
    elif (t - last_kept_t) >= 30: keep, reason="min-gap-forced"   # heartbeat every 30s
```

The `min-gap-forced` heartbeat ensures static videos (e.g., a webcam pointed at a whiteboard) still get a few key-frames for downstream summarization context.

### Phase F — Save + manifest (~30 min)
```python
def _save_frame(frame_bgr, idx, ts, output_dir):
    name = f"frame_{idx:06d}_t{ts:.2f}.png"
    path = output_dir / name
    cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return path
```

After all frames extracted, dump:
```json
{
  "version": "1",
  "video_path": "data/raw/<id>/video.mp4",
  "sample_fps": 1.0,
  "ssim_threshold": 0.92,
  "min_gap_sec": 1.5,
  "frames": [
    {"index": 0, "timestamp": 0.0, "path": "frame_000000_t0.00.png",
     "width": 640, "height": 360, "ssim_to_prev": null, "reason": "first"},
    ...
  ],
  "total_candidates": 312,
  "total_kept": 47
}
```

### Phase G — Live-mode chunk variant (~30 min)
```python
def extract_keyframes_for_chunks(chunk_dir, output_dir, **kwargs) -> list[FrameExtractionResult]:
    # Maintain a "last_kept_gray" buffer across chunks so the first frame
    # of chunk N is compared against the last kept frame of chunk N-1.
```

This is important: without cross-chunk state, each chunk would always emit its first frame, defeating SSIM filtering for live mode.

### Phase H — CLI (~20 min)
Standard argparse. On success, print path to `frames.json`.

### Phase I — Tests (~2 hr)

Generate synthetic test videos in fixtures (no checked-in MP4s):

```python
@pytest.fixture
def static_video(tmp_path):  # 5 s, all gray
    path = tmp_path / "static.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                    "-i", "color=c=gray:s=320x240:d=5:r=10",
                    "-c:v", "libx264", str(path)], check=True, capture_output=True)
    return path

@pytest.fixture
def changing_video(tmp_path):  # 5 s, color changes every second
    path = tmp_path / "changing.mp4"
    cmd = ["ffmpeg", "-y"]
    parts = []
    for i, color in enumerate(["red","green","blue","yellow","cyan"]):
        cmd += ["-f","lavfi","-i", f"color=c={color}:s=320x240:d=1:r=10"]
        parts.append(f"[{i}:v]")
    cmd += ["-filter_complex", "".join(parts) + "concat=n=5:v=1:a=0[v]",
            "-map", "[v]", "-c:v", "libx264", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path
```

Tests:
1. **Static video** — assert `total_kept == 1` (only the first frame, plus possibly the heartbeat-forced one if duration > 30 s).
2. **Changing video** — assert `total_kept >= 5` (one per color change).
3. **min_gap respected** — set `min_gap_sec=2`, sample_fps=10; assert no two kept timestamps closer than 2 s.
4. **SSIM ordering** — manually craft two near-identical synthetic frames; assert SSIM > 0.99 → not kept; alter pixels; SSIM drops → kept.
5. **Manifest schema** — load `frames.json`, validate keys.
6. **Idempotency** — run twice into the same dir; second run errors loudly OR is a no-op (configurable). Default: refuse if dir non-empty.
7. **Live chunk continuity** — split changing_video into 1-s chunks; run chunk variant; assert behavior matches single-file variant within ±1 frame.

---

## 5. File Layout After Plan 3.1
```
src/vision/
  __init__.py
  frame_extractor.py
  errors.py
tests/vision/
  __init__.py
  test_frame_extractor.py
  conftest.py
data/frames/            (created at runtime)
```

---

## 6. Dependencies
```
opencv-python>=4.10.0
scikit-image>=0.24.0
numpy
```

---

## 7. Acceptance Criteria

- [ ] CLI run on a 60 s video produces ≤ 30 PNGs (with default thresholds) and a valid `frames.json`.
- [ ] Static-video test extracts ≤ 2 frames.
- [ ] Changing-video test extracts ≥ N frames where N = number of scene changes.
- [ ] All saved PNG files open in any image viewer (smoke test: `cv2.imread` returns non-None).
- [ ] All unit tests pass on Windows and Linux.
- [ ] No memory leak when processing a 1-hour video (RSS stable, ≤ ~500 MB).

---

## 8. Edge Cases & Pitfalls

1. **`cap.set(POS_FRAMES)` inaccuracy on VFR videos** — fall back to sequential reading if `cap.get(CAP_PROP_POS_FRAMES)` after seek doesn't match request within ±2 frames.
2. **Black frames at fade-outs** — they have low SSIM to anything; would all be kept as separate "events." Add a heuristic: if mean pixel intensity < 5 or > 250, treat as fade and don't keep more than one consecutive.
3. **Video without keyframes / corrupted GOP** — OpenCV returns `ok=False`; raise `CannotOpenVideoError` with the file path and ffprobe output.
4. **Memory with very high resolution** — always resize to `resize_width` before SSIM and before saving. Save PNG at 640 px width by default.
5. **PNG vs JPEG** — PNG is lossless; OCR (Plan 3.3) benefits from no compression artifacts. Use PNG. JPEG saves disk but hurts OCR.
6. **Zero-duration video** — fail fast with `EmptyVideoError`.
7. **scikit-image `data_range` argument** — required for grayscale uint8; without it, SSIM returns warnings and incorrect values.
8. **Rotated videos with metadata-only orientation** — OpenCV ignores rotation metadata. Pre-check `ffprobe ... -show_entries stream_side_data_list` and rotate via `cv2.rotate` if a rotate tag is present (90/180/270).

---

## 9. Out of Scope

- Image enhancement / preprocessing (Plan 3.2).
- OCR / object detection (Plan 3.3).
- Saving lossy thumbnails for display (could be added — small JPEG copies under `frames/thumbs/`).

---

## 10. Definition of Done

A developer can run the CLI on any MP4, get a `frames.json` and a folder of PNGs containing only "interesting" frames, and verify visually that adjacent kept frames look meaningfully different — using only this plan file.
