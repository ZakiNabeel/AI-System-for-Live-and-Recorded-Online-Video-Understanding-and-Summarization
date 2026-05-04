# Plan 3.2 — DIP Enhancements

> **Self-contained scope.** Apply classical Digital Image Processing operations (grayscale, denoise, contrast, sharpen, morphology) to the key-frames produced by Plan 3.1, producing OCR-ready images that significantly improve text-detection recall in Plan 3.3. This is the **DIP-coursework headline module** — it is deliberately implemented with classical OpenCV ops (no neural nets) so it directly demonstrates DIP fundamentals.

---

## 1. Objective

Build `src/vision/enhancer.py` that:

1. Reads PNG key-frames from `data/frames/<run_id>/`.
2. Produces, for each frame, an enhanced version optimized for OCR text extraction.
3. Saves enhanced frames alongside originals as `frame_NNNNNN_t<seconds>.enhanced.png`.
4. Optionally produces one or more "diagnostic" intermediate images per frame (`*.gray.png`, `*.thresh.png`) for the project report.
5. Records all enhancement parameters in `enhancements.json` for reproducibility.

This module is **independent of Plan 3.3** — its output contract is "a PNG per input frame, same dimensions, single channel uint8."

---

## 2. Contract

### Function signature
```python
def enhance_frames(
    frames_json_path: Path,
    output_dir: Path | None = None,    # default: same dir as input frames
    profile: Literal["default", "screen", "whiteboard", "scene"] = "default",
    save_intermediates: bool = False,
) -> EnhancementResult: ...
```

### Result + per-frame record
```python
@dataclass
class EnhancedFrame:
    source_path: Path
    enhanced_path: Path
    intermediates: dict[str, Path]   # name -> path; empty if save_intermediates=False
    profile: str
    parameters: dict                  # exact params used (for the report)

@dataclass
class EnhancementResult:
    frames: list[EnhancedFrame]
    profile: str
    parameters: dict
```

### CLI
```
python -m src.vision.enhancer --frames data/frames/<id>/frames.json [--profile default] [--save-intermediates]
```

---

## 3. Profiles — Pre-Tuned Parameter Sets

Different content types need different DIP pipelines. Provide four built-in profiles:

| Profile | Use case | Strategy |
|---|---|---|
| `default` | Mixed content (auto) | Light denoise + adaptive threshold |
| `screen` | Screen recordings, slides, code editors | Aggressive sharpening + Otsu threshold (text is high-contrast) |
| `whiteboard` | Whiteboard / blackboard photos | Strong contrast equalization + adaptive threshold |
| `scene` | Real-world video (lectures with cameras, signs, etc.) | Bilateral filter + CLAHE; no thresholding (preserve grayscale for OCR) |

Each profile is a `dict` of params consumed by the pipeline (see §4). Store profiles in `src/vision/enhance_profiles.py`.

---

## 4. The DIP Pipeline (in order)

Each step is conditional on profile parameters; some can be turned off.

### Step 1 — Read & convert to grayscale
```python
img = cv2.imread(str(path), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
**Why:** OCR engines and morphological ops work on a single channel. Grayscale also halves processing cost.

### Step 2 — Denoise (Gaussian / Bilateral)
- `default` / `screen`: `cv2.GaussianBlur(gray, (3,3), 0)` — fast, removes salt-and-pepper noise.
- `whiteboard` / `scene`: `cv2.bilateralFilter(gray, 9, 75, 75)` — preserves edges, removes photo-noise.

**Why:** Noise creates spurious "text-like" features; threshold becomes meaningless; OCR confuses speckles with characters.

### Step 3 — Contrast enhancement (CLAHE)
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(denoised)
```
Used in `whiteboard` and `scene`. **Why:** uneven lighting (room light on whiteboard) compresses dynamic range; CLAHE rescues local contrast without amplifying global noise.

### Step 4 — Sharpening (unsharp mask)
```python
blur = cv2.GaussianBlur(enhanced, (0,0), sigmaX=2.0)
sharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
```
Used in `screen` profile. **Why:** sub-pixel rendering and video compression blur edges of small text.

### Step 5 — Threshold (binarization)
- `screen`: Otsu's method — text vs. background histogram is bimodal.
  ```python
  _, binary = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  ```
- `default`, `whiteboard`: adaptive Gaussian threshold (handles uneven background).
  ```python
  binary = cv2.adaptiveThreshold(
      img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
  ```
- `scene`: skip (Tesseract performs better on grayscale here).

**Why:** OCR is heavily threshold-dependent. The right threshold dramatically increases character-recognition rate.

### Step 6 — Invert if needed
If the binarized image is mostly black with white text (dark theme code editor), invert so text is dark on light:
```python
mean = binary.mean()
if mean < 127:
    binary = cv2.bitwise_not(binary)
```

### Step 7 — Morphology (dilation / opening)
- `screen`: thin antialiased text is fragile after threshold; **dilate** with a 2×2 kernel to reconnect pixels.
  ```python
  kernel = np.ones((2,2), np.uint8)
  binary = cv2.dilate(binary, kernel, iterations=1)
  ```
- `whiteboard`: marker strokes are often broken; same 2×2 dilation.
- `default`: small `cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2,2)))` to remove specks while keeping characters intact.

**Why:** Tesseract needs continuous strokes; one-pixel breaks dramatically reduce confidence.

### Step 8 — (Optional) Deskew
For photographed whiteboards / scanned slides, detect text orientation and rotate to horizontal.

```python
def deskew(image, max_angle=10):
    coords = np.column_stack(np.where(image > 0))
    if len(coords) < 50: return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > max_angle: return image
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```
Used in `whiteboard` profile only.

---

## 5. Phased Implementation

### Phase A — Skeleton + profile registry (~30 min)
1. Create `src/vision/enhancer.py`, `src/vision/enhance_profiles.py`.
2. Define `PROFILES: dict[str, dict]` with the four parameter sets.

Example `default` profile:
```python
{
    "grayscale": True,
    "denoise":   {"method": "gaussian", "ksize": 3},
    "clahe":     None,
    "sharpen":   None,
    "threshold": {"method": "adaptive_gaussian", "block": 31, "C": 10},
    "invert_if_dark": True,
    "morph":     {"op": "open", "kernel": [2,2], "iters": 1},
    "deskew":    False,
}
```

### Phase B — Step implementations (~2 hr)
Each step lives as a tiny pure function in `src/vision/dip_steps.py`:
```python
def grayscale(bgr): ...
def denoise(gray, method, ksize=3): ...
def clahe(gray, clip_limit=2.0, tile=(8,8)): ...
def sharpen(gray, sigma=2.0, amount=0.5): ...
def binarize(gray, method, **kw): ...
def invert_if_dark(binary): ...
def morph(binary, op, kernel, iters=1): ...
def deskew(binary, max_angle=10): ...
```
Each takes a `np.ndarray`, returns a `np.ndarray`. Easy to unit-test.

### Phase C — Pipeline runner (~45 min)
```python
def apply_profile(bgr_image, profile_params, *, collect_intermediates=False):
    intermediates = {}
    img = bgr_image
    if profile_params["grayscale"]:
        img = grayscale(img)
        if collect_intermediates: intermediates["gray"] = img.copy()
    if profile_params["denoise"]:
        img = denoise(img, **profile_params["denoise"])
    if profile_params["clahe"]:
        img = clahe(img, **profile_params["clahe"])
    if profile_params["sharpen"]:
        img = sharpen(img, **profile_params["sharpen"])
    if profile_params["threshold"]:
        img = binarize(img, **profile_params["threshold"])
        if collect_intermediates: intermediates["thresh"] = img.copy()
    if profile_params["invert_if_dark"]:
        img = invert_if_dark(img)
    if profile_params["morph"]:
        img = morph(img, **profile_params["morph"])
    if profile_params["deskew"]:
        img = deskew(img)
    return img, intermediates
```

### Phase D — File I/O wrapper (~30 min)
- Read `frames.json` from Plan 3.1.
- For each `KeyFrame`, run pipeline → save `*.enhanced.png` next to original.
- Save intermediates with suffixes (`*.gray.png`, `*.thresh.png`) when requested.
- Write `enhancements.json` with the per-frame record from §2.

### Phase E — CLI (~20 min)
Standard argparse with `--profile`, `--save-intermediates`. Print summary stats: `enhanced 47/47 frames; avg time/frame: 32 ms`.

### Phase F — Auto-profile detection (~optional, 1 hr)
Heuristic to pick a profile when `--profile auto`:

```python
def detect_profile(gray):
    h_var = np.var(np.diff(gray.mean(axis=0)))    # column-mean variance
    edges = cv2.Canny(gray, 50, 150).mean()
    if edges > 50: return "screen"           # lots of sharp edges → text-heavy slides
    elif gray.std() < 30: return "whiteboard" # low contrast
    else: return "scene"
```
Mark as best-effort; document the heuristic.

### Phase G — Tests (~2 hr)

1. **Each step in isolation** — for each of the 8 steps in §4, build a known-input `np.ndarray`, assert the output has expected properties (e.g., `binarize` produces only 0 and 255 values; `deskew` preserves shape).
2. **Synthetic text frame** — generate a frame with `cv2.putText("ABCDEF", ...)` on a noisy gray background; run `default` profile; assert output is binary, OCR-ready (test by running pytesseract on it, expect ≥ 4 of 6 chars correct).
3. **Profile dispatch** — call with each profile name; assert correct param set is logged in `enhancements.json`.
4. **Intermediates flag** — assert `*.gray.png` and `*.thresh.png` exist iff `save_intermediates=True`.
5. **Deskew** — synthetic frame with text rotated 5°; assert post-deskew text bounding box is more horizontal (compare `cv2.minAreaRect`).
6. **Idempotency** — calling twice produces identical output bytes.

---

## 6. File Layout After Plan 3.2
```
src/vision/
  enhancer.py
  enhance_profiles.py
  dip_steps.py
tests/vision/
  test_dip_steps.py
  test_enhancer.py
  fixtures/synthetic_text.py     # helper to render test frames
```

---

## 7. Dependencies
Already installed in Plan 3.1: `opencv-python`, `numpy`. No new packages required.

---

## 8. Acceptance Criteria

- [ ] CLI run produces one `*.enhanced.png` per input PNG.
- [ ] All four profiles run without error on the same input set.
- [ ] `enhancements.json` records every parameter used per frame.
- [ ] Each step has a corresponding unit test (8 tests minimum).
- [ ] On a synthetic text frame, default profile improves Tesseract OCR character accuracy by ≥ 30 % vs. the unenhanced original.
- [ ] No frame causes a crash even if it is mostly black, mostly white, or extremely small.

---

## 9. Edge Cases & Pitfalls

1. **Image too small** (< 50 px width) — adaptive threshold block size becomes invalid. Skip enhancement and copy original through; warn in log.
2. **All-black or all-white frames** — Otsu returns trivial threshold; skip threshold step; binarize with fixed value.
3. **Color text on color background** — grayscale conversion can collapse contrast (e.g., red text on blue). Add a `--color-aware` flag that runs the pipeline on each channel independently and picks the highest-contrast result.
4. **Large frames** (> 1920 px wide) — bilateral filter is O(n²) in kernel size; processing can take seconds. Either downscale to 1280 px first or warn.
5. **Memory** — process frames sequentially, never load all at once.
6. **Already-binarized frames** (e.g., screenshots of B&W documents) — threshold becomes a no-op; that's fine, but add an early-exit check to save time.
7. **Profile parameter typos** — validate `profile_params` against a Pydantic schema or a `set` of allowed keys; raise on unknown keys.
8. **OpenCV adaptive_threshold block size must be odd** — clamp/round-up.

---

## 10. Out of Scope

- Neural-net-based enhancement (super-resolution, denoising autoencoders).
- Detecting/removing watermarks or logos.
- Color text segmentation (color-aware mode is a stretch goal).
- OCR (Plan 3.3).

---

## 11. Definition of Done

A developer can run `python -m src.vision.enhancer --frames frames.json --profile screen --save-intermediates` on the output of Plan 3.1 and visually inspect the resulting binarized PNGs to confirm text is sharp, clean, and OCR-ready — using only this plan file.
