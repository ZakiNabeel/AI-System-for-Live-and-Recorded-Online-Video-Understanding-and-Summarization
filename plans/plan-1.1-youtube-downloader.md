# Plan 1.1 — YouTube Downloader

> **Self-contained scope.** This plan defines a single CLI-callable Python module that, given a YouTube URL, downloads the video to a known location and returns metadata. No other module of the project is required to implement or test this plan.

---

## 1. Objective

Build a Python module `src/ingest/youtube_downloader.py` that:

1. Accepts a YouTube URL (regular video, Shorts, or live archive).
2. Downloads the **video file** (with audio merged) at a configurable resolution.
3. Saves it to `data/raw/<run_id>/video.<ext>`.
4. Returns a structured metadata dict (title, duration, channel, etc.).

This module is the **batch-mode** entry to the pipeline. Live streams are handled by Plan 1.2.

---

## 2. Contract (Inputs & Outputs)

### Function signature
```python
def download_youtube_video(
    url: str,
    run_id: str,
    output_root: Path = Path("data/raw"),
    max_height: int = 720,
    prefer_codec: str = "mp4",
) -> VideoDownloadResult: ...
```

### `VideoDownloadResult` dataclass
```python
@dataclass
class VideoDownloadResult:
    run_id: str
    url: str
    video_path: Path        # absolute path to downloaded file
    title: str
    duration_sec: float
    channel: str
    upload_date: str        # ISO 8601 (YYYY-MM-DD)
    width: int
    height: int
    fps: float
    has_subtitles: bool     # True if YouTube auto/uploaded subs exist
    available_subtitle_langs: list[str]
    raw_metadata: dict      # full yt-dlp info dict
```

### CLI
```
python -m src.ingest.youtube_downloader --url "<URL>" [--run-id RUN_ID] [--max-height 720]
```
If `--run-id` is omitted, generate a UUID4. Print the resulting `VideoDownloadResult` as JSON to stdout.

---

## 3. Dependencies

| Package | Purpose | Notes |
|---|---|---|
| `yt-dlp` | Video download + metadata | Pin to `>=2024.08.06` |
| `ffmpeg` | Muxing audio/video, codec conversion | **System binary** — must be on PATH |
| `python-dotenv` | Load `.env` if a YouTube cookies file path is configured | Optional |

Add to `requirements.txt`:
```
yt-dlp>=2024.08.06
python-dotenv>=1.0.1
```

ffmpeg install check (PowerShell): `ffmpeg -version`. If missing, instruct user to install via `winget install Gyan.FFmpeg` and restart the shell.

---

## 4. Phased Implementation

### Phase A — Skeleton & dataclass (~30 min)
1. Create folders: `src/ingest/`, `tests/ingest/`, `data/raw/`.
2. Create `src/__init__.py`, `src/ingest/__init__.py`.
3. Define the `VideoDownloadResult` dataclass exactly as in §2.
4. Stub `download_youtube_video()` that raises `NotImplementedError`.
5. Verify `python -c "from src.ingest.youtube_downloader import VideoDownloadResult"` works.

### Phase B — yt-dlp integration (~1 hr)
1. Inside the function, build a `yt_dlp.YoutubeDL` options dict:
   ```python
   ydl_opts = {
       "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]",
       "outtmpl": str(output_dir / "video.%(ext)s"),
       "merge_output_format": prefer_codec,
       "quiet": True,
       "no_warnings": True,
       "writesubtitles": False,        # subtitles handled in Plan 2.2
       "writeautomaticsub": False,
       "noplaylist": True,
       "retries": 3,
   }
   ```
2. Create `output_dir = output_root / run_id` (mkdir parents=True, exist_ok=True).
3. Call `ydl.extract_info(url, download=True)`.
4. Glob the resulting file: `next(output_dir.glob("video.*"))`.

### Phase C — Metadata extraction (~30 min)
Map yt-dlp's `info` dict into `VideoDownloadResult`:
| Result field | Source key in yt-dlp `info` |
|---|---|
| `title` | `title` |
| `duration_sec` | `duration` |
| `channel` | `channel` or `uploader` |
| `upload_date` | `upload_date` (reformat `YYYYMMDD` → `YYYY-MM-DD`) |
| `width`, `height`, `fps` | `width`, `height`, `fps` |
| `has_subtitles` | `bool(info.get("subtitles") or info.get("automatic_captions"))` |
| `available_subtitle_langs` | sorted union of keys in `subtitles` and `automatic_captions` |

### Phase D — Error handling (~30 min)
Catch and translate these to friendly errors:
- `yt_dlp.utils.DownloadError` containing "Private video" → `PrivateVideoError`
- `yt_dlp.utils.DownloadError` containing "Video unavailable" → `UnavailableVideoError`
- Network errors (`urllib.error.URLError`) → `NetworkError`
- ffmpeg merge failure (DownloadError mentioning "ffmpeg") → `FFmpegMissingError` with install hint

Define each as a subclass of a base `IngestError(Exception)` in `src/ingest/errors.py`.

### Phase E — CLI wrapper (~30 min)
1. In `if __name__ == "__main__":` block, use `argparse` with flags from §2.
2. Generate `run_id = str(uuid.uuid4())` if not provided.
3. Print result as JSON: `print(json.dumps(asdict(result), default=str, indent=2))`.

### Phase F — Tests (~1 hr)
Create `tests/ingest/test_youtube_downloader.py`:

1. **Unit (mocked)** — patch `yt_dlp.YoutubeDL.extract_info` to return a fake info dict; assert `VideoDownloadResult` fields are mapped correctly. No network.
2. **Integration (network)** — mark with `@pytest.mark.network`; download a tiny known-stable video (e.g., a CC0 short clip you control or `https://www.youtube.com/watch?v=jNQXAC9IVRw` — "Me at the zoo", 19s). Assert file exists and `duration_sec ≈ 19`.
3. **Error mapping** — patch `extract_info` to raise `DownloadError("Private video")`; assert `PrivateVideoError` is raised.

Run network tests only on demand: `pytest -m network`.

---

## 5. File Layout After Plan 1.1
```
src/
  __init__.py
  ingest/
    __init__.py
    youtube_downloader.py
    errors.py
tests/
  ingest/
    __init__.py
    test_youtube_downloader.py
data/
  raw/                    (empty, populated at runtime)
requirements.txt
```

---

## 6. Acceptance Criteria

- [ ] `pip install -r requirements.txt` succeeds in a fresh venv.
- [ ] `python -m src.ingest.youtube_downloader --url <test_url>` produces `data/raw/<run_id>/video.mp4` and prints valid JSON.
- [ ] `pytest tests/ingest/ -m "not network"` passes with 100% of unit tests green.
- [ ] `pytest tests/ingest/ -m network` downloads the canary video successfully when run manually.
- [ ] Calling with an invalid URL prints a single-line friendly error and exits with code 2.
- [ ] No file outside `data/raw/<run_id>/` is created during normal operation.

---

## 7. Edge Cases & Pitfalls

1. **Live-stream URLs accidentally passed here** — yt-dlp will block until the stream ends. Detect `info.get("is_live") is True` *before* download; raise `LiveStreamNotSupportedError` and tell the caller to use Plan 1.2.
2. **Age-restricted / login-walled videos** — out of scope; surface the yt-dlp error verbatim.
3. **Very long videos (>2 hr)** — issue a warning to the log if `duration_sec > 7200`; don't fail.
4. **Disk space** — before download, check `shutil.disk_usage(output_root).free > 2 * estimated_size`. If unavailable, skip the check.
5. **Filename collisions** — `outtmpl` uses fixed `video.%(ext)s` inside a UUID folder, so collisions are impossible by design. Do **not** include video title in the filename (breaks downstream globbing).
6. **Windows paths** — always use `pathlib.Path`, never raw strings with backslashes.

---

## 8. Out of Scope (for this plan)

- Subtitle/transcript download (Plan 2.2 will do that separately).
- Audio-only extraction (Plan 2.1).
- Live streams (Plan 1.2).
- Any UI — CLI only.

---

## 9. Definition of Done

A teammate clones the repo, runs `pip install -r requirements.txt`, then runs the CLI command from §2 with any public YouTube URL, and gets a working `.mp4` file plus a JSON metadata blob — without reading any other plan file.
