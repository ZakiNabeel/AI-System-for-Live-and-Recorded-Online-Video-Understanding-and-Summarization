# Plan 2.1 — Audio Extraction

> **Self-contained scope.** Convert any video file (full recorded MP4 from Plan 1.1, or a single live chunk `.ts` from Plan 1.2) into a normalized mono 16 kHz WAV file ready for speech-to-text. This plan does not require any STT model and can be tested with `ffprobe` alone.

---

## 1. Objective

Build `src/audio/extractor.py` that:

1. Accepts a path to a video file.
2. Strips out the audio track using ffmpeg.
3. Resamples to **16 000 Hz**, **mono**, **PCM s16le** (the format every modern STT system expects).
4. Writes to `data/audio/<run_id>/audio.wav` (or `chunk_NNNNNN.wav` for live chunks).
5. Returns metadata: duration, sample_rate, channel_count, file_size.

Keep this module pure I/O — no model loading. STT happens in Plan 2.2.

---

## 2. Contract

### Function signature
```python
def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
) -> AudioExtractionResult: ...
```

### Result dataclass
```python
@dataclass
class AudioExtractionResult:
    audio_path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    bits_per_sample: int
    file_size_bytes: int
    source_video: Path
```

### Batch helper for live chunks
```python
def extract_audio_for_chunks(
    chunk_dir: Path,
    output_dir: Path,
    sample_rate: int = 16000,
) -> Iterator[AudioExtractionResult]: ...
```
Yields one result per chunk in ascending index order. Skips any chunk whose audio already exists in `output_dir` (idempotent).

### CLI
```
python -m src.audio.extractor --video data/raw/<id>/video.mp4 --output data/audio/<id>/audio.wav
python -m src.audio.extractor --chunk-dir data/raw/<id> --output-dir data/audio/<id>
```

---

## 3. Dependencies

| Package / Tool | Purpose |
|---|---|
| `ffmpeg` system binary | The only thing actually doing work |
| `ffmpeg-python` (optional) | Pythonic wrapper, but plain `subprocess` is fine and pinning-free |

No new pip packages required if we shell out directly. Recommended approach: **plain subprocess** — fewer moving parts.

---

## 4. The Exact ffmpeg Command

```
ffmpeg -hide_banner -loglevel error \
       -y if overwrite else -n \
       -i <video_path> \
       -vn \
       -acodec pcm_s16le \
       -ar <sample_rate> \
       -ac 1 if mono else 2 \
       <output_path>
```

Flag reasoning:
- `-vn` — drop video.
- `-acodec pcm_s16le` — raw signed 16-bit little-endian PCM, the WAV default everyone expects.
- `-ar` — resample target rate.
- `-ac 1` — downmix to mono (Whisper, Vosk, etc. all want mono).
- `-y` / `-n` — overwrite vs. fail if exists.

---

## 5. Phased Implementation

### Phase A — Single-file extraction (~45 min)
1. Create `src/audio/__init__.py`, `src/audio/extractor.py`.
2. Implement `extract_audio()`:
   - Validate `video_path` exists; if not, raise `FileNotFoundError` with full path.
   - `output_path.parent.mkdir(parents=True, exist_ok=True)`.
   - Build the arg list. Use `["ffmpeg", "-hide_banner", "-loglevel", "error", ...]`.
   - `subprocess.run(args, check=True, capture_output=True, text=True)`.
   - On `CalledProcessError`, raise `AudioExtractionError(f"ffmpeg failed: {e.stderr}")`.
3. After success, call `ffprobe` to populate the result dataclass (see Phase B).

### Phase B — ffprobe metadata helper (~30 min)
```python
def _probe_audio(path: Path) -> dict:
    args = ["ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", "-select_streams", "a:0",
            str(path)]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return json.loads(out)
```
Map JSON fields to result fields:
| Result | JSON path |
|---|---|
| `duration_sec` | `format.duration` (float) |
| `sample_rate` | `streams[0].sample_rate` (int) |
| `channels` | `streams[0].channels` |
| `bits_per_sample` | `streams[0].bits_per_sample` (default 16) |
| `file_size_bytes` | `path.stat().st_size` |

### Phase C — Idempotency (~15 min)
In `extract_audio`:
- If `output_path.exists()` and `overwrite is False`, skip ffmpeg, just probe and return.
- If `output_path.exists()` and `overwrite is True`, ffmpeg's `-y` will replace it.

### Phase D — Batch chunk helper (~30 min)
```python
def extract_audio_for_chunks(chunk_dir, output_dir, sample_rate=16000):
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(chunk_dir.glob("chunk_*.ts")) + \
             sorted(chunk_dir.glob("chunk_*.mp4"))
    for chunk in sorted(chunks):
        out = output_dir / (chunk.stem + ".wav")
        yield extract_audio(chunk, out, sample_rate=sample_rate)
```

### Phase E — CLI (~20 min)
Two mutually exclusive arg groups: `(--video, --output)` or `(--chunk-dir, --output-dir)`. Print result(s) as JSON lines.

### Phase F — Tests (~1 hr)

Generate a deterministic test asset on the fly to avoid checked-in binaries:
```python
@pytest.fixture
def silent_video(tmp_path):
    out = tmp_path / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3:r=10",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(out)
    ], check=True, capture_output=True)
    return out
```

1. **Happy path** — extract from `silent_video`; assert WAV exists, `sample_rate == 16000`, `channels == 1`, `2.9 < duration_sec < 3.1`.
2. **Idempotency** — call twice without overwrite; assert second call did not invoke ffmpeg (use `monkeypatch` on `subprocess.run` and count calls; or check mtime unchanged).
3. **Overwrite** — call with `overwrite=True`; assert mtime updates.
4. **Missing file** — point at non-existent path; expect `FileNotFoundError`.
5. **No audio track** — generate a silent video without an audio stream; assert `AudioExtractionError` is raised with helpful message.
6. **Batch mode** — produce 3 fake chunks; assert helper yields 3 results in order; assert second call is a no-op (idempotent).

---

## 6. File Layout After Plan 2.1
```
src/audio/
  __init__.py
  extractor.py
  errors.py            # AudioExtractionError
tests/audio/
  __init__.py
  test_extractor.py
  conftest.py          # silent_video fixture
data/audio/            (created at runtime)
```

---

## 7. Acceptance Criteria

- [ ] `extract_audio(video_path, output_path)` produces a 16 kHz mono PCM WAV.
- [ ] `ffprobe` confirms exactly: 1 channel, 16000 Hz, s16, no video stream.
- [ ] Re-running on existing file is a no-op unless `overwrite=True`.
- [ ] Batch helper processes all chunks in a live-mode folder in order.
- [ ] All unit tests green.
- [ ] CLI prints JSON metadata on success.

---

## 8. Edge Cases & Pitfalls

1. **Video has no audio track** — `ffmpeg` exits 0 but produces a 0-byte WAV. Always re-probe and fail if `duration_sec == 0`.
2. **Multiple audio tracks** — pick the first by default (`-map 0:a:0`); add a `--track` CLI flag if needed later.
3. **Variable-frame-rate or weird sample rates** — `-ar 16000` forces resample; safe.
4. **Very large videos** — ffmpeg streams; memory is not a concern. But disk: a 2-hour mono 16 kHz s16 WAV ≈ 230 MB. Acceptable.
5. **Filename Unicode** — pass paths via `str(Path)` (handles Windows backslashes); avoid passing through a shell.
6. **Concurrency** — running multiple ffmpeg processes is fine, but cap parallelism at `os.cpu_count() // 2` if used in batch (Plan 5.1 may need this).
7. **`.ts` chunks with discontinuities** — these can confuse ffmpeg's duration probe. Add `-fflags +igndts` if ffprobe duration looks off.

---

## 9. Out of Scope

- Speech-to-text (Plan 2.2).
- Audio denoising, normalization, or VAD (could be added later under `src/audio/preprocess.py`; not required by minimum spec).
- Streaming/online audio extraction (not needed; chunks are extracted in batch as they complete).

---

## 10. Definition of Done

A developer reading only this file can run `python -m src.audio.extractor --video <any_mp4> --output out.wav` and get a 16 kHz mono PCM WAV file plus JSON metadata, on Windows or Linux, with only ffmpeg installed.
