# Plan 1.3 — Orchestrator Skeleton

> **Self-contained scope.** This plan defines the top-level *router* that decides whether an input URL should be treated as a recorded video (Plan 1.1 path) or a live stream (Plan 1.2 path), and that creates the canonical run directory and config used by every later module. No actual ML/CV work happens here — it is plumbing only.

---

## 1. Objective

Build `src/orchestrator.py` that:

1. Accepts a URL and a `mode` flag (`auto`, `recorded`, `live`).
2. Auto-detects mode when set to `auto` (probes whether the URL is a live stream).
3. Generates a unique `run_id` and creates the full per-run folder skeleton.
4. Loads global config (`config.yaml`) and per-run overrides.
5. Dispatches to either `download_youtube_video` (Plan 1.1) or `start_live_capture` (Plan 1.2).
6. Writes a `manifest.json` describing what was ingested and where.
7. Returns a `RunContext` object that downstream modules consume.

This plan does **not** depend on Plans 2.x–5.x existing. It only depends on the public function signatures of Plans 1.1 and 1.2 — both of which are stable contracts (see those plans). To allow this plan to be developed in parallel, dummy stubs of those functions may be used during testing.

---

## 2. Contract

### `RunContext` dataclass
```python
@dataclass
class RunContext:
    run_id: str
    mode: Literal["recorded", "live"]
    url: str
    started_at: datetime
    paths: RunPaths              # see below
    config: dict                 # merged config
    ingest_result: VideoDownloadResult | LiveCaptureHandle | None
    manifest_path: Path
```

### `RunPaths`
```python
@dataclass(frozen=True)
class RunPaths:
    root: Path                   # data/.../<run_id>
    raw: Path                    # data/raw/<run_id>
    audio: Path                  # data/audio/<run_id>
    frames: Path                 # data/frames/<run_id>
    intermediate: Path           # data/intermediate/<run_id>
    output: Path                 # data/output/<run_id>
    log_file: Path               # logs/<run_id>.log
```

### Top-level function
```python
def run(
    url: str,
    mode: Literal["auto", "recorded", "live"] = "auto",
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
) -> RunContext: ...
```

### CLI
```
python -m src.orchestrator --url "<URL>" [--mode auto|recorded|live] [--run-id ID]
```
Prints the manifest path on success.

---

## 3. Dependencies

| Package | Purpose |
|---|---|
| `pyyaml` | Load `config.yaml` |
| `python-dotenv` | Load `.env` |

Add to `requirements.txt`:
```
pyyaml>=6.0.2
```

Plus the dependencies of Plans 1.1 and 1.2 (yt-dlp, streamlink, ffmpeg).

---

## 4. Phased Implementation

### Phase A — Folder & config foundations (~30 min)
1. Create `config.yaml` at project root with defaults:
   ```yaml
   ingest:
     max_height: 720
     chunk_seconds: 10
   audio: { sample_rate: 16000, mono: true }
   speech:  { engine: whisper, model: small.en }
   frames:  { ssim_threshold: 0.92, min_gap_sec: 1.5 }
   llm:     { provider: anthropic, model: claude-sonnet-4-6 }
   ```
2. Create `.env.example`:
   ```
   ANTHROPIC_API_KEY=
   OPENAI_API_KEY=
   ```
3. Create `src/config.py` with `load_config(path) -> dict` that merges `config.yaml` with environment variables.

### Phase B — Run directory factory (~30 min)
```python
def create_run_paths(run_id: str, base: Path = Path(".")) -> RunPaths:
    paths = RunPaths(
        root=base / "data" / "runs" / run_id,
        raw=base / "data" / "raw" / run_id,
        audio=base / "data" / "audio" / run_id,
        frames=base / "data" / "frames" / run_id,
        intermediate=base / "data" / "intermediate" / run_id,
        output=base / "data" / "output" / run_id,
        log_file=base / "logs" / f"{run_id}.log",
    )
    for p in [paths.raw, paths.audio, paths.frames,
              paths.intermediate, paths.output]:
        p.mkdir(parents=True, exist_ok=True)
    paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    return paths
```

### Phase C — Mode detection (~30 min)
```python
def detect_mode(url: str) -> Literal["recorded", "live"]:
    """Probe the URL with yt-dlp metadata-only (no download)."""
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return "live" if info.get("is_live") else "recorded"
```
- If `mode != "auto"`, skip detection and trust the user.
- If detection fails (network, unsupported site), raise `ModeDetectionError`.

### Phase D — Logging setup (~20 min)
Create `src/logging_setup.py` with `configure_logging(log_file, level=INFO)`:
- Root logger to file + stderr.
- Format: `%(asctime)s [%(levelname)s] %(name)s: %(message)s`.
- Per-run log file is `logs/<run_id>.log`.

Call from `run()` immediately after creating `RunPaths`.

### Phase E — Dispatch (~45 min)
```python
def run(url, mode="auto", run_id=None, config_path=Path("config.yaml")):
    config = load_config(config_path)
    run_id = run_id or str(uuid.uuid4())
    paths = create_run_paths(run_id)
    configure_logging(paths.log_file)
    started = datetime.utcnow()
    if mode == "auto":
        mode = detect_mode(url)
    if mode == "recorded":
        ingest_result = download_youtube_video(
            url=url, run_id=run_id,
            output_root=paths.raw.parent,
            max_height=config["ingest"]["max_height"],
        )
    elif mode == "live":
        ingest_result = start_live_capture(
            url=url, run_id=run_id,
            output_root=paths.raw.parent,
            chunk_seconds=config["ingest"]["chunk_seconds"],
        )
    else:
        raise ValueError(mode)
    manifest_path = _write_manifest(paths, run_id, mode, url, started, ingest_result)
    return RunContext(run_id=run_id, mode=mode, url=url, started_at=started,
                      paths=paths, config=config,
                      ingest_result=ingest_result, manifest_path=manifest_path)
```

### Phase F — Manifest writer (~30 min)
`manifest.json` lives at `paths.intermediate / "manifest.json"` and is the single source of truth for downstream modules. Schema:
```json
{
  "run_id": "...",
  "mode": "recorded",
  "url": "...",
  "started_at": "2026-05-04T12:34:56Z",
  "ingest": {
    "video_path": "data/raw/<id>/video.mp4",
    "title": "...",
    "duration_sec": 312.5,
    "has_subtitles": true,
    "available_subtitle_langs": ["en"]
  },
  "stages": {
    "ingest":   {"status": "complete", "completed_at": "..."},
    "audio":    {"status": "pending"},
    "stt":      {"status": "pending"},
    "frames":   {"status": "pending"},
    "ocr":      {"status": "pending"},
    "fusion":   {"status": "pending"},
    "summary":  {"status": "pending"}
  }
}
```
For `mode == "live"`, the `ingest` block instead lists `chunk_dir` and is updated by the live chunker callback.

Provide helper `update_manifest(manifest_path, **fields)` for downstream modules to atomically update stage status (write to temp file, then `os.replace`).

### Phase G — CLI (~20 min)
Standard `argparse` wrapper. On success print `manifest_path`. On failure, log full traceback to `paths.log_file`, print one-line error to stderr, exit code 1.

### Phase H — Tests (~1 hr)

1. **`create_run_paths`** — call with a UUID; assert all directories exist; assert idempotent (call twice, no error).
2. **`detect_mode`** — patch `yt_dlp.YoutubeDL.extract_info` to return `{"is_live": True}` and `{"is_live": False}`; assert outputs.
3. **Dispatch with stubs** — monkeypatch `download_youtube_video` and `start_live_capture` to return mock objects; run `run(...)` with each mode; assert correct branch was called and manifest contains expected fields.
4. **Manifest atomicity** — write a manifest, then call `update_manifest` with new fields; assert old fields preserved and new ones merged.
5. **Bad mode value** — `run(mode="weird")` raises `ValueError`.

---

## 5. File Layout After Plan 1.3
```
src/
  orchestrator.py
  config.py
  logging_setup.py
  paths.py             # RunPaths, create_run_paths
  manifest.py          # write_manifest, update_manifest
config.yaml
.env.example
logs/                  (created at runtime)
data/
  raw/
  audio/
  frames/
  intermediate/
  output/
tests/
  test_orchestrator.py
  test_paths.py
  test_manifest.py
```

---

## 6. Acceptance Criteria

- [ ] `python -m src.orchestrator --url <recorded_url>` creates the full folder skeleton and a valid `manifest.json` with `mode == "recorded"`.
- [ ] Same command with a live URL detects it and routes to the live chunker.
- [ ] `--mode recorded` with a live URL still tries to download (and surfaces the eventual error from Plan 1.1) — i.e., the override is respected.
- [ ] `manifest.json` validates against the schema in §4 Phase F.
- [ ] All unit tests pass without network.
- [ ] Two consecutive runs with different URLs produce two distinct, non-colliding `data/.../<run_id>/` trees.

---

## 7. Edge Cases & Pitfalls

1. **`run_id` collision** — UUID4 collisions are vanishingly rare, but if `data/raw/<run_id>` already exists, raise `RunIdConflictError` rather than overwriting.
2. **`config.yaml` missing** — fall back to a hardcoded default dict; warn in log.
3. **Partial failures** — if `download_youtube_video` raises after creating the raw folder, leave the folder in place but write `manifest.json` with `stages.ingest.status = "failed"` and the exception message. Downstream modules check status before proceeding.
4. **Concurrent runs** — every run uses its own UUID directory; there is no shared mutable state. Safe by construction.
5. **Manifest writes from background threads (live mode)** — funnel all manifest updates through `update_manifest()`, which uses `os.replace()` for atomic swap. Add a `threading.Lock` around the read-modify-write inside the helper.
6. **Path separators** — never join paths with `+` or string formatting; always `Path / "subdir"`.

---

## 8. Out of Scope

- Any actual processing past ingestion (Plans 2.x onwards).
- Resume/retry logic across runs (consider in Plan 5.1).
- Web UI.

---

## 9. Definition of Done

A new contributor reading only this file plus Plans 1.1 and 1.2 can run `python -m src.orchestrator --url <any_youtube_or_live_url>` and see a fully populated `data/<.../>` tree with a valid `manifest.json`, plus a structured log file.
