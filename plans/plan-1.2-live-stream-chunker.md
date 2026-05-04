# Plan 1.2 — Live Stream Chunker

> **Self-contained scope.** This plan defines a Python module that connects to a live video stream URL and saves it to disk as a sequence of fixed-duration chunk files, suitable for downstream near-real-time processing. No other module is required to implement this plan.

---

## 1. Objective

Build `src/ingest/live_chunker.py` that:

1. Connects to a live HLS/DASH/RTMP stream **or** a YouTube Live URL.
2. Records the stream into rolling **N-second chunks** (default 10 s).
3. Names chunks `chunk_000001.ts` (zero-padded, ascending) so file order = time order.
4. Emits a chunk-ready event (callback or queue) as soon as each file is fully written.
5. Stops cleanly on `SIGINT` or after a configurable time limit.

The chunker runs as either a foreground CLI or a background subprocess controlled by Plan 1.3.

---

## 2. Contract (Inputs & Outputs)

### Function signature
```python
def start_live_capture(
    url: str,
    run_id: str,
    chunk_seconds: int = 10,
    output_root: Path = Path("data/raw"),
    max_duration_sec: int | None = None,   # None = run until stopped
    on_chunk_ready: Callable[[Path, int], None] | None = None,
    container: str = "ts",                  # "ts" or "mp4"
) -> LiveCaptureHandle: ...
```

### `LiveCaptureHandle` (returned object)
```python
class LiveCaptureHandle:
    run_id: str
    output_dir: Path
    process: subprocess.Popen     # the ffmpeg subprocess
    watcher_thread: threading.Thread
    def stop(self, timeout: float = 5.0) -> None: ...   # graceful shutdown
    def wait(self) -> int: ...                          # blocks until done, returns exit code
    @property
    def is_running(self) -> bool: ...
```

### `on_chunk_ready` callback
Called once per fully-written chunk. Signature:
```python
on_chunk_ready(chunk_path: Path, chunk_index: int) -> None
```
Must be thread-safe (the watcher thread invokes it).

### CLI
```
python -m src.ingest.live_chunker --url "<URL>" [--run-id RUN_ID] [--chunk-seconds 10] [--max-duration 600]
```
Prints one JSON line per chunk to stdout: `{"chunk_index": 1, "path": "...", "size_bytes": ..., "wall_time": "..."}`.

---

## 3. Dependencies

| Package / Tool | Purpose |
|---|---|
| `ffmpeg` system binary | Connect to stream, segment output |
| `streamlink` | Resolve YouTube Live / Twitch URLs to a playable HLS URL |
| `yt-dlp` (already installed via Plan 1.1) | Fallback live-URL resolver |

Add to `requirements.txt`:
```
streamlink>=6.7.0
```

---

## 4. Why ffmpeg `-f segment` (not OpenCV)

OpenCV reads frame-by-frame and would require manual encoding to produce a playable file. ffmpeg's `segment` muxer is purpose-built: it copies the network stream straight to disk, splits at exact wall-clock boundaries, and handles the codec passthrough. Use ffmpeg.

The exact command template:
```
ffmpeg -hide_banner -loglevel warning \
       -i <RESOLVED_URL> \
       -c copy \
       -f segment \
       -segment_time <chunk_seconds> \
       -reset_timestamps 1 \
       -segment_format <container> \
       <output_dir>/chunk_%06d.<container>
```

`-c copy` is critical: no re-encode means low CPU and zero quality loss. Use container `ts` (MPEG-TS) by default — it's resilient to mid-file truncation if the process dies.

---

## 5. Phased Implementation

### Phase A — URL resolver (~45 min)
Many live URLs (YouTube Live, Twitch) need translation to a direct HLS/DASH manifest before ffmpeg can read them.

```python
def resolve_stream_url(url: str) -> str:
    # 1) Try streamlink first (best for YT Live / Twitch)
    try:
        streams = streamlink.streams(url)
        if streams:
            return streams.get("best").url
    except Exception:
        pass
    # 2) Fallback: yt-dlp manifest URL
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info.get("is_live"):
            raise NotALiveStreamError(url)
        return info["url"]   # direct manifest
    raise UnresolvableStreamError(url)
```

### Phase B — Output dir + ffmpeg launch (~45 min)
1. `output_dir = output_root / run_id` (mkdir parents=True, exist_ok=True).
2. Build the ffmpeg arg list (a Python list, not a shell string — prevents injection on Windows).
3. Launch via `subprocess.Popen(..., stdout=PIPE, stderr=PIPE, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)` on Windows so we can send `CTRL_BREAK_EVENT` for clean shutdown.
4. If `max_duration_sec` is set, append `-t <max_duration_sec>` to args.

### Phase C — Chunk-completion watcher (~1 hr)
ffmpeg writes a chunk file *while* the next one is being recorded, but it does not flush until the next segment opens. Detection strategy:

**Watch the directory; a chunk is "complete" the moment the next index appears.**

```python
def _watch_chunks(output_dir, callback, stop_event):
    seen = set()
    pattern = re.compile(r"chunk_(\d{6})\.\w+$")
    while not stop_event.is_set():
        time.sleep(0.5)
        files = sorted(output_dir.glob("chunk_*.*"))
        if len(files) < 2:
            continue
        # All files except the last (currently being written) are complete.
        for f in files[:-1]:
            if f in seen:
                continue
            seen.add(f)
            idx = int(pattern.search(f.name).group(1))
            try:
                callback(f, idx)
            except Exception:
                logger.exception("chunk callback failed")
    # On stop, mark the last file complete too if process exited.
    ...
```

Run this in a daemon `threading.Thread`.

### Phase D — Graceful shutdown (~30 min)
`LiveCaptureHandle.stop()` must:
1. Set the `stop_event` for the watcher thread.
2. Send `CTRL_BREAK_EVENT` (Windows) or `SIGINT` (Unix) to the ffmpeg process group — this lets ffmpeg finalize the current chunk before exiting. **Do not** use `process.kill()`; it leaves the trailing chunk corrupt.
3. `process.wait(timeout=timeout)`. If timeout exceeded, then `process.kill()`.
4. Join the watcher thread (timeout=2s).
5. After process exit, if a final chunk file exists and was not yet emitted, call the callback for it.

Register a `signal.signal(SIGINT, ...)` handler in the CLI entrypoint that calls `handle.stop()`.

### Phase E — CLI wrapper (~30 min)
```python
def main():
    args = parse_args()
    run_id = args.run_id or str(uuid.uuid4())
    def emit(path, idx):
        print(json.dumps({
            "chunk_index": idx,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "wall_time": datetime.utcnow().isoformat() + "Z",
        }), flush=True)
    handle = start_live_capture(args.url, run_id,
                                chunk_seconds=args.chunk_seconds,
                                max_duration_sec=args.max_duration,
                                on_chunk_ready=emit)
    try:
        handle.wait()
    except KeyboardInterrupt:
        handle.stop()
```

### Phase F — Tests (~1 hr 30 min)

1. **URL resolver unit test** — patch `streamlink.streams` and `yt_dlp.YoutubeDL` to return controlled values; verify both branches and the not-live error path.
2. **Watcher logic test** — pre-populate a tmp dir with `chunk_000001.ts`, `chunk_000002.ts`; run the watcher for 2 s; assert callback fires for chunk 1 only.
3. **End-to-end smoke test (manual)** — point at a public test HLS feed, e.g., `https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8`. Run for 30 s with `chunk_seconds=5`. Assert ≥ 5 chunk files exist and each is a valid playable `.ts`.
4. **Shutdown test** — start a capture; call `.stop()` after 8 s; assert process exit code is 0 (or 255 on SIGINT, which is also acceptable for ffmpeg) and the last chunk is non-zero size and plays in VLC.

Mark the live test with `@pytest.mark.live`.

---

## 6. File Layout After Plan 1.2
```
src/ingest/
  live_chunker.py
  stream_resolver.py     # holds resolve_stream_url()
  errors.py              # extend with LiveStream-* errors
tests/ingest/
  test_live_chunker.py
  test_stream_resolver.py
```

---

## 7. Acceptance Criteria

- [ ] CLI run against a public test HLS URL produces ≥ 1 chunk file every `chunk_seconds`.
- [ ] Each chunk is independently playable (open in VLC, plays without errors).
- [ ] `Ctrl+C` produces a clean shutdown — no orphaned ffmpeg process visible in Task Manager.
- [ ] `on_chunk_ready` is called exactly once per completed chunk, in ascending order.
- [ ] Unit tests pass without network (`pytest -m "not live"`).
- [ ] Manual live test passes against the canary URL in §5 Phase F.

---

## 8. Edge Cases & Pitfalls

1. **Stream stalls mid-recording** — ffmpeg may sit silent. Add `-rw_timeout 30000000` (30 s in microseconds) to the ffmpeg args to force an error if no data arrives.
2. **First chunk callback fires only after second chunk starts** — by design (we can't know chunk 1 is done until chunk 2 opens). Document this. Min latency for "chunk available" ≈ `chunk_seconds`.
3. **Container choice** — `ts` is fault-tolerant; `mp4` is not (mp4 needs the moov atom written at the end). Default to `ts` and only allow `mp4` when `max_duration_sec` is set.
4. **YouTube Live with DRM** — streamlink will fail; surface error verbatim.
5. **Two captures with the same run_id** — refuse to start if `output_dir` already contains `chunk_*` files; require `--force`.
6. **Disk filling up** — log a warning if free space drops below 1 GB; don't auto-delete.
7. **Windows process groups** — `CREATE_NEW_PROCESS_GROUP` is required so `CTRL_BREAK_EVENT` only kills ffmpeg, not the parent Python process.

---

## 9. Out of Scope

- Decoding/transcoding chunks (downstream plans handle that).
- Adaptive bitrate switching.
- Re-stitching chunks into a single file (use `ffmpeg -f concat` later if needed).
- UI.

---

## 10. Definition of Done

A developer can run the CLI command from §2 against `https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8`, see chunk-ready JSON lines stream to stdout every ~10 s, hit `Ctrl+C`, and find a complete sequence of playable `.ts` files in `data/raw/<run_id>/` — using only this plan file.
