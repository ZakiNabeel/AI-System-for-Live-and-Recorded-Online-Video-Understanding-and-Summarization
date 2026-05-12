# Plan 6.2 — Live Stream UI Integration (Streamlit)

> **Self-contained scope.** Wire live-stream mode into the existing Streamlit `app.py` so users can start, monitor, and stop a live analysis session from the browser — without needing a terminal. The live pipeline backend (`src/pipeline_live.py`) already works; this plan adds only the UI layer. Reading only this file is sufficient to implement.

---

## 1. Objective

Extend `app.py` so that when the user selects **Mode = live**:

1. Clicking **Run Pipeline** launches `pipeline_live` in a background thread.
2. The UI polls the run's manifest and rolling-summary state file and refreshes every few seconds — displaying:
   - Chunk count processed.
   - Rolling transcript excerpt (last N sentences).
   - Rolling summary (live-updating text area).
   - Frames extracted so far (thumbnail strip).
3. A **Stop** button gracefully terminates the live capture.
4. When the session ends (user stops or stream ends), the final summary is rendered in the same tabs as recorded mode.

---

## 2. Current State

`app.py` line 73–77 currently shows a warning and does nothing for live mode:
```python
elif mode == "live":
    st.warning(
        "Live mode is not supported in the Streamlit UI streaming path. "
        "Use `python -m src.pipeline_live --url <URL>` from the terminal."
    )
```

`src/pipeline_live.py` already:
- Accepts a URL and produces rolling outputs in `data/intermediate/<run_id>/` and `data/output/<run_id>/`.
- Writes a `manifest.json` (stage tracking) and `rolling_summary.json` (latest rolling summary text).
- Produces `fused.json` updated incrementally.

---

## 3. Architecture

### 3.1 Thread-based live runner

Live mode cannot use the `run_pipeline_streaming` generator (which yields per stage) because live analysis is indefinite. Instead, run in a thread:

```python
import threading
from src.pipeline_live import run_pipeline_live, LiveHandle

def _live_thread(url: str, run_id: str, opts: dict, handle_box: list, error_box: list) -> None:
    try:
        handle = run_pipeline_live(
            url=url,
            run_id=run_id,
            enable_captions=opts.get("captions", False),
            domain=opts.get("domain") or None,
        )
        handle_box.append(handle)
    except Exception as exc:
        error_box.append(str(exc))
```

`LiveHandle` is a dataclass returned by `run_pipeline_live` that exposes a `.stop()` method and a `.run_id` field. If `LiveHandle` does not yet exist in `pipeline_live.py`, it must be added (see §4).

### 3.2 Session state keys

```python
# Keys used in st.session_state for live mode
LIVE_THREAD_KEY   = "live_thread"
LIVE_HANDLE_KEY   = "live_handle"
LIVE_RUN_ID_KEY   = "live_run_id"
LIVE_ERROR_KEY    = "live_error"
LIVE_RUNNING_KEY  = "live_running"
```

### 3.3 Polling loop

Streamlit has no native background-update mechanism. Use `st.rerun()` inside a loop guarded by `time.sleep`:

```python
if st.session_state.get(LIVE_RUNNING_KEY):
    # Display current state
    _render_live_status(run_id)
    time.sleep(3)          # wait 3 s
    st.rerun()             # triggers full script re-run → polls again
```

This creates a 3-second polling loop that auto-refreshes the UI while the live thread runs.

---

## 4. Required Backend Additions

### 4.1 `LiveHandle` in `src/pipeline_live.py`

If not present, add:

```python
@dataclass
class LiveHandle:
    run_id: str
    output_dir: Path
    intermediate_dir: Path
    _stop_event: threading.Event = field(default_factory=threading.Event)

    def stop(self) -> None:
        """Signal the live capture loop to stop after the current chunk."""
        self._stop_event.set()

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()
```

Modify `run_pipeline_live()` to:
1. Create the `LiveHandle` before starting capture.
2. Pass `stop_event` into the chunker so it checks `stop_event.is_set()` between chunks.
3. Return the `LiveHandle` immediately (the capture runs on a background thread internally).

### 4.2 Rolling summary file

`pipeline_live.py` should write/update `data/output/<run_id>/rolling_summary.json` after each chunk:

```json
{
  "chunk_count": 5,
  "last_updated": "2026-05-12T14:32:01Z",
  "summary": "The speaker is discussing Python decorators...",
  "transcript_tail": "...and that is why we use @property. Next we'll look at...",
  "frames_extracted": 12
}
```

If this file does not already exist in `pipeline_live.py`, add writes to it.

---

## 5. UI Implementation

Replace the current live-mode warning block in `app.py` with the full live UI logic. Here is the complete structure:

```python
# ---- LIVE MODE ----
elif mode == "live":
    _render_live_mode(url, run_id_input, enable_captions, enable_qa, domain)
```

```python
def _render_live_mode(url, run_id_input, enable_captions, enable_qa, domain):
    opts = {"captions": enable_captions, "qa": enable_qa, "domain": domain}

    # ── Controls ──────────────────────────────────────────────────────────
    col_start, col_stop = st.columns([1, 1])
    start_btn = col_start.button("▶  Start Live Analysis", type="primary", use_container_width=True,
                                  disabled=st.session_state.get(LIVE_RUNNING_KEY, False))
    stop_btn  = col_stop.button("⏹  Stop", use_container_width=True,
                                 disabled=not st.session_state.get(LIVE_RUNNING_KEY, False))

    # ── Start ──────────────────────────────────────────────────────────────
    if start_btn:
        if not url.strip():
            st.error("Please enter a live stream URL first.")
            return

        import uuid
        run_id = run_id_input.strip() or str(uuid.uuid4())[:8]
        handle_box, error_box = [], []

        t = threading.Thread(
            target=_live_thread, args=(url, run_id, opts, handle_box, error_box), daemon=True
        )
        t.start()
        # Wait briefly for handle to be set (or error)
        import time; time.sleep(1.5)

        if error_box:
            st.error(f"Failed to start live capture: {error_box[0]}")
            return
        if handle_box:
            st.session_state[LIVE_HANDLE_KEY]  = handle_box[0]
        st.session_state[LIVE_THREAD_KEY]  = t
        st.session_state[LIVE_RUN_ID_KEY]  = run_id
        st.session_state[LIVE_RUNNING_KEY] = True
        st.rerun()

    # ── Stop ───────────────────────────────────────────────────────────────
    if stop_btn:
        handle = st.session_state.get(LIVE_HANDLE_KEY)
        if handle:
            handle.stop()
        st.session_state[LIVE_RUNNING_KEY] = False
        st.info("Stop signal sent. Finalizing current chunk...")
        st.rerun()

    # ── Live status display ────────────────────────────────────────────────
    run_id = st.session_state.get(LIVE_RUN_ID_KEY)
    if run_id:
        _render_live_status(run_id)

    # ── Polling loop ──────────────────────────────────────────────────────
    if st.session_state.get(LIVE_RUNNING_KEY):
        import time
        time.sleep(3)
        st.rerun()
```

```python
def _render_live_status(run_id: str) -> None:
    """Render current state of a live run from rolling_summary.json."""
    import json, time
    from pathlib import Path

    rolling_path = Path("data") / "output" / run_id / "rolling_summary.json"
    if not rolling_path.exists():
        st.info("Waiting for first chunk to process...")
        return

    try:
        state = json.loads(rolling_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not read live state file.")
        return

    chunk_count    = state.get("chunk_count", 0)
    last_updated   = state.get("last_updated", "")
    summary        = state.get("summary", "")
    transcript_tail = state.get("transcript_tail", "")
    frames_count   = state.get("frames_extracted", 0)

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Chunks Processed", chunk_count)
    col2.metric("Frames Extracted", frames_count)
    col3.metric("Last Updated", last_updated[11:19] if len(last_updated) > 18 else last_updated)

    # Rolling summary
    if summary:
        st.subheader("Rolling Summary")
        st.info(summary)

    # Transcript tail
    if transcript_tail:
        with st.expander("Recent Transcript"):
            st.text(transcript_tail)

    # Frame thumbnails (last 6 frames)
    frames_dir = Path("data") / "frames" / run_id
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.png"))[-6:]
        if frame_files:
            st.subheader("Recent Frames")
            cols = st.columns(min(len(frame_files), 6))
            for col, fp in zip(cols, frame_files):
                col.image(str(fp), use_column_width=True)
```

---

## 6. Final Summary Display

When the user clicks Stop and the thread completes, display the final outputs in the same tab structure as recorded mode. Add after the live status display:

```python
# Show final output once stopped
if not st.session_state.get(LIVE_RUNNING_KEY) and run_id:
    output_dir = Path("data") / "output" / run_id
    summary_md = output_dir / "summary.md"
    if summary_md.exists():
        st.divider()
        st.subheader("Final Analysis")
        tab_summary, tab_report = st.tabs(["Summary", "Report JSON"])
        with tab_summary:
            st.markdown(summary_md.read_text(encoding="utf-8"))
        with tab_report:
            report_json = output_dir / "report.json"
            if report_json.exists():
                st.json(json.loads(report_json.read_text(encoding="utf-8")))
```

---

## 7. Edge Cases

| Scenario | Handling |
|----------|----------|
| URL is a members-only/private stream | `pipeline_live` raises; error shown in `error_box`; UI shows error message |
| `rolling_summary.json` not yet written | UI shows "Waiting for first chunk..." |
| Thread crashes mid-run | Thread death detected via `t.is_alive()` check; set `LIVE_RUNNING_KEY=False` |
| User refreshes browser mid-run | `session_state` lost; run_id input field lets user re-attach by entering the same run_id |
| ffmpeg not installed | `pipeline_live` raises `FFmpegMissingError`; shown as user-friendly error |

---

## 8. Files Changed

| File | Change |
|------|--------|
| `app.py` | Replace live-mode warning with full `_render_live_mode()` + `_render_live_status()` functions |
| `src/pipeline_live.py` | Add `LiveHandle` dataclass; add rolling_summary.json writes; pass stop_event to chunker |

No new files required (all logic goes into the two existing files).

---

## 9. Phased Execution

| Phase | Task | Effort |
|-------|------|--------|
| A | Add `LiveHandle` to `pipeline_live.py` and wire stop_event | 45 min |
| B | Add `rolling_summary.json` writes to `pipeline_live.py` | 30 min |
| C | Write `_render_live_mode()` function in `app.py` | 1 hr |
| D | Write `_render_live_status()` function in `app.py` | 30 min |
| E | Write final summary display block | 20 min |
| F | Manual test with a real YouTube live stream URL | 30 min |

**Total estimated effort: ~3.5 hours**

---

## 10. Acceptance Criteria

- [ ] Selecting Mode = live and clicking Start shows live metrics without a terminal.
- [ ] Rolling summary updates every ~3 seconds while the stream is active.
- [ ] Stop button sends signal; capture stops gracefully (no zombie ffmpeg process).
- [ ] Final summary renders in tabs after stopping.
- [ ] Selecting Mode = recorded still works exactly as before (no regression).
- [ ] Error (bad URL, ffmpeg missing) shown as `st.error()` message — no crash or blank screen.

---

## 11. Definition of Done

A user opens `http://localhost:8501`, selects Mode = live, pastes a YouTube Live URL, clicks Start, and sees live metrics and rolling summary updating every few seconds — all within the Streamlit browser UI, no terminal needed.
