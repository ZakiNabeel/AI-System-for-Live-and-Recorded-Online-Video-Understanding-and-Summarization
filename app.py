"""Streamlit UI for the DIP Video Understanding pipeline."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

# Fix Windows console encoding for Unicode output (EasyOCR progress bars etc.)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import streamlit as st

# Show INFO logs in the terminal where Streamlit is launched
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

st.set_page_config(page_title="DIP Video Understanding", layout="wide")

# ---------------------------------------------------------------------------
# Session-state keys for live mode
# ---------------------------------------------------------------------------
_LIVE_HANDLE_KEY  = "live_handle"
_LIVE_THREAD_KEY  = "live_thread"
_LIVE_RUN_ID_KEY  = "live_run_id"
_LIVE_RUNNING_KEY = "live_running"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _live_thread(url: str, run_id: str, opts: dict, handle_box: list, error_box: list) -> None:
    """Start the live pipeline in a background thread."""
    try:
        from src.pipeline_live import run_live_pipeline
        handle = run_live_pipeline(
            url=url,
            run_id=run_id,
            enable_captions=opts.get("captions", False),
            enable_qa=opts.get("qa", False),
            domain=opts.get("domain") or None,
        )
        handle_box.append(handle)
    except Exception as exc:
        error_box.append(str(exc))


def _render_live_status(run_id: str) -> None:
    """Display current live-pipeline metrics from rolling_summary.json."""
    rolling_path = Path("data") / "output" / run_id / "rolling_summary.json"
    if not rolling_path.exists():
        st.info("⏳ Waiting for first chunk to process…")
        return

    try:
        state = json.loads(rolling_path.read_text(encoding="utf-8"))
    except Exception:
        st.warning("Could not read live state file.")
        return

    chunk_count     = state.get("chunk_count", 0)
    last_updated    = state.get("last_updated", "")
    summary_text    = state.get("summary", "")
    transcript_tail = state.get("transcript_tail", "")
    frames_count    = state.get("frames_extracted", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks Processed", chunk_count)
    c2.metric("Frames Extracted", frames_count)
    c3.metric("Last Updated", last_updated[11:19] if len(last_updated) > 18 else last_updated)

    if summary_text:
        st.subheader("Rolling Summary")
        st.info(summary_text)

    if transcript_tail:
        with st.expander("Recent Transcript"):
            st.text(transcript_tail)

    frames_dir = Path("data") / "frames" / run_id
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("frame_*.png"))[-6:]
        if frame_files:
            st.subheader("Recent Frames")
            cols = st.columns(min(len(frame_files), 6))
            for col, fp in zip(cols, frame_files):
                col.image(str(fp), use_column_width=True)


def _render_live_ui(url: str, run_id_input: str, enable_captions: bool, enable_qa: bool, domain: str) -> None:
    """Full live-stream analysis UI with start/stop/polling."""
    opts = {"captions": enable_captions, "qa": enable_qa, "domain": domain}
    is_running = st.session_state.get(_LIVE_RUNNING_KEY, False)

    col_start, col_stop = st.columns([1, 1])
    start_btn = col_start.button(
        "▶  Start Live Analysis", type="primary",
        use_container_width=True, disabled=is_running,
    )
    stop_btn = col_stop.button(
        "⏹  Stop", use_container_width=True, disabled=not is_running,
    )

    # ── Start ──────────────────────────────────────────────────────────────
    if start_btn:
        if not url.strip():
            st.error("Please enter a live stream URL first.")
            return

        import uuid
        new_run_id = run_id_input.strip() or str(uuid.uuid4())[:8]
        handle_box: list = []
        error_box:  list = []

        t = threading.Thread(
            target=_live_thread,
            args=(url, new_run_id, opts, handle_box, error_box),
            daemon=True,
        )
        t.start()
        time.sleep(2.0)  # Brief wait for pipeline init or fast-fail

        if error_box:
            st.error(f"Failed to start live capture: {error_box[0]}")
            return

        if handle_box:
            st.session_state[_LIVE_HANDLE_KEY] = handle_box[0]
        st.session_state[_LIVE_THREAD_KEY]  = t
        st.session_state[_LIVE_RUN_ID_KEY]  = new_run_id
        st.session_state[_LIVE_RUNNING_KEY] = True
        st.rerun()

    # ── Stop ───────────────────────────────────────────────────────────────
    if stop_btn:
        handle = st.session_state.get(_LIVE_HANDLE_KEY)
        if handle is not None:
            try:
                handle.stop(timeout=10.0)
            except Exception:
                pass
        st.session_state[_LIVE_RUNNING_KEY] = False
        st.info("Stop signal sent. Finalizing …")
        st.rerun()

    # ── Status display ────────────────────────────────────────────────────
    run_id = st.session_state.get(_LIVE_RUN_ID_KEY)
    if run_id:
        _render_live_status(run_id)

    # ── Polling loop — auto-refresh every 3 s ─────────────────────────────
    if st.session_state.get(_LIVE_RUNNING_KEY):
        t = st.session_state.get(_LIVE_THREAD_KEY)
        if t is not None and not t.is_alive():
            st.session_state[_LIVE_RUNNING_KEY] = False
        else:
            time.sleep(3)
            st.rerun()

    # ── Final output once stopped ─────────────────────────────────────────
    if not st.session_state.get(_LIVE_RUNNING_KEY) and run_id:
        output_dir  = Path("data") / "output" / run_id
        summary_md  = output_dir / "summary.md"
        report_json = output_dir / "report.json"
        if summary_md.exists():
            st.divider()
            st.subheader("Final Analysis")
            tab_sum, tab_rep = st.tabs(["Summary", "Report JSON"])
            with tab_sum:
                st.markdown(_read_file(summary_md))
            with tab_rep:
                if report_json.exists():
                    st.json(json.loads(_read_file(report_json)))


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["auto", "recorded", "live"], index=0)
    run_id_input = st.text_input("Run ID (optional)", value="", help="Leave blank for a new run.")
    enable_captions = st.checkbox("Enable image captions", value=False)
    enable_qa = st.checkbox("Generate Q&A pairs", value=False)
    domain = st.text_input(
        "Domain hint",
        value="",
        placeholder="education | trading | medical | law | tutorial-strategy",
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("DIP Video Understanding & Summarization")
st.markdown("Enter a YouTube URL (recorded or live) and click **Run Pipeline**.")

url = st.text_input("YouTube URL or live-stream URL", placeholder="https://www.youtube.com/watch?v=...")

# ---------------------------------------------------------------------------
# Recorded / auto mode
# ---------------------------------------------------------------------------

if mode in ("recorded", "auto"):
    col_run, col_resume = st.columns([1, 1])
    run_btn    = col_run.button("▶  Run Pipeline", type="primary", use_container_width=True)
    resume_btn = col_resume.button("↩  Resume (same run ID)", use_container_width=True)

    if run_btn or resume_btn:
        if not url.strip():
            st.error("Please enter a URL first.")
        else:
            from src.pipeline import run_pipeline_streaming, STAGE_NAMES

            opts = {"captions": enable_captions, "qa": enable_qa, "domain": domain}
            rid = run_id_input.strip() or None

            progress_bar      = st.progress(0.0)
            status_text       = st.empty()
            stage_table       = st.empty()
            error_placeholder = st.empty()
            run_id_used       = rid or "unknown"

            stage_status: dict[str, str] = {s: "pending" for s in STAGE_NAMES}

            try:
                gen = run_pipeline_streaming(
                    url=url,
                    mode=mode,
                    run_id=rid,
                    resume=resume_btn,
                    enable_captions=enable_captions,
                    enable_qa=enable_qa,
                    domain=domain or None,
                )
                for stage_name, fraction, actual_run_id in gen:
                    run_id_used = actual_run_id
                    stage_status[stage_name] = "running"
                    progress_bar.progress(min(fraction, 1.0))
                    status_text.markdown(f"**Running stage:** `{stage_name}`")
                    rows = " | ".join(
                        f"`{n}` {'✓' if s == 'complete' else ('⟳' if s == 'running' else '…')}"
                        for n, s in stage_status.items()
                    )
                    stage_table.markdown(rows)
                    stage_status[stage_name] = "complete"

                progress_bar.progress(1.0)
                status_text.markdown("**Pipeline complete!** ✅")
                stage_table.markdown(" | ".join(f"`{n}` ✓" for n in STAGE_NAMES))

            except Exception as exc:
                # Unwrap RetryError to show the real cause
                cause = getattr(exc, '__cause__', None) or getattr(exc, '__context__', None)
                real_exc = cause if cause is not None else exc
                msg = str(real_exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    error_placeholder.error(
                        "**Gemini API quota exceeded (429).** You've hit the free tier daily limit. "
                        "Wait a few minutes and click **Resume (same run ID)** to continue from where it stopped."
                    )
                else:
                    error_placeholder.error(f"Pipeline failed: {real_exc}")
                st.stop()

            output_dir   = Path("data") / "output" / run_id_used
            summary_md   = output_dir / "summary.md"
            summary_html = output_dir / "summary.html"
            report_json  = output_dir / "report.json"
            perf_report  = output_dir / "performance_report.json"

            # Show log file in UI
            log_file = Path("data") / "logs" / f"{run_id_used}.log"
            if log_file.exists():
                with st.expander("📋 Backend Logs", expanded=False):
                    st.code(_read_file(log_file), language=None)

            st.divider()
            tab_summary, tab_report, tab_perf = st.tabs(["Summary", "Report JSON", "Performance"])

            with tab_summary:
                if summary_md.exists():
                    md_text = _read_file(summary_md)
                    # Strip markdown image tags — we render frames below with st.image
                    import re
                    md_no_images = re.sub(r'!\[.*?\]\(.*?\)', '', md_text)
                    st.markdown(md_no_images)

                    # Render frames from report.json visuals section
                    if report_json.exists():
                        try:
                            rj = json.loads(_read_file(report_json))
                            frames_dir = Path("data") / "frames" / run_id_used
                            enhanced_dir = frames_dir / "enhanced"
                            # collect all frame pngs, prefer enhanced
                            src_dir = enhanced_dir if enhanced_dir.exists() else frames_dir
                            frame_files = sorted(src_dir.glob("*.png"))[:8]
                            if frame_files:
                                st.subheader("Extracted Frames")
                                cols = st.columns(min(len(frame_files), 4))
                                for col, fp in zip(cols * 2, frame_files):
                                    col.image(str(fp), use_container_width=True, caption=fp.stem)
                        except Exception:
                            pass
                elif summary_html.exists():
                    st.components.v1.html(_read_file(summary_html), height=600, scrolling=True)
                else:
                    st.info("No summary output found yet.")

            with tab_report:
                if report_json.exists():
                    st.json(json.loads(_read_file(report_json)))
                else:
                    st.info("No report.json found yet.")

            with tab_perf:
                if perf_report.exists():
                    perf = json.loads(_read_file(perf_report))
                    st.metric("Total time (s)", f"{perf.get('total_sec', 0):.1f}")
                    timings = perf.get("stage_timings_sec", {})
                    if timings:
                        import pandas as pd
                        df = pd.DataFrame(
                            {"Stage": list(timings.keys()), "Seconds": list(timings.values())}
                        )
                        st.bar_chart(df.set_index("Stage"))
                else:
                    st.info("No performance report found yet.")

# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

elif mode == "live":
    _render_live_ui(url, run_id_input, enable_captions, enable_qa, domain)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "DIP Video Understanding System — run `streamlit run app.py` from the project root. "
    "API keys must be set in `.env` or environment variables."
)
