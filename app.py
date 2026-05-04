"""Streamlit UI for the DIP Video Understanding pipeline."""

from __future__ import annotations

import threading
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="DIP Video Understanding", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _pipeline_thread(url: str, mode: str, run_id: str | None, opts: dict, result_box: list) -> None:
    """Run pipeline in a background thread and store result/exception in result_box."""
    try:
        from src.pipeline import run_pipeline
        result = run_pipeline(
            url=url,
            mode=mode,
            run_id=run_id or None,
            enable_captions=opts.get("captions", False),
            enable_qa=opts.get("qa", False),
            domain=opts.get("domain") or None,
        )
        result_box.append(("ok", result))
    except Exception as exc:
        result_box.append(("err", str(exc)))


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["auto", "recorded", "live"], index=0)
    run_id_input = st.text_input("Run ID (optional)", value="", help="Leave blank for a new run.")
    enable_captions = st.checkbox("Enable image captions", value=False)
    enable_qa = st.checkbox("Generate Q&A pairs", value=False)
    domain = st.text_input("Domain hint", value="", placeholder="e.g. education")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("DIP Video Understanding & Summarization")
st.markdown("Enter a YouTube URL (recorded or live) and click **Run Pipeline**.")

url = st.text_input("YouTube URL or live-stream URL", placeholder="https://www.youtube.com/watch?v=...")

col_run, col_resume = st.columns([1, 1])
run_btn = col_run.button("▶  Run Pipeline", type="primary", use_container_width=True)
resume_btn = col_resume.button("↩  Resume (same run ID)", use_container_width=True)

# ---------------------------------------------------------------------------
# Streaming run (recorded mode only — live runs via separate live module)
# ---------------------------------------------------------------------------

if run_btn or resume_btn:
    if not url.strip():
        st.error("Please enter a URL first.")
    elif mode == "live":
        st.warning(
            "Live mode is not supported in the Streamlit UI streaming path. "
            "Use `python -m src.pipeline_live --url <URL>` from the terminal."
        )
    else:
        from src.pipeline import run_pipeline_streaming, STAGE_NAMES

        opts = {"captions": enable_captions, "qa": enable_qa, "domain": domain}
        rid = run_id_input.strip() or None

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        stage_table = st.empty()

        stage_status: dict[str, str] = {s: "pending" for s in STAGE_NAMES}
        error_placeholder = st.empty()

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
            for stage_name, fraction in gen:
                stage_status[stage_name] = "running"
                progress_bar.progress(min(fraction, 1.0))
                status_text.markdown(f"**Running stage:** `{stage_name}`")
                rows = " | ".join(
                    f"`{n}` {'✓' if s == 'complete' else ('⟳' if s == 'running' else '…')}"
                    for n, s in stage_status.items()
                )
                stage_table.markdown(rows)
                # After the stage completes the generator yields again with updated fraction
                stage_status[stage_name] = "complete"

            progress_bar.progress(1.0)
            status_text.markdown("**Pipeline complete!** ✅")

        except Exception as exc:
            error_placeholder.error(f"Pipeline failed: {exc}")
            st.stop()

        # ---- Display results ----
        from src.pipeline import _manifest_path_for
        import json

        if rid:
            mp = _manifest_path_for(rid)
            if mp.exists():
                data = json.loads(mp.read_text())
                run_id_used = data.get("run_id", rid)
            else:
                run_id_used = rid
        else:
            run_id_used = "unknown"

        output_dir = Path("data") / "output" / run_id_used
        summary_md = output_dir / "summary.md"
        summary_html = output_dir / "summary.html"
        report_json = output_dir / "report.json"
        perf_report = output_dir / "performance_report.json"

        st.divider()
        tab_summary, tab_report, tab_perf = st.tabs(["Summary", "Report JSON", "Performance"])

        with tab_summary:
            if summary_md.exists():
                st.markdown(_read_file(summary_md))
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
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "DIP Video Understanding System — run `streamlit run app.py` from the project root. "
    "API keys must be set in `.env` or environment variables."
)
