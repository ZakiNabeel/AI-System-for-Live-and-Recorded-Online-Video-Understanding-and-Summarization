# Demo Guide

This folder contains everything needed to reproduce and present the project demo.

## Pre-run Outputs

`recorded_demo/` contains the full pipeline output for the demo video documented in `recorded_demo/video_info.json`. You can review these files without running the pipeline.

## Quick Start

### Option A — View Pre-run Results (no API key needed)

Open `recorded_demo/summary.md` in any Markdown viewer, or open `recorded_demo/summary.html` in a browser.

### Option B — Reproduce the Recorded Demo

```bash
# From project root — requires API key in .env
bash demo/demo_commands.sh
```

### Option C — Run via Web UI

```bash
streamlit run app.py
```

1. Select **Mode = recorded**
2. Paste the URL from `recorded_demo/video_info.json`
3. Optionally select a domain (e.g. `tutorial-strategy`)
4. Click **Run Pipeline**
5. View Summary / Report JSON / Performance tabs

### Option D — Live Stream Demo

```bash
streamlit run app.py
```

1. Select **Mode = live**
2. Paste any active YouTube Live URL
3. Click **Start Live Analysis**
4. Observe rolling metrics and summary updating every ~3 seconds
5. Click **Stop** to finalize

Or use the CLI directly:

```bash
python -m src.pipeline_live --url "https://www.youtube.com/live/STREAM_ID"
```

## Requirements

- `pip install -r requirements.txt`
- `.env` file at project root with at least one of:
  - `ANTHROPIC_API_KEY=sk-ant-...`
  - `GEMINI_API_KEY=AI...`
- `ffmpeg` installed and on PATH (`ffmpeg -version` should work)

## Folder Contents

```
demo/
  README.md                    This file
  demo_script.md               Spoken walkthrough script for presenting the demo
  demo_commands.sh             Shell commands to reproduce both demos
  recorded_demo/
    video_info.json            Title, URL, duration of the demo video
    summary.md                 Full markdown summary output
    summary.html               HTML report with embedded frame images
    report.json                Full structured JSON output
    chapters.txt               Time-stamped chapter list
    report_card.md             Statistics (tokens, time, model)
    performance_report.json    Per-stage timing breakdown
    strategy.md                Tutorial strategy (if domain=tutorial-strategy)
    strategy.py                Python skeleton (if programming tutorial)
    frames/                    Representative keyframe screenshots
  live_demo/
    screenshot_01_start.png    Browser showing live UI before starting
    screenshot_02_running.png  Live metrics with chunk_count > 0
    screenshot_03_summary.png  Rolling summary visible
    screenshot_04_final.png    Final summary after stopping
    live_demo_notes.md         Notes: stream used, observed output, timestamps
```
