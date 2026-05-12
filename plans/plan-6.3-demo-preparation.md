# Plan 6.3 — Demo Preparation

> **Self-contained scope.** Prepare everything needed to run and record the required project demo: one live stream demo and one recorded YouTube video demo. Produce demo scripts, sample output artifacts, and a concise demo guide. This plan requires no new source code — it is about running the existing pipeline, capturing outputs, and packaging them as deliverables.

---

## 1. Objective

Produce the following demo artifacts:
1. `demo/recorded_demo/` — full pipeline output for a well-chosen YouTube video.
2. `demo/live_demo/` — screenshots or screen-recording instructions for live mode.
3. `demo/demo_script.md` — step-by-step spoken script for presenting both demos.
4. `demo/demo_commands.sh` — shell commands to reproduce both demos exactly.
5. `demo/README.md` — how to run the demos.

---

## 2. Video Selection Criteria

### 2.1 Recorded Demo Video Requirements
The chosen video must:
- Be publicly available on YouTube (no age-gate or member-only).
- Be **5–15 minutes long** — long enough to show multi-chapter output, short enough to process in demo time.
- Have **clear speech** (English preferred; subtitles available as fallback).
- Have **on-screen text** (slides, code, or captions) so OCR adds visible value.
- Ideally match a domain profile (education or tutorial-strategy) for a richer demo.

**Recommended candidate types:**
- A Python tutorial (shows `tutorial-strategy` domain mode well).
- A data-science or machine-learning lecture (shows `education` domain mode).
- A short TED-style talk with slides.

### 2.2 Live Stream Demo Requirements
- Any ongoing YouTube Live stream (news, a lecture stream, a gaming stream, etc.).
- Alternatively, a YouTube video with `?live=1` emulation is acceptable for demo purposes.
- Document the exact URL used and timestamp in `demo_script.md`.

---

## 3. Demo Output Structure

```
demo/
  README.md                     # How to reproduce
  demo_script.md                # Spoken walkthrough script
  demo_commands.sh              # Exact shell commands (copy-paste ready)

  recorded_demo/
    video_info.json             # Title, URL, duration, channel
    summary.md                  # Full markdown summary output
    summary.html                # HTML report (embedded images)
    report.json                 # Full structured JSON
    chapters.txt                # YouTube-style chapters
    report_card.md              # Stats (tokens used, time, model)
    performance_report.json     # Stage timings
    study_notes.md              # If education domain used
    strategy.md + strategy.py   # If tutorial-strategy domain used
    frames/                     # 6–10 representative keyframe screenshots
      frame_0001.png ...

  live_demo/
    screenshot_01_start.png     # Browser showing live mode UI after clicking Start
    screenshot_02_running.png   # Live metrics panel with chunk count > 0
    screenshot_03_summary.png   # Rolling summary visible in UI
    screenshot_04_final.png     # Final summary after stopping
    live_demo_notes.md          # What stream was used, when, observed output
```

---

## 4. Demo Script (`demo/demo_script.md`)

The script below is a template — fill in the blanks with the actual video title and observed outputs before the final demo.

---

### Part 1: Recorded Video Demo (~5 minutes)

**[Slide / screen: show the Streamlit UI homepage]**

> "This is the DIP Video Understanding and Summarization System. It accepts any YouTube URL and produces a full multimodal analysis — combining speech, on-screen text, and visual content."

**[Type the recorded video URL into the URL input box]**

> "I'm using a [VIDEO TITLE] video, which is about [TOPIC]. It's [DURATION] long."

**[Select Mode = recorded, check 'Enable image captions', select domain if applicable]**

> "I'll enable image captioning so the system also describes what it sees on-screen, not just what's said."

**[Click Run Pipeline — progress bar appears]**

> "The pipeline runs 10 stages. First it downloads the video, then extracts audio, transcribes speech using faster-whisper, aligns timestamps, extracts key frames using SSIM-based scene detection, applies DIP enhancements — CLAHE, denoising, sharpening — runs OCR on the enhanced frames, fuses all modalities, sends to Claude for summarization, and finally formats the outputs."

**[Wait for completion — tab to Summary]**

> "Here's the full summary. Notice the time-stamped key points and the automatically generated chapters."

**[Tab to Report JSON]**

> "The structured JSON output contains the summary, key points, detected events, chapters, and Q&A pairs — all machine-readable."

**[Open summary.html in browser if possible]**

> "And here's the HTML report with embedded frame screenshots at the relevant timestamps."

**[If domain used — show study_notes.md or strategy.py]**

> "Because I used [education/tutorial-strategy] domain mode, the system also generated [study notes with definitions and worked examples / a Python skeleton with one function per tutorial step]."

---

### Part 2: Live Stream Demo (~3 minutes)

**[Switch to Mode = live in sidebar]**

> "Now for live mode. This runs the same pipeline in real time on a streaming video."

**[Enter live stream URL]**

> "I'm connecting to [STREAM NAME/URL]. The system captures 10-second audio+video chunks, transcribes each one, extracts frames, and updates the rolling summary every 30 seconds."

**[Click Start Live Analysis]**

> "The metrics panel shows chunk count and frames extracted updating in real time. Here's the rolling summary building up as more audio comes in."

**[Click Stop after ~1 minute]**

> "When we stop, the system finalizes and renders the full output — same format as recorded mode."

---

### Part 3: Wrap-up (~1 minute)

> "To summarize: the system handles both live and recorded video, combines speech transcription with visual OCR and captioning, performs multimodal LLM summarization, and supports domain-specific output modes for education, trading, medical, law, and tutorial strategy extraction. All outputs are timestamped and available in Markdown, HTML, and JSON formats."

---

## 5. Demo Commands (`demo/demo_commands.sh`)

```bash
#!/usr/bin/env bash
# Demo reproduction script
# Run from project root with: bash demo/demo_commands.sh

# Prerequisites:
#   - pip install -r requirements.txt
#   - .env with ANTHROPIC_API_KEY (or GEMINI_API_KEY) set
#   - ffmpeg installed and on PATH

set -e

echo "=== DIP Video Understanding Demo ==="

# ── Recorded Demo ──────────────────────────────────────────────────────────
RECORDED_URL="<PASTE_YOUTUBE_URL_HERE>"
DOMAIN="tutorial-strategy"   # change to "education" or "" as needed

echo ""
echo "--- Running recorded pipeline ---"
echo "URL: $RECORDED_URL"
echo "Domain: $DOMAIN"

python -m src.pipeline \
  --url "$RECORDED_URL" \
  --mode recorded \
  --domain "$DOMAIN" \
  --enable-captions \
  --enable-qa

echo ""
echo "Recorded demo complete. Outputs in: data/output/<run_id>/"

# ── Live Demo ──────────────────────────────────────────────────────────────
echo ""
echo "--- Live demo ---"
echo "To run live mode via UI: streamlit run app.py"
echo "  Then select Mode = live and enter a live stream URL."
echo ""
echo "To run live mode via CLI (alternative):"
echo "  python -m src.pipeline_live --url <LIVE_STREAM_URL>"
```

---

## 6. Executing the Recorded Demo

### Step 1 — Choose and verify the video
```bash
# Test that the video is accessible
python -c "
import yt_dlp
url = '<YOUR_URL>'
with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
    info = ydl.extract_info(url, download=False)
    print(f'Title: {info[\"title\"]}')
    print(f'Duration: {info[\"duration\"]}s')
    print(f'Has subtitles: {bool(info.get(\"subtitles\"))}')
"
```

### Step 2 — Run the pipeline
```bash
python -m src.pipeline \
  --url "<YOUR_URL>" \
  --mode recorded \
  --domain tutorial-strategy \
  --enable-captions \
  --enable-qa
```

### Step 3 — Copy outputs to demo folder
```bash
RUN_ID=$(ls -t data/output/ | head -1)
mkdir -p demo/recorded_demo/frames
cp data/output/$RUN_ID/summary.md        demo/recorded_demo/
cp data/output/$RUN_ID/summary.html      demo/recorded_demo/
cp data/output/$RUN_ID/report.json       demo/recorded_demo/
cp data/output/$RUN_ID/chapters.txt      demo/recorded_demo/
cp data/output/$RUN_ID/report_card.md    demo/recorded_demo/
cp data/output/$RUN_ID/performance_report.json demo/recorded_demo/
# Domain extras
cp data/output/$RUN_ID/strategy.md       demo/recorded_demo/ 2>/dev/null || true
cp data/output/$RUN_ID/strategy.py       demo/recorded_demo/ 2>/dev/null || true
cp data/output/$RUN_ID/study_notes.md    demo/recorded_demo/ 2>/dev/null || true
# Copy representative frames
cp data/frames/$RUN_ID/frame_*.png       demo/recorded_demo/frames/ 2>/dev/null || true
# Save video metadata
python -c "
import json, sys
with open('data/intermediate/$RUN_ID/manifest.json') as f:
    m = json.load(f)
print(json.dumps({'run_id': '$RUN_ID', 'url': m.get('url',''), 'title': m.get('title','')}, indent=2))
" > demo/recorded_demo/video_info.json
```

---

## 7. Live Demo Screenshots

After completing Plan 6.2 (Live Stream UI), take screenshots at these exact moments:

| Screenshot | When to take | File name |
|------------|-------------|-----------|
| 01 — Start screen | After entering URL, before clicking Start | `screenshot_01_start.png` |
| 02 — Running | After first chunk processes (chunk_count ≥ 1) | `screenshot_02_running.png` |
| 03 — Rolling summary | When rolling summary text is non-empty | `screenshot_03_summary.png` |
| 04 — Final output | After clicking Stop and final summary appears | `screenshot_04_final.png` |

Save to `demo/live_demo/`.

Write `demo/live_demo/live_demo_notes.md`:
```markdown
# Live Demo Notes

- Stream URL used: <URL>
- Stream name: <TITLE>
- Date/time of demo: <DATETIME>
- Chunks processed: <N>
- Observed rolling summary excerpt: "<EXCERPT>"
- Total run time: <MINUTES> minutes
```

---

## 8. `demo/README.md` Content

```markdown
# Demo Guide

## Quick Start

### Recorded Demo (automated)
```bash
# From project root:
bash demo/demo_commands.sh
```
Pre-run outputs are in `demo/recorded_demo/`.

### Live Demo (UI)
1. `streamlit run app.py`
2. Select **Mode = live**
3. Enter a YouTube Live URL
4. Click **Start Live Analysis**
5. Observe rolling summary updating
6. Click **Stop** to finalize

### Recorded Demo (UI)
1. `streamlit run app.py`
2. Select **Mode = recorded**
3. Enter the URL from `demo/recorded_demo/video_info.json`
4. Click **Run Pipeline**
5. View Summary / Report JSON / Performance tabs

## Pre-run Outputs

The `recorded_demo/` folder contains pre-run outputs for the video documented in `video_info.json`.
These can be reviewed without running the pipeline.

## Requirements
- `pip install -r requirements.txt`
- `.env` file with `ANTHROPIC_API_KEY` (or `GEMINI_API_KEY`)
- `ffmpeg` installed and on `PATH`
```

---

## 9. Phased Execution

| Phase | Task | Effort |
|-------|------|--------|
| A | Select and test the recorded demo video | 20 min |
| B | Run full pipeline on chosen video | 30–60 min (pipeline runtime) |
| C | Copy outputs into `demo/recorded_demo/` | 15 min |
| D | Fill in `demo_script.md` with real video title and output excerpts | 30 min |
| E | Fill in `demo_commands.sh` with actual URL | 10 min |
| F | Run live demo and take screenshots (after Plan 6.2 is done) | 30 min |
| G | Write `live_demo_notes.md` | 10 min |

**Total estimated effort: ~2.5 hours** (excluding pipeline runtime)

---

## 10. Acceptance Criteria

- [ ] `demo/recorded_demo/summary.md` exists and contains at least 3 chapters and 5 key points.
- [ ] `demo/recorded_demo/report.json` is valid JSON with `full_summary`, `key_points`, `events`, `chapters`.
- [ ] `demo/recorded_demo/frames/` contains at least 5 frame screenshots.
- [ ] `demo/live_demo/` contains at least 2 screenshots showing live metrics updating.
- [ ] `demo/demo_script.md` contains the video title, URL, and filled-in output excerpts.
- [ ] `demo/demo_commands.sh` runs end-to-end without error on a clean environment (with API key set).
- [ ] `demo/README.md` contains clear instructions that a reviewer can follow.

---

## 11. Definition of Done

A project reviewer can open `demo/README.md`, follow the instructions, reproduce the recorded demo outputs in under 10 minutes, and see the pre-run results in `demo/recorded_demo/` matching the demo script — confirming both recorded and live modes work as claimed.
