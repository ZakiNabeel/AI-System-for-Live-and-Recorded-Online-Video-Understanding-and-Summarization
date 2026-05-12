# Demo Presentation Script

**Duration:** ~10 minutes total  
**Video used:** See `recorded_demo/video_info.json`

---

## Part 1: System Overview (1 minute)

> "This is the DIP Video Understanding and Summarization System. It accepts any YouTube URL — live or recorded — and produces a full multimodal analysis by combining speech transcription, digital image processing, OCR, and large language model reasoning."

**[Show the Streamlit homepage at localhost:8501]**

> "The interface has two modes: recorded mode for YouTube videos, and live mode for ongoing streams. Let me start with a recorded video demo."

---

## Part 2: Recorded Video Demo (5 minutes)

**[In sidebar: select Mode = recorded]**

> "I'm using a [FILL IN: VIDEO TITLE] video, which is about [FILL IN: TOPIC]. It's [FILL IN: DURATION] long."

**[Paste the URL from `recorded_demo/video_info.json` into the URL field]**

**[Optional: set Domain to `tutorial-strategy` or `education`]**

> "I'll set the domain to [tutorial-strategy / education] so the system extracts domain-specific structured output alongside the standard summary."

**[Click Run Pipeline — progress bar appears]**

> "The pipeline runs 10 stages automatically:"
> - "Stage 1 — Ingest: downloads the video using yt-dlp"
> - "Stage 2 — Audio: extracts 16kHz mono WAV using ffmpeg"
> - "Stage 3 — STT: transcribes speech with faster-whisper"
> - "Stage 4 — Align: filters hallucinations and aligns timestamps"
> - "Stage 5 — Frames: extracts keyframes using SSIM scene-change detection"
> - "Stage 6 — Enhance: applies DIP profiles — CLAHE, denoising, sharpening"
> - "Stage 7 — OCR: runs EasyOCR on enhanced frames to extract on-screen text"
> - "Stage 8 — Fuse: merges speech and visual events into a multimodal timeline"
> - "Stage 9 — Summarize: sends to Claude for chunked + global summarization"
> - "Stage 10 — Format: renders Markdown, HTML, JSON, chapters, and report card"

**[Wait for completion — pipeline completes in ~X minutes for a Y-minute video]**

**[Click Summary tab]**

> "Here's the full summary. Notice the time-stamped key points — each one links back to the exact second in the video where that information appeared."

> "The system automatically generated [N] chapters — these are YouTube-style chapter markers, derived entirely from the content without any manual input."

**[Click Report JSON tab]**

> "The structured JSON output contains the full summary, key points, detected events, chapters, and Q&A pairs — all machine-readable and ready for downstream use."

**[Open summary.html in browser if available]**

> "The HTML report embeds the actual keyframe screenshots at the timestamps where they appeared."

**[If domain extras present — show study_notes.md or strategy.py]**

> "Because I used [education / tutorial-strategy] domain mode, the system also produced:"
> - **education:** "a study notes file with learning objectives, definitions, and worked examples"
> - **tutorial-strategy:** "a structured workflow file — and a Python skeleton with one function per tutorial step, complete with timestamps and expected outcomes in the docstrings"

---

## Part 3: Live Stream Demo (3 minutes)

**[In sidebar: select Mode = live]**

> "Now for live mode. This runs the same pipeline in real time on a streaming video."

**[Enter a YouTube Live URL]**

> "I'm connecting to [FILL IN: STREAM NAME / URL]. The system captures 10-second audio and video chunks using ffmpeg, transcribes each one with faster-whisper, runs OCR on extracted frames, and updates a rolling summary every 30 seconds."

**[Click Start Live Analysis]**

> "The metrics panel updates every 3 seconds. You can see the chunk count incrementing as each 10-second segment is processed, and the rolling summary building up as more content comes in."

**[Wait ~1 minute for 3-4 chunks]**

> "Here's the rolling summary so far — it's coherent even though we're only partway through."

**[Click Stop]**

> "When we stop, the system finalizes with a full output pass — producing the same Markdown, HTML, and JSON deliverables as recorded mode."

---

## Part 4: Wrap-up (1 minute)

> "To summarize:"
> 
> "The system handles both live and recorded video, combining automatic speech recognition, SSIM-based keyframe extraction, multi-profile DIP enhancement, EasyOCR, and Claude for multimodal summarization."
> 
> "It supports five domain-specific output modes: education, trading, medical, law, and tutorial strategy — each producing structured deliverables tailored to that domain."
> 
> "All outputs are timestamped and available in Markdown, HTML, and JSON formats."
> 
> "The Streamlit UI makes it accessible without any command-line knowledge, while the CLI supports scripted and batch use."

---

## Notes for Presenter

- Have the Streamlit app running before the demo starts: `streamlit run app.py`
- Pre-load the recorded_demo outputs in a separate browser tab for the "already run" fallback
- If pipeline is slow, show `recorded_demo/summary.md` from the pre-run outputs while the live run finishes
- For live demo: pick a news stream or sports stream that has clear speech
- The `--domain tutorial-strategy` flag is the most impressive demo; use a Python tutorial video
