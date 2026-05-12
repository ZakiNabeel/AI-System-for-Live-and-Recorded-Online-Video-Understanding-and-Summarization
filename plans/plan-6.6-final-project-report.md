# Plan 6.6 — Final Project Report

> **Self-contained scope.** Write the Final Project Report required by the PRD. This is a formal academic/technical report documenting the system's motivation, design, implementation, results, and conclusions. It draws on the architecture document (Plan 6.4), performance evaluation (Plan 6.5), and demo outputs (Plan 6.3). No source code is written in this plan.

---

## 1. Objective

Produce `docs/final_report.md` (and optionally `docs/final_report.pdf` via Pandoc) — a complete project report that satisfies the PRD deliverable requirement for a "Final Project Report."

---

## 2. Report Outline

The report should be 15–25 pages when rendered (printed/PDF). Every section is mandatory.

```
Title Page
Abstract
Table of Contents
1. Introduction
2. Problem Statement and Motivation
3. Related Work
4. System Architecture
5. Module Design and Implementation
   5.1 Media Ingestion
   5.2 Audio Processing & Speech-to-Text
   5.3 Visual Processing (DIP Pipeline)
   5.4 Multimodal Fusion
   5.5 LLM-Based Summarization
   5.6 Domain-Specific Analysis
   5.7 Output Generation
   5.8 Master Pipeline & UI
6. Live Video Analysis
7. Recorded Video Analysis
8. Results and Evaluation
9. Discussion
10. Limitations and Future Work
11. Conclusion
References
Appendices
```

---

## 3. Section-by-Section Writing Guide

### Title Page
```
Design and Implementation of an AI System for
Live and Recorded Online Video Understanding and Summarization

[Your Name(s)]
[Course / Institution]
[Date: Month Year]
```

---

### Abstract (250 words)

The abstract must cover:
1. **What problem** is addressed: the difficulty of extracting information from video.
2. **What was built**: the system, its modes (live/recorded), and its outputs.
3. **Key technical contributions**: multimodal fusion, DIP-enhanced OCR, domain-specific extraction.
4. **Key result**: successful end-to-end demo; quantitative highlights (e.g., "processes a 10-minute video in under 4 minutes").

Template:
> "The exponential growth of online video content has created a critical need for automated systems capable of understanding and summarizing video at scale. In this project, we design and implement an end-to-end AI system that analyzes both live and recorded online videos, combining automatic speech recognition (ASR), digital image processing (DIP), optical character recognition (OCR), and large language model (LLM) reasoning to produce structured, time-stamped summaries. The system supports YouTube video download, live stream capture, speech-to-text with timestamp alignment, SSIM-based keyframe extraction, multi-profile DIP enhancement, multi-engine OCR, multimodal event fusion, and LLM-based global summarization with domain-specific output modes for education, trading, medical, law, and tutorial strategy. In evaluation, the system processed a [N]-minute video in [X] minutes on commodity hardware, producing a coherent summary with [N] key points, [N] chapters, and [N] detected events. Domain mode for tutorial-strategy additionally generated structured workflow files including an executable Python skeleton. The system is implemented in Python using yt-dlp, faster-whisper, OpenCV, EasyOCR, and the Anthropic Claude API, with a Streamlit web interface for interactive use."

---

### 1. Introduction

Cover:
- The explosion of online video content (YouTube: 500 hours uploaded/minute).
- The manual effort required to extract knowledge from video.
- Why automated systems are valuable: accessibility, research, knowledge management.
- Overview of this project's goals (what it does, what modes it supports).
- Organization of the rest of the report.

---

### 2. Problem Statement and Motivation

Cover:
- Formal problem statement: "Given a URL for a live or pre-recorded video, produce a structured analysis including a text transcript with timestamps, a coherent summary, key points, detected events, chapters, and optionally domain-specific structured data."
- Challenges: multimodal data fusion; ASR accuracy; variable visual quality; LLM context limits.
- Use cases: lecture summarization, trading signal extraction, medical education review, tutorial workflow extraction.

---

### 3. Related Work

Brief survey (1 page) covering:
- **ASR systems**: Whisper (Radford et al., 2022), faster-whisper, YouTube's auto-captions.
- **Video summarization**: prior NLP-based approaches, video captioning models (BLIP-2, etc.).
- **OCR in video**: text detection pipelines (EAST, CRAFT), Tesseract history.
- **LLM-based summarization**: GPT-4, Claude, prompt engineering for structured output.
- **Multimodal systems**: VideoChat, Vid2Seq, InternVideo.
- **What this project does differently**: full end-to-end pipeline, domain-specific extraction, live + recorded modes, DIP-enhanced OCR.

Note: cite as "[Author, Year]" format — no need for a formal bibliography tool, but list references at the end.

---

### 4. System Architecture

Include the system overview diagram from Plan 6.4 (`docs/diagrams/system_overview.png`).

Text covering:
- 10-stage pipeline design.
- Data directory layout per run.
- Manifest-based resumability.
- Live vs recorded mode routing.
- Configuration system (config.yaml + .env).

---

### 5. Module Design and Implementation

For each of the 8 subsections, write 1–2 pages:

#### 5.1 Media Ingestion
- YouTube downloader: yt-dlp wrapper, metadata extraction, subtitle detection.
- Live chunker: ffmpeg-based 10-second rolling chunks, streamlink URL resolution.
- RunContext: per-run isolated directory, UUID-based run_id.

#### 5.2 Audio Processing & Speech-to-Text
- Audio extraction: ffmpeg normalization to 16 kHz mono WAV.
- Backend selection logic: YouTube subtitles → faster-whisper → OpenAI API.
- faster-whisper: CTranslate2-based, word-level timestamps.
- Hallucination filtering: pattern-based heuristic for common ASR artifacts.
- Transcript alignment: gap/overlap fixing, sentence boundary splitting.
- Output: SRT and WebVTT subtitle files.

#### 5.3 Visual Processing (DIP Pipeline)
This is the most important section — detail the DIP work:
- SSIM-based keyframe extraction: explain SSIM, threshold, min gap.
- Enhancement profiles table:
  | Profile | Operations Applied |
  |---------|--------------------|
  | default | Gaussian denoise → CLAHE → sharpen |
  | screen | Bilateral denoise → CLAHE → adaptive threshold |
  | whiteboard | Grayscale → denoise → Otsu binarize → morph clean → deskew |
  | scene | CLAHE → color boost |
- Each DIP operation described: why it helps OCR.
- OCR engines: Tesseract vs EasyOCR comparison.
- Image captioning: Claude Vision / GPT-4V / Gemini Vision backends.
- Include before/after images from demo.

#### 5.4 Multimodal Fusion
- FusedEvent schema.
- Time-windowing: how speech and visual events at similar timestamps are merged.
- Event types: speech, visual, speech+visual.
- Token-aware chunking for LLM context.

#### 5.5 LLM-Based Summarization
- Two-pass architecture.
- Prompt design: chunk prompt and global prompt.
- Structured JSON output extraction and validation.
- Provider comparison.
- Rolling mode for live.

#### 5.6 Domain-Specific Analysis
- DomainProfile Protocol design.
- Five profiles with their extracted fields and extra outputs.
- Tutorial-strategy pseudocode generation algorithm.
- Example: show a `strategy.py` snippet from a demo run.

#### 5.7 Output Generation
- Five output formats: Markdown, HTML, JSON, chapters, report card.
- HTML report: embedded frame images, CSS styling.
- Performance report generation.

#### 5.8 Master Pipeline & UI
- Stage registry and dependency tracking.
- Resumable runs via manifest.json.
- Streamlit UI: both recorded (streaming progress) and live (polling) modes.
- CLI interface.

---

### 6. Live Video Analysis

Dedicated section (1 page):
- Architecture of live mode (diagram from Plan 6.4 `live_pipeline.png`).
- Chunk-based processing cycle.
- Rolling summary mechanism.
- UI polling implementation.
- Observed latency in demo.

---

### 7. Recorded Video Analysis

Dedicated section (1 page):
- Full pipeline walkthrough with the demo video.
- Show transcript excerpt (first 5 lines with timestamps).
- Show OCR output from a frame (before/after DIP enhancement).
- Show chapter list.
- Show summary excerpt.

---

### 8. Results and Evaluation

Import data from Plan 6.5:
- Speed metrics table (all 10 stages with actual times).
- Content quality metrics (n_words, n_key_points, n_chapters, n_events).
- OCR confidence stats.
- Token usage and estimated API cost.
- Robustness test results.
- Any WER/ROUGE scores.

Include at least 2 charts:
- Bar chart of stage timings.
- OCR confidence distribution (histogram or box plot).

---

### 9. Discussion

- What worked well: multimodal fusion improved summary quality vs speech-only; DIP enhancement significantly improved OCR on whiteboard/slide content; two-pass LLM approach produced more consistent JSON output.
- What was challenging: LLM occasionally ignored domain prompt addenda; live stream URL resolution is fragile; hallucination filtering too aggressive on some accented speech.
- Observations on domain-specific mode: what the tutorial-strategy profile extracted from a real video.

---

### 10. Limitations and Future Work

- **Speaker diarization**: schema supports it but not implemented; would require pyannote.audio.
- **Real-time live UI latency**: 10–30s lag inherent to chunk-based processing.
- **Language support**: English-first; OCR supports 80+ languages but LLM prompts are English.
- **Video content moderation**: no filtering of inappropriate content.
- **Strategy execution**: `strategy.py` is a skeleton; users must implement each function body.
- **Future work**: RAG-based interactive Q&A over the full transcript; streaming LLM responses; WebSocket-based live UI; auto-domain detection from title/thumbnail.

---

### 11. Conclusion

One page:
- Restate what was built and its capabilities.
- Highlight that both required modules (live and recorded) are fully functional.
- Highlight the DIP contributions (SSIM extraction, 4 enhancement profiles, multi-engine OCR).
- Highlight the domain-specific extension (5 profiles, tutorial-strategy pseudocode generation).
- State that all PRD deliverables are met.

---

### References

At minimum, cite:
1. Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. (Whisper)
2. Wang et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. (SSIM)
3. Smith, R. (2007). An Overview of the Tesseract OCR Engine.
4. yt-dlp project. https://github.com/yt-dlp/yt-dlp
5. Anthropic. (2024). Claude Technical Report.
6. Streamlit. https://streamlit.io
7. OpenCV documentation. https://docs.opencv.org

---

### Appendices

- **Appendix A**: Full configuration reference (config.yaml annotated).
- **Appendix B**: Sample `report.json` (abbreviated, from demo run).
- **Appendix C**: Sample `strategy.py` output (from tutorial-strategy demo).
- **Appendix D**: Installation and setup instructions.

---

## 4. Converting to PDF

```bash
# Option 1: Pandoc + LaTeX (best formatting)
pandoc docs/final_report.md \
  -o docs/final_report.pdf \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  --highlight-style=tango \
  -V geometry:margin=1in \
  -V fontsize=11pt

# Option 2: Pandoc + wkhtmltopdf (HTML → PDF, preserves markdown styling)
pandoc docs/final_report.md -o docs/final_report.html
wkhtmltopdf docs/final_report.html docs/final_report.pdf

# Option 3: VS Code → Print → Save as PDF (simplest)
# Open docs/final_report.md in VS Code
# Ctrl+Shift+P → "Markdown: Open Preview to the Side"
# Right-click preview → Print → Save as PDF
```

---

## 5. Phased Execution

| Phase | Task | Effort |
|-------|------|--------|
| A | Write Abstract + Introduction + Problem Statement | 1 hr |
| B | Write Related Work (3.x) | 45 min |
| C | Write Architecture section (4) using Plan 6.4 diagrams | 30 min |
| D | Write Module sections 5.1–5.4 | 2 hr |
| E | Write Module sections 5.5–5.8 | 1.5 hr |
| F | Write Live Analysis (6) and Recorded Analysis (7) | 1 hr |
| G | Write Results section (8) using Plan 6.5 data | 1 hr |
| H | Write Discussion, Limitations, Conclusion (9–11) | 1 hr |
| I | Write References and Appendices | 30 min |
| J | Review, proofread, convert to PDF | 1 hr |

**Total estimated effort: ~10 hours**

---

## 6. Acceptance Criteria

- [ ] `docs/final_report.md` exists and covers all 11 sections.
- [ ] Abstract is ≤ 300 words and summarizes the key contributions.
- [ ] Section 5.3 (DIP Pipeline) is detailed: describes SSIM, all 4 enhancement profiles, and each DIP operation.
- [ ] Section 8 contains actual numeric results (not placeholders) from a real pipeline run.
- [ ] At least one diagram/figure is embedded per major section.
- [ ] References section has at least 5 citations.
- [ ] PDF export is clean and readable.

---

## 7. Definition of Done

A project supervisor can open `docs/final_report.pdf`, read a complete 15–25 page academic report covering the design, implementation, and evaluation of the system, and verify all PRD deliverables are addressed — without looking at source code.
