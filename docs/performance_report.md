# Performance Evaluation Report

**System:** DIP Video Understanding & Summarization  
**Date:** 2026-05-12  
**Evaluator:** [Fill in name]

---

## 1. Methodology

Performance metrics are collected automatically by the pipeline and stored as JSON files alongside the outputs. The `scripts/evaluate_performance.py` script reads these files and aggregates them into this report.

**Metrics sources:**
- **Speed/latency:** `data/output/<run_id>/performance_report.json` — written by `pipeline.py` at the end of each run; records per-stage wall-clock time.
- **Content quality:** `data/output/<run_id>/report.json` — contains structured summary with key points, events, chapters, and token usage.
- **Transcript stats:** `data/intermediate/<run_id>/transcript.aligned.json` — segment and word counts.
- **OCR confidence:** `data/intermediate/<run_id>/visual.json` — per-region confidence scores from EasyOCR.
- **Disk usage:** File sizes in `data/` subdirectories.

**To regenerate metrics from your own runs:**
```bash
python scripts/evaluate_performance.py
# Results in docs/performance_data/metrics.json and metrics_report.md
```

---

## 2. Test Setup

| Component | Value |
|-----------|-------|
| Platform | Windows 11 Pro |
| CPU | [Fill in: e.g. Intel Core i7-12700H] |
| RAM | [Fill in: e.g. 16 GB] |
| GPU | [Fill in: e.g. NVIDIA RTX 3060 / None] |
| Python version | 3.13 |
| STT backend | faster-whisper (small.en model) |
| LLM provider | Anthropic Claude (claude-3-5-sonnet-20241022) / Gemini |
| OCR engine | EasyOCR (en) |
| Test video | [Fill in title and URL from demo/recorded_demo/video_info.json] |
| Video duration | [Fill in: e.g. 8.5 minutes] |
| Date of evaluation | 2026-05-12 |

---

## 3. Recorded Video Demo Results

### 3.1 Speed Metrics

*Fill in the stage timings from `data/output/<run_id>/performance_report.json` after running the demo.*

| Stage | Time (s) | % of Total |
|-------|----------|------------|
| ingest | — | — |
| audio | — | — |
| stt | — | — |
| align | — | — |
| frames | — | — |
| enhance | — | — |
| ocr | — | — |
| fuse | — | — |
| summarize | — | — |
| format | — | — |
| **TOTAL** | **—** | **100%** |

> **Note:** Run `python scripts/evaluate_performance.py` to auto-populate this table from actual run data.

**Key observations:**
- STT is typically the longest non-LLM stage (proportional to video duration)
- LLM summarization dominates for videos > 10 minutes due to API latency
- Frame extraction + OCR combined is usually < 20% of total time

**STT speed ratio:** [Fill in: audio_duration_sec / stt_elapsed_sec] × real time  
*(faster-whisper typically achieves 10–30× real-time speed on CPU)*

### 3.2 Transcript Quality

| Metric | Value |
|--------|-------|
| Segments transcribed | — |
| Words transcribed | — |
| Sentences built | — |
| Transcript source | local-whisper |
| Estimated WER vs YouTube captions | [Fill in or N/A] |
| Hallucinations filtered | [Fill in from aligner logs] |

**WER measurement (if available):**
```bash
pip install jiwer
python -c "
from jiwer import wer
reference = open('reference_transcript.txt').read()
hypothesis = open('data/intermediate/<run_id>/transcript.aligned.json').read()
# ... parse and compare
"
```

### 3.3 Visual Analysis Metrics

| Metric | Value |
|--------|-------|
| Keyframes extracted | — |
| Frames per minute of video | — |
| OCR text regions detected | — |
| OCR average confidence | — |
| OCR minimum confidence | — |
| Enhancement profile used | default |

**Frame extraction rate:** SSIM threshold 0.92 produces approximately [N] frames/minute for this video type.

**OCR confidence distribution:** [Fill in: average, std dev from actual run]

### 3.4 LLM Summarization Quality

| Metric | Value |
|--------|-------|
| LLM chunks | — |
| Key points extracted | — |
| Events detected | — |
| Chapters generated | — |
| Input tokens | — |
| Output tokens | — |
| LLM elapsed time (s) | — |
| Estimated API cost (USD) | — |

**Estimated cost:** Claude Sonnet input = $3.00/Mtok, output = $15.00/Mtok.

**Manual quality assessment (1–5):**

| Dimension | Score | Notes |
|-----------|-------|-------|
| Summary accuracy | — | Does it correctly describe the video? |
| Summary completeness | — | Are major topics covered? |
| Key point relevance | — | Are key points genuinely important? |
| Chapter boundary accuracy | — | Do chapters match actual topic changes? |
| Overall coherence | — | Is the summary well-written? |

### 3.5 Resource Utilization

| Resource | Value |
|----------|-------|
| Peak RAM usage (MB) | — |
| Raw video disk (MB) | — |
| Audio disk (MB) | — |
| Frames disk (MB) | — |
| Intermediate disk (MB) | — |
| Output disk (MB) | — |
| **Total disk per run (MB)** | **—** |

---

## 4. Live Stream Demo Results

### 4.1 Latency Metrics

| Metric | Value |
|--------|-------|
| Stream URL used | [Fill in] |
| Chunk size | 10 seconds |
| Chunks processed | [Fill in] |
| Time from Start to first chunk processed | ~10–15 s |
| Time from chunk ready to rolling summary update | ~15–30 s |
| Rolling summary update interval | 30 s |
| Total session duration | [Fill in] |

**Observations:**
- The inherent minimum latency is 10 s (chunk size) + STT time + fuse + summarize
- For a 10-second chunk, faster-whisper processes audio in ~1–3 s on CPU
- LLM rolling summary adds ~5–15 s per 30-second update cycle

### 4.2 Rolling Summary Quality

*Fill in during live demo.*

| After N chunks | Summary coherence | Notes |
|----------------|-------------------|-------|
| 1 | — | First chunk only |
| 3 | — | ~30 seconds of video |
| 6 | — | ~60 seconds of video |

---

## 5. Robustness Tests

The following scenarios were tested to verify the system handles edge cases gracefully:

| Scenario | Test Method | Result | Notes |
|----------|-------------|--------|-------|
| Video with no YouTube subtitles | Short unlisted video | Pass / Fail | STT fallback should activate |
| Video with no on-screen text | Pure talking-head video | Pass / Fail | OCR returns empty; summary from speech only |
| Very short video (< 1 min) | 30-second clip | Pass / Fail | Single-pass, no chunking needed |
| Very long video (> 30 min) | 45-minute lecture | Pass / Fail | Multi-pass chunking should activate |
| Invalid URL | Garbage string | Pass / Fail | Should raise clear error, not crash |

*Fill in Pass/Fail and Notes after testing each scenario.*

---

## 6. Comparison with Baselines

| Approach | Summary Quality | Processing Time | Cost | Notes |
|----------|----------------|-----------------|------|-------|
| **This system** | High (multi-modal) | ~X min for Y min video | ~$X per run | Full pipeline |
| YouTube auto-summary | Basic (no chapters/KPs) | Instant | Free | Only for eligible videos |
| Manual summarization | Highest | 30+ min per video | Human time | Gold standard |
| Speech-only (no vision) | Medium | Faster (skip frames) | Lower LLM cost | Missing visual insights |
| GPT-4 transcript summary | High (speech only) | Depends on video length | Variable | No visual component |

---

## 7. Limitations

### 7.1 Speed Limitations
- **LLM API latency** is the main bottleneck for long videos. A 30-minute video with 8K-token chunks requires ~4 LLM calls; at 15 s/call average, this adds ~60 s.
- **faster-whisper on CPU** is ~10–30× real time. A 10-minute video takes ~20–60 s for STT.
- **EasyOCR model loading** takes ~5 s on first run (model download/initialization). Subsequent runs are faster.

### 7.2 Quality Limitations
- **STT accuracy** degrades with accented speech, background noise, or technical jargon.
- **OCR accuracy** depends on video quality — low-resolution or blurry frames produce poor results even after DIP enhancement.
- **LLM hallucination** — the model occasionally invents timestamps or events not present in the input. Timestamp validation (`validate_timestamps`) catches out-of-range values.
- **Domain extraction** — domain-specific fields are only extracted if the LLM correctly follows the addendum instructions. Instruction following varies by model and video length.

### 7.3 Resource Limitations
- **Memory:** EasyOCR requires ~500 MB RAM for model weights. LLaVA captioning requires ~4–8 GB.
- **Disk:** A 10-minute 1080p video generates ~500 MB–1 GB of intermediate files.
- **API cost:** A 10-minute video with Claude Sonnet costs ~$0.05–$0.20 depending on content density.

---

## 8. Conclusions

*Fill in after collecting actual data from demo runs.*

**Key findings:**
1. [Fill in: e.g. "The system processes a 10-minute video in under 4 minutes on a standard laptop."]
2. [Fill in: e.g. "OCR confidence averages 85% on slide-based content after DIP enhancement."]
3. [Fill in: e.g. "The two-pass LLM approach produces coherent chapters with 90%+ boundary accuracy."]
4. [Fill in: e.g. "Tutorial-strategy domain mode successfully extracts structured steps from programming tutorials."]

**Recommendation:** The system is suitable for academic and research use cases. For production deployment, consider caching models, pre-warming EasyOCR, and adding a cost cap on LLM calls.
