# Architecture Design

## System: DIP Video Understanding & Summarization

---

## 1. System Overview

The DIP Video Understanding & Summarization System is an end-to-end AI pipeline that processes both live and recorded online videos. It extracts spoken content via automatic speech recognition (ASR), analyzes visual frames using Digital Image Processing (DIP) and OCR, merges these modalities into a unified event timeline, and synthesizes structured summaries using a large language model (LLM).

**Key capabilities:**
- YouTube video download and live stream capture
- Multi-backend speech-to-text with timestamp alignment
- SSIM-based keyframe extraction and DIP enhancement
- Multi-engine OCR and optional LLM-based image captioning
- Multimodal fusion into a time-indexed event document
- Two-pass LLM summarization (per-chunk + global synthesis)
- Five domain-specific output modes
- Resumable 10-stage pipeline with Streamlit web UI

**Technology stack:**

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Video/Audio | yt-dlp, streamlink, ffmpeg |
| ASR | faster-whisper, OpenAI Whisper API, Google Gemini STT |
| Computer Vision | OpenCV, NumPy, Pillow |
| OCR | Tesseract, EasyOCR |
| LLM | Anthropic Claude, OpenAI GPT-4, Google Gemini, Ollama |
| Web UI | Streamlit |
| Testing | pytest |
| Config | YAML + python-dotenv |

**System diagram:**

![System Overview](diagrams/system_overview.mmd)

*(Render with Mermaid CLI: `mmdc -i docs/diagrams/system_overview.mmd -o docs/diagrams/system_overview.png`)*

---

## 2. Module Descriptions

### 2.1 `src/ingest/` — Media Ingestion

Responsible for acquiring video data from the source.

| File | Role |
|------|------|
| `youtube_downloader.py` | Downloads YouTube videos using yt-dlp; extracts metadata (title, duration, subtitles, resolution); falls back to lower resolution if needed |
| `live_chunker.py` | Captures live streams via ffmpeg into 10-second rolling chunks; exposes `on_chunk_ready` callback |
| `stream_resolver.py` | Resolves live-page URLs to playable HLS/DASH/RTMP streams using streamlink with yt-dlp fallback |
| `errors.py` | Domain exceptions: `PrivateVideoError`, `FFmpegMissingError`, `StreamNotFoundError` |

**Inputs:** URL string  
**Outputs:** `data/raw/<run_id>/video.mp4` (recorded) or chunk files (live)

---

### 2.2 `src/audio/` — Audio Processing

Extracts audio from video and normalizes it for ASR.

| File | Role |
|------|------|
| `extractor.py` | Runs ffmpeg to extract audio, normalize to 16 kHz mono WAV; probes duration via ffprobe |

**Inputs:** `video.mp4`  
**Outputs:** `data/audio/<run_id>/audio.wav`

---

### 2.3 `src/speech/` — Speech-to-Text

Transcribes audio to text with word-level timestamps.

| File | Role |
|------|------|
| `transcriber.py` | Top-level dispatcher; tries YouTube subtitles first, falls back to ASR backends |
| `backends/local_whisper.py` | faster-whisper (CTranslate2); local, free, word-level timestamps |
| `backends/openai_api.py` | OpenAI Whisper API; handles large files via segmentation |
| `backends/youtube_subs.py` | Extracts YouTube auto-captions (free, no audio processing) |
| `backends/gemini_stt.py` | Google Gemini STT (experimental) |
| `aligner.py` | Alignment pipeline: hallucination filter → gap/overlap fix → sentence builder |
| `aligner_steps.py` | Individual steps: `drop_hallucinations`, `merge_tiny_segments`, `split_overlong`, `polish_text` |
| `hallucination_filter.py` | Regex patterns to drop common ASR artifacts ("thanks for watching", repetitions) |
| `subtitles.py` | Writes SRT and WebVTT subtitle files from aligned transcript |
| `schema.py` | `Transcript`, `TranscriptSegment`, `Word`, `Sentence`, `AlignedTranscript` dataclasses |

**Inputs:** `audio.wav` or YouTube URL  
**Outputs:** `transcript.json`, `transcript.aligned.json`, `.srt`, `.vtt`

---

### 2.4 `src/vision/` — Visual Processing (DIP Pipeline)

Extracts and analyzes visual content from video frames.

| File | Role |
|------|------|
| `frame_extractor.py` | SSIM-based keyframe detection; configurable threshold, sample rate, min gap |
| `enhancer.py` | Applies pre-tuned DIP enhancement profiles to frames |
| `dip_steps.py` | Individual DIP operations: grayscale, Gaussian/bilateral denoise, CLAHE, sharpen, threshold, binarize, morph, deskew |
| `enhance_profiles.py` | 4 profiles: `default`, `screen`, `whiteboard`, `scene` |
| `extractor.py` | Orchestrates OCR + optional captioning on enhanced frames |
| `ocr/tesseract_ocr.py` | Tesseract OCR backend |
| `ocr/easyocr_ocr.py` | EasyOCR backend (ML-based, 80+ languages, confidence scores) |
| `captioning/claude_captioner.py` | Claude Vision API captioning |
| `captioning/openai_captioner.py` | GPT-4 Vision captioning |
| `captioning/gemini_captioner.py` | Gemini Vision captioning |
| `captioning/cache.py` | JSON cache to avoid re-captioning identical frames |
| `schema.py` | `Frame`, `VisualExtraction` dataclasses |

**Inputs:** `video.mp4`  
**Outputs:** `data/frames/<run_id>/`, `data/frames/<run_id>/enhanced/`, `visual.json`

---

### 2.5 `src/fusion/` — Multimodal Fusion

Merges speech and visual events into a unified timeline.

| File | Role |
|------|------|
| `fuser.py` | Time-windows aligned transcript + visual extraction into `FusedEvent` objects |
| `chunker.py` | Splits large `FusedDocument` into token-budget-aware chunks for LLM |
| `tokens.py` | Token counter (tiktoken-based) for both OpenAI and Anthropic models |
| `schema.py` | `FusedEvent`, `FusedDocument` dataclasses |

**Inputs:** `transcript.aligned.json`, `visual.json`  
**Outputs:** `fused.json`

---

### 2.6 `src/llm/` — LLM Summarization

Sends fused content to an LLM and parses structured output.

| File | Role |
|------|------|
| `summarizer.py` | Orchestrates two-pass summarization; injects domain addenda |
| `providers.py` | `AnthropicProvider`, `OpenAIProvider`, `OllamaProvider`, `GeminiProvider` |
| `prompts_system.py` | Loads prompt templates; injects `domain_addendum` before INPUT section |
| `prompts/chunk.txt` | Per-chunk analysis prompt |
| `prompts/global.txt` | Global synthesis prompt |
| `prompts/domains/*.txt` | Per-domain instruction addenda |
| `parsing.py` | `strip_and_parse_json`: recovers JSON from LLM outputs with fences or prose |
| `schema.py` | `Summary`, `KeyPoint`, `DetectedEvent`, `Chapter`, `QAPair` dataclasses |

**Inputs:** `fused.json`  
**Outputs:** `summary.raw.json`

---

### 2.7 `src/domain/` — Domain-Specific Analysis

Adds domain-specific LLM instructions and generates extra output files.

| File | Role |
|------|------|
| `base.py` | `DomainProfile` Protocol definition |
| `registry.py` | `DOMAINS` dict, `get_domain()`, `UnknownDomainError` |
| `education.py` | Extracts learning objectives, definitions, worked examples → `study_notes.md` |
| `trading.py` | Extracts tickers, signals, indicators → `trade_log.csv` + disclaimer |
| `medical.py` | Extracts conditions, treatments, evidence levels → `clinical_notes.md` + disclaimer |
| `law.py` | Extracts cases, statutes, principles → `case_brief.md` + disclaimer |
| `tutorial_strategy.py` | Extracts task, prerequisites, steps, decisions → `strategy.json`, `strategy.md` |
| `pseudocode.py` | Generates Python skeleton from extracted steps → `strategy.py` |

**Inputs:** `summary.raw.json` (already on disk), `fused.json`  
**Outputs:** Domain-specific extra files in `data/output/<run_id>/`

---

### 2.8 `src/output/` — Output Formatting

Renders all deliverables in multiple formats.

| File | Role |
|------|------|
| `formatter.py` | Orchestrates all renderers; calls domain `post_process()` |
| `markdown_renderer.py` | Markdown report with summary, key points, events, chapters, Q&A |
| `html_renderer.py` | HTML report with embedded frame images (base64 data URIs) |
| `chapters.py` | Plain-text chapter list with timestamps |
| `report_card.py` | Stats: word count, token usage, model, stage timings |
| `schema.py` | `FormatResult` dataclass |

**Inputs:** `summary.raw.json`, `transcript.aligned.json`, `fused.json`, `visual.json`  
**Outputs:** `summary.md`, `summary.html`, `report.json`, `chapters.txt`, `report_card.md`, `performance_report.json`

---

## 3. Pipeline Stages

The master pipeline (`src/pipeline.py`) executes 10 stages in order. Each stage is a thin wrapper in `src/stages/`.

| # | Stage | Module | Key Input | Key Output |
|---|-------|--------|-----------|------------|
| 1 | ingest | src/ingest | URL | video.mp4 |
| 2 | audio | src/audio | video.mp4 | audio.wav |
| 3 | stt | src/speech | audio.wav | transcript.json |
| 4 | align | src/speech | transcript.json | transcript.aligned.json |
| 5 | frames | src/vision | video.mp4 | frames.json + PNGs |
| 6 | enhance | src/vision | frames.json | enhancements.json |
| 7 | ocr | src/vision | enhancements.json | visual.json |
| 8 | fuse | src/fusion | aligned + visual | fused.json |
| 9 | summarize | src/llm + domain | fused.json | summary.raw.json |
| 10 | format | src/output + domain | summary.raw.json | all output files |

**Resumability:** Each stage records its status in `manifest.json`. A failed or interrupted run can be resumed with `--run-id <id>` — completed stages are skipped automatically.

**Data isolation:** All files for a run are stored under `data/<stage>/<run_id>/`. Multiple runs never share state.

---

## 4. Live vs Recorded Pipeline

### Recorded Mode

Linear 10-stage pipeline. All stages execute once on the full video. Suitable for any YouTube URL.

### Live Mode

`src/pipeline_live.py` runs a producer-consumer architecture:

1. **Producer:** ffmpeg captures stream into 10-second chunk files, enqueues them.
2. **Consumer (worker thread):** Dequeues chunks, runs audio → STT → align → frames → enhance → OCR for each.
3. **Accumulator:** Merges all chunk outputs with time offsets into running `transcript.aligned.json` and `visual.json`.
4. **Rolling pass:** Every 30 seconds, runs fuse → summarize → format on accumulated data.
5. **UI polling:** `rolling_summary.json` is written after each rolling pass; Streamlit polls it every 3 seconds.

**Diagram:** See `docs/diagrams/live_pipeline.mmd`

---

## 5. DIP (Digital Image Processing) Pipeline

This section details the visual processing pipeline, which is central to the project's DIP contribution.

### 5.1 Keyframe Extraction (`src/vision/frame_extractor.py`)

The system uses **Structural Similarity Index Measure (SSIM)** to detect scene changes:

1. Sample a frame every N seconds (configurable, default: 1 s).
2. Compute SSIM between the candidate frame and the last accepted keyframe (converted to grayscale).
3. If SSIM falls below the threshold (default: 0.92), the frame represents a scene change — accept it.
4. Enforce a minimum gap (default: 1.5 s) to avoid bursts of near-identical frames during fast cuts.

This produces 5–30 keyframes per minute of video, focusing on moments where the visual content meaningfully changed.

### 5.2 Enhancement Profiles

Four pre-tuned enhancement profiles optimize frames for OCR readability:

| Profile | Designed for | DIP Operations |
|---------|-------------|----------------|
| `default` | General videos | Gaussian denoise → CLAHE → sharpen |
| `screen` | Screencasts, presentations | Bilateral denoise → CLAHE → adaptive threshold |
| `whiteboard` | Whiteboards, handwriting | Grayscale → denoise → Otsu binarize → morph close → deskew |
| `scene` | Natural scenes, B-roll | CLAHE → color boost |

### 5.3 DIP Operations (`src/vision/dip_steps.py`)

| Operation | Method | Purpose |
|-----------|--------|---------|
| Grayscale | `cv2.cvtColor(BGR2GRAY)` | Reduces noise dimensions; OCR works on intensity |
| Gaussian denoise | `cv2.GaussianBlur` | Removes high-freq sensor noise |
| Bilateral denoise | `cv2.bilateralFilter` | Edge-preserving noise reduction |
| CLAHE | `cv2.createCLAHE` | Adaptive histogram equalization; improves contrast on dark slides |
| Sharpening | Convolution kernel `[-1,-1,-1; -1,9,-1; -1,-1,-1]` | Enhances text edges |
| Adaptive threshold | `cv2.adaptiveThreshold` | Binarizes under uneven lighting |
| Otsu binarize | `cv2.threshold(...OTSU)` | Global optimal threshold for high-contrast text |
| Morphological close | `cv2.morphologyEx(CLOSE)` | Fills gaps in broken characters |
| Deskew | Hough line transform → rotation correction | Corrects tilted camera or document |

### 5.4 OCR Engines

| Engine | Type | Strengths | Weaknesses |
|--------|------|-----------|------------|
| Tesseract | Traditional (rule-based) | Fast, deterministic, no GPU | Struggles with handwriting, low contrast |
| EasyOCR | ML-based (CRAFT + CRNN) | Handles varied fonts, 80+ languages, confidence scores | Slower, requires model download |

The system defaults to EasyOCR when available and falls back to Tesseract.

---

## 6. Speech Understanding

### Backend Selection Logic

```
if YouTube subtitles available and preferred:
    use youtube_subs backend
elif OPENAI_API_KEY set and audio > 25 MB:
    use openai_api backend (handles large files)
elif GEMINI_API_KEY set and gemini engine selected:
    use gemini_stt backend
else:
    use local_whisper backend (default, always available)
```

### Hallucination Filtering

`hallucination_filter.py` drops segments matching common ASR artifacts:
- Phrases like "thank you for watching", "subscribe", "like and share"
- Repeated words/characters (e.g., "um um um um")
- Very short segments < 0.3 seconds with < 3 words
- Segments with confidence below threshold

### Alignment Steps

1. `drop_hallucinations` — remove artifact segments
2. `merge_tiny_segments` — join segments < 1 s
3. `split_overlong` — split segments > 30 s at punctuation boundaries
4. `polish_text` — normalize whitespace, fix punctuation
5. Sentence boundary detection — group words into readable sentences
6. Optional forced alignment — wav2vec2 for word-level precision (if installed)

---

## 7. Multimodal Fusion

### FusedEvent Schema

```python
@dataclass
class FusedEvent:
    t_start: float          # seconds
    t_end: float
    kind: Literal["speech", "visual", "speech+visual"]
    speech_text: str | None # from aligned transcript
    visual_text: str | None # from OCR
    visual_caption: str | None  # from image captioner
    frame_path: str | None
```

### Time-Windowing Logic

For each 5-second window in the video:
1. Collect all aligned transcript sentences whose `t_start` falls in the window.
2. Collect all visual frames whose timestamp falls in the window.
3. If both exist → `kind = "speech+visual"`
4. If only speech → `kind = "speech"`
5. If only visual → `kind = "visual"`

### Token-Aware Chunking

The `chunker.py` splits `FusedDocument` into chunks of max 8,000 tokens (tiktoken count), with 200-token overlap at boundaries. This ensures LLM calls stay within context limits even for hour-long videos.

---

## 8. LLM Summarization

### Two-Pass Architecture

**Pass 1 — Per-chunk:** For each token-budget chunk, call `prompt_chunk.txt`:
- Extract `local_summary`, `key_points`, `events`
- Domain addendum injected before INPUT section

**Pass 2 — Global synthesis:** Combine all chunk results, call `prompt_global.txt`:
- Produce `full_summary`, `short_summary`, `chapters`, `merged_key_points`, `merged_events`
- Domain addendum injected at prompt start

This two-pass approach is more reliable than single long-context calls because:
- Per-chunk responses are small, structured JSON — easier to validate
- Global pass sees summaries, not raw events — reduces token cost
- Failed chunks can be retried independently

### Structured Output Recovery

`parsing.py → strip_and_parse_json` handles LLM formatting quirks:
1. Try `json.loads(text)` directly
2. On failure: strip markdown fences (` ```json ... ``` `)
3. On failure: regex-find the largest `{...}` block
4. On failure: raise `LLMOutputParseError` (triggers tenacity retry)

### Provider Comparison

| Provider | Default Model | Strengths | Notes |
|----------|-------------|-----------|-------|
| Anthropic Claude | claude-3-5-sonnet-20241022 | Excellent JSON adherence, long context | Recommended default |
| OpenAI GPT | gpt-4o | `response_format=json_object` guarantee | Good fallback |
| Google Gemini | gemini-2.5-flash-lite | Fast, large context (1M tokens) | Good for long videos |
| Ollama | llama2 | Local, free, private | Quality lower than cloud |

---

## 9. Domain-Specific Analysis

### Overview

The `src/domain/` package implements a `DomainProfile` Protocol. When `--domain <name>` is set:

1. The profile's `chunk_prompt_addendum()` is injected into the chunk prompt before the INPUT section.
2. The LLM returns extra fields in its JSON output.
3. After formatting, `post_process()` reads those fields from `report.json` and writes domain-specific files.

### Profile Summary

| Domain | Extra Extracted Fields | Extra Output Files | Disclaimer |
|--------|----------------------|-------------------|-----------|
| `education` | learning_objectives, worked_examples, definitions | `study_notes.md` | None |
| `trading` | tickers_mentioned, signals, indicators, risk_warnings | `trade_log.csv` | Financial advice disclaimer |
| `medical` | conditions_discussed, treatments_mentioned, evidence_levels, key_clinical_points | `clinical_notes.md` | Medical disclaimer |
| `law` | cases_cited, statutes, legal_principles, arguments | `case_brief.md` | Legal disclaimer |
| `tutorial-strategy` | task, prerequisites, steps, decision_points | `strategy.json`, `strategy.md`, `strategy.py`* | None |

*`strategy.py` generated only when `is_programming_tutorial()` returns True.

### Tutorial-Strategy Pseudocode Generation

`pseudocode.py → render_python_skeleton()`:
1. Checks if task/steps contain programming keywords (python, git, docker, npm, etc.)
2. Generates one Python function per step with: timestamp, expected outcome, parameters in docstring
3. Validates with `ast.parse()` — falls back to comment block on syntax error
4. Includes decision branches as inline `# if / else` comments

---

## 10. Design Decisions & Trade-offs

| Decision | Alternative | Reason Chosen |
|----------|-------------|---------------|
| faster-whisper as default STT | OpenAI Whisper API | No API cost; runs offline; comparable accuracy on English |
| SSIM for keyframe detection | Fixed-interval sampling (1 fps) | Adapts to content; skips static frames; fewer frames to process |
| Two-pass LLM summarization | Single long-context call | More reliable structured JSON; per-chunk validation; lower per-call cost |
| DomainProfile Protocol | Inheritance hierarchy | Protocols are more flexible; no base class; easy to add external profiles |
| `report.json` as post-process input | Re-calling LLM for domain | Domain runs after formatting; `report.json` already has all summary data |
| Streamlit for UI | Flask + React / Gradio | Fast iteration; no frontend build step; built-in polling via `st.rerun()` |
| Mermaid for architecture diagrams | draw.io / Visio | Text-based; version-controllable; CI-renderable |
| Per-run isolated directories | Shared output folder | Multiple parallel runs never conflict; easy cleanup; audit trail |
| Manifest-based resumability | Always re-run all stages | Long pipelines (LLM calls) are expensive; resume saves time and cost |

---

## 11. Directory Layout

```
c:\Users\zakin\Documents\DIP Project\
  app.py                     Streamlit web UI
  config.yaml                Main configuration
  .env.example               API key template
  requirements.txt           Python dependencies
  pytest.ini                 Test markers (network, live, e2e)

  src/
    config.py                YAML config loader
    paths.py                 RunPaths dataclass
    manifest.py              Atomic manifest writer
    logging_setup.py         Per-run log files
    orchestrator.py          RunContext creation + mode routing
    pipeline.py              10-stage master pipeline
    pipeline_live.py         Live-stream pipeline + rolling summarizer

    ingest/                  Media ingestion
    audio/                   Audio extraction
    speech/                  STT + alignment + subtitles
    vision/                  Frame extraction + DIP + OCR
    fusion/                  Multimodal fusion
    llm/                     LLM summarization + providers + prompts
    domain/                  Domain-specific profiles
    output/                  Output formatting
    stages/                  Thin stage wrappers

  tests/                     pytest tests (mirrors src/)
  plans/                     Implementation plan documents
  docs/                      Architecture + performance reports
  demo/                      Demo scripts and pre-run outputs
  scripts/                   Utility scripts (evaluation, etc.)

  data/                      Per-run working data (auto-created)
    raw/<run_id>/            Downloaded video
    audio/<run_id>/          Extracted WAV
    frames/<run_id>/         Keyframes + enhanced
    intermediate/<run_id>/   Transcripts, visual, fused, summary
    output/<run_id>/         Final deliverables
  logs/                      Per-run log files
```

---

## 12. Rendering Diagrams

```bash
# Install Mermaid CLI (requires Node.js)
npm install -g @mermaid-js/mermaid-cli

# Render all diagrams to PNG
for f in docs/diagrams/*.mmd; do
    mmdc -i "$f" -o "${f%.mmd}.png" -b white -w 1600
done
```

Alternatively, paste `.mmd` content into https://mermaid.live and export as PNG/SVG.
