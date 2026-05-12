# DIP Project — Implementation Plans

This folder contains the **complete, sub-divided implementation plan** for the AI Video Understanding & Summarization system (see `../PRD.md`).

Each plan file is **self-contained**: it describes one atomic deliverable with its own inputs, outputs, dependencies, code structure, tests, and acceptance criteria. A coding assistant should be able to execute any single plan file without reading the others.

---

## Execution Order

Plans must generally be completed in the order listed below, because later plans consume the file/data outputs of earlier ones. Each plan defines a clear **contract** (inputs/outputs) so adjacent plans can be developed independently as long as the contract is honored.

### Module 1 — Media Ingestion & Routing
- [Plan 1.1 — YouTube Downloader](plan-1.1-youtube-downloader.md)
- [Plan 1.2 — Live Stream Chunker](plan-1.2-live-stream-chunker.md)
- [Plan 1.3 — Orchestrator Skeleton](plan-1.3-orchestrator-skeleton.md)

### Module 2 — Audio & Speech Engine
- [Plan 2.1 — Audio Extraction](plan-2.1-audio-extraction.md)
- [Plan 2.2 — Speech-to-Text](plan-2.2-speech-to-text.md)
- [Plan 2.3 — Timestamp Alignment](plan-2.3-timestamp-alignment.md)

### Module 3 — Visual Processing (DIP/CV Engine)
- [Plan 3.1 — Frame Extraction & Differencing](plan-3.1-frame-extraction.md)
- [Plan 3.2 — DIP Enhancements](plan-3.2-dip-enhancements.md)
- [Plan 3.3 — Visual Content Extraction (OCR)](plan-3.3-visual-content-extraction.md)

### Module 4 — Multimodal Reasoning & Output Engine
- [Plan 4.1 — Data Fusion](plan-4.1-data-fusion.md)
- [Plan 4.2 — LLM Summarization](plan-4.2-llm-summarization.md)
- [Plan 4.3 — Output Formatting](plan-4.3-output-formatting.md)

### Module 5 — Final Polish
- [Plan 5.1 — Master Pipeline](plan-5.1-master-pipeline.md)
- [Plan 5.2 — Domain-Specific Logic (Bonus)](plan-5.2-domain-specific.md)

### Module 6 — Remaining Required Work
- [Plan 6.1 — Domain & Strategy Extraction (Implementation)](plan-6.1-domain-strategy-implementation.md)
- [Plan 6.2 — Live Stream UI Integration](plan-6.2-live-stream-ui.md)
- [Plan 6.3 — Demo Preparation](plan-6.3-demo-preparation.md)
- [Plan 6.4 — Architecture Design Document & Diagrams](plan-6.4-architecture-diagram.md)
- [Plan 6.5 — Performance Evaluation Report](plan-6.5-performance-evaluation-report.md)
- [Plan 6.6 — Final Project Report](plan-6.6-final-project-report.md)

---

## Global Conventions

These conventions are referenced by every individual plan. If a plan does not say otherwise, assume:

| Topic | Convention |
|---|---|
| Language | Python 3.11+ |
| Project root | `c:\Users\zakin\Documents\DIP Project` |
| Source folder | `src/` |
| Working files | `data/raw/`, `data/audio/`, `data/frames/`, `data/intermediate/`, `data/output/` |
| Logs | `logs/` (one file per run, named `<timestamp>_<run_id>.log`) |
| Config | `config.yaml` at project root |
| Env vars | `.env` at project root (never committed) |
| Testing | `pytest`, tests live in `tests/`, mirror source layout |
| Style | `black`, `ruff`, type hints everywhere |
| Logging | Python `logging` module, never `print()` in library code |
| IDs | Each pipeline run gets a UUID4 `run_id` used as a folder name under `data/` |

### Standard data-folder layout for one run
```
data/
  raw/<run_id>/video.mp4
  audio/<run_id>/audio.wav
  frames/<run_id>/frame_0001.png ...
  intermediate/<run_id>/transcript.json
  intermediate/<run_id>/visual.json
  intermediate/<run_id>/fused.json
  output/<run_id>/summary.md
  output/<run_id>/report.json
```

Every plan reads/writes inside `data/<stage>/<run_id>/` so multiple runs never collide.
