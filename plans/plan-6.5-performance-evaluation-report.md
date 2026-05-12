# Plan 6.5 — Performance Evaluation Report

> **Self-contained scope.** Design and produce the Performance Evaluation Report required by the PRD. This covers: what metrics to measure, how to collect them (using data already logged by the pipeline), and the format of the final `docs/performance_report.md` document. A companion script `scripts/evaluate_performance.py` automates metric collection from existing pipeline runs.

---

## 1. Objective

Produce:
1. `docs/performance_report.md` — the graded deliverable document.
2. `scripts/evaluate_performance.py` — script that reads `performance_report.json` from one or more runs and computes summary statistics.
3. `docs/performance_data/` — raw data files collected from demo runs.

---

## 2. Metrics to Evaluate

### 2.1 Speed / Latency
| Metric | How measured | Source |
|--------|-------------|--------|
| Total pipeline time (seconds) | `performance_report.json → total_sec` | Already logged |
| Per-stage latency | `performance_report.json → stage_timings_sec` | Already logged |
| STT speed ratio | `audio_duration_sec / stt_elapsed_sec` | Compute from manifest + timing |
| Frames per second (extraction) | `n_frames / frames_elapsed_sec` | Compute |
| LLM tokens/second | `total_output_tokens / llm_elapsed_sec` | Compute from token_usage |

### 2.2 Accuracy / Quality
| Metric | How measured | Notes |
|--------|-------------|-------|
| Transcript Word Error Rate (WER) | Compare to human reference or YouTube auto-captions | Use `jiwer` library |
| Hallucination rate | Count filtered segments / total segments | Logged in aligner |
| OCR confidence | Average confidence score from EasyOCR | In `visual.json → frames[*].ocr_results[*].confidence` |
| Summary ROUGE-1/2/L | Compare to human-written reference summary | Use `rouge-score` library |
| Chapter boundary accuracy | Manual check: do chapters match actual topic changes | Manual review |

### 2.3 Resource Utilization
| Metric | How measured |
|--------|-------------|
| Peak memory (MB) | `tracemalloc` or `psutil` in pipeline (add wrapper) |
| Disk space per run | Size of `data/<stage>/<run_id>/` directories |
| API cost estimate | `input_tokens * price + output_tokens * price` (logged in token_usage) |

### 2.4 Robustness
| Scenario | Test |
|----------|------|
| Video with no subtitles | Forced STT fallback — verify transcript produced |
| Video with no on-screen text | OCR returns empty — verify no crash, summary still produced |
| Very short video (< 1 min) | Pipeline handles gracefully |
| Very long video (> 30 min) | Multi-pass chunking activates — verify complete output |
| Live stream with gaps | Chunker handles short gaps — no crash |

---

## 3. Evaluation Script (`scripts/evaluate_performance.py`)

```python
#!/usr/bin/env python3
"""
Collect and report performance metrics from one or more pipeline runs.

Usage:
    python scripts/evaluate_performance.py [--run-ids <id1> <id2> ...]
    # If no --run-ids given, uses all runs found in data/output/
"""

from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


DATA_ROOT = Path("data")
DOCS_ROOT = Path("docs/performance_data")


def collect_run_metrics(run_id: str) -> dict:
    """Collect all available metrics for a single run."""
    metrics: dict = {"run_id": run_id}

    # Performance report
    perf_path = DATA_ROOT / "output" / run_id / "performance_report.json"
    if perf_path.exists():
        perf = json.loads(perf_path.read_text())
        metrics["total_sec"] = perf.get("total_sec", 0)
        metrics["stage_timings"] = perf.get("stage_timings_sec", {})

    # Token usage from summary
    report_path = DATA_ROOT / "output" / run_id / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        metrics["token_usage"] = report.get("token_usage", {})
        metrics["model"] = report.get("model", "")
        metrics["provider"] = report.get("provider", "")
        metrics["n_chunks"] = report.get("n_chunks", 1)
        metrics["n_key_points"] = len(report.get("key_points", []))
        metrics["n_events"] = len(report.get("events", []))
        metrics["n_chapters"] = len(report.get("chapters", []))

    # Transcript metrics
    transcript_path = DATA_ROOT / "intermediate" / run_id / "transcript.aligned.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        segments = transcript.get("segments", [])
        metrics["n_segments"] = len(segments)
        metrics["transcript_duration_sec"] = transcript.get("duration_sec", 0)
        words = sum(len(s.get("words", [])) for s in segments)
        metrics["n_words"] = words

    # Frame metrics
    frames_path = DATA_ROOT / "frames" / run_id / "frames.json"
    if frames_path.exists():
        frames = json.loads(frames_path.read_text())
        metrics["n_frames"] = len(frames.get("frames", []))

    # Visual metrics (OCR confidence)
    visual_path = DATA_ROOT / "intermediate" / run_id / "visual.json"
    if visual_path.exists():
        visual = json.loads(visual_path.read_text())
        confidences = []
        for frame in visual.get("frames", []):
            for ocr in frame.get("ocr_results", []):
                if "confidence" in ocr:
                    confidences.append(float(ocr["confidence"]))
        if confidences:
            metrics["ocr_avg_confidence"] = statistics.mean(confidences)
            metrics["ocr_n_text_regions"] = len(confidences)

    # Disk usage
    for stage in ["raw", "audio", "frames", "intermediate", "output"]:
        stage_dir = DATA_ROOT / stage / run_id
        if stage_dir.exists():
            size_bytes = sum(f.stat().st_size for f in stage_dir.rglob("*") if f.is_file())
            metrics[f"disk_{stage}_mb"] = size_bytes / 1_048_576

    return metrics


def print_report(all_metrics: list[dict]) -> str:
    """Format a Markdown performance report from collected metrics."""
    lines = ["# Performance Evaluation Report\n"]
    lines.append(f"Runs analyzed: {len(all_metrics)}\n")

    for m in all_metrics:
        run_id = m["run_id"]
        lines.append(f"\n## Run: `{run_id}`\n")

        # Timing
        total = m.get("total_sec", 0)
        lines.append(f"**Total pipeline time:** {total:.1f}s\n")
        timings = m.get("stage_timings", {})
        if timings:
            lines.append("| Stage | Time (s) |")
            lines.append("|-------|----------|")
            for stage, t in timings.items():
                lines.append(f"| {stage} | {t:.2f} |")
            lines.append("")

        # Content stats
        dur = m.get("transcript_duration_sec", 0)
        if dur:
            lines.append(f"**Video duration:** {dur/60:.1f} min")
        lines.append(f"**Transcript segments:** {m.get('n_segments', 'N/A')}")
        lines.append(f"**Words transcribed:** {m.get('n_words', 'N/A')}")
        lines.append(f"**Keyframes extracted:** {m.get('n_frames', 'N/A')}")
        if "ocr_avg_confidence" in m:
            lines.append(f"**OCR avg confidence:** {m['ocr_avg_confidence']:.1%}")
            lines.append(f"**OCR text regions:** {m.get('ocr_n_text_regions', 0)}")
        lines.append("")

        # Summary quality
        lines.append(f"**Key points extracted:** {m.get('n_key_points', 'N/A')}")
        lines.append(f"**Events detected:** {m.get('n_events', 'N/A')}")
        lines.append(f"**Chapters:** {m.get('n_chapters', 'N/A')}")
        lines.append(f"**LLM chunks:** {m.get('n_chunks', 'N/A')}")
        lines.append(f"**Model:** {m.get('model', 'N/A')} ({m.get('provider', 'N/A')})")
        tu = m.get("token_usage", {})
        if tu:
            lines.append(f"**Tokens (in/out):** {tu.get('input_tokens', 0):,} / {tu.get('output_tokens', 0):,}")
        lines.append("")

        # Disk
        total_disk = sum(m.get(f"disk_{s}_mb", 0) for s in ["raw", "audio", "frames", "intermediate", "output"])
        lines.append(f"**Total disk usage:** {total_disk:.1f} MB")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-ids", nargs="*", help="Run IDs to evaluate")
    parser.add_argument("--output", default="docs/performance_data/metrics.json")
    args = parser.parse_args()

    output_dir = Path("data/output")
    if args.run_ids:
        run_ids = args.run_ids
    else:
        run_ids = [d.name for d in output_dir.iterdir() if d.is_dir()] if output_dir.exists() else []

    if not run_ids:
        print("No runs found. Run the pipeline first.")
        return

    all_metrics = [collect_run_metrics(rid) for rid in run_ids]

    # Save raw metrics
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {args.output}")

    # Print report
    report = print_report(all_metrics)
    print(report)


if __name__ == "__main__":
    main()
```

---

## 4. `docs/performance_report.md` Structure

The final document must be written (not auto-generated) after running the demo, using the script output as raw data. Structure:

```
# Performance Evaluation Report
## 1. Methodology
## 2. Test Setup
## 3. Recorded Video Demo Results
   ### 3.1 Speed Metrics
   ### 3.2 Content Quality Metrics
   ### 3.3 OCR & Visual Analysis Metrics
   ### 3.4 LLM Summarization Quality
   ### 3.5 Resource Utilization
## 4. Live Stream Demo Results
   ### 4.1 Latency Metrics
   ### 4.2 Rolling Summary Quality
## 5. Robustness Tests
## 6. Comparison with Baselines
## 7. Limitations
## 8. Conclusions
```

### Section-by-section content guide:

**§1 Methodology** — describe how each metric was measured; cite specific files (e.g., `performance_report.json`, `visual.json`); explain WER calculation if used.

**§2 Test Setup** — hardware specs (CPU/GPU, RAM); Python version; model versions; video titles and URLs used; date of evaluation.

**§3 Recorded Demo Results**

*Speed metrics table:*
```
| Stage         | Time (s) | % of total |
|---------------|----------|------------|
| ingest        | X.X      | X%         |
| audio         | X.X      | X%         |
| stt           | X.X      | X%         |
| align         | X.X      | X%         |
| frames        | X.X      | X%         |
| enhance       | X.X      | X%         |
| ocr           | X.X      | X%         |
| fuse          | X.X      | X%         |
| summarize     | X.X      | X%         |
| format        | X.X      | X%         |
| **TOTAL**     | **X.X**  | **100%**   |
```

*Content quality:* Number of words transcribed, WER estimate vs YouTube captions, number of OCR text regions detected, OCR average confidence.

*Summary quality:* Number of key points, events, chapters. ROUGE scores if reference available. Manual quality score 1–5 for coherence, accuracy, completeness.

**§4 Live Stream Results** — chunk processing latency (time from chunk end to summary update), rolling summary coherence (manual), total session duration tested.

**§5 Robustness Tests** — table of 5 scenarios from §2.4, with pass/fail and notes.

**§6 Baseline Comparison**
| Approach | Method | Quality | Speed | Cost |
|----------|--------|---------|-------|------|
| This system | full pipeline | High | ~Xmin | ~$X |
| YouTube auto-summary | YouTube feature | Basic | Instant | Free |
| Manual summarization | Human | Highest | 30+ min | High |
| Transcript only (no vision) | STT only | Medium | Faster | Lower |

**§7 Limitations** — live stream latency bound by chunk size; OCR accuracy depends on video quality; LLM cost scales with video length.

**§8 Conclusions** — key findings, what worked well, what to improve.

---

## 5. Running the Evaluation

```bash
# Run the evaluation script on all completed runs
python scripts/evaluate_performance.py --output docs/performance_data/metrics.json

# Run on a specific run
python scripts/evaluate_performance.py --run-ids <run_id>
```

Then manually write `docs/performance_report.md` using the script output as data, following the structure in §4.

---

## 6. Dependencies

Add to `requirements.txt` if not present (for optional WER and ROUGE evaluation):
```
jiwer>=3.0.3        # WER calculation
rouge-score>=0.1.2  # ROUGE scores
psutil>=5.9.0       # Memory monitoring
```

These are optional — the core metrics (speed, token usage, OCR confidence) come from data already on disk.

---

## 7. Phased Execution

| Phase | Task | Effort |
|-------|------|--------|
| A | Write `scripts/evaluate_performance.py` | 1 hr |
| B | Run evaluation script on demo run outputs | 15 min |
| C | Write §1–3 of `docs/performance_report.md` from script data | 1 hr |
| D | Write §4–8 manually from live demo observation | 1 hr |
| E | Add ROUGE/WER metrics if time permits | 45 min |

**Total estimated effort: ~4 hours**

---

## 8. Acceptance Criteria

- [ ] `scripts/evaluate_performance.py` runs without error on any completed pipeline run.
- [ ] `docs/performance_data/metrics.json` exists with at least one run's data.
- [ ] `docs/performance_report.md` exists and covers all 8 sections.
- [ ] Speed metrics table has actual numbers (not placeholders) from a real pipeline run.
- [ ] OCR confidence metrics are present (from `visual.json`).
- [ ] Robustness test table has pass/fail for at least 3 of the 5 scenarios.

---

## 9. Definition of Done

A reviewer opens `docs/performance_report.md` and finds a complete performance evaluation with actual numeric data from real pipeline runs, covering speed, quality, resource usage, and robustness — supported by raw data files in `docs/performance_data/`.
