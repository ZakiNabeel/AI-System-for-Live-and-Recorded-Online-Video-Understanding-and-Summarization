#!/usr/bin/env python3
"""
Collect and report performance metrics from one or more pipeline runs.

Usage:
    python scripts/evaluate_performance.py
    python scripts/evaluate_performance.py --run-ids <id1> <id2>
    python scripts/evaluate_performance.py --output docs/performance_data/metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATA_ROOT = Path("data")
DOCS_ROOT = Path("docs") / "performance_data"


def collect_run_metrics(run_id: str) -> dict:
    """Collect all available metrics for a single pipeline run."""
    metrics: dict = {"run_id": run_id}

    # ── Performance report ────────────────────────────────────────────────
    perf_path = DATA_ROOT / "output" / run_id / "performance_report.json"
    if perf_path.exists():
        try:
            perf = json.loads(perf_path.read_text(encoding="utf-8"))
            metrics["total_sec"] = perf.get("total_sec", 0)
            metrics["stage_timings"] = perf.get("stage_timings_sec", {})
        except Exception:
            pass

    # ── Summary / report.json ─────────────────────────────────────────────
    report_path = DATA_ROOT / "output" / run_id / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report.get("summary", {})
            stats   = report.get("stats", {})
            meta    = report.get("metadata", {})
            metrics["url"]         = report.get("url", "")
            metrics["model"]       = meta.get("model", "")
            metrics["provider"]    = meta.get("provider", "")
            metrics["n_chunks"]    = summary.get("n_chunks", 1)
            metrics["n_key_points"] = len(summary.get("key_points", []))
            metrics["n_events"]    = len(summary.get("events", []))
            metrics["n_chapters"]  = len(summary.get("chapters", []))
            metrics["elapsed_llm_sec"] = summary.get("elapsed_sec", 0)
            metrics["token_usage"] = summary.get("token_usage", {})
            metrics["duration_sec"] = stats.get("duration_sec", 0)
            metrics["frame_count"] = stats.get("frame_count", 0)
            metrics["ocr_lines"]   = stats.get("ocr_lines_total", 0)
            metrics["summary_word_count"] = len(
                (summary.get("full_summary") or "").split()
            )
        except Exception:
            pass

    # ── Transcript metrics ────────────────────────────────────────────────
    transcript_path = DATA_ROOT / "intermediate" / run_id / "transcript.aligned.json"
    if transcript_path.exists():
        try:
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            segments = transcript.get("segments", [])
            metrics["n_segments"] = len(segments)
            metrics["transcript_duration_sec"] = transcript.get("duration_sec", 0)
            metrics["transcript_source"] = transcript.get("source", "")
            words = sum(len(s.get("words", [])) for s in segments)
            metrics["n_words"] = words
            sentences = transcript.get("sentences", [])
            metrics["n_sentences"] = len(sentences)
        except Exception:
            pass

    # ── Frame metrics ─────────────────────────────────────────────────────
    frames_path = DATA_ROOT / "frames" / run_id / "frames.json"
    if frames_path.exists():
        try:
            frames_data = json.loads(frames_path.read_text(encoding="utf-8"))
            metrics["n_keyframes"] = len(frames_data.get("frames", []))
        except Exception:
            pass

    # ── OCR confidence stats ──────────────────────────────────────────────
    visual_path = DATA_ROOT / "intermediate" / run_id / "visual.json"
    if visual_path.exists():
        try:
            visual = json.loads(visual_path.read_text(encoding="utf-8"))
            confidences = []
            for frame in visual.get("frames", []):
                for ocr in frame.get("ocr_results", []):
                    conf = ocr.get("confidence")
                    if conf is not None:
                        confidences.append(float(conf))
            if confidences:
                metrics["ocr_avg_confidence"]  = round(statistics.mean(confidences), 4)
                metrics["ocr_min_confidence"]  = round(min(confidences), 4)
                metrics["ocr_n_text_regions"]  = len(confidences)
        except Exception:
            pass

    # ── Disk usage ────────────────────────────────────────────────────────
    total_bytes = 0
    for stage in ("raw", "audio", "frames", "intermediate", "output"):
        stage_dir = DATA_ROOT / stage / run_id
        if stage_dir.exists():
            size = sum(
                f.stat().st_size for f in stage_dir.rglob("*") if f.is_file()
            )
            metrics[f"disk_{stage}_mb"] = round(size / 1_048_576, 2)
            total_bytes += size
    metrics["disk_total_mb"] = round(total_bytes / 1_048_576, 2)

    return metrics


def format_report(all_metrics: list[dict]) -> str:
    """Format a Markdown performance report from collected metrics."""
    lines = [
        "# Performance Metrics Report\n",
        f"**Runs analyzed:** {len(all_metrics)}\n",
        "---\n",
    ]

    for m in all_metrics:
        run_id = m["run_id"]
        lines.append(f"## Run: `{run_id}`\n")

        url = m.get("url", "")
        if url:
            lines.append(f"**URL:** {url}\n")

        duration = m.get("duration_sec") or m.get("transcript_duration_sec", 0)
        if duration:
            lines.append(f"**Video duration:** {duration / 60:.1f} min\n")

        lines.append(f"**Model:** {m.get('model', 'N/A')} ({m.get('provider', 'N/A')})\n")

        # Stage timings
        timings = m.get("stage_timings", {})
        total   = m.get("total_sec", 0)
        if timings:
            lines.append("\n### Stage Timings\n")
            lines.append("| Stage | Time (s) | % of Total |")
            lines.append("|-------|----------|------------|")
            for stage, t in timings.items():
                pct = (t / total * 100) if total else 0
                lines.append(f"| {stage} | {t:.2f} | {pct:.1f}% |")
            lines.append(f"| **TOTAL** | **{total:.1f}** | **100%** |")
            lines.append("")

        # Content stats
        lines.append("\n### Content Statistics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        stat_rows = [
            ("Segments transcribed",   m.get("n_segments", "N/A")),
            ("Words transcribed",      m.get("n_words", "N/A")),
            ("Sentences",              m.get("n_sentences", "N/A")),
            ("Transcript source",      m.get("transcript_source", "N/A")),
            ("Keyframes extracted",    m.get("n_keyframes", "N/A")),
            ("OCR text regions",       m.get("ocr_n_text_regions", "N/A")),
            ("OCR avg confidence",     f"{m['ocr_avg_confidence']:.1%}" if "ocr_avg_confidence" in m else "N/A"),
            ("LLM key points",         m.get("n_key_points", "N/A")),
            ("LLM events detected",    m.get("n_events", "N/A")),
            ("LLM chapters",           m.get("n_chapters", "N/A")),
            ("LLM chunks",             m.get("n_chunks", "N/A")),
            ("Summary word count",     m.get("summary_word_count", "N/A")),
        ]
        for label, value in stat_rows:
            lines.append(f"| {label} | {value} |")
        lines.append("")

        # Token usage
        tu = m.get("token_usage", {})
        if tu:
            lines.append("\n### Token Usage\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Input tokens  | {tu.get('input_tokens', 0):,} |")
            lines.append(f"| Output tokens | {tu.get('output_tokens', 0):,} |")
            total_tok = tu.get("input_tokens", 0) + tu.get("output_tokens", 0)
            lines.append(f"| Total tokens  | {total_tok:,} |")
            lines.append(f"| LLM time (s)  | {m.get('elapsed_llm_sec', 0):.1f} |")
            lines.append("")

        # Disk usage
        lines.append("\n### Disk Usage\n")
        lines.append("| Stage | MB |")
        lines.append("|-------|----|")
        for stage in ("raw", "audio", "frames", "intermediate", "output"):
            mb = m.get(f"disk_{stage}_mb", 0)
            lines.append(f"| {stage} | {mb:.1f} |")
        lines.append(f"| **Total** | **{m.get('disk_total_mb', 0):.1f}** |")
        lines.append("")
        lines.append("---\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DIP pipeline performance")
    parser.add_argument("--run-ids", nargs="*", help="Specific run IDs to evaluate")
    parser.add_argument(
        "--output",
        default="docs/performance_data/metrics.json",
        help="Path to write metrics JSON",
    )
    parser.add_argument(
        "--report",
        default="docs/performance_data/metrics_report.md",
        help="Path to write Markdown report",
    )
    args = parser.parse_args()

    # Discover run IDs
    output_dir = DATA_ROOT / "output"
    if args.run_ids:
        run_ids = args.run_ids
    elif output_dir.exists():
        run_ids = [d.name for d in sorted(output_dir.iterdir()) if d.is_dir()]
    else:
        run_ids = []

    if not run_ids:
        print("No pipeline runs found. Run the pipeline first.")
        print("  python -m src.pipeline --url <URL>")
        return

    print(f"Evaluating {len(run_ids)} run(s): {run_ids}")
    all_metrics = [collect_run_metrics(rid) for rid in run_ids]

    # Save raw metrics JSON
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Metrics saved to: {output_path}")

    # Save Markdown report
    report = format_report(all_metrics)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to:  {report_path}")

    # Print summary to stdout
    for m in all_metrics:
        total = m.get("total_sec", 0)
        dur   = m.get("duration_sec") or m.get("transcript_duration_sec", 0)
        print(
            f"\n[{m['run_id']}] total={total:.1f}s "
            f"video={dur/60:.1f}min "
            f"kp={m.get('n_key_points', '?')} "
            f"ch={m.get('n_chapters', '?')} "
            f"tok={m.get('token_usage', {}).get('input_tokens', 0):,}in"
        )


if __name__ == "__main__":
    main()
