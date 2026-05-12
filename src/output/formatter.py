"""Main output formatter orchestrating all renderers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.fusion.schema import FusedDocument, load_fused_document
from src.llm.schema import Summary, load_summary
from src.vision.schema import VisualExtraction, load_visual_extraction

from .chapters import render_chapters_txt
from .html_renderer import render_html
from .markdown_renderer import render_markdown
from .report_card import render_report_card
from .schema import FormatResult

logger = logging.getLogger(__name__)


def render_report_json(
    summary: Summary,
    fused: FusedDocument,
    visual: VisualExtraction,
    youtube_url: str | None = None,
) -> dict:
    """Generate the canonical JSON report payload."""
    return {
        "run_id": summary.run_id,
        "url": youtube_url,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": asdict(summary),
        "stats": {
            "transcript_word_count": len(summary.full_summary.split()) if summary.full_summary else 0,
            "word_count": len(summary.full_summary.split()) if summary.full_summary else 0,
            "frame_kept_count": len(visual.frames),
            "frame_count": len(visual.frames),
            "ocr_lines_total": sum(len(frame.lines) for frame in visual.frames),
            "duration_sec": fused.duration_sec,
            "events_count": len(fused.events),
            "token_usage": summary.token_usage,
        },
        "metadata": {
            "speech_source": fused.speech_source,
            "ocr_engine": fused.ocr_engine,
            "has_captions": fused.has_captions,
            "model": summary.model,
            "provider": summary.provider,
        },
        "deliverables": {
            "markdown": "summary.md",
            "html": "summary.html",
            "json": "report.json",
            "chapters": "chapters.txt",
            "report_card": "report_card.md",
        },
    }


def format_outputs(
    summary_path: Path,
    transcript_path: Path,
    fused_path: Path,
    visual_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    youtube_url: str | None = None,
    embed_images: bool = True,
    image_max_width: int = 480,
    domain: str | None = None,
) -> FormatResult:
    """Produce all output deliverables for a run."""

    for path in (summary_path, transcript_path, fused_path, visual_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    summary = load_summary(summary_path)
    fused = load_fused_document(fused_path)
    visual = load_visual_extraction(visual_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_content = render_markdown(summary, fused, visual, youtube_url=youtube_url)
    html_content = render_html(
        markdown_content,
        output_dir,
        embed_images=embed_images,
        max_width=image_max_width,
        run_id=run_id,
    )
    report_json = render_report_json(summary, fused, visual, youtube_url=youtube_url)
    chapters_content = render_chapters_txt(summary)
    report_card_content = render_report_card(output_dir)

    markdown_path = output_dir / "summary.md"
    html_path = output_dir / "summary.html"
    json_path = output_dir / "report.json"
    chapters_path = output_dir / "chapters.txt"
    report_card_path = output_dir / "report_card.md"

    markdown_path.write_text(markdown_content, encoding="utf-8")
    html_path.write_text(html_content, encoding="utf-8")
    json_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")
    chapters_path.write_text(chapters_content, encoding="utf-8")
    report_card_path.write_text(report_card_content, encoding="utf-8")

    total_size = sum(path.stat().st_size for path in [markdown_path, html_path, json_path, chapters_path, report_card_path])

    # Domain post-processing — run after core outputs are on disk
    extra_files: list[Path] = []
    if domain:
        try:
            from src.domain.registry import get_domain
            profile = get_domain(domain)
            summary_dict = json.loads(json_path.read_text(encoding="utf-8"))
            fused_dict = json.loads(Path(fused_path).read_text(encoding="utf-8"))
            extra_files = profile.post_process(summary_dict, fused_dict, output_dir)
            for p in extra_files:
                logger.info("Domain '%s' extra: %s", domain, p)
        except Exception as exc:
            logger.warning("Domain post-processing failed (non-fatal): %s", exc)

    return FormatResult(
        markdown=markdown_path,
        html=html_path,
        report_json=json_path,
        chapters_txt=chapters_path,
        report_card=report_card_path,
        total_size_bytes=total_size,
        extra_files=extra_files,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format DIP outputs into final deliverables")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--transcript", required=True, type=Path)
    parser.add_argument("--fused", required=True, type=Path)
    parser.add_argument("--visual", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--youtube-url", default=None)
    parser.add_argument("--embed-images", action="store_true")
    parser.add_argument("--image-max-width", type=int, default=480)
    args = parser.parse_args()

    format_outputs(
        summary_path=args.summary,
        transcript_path=args.transcript,
        fused_path=args.fused,
        visual_path=args.visual,
        output_dir=args.out_dir,
        run_id=args.run_id,
        youtube_url=args.youtube_url,
        embed_images=args.embed_images,
        image_max_width=args.image_max_width,
    )
