"""CLI for output formatter."""

import argparse
import logging
import sys
from pathlib import Path

from .formatter import format_outputs


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Format DIP output to final deliverables (Markdown, HTML, JSON, chapters, report card)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.output.cli \\
    --summary data/intermediate/run123/summary.raw.json \\
    --transcript data/intermediate/run123/transcript.aligned.json \\
    --fused data/intermediate/run123/fused.json \\
    --visual data/intermediate/run123/visual.json \\
    --out-dir data/output/run123

  python -m src.output.cli \\
    --summary data/intermediate/run123/summary.raw.json \\
    --transcript data/intermediate/run123/transcript.aligned.json \\
    --fused data/intermediate/run123/fused.json \\
    --visual data/intermediate/run123/visual.json \\
    --out-dir data/output/run123 \\
    --youtube-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \\
    --embed-images
        """,
    )

    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="Path to summary.raw.json (from Plan 4.2)",
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=Path,
        help="Path to transcript.aligned.json (from Plan 2.3)",
    )
    parser.add_argument(
        "--fused",
        required=True,
        type=Path,
        help="Path to fused.json (from Plan 4.1)",
    )
    parser.add_argument(
        "--visual",
        required=True,
        type=Path,
        help="Path to visual.json (from Plan 3.3)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for deliverables",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Run ID (extracted from paths if not specified)",
    )
    parser.add_argument(
        "--youtube-url",
        default=None,
        help="YouTube URL for timestamp linking",
    )
    parser.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed images in HTML as data URIs (default: relative paths)",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=480,
        help="Max width for embedded images (default: 480)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Infer run_id if not specified
    run_id = args.run_id
    if not run_id:
        # Try to extract from paths
        run_id = args.summary.parent.name

    logger.info(f"Formatting outputs for run: {run_id}")

    try:
        result = format_outputs(
            summary_path=args.summary,
            transcript_path=args.transcript,
            fused_path=args.fused,
            visual_path=args.visual,
            output_dir=args.out_dir,
            run_id=run_id,
            youtube_url=args.youtube_url,
            embed_images=args.embed_images,
            image_max_width=args.image_max_width,
        )

        logger.info("✅ Formatting complete!")
        logger.info(f"Output directory: {result.markdown.parent}")
        logger.info(f"Markdown: {result.markdown}")
        logger.info(f"HTML: {result.html}")
        logger.info(f"JSON: {result.report_json}")
        logger.info(f"Chapters: {result.chapters_txt}")
        logger.info(f"Report Card: {result.report_card}")
        logger.info(f"Total size: {result.total_size_bytes} bytes")

        return 0

    except Exception as e:
        logger.error(f"❌ Formatting failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
