"""CLI for LLM summarizer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.llm.summarizer import summarize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Summarize fused video content using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.llm.summarizer \\
    --fused data/intermediate/run123/fused.json \\
    --out data/intermediate/run123/summary.raw.json

  python -m src.llm.summarizer \\
    --fused data/intermediate/run123/fused.json \\
    --out data/intermediate/run123/summary.raw.json \\
    --provider openai --model gpt-4o \\
    --style detailed --qa
        """,
    )

    parser.add_argument(
        "--fused",
        type=Path,
        required=True,
        help="Path to fused.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write summary.raw.json",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override default model for provider",
    )
    parser.add_argument(
        "--style",
        choices=["concise", "detailed", "bullet-only"],
        default="detailed",
        help="Summary style (default: detailed)",
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Generate Q&A pairs",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Domain-specific hooks (e.g., education, trading)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        summary = summarize(
            args.fused,
            args.out,
            provider=args.provider,
            model=args.model,
            style=args.style,
            enable_qa=args.qa,
            domain=args.domain,
        )

        # Print summary stats
        print(f"\n✓ Summary complete")
        print(f"  Provider: {summary.provider}")
        print(f"  Model: {summary.model}")
        print(f"  Tokens: {summary.token_usage.get('input_tokens', 0)} input, {summary.token_usage.get('output_tokens', 0)} output")
        print(f"  Elapsed: {summary.elapsed_sec:.1f}s")
        print(f"  Output: {args.out}")
        print(f"\n  Key points: {len(summary.key_points)}")
        print(f"  Events: {len(summary.events)}")
        print(f"  Chapters: {len(summary.chapters)}")
        if summary.qa_pairs:
            print(f"  Q&A pairs: {len(summary.qa_pairs)}")

        return 0

    except Exception as e:
        LOGGER.exception(f"Summarization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
