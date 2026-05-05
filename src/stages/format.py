"""Output formatting stage."""

from __future__ import annotations

from src.output.formatter import format_outputs


def run_format(ctx) -> None:
    paths = ctx.run_ctx.paths
    result = format_outputs(
        summary_path=paths.intermediate / "summary.raw.json",
        transcript_path=paths.intermediate / "transcript.aligned.json",
        fused_path=paths.intermediate / "fused.json",
        visual_path=paths.intermediate / "visual.json",
        output_dir=paths.output,
        run_id=ctx.run_ctx.run_id,
        youtube_url=ctx.run_ctx.url,
    )
    ctx.log.info("Outputs written to %s", paths.output)
    ctx.log.info("  Markdown:    %s", result.markdown)
    ctx.log.info("  HTML:        %s", result.html)
    ctx.log.info("  Report:      %s", result.report_json)
    ctx.log.info("  Chapters:    %s", result.chapters_txt)
    ctx.log.info("  Report card: %s", result.report_card)
