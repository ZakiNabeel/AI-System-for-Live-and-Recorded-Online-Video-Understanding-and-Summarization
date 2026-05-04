"""LLM summarization stage."""

from __future__ import annotations

from src.llm.summarizer import summarize


def run_summarize(ctx) -> None:
    paths = ctx.run_ctx.paths
    fused_path = paths.intermediate / "fused.json"
    summary_path = paths.intermediate / "summary.raw.json"

    cfg = ctx.config.get("llm", {})
    summarize(
        fused_path=fused_path,
        output_path=summary_path,
        provider=cfg.get("provider", "anthropic"),
        model=cfg.get("model"),
        enable_qa=ctx.enable_qa,
        domain=ctx.domain,
    )
    ctx.log.info("Summary -> %s", summary_path)
