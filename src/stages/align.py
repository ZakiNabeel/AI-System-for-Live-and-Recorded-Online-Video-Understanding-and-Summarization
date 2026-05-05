"""Transcript alignment stage."""

from __future__ import annotations

from src.speech.aligner import align_transcript


def run_align(ctx) -> None:
    paths = ctx.run_ctx.paths
    transcript_path = paths.intermediate / "transcript.json"
    audio_path = paths.audio / "audio.wav"

    result = align_transcript(
        transcript_path=transcript_path,
        audio_path=audio_path if audio_path.exists() else None,
        output_dir=paths.intermediate,
    )
    ctx.log.info(
        "Aligned: %d sentences, %d words -> %s",
        result.sentence_count,
        result.word_count,
        result.aligned_json,
    )
