"""Speech-to-text stage."""

from __future__ import annotations

from src.speech.transcriber import transcribe
from src.speech.schema import save_transcript


def run_stt(ctx) -> None:
    paths = ctx.run_ctx.paths
    audio_path = paths.audio / "audio.wav"
    transcript_path = paths.intermediate / "transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ctx.config.get("speech", {})

    # Prefer YouTube subtitles; fall back to local audio if available.
    transcript = transcribe(
        audio_path=audio_path if audio_path.exists() else None,
        youtube_url=ctx.run_ctx.url,
        model=cfg.get("model", "small.en"),
        use_youtube_subs_if_available=True,
    )
    save_transcript(transcript, transcript_path)
    ctx.log.info(
        "Transcript saved: %s (%d segments, source=%s)",
        transcript_path,
        len(transcript.segments),
        transcript.source,
    )
