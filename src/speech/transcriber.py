"""Top-level speech transcription dispatcher and CLI."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import wave
from pathlib import Path
from typing import Literal, Protocol, Sequence

from .backends import LocalWhisperBackend, OpenAIWhisperBackend, fetch_youtube_transcript
from .errors import AudioTooShortError, NoBackendAvailableError
from .schema import Transcript, save_transcript

BackendName = Literal["auto", "local-whisper", "openai-whisper"]


class TranscriberBackend(Protocol):
    def transcribe(self, audio_path: Path, language: str | None) -> Transcript:
        ...


def _audio_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except (wave.Error, EOFError, OSError):
        return 0.0


def _resolve_backend(backend: BackendName) -> Literal["local-whisper", "openai-whisper"]:
    if backend == "local-whisper":
        return "local-whisper"
    if backend == "openai-whisper":
        if not os.getenv("OPENAI_API_KEY"):
            raise NoBackendAvailableError("OPENAI_API_KEY is required for openai-whisper backend.")
        return "openai-whisper"
    if backend != "auto":
        raise ValueError(f"Unsupported backend: {backend}")

    if importlib.util.find_spec("faster_whisper") is not None:
        return "local-whisper"
    if os.getenv("OPENAI_API_KEY"):
        return "openai-whisper"
    raise NoBackendAvailableError(
        "No backend available: install faster-whisper or set OPENAI_API_KEY for openai-whisper."
    )


def transcribe(
    audio_path: Path | None = None,
    *,
    youtube_url: str | None = None,
    backend: BackendName = "auto",
    model: str = "small.en",
    language: str | None = None,
    use_youtube_subs_if_available: bool = True,
) -> Transcript:
    if audio_path is None and youtube_url is None:
        raise ValueError("At least one of audio_path or youtube_url must be provided.")

    if youtube_url and use_youtube_subs_if_available:
        sub_transcript = fetch_youtube_transcript(youtube_url)
        if sub_transcript is not None:
            return sub_transcript

    if audio_path is None:
        raise ValueError("audio_path required when YouTube subtitles are unavailable.")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: '{audio_path.resolve()}'")

    duration = _audio_duration_seconds(audio_path)
    if duration < 1.0:
        raise AudioTooShortError("Audio is shorter than 1 second.")

    resolved_backend = _resolve_backend(backend)
    if resolved_backend == "local-whisper":
        backend_impl: TranscriberBackend = LocalWhisperBackend(model_name=model)
    else:
        backend_impl = OpenAIWhisperBackend(model="whisper-1")

    return backend_impl.transcribe(audio_path=audio_path, language=language)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe WAV audio into structured transcript JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=Path, help="Path to WAV audio file.")
    group.add_argument("--youtube", type=str, help="YouTube URL.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "local-whisper", "openai-whisper"])
    parser.add_argument("--model", type=str, default="small.en", help="Model name for local backend.")
    parser.add_argument("--language", type=str, default=None, help="Language hint (e.g., en).")
    parser.add_argument(
        "--no-youtube-subs",
        action="store_true",
        help="Disable YouTube subtitles fast path and force model-based transcription.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output transcript JSON path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        transcript = transcribe(
            audio_path=args.audio,
            youtube_url=args.youtube,
            backend=args.backend,
            model=args.model,
            language=args.language,
            use_youtube_subs_if_available=not args.no_youtube_subs,
        )
        save_transcript(transcript, args.out)
        elapsed = time.perf_counter() - started
        word_count = sum(len(segment.words) for segment in transcript.segments)
        duration = transcript.duration_sec
        duration_str = "nan" if math.isnan(duration) else f"{duration:.1f}"
        print(
            f"[INFO] transcribed {duration_str} s in {elapsed:.1f} s, {word_count} words",
            file=sys.stderr,
        )
        return 0
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

