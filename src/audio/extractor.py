"""Extract normalized WAV audio from recorded videos or live chunks."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .errors import AudioExtractionError


@dataclass(frozen=True)
class AudioExtractionResult:
    audio_path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    bits_per_sample: int
    file_size_bytes: int
    source_video: Path


def _run_subprocess(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioExtractionError(f"Required binary not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "unknown subprocess error"
        raise AudioExtractionError(f"{args[0]} failed: {detail}") from exc


def _probe_audio(path: Path) -> dict:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-select_streams",
        "a:0",
        str(path),
    ]
    out = _run_subprocess(args).stdout
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise AudioExtractionError(f"ffprobe produced invalid JSON for '{path}'.") from exc


def _build_result(audio_path: Path, source_video: Path) -> AudioExtractionResult:
    probe = _probe_audio(audio_path)
    streams = probe.get("streams") or []
    if not streams:
        raise AudioExtractionError(
            f"Extracted file has no audio stream: '{audio_path}'. "
            "Source may have no audio track."
        )

    stream = streams[0]
    fmt = probe.get("format") or {}

    try:
        duration_sec = float(fmt.get("duration", 0.0))
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        bits_per_sample = int(stream.get("bits_per_sample") or 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioExtractionError(
            f"Unable to parse audio metadata for '{audio_path}': {probe}"
        ) from exc

    if duration_sec <= 0:
        raise AudioExtractionError(
            f"Extracted audio has zero duration: '{audio_path}'. "
            "Source may not contain a valid audio track."
        )

    return AudioExtractionResult(
        audio_path=audio_path,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        file_size_bytes=audio_path.stat().st_size,
        source_video=source_video,
    )


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
) -> AudioExtractionResult:
    """Extract audio from a video and normalize to WAV PCM s16le."""
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: '{video_path.resolve()}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return _build_result(output_path, video_path)

    channels = 1 if mono else 2
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]

    _run_subprocess(args)
    if not output_path.exists():
        raise AudioExtractionError(f"ffmpeg completed but no file was created: '{output_path}'.")

    return _build_result(output_path, video_path)


def extract_audio_for_chunks(
    chunk_dir: Path,
    output_dir: Path,
    sample_rate: int = 16000,
) -> Iterator[AudioExtractionResult]:
    """Extract normalized audio for all chunk files in ascending order."""
    chunk_dir = Path(chunk_dir)
    output_dir = Path(output_dir)

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: '{chunk_dir.resolve()}'")
    if not chunk_dir.is_dir():
        raise NotADirectoryError(f"Chunk path is not a directory: '{chunk_dir.resolve()}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(chunk_dir.glob("chunk_*.ts")) + sorted(chunk_dir.glob("chunk_*.mp4"))
    for chunk in sorted(chunks):
        out_path = output_dir / f"{chunk.stem}.wav"
        yield extract_audio(
            video_path=chunk,
            output_path=out_path,
            sample_rate=sample_rate,
            mono=True,
            overwrite=False,
        )


def _result_to_json_line(result: AudioExtractionResult) -> str:
    payload = asdict(result)
    payload["audio_path"] = str(result.audio_path)
    payload["source_video"] = str(result.source_video)
    return json.dumps(payload, ensure_ascii=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract normalized WAV audio from video.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Path to source video file.")
    group.add_argument("--chunk-dir", type=Path, help="Directory containing chunk_*.ts/mp4 files.")
    parser.add_argument("--output", type=Path, help="Output WAV path for --video mode.")
    parser.add_argument("--output-dir", type=Path, help="Output WAV directory for --chunk-dir mode.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file in --video mode.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.video is not None:
            if args.output is None:
                parser.error("--output is required when using --video")
            result = extract_audio(
                video_path=args.video,
                output_path=args.output,
                sample_rate=args.sample_rate,
                mono=True,
                overwrite=args.overwrite,
            )
            print(_result_to_json_line(result))
            return 0

        if args.output_dir is None:
            parser.error("--output-dir is required when using --chunk-dir")
        for result in extract_audio_for_chunks(
            chunk_dir=args.chunk_dir,
            output_dir=args.output_dir,
            sample_rate=args.sample_rate,
        ):
            print(_result_to_json_line(result))
        return 0
    except (AudioExtractionError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

