"""Transcript alignment pipeline and CLI."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

from .aligner_steps import (
    build_sentences,
    drop_hallucinations,
    fix_gaps_and_overlaps,
    merge_tiny_segments,
    polish_text,
    split_overlong_segments,
)
from .forced_align import ForcedAlignmentNotInstalledError, forced_realign
from .schema import AlignedTranscript, Transcript, load_transcript, save_transcript
from .subtitles import write_srt, write_vtt


@dataclass(frozen=True)
class AlignedTranscriptResult:
    aligned_json: Path
    srt: Path
    vtt: Path
    sentence_count: int
    word_count: int


def _clean_segments(
    segments,
    *,
    min_segment_sec: float,
    max_segment_sec: float,
):
    cleaned = drop_hallucinations(segments)
    cleaned = fix_gaps_and_overlaps(cleaned)
    cleaned = merge_tiny_segments(cleaned, min_segment_sec)
    cleaned = fix_gaps_and_overlaps(cleaned)
    cleaned = split_overlong_segments(cleaned, max_segment_sec)
    cleaned = fix_gaps_and_overlaps(cleaned)
    return [
        replace(
            segment,
            text=polish_text(segment.text),
            words=list(segment.words),
        )
        for segment in cleaned
    ]


def align_transcript(
    transcript_path: Path,
    *,
    audio_path: Path | None = None,
    output_dir: Path,
    mode: Literal["clean", "forced"] = "clean",
    min_segment_sec: float = 1.0,
    max_segment_sec: float = 12.0,
) -> AlignedTranscriptResult:
    transcript = load_transcript(transcript_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "forced":
        resolved_audio_path = audio_path or transcript.audio_path
        if resolved_audio_path is None:
            raise ValueError("audio_path is required when mode='forced'.")
        transcript = forced_realign(transcript, resolved_audio_path)
    elif mode != "clean":
        raise ValueError(f"Unsupported mode: {mode}")

    cleaned_segments = _clean_segments(
        transcript.segments,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )
    sentences = build_sentences(cleaned_segments, transcript.language)

    aligned = AlignedTranscript(
        segments=cleaned_segments,
        language=transcript.language,
        duration_sec=cleaned_segments[-1].end if cleaned_segments else transcript.duration_sec,
        source=transcript.source,
        audio_path=transcript.audio_path,
        raw_response=transcript.raw_response,
        sentences=sentences,
    )

    aligned_json = output_dir / "transcript.aligned.json"
    srt_path = output_dir / "transcript.srt"
    vtt_path = output_dir / "transcript.vtt"

    save_transcript(aligned, aligned_json)
    write_srt(sentences, srt_path)
    write_vtt(sentences, vtt_path)

    return AlignedTranscriptResult(
        aligned_json=aligned_json,
        srt=srt_path,
        vtt=vtt_path,
        sentence_count=len(sentences),
        word_count=sum(len(segment.words) for segment in cleaned_segments),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align transcript timestamps and write subtitles.")
    parser.add_argument("--in", dest="input_path", type=Path, required=True, help="Input transcript JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for aligned outputs.")
    parser.add_argument("--audio", type=Path, default=None, help="Audio file for forced alignment.")
    parser.add_argument("--mode", choices=["clean", "forced"], default="clean")
    parser.add_argument("--min-segment-sec", type=float, default=1.0)
    parser.add_argument("--max-segment-sec", type=float, default=12.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    started = time.perf_counter()
    try:
        result = align_transcript(
            args.input_path,
            audio_path=args.audio,
            output_dir=args.out_dir,
            mode=args.mode,
            min_segment_sec=args.min_segment_sec,
            max_segment_sec=args.max_segment_sec,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "aligned_json": str(result.aligned_json),
                    "srt": str(result.srt),
                    "vtt": str(result.vtt),
                    "sentence_count": result.sentence_count,
                    "word_count": result.word_count,
                    "elapsed_sec": round(elapsed, 3),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except ForcedAlignmentNotInstalledError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
