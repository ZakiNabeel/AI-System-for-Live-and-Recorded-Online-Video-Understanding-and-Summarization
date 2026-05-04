"""OpenAI Whisper API backend."""

from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

from ..errors import AudioTooShortError, BackendUnavailableError, TranscriptionError
from ..schema import Transcript, TranscriptSegment, Word

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SEGMENT_SECONDS = 600  # 10 min keeps 16k mono PCM safely under 25 MB.


def _as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    if isinstance(response, dict):
        return response
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    raise TranscriptionError("Unexpected OpenAI response type.")


def _audio_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0
            return float(frames) / float(rate)
    except (wave.Error, EOFError, OSError) as exc:
        raise TranscriptionError(f"Unable to read WAV duration for '{path}'.") from exc


class OpenAIWhisperBackend:
    def __init__(self, model: str = "whisper-1", client: Any | None = None):
        self.model = model
        if client is not None:
            self.client = client
            return
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailableError("openai package is not installed.") from exc
        self.client = OpenAI()

    def _slice_audio_with_ffmpeg(self, audio_path: Path, output_dir: Path) -> list[Path]:
        output_pattern = output_dir / "part_%03d.wav"
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-f",
            "segment",
            "-segment_time",
            str(SEGMENT_SECONDS),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_pattern),
        ]
        try:
            subprocess.run(args, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise TranscriptionError("ffmpeg binary not found for OpenAI chunk slicing.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise TranscriptionError(f"ffmpeg slicing failed: {detail}") from exc

        parts = sorted(output_dir.glob("part_*.wav"))
        if not parts:
            raise TranscriptionError("ffmpeg produced no chunk files for large audio.")
        return parts

    def _transcribe_file(self, path: Path, language: str | None) -> dict[str, Any]:
        with path.open("rb") as handle:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=handle,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        return _as_dict(response)

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: '{audio_path.resolve()}'")

        total_duration = _audio_duration_seconds(audio_path)
        if total_duration < 1.0:
            raise AudioTooShortError("Audio is shorter than 1 second.")

        raw_chunks: list[dict[str, Any]] = []
        merged_segments: list[TranscriptSegment] = []
        detected_language = language or "unknown"

        if audio_path.stat().st_size <= MAX_UPLOAD_BYTES:
            chunk_paths = [audio_path]
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="openai-whisper-chunks-"))
            chunk_paths = self._slice_audio_with_ffmpeg(audio_path, temp_dir)

        offset = 0.0
        try:
            for chunk_path in chunk_paths:
                chunk_duration = _audio_duration_seconds(chunk_path)
                payload = self._transcribe_file(chunk_path, language)
                raw_chunks.append(payload)

                chunk_language = payload.get("language")
                if isinstance(chunk_language, str) and chunk_language:
                    detected_language = chunk_language

                words_payload = payload.get("words") or []
                global_words: list[Word] = []
                for item in words_payload:
                    text = str(item.get("word", "")).strip()
                    global_words.append(
                        Word(
                            start=float(item.get("start", 0.0)) + offset,
                            end=float(item.get("end", 0.0)) + offset,
                            text=text,
                            confidence=math.nan,
                        )
                    )

                for seg in payload.get("segments") or []:
                    seg_start = float(seg.get("start", 0.0)) + offset
                    seg_end = float(seg.get("end", seg_start)) + offset

                    seg_words_payload = seg.get("words")
                    if seg_words_payload:
                        seg_words = [
                            Word(
                                start=float(word.get("start", 0.0)) + offset,
                                end=float(word.get("end", 0.0)) + offset,
                                text=str(word.get("word", "")).strip(),
                                confidence=math.nan,
                            )
                            for word in seg_words_payload
                        ]
                    else:
                        seg_words = [
                            word
                            for word in global_words
                            if word.start >= seg_start - 1e-9 and word.end <= seg_end + 1e-9
                        ]

                    merged_segments.append(
                        TranscriptSegment(
                            start=seg_start,
                            end=seg_end,
                            text=str(seg.get("text", "")).strip(),
                            words=seg_words,
                            language=detected_language,
                        )
                    )

                offset += chunk_duration
        finally:
            if chunk_paths and chunk_paths[0] != audio_path:
                for part in chunk_paths:
                    part.unlink(missing_ok=True)
                try:
                    chunk_paths[0].parent.rmdir()
                except OSError:
                    pass

        duration_sec = merged_segments[-1].end if merged_segments else total_duration
        return Transcript(
            segments=merged_segments,
            language=detected_language,
            duration_sec=duration_sec,
            source="openai-whisper",
            audio_path=audio_path,
            raw_response={"chunks": raw_chunks},
        )

