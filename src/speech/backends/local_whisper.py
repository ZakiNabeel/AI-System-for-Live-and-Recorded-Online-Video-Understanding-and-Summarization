"""Local faster-whisper backend."""

from __future__ import annotations

from pathlib import Path

from ..errors import BackendUnavailableError
from ..schema import Transcript, TranscriptSegment, Word


class LocalWhisperBackend:
    def __init__(self, model_name: str = "small.en", device: str = "auto", compute_type: str = "auto"):
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailableError(
                "faster-whisper is not installed. Install faster-whisper>=1.0.3."
            ) from exc

        if device == "auto":
            try:
                import torch  # type: ignore[import-not-found]

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        info_language = getattr(info, "language", None) or language or "unknown"
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        segments: list[TranscriptSegment] = []
        for seg in segments_iter:
            words = [
                Word(
                    start=float(word.start),
                    end=float(word.end),
                    text=str(getattr(word, "word", "")).strip(),
                    confidence=float(getattr(word, "probability", 0.0)),
                )
                for word in (getattr(seg, "words", None) or [])
            ]
            segments.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=str(seg.text).strip(),
                    words=words,
                    language=info_language,
                )
            )

        return Transcript(
            segments=segments,
            language=info_language,
            duration_sec=duration if duration > 0 else (segments[-1].end if segments else 0.0),
            source="local-whisper",
            audio_path=Path(audio_path),
            raw_response=None,
        )

