"""Google Gemini audio transcription backend."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..schema import Transcript, TranscriptSegment, Word

LOGGER = logging.getLogger(__name__)

_TRANSCRIBE_PROMPT = """\
Transcribe the audio file completely and accurately.
Return ONLY valid JSON — no markdown, no extra text — in this exact structure:
{
  "language": "<detected language code, e.g. en>",
  "duration_sec": <total duration as a float>,
  "segments": [
    {"start": <float seconds>, "end": <float seconds>, "text": "<spoken text>"},
    ...
  ]
}

Rules:
- Split into short segments of 5-15 seconds each.
- Estimate timestamps using ~2-3 words per second as a guide.
- Segments must not overlap and must cover the full audio in order.
- If the audio contains no speech, return segments as an empty array.
"""

# Gemini inline data limit (bytes). Above this, use the File API.
_INLINE_SIZE_LIMIT = 19 * 1024 * 1024  # 19 MB


class GeminiSTTBackend:
    """Transcribe audio using Gemini's multimodal audio understanding."""

    def __init__(self, model: str = "gemini-1.5-flash") -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError("google-genai package required: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self.model = model

    def transcribe(self, audio_path: Path, language: str | None = None) -> Transcript:
        audio_path = Path(audio_path)
        audio_bytes = audio_path.read_bytes()
        file_size = len(audio_bytes)

        LOGGER.info("Gemini STT: %s (%.1f MB)", audio_path.name, file_size / 1024 ** 2)

        if file_size <= _INLINE_SIZE_LIMIT:
            raw = self._transcribe_inline(audio_bytes)
        else:
            raw = self._transcribe_via_file_api(audio_path)

        return _build_transcript(raw, audio_path)

    # ------------------------------------------------------------------

    def _transcribe_inline(self, audio_bytes: bytes) -> dict:
        part = self._types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        response = self._client.models.generate_content(
            model=self.model,
            contents=[part, _TRANSCRIBE_PROMPT],
        )
        return _parse_response(getattr(response, "text", "") or "")

    def _transcribe_via_file_api(self, audio_path: Path) -> dict:
        LOGGER.info("Uploading %s via Gemini File API …", audio_path.name)
        uploaded = self._client.files.upload(
            file=str(audio_path),
            config={"mime_type": "audio/wav"},
        )
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[uploaded, _TRANSCRIBE_PROMPT],
            )
            return _parse_response(getattr(response, "text", "") or "")
        finally:
            try:
                self._client.files.delete(name=uploaded.name)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    """Strip markdown fences and parse the JSON payload."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini STT returned non-JSON output: {text[:300]}") from exc


def _build_transcript(raw: dict, audio_path: Path) -> Transcript:
    language = str(raw.get("language") or "en")
    duration_sec = float(raw.get("duration_sec") or 0.0)
    raw_segments = raw.get("segments") or []

    segments: list[TranscriptSegment] = []
    for seg in raw_segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start + 1.0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                words=[],          # Gemini doesn't provide word-level timestamps
                language=language,
                speaker=None,
            )
        )

    # Use last segment end as duration if Gemini didn't report it
    if duration_sec <= 0 and segments:
        duration_sec = segments[-1].end

    return Transcript(
        segments=segments,
        language=language,
        duration_sec=duration_sec,
        source="gemini-stt",
        audio_path=audio_path,
        raw_response=raw,
    )
