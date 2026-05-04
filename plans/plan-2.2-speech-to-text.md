# Plan 2.2 — Speech-to-Text

> **Self-contained scope.** Convert a 16 kHz mono WAV file (the output contract of Plan 2.1) into a structured transcript with word-level timestamps. The plan supports two backends — local `faster-whisper` and the cloud OpenAI Whisper API — selectable by config. No knowledge of the rest of the pipeline is required.

---

## 1. Objective

Build `src/speech/transcriber.py` that:

1. Accepts a path to a WAV (and optionally a YouTube URL for direct subtitle pull).
2. Produces a `Transcript` object with segments and word-level timestamps.
3. Supports two backends behind a common interface:
   - `local-whisper` — `faster-whisper`, runs on CPU/GPU offline (no API key).
   - `openai-whisper` — cloud API, `OPENAI_API_KEY` required.
4. (Bonus path) If a YouTube URL is provided and YouTube auto/uploaded subtitles exist, pull those instead — much faster and free. Falls back to model transcription otherwise.

Plan 2.3 then post-processes this transcript to align/clean timestamps. This plan only needs to produce a *correct, structured* transcript.

---

## 2. Contract

### Core dataclasses
```python
@dataclass
class Word:
    start: float        # seconds
    end: float          # seconds
    text: str
    confidence: float   # 0..1, or NaN if unknown

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: list[Word]
    language: str       # ISO 639-1, e.g. "en"
    speaker: str | None = None   # optional, populated by diarization later

@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str
    duration_sec: float
    source: Literal["local-whisper", "openai-whisper", "youtube-subtitles"]
    audio_path: Path | None
    raw_response: dict | None    # full backend response for debugging
```

### Top-level function
```python
def transcribe(
    audio_path: Path | None = None,
    *,
    youtube_url: str | None = None,
    backend: Literal["auto", "local-whisper", "openai-whisper"] = "auto",
    model: str = "small.en",
    language: str | None = None,    # None = auto-detect
    use_youtube_subs_if_available: bool = True,
) -> Transcript: ...
```

Rules:
- Exactly one of `audio_path` / `youtube_url` must be provided (or both: subs first, fallback to audio).
- `backend == "auto"` resolves to `local-whisper` if `faster-whisper` is importable, else `openai-whisper` if `OPENAI_API_KEY` is set, else raises.

### Persisting a transcript
```python
def save_transcript(transcript: Transcript, path: Path) -> None: ...
def load_transcript(path: Path) -> Transcript: ...
```
On disk format: JSON, schema in §6.

### CLI
```
python -m src.speech.transcriber --audio <wav> --out <json>
python -m src.speech.transcriber --youtube <url> --out <json>
```

---

## 3. Dependencies

| Backend | Packages |
|---|---|
| Local | `faster-whisper>=1.0.3` (uses CTranslate2 — much faster than openai-whisper) |
| Cloud | `openai>=1.40.0` |
| YouTube subs | `youtube-transcript-api>=0.6.2` |
| All | `numpy`, `soundfile` (for length probing) |

Add to `requirements.txt`:
```
faster-whisper>=1.0.3
openai>=1.40.0
youtube-transcript-api>=0.6.2
soundfile>=0.12.1
```

GPU acceleration: `faster-whisper` will auto-detect CUDA. On CPU, use `compute_type="int8"` for ~3x speedup.

---

## 4. Phased Implementation

### Phase A — Backend interface (~30 min)
```python
class TranscriberBackend(Protocol):
    def transcribe(self, audio_path: Path, language: str | None) -> Transcript: ...
```

### Phase B — Local Whisper backend (~1 hr 30 min)
File `src/speech/backends/local_whisper.py`:

```python
class LocalWhisperBackend:
    def __init__(self, model_name="small.en", device="auto", compute_type="auto"):
        from faster_whisper import WhisperModel
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language=None):
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,            # skip silence
            vad_parameters={"min_silence_duration_ms": 500},
        )
        out_segments = []
        for s in segments:                          # generator!
            words = [Word(w.start, w.end, w.word, w.probability) for w in (s.words or [])]
            out_segments.append(TranscriptSegment(
                start=s.start, end=s.end, text=s.text.strip(),
                words=words, language=info.language,
            ))
        return Transcript(
            segments=out_segments, language=info.language,
            duration_sec=info.duration,
            source="local-whisper", audio_path=audio_path,
            raw_response=None,
        )
```

Key points:
- `faster-whisper`'s `transcribe` returns a **generator** — must iterate to populate.
- `word_timestamps=True` is required for downstream Plan 2.3.
- `vad_filter=True` cuts silent gaps (massive speedup on lectures with long pauses).

Available local models (pick via `--model`):
| Model | Size on disk | English-only | Use case |
|---|---|---|---|
| `tiny.en` | 39 MB | yes | Fastest, lower accuracy |
| `base.en` | 74 MB | yes | Decent for clean speech |
| `small.en` | 244 MB | yes | **Default** — good balance |
| `medium.en` | 769 MB | yes | Better for noisy audio |
| `large-v3` | 1.55 GB | no | Best, multilingual, slow on CPU |

### Phase C — OpenAI Whisper API backend (~45 min)
File `src/speech/backends/openai_api.py`:

```python
class OpenAIWhisperBackend:
    def __init__(self, model="whisper-1"):
        from openai import OpenAI
        self.client = OpenAI()   # picks up OPENAI_API_KEY
        self.model = model

    def transcribe(self, audio_path, language=None):
        with open(audio_path, "rb") as f:
            resp = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        # resp.segments: list of {start, end, text, ...}
        # resp.words:    list of {start, end, word}
        ...
```

Rules:
- Cloud API rejects files > 25 MB. **Pre-check** size; if too big, slice the WAV with ffmpeg into ≤ 24-min chunks (16 kHz mono s16 ≈ 1 MB / minute), transcribe each, then stitch with offset adjustment.
- API has no `confidence` per word; set `confidence = math.nan`.

### Phase D — YouTube subtitles backend (~45 min)
File `src/speech/backends/youtube_subs.py`:

```python
def fetch_youtube_transcript(url, languages=("en",)) -> Transcript | None:
    video_id = _parse_video_id(url)
    try:
        api = YouTubeTranscriptApi()
        items = api.fetch(video_id, languages=list(languages)).snippets
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    segments = [
        TranscriptSegment(
            start=item.start, end=item.start + item.duration,
            text=item.text, words=[],   # YouTube does not give word timing
            language=languages[0],
        )
        for item in items
    ]
    return Transcript(segments=segments, language=languages[0],
                      duration_sec=segments[-1].end if segments else 0,
                      source="youtube-subtitles", audio_path=None,
                      raw_response={"items": [...]})
```

Note: YouTube subs don't have word-level timing; downstream Plan 2.3 must detect this and either skip word-level features or re-time with forced alignment (out of scope here — flag in the segment as `words == []`).

### Phase E — Top-level dispatcher + auto-fallback (~45 min)
```python
def transcribe(audio_path=None, *, youtube_url=None, backend="auto",
               model="small.en", language=None,
               use_youtube_subs_if_available=True):
    # 1) Try YouTube subs first if requested
    if youtube_url and use_youtube_subs_if_available:
        sub_transcript = fetch_youtube_transcript(youtube_url)
        if sub_transcript is not None:
            return sub_transcript
    # 2) Need an audio file from here on
    if audio_path is None:
        raise ValueError("audio_path required when YouTube subs unavailable")
    # 3) Resolve backend
    backend = _resolve_backend(backend)
    if backend == "local-whisper":
        return LocalWhisperBackend(model).transcribe(audio_path, language)
    if backend == "openai-whisper":
        return OpenAIWhisperBackend("whisper-1").transcribe(audio_path, language)
    raise ValueError(backend)
```

### Phase F — Save / load (~30 min)
JSON schema (one file, ~1–10 MB for hour-long video):
```json
{
  "version": "1",
  "language": "en",
  "duration_sec": 312.5,
  "source": "local-whisper",
  "audio_path": "data/audio/<id>/audio.wav",
  "segments": [
    {
      "start": 0.0, "end": 4.32, "text": "Hello and welcome.",
      "language": "en", "speaker": null,
      "words": [
        {"start": 0.0, "end": 0.4, "text": "Hello", "confidence": 0.99},
        ...
      ]
    }
  ]
}
```

Use `dataclasses.asdict` + a `default=str` JSON encoder.

### Phase G — CLI (~30 min)
- Mutually exclusive: `--audio` xor `--youtube`.
- `--backend`, `--model`, `--language`, `--out`.
- On success, also emit a tiny stats line to stderr: `[INFO] transcribed 312.5 s in 41.2 s, 754 words`.

### Phase H — Tests (~2 hr)

1. **Schema round-trip** — build a tiny `Transcript` object, save, reload, assert deep equality.
2. **Auto backend resolution** — patch `importlib` to make `faster_whisper` unavailable; assert auto picks `openai-whisper` if env var set; else raises `NoBackendAvailableError`.
3. **YouTube subs path** — patch `YouTubeTranscriptApi` to return canned data; assert `Transcript.source == "youtube-subtitles"` and segments are constructed correctly.
4. **YouTube subs missing** — patch to raise `TranscriptsDisabled`; assert function returns `None` and dispatcher falls through to audio path.
5. **Local backend smoke** (slow, mark `@pytest.mark.slow`) — generate a 3 s WAV with TTS or a fixed pre-recorded sample saying "one two three"; assert ≥ 3 words detected, all `confidence > 0.5`, language == "en".
6. **OpenAI backend size guard** — feed a 30 MB fake WAV; assert it slices into 2 chunks, transcribes each, and stitches offsets correctly (mock the API).
7. **CLI** — invoke as subprocess against a small WAV; assert JSON file exists and validates against schema.

---

## 5. File Layout After Plan 2.2
```
src/speech/
  __init__.py
  transcriber.py
  schema.py            # Word, TranscriptSegment, Transcript + save/load
  backends/
    __init__.py
    local_whisper.py
    openai_api.py
    youtube_subs.py
  errors.py
tests/speech/
  test_transcriber.py
  test_schema.py
  test_backends_local.py
  test_backends_openai.py
  test_backends_youtube.py
  fixtures/three_words.wav
```

---

## 6. Acceptance Criteria

- [ ] `transcribe(audio_path=...)` produces a `Transcript` with non-empty segments for any speech-bearing WAV.
- [ ] Word-level timestamps are populated when using a Whisper backend (not for YouTube subs).
- [ ] `save_transcript` then `load_transcript` round-trips losslessly.
- [ ] Backend auto-resolves correctly given different environment conditions (covered by tests).
- [ ] OpenAI backend handles files > 25 MB by slicing.
- [ ] CLI works for both `--audio` and `--youtube` modes.
- [ ] All non-slow tests pass without GPU and without API keys.

---

## 7. Edge Cases & Pitfalls

1. **Hallucinated repeats on silent audio** — Whisper invents text on long silences. `vad_filter=True` mitigates; also drop segments where `len(words) == 0` and `text` is one of a known hallucination list ("you", "thanks for watching", etc.) — keep this list in `src/speech/hallucination_filter.py`.
2. **Language auto-detect on multilingual audio** — set `language` explicitly if known; auto-detect samples only the first 30 s.
3. **Model download time** — first call to `WhisperModel("small.en")` downloads ~250 MB. Document this in README; consider pre-downloading in CI.
4. **Cuda OOM on `large-v3`** — fall back to CPU with int8 if CUDA OOM detected; log a warning.
5. **API rate limits** — wrap OpenAI calls with retry (`tenacity`, exponential backoff, max 5 attempts).
6. **YouTube subs are auto-generated and noisy** — the `is_generated` flag in `youtube-transcript-api` indicates auto-captions. Prefer manual captions when both exist.
7. **Audio file shorter than 1 s** — Whisper crashes. Pre-check duration; raise `AudioTooShortError`.
8. **API filename header** — OpenAI uses the filename to guess format; ensure the file is opened in binary mode and the API call's filename ends in `.wav`.

---

## 8. Out of Scope

- Speaker diarization (could be added in a future plan; `speaker` field is reserved).
- Punctuation post-processing (Whisper already does this; further cleanup in Plan 2.3 if needed).
- Translation (`task="translate"` parameter is intentionally not exposed).

---

## 9. Definition of Done

A developer can take any 16 kHz mono WAV from Plan 2.1, run `python -m src.speech.transcriber --audio file.wav --out transcript.json`, and get a structured JSON file matching the §6 schema — using only this plan file, with either a local model or an `OPENAI_API_KEY`.
