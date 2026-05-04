# Plan 2.3 — Timestamp Alignment & Transcript Refinement

> **Self-contained scope.** Take a raw `Transcript` JSON (the output schema of Plan 2.2) and produce a *clean, sentence-segmented, well-timed* transcript suitable for downstream summarization and SRT export. This plan adds no new model — it is pure post-processing.

---

## 1. Objective

Build `src/speech/aligner.py` that:

1. Loads a `Transcript` JSON (Plan 2.2 schema).
2. Produces three output artifacts in `data/intermediate/<run_id>/`:
   - `transcript.aligned.json` — refined transcript, same schema, but with cleaner sentence boundaries, gap fixes, and merged short segments.
   - `transcript.srt` — standard SRT subtitle file.
   - `transcript.vtt` — WebVTT file.
3. Optionally (when YouTube subs were used and word-level timing is missing) approximates word-level timing by uniformly distributing words across each segment's duration.
4. Optionally re-times against the audio using a *forced alignment* tool (`stable-ts`) when configured, for highest accuracy.

This plan is the **last step before fusion (Plan 4.1)**, where speech meets visual data.

---

## 2. Contract

### Input
- `transcript_path: Path` — JSON conforming to Plan 2.2 schema.
- `audio_path: Path | None` — required only for forced-alignment mode.

### Output dataclass (extends Plan 2.2 schema)
```python
@dataclass
class AlignedTranscript(Transcript):
    sentences: list[Sentence]   # NEW — what summarizer/fusion consumes

@dataclass
class Sentence:
    start: float
    end: float
    text: str
    word_count: int
    segment_indices: list[int]   # which raw segments this sentence came from
```

### Top-level function
```python
def align_transcript(
    transcript_path: Path,
    *,
    audio_path: Path | None = None,
    output_dir: Path,
    mode: Literal["clean", "forced"] = "clean",
    min_segment_sec: float = 1.0,
    max_segment_sec: float = 12.0,
) -> AlignedTranscriptResult: ...

@dataclass
class AlignedTranscriptResult:
    aligned_json: Path
    srt: Path
    vtt: Path
    sentence_count: int
    word_count: int
```

### CLI
```
python -m src.speech.aligner --in transcript.json --out-dir data/intermediate/<id> [--mode clean|forced --audio audio.wav]
```

---

## 3. Dependencies

| Package | Purpose | Required for |
|---|---|---|
| `webvtt-py>=0.5.1` | VTT writing | always |
| `srt>=3.5.3` | SRT writing | always |
| `stable-ts>=2.17.0` | Forced re-alignment | only `mode="forced"` |
| `nltk>=3.8` or `pysbd>=0.3.4` | Sentence segmentation | always |

Recommend `pysbd` (Pragmatic Sentence Boundary Detector) — single dependency, no model download:
```
pysbd>=0.3.4
srt>=3.5.3
webvtt-py>=0.5.1
stable-ts>=2.17.0
```

---

## 4. The Cleaning Pipeline (mode="clean")

Apply, in order:

### Step 1 — Drop hallucinations
Filter segments where:
- `text.strip()` matches a known hallucination pattern (e.g., `"thanks for watching"`, `"you"` alone, `"."`).
- AND `len(words) == 0` OR all words have `confidence < 0.3`.

### Step 2 — Fix gap & overlap
For consecutive segments `s[i], s[i+1]`:
- If `s[i+1].start < s[i].end` (overlap due to model error), set `s[i+1].start = s[i].end + 0.01`.
- If `s[i+1].start - s[i].end > 5.0`, leave as-is (genuine silence). Do not "stretch."

### Step 3 — Merge tiny segments
If `s[i].end - s[i].start < min_segment_sec` AND adjacent segments exist, merge into the nearer neighbor (concatenate text, union words, take outer bounds).

### Step 4 — Split overlong segments
If `s[i].end - s[i].start > max_segment_sec`, split at the nearest sentence boundary inside `text` (using `pysbd`). Distribute words to the new segments by word `start` time.

### Step 5 — Sentence segmentation
Concatenate all segment texts (with spaces) and run `pysbd.Segmenter(language="en")`. For each sentence, compute `start = first_word.start`, `end = last_word.end`. If the source had no word-level timing, distribute uniformly across the parent segment's duration.

### Step 6 — Capitalization & punctuation polish
- Strip leading/trailing spaces.
- Fix `" ." → "."`, `" ," → ","`.
- Capitalize the first letter of each sentence if Whisper missed it.

---

## 5. Forced Alignment (mode="forced") — Optional High-Accuracy Mode

`stable-ts` re-runs Whisper with constrained word-level alignment using the original audio. This corrects timestamp drift on long audio (Whisper often shifts by several seconds at the 30-min mark).

```python
import stable_whisper
model = stable_whisper.load_faster_whisper("small.en")
result = model.transcribe_stable(audio_path, regroup=True)
# Then convert result.segments back to our Transcript schema.
```

Use this mode only when:
- The transcript came from `local-whisper` or `openai-whisper` (not YouTube subs — those need different alignment).
- The user explicitly asked for it (slow: similar to a fresh transcription).

---

## 6. SRT / VTT Writers

### SRT
```
1
00:00:00,000 --> 00:00:04,320
Hello and welcome.

2
00:00:04,400 --> 00:00:09,800
Today we'll cover three topics.
```

Use `srt.compose([srt.Subtitle(idx, td_start, td_end, text), ...])`. One subtitle per **sentence** (not per Whisper segment) — gives nicer subtitles.

Limit each cue to ≤ 84 characters / 2 lines (CEA-608 readability convention). If a sentence is longer, split with `textwrap.wrap(text, 42)` and time-divide proportionally to character count.

### VTT
Same content, written via `webvtt-py`. Header line `WEBVTT` required.

---

## 7. Phased Implementation

### Phase A — Schema extension (~20 min)
Extend Plan 2.2's `schema.py` (or create `aligned_schema.py`) with `Sentence` and `AlignedTranscript`. Make sure `save/load` round-trips both.

### Phase B — Cleaning steps (~2 hr)
Implement steps 1–6 from §4 as small pure functions in `src/speech/aligner_steps.py`. Each takes and returns a `list[TranscriptSegment]`. Easy to unit-test in isolation.

### Phase C — Sentence builder (~45 min)
```python
def build_sentences(segments: list[TranscriptSegment]) -> list[Sentence]:
    seg = pysbd.Segmenter(language="en", clean=False)
    full_text = " ".join(s.text for s in segments)
    raw_sentences = seg.segment(full_text)
    # Walk through words across all segments to find each sentence's bounds.
    ...
```

If words are missing, fall back to "evenly slice each segment by sentence character-length proportion."

### Phase D — SRT/VTT writers (~45 min)
File `src/speech/subtitles.py` with `write_srt(sentences, out_path)` and `write_vtt(sentences, out_path)`.

### Phase E — Forced-alignment backend (~1 hr, optional)
Stub if `stable-ts` not installed; clearly raise `ForcedAlignmentNotInstalledError`.

### Phase F — CLI & main (~30 min)
Compose the pipeline:
```python
def align_transcript(transcript_path, *, audio_path=None, output_dir, mode="clean", ...):
    raw = load_transcript(transcript_path)
    if mode == "forced":
        raw = forced_realign(raw, audio_path)
    cleaned = drop_hallucinations(raw.segments)
    cleaned = fix_gaps_and_overlaps(cleaned)
    cleaned = merge_tiny(cleaned, min_segment_sec)
    cleaned = split_overlong(cleaned, max_segment_sec)
    cleaned = polish_text(cleaned)
    sentences = build_sentences(cleaned)
    aligned = AlignedTranscript(segments=cleaned, sentences=sentences, ...)
    aligned_path = output_dir / "transcript.aligned.json"
    srt_path = output_dir / "transcript.srt"
    vtt_path = output_dir / "transcript.vtt"
    save_transcript(aligned, aligned_path)
    write_srt(sentences, srt_path); write_vtt(sentences, vtt_path)
    return AlignedTranscriptResult(...)
```

### Phase G — Tests (~2 hr)

1. **Each cleaning step in isolation** — feed a tiny `list[TranscriptSegment]`; assert the post-condition holds (no overlaps, no tiny segments, etc.).
2. **Sentence splitting** — give multi-sentence segment text; assert correct sentence count and that timestamps fall within parent segment bounds.
3. **YouTube-subs path (no word timing)** — verify uniform word distribution produces monotonic timestamps.
4. **SRT validity** — write a sample, then re-parse with `srt.parse`; assert round-trip equality on text and roughly equal timestamps.
5. **VTT validity** — same with `webvtt.read_buffer`.
6. **Hallucination filter** — feed segments with known hallucination strings; assert they are removed.
7. **Idempotency** — running `align_transcript` twice on the same input produces identical output (deterministic).
8. **Long-cue line wrapping** — give a 200-char sentence; assert SRT cue is split into ≤ 2 lines of ≤ 42 chars each.

---

## 8. File Layout After Plan 2.3
```
src/speech/
  aligner.py
  aligner_steps.py
  subtitles.py
  hallucination_filter.py
  forced_align.py        # optional stable-ts wrapper
tests/speech/
  test_aligner_steps.py
  test_subtitles.py
  test_aligner.py
```

---

## 9. Acceptance Criteria

- [ ] Given a Plan 2.2 JSON, produces `transcript.aligned.json`, `transcript.srt`, `transcript.vtt` in the output dir.
- [ ] No segment in the output has `start > end` or `start < previous.end`.
- [ ] Sentences are non-empty, monotonically ordered by `start`.
- [ ] SRT and VTT files validate against their parsers.
- [ ] Re-running on the same input produces byte-identical SRT (deterministic order).
- [ ] All unit tests green.
- [ ] (If `--mode forced` used) re-aligned timestamps differ from input by no more than the forced-alignment tool's spec; transcript text is preserved.

---

## 10. Edge Cases & Pitfalls

1. **Empty transcript** — produce empty SRT/VTT (with valid headers) and zero sentences. Don't crash.
2. **Single-word transcript** — sentence segmentation may produce zero sentences; fallback: treat each segment text as its own sentence.
3. **Non-English content** — `pysbd` supports several languages; pass `transcript.language` through. If unsupported, fall back to splitting on `[.!?]` regex.
4. **Word timestamps slightly out of order inside a segment** — happens with Whisper; sort by `start` before building sentences.
5. **Long single sentence with no internal punctuation** — split at max 12 s by word count proportionally; mark in metadata that this is a synthetic break.
6. **Overlap with neighboring segments after merge** — re-run `fix_gaps_and_overlaps` after merge step.
7. **SRT character encoding** — write UTF-8 with BOM for Windows Notepad compatibility (`encoding="utf-8-sig"`).

---

## 11. Out of Scope

- Speaker diarization (would integrate at this stage if added later — the `speaker` field already exists in the schema).
- Translation.
- Punctuation restoration on YouTube auto-caps (already lowercase + missing periods); document but don't fix here.

---

## 12. Definition of Done

A developer can take any Plan 2.2 transcript JSON, run `python -m src.speech.aligner --in transcript.json --out-dir out/`, and get three files (aligned JSON, SRT, VTT) — all valid and consistent — using only this plan file.
