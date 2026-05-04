# Plan 4.1 — Data Fusion

> **Self-contained scope.** Merge the aligned transcript (Plan 2.3) and the visual extraction (Plan 3.3) into a single, time-ordered, LLM-ready JSON payload. This module is **pure data transformation** — no models, no I/O beyond JSON read/write — and has the cleanest contract in the project.

---

## 1. Objective

Build `src/fusion/fuser.py` that:

1. Reads two JSON files: `transcript.aligned.json` and `visual.json`.
2. Produces `fused.json` — a chronologically ordered list of multimodal events.
3. Optionally produces `fused_chunked.json` — the same content split into LLM-context-sized chunks (e.g., 8 K, 32 K tokens) for downstream summarization.

This is the deterministic glue between the audio and visual halves of the pipeline. It must be reproducible (same inputs → byte-identical output).

---

## 2. Contract

### Output dataclasses
```python
@dataclass
class FusedEvent:
    """A single timestamped multimodal event."""
    t_start: float
    t_end: float
    kind: Literal["speech", "visual", "speech+visual"]
    speech_text: str | None         # the sentence(s) spoken in this window
    speech_segment_indices: list[int]   # indices into transcript.sentences
    visual_text: str | None         # OCR text of the nearest key-frame
    visual_caption: str | None      # vision-LM caption (if present)
    frame_index: int | None
    frame_path: str | None          # relative path
    notes: list[str]                # any qualitative tags ("scene-change", "long-pause")

@dataclass
class FusedDocument:
    run_id: str
    duration_sec: float
    language: str
    events: list[FusedEvent]
    speech_source: str              # echo of Transcript.source
    ocr_engine: str
    has_captions: bool
```

### Top-level function
```python
def fuse(
    transcript_path: Path,
    visual_path: Path,
    output_path: Path,
    *,
    run_id: str,
    window_sec: float = 5.0,        # event granularity
    max_chunk_tokens: int = 8000,
    chunk_overlap_tokens: int = 200,
) -> FusedDocument: ...
```

### CLI
```
python -m src.fusion.fuser \
    --transcript transcript.aligned.json \
    --visual visual.json \
    --out fused.json \
    --run-id <id>
```

---

## 3. The Fusion Algorithm

### Step 1 — Build a time-windowed grid
Slice the timeline `[0, duration]` into windows of `window_sec` seconds. For 5 s windows on a 600 s video, you get 120 windows.

### Step 2 — Bucket sentences into windows
For each `Sentence` in `transcript.sentences`:
- Compute the windows it overlaps: `floor(s.start / W)` to `floor(s.end / W)`.
- Append the sentence to each overlapped window's bucket.

### Step 3 — Bucket frames into windows
For each `FrameVisual`:
- It belongs to window `floor(frame.timestamp / W)`.
- A window may have 0, 1, or many frames. If many, pick the one with the most OCR text (`len(frame.text)`).

### Step 4 — Emit events
For each window in order:
```python
sentences = window.speech
frame     = window.best_frame   # may be None
if not sentences and not frame:
    continue                    # silent + no visual change → skip
event = FusedEvent(
    t_start=window.start, t_end=window.end,
    kind=("speech+visual" if sentences and frame else
          "speech" if sentences else "visual"),
    speech_text=" ".join(s.text for s in sentences) or None,
    speech_segment_indices=[s.index for s in sentences],
    visual_text=frame.text if frame else None,
    visual_caption=frame.caption if frame else None,
    frame_index=frame.frame_index if frame else None,
    frame_path=str(Path(frame.frame_path).relative_to(...)) if frame else None,
    notes=_tag_notes(window),
)
events.append(event)
```

### Step 5 — Add qualitative notes
Tag windows with helpful flags downstream LLMs can use:
- `"long-pause"` — gap of > 8 s between consecutive non-empty windows.
- `"scene-change"` — frame whose `reason == "ssim-drop"` AND there is also speech in this window.
- `"silent-visual"` — visual but no speech (e.g., a slide change with no narration).

### Step 6 — Chunking (optional secondary output)
For very long videos, the LLM cannot ingest the entire fused doc. Produce `fused_chunked.json`:

```python
def chunk_events(events, max_tokens, overlap_tokens) -> list[FusedDocument]:
    # Use a tokenizer (tiktoken for OpenAI, or anthropic.Anthropic().count_tokens()).
    # Greedy fill: add events to current chunk until adding next would exceed max.
    # When rolling over, include the trailing `overlap_tokens` of text in the new chunk.
```

Each chunk is a self-contained mini-`FusedDocument` with its own time range and continuous `events` slice.

---

## 4. Phased Implementation

### Phase A — Skeleton + dataclasses (~30 min)
Create `src/fusion/__init__.py`, `src/fusion/fuser.py`, `src/fusion/schema.py`. Define dataclasses + JSON serialization helpers.

### Phase B — Window builder (~45 min)
```python
def make_windows(duration, window_sec) -> list[tuple[float, float]]:
    n = math.ceil(duration / window_sec)
    return [(i * window_sec, min((i + 1) * window_sec, duration))
            for i in range(n)]
```

### Phase C — Bucket helpers (~45 min)
```python
def bucket_sentences(sentences, window_sec):
    buckets = defaultdict(list)
    for s in sentences:
        i_start = int(s.start // window_sec)
        i_end   = int(s.end   // window_sec)
        for i in range(i_start, i_end + 1):
            buckets[i].append(s)
    return buckets

def bucket_frames(frames, window_sec):
    buckets = defaultdict(list)
    for f in frames:
        i = int(f.timestamp // window_sec)
        buckets[i].append(f)
    return buckets

def best_frame(frames):
    if not frames: return None
    return max(frames, key=lambda f: (len(f.text), 1 if f.caption else 0))
```

### Phase D — Event emitter (~45 min)
The function that walks windows in order and produces `FusedEvent`s per §3 Step 4.

### Phase E — Note tagger (~30 min)
Pure functions in `src/fusion/notes.py`. Each takes window data and returns `list[str]`.

### Phase F — Chunker (~1 hr)
Implement `chunk_events`. Use **anthropic** tokenizer if Plan 4.2 chose Claude; otherwise `tiktoken`. Detect via config.

### Phase G — CLI + main fuse() (~30 min)
Compose the full pipeline. Sort events by `t_start`. Write JSON.

### Phase H — Tests (~1 hr 30 min)

Tests are easy here because the function is deterministic and pure.

1. **Empty inputs** — empty transcript + empty visual → `events == []`. Don't crash.
2. **Speech only** — transcript with two sentences, no visual; assert two `kind="speech"` events.
3. **Visual only** — frames with OCR but no transcript; assert `kind="visual"` events.
4. **Overlap** — sentence spans two windows; assert it appears in both event buckets.
5. **Best-frame selection** — two frames in same window, one with more OCR text; assert the longer is chosen.
6. **Note tagging** — 12 s gap between two non-empty windows; assert the second one has `"long-pause"` note.
7. **Chunking math** — fake event list summing to 20 K tokens with `max_chunk_tokens=8000`; assert ≥ 3 chunks; assert no event is duplicated except in the overlap region.
8. **Determinism** — call `fuse()` twice with same inputs; compare JSON outputs byte-for-byte.

---

## 5. File Layout After Plan 4.1
```
src/fusion/
  __init__.py
  fuser.py
  schema.py
  notes.py
  chunker.py
  tokens.py             # token counting helpers
tests/fusion/
  test_fuser.py
  test_chunker.py
  test_notes.py
```

---

## 6. Dependencies

| Package | Purpose |
|---|---|
| `tiktoken` | Token counting (OpenAI tokenizer) |
| `anthropic` | Already installed; provides `count_tokens` |

Add:
```
tiktoken>=0.7.0
```

---

## 7. Acceptance Criteria

- [ ] CLI run produces `fused.json` validating against the schema in §2.
- [ ] Events are sorted by `t_start` and have non-decreasing `t_end`.
- [ ] Every sentence and every frame appears in at least one event (or is intentionally dropped — log how many).
- [ ] Output is byte-identical between runs on the same input.
- [ ] Chunked variant respects `max_chunk_tokens` (no chunk exceeds it).
- [ ] All unit tests pass without any external API or model.

---

## 8. Edge Cases & Pitfalls

1. **Transcript and visual have different durations** — use `max(transcript.duration_sec, max(frame.timestamp))`. Don't truncate either.
2. **Sentence with `end < start`** — log warning, skip.
3. **Frame timestamp beyond transcript duration** — keep it; emit a `kind="visual"` event past the speech end.
4. **Window straddles end of video** — last window is clamped to actual duration.
5. **Tokenizer mismatch** — if user is using Claude downstream but counter uses `tiktoken`, chunk sizes will be inaccurate. Fall back gracefully; document this in the chunker docstring.
6. **Path serialization** — store frame paths *relative to the project root*, not absolute, so the JSON is portable across machines.
7. **Very dense visual content (100 s of frames in a 5 s window)** — `best_frame` is O(n) and fine, but loading them all into memory is unnecessary. Process in a single sweep.
8. **Time precision** — round all timestamps to 3 decimal places in JSON output for consistent diffing.

---

## 9. Out of Scope

- Any LLM call (that's Plan 4.2).
- Embedding-based semantic chunking (chunker is purely length-based).
- Cross-modal alignment beyond windowing (no fancy "find the slide that matches what the speaker just said" — keep it simple; the LLM does that reasoning in the next step).

---

## 10. Definition of Done

A developer can take Plan 2.3's `transcript.aligned.json` and Plan 3.3's `visual.json`, run `python -m src.fusion.fuser ...`, and get a deterministic `fused.json` ready to feed into the LLM step — using only this plan file.
