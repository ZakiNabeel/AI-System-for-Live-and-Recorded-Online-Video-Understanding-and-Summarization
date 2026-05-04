# Plan 5.1 — Master Pipeline

> **Self-contained scope.** Tie all earlier plans together into a single CLI entrypoint and an optional Streamlit UI. This plan adds no new ML — it only wires existing modules into a robust, resumable, observable pipeline. Reading only this file (plus the §1 contract list of upstream functions) is enough to implement.

---

## 1. Objective

Build `src/pipeline.py` (and a minimal `app.py` Streamlit demo) that:

1. Accepts a single URL (live or recorded) plus options.
2. Runs every stage in correct order:
   `orchestrator → audio → STT → align → frames → enhance → OCR/captions → fuse → summarize → format`
3. Records progress in `manifest.json` (Plan 1.3 schema) so a crashed run can be resumed.
4. Honors a `--skip` flag for stages already complete.
5. Provides a Streamlit UI showing live progress, frame previews, and the final summary.

This is the **MVP demo entrypoint** for the project's deliverable demo.

---

## 2. Contract

### Upstream functions consumed (stable contracts from earlier plans)
| Plan | Function |
|---|---|
| 1.3 | `orchestrator.run(url, mode, run_id)` → `RunContext` |
| 2.1 | `audio.extractor.extract_audio(video, out)` |
| 2.2 | `speech.transcriber.transcribe(audio_path or youtube_url)` |
| 2.3 | `speech.aligner.align_transcript(transcript_path, audio_path, output_dir)` |
| 3.1 | `vision.frame_extractor.extract_keyframes(video, out_dir)` |
| 3.2 | `vision.enhancer.enhance_frames(frames_json)` |
| 3.3 | `vision.extractor.extract_visual_content(manifest, out_dir)` |
| 4.1 | `fusion.fuser.fuse(transcript, visual, out, run_id=...)` |
| 4.2 | `llm.summarizer.summarize(fused, out)` |
| 4.3 | `output.formatter.format_outputs(...)` |

### Top-level function
```python
def run_pipeline(
    url: str,
    *,
    mode: Literal["auto", "recorded", "live"] = "auto",
    run_id: str | None = None,
    config_path: Path = Path("config.yaml"),
    skip: set[str] = frozenset(),       # stage names to skip
    only: set[str] | None = None,       # if set, run only these
    resume: bool = True,                # auto-skip stages already complete in manifest
    domain: str | None = None,
    enable_captions: bool = False,
    enable_qa: bool = False,
) -> PipelineResult: ...
```

### Result
```python
@dataclass
class PipelineResult:
    run_context: RunContext
    output_dir: Path
    summary_md: Path
    summary_html: Path
    report_json: Path
    timings: dict[str, float]   # stage_name -> seconds
    errors: list[StageError]    # non-fatal stage failures
```

### CLI
```
python -m src.pipeline --url "<URL>" [--mode auto] [--run-id ID]
                      [--skip frames,ocr] [--only summarize,format]
                      [--no-resume] [--captions] [--qa] [--domain education]
```

---

## 3. Stage Registry

A list-driven design makes the pipeline trivial to extend / re-order:

```python
STAGES: list[Stage] = [
    Stage("ingest",    func=_run_ingest,    inputs=[],                       outputs=["video"]),
    Stage("audio",     func=_run_audio,     inputs=["video"],                outputs=["audio"]),
    Stage("stt",       func=_run_stt,       inputs=["audio"],                outputs=["transcript"]),
    Stage("align",     func=_run_align,     inputs=["transcript", "audio"],  outputs=["aligned", "srt"]),
    Stage("frames",    func=_run_frames,    inputs=["video"],                outputs=["frames"]),
    Stage("enhance",   func=_run_enhance,   inputs=["frames"],               outputs=["enhanced"]),
    Stage("ocr",       func=_run_ocr,       inputs=["enhanced"],             outputs=["visual"]),
    Stage("fuse",      func=_run_fuse,      inputs=["aligned", "visual"],    outputs=["fused"]),
    Stage("summarize", func=_run_summarize, inputs=["fused"],                outputs=["summary"]),
    Stage("format",    func=_run_format,    inputs=["summary","aligned","fused","visual"],
                                            outputs=["markdown","html","report","chapters","report_card"]),
]
```

Each `Stage`:
```python
@dataclass
class Stage:
    name: str
    func: Callable[[StageCtx], StageResult]
    inputs: list[str]
    outputs: list[str]
```

`StageCtx` carries `RunContext`, the resolved paths, the merged config, and a logger.

---

## 4. Manifest-Driven Resume

The orchestrator (Plan 1.3) initializes `manifest.json["stages"][name] = {"status": "pending"}` for each stage. The pipeline:

1. **Before** running a stage, checks `manifest.stages[name].status`.
   - `complete` → if `resume=True`, skip; else run and overwrite.
   - `failed` or `pending` → run.
2. **After** a stage:
   - On success: `update_manifest(stages.<name> = {status:"complete", completed_at:..., outputs:{...}})`.
   - On failure: `status:"failed", error: "...", traceback:"..."`.

Atomic update via `manifest.update_manifest` (already implemented in Plan 1.3).

---

## 5. Live-Mode Special Handling

In live mode, ingestion never "completes" until the user stops it. Pipeline behavior:

1. `ingest` stage starts the live chunker (Plan 1.2) and **returns a handle** immediately.
2. A background watcher processes each completed chunk through stages 2 (`audio`), 3 (`stt`), 5 (`frames`), 6 (`enhance`), 7 (`ocr`), 8 (`fuse`) per-chunk, appending to running JSON files.
3. Stage 9 (`summarize`) runs in **rolling mode** (Plan 4.2 §5 Phase F), regenerating the global summary every `rolling_interval_sec` (default 30 s).
4. Stage 10 (`format`) regenerates `summary.md`/`summary.html` whenever the rolling summary updates.
5. On `Ctrl+C`: stop the chunker, do one final pass on the trailing chunk, then run `summarize` and `format` once more for the final state.

Implement live mode in `src/pipeline_live.py` as a separate module to keep the recorded path simple.

---

## 6. Phased Implementation

### Phase A — Stage skeleton + dispatcher (~1 hr 30 min)
1. Create `src/pipeline.py`, `src/stages/` (one file per stage wrapper).
2. Each `_run_<stage>` is a 5–15-line wrapper that loads inputs from `RunPaths`, calls the upstream function, and persists outputs to fixed locations.
3. Dispatcher:
   ```python
   for stage in STAGES:
       if stage.name in skip: log skip; continue
       if only and stage.name not in only: continue
       if resume and manifest.stages[stage.name].status == "complete": continue
       try:
           t0 = time.time()
           stage.func(ctx)
           timings[stage.name] = time.time() - t0
           manifest.update(stage="complete")
       except Exception as e:
           manifest.update(stage="failed", error=str(e))
           if not stage.optional: raise
   ```

### Phase B — Logging & progress (~1 hr)
- One log line per stage start/end with elapsed time.
- Use `tqdm` for inner loops (per-frame OCR, per-chunk audio).
- Optional `--quiet` flag for CI.

### Phase C — CLI argparse (~30 min)
Standard. Print final summary path on success.

### Phase D — Live mode (~3 hr)
1. Implement `pipeline_live.py` with a producer-consumer queue.
   - Producer: `start_live_capture` callback enqueues `(chunk_index, chunk_path)`.
   - Consumer thread pool: dequeues and runs the per-chunk pipeline.
   - Periodic timer triggers `summarize` (rolling) + `format`.
2. Provide a single entry function `run_live_pipeline(url, run_id, ...)`.

### Phase E — Streamlit UI (~3 hr)
File `app.py` at project root:

```python
import streamlit as st
from src.pipeline import run_pipeline

st.title("DIP Video Understanding")
url = st.text_input("YouTube URL or live stream")
mode = st.selectbox("Mode", ["auto", "recorded", "live"])
if st.button("Run"):
    progress = st.progress(0.0)
    status = st.empty()
    for stage_name, fraction in run_pipeline_streaming(url, mode):
        progress.progress(fraction); status.text(f"Running {stage_name}…")
    st.markdown(open(report_path).read())
```

Add `run_pipeline_streaming` as a generator wrapper around `run_pipeline` that yields after each stage. For live mode, add a "Stop" button that calls `handle.stop()`.

### Phase F — End-to-end smoke test (~1 hr)
A real test case. Run on a 30 s public CC-licensed YouTube clip and assert all outputs exist:

```python
@pytest.mark.e2e
def test_full_pipeline(tmp_path):
    result = run_pipeline(url=CANARY_URL, run_id="test-e2e", mode="recorded")
    assert result.summary_md.exists()
    assert result.summary_html.exists()
    assert result.report_json.exists()
    assert (tmp_path / "data" / "output" / "test-e2e" / "chapters.txt").exists()
```

Mark `@pytest.mark.e2e`; document that this requires API keys + network.

### Phase G — Resume tests (~1 hr)
1. Run pipeline; halfway through, kill the process (or simulate by raising in `_run_ocr`).
2. Re-run with `resume=True`; assert that `ingest`, `audio`, `stt`, `align`, `frames`, `enhance` are skipped and only `ocr` onwards run.
3. Re-run with `resume=False, only={"summarize","format"}`; assert only those run.

### Phase H — Performance reporting (~30 min)
At the end, write `performance_report.json` with timings per stage and totals. Used by the project's "Performance Evaluation Report" deliverable.

---

## 7. File Layout After Plan 5.1
```
src/
  pipeline.py
  pipeline_live.py
  stages/
    __init__.py
    ingest.py
    audio.py
    stt.py
    align.py
    frames.py
    enhance.py
    ocr.py
    fuse.py
    summarize.py
    format.py
app.py                  # Streamlit UI at project root
tests/
  test_pipeline.py
  test_pipeline_resume.py
  test_pipeline_live.py
  test_pipeline_e2e.py
```

---

## 8. Dependencies
```
streamlit>=1.36.0
tqdm>=4.66.4
```

---

## 9. Acceptance Criteria

- [ ] `python -m src.pipeline --url <recorded_url>` runs end-to-end and produces all five outputs from Plan 4.3.
- [ ] `--skip frames,ocr` skips those stages; downstream stages still succeed if their inputs (or sensible defaults) exist.
- [ ] `--only summarize,format` runs only those (assuming earlier outputs exist on disk).
- [ ] Killing mid-run and re-running picks up exactly where it stopped.
- [ ] Live mode runs without crashing for at least 5 minutes; final outputs are produced after `Ctrl+C`.
- [ ] Streamlit UI launches with `streamlit run app.py` and processes a short YouTube video successfully.
- [ ] `performance_report.json` records non-zero timings for every executed stage.

---

## 10. Edge Cases & Pitfalls

1. **Stage failures in live mode** — must not kill the whole pipeline. Log error on the offending chunk, mark that chunk failed in manifest, continue with next chunk.
2. **YouTube subtitle short-circuit** — when Plan 2.2 returns subs from YouTube, the `audio` stage is unnecessary. Detect this and mark `audio.status = "skipped"` rather than failed.
3. **Very small / very short videos** — some stages produce empty intermediates (e.g., 0 frames extracted). Downstream stages must not crash on empty inputs (covered in earlier plans, verify here).
4. **Concurrent runs on the same machine** — each uses its own `run_id`; no shared state. But two `pip` installs can clash; document that `pip install` should be done once.
5. **Streamlit blocking** — long stages block the UI. Either use `st.status` with `expanded=True` and call `run_pipeline_streaming`, or run pipeline in a thread and poll status from manifest.
6. **Out-of-order chunk completion in live mode** — ffmpeg always writes chunks in order, but processing may finish out of order if pool size > 1. Process strictly in order per stage, or (better) keep a "pending fuse" queue keyed by chunk index.
7. **Disk usage growth** — auto-cleanup is intentionally not implemented — preserve all artifacts for grading. Document this in README.
8. **Cost** — log estimated LLM and API costs at end of run.
9. **Windows process group cleanup** — make sure `Ctrl+C` in Streamlit cleanly stops the live chunker (Plan 1.2 already supports this; ensure our wrapper passes the signal through).

---

## 11. Out of Scope

- Distributed execution / queue (Celery, RQ).
- Cloud deployment.
- Authentication / multi-user.

---

## 12. Definition of Done

A developer who has implemented Plans 1.1–4.3 can run `python -m src.pipeline --url <URL>` (recorded) or `streamlit run app.py` (UI) and produce the full set of deliverables. A second invocation with the same `run_id` after a crash resumes correctly — using only this plan file as guidance.
