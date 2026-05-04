# Plan 4.2 — LLM Summarization

> **Self-contained scope.** Take `fused.json` from Plan 4.1, send it to an LLM with carefully crafted prompts, and produce a structured summary JSON containing: full summary, time-stamped key points, detected events, and an optional Q&A index. This plan owns the prompt engineering and the LLM-call retry/streaming logic. It depends only on Plan 4.1's output schema.

---

## 1. Objective

Build `src/llm/summarizer.py` that:

1. Reads `fused.json`.
2. Splits it into chunks if it exceeds the chosen model's context window (use Plan 4.1's `chunker` if needed; otherwise process whole).
3. Calls the LLM **twice** for long videos:
   - **Pass 1 — per-chunk summary**: extract local key points, events, and decisions.
   - **Pass 2 — global synthesis**: combine all chunk summaries into a single coherent overview.
4. Writes `summary.raw.json` — the structured LLM output before formatting.
5. Implements rolling-summary mode for live runs: every N new events, regenerate the summary.

This plan does **not** produce the final user-facing files (Markdown, HTML). That's Plan 4.3.

---

## 2. Contract

### Output dataclass
```python
@dataclass
class KeyPoint:
    timestamp: float        # seconds (start of relevant window)
    text: str
    confidence: Literal["low", "medium", "high"]
    source_event_indices: list[int]   # indices into fused.events

@dataclass
class DetectedEvent:
    timestamp: float
    event_type: str        # free-form: "topic-change", "demo-start", "Q&A", etc.
    description: str
    source_event_indices: list[int]

@dataclass
class Summary:
    run_id: str
    full_summary: str               # 3–6 paragraphs
    short_summary: str              # 1–2 sentences (for thumbnails / TTS)
    key_points: list[KeyPoint]      # bullet list with timestamps
    events: list[DetectedEvent]
    chapters: list[Chapter]         # YouTube-style chapters (auto-generated)
    qa_pairs: list[QAPair] | None   # if --qa enabled, sample Q&A
    model: str
    provider: str
    chunked: bool
    n_chunks: int
    elapsed_sec: float
    token_usage: dict               # input/output token counts

@dataclass
class Chapter:
    t_start: float
    t_end: float
    title: str

@dataclass
class QAPair:
    question: str
    answer: str
    timestamp: float
```

### Top-level function
```python
def summarize(
    fused_path: Path,
    output_path: Path,
    *,
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic",
    model: str | None = None,        # default chosen per provider below
    style: Literal["concise", "detailed", "bullet-only"] = "detailed",
    enable_qa: bool = False,
    domain: str | None = None,       # e.g., "education", "trading" — see Plan 5.2
    rolling: bool = False,
    rolling_state_path: Path | None = None,
) -> Summary: ...
```

### Default models
| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-6` |
| `openai` | `gpt-4o` |
| `ollama` | `llama3.1:8b-instruct` |

### CLI
```
python -m src.llm.summarizer --in fused.json --out summary.raw.json [--provider anthropic] [--style detailed] [--qa]
```

---

## 3. Prompts

Prompts live in `src/llm/prompts/` as `.txt` files so they can be reviewed/edited without touching code.

### Pass 1 — `prompt_chunk.txt`
```
You are an expert video analyst. Below is a chronological transcript of a video segment, fused
with on-screen text (OCR) and visual descriptions.

Your job is to produce a strict JSON object with this exact shape:

{
  "local_summary": "2–3 sentences",
  "key_points": [
    {"timestamp": <seconds float>, "text": "...", "confidence": "low|medium|high"},
    ...
  ],
  "events": [
    {"timestamp": <seconds float>, "event_type": "...", "description": "..."}
  ]
}

Rules:
- Use timestamps from the input events (the t_start of the originating event).
- Do NOT invent facts. If a key point relies on a visual, mention it explicitly.
- "events" should capture topic transitions, demos starting/ending, decisions made, questions
  asked or answered, slides/screens that introduce new sections.
- Output ONLY the JSON. No prose, no markdown fences.

INPUT:
{{events_json}}
```

### Pass 2 — `prompt_global.txt`
```
You are an expert editor. You have N local summaries of consecutive segments of the same video.
Combine them into a single coherent global summary.

Output strict JSON:

{
  "full_summary": "3–6 paragraph narrative",
  "short_summary": "1–2 sentence elevator pitch",
  "chapters": [
    {"t_start": ..., "t_end": ..., "title": "..."},
    ...
  ],
  "merged_key_points": [...],   // dedupe and sort by timestamp
  "merged_events": [...]        // dedupe and sort by timestamp
}

Rules:
- Chapters should partition the video without gaps; titles 3–8 words.
- Drop key_points that are minor; aim for 6–15 in total.
- Preserve original timestamps.
- Output ONLY JSON.

INPUT:
{{local_summaries_json}}
```

### Style switches
- `concise`: append "Keep `full_summary` ≤ 100 words." to prompt_global.
- `detailed` (default): no change.
- `bullet-only`: replace `full_summary` with `bulleted_summary` (a JSON array of bullet strings).

### Domain hook (Plan 5.2)
If `domain` is set, prepend `prompt_domain_<domain>.txt` (loaded from `prompts/domains/`). E.g., `domain=education` adds: "Identify the learning objectives stated or implied. Note any worked examples and label them clearly."

---

## 4. Provider Adapters

Common interface:
```python
class LLMProvider(Protocol):
    def complete_json(self, system: str, user: str, *, max_tokens: int = 4096,
                      temperature: float = 0.2) -> tuple[dict, dict]: ...
    # returns (parsed_json, usage_dict)
```

### Anthropic adapter
```python
class AnthropicProvider:
    def __init__(self, model="claude-sonnet-4-6"):
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model

    def complete_json(self, system, user, *, max_tokens=4096, temperature=0.2):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}])
        text = resp.content[0].text.strip()
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens}
        return _strip_and_parse_json(text), usage
```

### OpenAI adapter
Use `response_format={"type": "json_object"}` — guarantees valid JSON output. Otherwise same shape.

### Ollama adapter
```python
class OllamaProvider:
    def complete_json(self, system, user, *, max_tokens=4096, temperature=0.2):
        import requests
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": self.model, "prompt": f"{system}\n\n{user}",
            "format": "json", "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False
        })
        r.raise_for_status()
        return json.loads(r.json()["response"]), {"input_tokens": 0, "output_tokens": 0}
```

### Robust JSON parsing
LLMs occasionally wrap JSON in markdown fences or add prose. `_strip_and_parse_json` should:
1. Try `json.loads(text)` directly.
2. On failure, regex-find the largest `{...}` block and try again.
3. On second failure, raise `LLMOutputParseError(text)` with the bad text attached.

### Retries
Wrap each call with `tenacity` — retry on `RateLimitError`, `APIConnectionError`, `LLMOutputParseError`. Max 3 attempts, exponential backoff.

---

## 5. Phased Implementation

### Phase A — Skeleton + dataclasses (~30 min)
Create files; define `Summary` and friends; stub `summarize()`.

### Phase B — Provider adapters (~2 hr)
Implement all three. Register in a factory:
```python
_providers = {
    "anthropic": AnthropicProvider,
    "openai":   OpenAIProvider,
    "ollama":   OllamaProvider,
}
```

### Phase C — Prompt loader (~30 min)
```python
def load_prompt(name: str, **vars) -> str:
    text = (Path(__file__).parent / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
    return text.format(**{k: json.dumps(v, default=str) if not isinstance(v, str) else v
                          for k, v in vars.items()})
```

### Phase D — Single-pass summarizer (~1 hr)
For short videos (< 8 K tokens of fused content):
1. Build `events_json = json.dumps(fused.events)`.
2. Call `prompt_chunk` then `prompt_global` with synthetic single-element list.
3. Map JSON result to `Summary` dataclass.

### Phase E — Multi-pass / chunked (~1 hr 30 min)
For long videos:
1. Run Plan 4.1 chunker if `len(events_json) > model_context_limit`.
2. Pass 1: for each chunk, call `prompt_chunk`, collect `local_summary`/`key_points`/`events`.
3. Pass 2: call `prompt_global` with the list of locals.
4. Merge into `Summary`.

Show progress to stderr (`Pass 1: chunk 3/12...`) so long jobs visibly run.

### Phase F — Rolling-summary mode (~1 hr 30 min, for live)
Behavior:
- State file `rolling_state.json` holds: last_summary, last_processed_event_index, total_event_count.
- On each call, read fused (which is being appended to live), grab events past `last_processed_event_index`, call `prompt_chunk` on the new slice, then `prompt_global` on `[old_summary_as_local, new_local]`.
- Atomic state update via temp-file + replace.
- Designed to be called from a watcher in Plan 5.1.

### Phase G — CLI (~30 min)
Standard argparse. `--rolling` requires `--rolling-state-path`. Print one-line summary to stderr; full JSON to `--out`.

### Phase H — Tests (~2 hr)

Goal: test logic without burning real API tokens. Use a fake provider.

```python
class FakeProvider:
    def __init__(self, scripted_responses): self._r = iter(scripted_responses)
    def complete_json(self, system, user, **kw): return next(self._r), {"input_tokens": 100, "output_tokens": 50}
```

1. **Single-pass happy path** — feed a tiny `fused.json`; provide canned local + global responses; assert resulting `Summary` has correct chapter list / key points.
2. **Multi-pass dispatch** — large `fused.json` that triggers chunking; assert N+1 calls (N locals + 1 global).
3. **JSON parse fallback** — fake provider returns text wrapped in ```` ```json ```` fences; assert `_strip_and_parse_json` recovers.
4. **Rolling state** — call rolling mode 3 times with growing `fused.json`; assert state file updates and `last_processed_event_index` advances.
5. **Provider selection** — `--provider openai` instantiates `OpenAIProvider` (mock). Don't actually call the API.
6. **Parse failure surfaces** — fake returns garbage; assert `LLMOutputParseError` raised after retries.
7. **Style switches** — `style=bullet-only` produces output with `bulleted_summary` populated and `full_summary` empty/null.

---

## 6. File Layout After Plan 4.2
```
src/llm/
  __init__.py
  summarizer.py
  schema.py
  parsing.py            # _strip_and_parse_json
  rolling.py
  providers/
    __init__.py
    base.py             # Protocol
    anthropic_provider.py
    openai_provider.py
    ollama_provider.py
  prompts/
    prompt_chunk.txt
    prompt_global.txt
    domains/
      education.txt
      trading.txt
      medical.txt
      law.txt
tests/llm/
  test_summarizer.py
  test_parsing.py
  test_rolling.py
  test_providers.py
```

---

## 7. Dependencies
Already added in earlier plans: `anthropic`, `openai`, `tiktoken`, `tenacity`. Add:
```
tenacity>=8.5.0
requests>=2.32.3
```

---

## 8. Acceptance Criteria

- [ ] CLI run on a Plan 4.1 `fused.json` produces a `summary.raw.json` matching the §2 schema.
- [ ] All three providers can be selected via `--provider`.
- [ ] JSON parsing recovers from common LLM formatting quirks (fenced output, trailing prose).
- [ ] Rolling mode updates incrementally and produces strictly increasing `last_processed_event_index`.
- [ ] All non-network tests pass with the fake provider; no real API key required for `pytest`.
- [ ] Token usage is recorded.
- [ ] Provider failures retry up to 3 times before raising.

---

## 9. Edge Cases & Pitfalls

1. **LLM ignores JSON instruction and returns prose** — `_strip_and_parse_json` recovers; retry once with a stricter system message; fail after 3.
2. **Timestamp drift in chapters** — if model returns chapters with overlapping or non-monotonic times, post-process to sort and clip.
3. **Hallucinated events not in input** — model may invent timestamps. Validate every output timestamp is `0 ≤ t ≤ duration`. Drop invalid entries.
4. **Empty fused doc** — return a `Summary` with empty fields and `full_summary="(no content detected)"`. Don't call the LLM.
5. **Token cost on long videos** — log estimated cost (`$ = (input_tokens * input_price + output_tokens * output_price)`). Provide `--max-cost` safety cap that aborts before exceeding it.
6. **Rate limits on Anthropic** — exponential backoff already covers; for very long jobs, add `--throttle-rps` flag.
7. **Concurrent rolling-mode invocations** — guard state file with a lockfile (`portalocker` or `fcntl`).
8. **Streaming** — for live UI, you may want to stream the global summary. Anthropic and OpenAI both support it; expose `--stream` flag that prints tokens as they arrive but still writes the final JSON. Implement only if time permits.
9. **Locale / language** — pass the transcript's `language` field into the prompts so Spanish content gets a Spanish summary. Add `target_language` override.

---

## 10. Out of Scope

- Final Markdown/HTML/PDF rendering (Plan 4.3).
- RAG / vector search (this is a one-shot summarization, not an interactive chat).
- Fine-tuning / training.

---

## 11. Definition of Done

A developer can take a Plan 4.1 `fused.json`, run `python -m src.llm.summarizer --in fused.json --out summary.raw.json --provider anthropic`, and get a JSON file that satisfies the schema in §2 — using only this plan file and an `ANTHROPIC_API_KEY`.
