# Plan 3.3 — Visual Content Extraction (OCR + Optional Captioning)

> **Self-contained scope.** Take the enhanced key-frames from Plan 3.2 and produce a structured JSON describing what is visible: extracted text (OCR), bounding boxes, confidence scores, and optionally a one-sentence vision-model caption per frame. This is the last visual stage before fusion.

---

## 1. Objective

Build `src/vision/extractor.py` that:

1. Reads `enhancements.json` from Plan 3.2 (or `frames.json` directly from Plan 3.1 if enhancement was skipped).
2. For each frame, runs OCR on the enhanced image and produces:
   - Plain text (concatenated reading order).
   - Per-line records with bounding boxes, confidence, language.
3. Optionally runs a vision-language model (Claude Vision, GPT-4o-mini, or local LLaVA) to produce a one-sentence caption describing the *non-text* content of the frame.
4. Emits `visual.json` — the structured output that Plan 4.1 consumes.

Two backends for OCR; two backends for captioning. Default config requires no API keys.

---

## 2. Contract

### Output dataclasses
```python
@dataclass
class TextLine:
    text: str
    confidence: float       # 0..100 (Tesseract style); rescale others to match
    bbox: tuple[int,int,int,int]   # (x, y, w, h) in original frame pixels
    language: str

@dataclass
class FrameVisual:
    timestamp: float        # from KeyFrame.timestamp
    frame_index: int
    frame_path: Path        # original PNG path (not enhanced)
    enhanced_path: Path | None
    text: str               # joined plain-text reading order
    lines: list[TextLine]
    caption: str | None     # vision-LM caption, optional
    caption_source: str | None
    has_text: bool          # convenience: len(lines) > 0 and total chars > 5
    raw_ocr: dict | None    # full backend response for debugging

@dataclass
class VisualExtractionResult:
    frames: list[FrameVisual]
    ocr_engine: str
    captioner: str | None
    elapsed_sec: float
```

### Top-level function
```python
def extract_visual_content(
    frames_manifest_path: Path,           # frames.json or enhancements.json
    output_dir: Path,
    *,
    ocr_engine: Literal["tesseract", "easyocr"] = "tesseract",
    languages: list[str] = ["eng"],
    enable_captions: bool = False,
    captioner: Literal["claude", "openai", "llava-local"] = "claude",
) -> VisualExtractionResult: ...
```

### CLI
```
python -m src.vision.extractor --manifest enhancements.json --out-dir data/intermediate/<id>
                               [--ocr tesseract] [--captions --captioner claude]
```

Output file: `output_dir/visual.json`.

---

## 3. Dependencies

| Component | Package / Tool | Notes |
|---|---|---|
| Tesseract OCR | `pytesseract>=0.3.10` + Tesseract binary | Free, fast, classical. Install binary via `winget install UB-Mannheim.TesseractOCR` |
| EasyOCR | `easyocr>=1.7.1` | NN-based, better on natural scenes, slower, downloads ~70 MB model |
| Claude Vision | `anthropic>=0.34.0` | Cloud, needs `ANTHROPIC_API_KEY` |
| OpenAI Vision | `openai>=1.40.0` | Cloud, needs `OPENAI_API_KEY` |
| LLaVA local | `ollama` + `llava:7b` model | Big download, GPU recommended |

Add to `requirements.txt`:
```
pytesseract>=0.3.10
easyocr>=1.7.1
anthropic>=0.34.0
Pillow>=10.4.0
```

---

## 4. OCR Backends

### Tesseract (default)
```python
import pytesseract
from PIL import Image

def ocr_tesseract(enhanced_path, original_path, languages):
    img = Image.open(enhanced_path)
    data = pytesseract.image_to_data(img, lang="+".join(languages),
                                     output_type=pytesseract.Output.DICT)
    lines: dict[int, list[int]] = {}
    for i, text in enumerate(data["text"]):
        if not text.strip(): continue
        line_id = data["block_num"][i] * 1000 + data["line_num"][i]
        lines.setdefault(line_id, []).append(i)
    out_lines = []
    for line_id, idxs in lines.items():
        words = [data["text"][i] for i in idxs]
        confs = [int(data["conf"][i]) for i in idxs if int(data["conf"][i]) >= 0]
        if not confs: continue
        x = min(data["left"][i] for i in idxs)
        y = min(data["top"][i] for i in idxs)
        w = max(data["left"][i] + data["width"][i] for i in idxs) - x
        h = max(data["top"][i] + data["height"][i] for i in idxs) - y
        out_lines.append(TextLine(
            text=" ".join(words), confidence=sum(confs)/len(confs),
            bbox=(x,y,w,h), language=languages[0]))
    return out_lines
```

Drop lines with `confidence < 30` and `len(text) < 3` (noise).

### EasyOCR
```python
import easyocr
reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
results = reader.readtext(str(enhanced_path))   # [(box, text, conf), ...]
```
Convert quadrilateral boxes to axis-aligned `(x, y, w, h)`. Rescale conf to 0..100 (multiply by 100).

Pick EasyOCR for non-Latin scripts, screenshots with stylized fonts, or photographs of signs.

---

## 5. Vision-LM Captioning Backends (optional)

### Claude Vision (recommended)
```python
def caption_claude(image_path: Path) -> str:
    import base64
    from anthropic import Anthropic
    client = Anthropic()
    img_bytes = image_path.read_bytes()
    b64 = base64.standard_b64encode(img_bytes).decode()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text",
                 "text": "Describe this video frame in one short sentence. "
                         "Focus on what is visually distinctive (objects, scene, "
                         "activity). Ignore any text on screen."}
            ]
        }]
    )
    return resp.content[0].text.strip()
```

### OpenAI / LLaVA
Same pattern with `client.chat.completions.create` and a `gpt-4o-mini` model (or Ollama HTTP for local LLaVA). Wrap each in a try/except — missing API key falls back to `caption=None`, `caption_source=None`.

### Caching
Captioning is the most expensive operation. Hash the *original* frame path's bytes (md5 of file contents) and store in `output_dir / "caption_cache.json"`. Skip re-captioning identical frames.

### Rate limiting / batching
Process at most 5 frames in parallel (use `concurrent.futures.ThreadPoolExecutor`). Add `tenacity` retry with exponential backoff.

---

## 6. Phased Implementation

### Phase A — Skeleton + dataclasses (~30 min)
Create files; define dataclasses; stub function.

### Phase B — Tesseract integration (~1 hr)
Implement `ocr_tesseract` per §4. Verify Tesseract is on PATH (`pytesseract.get_tesseract_version()`). If missing, raise `TesseractNotInstalledError` with install hint.

### Phase C — Manifest reader (~30 min)
Auto-detect manifest schema:
```python
data = json.loads(path.read_text())
if "version" not in data and "frames" not in data:
    raise ValueError("unknown manifest")
# both Plan 3.1 and Plan 3.2 manifests have a top-level "frames" list with timestamps
```

For each entry, find the **enhanced** image if available (look for `enhanced_path` field), else use the original.

### Phase D — EasyOCR backend (~45 min)
Stub then implement. Lazy-load the model (constructor on first use — cold start is ~3 s).

### Phase E — Captioner integration (~1 hr 30 min)
Implement Claude path; add OpenAI; stub LLaVA. Add the cache file. Add retry decorator.

### Phase F — Main pipeline (~45 min)
```python
def extract_visual_content(frames_manifest_path, output_dir, *, ocr_engine, languages,
                           enable_captions, captioner):
    manifest = json.loads(Path(frames_manifest_path).read_text())
    visuals = []
    for entry in manifest["frames"]:
        enhanced = Path(entry.get("enhanced_path") or entry["path"])
        original = Path(entry["path"])
        lines = run_ocr(ocr_engine, enhanced, original, languages)
        text = "\n".join(l.text for l in lines)
        caption = run_caption(captioner, original) if enable_captions else None
        visuals.append(FrameVisual(
            timestamp=entry["timestamp"], frame_index=entry["index"],
            frame_path=original, enhanced_path=enhanced,
            text=text, lines=lines, caption=caption,
            caption_source=captioner if enable_captions else None,
            has_text=len(text.strip()) > 5,
            raw_ocr=None,
        ))
    out = output_dir / "visual.json"
    out.write_text(_to_json(visuals), encoding="utf-8")
    return VisualExtractionResult(frames=visuals, ...)
```

### Phase G — CLI (~30 min)
Standard argparse. Print summary: `47 frames; 312 lines of text; 12 captioned; 32.4 s total`.

### Phase H — Tests (~2 hr)

1. **Tesseract on synthetic text** — render a 320×240 image with `cv2.putText("HELLO WORLD", ...)`; run ocr_tesseract; assert `"HELLO" in text and "WORLD" in text`. Mark `@pytest.mark.requires_tesseract`.
2. **EasyOCR on same** — same assertion, mark `@pytest.mark.slow`.
3. **OCR confidence filtering** — mock pytesseract to return low-confidence lines; assert they are dropped.
4. **Manifest auto-detect** — feed a Plan 3.1-style and Plan 3.2-style JSON; both should work.
5. **Caption cache** — mock the Claude client; call twice on same frame; assert client called only once.
6. **No-API-key fallback** — set captioning enabled but unset env vars; assert `caption=None` and a single warning logged.
7. **Empty frames list** — assert `visual.json` is written with `[]`.

---

## 7. File Layout After Plan 3.3
```
src/vision/
  extractor.py
  ocr/
    __init__.py
    tesseract_ocr.py
    easyocr_ocr.py
  captioning/
    __init__.py
    claude_captioner.py
    openai_captioner.py
    llava_captioner.py
    cache.py
tests/vision/
  test_extractor.py
  test_tesseract.py
  test_easyocr.py
  test_caption_cache.py
```

---

## 8. Acceptance Criteria

- [ ] CLI run on a Plan 3.2 output produces `visual.json` matching the schema.
- [ ] Tesseract path produces correct text on synthetic canary frame (`HELLO WORLD`).
- [ ] EasyOCR path is selectable via `--ocr easyocr`.
- [ ] Captioning is opt-in (`--captions`) and degrades gracefully without API keys.
- [ ] Caption cache prevents duplicate API calls.
- [ ] All non-slow / non-network tests pass.
- [ ] `has_text` is set correctly.

---

## 9. Edge Cases & Pitfalls

1. **Tesseract not on PATH (Windows)** — set `pytesseract.pytesseract.tesseract_cmd` from config or env var if PATH lookup fails.
2. **Language pack missing** — Tesseract errors with code 1 and `Failed loading language 'xxx'`. Catch and fall back to `eng` with a warning.
3. **EasyOCR cold-start memory** — ~1 GB RSS even for English-only. Document.
4. **Captioning latency** — 0.5–3 s per call. Default to disabled. Don't run on every frame in live mode.
5. **OCR on heavily-binarized image with white-on-black inverted** — Tesseract can fail. Plan 3.2 already inverts when needed; ensure `invert_if_dark` was applied.
6. **Confidence == -1 from Tesseract** — means "I have no idea"; treat as drop.
7. **Bounding boxes in original-frame coordinates** — if Plan 3.2 resized images, you must scale boxes back to the original frame's dimensions. Store the resize ratio in `enhancements.json` and apply here.
8. **Vision-LM hallucinations** — Claude/GPT may invent details. Keep captions short (one sentence, max ~120 tokens) and don't use captions to *replace* OCR text — they complement.
9. **Caption rate limits** — process sequentially or with `max_workers=3`; on 429, retry with backoff.
10. **Privacy** — captioning sends frames to a third party. Document this clearly in README.

---

## 10. Out of Scope

- Object detection (YOLO etc.) — not in MVP, could be added as `src/vision/detection.py`.
- Logo / face recognition.
- Video-level temporal models (e.g., I3D action recognition).

---

## 11. Definition of Done

A developer can take a Plan 3.2 `enhancements.json`, run `python -m src.vision.extractor --manifest enhancements.json --out-dir out/`, and get a `visual.json` listing extracted text per frame — using only this plan file and a Tesseract install.
