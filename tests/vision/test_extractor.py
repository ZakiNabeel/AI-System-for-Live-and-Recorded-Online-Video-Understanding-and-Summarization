from __future__ import annotations

import json
from pathlib import Path

from src.vision.extractor import TextLine, extract_visual_content


def test_extract_visual_content_writes_visual_json(
    monkeypatch,
    text_frame: Path,
    tmp_path: Path,
) -> None:
    enhanced = tmp_path / "frame_000000_t0.00.enhanced.png"
    enhanced.write_bytes(text_frame.read_bytes())
    manifest = tmp_path / "enhancements.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "1",
                "frames": [
                    {
                        "index": 0,
                        "timestamp": 0.0,
                        "source_path": text_frame.name,
                        "enhanced_path": enhanced.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_ocr(image_path: Path, languages: list[str]):
        return [
            {
                "text": "HELLO WORLD",
                "confidence": 91.0,
                "bbox": (1, 2, 30, 10),
                "language": languages[0],
            }
        ], {"fake": True}

    monkeypatch.setattr("src.vision.extractor.ocr_tesseract", fake_ocr)

    result = extract_visual_content(manifest, tmp_path / "intermediate", ocr_engine="tesseract")
    visual_json = json.loads((tmp_path / "intermediate" / "visual.json").read_text(encoding="utf-8"))

    assert result.frames[0].text == "HELLO WORLD"
    assert result.frames[0].has_text is True
    assert isinstance(result.frames[0].lines[0], TextLine)
    assert visual_json["frames"][0]["lines"][0]["bbox"] == [1, 2, 30, 10]


def test_caption_cache_reuses_identical_frame(
    monkeypatch,
    text_frame: Path,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "frames.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "1",
                "frames": [
                    {"index": 0, "timestamp": 0.0, "path": text_frame.name},
                    {"index": 1, "timestamp": 1.0, "path": text_frame.name},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = {"count": 0}

    def fake_ocr(image_path: Path, languages: list[str]):
        return [], {"fake": True}

    def fake_caption(image_path: Path) -> str:
        calls["count"] += 1
        return "A synthetic text frame."

    monkeypatch.setattr("src.vision.extractor.ocr_tesseract", fake_ocr)
    monkeypatch.setattr("src.vision.extractor.caption_claude", fake_caption)

    result = extract_visual_content(
        manifest,
        tmp_path / "intermediate",
        ocr_engine="tesseract",
        enable_captions=True,
        captioner="claude",
    )

    assert calls["count"] == 1
    assert [frame.caption for frame in result.frames] == [
        "A synthetic text frame.",
        "A synthetic text frame.",
    ]
