"""Google Gemini Vision captioning adapter (google-genai SDK)."""

from __future__ import annotations

import os
from pathlib import Path


def caption_gemini(image_path: Path) -> str | None:
    """Return a one-sentence visual caption using Gemini Vision, or None when unavailable."""
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types as genai_types
        from PIL import Image
    except ImportError:
        return None

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    image = Image.open(str(image_path))

    response = client.models.generate_content(
        model=os.getenv("GEMINI_VISION_MODEL", "gemini-1.5-flash"),
        contents=[
            image,
            "Describe this video frame in one short sentence. "
            "Focus on visually distinctive objects, scene, and activity. "
            "Ignore any text on screen.",
        ],
    )
    text = (getattr(response, "text", "") or "").strip()
    return text or None
