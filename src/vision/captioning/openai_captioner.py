"""OpenAI vision captioning adapter."""

from __future__ import annotations

import base64
import os
from pathlib import Path


def caption_openai(image_path: Path) -> str | None:
    """Return a one-sentence visual caption, or None when unavailable."""

    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    client = OpenAI()
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini"),
        max_tokens=120,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this video frame in one short sentence. "
                            "Focus on visually distinctive objects, scene, and activity. "
                            "Ignore any text on screen."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip() or None
