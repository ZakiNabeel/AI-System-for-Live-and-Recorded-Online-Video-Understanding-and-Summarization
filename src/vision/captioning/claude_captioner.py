"""Claude Vision captioning adapter."""

from __future__ import annotations

import base64
import os
from pathlib import Path


def caption_claude(image_path: Path) -> str | None:
    """Return a one-sentence visual caption, or None when unavailable."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    img_bytes = Path(image_path).read_bytes()
    encoded = base64.standard_b64encode(img_bytes).decode("ascii")
    client = Anthropic()
    response = client.messages.create(
        model=os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6"),
        max_tokens=120,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe this video frame in one short sentence. "
                            "Focus on visually distinctive objects, scene, and activity. "
                            "Ignore any text on screen."
                        ),
                    },
                ],
            }
        ],
    )
    if not response.content:
        return None
    return getattr(response.content[0], "text", "").strip() or None
