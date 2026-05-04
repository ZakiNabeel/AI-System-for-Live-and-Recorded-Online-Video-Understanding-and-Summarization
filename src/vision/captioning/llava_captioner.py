"""Local LLaVA captioning adapter through Ollama."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def caption_llava_local(image_path: Path) -> str | None:
    """Return a one-sentence visual caption from local Ollama LLaVA, if running."""

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    payload = {
        "model": "llava:7b",
        "prompt": (
            "Describe this video frame in one short sentence. Focus on visually "
            "distinctive objects, scene, and activity. Ignore any text on screen."
        ),
        "images": [encoded],
        "stream": False,
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None
    return str(data.get("response", "")).strip() or None
