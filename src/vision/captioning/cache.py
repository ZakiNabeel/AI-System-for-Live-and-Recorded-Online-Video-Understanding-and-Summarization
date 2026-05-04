"""Content-addressed caption cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class CaptionCache:
    """JSON-backed cache keyed by image bytes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}

    def key_for(self, image_path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        digest.update(Path(image_path).read_bytes())
        return digest.hexdigest()

    def get(self, image_path: Path) -> str | None:
        return self._data.get(self.key_for(image_path))

    def set(self, image_path: Path, caption: str) -> None:
        self._data[self.key_for(image_path)] = caption
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
