"""Pre-tuned classical DIP profiles for OCR-oriented frame enhancement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "grayscale": True,
        "denoise": {"method": "gaussian", "ksize": 3},
        "clahe": None,
        "sharpen": None,
        "threshold": {"method": "adaptive_gaussian", "block": 31, "c": 10},
        "invert_if_dark": True,
        "morph": {"op": "open", "kernel": [2, 2], "iters": 1},
        "deskew": False,
    },
    "screen": {
        "grayscale": True,
        "denoise": {"method": "gaussian", "ksize": 3},
        "clahe": None,
        "sharpen": {"sigma": 2.0, "amount": 0.5},
        "threshold": {"method": "otsu"},
        "invert_if_dark": True,
        "morph": {"op": "dilate", "kernel": [2, 2], "iters": 1},
        "deskew": False,
    },
    "whiteboard": {
        "grayscale": True,
        "denoise": {"method": "bilateral", "d": 9, "sigma_color": 75, "sigma_space": 75},
        "clahe": {"clip_limit": 2.0, "tile": [8, 8]},
        "sharpen": None,
        "threshold": {"method": "adaptive_gaussian", "block": 31, "c": 10},
        "invert_if_dark": True,
        "morph": {"op": "dilate", "kernel": [2, 2], "iters": 1},
        "deskew": {"max_angle": 10},
    },
    "scene": {
        "grayscale": True,
        "denoise": {"method": "bilateral", "d": 9, "sigma_color": 75, "sigma_space": 75},
        "clahe": {"clip_limit": 2.0, "tile": [8, 8]},
        "sharpen": None,
        "threshold": None,
        "invert_if_dark": False,
        "morph": None,
        "deskew": False,
    },
}


def get_profile(name: str) -> dict[str, Any]:
    """Return a mutable copy of a named enhancement profile."""

    try:
        return deepcopy(PROFILES[name])
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown enhancement profile '{name}'. Choose one of: {choices}") from exc
