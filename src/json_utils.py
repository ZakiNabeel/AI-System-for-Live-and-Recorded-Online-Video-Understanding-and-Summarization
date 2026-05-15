"""JSON utilities: encoder that handles numpy and Path types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SafeEncoder(json.JSONEncoder):
    """Serialize numpy scalars, ndarrays, and Path objects."""

    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("cls", SafeEncoder)
    return json.dumps(obj, **kwargs)


def dump(obj: Any, fp: Any, **kwargs: Any) -> None:
    kwargs.setdefault("cls", SafeEncoder)
    json.dump(obj, fp, **kwargs)
