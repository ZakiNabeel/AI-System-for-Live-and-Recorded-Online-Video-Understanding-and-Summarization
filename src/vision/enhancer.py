"""Apply classical DIP enhancement profiles to extracted key-frames."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from .dip_steps import binarize, clahe, denoise, deskew, grayscale, invert_if_dark, morph, sharpen
from .enhance_profiles import PROFILES, get_profile
from .errors import EnhancementError

EnhancementProfile = Literal["default", "screen", "whiteboard", "scene"]


@dataclass(frozen=True)
class EnhancedFrame:
    source_path: Path
    enhanced_path: Path
    intermediates: dict[str, Path]
    profile: str
    parameters: dict[str, Any]
    timestamp: float | None = None
    index: int | None = None


@dataclass(frozen=True)
class EnhancementResult:
    frames: list[EnhancedFrame]
    profile: str
    parameters: dict[str, Any]


def enhance_frames(
    frames_json_path: Path,
    output_dir: Path | None = None,
    profile: EnhancementProfile = "default",
    save_intermediates: bool = False,
) -> EnhancementResult:
    """Enhance every frame listed in a Plan 3.1 frames manifest."""

    frames_json_path = Path(frames_json_path)
    manifest = _read_manifest(frames_json_path)
    source_base = frames_json_path.parent
    target_dir = Path(output_dir) if output_dir is not None else source_base
    target_dir.mkdir(parents=True, exist_ok=True)
    params = get_profile(profile)
    enhanced_frames: list[EnhancedFrame] = []

    for entry in manifest.get("frames", []):
        source_path = _resolve_manifest_path(entry["path"], source_base)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EnhancementError(f"Could not read frame image: '{source_path}'")

        enhanced, intermediates = apply_profile(
            image,
            params,
            collect_intermediates=save_intermediates,
        )
        enhanced_path = target_dir / f"{source_path.stem}.enhanced.png"
        _write_image(enhanced_path, enhanced)

        intermediate_paths: dict[str, Path] = {}
        for name, intermediate in intermediates.items():
            path = target_dir / f"{source_path.stem}.{name}.png"
            _write_image(path, intermediate)
            intermediate_paths[name] = path.resolve()

        enhanced_frames.append(
            EnhancedFrame(
                source_path=source_path.resolve(),
                enhanced_path=enhanced_path.resolve(),
                intermediates=intermediate_paths,
                profile=profile,
                parameters=params,
                timestamp=entry.get("timestamp"),
                index=entry.get("index"),
            )
        )

    result = EnhancementResult(frames=enhanced_frames, profile=profile, parameters=params)
    _write_enhancement_manifest(
        frames_json_path=frames_json_path,
        output_dir=target_dir,
        result=result,
        save_intermediates=save_intermediates,
    )
    return result


def apply_profile(
    bgr_image: np.ndarray,
    profile_params: dict[str, Any],
    *,
    collect_intermediates: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the configured DIP pipeline and optionally return snapshots."""

    intermediates: dict[str, np.ndarray] = {}
    image = bgr_image
    if profile_params.get("grayscale"):
        image = grayscale(image)
        if collect_intermediates:
            intermediates["gray"] = image.copy()
    if profile_params.get("denoise"):
        image = denoise(image, **profile_params["denoise"])
    if profile_params.get("clahe"):
        image = clahe(image, **profile_params["clahe"])
    if profile_params.get("sharpen"):
        image = sharpen(image, **profile_params["sharpen"])
    if profile_params.get("threshold"):
        image = binarize(image, **profile_params["threshold"])
        if collect_intermediates:
            intermediates["thresh"] = image.copy()
    if profile_params.get("invert_if_dark"):
        image = invert_if_dark(image)
    if profile_params.get("morph"):
        image = morph(image, **profile_params["morph"])
    if profile_params.get("deskew"):
        deskew_params = profile_params["deskew"] if isinstance(profile_params["deskew"], dict) else {}
        image = deskew(image, **deskew_params)
    return image, intermediates


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Frames manifest not found: '{path.resolve()}'")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnhancementError(f"Invalid JSON manifest: '{path}'") from exc
    if "frames" not in data or not isinstance(data["frames"], list):
        raise EnhancementError(f"Manifest does not contain a frames list: '{path}'")
    return data


def _resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise EnhancementError(f"Failed to write image: '{path}'")


def _write_enhancement_manifest(
    *,
    frames_json_path: Path,
    output_dir: Path,
    result: EnhancementResult,
    save_intermediates: bool,
) -> Path:
    payload = {
        "version": "1",
        "frames_manifest": str(frames_json_path.resolve()),
        "profile": result.profile,
        "parameters": result.parameters,
        "save_intermediates": save_intermediates,
        "frames": [
            _enhanced_frame_to_manifest(frame, output_dir)
            for frame in result.frames
        ],
    }
    path = output_dir / "enhancements.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def _enhanced_frame_to_manifest(frame: EnhancedFrame, output_dir: Path) -> dict[str, Any]:
    payload = asdict(frame)
    payload["path"] = _portable_path(frame.source_path, output_dir)
    payload["source_path"] = _portable_path(frame.source_path, output_dir)
    payload["enhanced_path"] = _portable_path(frame.enhanced_path, output_dir)
    payload["intermediates"] = {
        name: _portable_path(path, output_dir)
        for name, path in frame.intermediates.items()
    }
    return payload


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enhance key-frames for OCR.")
    parser.add_argument("--frames", type=Path, required=True, help="Path to frames.json.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to frame directory.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="default")
    parser.add_argument("--save-intermediates", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        result = enhance_frames(
            frames_json_path=args.frames,
            output_dir=args.out_dir,
            profile=args.profile,
            save_intermediates=args.save_intermediates,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    elapsed = time.perf_counter() - started
    avg_ms = elapsed * 1000 / max(len(result.frames), 1)
    out_dir = args.out_dir or args.frames.parent
    print(f"enhanced {len(result.frames)}/{len(result.frames)} frames; avg time/frame: {avg_ms:.1f} ms; {Path(out_dir) / 'enhancements.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
