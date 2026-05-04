"""Small, testable classical Digital Image Processing steps."""

from __future__ import annotations

import cv2
import numpy as np


def grayscale(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, method: str, ksize: int = 3, **kwargs: int) -> np.ndarray:
    if method == "gaussian":
        odd_ksize = _odd_at_least(ksize, 3)
        return cv2.GaussianBlur(gray, (odd_ksize, odd_ksize), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(
            gray,
            int(kwargs.get("d", 9)),
            int(kwargs.get("sigma_color", 75)),
            int(kwargs.get("sigma_space", 75)),
        )
    raise ValueError(f"Unsupported denoise method: {method}")


def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: list[int] | tuple[int, int] = (8, 8)) -> np.ndarray:
    tile_size = (int(tile[0]), int(tile[1]))
    return cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_size).apply(gray)


def sharpen(gray: np.ndarray, sigma: float = 2.0, amount: float = 0.5) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(sigma))
    weight = 1.0 + float(amount)
    return cv2.addWeighted(gray, weight, blur, -float(amount), 0)


def binarize(gray: np.ndarray, method: str, block: int = 31, c: int = 10) -> np.ndarray:
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    if method == "adaptive_gaussian":
        block_size = _odd_at_least(block, 3)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            int(c),
        )
    raise ValueError(f"Unsupported threshold method: {method}")


def invert_if_dark(binary: np.ndarray) -> np.ndarray:
    if float(binary.mean()) < 127.0:
        return cv2.bitwise_not(binary)
    return binary


def morph(binary: np.ndarray, op: str, kernel: list[int] | tuple[int, int], iters: int = 1) -> np.ndarray:
    kernel_arr = np.ones((int(kernel[1]), int(kernel[0])), np.uint8)
    if op == "open":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_arr, iterations=int(iters))
    if op == "dilate":
        return cv2.dilate(binary, kernel_arr, iterations=int(iters))
    if op == "erode":
        return cv2.erode(binary, kernel_arr, iterations=int(iters))
    raise ValueError(f"Unsupported morphology op: {op}")


def deskew(image: np.ndarray, max_angle: float = 10) -> np.ndarray:
    coords = np.column_stack(np.where(image < 255))
    if len(coords) < 50:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > max_angle:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _odd_at_least(value: int, minimum: int) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1
