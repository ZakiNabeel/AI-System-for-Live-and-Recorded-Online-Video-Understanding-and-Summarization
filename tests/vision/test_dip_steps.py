from __future__ import annotations

import cv2
import numpy as np

from src.vision.dip_steps import binarize, clahe, denoise, grayscale, invert_if_dark, morph, sharpen


def test_binarize_outputs_binary_values() -> None:
    gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (64, 1))

    binary = binarize(gradient, method="otsu")

    assert set(np.unique(binary)).issubset({0, 255})


def test_dip_steps_preserve_shape() -> None:
    bgr = np.full((80, 120, 3), 180, dtype=np.uint8)
    cv2.rectangle(bgr, (20, 20), (100, 55), (10, 10, 10), -1)

    gray = grayscale(bgr)
    processed = denoise(gray, method="gaussian", ksize=3)
    processed = clahe(processed)
    processed = sharpen(processed)
    processed = binarize(processed, method="adaptive_gaussian")
    processed = invert_if_dark(processed)
    processed = morph(processed, op="open", kernel=[2, 2])

    assert processed.shape == gray.shape
    assert processed.dtype == np.uint8
