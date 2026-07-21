"""
OpenCV-based preprocessing utilities for surface-defect images.

These functions form the shared image-processing toolkit reused between
this project and the thesis pipeline (contour extraction, thresholding,
edge detection) — kept dependency-light and pure-function style so they
are easy to unit test and reuse in the thesis codebase.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    """Load an image from disk as a single-channel grayscale array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def resize(image: np.ndarray, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """Resize an image to a fixed (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32."""
    return image.astype(np.float32) / 255.0


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Light denoising via Non-Local Means (mostly a no-op on NEU images,
    but kept for parity with the noisier X-ray thesis pipeline)."""
    return cv2.fastNlMeansDenoising(image, h=strength)


def extract_edges(image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map — used for visual QA and as an optional extra
    input channel for the model."""
    return cv2.Canny(image, low, high)


def extract_contours(image: np.ndarray, thresh: int = 127) -> list[np.ndarray]:
    """Binary threshold + contour extraction. Returns a list of contours,
    largest first — same approach used to isolate the keyhole boundary
    from X-ray projections in the thesis."""
    _, binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return sorted(contours, key=cv2.contourArea, reverse=True)


def preprocess_pipeline(
    path: str, size: tuple[int, int] = (200, 200)
) -> np.ndarray:
    """Full preprocessing pipeline: load -> resize -> normalize.
    Returns a float32 array in [0, 1], shape (H, W)."""
    img = load_grayscale(path)
    img = resize(img, size)
    return normalize(img)
