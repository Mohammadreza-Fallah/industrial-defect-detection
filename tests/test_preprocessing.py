import numpy as np
import pytest

from src.utils.preprocessing import (
    resize,
    normalize,
    extract_edges,
    extract_contours,
)


@pytest.fixture
def dummy_image():
    """A synthetic 100x100 grayscale image with a white square defect."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 255
    return img


def test_resize_changes_shape(dummy_image):
    resized = resize(dummy_image, (50, 50))
    assert resized.shape == (50, 50)


def test_normalize_range(dummy_image):
    normed = normalize(dummy_image)
    assert normed.dtype == np.float32
    assert normed.min() >= 0.0
    assert normed.max() <= 1.0


def test_extract_edges_returns_binary_like_map(dummy_image):
    edges = extract_edges(dummy_image)
    assert edges.shape == dummy_image.shape
    assert set(np.unique(edges)).issubset({0, 255})


def test_extract_contours_finds_the_square(dummy_image):
    contours = extract_contours(dummy_image)
    assert len(contours) >= 1
    # Largest contour should roughly match the 40x40 square (1600 px area)
    import cv2

    area = cv2.contourArea(contours[0])
    assert 1000 < area < 2000
