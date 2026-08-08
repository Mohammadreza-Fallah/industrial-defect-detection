"""
Detects near-duplicate images between the NEU-DET train and validation
splits using perceptual hashing (pHash). A known issue with some
distributions of NEU-DET is that images are cropped from a small number
of source micrographs, so train/validation splits done naively (by
image, not by source micrograph) can leak near-identical crops across
the split — which would explain suspiciously high validation accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def phash(image_path: str, hash_size: int = 8) -> np.ndarray:
    """Computes a simple perceptual hash (DCT-free average-hash variant)
    for an image: resize to (hash_size+1, hash_size), compare adjacent
    pixels. Returns a boolean array — Hamming distance between two
    hashes measures visual similarity (0 = identical)."""
    img = Image.open(image_path).convert("L").resize(
        (hash_size + 1, hash_size), Image.LANCZOS
    )
    pixels = np.asarray(img, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def find_near_duplicates(
    train_dir: str,
    val_dir: str,
    class_names: list[str],
    max_distance: int = 5,
) -> list[dict]:
    """Compares every validation image against every train image of the
    SAME class (comparing across all classes would be much slower and
    isn't needed — we only care about same-class leakage, which is what
    would inflate accuracy).

    Returns a list of dicts: {class, train_file, val_file, distance}
    for every pair with Hamming distance <= max_distance, sorted by
    distance (most suspicious first).
    """
    results = []

    for class_name in class_names:
        train_class_dir = Path(train_dir) / class_name
        val_class_dir = Path(val_dir) / class_name
        if not train_class_dir.is_dir() or not val_class_dir.is_dir():
            continue

        train_files = sorted(train_class_dir.glob("*"))
        val_files = sorted(val_class_dir.glob("*"))

        train_hashes = {f: phash(str(f)) for f in train_files}
        val_hashes = {f: phash(str(f)) for f in val_files}

        for val_f, val_h in val_hashes.items():
            best_dist = None
            best_train_f = None
            for train_f, train_h in train_hashes.items():
                d = hamming_distance(val_h, train_h)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_train_f = train_f

            if best_dist is not None and best_dist <= max_distance:
                results.append(
                    {
                        "class": class_name,
                        "train_file": str(best_train_f),
                        "val_file": str(val_f),
                        "distance": best_dist,
                    }
                )

    return sorted(results, key=lambda r: r["distance"])
