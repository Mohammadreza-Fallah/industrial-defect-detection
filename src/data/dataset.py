"""
PyTorch Dataset for the NEU Surface Defect Database (object-detection
distribution — images + Pascal-VOC-style XML annotations).

Expected folder layout (as shipped by the "NEU-DET" Kaggle release):

    <root_dir>/               (e.g. NEU-DET/train or NEU-DET/validation)
    ├── images/
    │   ├── crazing/*.jpg
    │   ├── inclusion/*.jpg
    │   ├── patches/*.jpg
    │   ├── pitted_surface/*.jpg
    │   ├── rolled-in_scale/*.jpg
    │   └── scratches/*.jpg
    └── annotations/
        ├── crazing_1.xml
        ├── patches_35.xml
        └── ...              (flat, one XML per image, prefixed by class)

For Phase 1/2 (classification), only the `images/` folder is used — the
class is taken from the subfolder name. The `annotations/` XML files
(bounding boxes) are reserved for Phase 3 (detection) and not parsed here.

See docs/dataset.md for download instructions and class descriptions.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.preprocessing import load_grayscale, normalize, resize

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


class NEUDataset(Dataset):
    """Loads NEU-DET images and their class labels.

    Args:
        root_dir: path to NEU-DET/train or NEU-DET/validation (the folder
            that directly contains an `images/` subfolder).
        image_size: (width, height) to resize images to.
        transform: optional callable applied to the normalized image
            (e.g. torchvision augmentations).
    """

    def __init__(
        self,
        root_dir: str,
        image_size: tuple[int, int] = (200, 200),
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.images_dir = self._resolve_images_dir(self.root_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples: list[tuple[str, int]] = self._index_samples()

    @staticmethod
    def _resolve_images_dir(root_dir: Path) -> Path:
        """Supports both `root_dir/images/<class>/...` (this dataset's
        actual layout) and a flat `root_dir/<class>/...` layout, in case
        a differently-packaged copy of NEU-DET is used later."""
        if (root_dir / "images").is_dir():
            return root_dir / "images"
        return root_dir

    def _index_samples(self) -> list[tuple[str, int]]:
        samples = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.images_dir / class_name
            if not class_dir.is_dir():
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    samples.append((str(class_dir / fname), class_idx))

        if not samples:
            raise FileNotFoundError(
                f"No images found under {self.images_dir}. "
                "Check the folder layout described in docs/dataset.md — "
                "did you download and place the NEU-DET dataset yet?"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = load_grayscale(path)
        img = resize(img, self.image_size)
        img = normalize(img)  # float32 in [0, 1], shape (H, W)

        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

    def class_distribution(self) -> dict[str, int]:
        """Count of samples per class — useful for the EDA notebook."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts
