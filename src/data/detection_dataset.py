"""
PyTorch Dataset for NEU-DET object detection: returns an image together
with its bounding boxes and defect-class labels (unlike NEUDataset,
which only returns a single whole-image class label).

Folder layout expected (see docs/dataset.md):
    <root_dir>/
    ├── images/
    │   ├── crazing/crazing_1.jpg, crazing_2.jpg, ...
    │   └── ...
    └── annotations/
        ├── crazing_1.xml, crazing_2.xml, ...   (flat, not per-class!)
        └── ...
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.annotation_parser import parse_annotation
from src.utils.preprocessing import load_grayscale, normalize

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Label 0 is reserved for "background" in most detection frameworks
# (e.g. torchvision's models), so real classes start at 1.
CLASS_TO_LABEL = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}


class DetectionDataset(Dataset):
    """Loads NEU-DET images together with their bounding-box annotations.

    Args:
        root_dir: path to NEU-DET/train or NEU-DET/validation (the
            folder that contains `images/` and `annotations/`).
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.annotations_dir = self.root_dir / "annotations"

        samples = []
        for class_name in CLASS_NAMES:
            class_dir = self.images_dir / class_name
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                ann_path = self.annotations_dir / f"{img_path.stem}.xml"
                if ann_path.exists():
                    samples.append((img_path, ann_path))
        self.samples = samples

        if not self.samples:
            raise FileNotFoundError(
                f"No image/annotation pairs found under {self.root_dir}. "
                "Check that both images/ and annotations/ exist."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        img = load_grayscale(str(img_path))
        img = normalize(img)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        annotation = parse_annotation(ann_path)
        boxes, labels = [], []
        for obj in annotation["objects"]:
            boxes.append(obj["bbox"])
            labels.append(CLASS_TO_LABEL[obj["name"]])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes_tensor, "labels": labels_tensor}
        return img_tensor, target


def detection_collate_fn(batch):
    """Custom collate function for DataLoader.

    Why it's needed: the default DataLoader collate tries to torch.stack
    every sample's tensors into one big batch tensor. That works fine for
    `img_tensor` (every image is the same 200x200 shape), but each
    image's `boxes`/`labels` have a DIFFERENT length (one image might
    have 1 defect, another 3) — stacking tensors of different shapes
    raises an error.

    The fix: keep images stacked (they're all the same shape), but leave
    the per-image targets as a plain Python list of dicts instead of
    trying to stack them. This is the standard pattern used by
    torchvision's detection models (e.g. Faster R-CNN), which expect
    exactly this format: (images_tensor, list_of_target_dicts).

    Args:
        batch: a list of (img_tensor, target_dict) tuples, one per
            sample, as produced by DetectionDataset.__getitem__.

    Returns:
        (images, targets) where:
            images: a single tensor of shape (batch_size, 1, H, W)
            targets: a list of target dicts, length == batch_size,
                each unchanged from what __getitem__ returned
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = DetectionDataset("data/raw/NEU-DET/train")
    print(f"Dataset size: {len(dataset)}")

    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Target: {target}")

    # Sanity-check the collate_fn with a small batch containing images
    # that have DIFFERENT numbers of bounding boxes.
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))
    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch targets (list of {len(targets)} dicts):")
    for i, t in enumerate(targets):
        print(f"  sample {i}: {t['boxes'].shape[0]} box(es), labels={t['labels'].tolist()}")
