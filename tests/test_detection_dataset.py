"""Tests for DetectionDataset and detection_collate_fn.

Builds a tiny synthetic NEU-DET-shaped dataset in a temp folder, so the
tests run anywhere without the real dataset being downloaded.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.detection_dataset import (
    CLASS_TO_LABEL,
    DetectionDataset,
    detection_collate_fn,
)


def _write_xml(path: Path, filename: str, objects: list[tuple[str, tuple]]) -> None:
    """objects: list of (class_name, (xmin, ymin, xmax, ymax))"""
    obj_xml = "".join(
        f"<object><name>{name}</name><difficult>0</difficult>"
        f"<bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for name, b in objects
    )
    path.write_text(
        f"<annotation><filename>{filename}</filename>"
        f"<size><width>200</width><height>200</height><depth>1</depth></size>"
        f"{obj_xml}</annotation>"
    )


@pytest.fixture
def fake_dataset_root(tmp_path: Path) -> Path:
    """Two images with DIFFERENT numbers of boxes — the case that breaks
    the default DataLoader collate and motivates detection_collate_fn."""
    root = tmp_path / "train"
    (root / "images" / "crazing").mkdir(parents=True)
    (root / "images" / "scratches").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)

    rng = np.random.default_rng(0)

    img = Image.fromarray(rng.integers(0, 255, (200, 200), dtype=np.uint8))
    img.save(root / "images" / "crazing" / "crazing_1.jpg")
    _write_xml(
        root / "annotations" / "crazing_1.xml",
        "crazing_1.jpg",
        [("crazing", (2, 2, 193, 194))],
    )

    img2 = Image.fromarray(rng.integers(0, 255, (200, 200), dtype=np.uint8))
    img2.save(root / "images" / "scratches" / "scratches_1.jpg")
    _write_xml(
        root / "annotations" / "scratches_1.xml",
        "scratches_1.jpg",
        [
            ("scratches", (1, 1, 10, 10)),
            ("scratches", (20, 20, 30, 30)),
            ("scratches", (40, 40, 50, 50)),
        ],
    )

    return root


def test_dataset_finds_image_annotation_pairs(fake_dataset_root):
    dataset = DetectionDataset(str(fake_dataset_root))
    assert len(dataset) == 2


def test_getitem_returns_image_and_target(fake_dataset_root):
    dataset = DetectionDataset(str(fake_dataset_root))
    img, target = dataset[0]

    # 3 channels because the pretrained detection backbone expects RGB
    assert img.shape == (3, 200, 200)
    assert img.dtype == torch.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0

    assert set(target.keys()) == {"boxes", "labels"}
    assert target["boxes"].shape[1] == 4
    assert target["boxes"].shape[0] == target["labels"].shape[0]


def test_single_channel_mode(fake_dataset_root):
    dataset = DetectionDataset(str(fake_dataset_root), to_3channel=False)
    img, _ = dataset[0]
    assert img.shape == (1, 200, 200)


def test_labels_start_at_one_not_zero(fake_dataset_root):
    """Label 0 is reserved for background by torchvision detection models."""
    dataset = DetectionDataset(str(fake_dataset_root))
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        assert int(target["labels"].min()) >= 1
    assert CLASS_TO_LABEL["crazing"] == 1


def test_missing_annotation_is_skipped(fake_dataset_root):
    """An image with no matching XML should not appear in the samples."""
    orphan = fake_dataset_root / "images" / "crazing" / "crazing_999.jpg"
    Image.fromarray(np.zeros((200, 200), dtype=np.uint8)).save(orphan)

    dataset = DetectionDataset(str(fake_dataset_root))
    assert len(dataset) == 2  # still 2, the orphan is ignored


def test_empty_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        DetectionDataset(str(tmp_path))


def test_collate_fn_handles_variable_box_counts(fake_dataset_root):
    """The whole point of the custom collate: images stack, targets don't."""
    dataset = DetectionDataset(str(fake_dataset_root))
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn
    )
    images, targets = next(iter(loader))

    assert images.shape == (2, 3, 200, 200)
    assert isinstance(targets, list)
    assert len(targets) == 2

    box_counts = sorted(t["boxes"].shape[0] for t in targets)
    assert box_counts == [1, 3]  # different lengths survived collation
