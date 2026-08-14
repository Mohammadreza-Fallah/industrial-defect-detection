"""
Faster R-CNN model for NEU-DET defect detection, built on a
torchvision-provided, COCO-pretrained backbone.

Two backbone options are supported:

  - "resnet50"  (~42M params) — higher accuracy ceiling, but heavy.
    Needs a reasonably powerful GPU with several GB of VRAM.
  - "mobilenet" (~19M params) — much lighter and faster, designed for
    constrained hardware. Lower accuracy ceiling, but trains in a
    fraction of the time and fits in ~2GB VRAM.

Pick "mobilenet" on an entry-level laptop GPU (e.g. MX330) or CPU;
pick "resnet50" if you have a dedicated training GPU.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int, backbone: str = "mobilenet") -> nn.Module:
    """Loads a COCO-pretrained Faster R-CNN and replaces its box-prediction
    head for `num_classes` (including background as class 0).

    Args:
        num_classes: total number of classes INCLUDING background.
            For NEU-DET's 6 defect types, pass 7 (6 defects + background).
        backbone: "mobilenet" (default, light) or "resnet50" (heavy).
    """
    if backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    elif backbone == "mobilenet":
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
    else:
        raise ValueError(f"Unknown backbone {backbone!r} — use 'mobilenet' or 'resnet50'.")

    # The pretrained head was trained for 91 COCO classes — replace it
    # with a fresh head sized for our number of classes. Everything else
    # (backbone, region proposal network) keeps its pretrained weights,
    # which we fine-tune during training.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
