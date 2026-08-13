"""
Faster R-CNN model for NEU-DET defect detection, built on a
torchvision-provided, COCO-pretrained backbone.

Why Faster R-CNN (vs. building a detector from scratch): object detection
models are complex (region proposal networks, anchor boxes, non-max
suppression...) — reimplementing one from scratch is its own multi-week
project. Using torchvision's implementation and only replacing the final
classification head (transfer learning, same idea as ResNet18 in Phase 2)
is standard practice and lets us focus on applying it correctly to this
dataset rather than reinventing detection architecture internals.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int) -> nn.Module:
    """Loads a COCO-pretrained Faster R-CNN and replaces its box-prediction
    head for `num_classes` (including background as class 0).

    Args:
        num_classes: total number of classes INCLUDING background.
            For NEU-DET's 6 defect types, pass 7 (6 defects + background).
    """
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # The pretrained head was trained for 91 COCO classes — replace it
    # with a fresh head sized for our number of classes. Everything
    # else (the backbone, region proposal network) keeps its
    # pretrained, general-purpose "how to find object-like regions"
    # weights, which we fine-tune during training.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
