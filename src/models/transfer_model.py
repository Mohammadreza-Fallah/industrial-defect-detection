"""
Transfer-learning model: ImageNet-pretrained ResNet18, fine-tuned for
NEU-DET 6-class defect classification.

NEUDataset yields single-channel (grayscale) tensors, but ResNet18 expects
3-channel input normalized with ImageNet statistics — `resnet_transform`
below handles both conversions and is meant to be passed as the
`transform` argument to `NEUDataset`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# ImageNet normalization stats (standard for any torchvision pretrained model)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def resnet_transform(img_tensor: torch.Tensor) -> torch.Tensor:
    """Converts a (1, H, W) grayscale tensor in [0, 1] to a (3, H, W)
    ImageNet-normalized tensor suitable for a pretrained ResNet."""
    img_3ch = img_tensor.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
    return (img_3ch - IMAGENET_MEAN) / IMAGENET_STD


def build_resnet18(num_classes: int = 6, freeze_backbone: bool = False) -> nn.Module:
    """Loads an ImageNet-pretrained ResNet18 and replaces the final FC
    layer for `num_classes` outputs.

    Args:
        num_classes: number of output classes.
        freeze_backbone: if True, freezes all conv layers and only trains
            the new final FC layer (faster, less prone to overfitting on
            small datasets, but a lower ceiling on accuracy). If False,
            the whole network is fine-tuned end-to-end (slower, usually
            higher accuracy on a dataset this size).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final classification layer (always trainable)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
