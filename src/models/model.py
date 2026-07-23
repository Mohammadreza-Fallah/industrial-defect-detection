"""
Baseline CNN for NEU-DET 6-class surface defect classification.

Deliberately simple — this is a baseline to establish a performance floor
before moving to a more capable architecture (transfer learning, or a
segmentation/detection model) in a later phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """A small CNN: 3 conv blocks + 2 FC layers.

    Input: (batch, 1, H, W) grayscale images (default H=W=200).
    Output: (batch, num_classes) raw logits.
    """

    def __init__(self, num_classes: int = 6, image_size: int = 200):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2

            # Block 2: 16 -> 32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4

            # Block 3: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/8
        )

        reduced = image_size // 8
        flat_dim = 64 * reduced * reduced

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
