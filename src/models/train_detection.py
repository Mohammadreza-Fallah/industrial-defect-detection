"""
Training and evaluation loop for the Faster R-CNN defect detector.

Key difference from the classification loops (src/models/train.py):
Faster R-CNN's forward pass behaves differently depending on mode:
  - model.train() + model(images, targets)  -> returns a dict of losses
    (no explicit loss function needed — it's built into the model)
  - model.eval()  + model(images)           -> returns predicted boxes/
    scores/labels per image (no targets needed)

Also, torchvision detection models expect `images` as a LIST of
(C, H, W) tensors (images can vary in size across the list), not one
stacked (B, C, H, W) tensor — so we unstack the batch our collate_fn
produces before passing it to the model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.detection_metrics import mean_average_precision


def train_one_epoch_detection(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Runs one training epoch. Returns the average total loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def evaluate_detection(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Runs the model in inference mode over the whole loader and
    computes mAP against the ground-truth targets.

    Returns:
        (mAP, per_class_ap)
    """
    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = [img.to(device) for img in images]
        predictions = model(images)  # no targets in eval mode

        all_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions])
        all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

    return mean_average_precision(all_predictions, all_targets, class_names, iou_threshold)


def fit_detection(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    epochs: int = 10,
    lr: float = 5e-3,
    checkpoint_path: str | None = "models/fasterrcnn.pt",
) -> dict:
    """Full training loop with best-checkpoint saving (by mAP).

    Uses SGD with momentum, the optimizer choice torchvision's own
    detection training references use (Faster R-CNN tends to train
    less reliably with Adam than the classifiers we've used so far).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    history = {"train_loss": [], "val_mAP": []}
    best_map = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_detection(model, train_loader, optimizer, device)
        val_map, per_class_ap = evaluate_detection(model, val_loader, device, class_names)

        history["train_loss"].append(train_loss)
        history["val_mAP"].append(val_map)

        print(f"Epoch {epoch:2d}/{epochs} | train_loss={train_loss:.4f} | val_mAP={val_map:.4f}")
        for name, ap in per_class_ap.items():
            print(f"    {name:<16}: AP={ap:.4f}")

        if checkpoint_path and val_map > best_map:
            best_map = val_map
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> saved new best checkpoint (val_mAP={val_map:.4f})")

    return history
