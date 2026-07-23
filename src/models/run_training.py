"""
Standalone CLI entry point to train the baseline classifier from the
terminal (reproducible, scriptable, usable in CI later) — same logic
that notebooks/02_baseline_model.ipynb uses interactively.

Usage:
    python -m src.models.run_training
    python -m src.models.run_training --epochs 20 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import NEUDataset
from src.models.model import BaselineCNN
from src.models.train import fit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline NEU-DET classifier")
    parser.add_argument("--data-root", default="data/raw/NEU-DET")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=200)
    parser.add_argument("--checkpoint", default="models/baseline_cnn.pt")
    parser.add_argument("--history-out", default="models/baseline_history.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = NEUDataset(
        f"{args.data_root}/train", image_size=(args.image_size, args.image_size)
    )
    val_set = NEUDataset(
        f"{args.data_root}/validation", image_size=(args.image_size, args.image_size)
    )
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = BaselineCNN(num_classes=6, image_size=args.image_size).to(device)

    history = fit(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
    )

    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {args.history_out}")


if __name__ == "__main__":
    main()
