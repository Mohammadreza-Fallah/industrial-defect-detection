"""
Lightweight mean Average Precision (mAP) calculator for object detection,
implemented without pycocotools (which requires a C build step that's
been unreliable in this project's Windows environment — see the earlier
torch/DLL issues). Uses torchvision.ops.box_iou for the IoU math and a
standard VOC-style (continuous, precision-envelope) AP calculation.

This trades a bit of precision-matching-COCO's-exact-numbers for
reliability and transparency — good enough to compare model checkpoints
against each other, which is all we need here.
"""

from __future__ import annotations

import torch
from torchvision.ops import box_iou


def compute_ap_for_class(
    all_predictions: list[dict],
    all_targets: list[dict],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float | None:
    """Computes Average Precision for a single class across a whole
    evaluation set.

    Args:
        all_predictions: one dict per image, each with 'boxes' (N,4),
            'scores' (N,), 'labels' (N,) tensors — model output.
        all_targets: one dict per image, each with 'boxes' (M,4),
            'labels' (M,) tensors — ground truth.
        class_id: which class label to compute AP for.
        iou_threshold: IoU above which a prediction counts as a match.

    Returns:
        AP as a float, or None if this class has no ground-truth
        instances in the evaluation set (AP is undefined in that case).
    """
    # Collect every prediction of this class, across all images, so we
    # can rank them globally by confidence score (standard AP procedure).
    pred_records = []  # (image_idx, box, score)
    for img_idx, pred in enumerate(all_predictions):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
            pred_records.append((img_idx, box, score.item()))
    pred_records.sort(key=lambda r: r[2], reverse=True)

    # Ground-truth boxes per image for this class, with a "used" flag so
    # each GT box can only be matched by one prediction (avoids
    # rewarding multiple detections of the same object).
    gt_by_image = {}
    num_gt = 0
    for img_idx, target in enumerate(all_targets):
        mask = target["labels"] == class_id
        boxes = target["boxes"][mask]
        gt_by_image[img_idx] = {"boxes": boxes, "matched": [False] * boxes.shape[0]}
        num_gt += boxes.shape[0]

    if num_gt == 0:
        return None

    tp = torch.zeros(len(pred_records))
    fp = torch.zeros(len(pred_records))

    for i, (img_idx, box, _score) in enumerate(pred_records):
        gt_info = gt_by_image.get(img_idx)
        if gt_info is None or gt_info["boxes"].shape[0] == 0:
            fp[i] = 1
            continue

        ious = box_iou(box.unsqueeze(0), gt_info["boxes"])[0]
        best_iou, best_idx = ious.max(0)
        best_idx = best_idx.item()

        if best_iou.item() >= iou_threshold and not gt_info["matched"][best_idx]:
            tp[i] = 1
            gt_info["matched"][best_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    recalls = (tp_cumsum / num_gt).tolist()
    precisions = (tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)).tolist()

    # VOC-style continuous AP: pad with sentinel points, take the
    # precision envelope (non-increasing from the right), then
    # integrate precision over recall.
    recalls = [0.0] + recalls + [recalls[-1] if recalls else 0.0]
    precisions = [0.0] + precisions + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def mean_average_precision(
    all_predictions: list[dict],
    all_targets: list[dict],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Computes mAP@iou_threshold, averaged over classes that have at
    least one ground-truth instance in the evaluation set.

    Args:
        class_names: ordered list of class names, where class_names[i]
            corresponds to label id (i + 1) — label 0 is background
            (see CLASS_TO_LABEL in detection_dataset.py).

    Returns:
        (mAP, per_class_ap) where per_class_ap maps class name -> AP
        (only for classes with ground-truth instances present).
    """
    per_class_ap = {}
    for idx, name in enumerate(class_names):
        class_id = idx + 1  # label 0 is background
        ap = compute_ap_for_class(all_predictions, all_targets, class_id, iou_threshold)
        if ap is not None:
            per_class_ap[name] = ap

    if not per_class_ap:
        return 0.0, {}

    mAP = sum(per_class_ap.values()) / len(per_class_ap)
    return mAP, per_class_ap
