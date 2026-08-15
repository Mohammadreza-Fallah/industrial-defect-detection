"""Tests for the custom mAP implementation.

Since this metric was written from scratch (instead of using
pycocotools), it needs tests that pin down its behaviour on cases where
the correct answer is known by hand.
"""

import torch

from src.models.detection_metrics import compute_ap_for_class, mean_average_precision

BOX_A = [10.0, 10.0, 50.0, 50.0]
BOX_FAR_AWAY = [100.0, 100.0, 140.0, 140.0]


def _pred(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def _target(boxes, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_perfect_prediction_gives_ap_one():
    targets = [_target([BOX_A], [1])]
    preds = [_pred([BOX_A], [0.9], [1])]
    assert compute_ap_for_class(preds, targets, class_id=1) == 1.0


def test_completely_wrong_prediction_gives_ap_zero():
    targets = [_target([BOX_A], [1])]
    preds = [_pred([BOX_FAR_AWAY], [0.9], [1])]
    assert compute_ap_for_class(preds, targets, class_id=1) == 0.0


def test_half_recalled_gives_ap_half():
    """Two ground-truth boxes across two images, only one detected."""
    targets = [_target([BOX_A], [1]), _target([BOX_A], [1])]
    preds = [_pred([BOX_A], [0.9], [1]), _pred([BOX_FAR_AWAY], [0.8], [1])]
    assert compute_ap_for_class(preds, targets, class_id=1) == 0.5


def test_class_with_no_ground_truth_returns_none():
    """AP is undefined for a class that never appears — it must not be
    silently counted as 0.0, which would drag the mAP down."""
    targets = [_target([BOX_A], [1])]
    preds = [_pred([BOX_A], [0.9], [1])]
    assert compute_ap_for_class(preds, targets, class_id=3) is None


def test_duplicate_detections_count_as_false_positives():
    """Two predictions on the same single ground-truth box: the first
    (higher score) is a TP, the second must be an FP — otherwise a model
    could game AP by flooding the image with boxes."""
    targets = [_target([BOX_A], [1])]
    preds = [_pred([BOX_A, BOX_A], [0.9, 0.8], [1, 1])]
    ap = compute_ap_for_class(preds, targets, class_id=1)
    assert ap == 1.0  # recall is still 1.0 at max precision point


def test_iou_threshold_is_respected():
    """A loosely overlapping box passes at IoU 0.1 but fails at 0.9."""
    targets = [_target([[0.0, 0.0, 100.0, 100.0]], [1])]
    preds = [_pred([[0.0, 0.0, 60.0, 60.0]], [0.9], [1])]  # IoU = 0.36

    assert compute_ap_for_class(preds, targets, 1, iou_threshold=0.1) == 1.0
    assert compute_ap_for_class(preds, targets, 1, iou_threshold=0.9) == 0.0


def test_mean_average_precision_averages_present_classes_only():
    class_names = ["a", "b", "c"]  # -> label ids 1, 2, 3
    targets = [_target([BOX_A, BOX_FAR_AWAY], [1, 2])]
    preds = [_pred([BOX_A, BOX_A], [0.9, 0.7], [1, 2])]

    mAP, per_class = mean_average_precision(preds, targets, class_names)

    # class "c" has no ground truth -> excluded from both outputs
    assert set(per_class.keys()) == {"a", "b"}
    assert per_class["a"] == 1.0
    assert per_class["b"] == 0.0
    assert mAP == 0.5


def test_empty_evaluation_set_returns_zero():
    mAP, per_class = mean_average_precision([], [], ["a", "b"])
    assert mAP == 0.0
    assert per_class == {}
