"""Object-detection metrics.

Sightline routes crews by detector confidence, so it needs mAP *and*
calibration. Written out rather than pulled from torchmetrics to keep the
eval harness importable without a deep-learning stack.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass(frozen=True)
class Detection:
    box: Box
    label: str
    score: float


@dataclass(frozen=True)
class GroundTruth:
    box: Box
    label: str


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(
    predictions: Sequence[tuple[int, Detection]],
    ground_truths: Sequence[tuple[int, GroundTruth]],
    label: str,
    iou_threshold: float,
) -> float:
    """All-point-interpolated AP for one class.

    `predictions` and `ground_truths` are (image_id, item) pairs so that a box
    can only match a target in its own image.
    """
    preds = sorted(
        (p for p in predictions if p[1].label == label),
        key=lambda p: p[1].score,
        reverse=True,
    )
    gts_by_image: dict[int, list[GroundTruth]] = {}
    for image_id, gt in ground_truths:
        if gt.label == label:
            gts_by_image.setdefault(image_id, []).append(gt)

    total_gt = sum(len(v) for v in gts_by_image.values())
    if total_gt == 0:
        return 0.0

    matched: set[tuple[int, int]] = set()
    tps: list[int] = []
    fps: list[int] = []

    for image_id, det in preds:
        candidates = gts_by_image.get(image_id, [])
        best_iou, best_idx = 0.0, -1
        for idx, gt in enumerate(candidates):
            if (image_id, idx) in matched:
                continue
            score = iou(det.box, gt.box)
            if score > best_iou:
                best_iou, best_idx = score, idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add((image_id, best_idx))
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    cum_tp = cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for tp, fp in zip(tps, fps, strict=True):
        cum_tp += tp
        cum_fp += fp
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    # Make precision monotonically decreasing, then integrate over recall.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls, strict=True):
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return ap


def mean_average_precision(
    predictions: Sequence[tuple[int, Detection]],
    ground_truths: Sequence[tuple[int, GroundTruth]],
    iou_thresholds: Sequence[float] = tuple(x / 100 for x in range(50, 100, 5)),
) -> dict[str, float]:
    """COCO-style mAP@50-95 plus mAP@50, and per-class AP@50.

    Per-class numbers are not decoration: a headline mAP that holds steady
    while one class collapses is the failure this project most needs to catch.
    """
    labels = sorted({gt.label for _, gt in ground_truths})
    if not labels:
        return {"map_50_95": 0.0, "map_50": 0.0}

    per_threshold: dict[float, list[float]] = {}
    for threshold in iou_thresholds:
        per_threshold[threshold] = [
            average_precision(predictions, ground_truths, label, threshold)
            for label in labels
        ]

    results: dict[str, float] = {
        "map_50_95": _mean(
            [ap for aps in per_threshold.values() for ap in aps]
        ),
        "map_50": _mean(per_threshold.get(0.5, [])),
    }
    for label, ap in zip(labels, per_threshold.get(0.5, []), strict=False):
        results[f"ap_50.{label}"] = ap
    return results


def mask_iou(pred: Sequence[Sequence[bool]], gold: Sequence[Sequence[bool]]) -> float:
    """IoU over two boolean masks of identical shape."""
    intersection = union = 0
    for prow, grow in zip(pred, gold, strict=True):
        for p, g in zip(prow, grow, strict=True):
            if p and g:
                intersection += 1
            if p or g:
                union += 1
    return intersection / union if union else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
