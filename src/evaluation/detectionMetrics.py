# Detection metrics on decoded detections: PASCAL VOC average precision (per class, mAP at one or several IoU
# thresholds), precision-recall curves, and a per-object error breakdown (correct / localization / class / missed).
# Detections are per-image (K, 6) [left, top, right, bottom, score, classId]; annotations are per-image
# (M, 5) [classId, centerX, centerY, width, height]; all coordinates normalized to [0, 1].

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.boxes import pairwiseIou, cxcywhToXyxy

# COCO-style IoU thresholds 0.50, 0.55, ..., 0.95; mAP over these rewards tight localization
COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
# Object size buckets by sqrt(area) in pixels, the same ones the ablation metrics use
SIZE_BUCKETS = (("small", 0.0, 24.0), ("medium", 24.0, 48.0), ("large", 48.0, math.inf))
ERROR_TYPES = ("correct", "localization", "classification", "missed")
# Same-class detections with IoU in [LOCALIZATION_MIN_IOU, iouThreshold) count as localization errors
LOCALIZATION_MIN_IOU = 0.1


@dataclass
class PrecisionRecallCurve:
    precision: Tensor  # (D,) precision after each detection of this class, in descending score order
    recall: Tensor  # (D,)
    scores: Tensor  # (D,)
    numObjects: int


def truthCorners(annotations: Tensor) -> Tensor:
    return cxcywhToXyxy(annotations[:, 1:5]) if annotations.numel() else torch.zeros(0, 4)


def matchClassDetections(
    detections: list[Tensor], annotations: list[Tensor], classId: int, iouThreshold: float
) -> tuple[Tensor, Tensor, int]:
    # VOC matching for one class: detections in descending score order; each one is a true positive when its
    # highest-IoU ground truth of this class has IoU >= threshold and is not already taken, otherwise a false
    # positive (duplicates of a matched object are false positives). Returns (scores, isTruePositive, numObjects)
    records = []  # (score, imageIndex, detection box)
    numObjects = 0
    for imageIndex, (imageDetections, imageAnnotations) in enumerate(zip(detections, annotations)):
        numObjects += int((imageAnnotations[:, 0].long() == classId).sum()) if imageAnnotations.numel() else 0
        for row in imageDetections[imageDetections[:, 5].long() == classId]:
            records.append((float(row[4]), imageIndex, row[:4]))
    records.sort(key=lambda record: -record[0])

    matched = {}
    truePositive = torch.zeros(len(records), dtype=torch.bool)
    for index, (_, imageIndex, box) in enumerate(records):
        imageAnnotations = annotations[imageIndex]
        if imageAnnotations.numel() == 0:
            continue
        classRows = torch.nonzero(imageAnnotations[:, 0].long() == classId, as_tuple=True)[0]
        if classRows.numel() == 0:
            continue
        ious = pairwiseIou(box.unsqueeze(0), truthCorners(imageAnnotations[classRows]))[0]
        bestIou, bestPosition = ious.max(dim=0)
        truthIndex = int(classRows[bestPosition])
        taken = matched.setdefault(imageIndex, set())
        if bestIou >= iouThreshold and truthIndex not in taken:
            taken.add(truthIndex)
            truePositive[index] = True
    scores = torch.tensor([record[0] for record in records], dtype=torch.float32)
    return scores, truePositive, numObjects


def precisionRecallCurve(
    detections: list[Tensor], annotations: list[Tensor], classId: int, iouThreshold: float = 0.5
) -> PrecisionRecallCurve:
    scores, truePositive, numObjects = matchClassDetections(detections, annotations, classId, iouThreshold)
    cumulativeTp = truePositive.float().cumsum(0)
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    precision = cumulativeTp / ranks if len(scores) else torch.zeros(0)
    recall = cumulativeTp / max(numObjects, 1) if len(scores) else torch.zeros(0)
    return PrecisionRecallCurve(precision, recall, scores, numObjects)


def averagePrecisionFromCurve(curve: PrecisionRecallCurve) -> float:
    # VOC 2010+ all-point interpolated AP: area under the precision envelope (max precision at recall >= r)
    if curve.numObjects == 0:
        return float("nan")
    if curve.recall.numel() == 0:
        return 0.0
    recall = torch.cat((torch.tensor([0.0]), curve.recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([0.0]), curve.precision, torch.tensor([0.0])))
    precision = torch.flip(torch.cummax(torch.flip(precision, [0]), 0).values, [0])
    steps = torch.nonzero(recall[1:] != recall[:-1], as_tuple=True)[0]
    return float(((recall[steps + 1] - recall[steps]) * precision[steps + 1]).sum())


def averagePrecision(
    detections: list[Tensor], annotations: list[Tensor], classId: int, iouThreshold: float = 0.5
) -> float:
    return averagePrecisionFromCurve(precisionRecallCurve(detections, annotations, classId, iouThreshold))


def meanAveragePrecision(
    detections: list[Tensor], annotations: list[Tensor], numClasses: int, iouThreshold: float = 0.5
) -> tuple[float, list[float]]:
    # (mAP over classes that have objects, per-class AP with nan for classes without objects)
    perClass = [averagePrecision(detections, annotations, classId, iouThreshold) for classId in range(numClasses)]
    present = [value for value in perClass if not math.isnan(value)]
    return (sum(present) / len(present) if present else float("nan")), perClass


def cocoStyleMap(detections: list[Tensor], annotations: list[Tensor], numClasses: int) -> float:
    # mAP averaged over IoU thresholds 0.50:0.05:0.95
    values = [meanAveragePrecision(detections, annotations, numClasses, threshold)[0]
              for threshold in COCO_IOU_THRESHOLDS]
    return sum(values) / len(values)


def objectSizeBucket(annotation: Tensor, imageSize: int) -> str:
    side = math.sqrt(max(float(annotation[3] * annotation[4]), 0.0)) * imageSize
    return next(name for name, low, high in SIZE_BUCKETS if low <= side < high)


def errorBreakdown(
    detections: list[Tensor],
    annotations: list[Tensor],
    imageSize: int,
    iouThreshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    # For every ground-truth object, the best outcome among the given detections (use an operating-point score
    # threshold before calling): correct (same class, IoU >= threshold) > localization (same class,
    # 0.1 <= IoU < threshold) > classification (other class, IoU >= threshold) > missed.
    # Returns counts keyed by size bucket and "all"
    counts = {bucket: dict.fromkeys(ERROR_TYPES, 0) for bucket in ("all", *[name for name, _, _ in SIZE_BUCKETS])}
    for imageDetections, imageAnnotations in zip(detections, annotations):
        if imageAnnotations.numel() == 0:
            continue
        if imageDetections.shape[0]:
            ious = pairwiseIou(truthCorners(imageAnnotations), imageDetections[:, :4])  # (M, K)
            sameClass = imageAnnotations[:, 0].long()[:, None] == imageDetections[:, 5].long()[None, :]
        for row, annotation in enumerate(imageAnnotations):
            outcome = "missed"
            if imageDetections.shape[0]:
                sameIou = ious[row][sameClass[row]]
                otherIou = ious[row][~sameClass[row]]
                if sameIou.numel() and sameIou.max() >= iouThreshold:
                    outcome = "correct"
                elif sameIou.numel() and sameIou.max() >= LOCALIZATION_MIN_IOU:
                    outcome = "localization"
                elif otherIou.numel() and otherIou.max() >= iouThreshold:
                    outcome = "classification"
            counts["all"][outcome] += 1
            counts[objectSizeBucket(annotation, imageSize)][outcome] += 1
    return counts
