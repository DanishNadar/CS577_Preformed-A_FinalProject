import json
import math

import pytest
import torch

from src.ablations.runAblations import runAblations
from src.evaluation.detectionMetrics import (
    averagePrecision,
    cocoStyleMap,
    errorBreakdown,
    meanAveragePrecision,
    precisionRecallCurve,
)
from src.evaluation.evaluate import evaluateAblations, evaluateCheckpoint, evaluatePredictions
from src.datasets.targetEncoder import encodeTarget


def detection(left, top, right, bottom, score, classId):
    return [left, top, right, bottom, score, classId]


def annotation(classId, left, top, right, bottom):
    # [classId, cx, cy, w, h] from corners
    return [classId, (left + right) / 2, (top + bottom) / 2, right - left, bottom - top]


def testPerfectDetectionsGiveApOne():
    annotations = [torch.tensor([annotation(0, 0.1, 0.1, 0.3, 0.3), annotation(1, 0.5, 0.5, 0.9, 0.9)])]
    detections = [torch.tensor([detection(0.1, 0.1, 0.3, 0.3, 0.9, 0), detection(0.5, 0.5, 0.9, 0.9, 0.8, 1)])]
    mapValue, perClass = meanAveragePrecision(detections, annotations, numClasses=3)
    assert perClass[0] == pytest.approx(1.0)
    assert perClass[1] == pytest.approx(1.0)
    assert math.isnan(perClass[2])  # no objects of class 2: excluded from the mean
    assert mapValue == pytest.approx(1.0)
    assert cocoStyleMap(detections, annotations, 3) == pytest.approx(1.0)


def testApHandComputedRanking():
    # Two objects of class 0 in two images. Ranked detections: TP (0.9), FP (0.8), TP (0.7)
    # Precision/recall: (1, 0.5), (0.5, 0.5), (2/3, 1). Envelope: 1 for r in (0, .5], 2/3 for (.5, 1]
    # AP = 0.5 * 1 + 0.5 * 2/3 = 5/6
    annotations = [torch.tensor([annotation(0, 0.0, 0.0, 0.2, 0.2)]), torch.tensor([annotation(0, 0.5, 0.5, 0.7, 0.7)])]
    detections = [
        torch.tensor([detection(0.0, 0.0, 0.2, 0.2, 0.9, 0), detection(0.8, 0.8, 1.0, 1.0, 0.8, 0)]),
        torch.tensor([detection(0.5, 0.5, 0.7, 0.7, 0.7, 0)]),
    ]
    curve = precisionRecallCurve(detections, annotations, 0)
    assert curve.precision.tolist() == pytest.approx([1.0, 0.5, 2 / 3])
    assert curve.recall.tolist() == pytest.approx([0.5, 0.5, 1.0])
    assert averagePrecision(detections, annotations, 0) == pytest.approx(5 / 6)


def testDuplicateDetectionIsFalsePositive():
    annotations = [torch.tensor([annotation(0, 0.1, 0.1, 0.5, 0.5)])]
    detections = [torch.tensor([detection(0.1, 0.1, 0.5, 0.5, 0.9, 0), detection(0.1, 0.1, 0.5, 0.5, 0.8, 0)])]
    curve = precisionRecallCurve(detections, annotations, 0)
    assert curve.precision.tolist() == pytest.approx([1.0, 0.5])
    assert averagePrecision(detections, annotations, 0) == pytest.approx(1.0)  # envelope keeps the first point


def testIouThresholdControlsMatch():
    # Box shifted by 0.2: IoU = 0.15 / 0.35 ~= 0.43 with the truth: miss at 0.5, hit at 0.4
    annotations = [torch.tensor([annotation(0, 0.0, 0.0, 0.5, 0.5)])]
    detections = [torch.tensor([detection(0.2, 0.0, 0.7, 0.5, 0.9, 0)])]
    assert averagePrecision(detections, annotations, 0, iouThreshold=0.5) == 0.0
    assert averagePrecision(detections, annotations, 0, iouThreshold=0.4) == pytest.approx(1.0)


def testErrorBreakdownCategories():
    imageSize = 112
    # small correct, medium localization error, large classification error, one small missed
    annotations = [torch.tensor([
        annotation(0, 0.0, 0.0, 0.15, 0.15),  # 16.8 px: small
        annotation(1, 0.3, 0.3, 0.6, 0.6),  # 33.6 px: medium
        annotation(2, 0.0, 0.5, 0.5, 1.0),  # 56 px: large
        annotation(0, 0.8, 0.0, 0.95, 0.15),  # small, nothing near it
    ])]
    detections = [torch.tensor([
        detection(0.0, 0.0, 0.15, 0.15, 0.9, 0),
        detection(0.4, 0.4, 0.7, 0.7, 0.9, 1),  # same class, IoU ~0.29
        detection(0.0, 0.5, 0.5, 1.0, 0.9, 0),  # wrong class, IoU 1
    ])]
    counts = errorBreakdown(detections, annotations, imageSize)
    assert counts["all"] == {"correct": 1, "localization": 1, "classification": 1, "missed": 1}
    assert counts["small"] == {"correct": 1, "localization": 0, "classification": 0, "missed": 1}
    assert counts["medium"]["localization"] == 1
    assert counts["large"]["classification"] == 1


def testEvaluatePredictionsOnPerfectModelOutput():
    objects = torch.tensor([[0, 0.3, 0.3, 0.2, 0.2], [2, 0.75, 0.7, 0.4, 0.5]], dtype=torch.float32)
    target, _ = encodeTarget(objects, 7, 3)
    targets = target.unsqueeze(0)
    predictions = torch.zeros(1, 7, 7, 13)
    predictions[..., 0:4] = targets[..., :4]
    predictions[..., 5:9] = targets[..., :4]
    predictions[..., 4] = targets[..., 4]
    predictions[..., 10:13] = targets[..., 5:]
    result = evaluatePredictions(predictions, targets, [objects], numBoxes=2, imageSize=112,
                                 classNames=["circle", "square", "triangle"])
    assert result.metrics["mAP50"] == pytest.approx(1.0)
    assert result.metrics["mAP50to95"] == pytest.approx(1.0)
    assert math.isnan(result.metrics["AP50_square"])
    assert result.metrics["MeanIou"] == pytest.approx(1.0)
    assert result.errors["all"]["correct"] == 2


def testEvaluateTinyAblationEndToEnd(tmp_path):
    shared = {
        "device": "cpu",
        "data.synthetic.root": str(tmp_path / "synthetic"),
        "data.synthetic.numTrain": 32,
        "data.synthetic.numVal": 16,
        "training.batchSize": 16,
        "training.epochs": 1,
    }
    checkpointRoot = tmp_path / "checkpoints"
    runAblations(["paper", "unitWeights"], [577], outputRoot=str(tmp_path / "results"),
                 checkpointRoot=str(checkpointRoot), sharedOverrides=shared)

    single = evaluateCheckpoint(checkpointRoot / "seed577/paper/last.pt")
    assert 0.0 <= single.metrics["mAP50"] <= 1.0
    assert sum(single.errors["all"].values()) == sum(rows.shape[0] for rows in single.annotations)

    outputRoot = tmp_path / "evaluation"
    summaryPath = evaluateAblations(checkpointRoot, outputRoot, figureDir=tmp_path / "figures")
    summary = summaryPath.read_text()
    assert "`paper`" in summary and "`unitWeights`" in summary and "mAP@0.5" in summary
    assert (outputRoot / "runMetrics.csv").exists() and (outputRoot / "errorBreakdown.csv").exists()
    curves = json.loads((outputRoot / "precisionRecall.json").read_text())
    assert set(curves) == {"paper/seed577", "unitWeights/seed577"}
    for name in ("mapByVariant.png", "precisionRecall.png", "errorBreakdown.png", "detectionExamples.png"):
        assert (tmp_path / "figures" / name).exists()
