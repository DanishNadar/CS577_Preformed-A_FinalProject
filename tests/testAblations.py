import json
import math

import pytest
import torch

from src.ablations.analysis import aggregate, analyze, runSummary
from src.ablations.metrics import (
    TrainingGradientTracker,
    detectionMetrics,
    responsibleBoxMetrics,
    sizeBucketMasks,
    termGradientNorms,
)
from src.ablations.runAblations import isRunComplete, runAblations
from src.ablations.variants import (
    PROPOSAL_VARIANTS,
    VARIANTS,
    assertFairComparison,
    buildVariantConfig,
    dataRootForSeed,
    variantOverrides,
)
from src.datasets.targetEncoder import encodeTarget
from src.losses.yoloLoss import YoloLoss, YoloLossConfig
from src.models.yoloModel import ModelConfig, createModel


def targetFor(objects):
    # objects: list of [classId, cx, cy, w, h] -> (1, 7, 7, 8) target
    target, _ = encodeTarget(torch.tensor(objects, dtype=torch.float32), 7, 3)
    return target.unsqueeze(0)


def perfectPrediction(target):
    # Both predictors copy the truth box, confidence 1 on occupied cells, one-hot class
    predictions = torch.zeros(1, 7, 7, 13)
    box = target[..., :4]
    predictions[..., 0:4] = box
    predictions[..., 5:9] = box
    predictions[..., 4] = target[..., 4]
    predictions[..., 10:13] = target[..., 5:]
    return predictions


def testProposalVariantsMatchProposal():
    assert PROPOSAL_VARIANTS == ("paper", "unitWeights", "directWh")
    paper, unit, direct = (VARIANTS[name] for name in PROPOSAL_VARIANTS)
    assert (paper.lambdaCoord, paper.lambdaNoobj, paper.useSqrtWidthHeight) == (5.0, 0.5, True)
    assert (unit.lambdaCoord, unit.lambdaNoobj, unit.useSqrtWidthHeight) == (1.0, 1.0, True)
    assert (direct.lambdaCoord, direct.lambdaNoobj, direct.useSqrtWidthHeight) == (5.0, 0.5, False)


def testVariantConfigsDifferOnlyInLoss():
    configs = [buildVariantConfig(name, seed) for seed in (577, 1) for name in VARIANTS]
    assertFairComparison(configs)
    paper = buildVariantConfig("paper", 577)
    assert paper.loss == YoloLossConfig()
    assert buildVariantConfig("directWh", 577).loss.useSqrtWidthHeight is False
    assert paper.data.synthetic.root == "data/synthetic"
    assert buildVariantConfig("paper", 1).data.synthetic.root == dataRootForSeed(1) != "data/synthetic"


def testFairnessCheckRejectsMismatchedSettings():
    configs = [buildVariantConfig("paper", 577), buildVariantConfig("unitWeights", 577, extraOverrides={"training.epochs": 3})]
    with pytest.raises(ValueError, match="training"):
        assertFairComparison(configs)


def testSharedOverridesCannotChangeLoss():
    with pytest.raises(ValueError, match="loss"):
        variantOverrides("paper", extraOverrides={"loss.lambdaNoobj": 2.0})


def testResponsibleMetricsPerfectPrediction():
    target = targetFor([[0, 0.3, 0.3, 0.1, 0.1], [1, 0.7, 0.7, 0.6, 0.5]])
    metrics = responsibleBoxMetrics(perfectPrediction(target), target, numBoxes=2, imageSize=112)
    assert metrics["valMeanIou"] == pytest.approx(1.0)
    assert metrics["valIouAt50"] == 1.0
    assert metrics["valRelativeSizeError"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["valClassAccuracy"] == 1.0
    assert (metrics["valCountSmall"], metrics["valCountMedium"], metrics["valCountLarge"]) == (1, 0, 1)
    assert math.isnan(metrics["valMeanIouMedium"])


def testSizeBucketsUsePixelSide():
    # sqrt(w*h) * 112: 0.1 -> 11.2 px small, 0.3 -> 33.6 px medium, 0.5 -> 56 px large
    target = targetFor([[0, 0.1, 0.1, 0.1, 0.1], [0, 0.5, 0.5, 0.3, 0.3], [0, 0.9, 0.9, 0.5, 0.5]])
    masks = sizeBucketMasks(target[..., :4], target[..., 4], 112)
    assert [int(mask.sum()) for mask in masks.values()] == [1, 1, 1]
    assert masks["Small"][0, 0, 0] and masks["Medium"][0, 3, 3] and masks["Large"][0, 6, 6]


def testResponsibleMetricsKnownIou():
    # Truth 0.2 x 0.2; responsible predictor shifted by half a width -> IoU 1/3, relative size error 0
    target = targetFor([[0, 0.5, 0.5, 0.2, 0.2]])
    predictions = torch.zeros(1, 7, 7, 13)
    predictions[0, 3, 3, :5] = torch.tensor([0.6 * 7 - 3, 0.5, 0.2, 0.2, 1.0])
    predictions[0, 3, 3, 5:10] = torch.tensor([0.5, 0.5, 0.01, 0.01, 0.0])
    metrics = responsibleBoxMetrics(predictions, target, 2, 112)
    assert metrics["valMeanIou"] == pytest.approx(1 / 3, rel=1e-5)
    assert metrics["valIouAt50"] == 0.0
    assert metrics["valRelativeSizeError"] == pytest.approx(0.0, abs=1e-6)


def testDetectionMetricsCountsClassMatches():
    target = targetFor([[0, 0.3, 0.3, 0.2, 0.2], [2, 0.7, 0.7, 0.3, 0.3]])
    predictions = perfectPrediction(target)
    annotations = [torch.tensor([[0, 0.3, 0.3, 0.2, 0.2], [2, 0.7, 0.7, 0.3, 0.3]])]
    assert detectionMetrics(predictions, annotations, 2) == {"valRecallAt50": 1.0, "valPrecisionAt50": 1.0}
    # Wrong class on the second object: one hit, one false positive, one miss
    predictions[0, 4, 4, 10:13] = torch.tensor([1.0, 0.0, 0.0])
    assert detectionMetrics(predictions, annotations, 2) == {"valRecallAt50": 0.5, "valPrecisionAt50": 0.5}


def testTermGradientNormsMatchManualBackward():
    model = createModel(ModelConfig(dropout=0.0), seed=0)
    images = torch.rand(2, 3, 112, 112, generator=torch.Generator().manual_seed(0))
    targets = torch.cat([targetFor([[0, 0.3, 0.4, 0.2, 0.3]]), targetFor([[1, 0.6, 0.5, 0.4, 0.2]])])
    lossFn = YoloLoss(YoloLossConfig(), 2)
    norms = termGradientNorms(model, images, targets, lossFn)

    model.zero_grad()
    lossFn(model(images), targets)["noObjectConfidence"].backward()
    manual = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    assert norms["probeGradNormNoObjectConfidence"] == pytest.approx(manual.item(), rel=1e-4)
    assert norms["probeNoobjToObjGradRatio"] == pytest.approx(
        norms["probeGradNormNoObjectConfidence"] / norms["probeGradNormObjectConfidence"])
    assert model.training  # restored


def testGradientTrackerRecordsEachStep():
    parameter = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with TrainingGradientTracker() as tracker:
        for scale in (1.0, 2.0):
            optimizer.zero_grad()
            (scale * torch.tensor([3.0, 4.0, 0.0]) @ parameter).backward()
            optimizer.step()
    optimizer.step()  # after exit: not recorded
    stats = tracker.popEpochStats()
    assert stats["trainGradNormMean"] == pytest.approx(7.5)
    assert stats["trainGradNormMax"] == pytest.approx(10.0)
    assert stats["trainNonFiniteSteps"] == 0
    assert tracker.popEpochStats()["trainNonFiniteSteps"] == 0  # emptied


def testRunAndAnalyzeTinyAblation(tmp_path):
    shared = {
        "device": "cpu",
        "data.synthetic.root": str(tmp_path / "synthetic"),
        "data.synthetic.numTrain": 32,
        "data.synthetic.numVal": 16,
        "training.batchSize": 16,
        "training.epochs": 2,
    }
    outputRoot, checkpointRoot = tmp_path / "results", tmp_path / "checkpoints"
    summaries = runAblations(["paper", "directWh"], [577], jobs=1, outputRoot=str(outputRoot),
                             checkpointRoot=str(checkpointRoot), sharedOverrides=shared)
    assert len(summaries) == 2
    history = json.loads((outputRoot / "seed577/directWh/history.json").read_text())
    assert len(history) == 2
    for key in ("valMeanIou", "valPaperTotal", "valRecallAt50", "trainGradNormMean", "probeNoobjToObjGradRatio"):
        assert math.isfinite(history[-1][key])
    assert isRunComplete(buildVariantConfig("paper", 577, outputRoot=outputRoot, checkpointRoot=checkpointRoot,
                                            extraOverrides=shared))
    # Already-complete runs are skipped
    assert runAblations(["paper"], [577], outputRoot=str(outputRoot), checkpointRoot=str(checkpointRoot),
                        sharedOverrides=shared) == []

    reportPath = analyze(outputRoot, tmp_path / "figures")
    report = reportPath.read_text()
    assert "`paper`" in report and "`directWh`" in report
    assert (outputRoot / "variantSummary.csv").exists()


def testPairedDifferenceAgainstPaper():
    def fakeRun(variant, seed, iou):
        row = {"epoch": 1, "valMeanIou": iou, "valPaperTotal": 1.0, "trainGradNormMean": 1.0,
               "trainGradNormMax": 1.0, "trainGradNormStd": 0.0, "trainNonFiniteSteps": 0,
               "probeNoobjToObjGradRatio": 1.0, "epochSeconds": 1.0}
        return {"variant": variant, "seed": seed, "history": [row]}
    runs = [fakeRun("paper", 1, 0.5), fakeRun("paper", 2, 0.6), fakeRun("unitWeights", 1, 0.4),
            fakeRun("unitWeights", 2, 0.7)]
    rows = {row["variant"]: row for row in aggregate([runSummary(run) for run in runs])}
    assert rows["unitWeights"]["valMeanIouDiffVsPaper"] == pytest.approx(0.0)
    assert rows["unitWeights"]["valMeanIouSeedsBelowPaper"] == 1
    assert rows["paper"]["valMeanIouMean"] == pytest.approx(0.55)
