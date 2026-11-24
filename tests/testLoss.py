import math

import pytest
import torch

from src.datasets.targetEncoder import encodeTarget
from src.losses.yoloLoss import (
    LOSS_COMPONENTS,
    YoloLoss,
    YoloLossConfig,
    assignResponsibleBoxes,
    noObjectConfidenceLoss,
    safeSqrt,
)
from src.models.yoloModel import ModelConfig, createModel


def singleObjectTarget(box=(0.5, 0.5, 0.4, 0.2), classId=1) -> torch.Tensor:
    # Batch of one target with a single object centered in cell (3, 3)
    target, _ = encodeTarget(torch.tensor([[classId, *box]]), 7, 3)
    return target.unsqueeze(0)


def predictionWithBoxes(firstBox, secondBox, confidences=(0.0, 0.0), classProbabilities=(0.0, 1.0, 0.0)):
    # Prediction tensor (1, 7, 7, 13), zero everywhere except cell (3, 3)
    predictions = torch.zeros(1, 7, 7, 13)
    predictions[0, 3, 3] = torch.tensor(
        [*firstBox, confidences[0], *secondBox, confidences[1], *classProbabilities]
    )
    return predictions


def fixedBatch(batchSize=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(batchSize, 3, 112, 112, generator=generator)
    targets = torch.stack([
        encodeTarget(torch.tensor([[i % 3, 0.2 + 0.15 * i, 0.3 + 0.1 * i, 0.1 + 0.05 * i, 0.2]]), 7, 3)[0]
        for i in range(batchSize)
    ])
    return images, targets


def testResponsibilityPicksHighestIou():
    target = singleObjectTarget()
    # Box 0 is far off in size, box 1 matches the truth exactly -> box 1 responsible, IoU 1
    predictions = predictionWithBoxes((0.5, 0.5, 0.05, 0.05), (0.5, 0.5, 0.4, 0.2))
    predictedBoxes = predictions[..., :10].reshape(1, 7, 7, 2, 5)[..., :4]
    responsibleMask, boxIous = assignResponsibleBoxes(predictedBoxes, target[..., :4], target[..., 4])
    assert responsibleMask[0, 3, 3].tolist() == [0.0, 1.0]
    assert responsibleMask.sum() == 1
    assert boxIous[0, 3, 3, 1] == pytest.approx(1.0)
    # 0.05 x 0.05 box fully inside the 0.4 x 0.2 truth: IoU = 0.0025 / 0.08
    assert boxIous[0, 3, 3, 0] == pytest.approx(0.0025 / 0.08, rel=1e-5)


def testResponsibilityKnownHalfOverlap():
    target = singleObjectTarget(box=(0.5, 0.5, 0.2, 0.2))
    # Box 0 is shifted right by half its width: intersection 0.1*0.2, union 0.06 -> IoU 1/3
    shiftedX = 0.6 * 7 - 3  # image center x = 0.6 expressed relative to cell column 3
    predictions = predictionWithBoxes((shiftedX, 0.5, 0.2, 0.2), (0.5, 0.5, 0.01, 0.01))
    predictedBoxes = predictions[..., :10].reshape(1, 7, 7, 2, 5)[..., :4]
    responsibleMask, boxIous = assignResponsibleBoxes(predictedBoxes, target[..., :4], target[..., 4])
    assert responsibleMask[0, 3, 3].tolist() == [1.0, 0.0]
    assert boxIous[0, 3, 3, 0] == pytest.approx(1 / 3, rel=1e-5)


def testLossHandComputedExample():
    target = singleObjectTarget(box=(0.5, 0.5, 0.36, 0.16), classId=1)
    # Responsible box 1: x off by 0.1, sqrt(w) off by 0.1 (0.25 vs 0.36), confidence 0.5 vs IoU
    predictions = predictionWithBoxes(
        (0.5, 0.5, 0.01, 0.01), (0.6, 0.5, 0.25, 0.16), confidences=(0.2, 0.5), classProbabilities=(0.1, 0.8, 0.1)
    )
    predictions[0, 0, 0, 4] = 0.4  # a confident predictor in an empty cell
    lossBreakdown = YoloLoss(YoloLossConfig(sqrtEpsilon=0.0))(predictions, target)

    # Box 1 corners: x from (3.6/7 - 0.125) to (3.6/7 + 0.125), truth from 0.32 to 0.68, same height
    predictedCenter = 3.6 / 7
    overlapWidth = min(predictedCenter + 0.125, 0.68) - max(predictedCenter - 0.125, 0.32)
    expectedIou = overlapWidth * 0.16 / (0.25 * 0.16 + 0.36 * 0.16 - overlapWidth * 0.16)

    expected = {
        "coordinateXy": 5 * 0.1**2,
        "coordinateWh": 5 * (0.5 - 0.6) ** 2,
        "objectConfidence": (0.5 - expectedIou) ** 2,
        "noObjectConfidence": 0.5 * (0.2**2 + 0.4**2),
        "classification": 0.1**2 + 0.2**2 + 0.1**2,
    }
    for name, value in expected.items():
        assert lossBreakdown[name].item() == pytest.approx(value, rel=1e-4), name
    assert lossBreakdown["total"].item() == pytest.approx(sum(expected.values()), rel=1e-4)


def testComponentsSumToTotalOnFixedBatch():
    images, targets = fixedBatch()
    model = createModel(ModelConfig(), seed=0)
    lossBreakdown = YoloLoss()(model(images), targets)
    assert set(lossBreakdown) == set(LOSS_COMPONENTS)
    assert all(torch.isfinite(value) for value in lossBreakdown.values())
    partsSum = sum(lossBreakdown[name] for name in ("coordinate", "objectConfidence", "noObjectConfidence", "classification"))
    torch.testing.assert_close(partsSum, lossBreakdown["total"])
    torch.testing.assert_close(lossBreakdown["coordinateXy"] + lossBreakdown["coordinateWh"], lossBreakdown["coordinate"])


def testFixedBatchLossIsReproducible():
    images, targets = fixedBatch()
    firstLoss = YoloLoss()(createModel(seed=0)(images), targets)["total"]
    secondLoss = YoloLoss()(createModel(seed=0)(images), targets)["total"]
    assert firstLoss.item() == secondLoss.item()


@pytest.mark.parametrize("outputActivation", ["sigmoid", "linear"])
@pytest.mark.parametrize("useSqrt", [True, False])
def testGradientsAreFinite(outputActivation, useSqrt):
    images, targets = fixedBatch()
    model = createModel(ModelConfig(outputActivation=outputActivation), seed=0)
    lossBreakdown = YoloLoss(YoloLossConfig(useSqrtWidthHeight=useSqrt))(model(images), targets)
    lossBreakdown["total"].backward()
    assert torch.isfinite(lossBreakdown["total"])
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def testSafeSqrtHandlesZeroAndNegativeWidths():
    widths = torch.tensor([-0.2, 0.0, 1e-12, 0.25], requires_grad=True)
    values = safeSqrt(widths)
    values.sum().backward()
    assert torch.isfinite(values).all() and torch.isfinite(widths.grad).all()
    assert values[3].item() == pytest.approx(0.5, rel=1e-5)
    assert values[0].item() < 0


def testNoObjectLossCoversNonResponsiblePredictorInOccupiedCell():
    confidence = torch.zeros(1, 7, 7, 2)
    confidence[0, 3, 3] = torch.tensor([0.5, 0.9])
    responsibleMask = torch.zeros(1, 7, 7, 2)
    responsibleMask[0, 3, 3, 1] = 1.0
    assert noObjectConfidenceLoss(confidence, responsibleMask).item() == pytest.approx(0.25)


def testLossWeightsAreConfigurable():
    images, targets = fixedBatch()
    predictions = createModel(seed=0)(images)
    paperLoss = YoloLoss(YoloLossConfig())(predictions, targets)
    unitLoss = YoloLoss(YoloLossConfig(lambdaCoord=1.0, lambdaNoobj=1.0))(predictions, targets)
    torch.testing.assert_close(unitLoss["coordinate"] * 5, paperLoss["coordinate"])
    torch.testing.assert_close(unitLoss["noObjectConfidence"] * 0.5, paperLoss["noObjectConfidence"])
    directLoss = YoloLoss(YoloLossConfig(useSqrtWidthHeight=False))(predictions, targets)
    assert not math.isclose(directLoss["coordinateWh"].item(), paperLoss["coordinateWh"].item())


def testConfidenceTargetOne():
    target = singleObjectTarget()
    predictions = predictionWithBoxes((0.5, 0.5, 0.05, 0.05), (0.5, 0.5, 0.2, 0.1), confidences=(0.0, 0.7))
    lossBreakdown = YoloLoss(YoloLossConfig(confidenceTarget="one"))(predictions, target)
    assert lossBreakdown["objectConfidence"].item() == pytest.approx(0.3**2, rel=1e-5)


def testOneOptimizerStepReducesLossOnFixedBatch():
    images, targets = fixedBatch(batchSize=8)
    model = createModel(ModelConfig(dropout=0.0), seed=0)
    lossFn = YoloLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    parametersBefore = [parameter.detach().clone() for parameter in model.parameters()]

    lossBefore = lossFn(model(images), targets)["total"]
    optimizer.zero_grad()
    lossBefore.backward()
    optimizer.step()
    with torch.no_grad():
        lossAfter = lossFn(model(images), targets)["total"]

    assert any(not torch.equal(before, after) for before, after in zip(parametersBefore, model.parameters()))
    assert lossAfter.item() < lossBefore.item()
