"""Reference YOLOv1 loss (Redmon et al., 2016, eq. 3).

    L = lambdaCoord * sum_ij 1obj_ij [(x - x^)^2 + (y - y^)^2]
      + lambdaCoord * sum_ij 1obj_ij [(sqrt w - sqrt w^)^2 + (sqrt h - sqrt h^)^2]
      +               sum_ij 1obj_ij (C - C^)^2
      + lambdaNoobj * sum_ij 1noobj_ij (C - C^)^2
      +               sum_i  1obj_i  sum_c (p(c) - p^(c))^2

1obj_ij marks predictor j in cell i as responsible: the cell contains an object center and
predictor j has the highest IoU with that object. 1noobj_ij is its complement, so the
non-responsible predictor in an occupied cell is pushed toward zero confidence too, as in the paper.
The confidence target for a responsible predictor is IoU(prediction, truth) by default.

Each term is a standalone function so it can be inspected, differentiated or replaced on its own.
The total is summed over cells and predictors and divided by the batch size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.boxes import boxIou, cellToImageBoxes, cxcywhToXyxy
from src.utils.decoding import splitPredictions

CONFIDENCE_TARGETS = ("iou", "one")
LOSS_COMPONENTS = ("coordinate", "coordinateXy", "coordinateWh", "objectConfidence", "noObjectConfidence",
                   "classification", "total")


@dataclass
class YoloLossConfig:
    lambdaCoord: float = 5.0
    lambdaNoobj: float = 0.5
    useSqrtWidthHeight: bool = True
    confidenceTarget: str = "iou"  # "iou" follows the paper; "one" is the common darknet-style shortcut
    sqrtEpsilon: float = 1e-6


def safeSqrt(values: Tensor, eps: float = 1e-6) -> Tensor:
    """sign(v) * sqrt(|v| + eps): finite value and gradient even for zero or negative widths."""
    return torch.sign(values) * torch.sqrt(values.abs() + eps)


def assignResponsibleBoxes(
    predictedBoxes: Tensor, targetBoxes: Tensor, objectMask: Tensor
) -> tuple[Tensor, Tensor]:
    """Pick the responsible predictor in every occupied cell.

    predictedBoxes (N, S, S, B, 4) and targetBoxes (N, S, S, 4) are cell-relative.
    Returns responsibleMask (N, S, S, B) in {0, 1} and boxIous (N, S, S, B).

    Runs under no_grad: argmax is not differentiable, and the paper treats the IoU confidence
    target as a constant, so no gradient should flow through either.
    Ties (e.g. both IoUs are 0 early in training) go to the lower predictor index.
    """
    gridSize = predictedBoxes.shape[1]
    numBoxes = predictedBoxes.shape[3]
    with torch.no_grad():
        predictedCorners = cxcywhToXyxy(cellToImageBoxes(predictedBoxes, gridSize))
        targetCorners = cxcywhToXyxy(cellToImageBoxes(targetBoxes, gridSize)).unsqueeze(-2)
        boxIous = boxIou(predictedCorners, targetCorners)
        bestPredictor = boxIous.argmax(dim=-1)
        responsibleMask = F.one_hot(bestPredictor, numBoxes).to(predictedBoxes.dtype) * objectMask.unsqueeze(-1)
    return responsibleMask, boxIous


def centerLoss(predictedBoxes: Tensor, targetBoxes: Tensor, responsibleMask: Tensor) -> Tensor:
    squaredError = (predictedBoxes[..., :2] - targetBoxes.unsqueeze(-2)[..., :2]).pow(2).sum(dim=-1)
    return (responsibleMask * squaredError).sum()


def sizeLoss(
    predictedBoxes: Tensor, targetBoxes: Tensor, responsibleMask: Tensor, useSqrt: bool = True, eps: float = 1e-6
) -> Tensor:
    predictedSize = predictedBoxes[..., 2:4]
    targetSize = targetBoxes.unsqueeze(-2)[..., 2:4]
    if useSqrt:
        predictedSize, targetSize = safeSqrt(predictedSize, eps), safeSqrt(targetSize, eps)
    squaredError = (predictedSize - targetSize).pow(2).sum(dim=-1)
    return (responsibleMask * squaredError).sum()


def objectConfidenceLoss(predictedConfidence: Tensor, targetConfidence: Tensor, responsibleMask: Tensor) -> Tensor:
    return (responsibleMask * (predictedConfidence - targetConfidence).pow(2)).sum()


def noObjectConfidenceLoss(predictedConfidence: Tensor, responsibleMask: Tensor) -> Tensor:
    return ((1.0 - responsibleMask) * predictedConfidence.pow(2)).sum()


def classificationLoss(predictedClasses: Tensor, targetClasses: Tensor, objectMask: Tensor) -> Tensor:
    squaredError = (predictedClasses - targetClasses).pow(2).sum(dim=-1)
    return (objectMask * squaredError).sum()


class YoloLoss(nn.Module):
    """predictions (N, S, S, B*5 + C), targets (N, S, S, 5 + C) -> dict of scalar loss tensors.

    Returned keys: coordinate (= coordinateXy + coordinateWh), coordinateXy, coordinateWh,
    objectConfidence, noObjectConfidence, classification, total. Every value already includes
    its lambda weight and the 1/N batch normalization, so the weighted parts add up to total.
    """

    def __init__(self, config: YoloLossConfig | None = None, numBoxes: int = 2):
        super().__init__()
        self.config = config or YoloLossConfig()
        self.numBoxes = numBoxes
        if self.config.confidenceTarget not in CONFIDENCE_TARGETS:
            raise ValueError(f"confidenceTarget must be one of {CONFIDENCE_TARGETS}")

    def forward(self, predictions: Tensor, targets: Tensor) -> dict[str, Tensor]:
        cfg = self.config
        batchSize = predictions.shape[0]
        predictedBoxes, predictedConfidence, predictedClasses = splitPredictions(predictions, self.numBoxes)
        targetBoxes, objectMask, targetClasses = targets[..., :4], targets[..., 4], targets[..., 5:]

        responsibleMask, boxIous = assignResponsibleBoxes(predictedBoxes, targetBoxes, objectMask)
        targetConfidence = boxIous if cfg.confidenceTarget == "iou" else torch.ones_like(boxIous)

        coordinateXy = cfg.lambdaCoord * centerLoss(predictedBoxes, targetBoxes, responsibleMask)
        coordinateWh = cfg.lambdaCoord * sizeLoss(
            predictedBoxes, targetBoxes, responsibleMask, cfg.useSqrtWidthHeight, cfg.sqrtEpsilon
        )
        breakdown = {
            "coordinateXy": coordinateXy,
            "coordinateWh": coordinateWh,
            "coordinate": coordinateXy + coordinateWh,
            "objectConfidence": objectConfidenceLoss(predictedConfidence, targetConfidence, responsibleMask),
            "noObjectConfidence": cfg.lambdaNoobj * noObjectConfidenceLoss(predictedConfidence, responsibleMask),
            "classification": classificationLoss(predictedClasses, targetClasses, objectMask),
        }
        breakdown = {name: value / batchSize for name, value in breakdown.items()}
        breakdown["total"] = (
            breakdown["coordinate"] + breakdown["objectConfidence"]
            + breakdown["noObjectConfidence"] + breakdown["classification"]
        )
        return breakdown
