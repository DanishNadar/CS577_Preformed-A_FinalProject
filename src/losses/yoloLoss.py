# Reference YOLOv1 loss (Redmon et al., 2016, eq. 3) with each term as a separate function

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
    # sign(v) * sqrt(|v| + eps) keeps values and gradients finite for zero or negative widths
    return torch.sign(values) * torch.sqrt(values.abs() + eps)


def assignResponsibleBoxes(
    predictedBoxes: Tensor, targetBoxes: Tensor, objectMask: Tensor
) -> tuple[Tensor, Tensor]:
    # Highest-IoU predictor per occupied cell; no_grad because argmax and the IoU confidence target are constants
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
    # Returns weighted, batch-averaged loss components that sum to 'total'

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
