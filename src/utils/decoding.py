# Turn raw model output into boxes, confidences and class scores

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.utils.boxes import cellToImageBoxes, cxcywhToXyxy, nonMaxSuppression


@dataclass
class DecodedPredictions:
    cellBoxes: Tensor  # (N, S, S, B, 4) cell-relative x, y and image-relative w, h
    imageBoxes: Tensor  # (N, S, S, B, 4) image-relative cxcywh
    confidence: Tensor  # (N, S, S, B)
    classProbabilities: Tensor  # (N, S, S, C)
    classScores: Tensor  # (N, S, S, B, C) = confidence * class probability (paper eq. at test time)


def splitPredictions(predictions: Tensor, numBoxes: int) -> tuple[Tensor, Tensor, Tensor]:
    # Split (N, S, S, B*5 + C) into boxes (N, S, S, B, 4), confidence (N, S, S, B), classes (N, S, S, C)
    batchSize, gridSize = predictions.shape[0], predictions.shape[1]
    boxValues = predictions[..., : numBoxes * 5].reshape(batchSize, gridSize, gridSize, numBoxes, 5)
    return boxValues[..., :4], boxValues[..., 4], predictions[..., numBoxes * 5 :]


def decodePredictions(predictions: Tensor, numBoxes: int) -> DecodedPredictions:
    cellBoxes, confidence, classProbabilities = splitPredictions(predictions, numBoxes)
    gridSize = predictions.shape[1]
    return DecodedPredictions(
        cellBoxes=cellBoxes,
        imageBoxes=cellToImageBoxes(cellBoxes, gridSize),
        confidence=confidence,
        classProbabilities=classProbabilities,
        classScores=confidence.unsqueeze(-1) * classProbabilities.unsqueeze(-2),
    )


def predictionsToDetections(
    predictions: Tensor,
    numBoxes: int,
    scoreThreshold: float = 0.2,
    nmsIouThreshold: float = 0.5,
) -> list[Tensor]:
    # Per-image (K, 6) [left, top, right, bottom, score, classId] detections after best-class scoring and class-wise NMS
    decoded = decodePredictions(predictions.detach(), numBoxes)
    batchSize = predictions.shape[0]
    allBoxes = cxcywhToXyxy(decoded.imageBoxes).reshape(batchSize, -1, 4).clamp(0.0, 1.0)
    bestScores, bestClasses = decoded.classScores.reshape(batchSize, -1, decoded.classScores.shape[-1]).max(dim=-1)

    detections = []
    for imageIndex in range(batchSize):
        keepMask = bestScores[imageIndex] >= scoreThreshold
        boxes = allBoxes[imageIndex][keepMask]
        scores = bestScores[imageIndex][keepMask]
        classIds = bestClasses[imageIndex][keepMask]

        keptRows = []
        for classId in classIds.unique():
            classIndices = torch.nonzero(classIds == classId, as_tuple=True)[0]
            keptIndices = classIndices[nonMaxSuppression(boxes[classIndices], scores[classIndices], nmsIouThreshold)]
            keptRows.append(
                torch.cat((boxes[keptIndices], scores[keptIndices, None], classIds[keptIndices, None].to(boxes.dtype)), 1)
            )
        imageDetections = torch.cat(keptRows) if keptRows else torch.zeros(0, 6, device=predictions.device)
        detections.append(imageDetections[imageDetections[:, 4].argsort(descending=True)])
    return detections
