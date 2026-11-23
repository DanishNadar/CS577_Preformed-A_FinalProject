"""Batch inference helpers for evaluation code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class InferenceOutputs:
    predictions: Tensor  # (N, S, S, B*5 + C), activated model output on CPU
    targets: Tensor  # (N, S, S, 5 + C) encoded ground truth
    annotations: list[Tensor]  # per image (K, 5): [classId, centerX, centerY, width, height]


@torch.no_grad()
def predict(model: nn.Module, images: Tensor, device: torch.device | str = "cpu") -> Tensor:
    model.eval()
    return model(images.to(device)).cpu()


@torch.no_grad()
def collectPredictions(model: nn.Module, dataLoader: DataLoader, device: torch.device | str = "cpu") -> InferenceOutputs:
    model.eval()
    predictionBatches, targetBatches, allAnnotations = [], [], []
    for images, targets, annotations in dataLoader:
        predictionBatches.append(model(images.to(device)).cpu())
        targetBatches.append(targets)
        allAnnotations.extend(annotations)
    return InferenceOutputs(torch.cat(predictionBatches), torch.cat(targetBatches), allAnnotations)
