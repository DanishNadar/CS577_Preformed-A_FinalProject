# Checkpoint save/load. Checkpoints are trusted local files written by train.py

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from src.models.yoloModel import ModelConfig, YoloModel


def saveCheckpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    bestValLoss: float,
    trainingHistory: list[dict],
    extraState: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
        "epoch": epoch,
        "seed": config.get("seed"),
        "config": config,
        "bestValLoss": bestValLoss,
        "trainingHistory": trainingHistory,
        **(extraState or {}),
    }
    # Write then rename so an interrupted save never leaves a truncated checkpoint behind
    temporaryPath = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporaryPath)
    os.replace(temporaryPath, path)
    return path


def loadCheckpoint(path: str | Path, device: str | torch.device = "cpu") -> dict:
    # weights_only=False: checkpoints also hold RNG states and config dicts, not just tensors
    return torch.load(Path(path), map_location=device, weights_only=False)


def loadModelFromCheckpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[YoloModel, dict]:
    # Rebuild the model from the checkpoint's stored config; returns (model in eval mode, checkpoint)
    checkpoint = loadCheckpoint(path, device)
    model = YoloModel(ModelConfig(**checkpoint["config"]["model"]))
    model.load_state_dict(checkpoint["modelState"])
    model.to(device).eval()
    return model, checkpoint
