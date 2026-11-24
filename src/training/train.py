# Baseline training: python -m src.training.train --config configs/baseline.yaml [--set key=value] [--resume path]

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.loaders import buildDataLoaders
from src.losses.yoloLoss import YoloLoss
from src.models.yoloModel import countParameters, createModel
from src.training.checkpoint import loadCheckpoint, saveCheckpoint
from src.training.history import prefixedMetrics, writeHistory
from src.utils.config import ExperimentConfig, configToDict, loadExperimentConfig, parseOverrideValue
from src.utils.reproducibility import captureRngState, resolveDevice, restoreRngState, seedEverything

# Called after validation each epoch; returned values are added to that epoch's history row
ExtraMetricsFn = Callable[[nn.Module, DataLoader, torch.device], dict]


@dataclass
class TrainingResult:
    trainingHistory: list[dict]
    bestValLoss: float
    lastCheckpointPath: Path
    bestCheckpointPath: Path
    totalSeconds: float


def runEpoch(
    model: nn.Module,
    dataLoader: DataLoader,
    lossFn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradientClipNorm: float | None = None,
) -> dict[str, float]:
    # Trains when an optimizer is given, otherwise evaluates without gradients
    isTraining = optimizer is not None
    model.train(isTraining)
    componentSums: dict[str, float] = {}
    sampleCount = 0

    with torch.set_grad_enabled(isTraining):
        for images, targets, _ in dataLoader:
            images, targets = images.to(device), targets.to(device)
            lossBreakdown = lossFn(model(images), targets)
            if isTraining:
                optimizer.zero_grad(set_to_none=True)
                lossBreakdown["total"].backward()
                if gradientClipNorm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradientClipNorm)
                optimizer.step()

            batchSize = images.shape[0]
            sampleCount += batchSize
            for name, value in lossBreakdown.items():
                componentSums[name] = componentSums.get(name, 0.0) + value.item() * batchSize

    return {name: total / sampleCount for name, total in componentSums.items()}


def train(
    config: ExperimentConfig,
    resumePath: str | Path | None = None,
    lossFn: nn.Module | None = None,
    extraMetricsFn: ExtraMetricsFn | None = None,
    verbose: bool = True,
) -> TrainingResult:
    # Train from scratch or resume; lossFn overrides the default YoloLoss(config.loss)
    seedEverything(config.seed)
    device = resolveDevice(config.device)
    trainingConfig = config.training
    outputDir, checkpointDir = Path(config.outputDir), Path(config.checkpointDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    configDict = configToDict(config)
    (outputDir / "config.json").write_text(json.dumps(configDict, indent=2))

    trainLoader, valLoader = buildDataLoaders(config.data, trainingConfig.batchSize, config.seed, device.type == "cuda")
    model = createModel(config.model, seed=config.seed).to(device)
    lossFn = lossFn or YoloLoss(config.loss, numBoxes=config.model.numBoxes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=trainingConfig.learningRate, weight_decay=trainingConfig.weightDecay
    )

    startEpoch, bestValLoss, trainingHistory = 1, math.inf, []
    if resumePath is not None:
        checkpoint = loadCheckpoint(resumePath, device)
        model.load_state_dict(checkpoint["modelState"])
        optimizer.load_state_dict(checkpoint["optimizerState"])
        startEpoch = checkpoint["epoch"] + 1
        bestValLoss = checkpoint["bestValLoss"]
        trainingHistory = checkpoint["trainingHistory"]
        restoreRngState(checkpoint["rngState"])
        trainLoader.generator.set_state(checkpoint["shuffleGeneratorState"])

    if verbose:
        print(f"device={device} params={countParameters(model):,} train={len(trainLoader.dataset)} "
              f"val={len(valLoader.dataset)} epochs={startEpoch}..{trainingConfig.epochs}")
        for name, dataset in (("train", trainLoader.dataset), ("val", valLoader.dataset)):
            if hasattr(dataset, "collisionSummary"):
                print(f"  {name} collisions: {dataset.collisionSummary()}")

    lastCheckpointPath, bestCheckpointPath = checkpointDir / "last.pt", checkpointDir / "best.pt"
    runStart = time.perf_counter()
    for epoch in range(startEpoch, trainingConfig.epochs + 1):
        epochStart = time.perf_counter()
        trainMetrics = runEpoch(model, trainLoader, lossFn, device, optimizer, trainingConfig.gradientClipNorm)
        valMetrics = runEpoch(model, valLoader, lossFn, device)
        historyRow = {
            "epoch": epoch,
            "learningRate": optimizer.param_groups[0]["lr"],
            **prefixedMetrics("train", trainMetrics),
            **prefixedMetrics("val", valMetrics),
        }
        if extraMetricsFn is not None:
            historyRow.update(extraMetricsFn(model, valLoader, device))
        historyRow["epochSeconds"] = time.perf_counter() - epochStart
        trainingHistory.append(historyRow)
        writeHistory(outputDir, trainingHistory)

        isBest = valMetrics["total"] < bestValLoss
        bestValLoss = min(bestValLoss, valMetrics["total"])
        extraState = {
            "rngState": captureRngState(),
            "shuffleGeneratorState": trainLoader.generator.get_state(),
        }
        saveCheckpoint(lastCheckpointPath, model, optimizer, epoch, configDict, bestValLoss, trainingHistory, extraState)
        if isBest:
            saveCheckpoint(bestCheckpointPath, model, optimizer, epoch, configDict, bestValLoss, trainingHistory, extraState)

        if verbose:
            print(f"epoch {epoch:3d} | train {trainMetrics['total']:.4f} | val {valMetrics['total']:.4f} "
                  f"(coord {valMetrics['coordinate']:.3f} obj {valMetrics['objectConfidence']:.3f} "
                  f"noobj {valMetrics['noObjectConfidence']:.3f} cls {valMetrics['classification']:.3f}) "
                  f"| {historyRow['epochSeconds']:.1f}s{' *' if isBest else ''}")

    return TrainingResult(trainingHistory, bestValLoss, lastCheckpointPath, bestCheckpointPath,
                          time.perf_counter() - runStart)


def parseArguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the baseline YOLOv1-style detector")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from, e.g. checkpoints/baseline/last.pt")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="dotted config override, e.g. --set training.epochs=5 (repeatable)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> TrainingResult:
    arguments = parseArguments(argv)
    overrides = {}
    for override in arguments.overrides:
        key, _, value = override.partition("=")
        overrides[key] = parseOverrideValue(value)
    config = loadExperimentConfig(arguments.config, overrides)
    result = train(config, resumePath=arguments.resume)
    print(f"done in {result.totalSeconds:.1f}s | best val loss {result.bestValLoss:.4f} | "
          f"checkpoints: {result.lastCheckpointPath}, {result.bestCheckpointPath}")
    return result


if __name__ == "__main__":
    main()
