# Train every (loss variant, seed) pair with identical data, model and optimizer settings.
# python -m src.ablations.runAblations [--variants paper unitWeights directWh] [--seeds 577 1 2] [--jobs 3]

from __future__ import annotations

import argparse
import contextlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch

from src.ablations.metrics import TrainingGradientTracker, makeAblationMetricsFn
from src.ablations.variants import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    VARIANTS,
    assertFairComparison,
    buildVariantConfig,
)
from src.datasets.syntheticShapes import ensureSyntheticDataset
from src.training.train import train
from src.utils.config import ExperimentConfig, parseOverrideValue


def isRunComplete(config: ExperimentConfig) -> bool:
    historyPath = Path(config.outputDir) / "history.json"
    if not historyPath.exists():
        return False
    history = json.loads(historyPath.read_text())
    return bool(history) and history[-1]["epoch"] >= config.training.epochs and "valMeanIou" in history[-1]


def trainVariant(config: ExperimentConfig, numThreads: int | None = None, logToFile: bool = True) -> dict:
    # Train one variant with the ablation metrics hooked in; stdout goes to <outputDir>/train.log when logToFile
    if numThreads:
        torch.set_num_threads(numThreads)
    outputDir = Path(config.outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    tracker = TrainingGradientTracker()
    metricsFn = makeAblationMetricsFn(config.loss, config.model.numBoxes, config.data.imageSize, tracker)
    logContext = open(outputDir / "train.log", "w") if logToFile else contextlib.nullcontext()
    with logContext as logFile, tracker:
        redirect = contextlib.redirect_stdout(logFile) if logToFile else contextlib.nullcontext()
        with redirect:
            result = train(config, extraMetricsFn=metricsFn, verbose=True)
    return {"experimentName": config.experimentName, "seconds": result.totalSeconds,
            "finalValMeanIou": result.trainingHistory[-1].get("valMeanIou")}


def prepareDatasets(configs: list[ExperimentConfig]) -> None:
    # Generate each seed's dataset once, before parallel jobs could race to write it
    for config in {config.data.synthetic.root: config for config in configs}.values():
        if config.data.dataset == "synthetic":
            ensureSyntheticDataset(config.data.synthetic, config.seed, config.data.imageSize, config.data.gridSize)


def runAblations(
    variantNames: list[str],
    seeds: list[int],
    jobs: int = 1,
    threadsPerJob: int | None = None,
    baseConfigPath: str = "configs/baseline.yaml",
    outputRoot: str = DEFAULT_OUTPUT_ROOT,
    checkpointRoot: str = DEFAULT_CHECKPOINT_ROOT,
    sharedOverrides: dict | None = None,
    force: bool = False,
) -> list[dict]:
    configs = [
        buildVariantConfig(name, seed, baseConfigPath, outputRoot, checkpointRoot, sharedOverrides)
        for seed in seeds for name in variantNames
    ]
    assertFairComparison(configs)
    pending = [config for config in configs if force or not isRunComplete(config)]
    for config in configs:
        if config not in pending:
            print(f"skip (complete): {config.experimentName}")
    if not pending:
        return []
    prepareDatasets(pending)

    if threadsPerJob is None:
        threadsPerJob = max(1, torch.get_num_threads() // max(jobs, 1))
    print(f"running {len(pending)} runs, {jobs} at a time, {threadsPerJob} threads each")
    summaries, started = [], time.perf_counter()
    if jobs <= 1:
        for config in pending:
            print(f"start {config.experimentName}", flush=True)
            summaries.append(trainVariant(config, threadsPerJob))
            print(f"done  {summaries[-1]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(trainVariant, config, threadsPerJob): config for config in pending}
            for future in as_completed(futures):
                summaries.append(future.result())
                print(f"done  {summaries[-1]} [{time.perf_counter() - started:.0f}s elapsed]", flush=True)
    return summaries


def parseArguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the YOLOv1 loss ablations")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[DEFAULT_SEED])
    parser.add_argument("--jobs", type=int, default=1, help="runs trained in parallel processes")
    parser.add_argument("--threads", type=int, default=None, help="torch threads per run (default: cores / jobs)")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--outputRoot", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpointRoot", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override applied to every run, e.g. --set training.epochs=5 (loss.* is rejected)")
    parser.add_argument("--force", action="store_true", help="retrain runs that already finished")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[dict]:
    arguments = parseArguments(argv)
    overrides = {}
    for override in arguments.overrides:
        key, _, value = override.partition("=")
        overrides[key] = parseOverrideValue(value)
    return runAblations(arguments.variants, arguments.seeds, arguments.jobs, arguments.threads, arguments.config,
                        arguments.outputRoot, arguments.checkpointRoot, overrides, arguments.force)


if __name__ == "__main__":
    main()
