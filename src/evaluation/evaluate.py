# Evaluate trained checkpoints on their validation split: mAP (VOC AP@0.5, AP@0.75, COCO-style AP@[.5:.95]),
# per-class AP, responsible-box IoU, recall/precision at the operating point and an error breakdown by object size.
# python -m src.evaluation.evaluate --checkpoint checkpoints/baseline/last.pt
# python -m src.evaluation.evaluate --ablations [--checkpointRoot checkpoints/ablations] [--figureDir docs/figures/evaluation]

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from src.ablations.metrics import detectionMetrics, responsibleBoxMetrics
from src.ablations.variants import DEFAULT_CHECKPOINT_ROOT, REFERENCE_VARIANT, VARIANTS
from src.datasets.loaders import DataConfig, buildDatasets, collateDetections
from src.evaluation.detectionMetrics import (
    ERROR_TYPES,
    SIZE_BUCKETS,
    cocoStyleMap,
    errorBreakdown,
    meanAveragePrecision,
    precisionRecallCurve,
)
from src.training.checkpoint import loadModelFromCheckpoint
from src.training.inference import collectPredictions
from src.utils.config import dataclassFromDict
from src.utils.decoding import predictionsToDetections

DEFAULT_EVALUATION_ROOT = "results/evaluation"
# AP needs the whole ranking, so keep nearly every scored box; the operating point matches the ablation metrics
AP_SCORE_THRESHOLD = 0.01
OPERATING_SCORE_THRESHOLD = 0.2
NMS_IOU_THRESHOLD = 0.5


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    errors: dict[str, dict[str, int]]
    classNames: list[str]
    curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # className -> precision/recall at 0.5
    predictions: Tensor | None = None
    annotations: list[Tensor] | None = None
    dataset: object | None = None  # validation dataset, for drawing example images


_datasetCache: dict[str, object] = {}


def validationDataset(dataValues: dict, seed: int):
    # Validation split described by a checkpoint's stored data config; cached because variants share a seed
    dataConfig = dataclassFromDict(DataConfig, dataValues)
    key = json.dumps(dataValues, sort_keys=True, default=str) + f"/{seed}"
    if key not in _datasetCache:
        _datasetCache[key] = buildDatasets(dataConfig, seed)[1]
    return _datasetCache[key]


def classNamesFor(dataValues: dict) -> list[str]:
    if dataValues.get("dataset", "synthetic") == "voc":
        return list(dataValues["voc"]["classes"])
    from src.datasets.syntheticShapes import CLASS_NAMES
    return list(CLASS_NAMES)


@torch.no_grad()
def evaluatePredictions(
    predictions: Tensor, targets: Tensor, annotations: list[Tensor], numBoxes: int, imageSize: int,
    classNames: list[str],
) -> EvaluationResult:
    numClasses = len(classNames)
    rankedDetections = predictionsToDetections(predictions, numBoxes, AP_SCORE_THRESHOLD, NMS_IOU_THRESHOLD)
    map50, perClass50 = meanAveragePrecision(rankedDetections, annotations, numClasses, 0.5)
    map75, _ = meanAveragePrecision(rankedDetections, annotations, numClasses, 0.75)
    metrics = {
        "mAP50": map50,
        "mAP75": map75,
        "mAP50to95": cocoStyleMap(rankedDetections, annotations, numClasses),
        **{f"AP50_{name}": value for name, value in zip(classNames, perClass50)},
    }
    responsible = responsibleBoxMetrics(predictions, targets, numBoxes, imageSize)
    metrics.update({key.removeprefix("val"): value for key, value in responsible.items() if "Count" not in key})
    operating = detectionMetrics(predictions, annotations, numBoxes, OPERATING_SCORE_THRESHOLD)
    metrics.update({key.removeprefix("val"): value for key, value in operating.items()})

    operatingDetections = predictionsToDetections(predictions, numBoxes, OPERATING_SCORE_THRESHOLD, NMS_IOU_THRESHOLD)
    errors = errorBreakdown(operatingDetections, annotations, imageSize)
    curves = {}
    for classId, name in enumerate(classNames):
        curve = precisionRecallCurve(rankedDetections, annotations, classId, 0.5)
        curves[name] = {"precision": curve.precision.tolist(), "recall": curve.recall.tolist()}
    return EvaluationResult(metrics, errors, classNames, curves, predictions, annotations)


def evaluateCheckpoint(checkpointPath: str | Path, device: str = "cpu", batchSize: int = 64) -> EvaluationResult:
    model, checkpoint = loadModelFromCheckpoint(checkpointPath, device)
    config = checkpoint["config"]
    dataset = validationDataset(config["data"], config["seed"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False, collate_fn=collateDetections)
    outputs = collectPredictions(model, loader, device)
    result = evaluatePredictions(outputs.predictions, outputs.targets, outputs.annotations, config["model"]["numBoxes"],
                                 config["data"]["imageSize"], classNamesFor(config["data"]))
    result.metrics["epoch"] = checkpoint["epoch"]
    result.dataset = dataset
    return result


def findAblationCheckpoints(checkpointRoot: str | Path, checkpointName: str) -> list[tuple[str, int, Path]]:
    found = []
    for path in sorted(Path(checkpointRoot).glob(f"seed*/*/{checkpointName}")):
        variant, seed = path.parent.name, int(path.parent.parent.name.removeprefix("seed"))
        if variant in VARIANTS:
            found.append((variant, seed, path))
    return found


def meanStd(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


SUMMARY_METRICS = (
    ("mAP50", "mAP@0.5"),
    ("mAP75", "mAP@0.75"),
    ("mAP50to95", "mAP@[.5:.95]"),
    ("MeanIou", "Mean IoU (responsible box)"),
    ("RecallAt50", "Recall@0.5 (score ≥ 0.2)"),
    ("PrecisionAt50", "Precision@0.5 (score ≥ 0.2)"),
)


def summaryMarkdown(runRows: list[dict], errorRows: list[dict], classNames: list[str], checkpointName: str) -> str:
    variants = [name for name in VARIANTS if any(row["variant"] == name for row in runRows)]
    seeds = sorted({row["seed"] for row in runRows})
    metrics = list(SUMMARY_METRICS) + [(f"AP50_{name}", f"AP@0.5, {name}") for name in classNames]

    def cell(variant: str, key: str) -> str:
        mean, std = meanStd([row[key] for row in runRows if row["variant"] == variant])
        return "—" if math.isnan(mean) else f"{mean:.3f} ± {std:.3f}"

    def pairedDiff(variant: str, key: str) -> str:
        reference = {row["seed"]: row[key] for row in runRows if row["variant"] == REFERENCE_VARIANT}
        diffs = [row[key] - reference[row["seed"]] for row in runRows
                 if row["variant"] == variant and row["seed"] in reference]
        if not diffs:
            return "—"
        better = sum(diff >= 0 for diff in diffs)
        return f"{statistics.fmean(diffs):+.3f} ({better}/{len(diffs)} ≥ paper)"

    lines = [
        "# Evaluation summary",
        "",
        f"Checkpoint: `{checkpointName}` of every run. Seeds: {', '.join(map(str, seeds))}. "
        "Validation split of each seed. Values are mean ± std over seeds.",
        "",
        f"AP uses VOC all-point interpolation over detections with score ≥ {AP_SCORE_THRESHOLD} after class-wise NMS "
        f"(IoU {NMS_IOU_THRESHOLD}). Recall, precision and the error breakdown use the operating point "
        f"score ≥ {OPERATING_SCORE_THRESHOLD}.",
        "",
        "## Detection metrics",
        "",
        "| Metric | " + " | ".join(f"`{variant}`" for variant in variants) + " |",
        "|---|" + "---|" * len(variants),
    ]
    for key, label in metrics:
        lines.append(f"| {label} | " + " | ".join(cell(variant, key) for variant in variants) + " |")

    lines += ["", "## Paired difference vs `paper` (mean over seeds)", "",
              "| Metric | " + " | ".join(f"`{variant}`" for variant in variants if variant != REFERENCE_VARIANT) + " |",
              "|---|" + "---|" * (len(variants) - 1)]
    for key, label in SUMMARY_METRICS[:3]:
        lines.append(f"| {label} | " + " | ".join(pairedDiff(variant, key) for variant in variants
                                                   if variant != REFERENCE_VARIANT) + " |")

    lines += ["", "## Error breakdown (share of ground-truth objects, all seeds pooled)", "",
              "Each object gets its best outcome among detections with score ≥ "
              f"{OPERATING_SCORE_THRESHOLD}: **correct** (right class, IoU ≥ 0.5), **localization** (right class, "
              "0.1 ≤ IoU < 0.5), **classification** (wrong class, IoU ≥ 0.5), **missed** (none of these).", ""]
    buckets = ["all"] + [name for name, _, _ in SIZE_BUCKETS]
    lines += ["| Variant | Objects | " + " | ".join(f"{bucket}: {kind}" for bucket in ("all",) for kind in ERROR_TYPES)
              + " | small: correct | medium: correct | large: correct |",
              "|---|---|" + "---|" * (len(ERROR_TYPES) + 3)]
    for variant in variants:
        totals = {bucket: dict.fromkeys(ERROR_TYPES, 0) for bucket in buckets}
        for row in errorRows:
            if row["variant"] == variant:
                totals[row["bucket"]] = {kind: totals[row["bucket"]][kind] + row[kind] for kind in ERROR_TYPES}
        allCount = sum(totals["all"].values())

        def share(bucket: str, kind: str) -> str:
            count = sum(totals[bucket].values())
            return f"{totals[bucket][kind] / count:.1%}" if count else "—"

        lines.append(f"| `{variant}` | {allCount} | " + " | ".join(share("all", kind) for kind in ERROR_TYPES) + " | "
                     + " | ".join(share(bucket, "correct") for bucket in buckets[1:]) + " |")
    lines.append("")
    return "\n".join(lines)


def writeCsv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluateAblations(
    checkpointRoot: str | Path = DEFAULT_CHECKPOINT_ROOT,
    outputRoot: str | Path = DEFAULT_EVALUATION_ROOT,
    checkpointName: str = "last.pt",
    figureDir: str | Path | None = None,
    device: str = "cpu",
) -> Path:
    checkpoints = findAblationCheckpoints(checkpointRoot, checkpointName)
    if not checkpoints:
        raise FileNotFoundError(f"No {checkpointName} under {checkpointRoot}/seed*/<variant>/; run the ablations first")
    outputRoot = Path(outputRoot)
    outputRoot.mkdir(parents=True, exist_ok=True)

    runRows, errorRows, results = [], [], {}
    for variant, seed, path in checkpoints:
        result = evaluateCheckpoint(path, device)
        results[(variant, seed)] = result
        runRows.append({"variant": variant, "seed": seed, **result.metrics})
        for bucket, counts in result.errors.items():
            errorRows.append({"variant": variant, "seed": seed, "bucket": bucket, **counts})
        print(f"{variant:>20} seed {seed:>3}: mAP@0.5 {result.metrics['mAP50']:.3f}  "
              f"mAP@[.5:.95] {result.metrics['mAP50to95']:.3f}  mean IoU {result.metrics['MeanIou']:.3f}", flush=True)

    classNames = next(iter(results.values())).classNames
    writeCsv(outputRoot / "runMetrics.csv", runRows)
    writeCsv(outputRoot / "errorBreakdown.csv", errorRows)
    curves = {f"{variant}/seed{seed}": result.curves for (variant, seed), result in results.items()}
    (outputRoot / "precisionRecall.json").write_text(json.dumps(curves))
    summaryPath = outputRoot / "summary.md"
    summaryPath.write_text(summaryMarkdown(runRows, errorRows, classNames, checkpointName))

    if figureDir is not None:
        from src.evaluation.figures import plotEvaluationFigures
        written = plotEvaluationFigures(runRows, errorRows, results, Path(figureDir))
        for path in written:
            print(f"wrote {path}")
    print(f"wrote {summaryPath}")
    return summaryPath


def parseArguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv1 checkpoints (mAP, IoU, error breakdown)")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint", help="evaluate one checkpoint and print its metrics as JSON")
    target.add_argument("--ablations", action="store_true", help="evaluate every ablation run's checkpoint")
    parser.add_argument("--checkpointRoot", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpointName", default="last.pt",
                        help="last.pt (final epoch, default) or best.pt (lowest val loss under the run's own loss)")
    parser.add_argument("--outputRoot", default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--figureDir", default=None, help="also write figures here, e.g. docs/figures/evaluation")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    arguments = parseArguments(argv)
    if arguments.checkpoint:
        result = evaluateCheckpoint(arguments.checkpoint, arguments.device)
        print(json.dumps({"metrics": result.metrics, "errors": result.errors}, indent=2))
    else:
        evaluateAblations(arguments.checkpointRoot, arguments.outputRoot, arguments.checkpointName,
                          arguments.figureDir, arguments.device)


if __name__ == "__main__":
    main()
