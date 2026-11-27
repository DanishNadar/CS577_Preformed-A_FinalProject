# Summarize ablation runs: per-run and per-variant tables (mean +/- std over seeds, paired differences vs the
# paper loss) and figures. python -m src.ablations.analysis [--outputRoot results/ablations]

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from src.ablations.variants import DEFAULT_OUTPUT_ROOT, REFERENCE_VARIANT, VARIANTS

# Metrics reported at the end of training; "higher" says which direction is better
FINAL_METRICS = {
    "valMeanIou": ("Mean IoU (responsible box)", "higher"),
    "valIouAt50": ("Responsible IoU >= 0.5", "higher"),
    "valMeanIouSmall": ("Mean IoU, small objects (<24 px)", "higher"),
    "valMeanIouMedium": ("Mean IoU, medium objects (24-48 px)", "higher"),
    "valMeanIouLarge": ("Mean IoU, large objects (>=48 px)", "higher"),
    "valRelativeSizeErrorSmall": ("Relative w/h error, small", "lower"),
    "valRelativeSizeErrorLarge": ("Relative w/h error, large", "lower"),
    "valRecallAt50": ("Recall@0.5 after NMS", "higher"),
    "valPrecisionAt50": ("Precision@0.5 after NMS", "higher"),
    "valClassAccuracy": ("Class accuracy (occupied cells)", "higher"),
    "valPaperTotal": ("Val loss, paper weights", "lower"),
    "valPaperNoObjectConfidence": ("Val no-object term, paper weights", "lower"),
}
# Last epochs averaged for the headline numbers, which is less noisy than one epoch
TAIL_EPOCHS = 3
SPIKE_FRACTION = 0.10
IOU_MILESTONE = 0.5

# Reference palette (light mode), fixed order by variant so a color always means the same loss
VARIANT_STYLES = {
    "paper": ("#257de0", "-"),
    "unitWeights": ("#f47230", "--"),
    "directWh": ("#00b894", "-."),
    "unitCoord": ("#f2a600", ":"),
    "unitNoobj": ("#ee7fac", (0, (5, 2, 1, 2))),
    "unitWeightsDirectWh": ("#6a4ce2", (0, (1, 1))),
}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#191919", "#666666", "#e6e6e6"


def loadRuns(outputRoot: str | Path) -> list[dict]:
    # One record per finished run: variant, seed, history rows
    runs = []
    for historyPath in sorted(Path(outputRoot).glob("seed*/*/history.json")):
        variant, seedFolder = historyPath.parent.name, historyPath.parent.parent.name
        history = json.loads(historyPath.read_text())
        if variant in VARIANTS and history and "valMeanIou" in history[-1]:
            runs.append({"variant": variant, "seed": int(seedFolder.removeprefix("seed")), "history": history})
    return runs


def tailMean(history: list[dict], key: str, tail: int = TAIL_EPOCHS) -> float:
    values = [row[key] for row in history[-tail:] if key in row and _isNumber(row[key])]
    return sum(values) / len(values) if values else float("nan")


def _isNumber(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def stabilityMetrics(history: list[dict]) -> dict[str, float]:
    # Training-stability summary for one run
    paperLoss = [row["valPaperTotal"] for row in history]
    spikes = sum(1 for previous, current in zip(paperLoss, paperLoss[1:]) if current > previous * (1 + SPIKE_FRACTION))
    milestone = next((row["epoch"] for row in history if row["valMeanIou"] >= IOU_MILESTONE), float("nan"))
    secondHalf = history[len(history) // 2:]
    return {
        "valLossSpikes": spikes,
        "epochsToIou50": milestone,
        "trainGradNormMeanAll": statistics.fmean(row["trainGradNormMean"] for row in history),
        "trainGradNormPeak": max(row["trainGradNormMax"] for row in history),
        "trainGradNormStdSecondHalf": statistics.fmean(row["trainGradNormStd"] for row in secondHalf),
        "nonFiniteSteps": sum(row["trainNonFiniteSteps"] for row in history),
        "probeNoobjToObjRatioFirst": history[0]["probeNoobjToObjGradRatio"],
        "probeNoobjToObjRatioMean": statistics.fmean(row["probeNoobjToObjGradRatio"] for row in history),
        "bestValMeanIou": max(row["valMeanIou"] for row in history),
        "bestValMeanIouEpoch": max(history, key=lambda row: row["valMeanIou"])["epoch"],
        "meanEpochSeconds": statistics.fmean(row["epochSeconds"] for row in history),
    }


def runSummary(run: dict) -> dict:
    summary = {"variant": run["variant"], "seed": run["seed"], "epochs": run["history"][-1]["epoch"]}
    summary.update({key: tailMean(run["history"], key) for key in FINAL_METRICS})
    summary.update(stabilityMetrics(run["history"]))
    return summary


def meanStd(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if _isNumber(value)]
    if not values:
        return float("nan"), float("nan")
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(runSummaries: list[dict]) -> list[dict]:
    # Per variant: mean/std over seeds, and the seed-paired difference from the paper loss
    metricKeys = [key for key in runSummaries[0] if key not in ("variant", "seed", "epochs")]
    reference = {row["seed"]: row for row in runSummaries if row["variant"] == REFERENCE_VARIANT}
    rows = []
    for variant in [name for name in VARIANTS if any(row["variant"] == name for row in runSummaries)]:
        variantRows = [row for row in runSummaries if row["variant"] == variant]
        aggregateRow = {"variant": variant, "numSeeds": len(variantRows)}
        for key in metricKeys:
            aggregateRow[f"{key}Mean"], aggregateRow[f"{key}Std"] = meanStd([row[key] for row in variantRows])
            pairedDiffs = [row[key] - reference[row["seed"]][key] for row in variantRows if row["seed"] in reference]
            aggregateRow[f"{key}DiffVsPaper"], aggregateRow[f"{key}DiffStd"] = meanStd(pairedDiffs)
            aggregateRow[f"{key}SeedsBelowPaper"] = sum(diff < 0 for diff in pairedDiffs)
        rows.append(aggregateRow)
    return rows


def writeCsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formatValue(mean: float, std: float, digits: int = 3) -> str:
    if not _isNumber(mean):
        return "n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def formatDiff(row: dict, key: str, digits: int = 3) -> str:
    diff = row[f"{key}DiffVsPaper"]
    if row["variant"] == REFERENCE_VARIANT or not _isNumber(diff):
        return "—"
    return f"{diff:+.{digits}f} ({row['numSeeds'] - row[f'{key}SeedsBelowPaper']}/{row['numSeeds']} ≥ paper)"


def markdownReport(aggregateRows: list[dict], runSummaries: list[dict]) -> str:
    seeds = sorted({row["seed"] for row in runSummaries})
    epochs = sorted({row["epochs"] for row in runSummaries})
    lines = [
        "# Loss ablation summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Epochs: {', '.join(map(str, epochs))}. "
        f"Values are the mean of the last {TAIL_EPOCHS} epochs per run, then mean ± std over seeds. "
        "Differences are paired by seed against the paper loss.",
        "",
        "## Variants",
        "",
        "| Variant | λcoord | λnoobj | sqrt(w,h) | Description |",
        "|---|---|---|---|---|",
    ]
    for row in aggregateRows:
        variant = VARIANTS[row["variant"]]
        lines.append(f"| `{variant.name}` | {variant.lambdaCoord:g} | {variant.lambdaNoobj:g} | "
                     f"{'yes' if variant.useSqrtWidthHeight else 'no'} | {variant.description} |")

    lines += ["", "## Final metrics", "", "| Metric | " + " | ".join(f"`{row['variant']}`" for row in aggregateRows) + " |",
              "|---" * (len(aggregateRows) + 1) + "|"]
    for key, (label, direction) in FINAL_METRICS.items():
        arrow = "↑" if direction == "higher" else "↓"
        lines.append(f"| {label} {arrow} | " + " | ".join(
            formatValue(row[f"{key}Mean"], row[f"{key}Std"]) for row in aggregateRows) + " |")

    lines += ["", "## Paired difference vs `paper` (mean over seeds)", "",
              "| Metric | " + " | ".join(f"`{row['variant']}`" for row in aggregateRows) + " |",
              "|---" * (len(aggregateRows) + 1) + "|"]
    for key in ("valMeanIou", "valMeanIouSmall", "valMeanIouLarge", "valRecallAt50", "valPaperTotal"):
        lines.append(f"| {FINAL_METRICS[key][0]} | " + " | ".join(formatDiff(row, key) for row in aggregateRows) + " |")

    stabilityLabels = {
        "epochsToIou50": ("Epochs until mean IoU ≥ 0.5", 1),
        "bestValMeanIou": ("Best mean IoU (any epoch)", 3),
        "valLossSpikes": (f"Val paper-loss spikes (>{SPIKE_FRACTION:.0%} rise)", 1),
        "trainGradNormMeanAll": ("Train grad norm, mean", 2),
        "trainGradNormPeak": ("Train grad norm, peak", 1),
        "trainGradNormStdSecondHalf": ("Train grad norm std, epochs 16-30", 2),
        "nonFiniteSteps": ("Non-finite steps", 0),
        "probeNoobjToObjRatioFirst": ("noobj / obj grad-norm ratio, epoch 1", 2),
        "probeNoobjToObjRatioMean": ("noobj / obj grad-norm ratio, mean", 2),
        "meanEpochSeconds": ("Seconds per epoch", 1),
    }
    lines += ["", "## Training dynamics and stability", "",
              "| Metric | " + " | ".join(f"`{row['variant']}`" for row in aggregateRows) + " |",
              "|---" * (len(aggregateRows) + 1) + "|"]
    for key, (label, digits) in stabilityLabels.items():
        lines.append(f"| {label} | " + " | ".join(
            formatValue(row[f"{key}Mean"], row[f"{key}Std"], digits) for row in aggregateRows) + " |")
    return "\n".join(lines) + "\n"


def epochSeries(runs: list[dict], variant: str, key: str) -> tuple[list[int], list[float], list[float]]:
    # Mean and std across seeds of one history column, epoch by epoch
    variantRuns = [run["history"] for run in runs if run["variant"] == variant]
    numEpochs = min(len(history) for history in variantRuns)
    epochs, means, stds = [], [], []
    for index in range(numEpochs):
        mean, std = meanStd([history[index][key] for history in variantRuns])
        epochs.append(variantRuns[0][index]["epoch"])
        means.append(mean)
        stds.append(std)
    return epochs, means, stds


def styleAxes(axes, title: str, ylabel: str, xlabel: str = "Epoch") -> None:
    axes.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, fontweight="bold")
    axes.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    axes.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#bdbdbd")
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plotFigures(runs: list[dict], aggregateRows: list[dict], figureDir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping figures (pip install matplotlib)")
        return []
    figureDir.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in aggregateRows]
    written = []

    curves = (
        ("valMeanIou", "Validation mean IoU of the responsible box", "Mean IoU", False, "iouCurves.png"),
        ("valPaperTotal", "Validation loss under the paper's weights", "Loss (paper λ, sqrt w/h)", False,
         "paperLossCurves.png"),
        ("trainGradNormMean", "Training gradient norm (mean per epoch)", "‖∇θ L‖₂", True, "gradNormCurves.png"),
        ("probeNoobjToObjGradRatio", "No-object vs object confidence gradient, fixed probe batch",
         "‖∇ noobj‖ / ‖∇ obj‖", True, "noobjRatioCurves.png"),
    )
    for key, title, ylabel, logScale, fileName in curves:
        figure, axes = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        for variant in variants:
            color, lineStyle = VARIANT_STYLES[variant]
            epochs, means, stds = epochSeries(runs, variant, key)
            axes.plot(epochs, means, color=color, linestyle=lineStyle, linewidth=2, label=variant)
            axes.fill_between(epochs, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)],
                              color=color, alpha=0.12, linewidth=0)
        if logScale:
            axes.set_yscale("log")
        styleAxes(axes, title, ylabel)
        axes.legend(frameon=False, fontsize=8, ncol=3, loc="best", labelcolor=TEXT_PRIMARY)
        figure.text(0.01, 0.01, "Line: mean over seeds. Band: ±1 std.", fontsize=7, color=TEXT_SECONDARY)
        figure.tight_layout(rect=(0, 0.03, 1, 1))
        figure.savefig(figureDir / fileName)
        plt.close(figure)
        written.append(figureDir / fileName)

    # Size-stratified IoU: the sqrt(w, h) claim is about small vs large boxes
    buckets = (("valMeanIouSmall", "Small\n<24 px"), ("valMeanIouMedium", "Medium\n24-48 px"),
               ("valMeanIouLarge", "Large\n≥48 px"))
    figure, axes = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    barWidth = 0.8 / len(variants)
    for index, row in enumerate(aggregateRows):
        color, _ = VARIANT_STYLES[row["variant"]]
        positions = [bucket + (index - (len(variants) - 1) / 2) * barWidth for bucket in range(len(buckets))]
        axes.bar(positions, [row[f"{key}Mean"] for key, _ in buckets], width=barWidth * 0.9, color=color,
                 yerr=[row[f"{key}Std"] for key, _ in buckets], error_kw={"ecolor": "#555555", "linewidth": 1},
                 label=row["variant"])
    axes.set_xticks(range(len(buckets)), [label for _, label in buckets])
    styleAxes(axes, "Final mean IoU by object size", "Mean IoU (last 3 epochs)", xlabel="Ground-truth √(w·h)")
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", labelcolor=TEXT_PRIMARY)
    axes.set_ylim(0, 1)
    figure.text(0.01, 0.01, "Bar: mean over seeds. Whisker: ±1 std.", fontsize=7, color=TEXT_SECONDARY)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(figureDir / "iouBySize.png")
    plt.close(figure)
    written.append(figureDir / "iouBySize.png")
    return written


def analyze(outputRoot: str | Path = DEFAULT_OUTPUT_ROOT, figureDir: str | Path | None = None) -> Path:
    outputRoot = Path(outputRoot)
    runs = loadRuns(outputRoot)
    if not runs:
        raise FileNotFoundError(f"No finished ablation runs under {outputRoot}")
    runSummaries = [runSummary(run) for run in runs]
    aggregateRows = aggregate(runSummaries)
    writeCsv(outputRoot / "runSummary.csv", runSummaries)
    writeCsv(outputRoot / "variantSummary.csv", aggregateRows)
    reportPath = outputRoot / "summary.md"
    reportPath.write_text(markdownReport(aggregateRows, runSummaries))
    plotFigures(runs, aggregateRows, Path(figureDir) if figureDir else outputRoot / "figures")
    return reportPath


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize the loss ablation runs")
    parser.add_argument("--outputRoot", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--figureDir", default=None, help="default: <outputRoot>/figures")
    arguments = parser.parse_args(argv)
    reportPath = analyze(arguments.outputRoot, arguments.figureDir)
    print(reportPath.read_text())


if __name__ == "__main__":
    main()
