# Evaluation figures: mAP per variant, precision-recall curves, error breakdown by object size, and example
# detections for the proposal's three losses side by side. Styling follows src/ablations/analysis.py.

from __future__ import annotations

import math
import statistics
from pathlib import Path

from src.ablations.analysis import TEXT_PRIMARY, TEXT_SECONDARY, VARIANT_STYLES, styleAxes
from src.ablations.variants import DEFAULT_SEED, PROPOSAL_VARIANTS, VARIANTS
from src.evaluation.detectionMetrics import ERROR_TYPES, SIZE_BUCKETS
from src.utils.decoding import predictionsToDetections

ERROR_COLORS = {"correct": "#00b894", "localization": "#f2a600", "classification": "#6a4ce2", "missed": "#bdbdbd"}
CLASS_COLORS = ("#257de0", "#f47230", "#ee7fac")
NUM_EXAMPLES = 6


def _meanStd(values: list[float]) -> tuple[float, float]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return float("nan"), 0.0
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def exampleIndices(annotations, count: int = NUM_EXAMPLES) -> list[int]:
    # Deterministic picks: the first images with several objects, so each panel shows a mix of sizes
    crowded = [index for index, rows in enumerate(annotations) if rows.shape[0] >= 2]
    return (crowded + [index for index in range(len(annotations)) if index not in crowded])[:count]


def plotEvaluationFigures(runRows: list[dict], errorRows: list[dict], results: dict, figureDir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed; skipping figures (pip install matplotlib)")
        return []
    figureDir.mkdir(parents=True, exist_ok=True)
    variants = [name for name in VARIANTS if any(row["variant"] == name for row in runRows)]
    written = []

    # 1. mAP per variant at three IoU strictness levels
    keys = (("mAP50", "mAP@0.5"), ("mAP75", "mAP@0.75"), ("mAP50to95", "mAP@[.5:.95]"))
    figure, axes = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    barWidth = 0.8 / len(variants)
    for index, variant in enumerate(variants):
        color, _ = VARIANT_STYLES[variant]
        stats = [_meanStd([row[key] for row in runRows if row["variant"] == variant]) for key, _ in keys]
        positions = [group + (index - (len(variants) - 1) / 2) * barWidth for group in range(len(keys))]
        axes.bar(positions, [mean for mean, _ in stats], width=barWidth * 0.9, color=color, label=variant,
                 yerr=[std for _, std in stats], error_kw={"ecolor": "#555555", "linewidth": 1})
    axes.set_xticks(range(len(keys)), [label for _, label in keys])
    axes.set_ylim(0, 1)
    styleAxes(axes, "Detection mAP on the validation split", "mAP", xlabel="IoU threshold")
    axes.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", labelcolor=TEXT_PRIMARY)
    figure.text(0.01, 0.01, "Bar: mean over seeds. Whisker: ±1 std.", fontsize=7, color=TEXT_SECONDARY)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    written.append(_save(figure, plt, figureDir / "mapByVariant.png"))

    # 2. Precision-recall at IoU 0.5, one panel per class, reference seed
    referenceSeed = DEFAULT_SEED if any(seed == DEFAULT_SEED for _, seed in results) else min(s for _, s in results)
    classNames = next(iter(results.values())).classNames
    figure, panels = plt.subplots(1, len(classNames), figsize=(10, 3.6), dpi=150, sharey=True)
    for panel, className in zip(panels, classNames):
        for variant in variants:
            result = results.get((variant, referenceSeed))
            if result is None:
                continue
            color, lineStyle = VARIANT_STYLES[variant]
            curve = result.curves[className]
            panel.plot(curve["recall"], curve["precision"], color=color, linestyle=lineStyle, linewidth=1.6,
                       label=variant)
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1.02)
        styleAxes(panel, className, "Precision" if panel is panels[0] else "", xlabel="Recall")
    panels[0].legend(frameon=False, fontsize=7, loc="lower left", labelcolor=TEXT_PRIMARY)
    figure.suptitle(f"Precision-recall at IoU 0.5 (seed {referenceSeed})", x=0.01, ha="left", fontsize=11,
                    fontweight="bold", color=TEXT_PRIMARY)
    figure.tight_layout()
    written.append(_save(figure, plt, figureDir / "precisionRecall.png"))

    # 3. Error breakdown by object size, pooled over seeds
    buckets = [name for name, _, _ in SIZE_BUCKETS]
    figure, panels = plt.subplots(1, len(buckets), figsize=(10, 3.8), dpi=150, sharey=True)
    for panel, bucket in zip(panels, buckets):
        lefts = [0.0] * len(variants)
        for kind in ERROR_TYPES:
            shares = []
            for variant in variants:
                rows = [row for row in errorRows if row["variant"] == variant and row["bucket"] == bucket]
                total = sum(row[key] for row in rows for key in ERROR_TYPES)
                shares.append(sum(row[kind] for row in rows) / total if total else 0.0)
            panel.barh(range(len(variants)), shares, left=lefts, color=ERROR_COLORS[kind], height=0.7,
                       label=kind if bucket == buckets[0] else None)
            lefts = [left + share for left, share in zip(lefts, shares)]
        panel.set_yticks(range(len(variants)), variants)
        panel.invert_yaxis()
        panel.set_xlim(0, 1)
        styleAxes(panel, f"{bucket.capitalize()} objects", "", xlabel="Share of ground-truth objects")
        panel.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        panel.grid(axis="y", visible=False)
    figure.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", labelcolor=TEXT_PRIMARY)
    figure.suptitle("What happens to each object (score ≥ 0.2, all seeds)", x=0.01, ha="left", fontsize=11,
                    fontweight="bold", color=TEXT_PRIMARY)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    written.append(_save(figure, plt, figureDir / "errorBreakdown.png"))

    # 4. Example detections: proposal variants (rows) on the same validation images (columns)
    rowVariants = [variant for variant in PROPOSAL_VARIANTS if (variant, referenceSeed) in results]
    if rowVariants:
        firstResult = results[(rowVariants[0], referenceSeed)]
        indices = exampleIndices(firstResult.annotations)
        figure, grid = plt.subplots(len(rowVariants), len(indices), figsize=(1.9 * len(indices), 2.05 * len(rowVariants)),
                                    dpi=150, squeeze=False)
        for rowIndex, variant in enumerate(rowVariants):
            result = results[(variant, referenceSeed)]
            detections = predictionsToDetections(result.predictions[indices], 2, 0.2)
            for column, imageIndex in enumerate(indices):
                axes = grid[rowIndex][column]
                image, _, annotations = result.dataset[imageIndex]
                size = image.shape[-1]
                axes.imshow(image.permute(1, 2, 0).numpy())
                for annotation in annotations:
                    centerX, centerY, width, height = (float(value) * size for value in annotation[1:5])
                    axes.add_patch(Rectangle((centerX - width / 2, centerY - height / 2), width, height, fill=False,
                                             edgecolor="white", linewidth=1.2, linestyle=(0, (3, 2))))
                for left, top, right, bottom, score, classId in detections[column].tolist():
                    color = CLASS_COLORS[int(classId) % len(CLASS_COLORS)]
                    axes.add_patch(Rectangle((left * size, top * size), (right - left) * size, (bottom - top) * size,
                                             fill=False, edgecolor=color, linewidth=1.6))
                    axes.text(left * size, top * size - 1, f"{result.classNames[int(classId)]} {score:.2f}",
                              fontsize=4.5, color="white", va="bottom",
                              bbox={"facecolor": color, "edgecolor": "none", "pad": 0.6})
                axes.set_xticks([])
                axes.set_yticks([])
                if column == 0:
                    axes.set_ylabel(variant, fontsize=9, color=TEXT_PRIMARY)
        figure.suptitle(f"Detections (score ≥ 0.2, solid, colored by class) vs ground truth (white dashed), "
                        f"seed {referenceSeed}", x=0.01, ha="left", fontsize=9, color=TEXT_PRIMARY)
        figure.tight_layout()
        written.append(_save(figure, plt, figureDir / "detectionExamples.png"))
    return written


def _save(figure, plt, path: Path) -> Path:
    figure.savefig(path)
    plt.close(figure)
    return path
