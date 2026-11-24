# Per-epoch history in history.json/history.csv: epoch, learningRate, epochSeconds, train<Component>, val<Component>

from __future__ import annotations

import csv
import json
from pathlib import Path

HISTORY_JSON = "history.json"
HISTORY_CSV = "history.csv"


def prefixedMetrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {prefix + name[0].upper() + name[1:]: value for name, value in metrics.items()}


def writeHistory(outputDir: str | Path, trainingHistory: list[dict]) -> None:
    outputDir = Path(outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)
    (outputDir / HISTORY_JSON).write_text(json.dumps(trainingHistory, indent=2))

    columnNames = list(dict.fromkeys(key for row in trainingHistory for key in row))
    with open(outputDir / HISTORY_CSV, "w", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=columnNames)
        writer.writeheader()
        writer.writerows(trainingHistory)


def readHistory(outputDir: str | Path) -> list[dict]:
    return json.loads((Path(outputDir) / HISTORY_JSON).read_text())
