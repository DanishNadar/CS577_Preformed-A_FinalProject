"""End-to-end CPU smoke test (well under a minute): tiny synthetic data -> train 2 epochs ->
reload checkpoint -> decode predictions -> read logs. Everything is written to a temporary folder.

    python scripts/smokeTest.py
"""

import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.datasets.loaders import buildDataLoaders  # noqa: E402
from src.losses.yoloLoss import YoloLoss  # noqa: E402
from src.training.checkpoint import loadModelFromCheckpoint  # noqa: E402
from src.training.history import readHistory  # noqa: E402
from src.training.inference import collectPredictions  # noqa: E402
from src.training.train import train  # noqa: E402
from src.utils.config import loadExperimentConfig  # noqa: E402
from src.utils.decoding import predictionsToDetections  # noqa: E402


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"SMOKE TEST FAILED: {message}")
    print(f"  ok  {message}")


def main() -> None:
    startTime = time.perf_counter()
    with tempfile.TemporaryDirectory() as workDir:
        config = loadExperimentConfig("configs/baseline.yaml", {
            "device": "cpu",
            "outputDir": f"{workDir}/results",
            "checkpointDir": f"{workDir}/checkpoints",
            "data.synthetic.root": f"{workDir}/synthetic",
            "data.synthetic.numTrain": 64,
            "data.synthetic.numVal": 32,
            "training.batchSize": 16,
            "training.epochs": 2,
        })

        print("training 2 epochs on 64/32 synthetic images")
        result = train(config)
        check(len(result.trainingHistory) == 2, "two history rows recorded")
        check(all(math.isfinite(row["trainTotal"]) for row in result.trainingHistory), "finite training losses")

        model, checkpoint = loadModelFromCheckpoint(result.bestCheckpointPath)
        check(checkpoint["epoch"] in (1, 2), f"best checkpoint reloaded (epoch {checkpoint['epoch']})")

        _, valLoader = buildDataLoaders(config.data, 16, config.seed)
        outputs = collectPredictions(model, valLoader)
        check(outputs.predictions.shape == (32, 7, 7, 13), f"predictions shape {tuple(outputs.predictions.shape)}")
        reloadedLoss = YoloLoss(config.loss)(outputs.predictions, outputs.targets)["total"].item()
        check(math.isclose(reloadedLoss, checkpoint["bestValLoss"], rel_tol=1e-3),
              f"reloaded model reproduces best val loss ({reloadedLoss:.4f})")

        detections = predictionsToDetections(outputs.predictions, numBoxes=config.model.numBoxes)
        check(len(detections) == 32, f"decoded detections for 32 images ({sum(len(d) for d in detections)} boxes)")
        check(readHistory(config.outputDir) == result.trainingHistory, "history.json readable")
        check(Path(config.outputDir, "history.csv").exists(), "history.csv written")

    print(f"SMOKE TEST PASSED in {time.perf_counter() - startTime:.1f}s (torch {torch.__version__})")


if __name__ == "__main__":
    main()
