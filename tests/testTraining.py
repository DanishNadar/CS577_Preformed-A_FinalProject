import csv

import torch

from src.datasets.loaders import buildDataLoaders
from src.datasets.targetEncoder import encodeTarget
from src.losses.yoloLoss import LOSS_COMPONENTS
from src.models.yoloModel import createModel
from src.training.checkpoint import loadCheckpoint, loadModelFromCheckpoint, saveCheckpoint
from src.training.history import readHistory
from src.training.inference import collectPredictions
from src.training.train import main, train
from src.utils.boxes import boxIou, cxcywhToXyxy
from src.utils.config import configToDict, loadExperimentConfig
from src.utils.decoding import decodePredictions, predictionsToDetections


def testCheckpointRoundTrip(tmp_path, tinyExperimentConfig):
    model = createModel(tinyExperimentConfig.model, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model(torch.rand(2, 3, 112, 112)).sum().backward()
    optimizer.step()
    history = [{"epoch": 1, "trainTotal": 1.0}]
    path = saveCheckpoint(tmp_path / "ckpt.pt", model, optimizer, 3, configToDict(tinyExperimentConfig), 0.5, history)

    restoredModel, checkpoint = loadModelFromCheckpoint(path)
    assert checkpoint["epoch"] == 3 and checkpoint["bestValLoss"] == 0.5
    assert checkpoint["seed"] == tinyExperimentConfig.seed and checkpoint["trainingHistory"] == history
    for original, restored in zip(model.state_dict().values(), restoredModel.state_dict().values()):
        assert torch.equal(original, restored)

    images = torch.rand(3, 3, 112, 112)
    model.eval()
    torch.testing.assert_close(model(images), restoredModel(images))

    restoredOptimizer = torch.optim.Adam(restoredModel.parameters(), lr=1e-3)
    restoredOptimizer.load_state_dict(checkpoint["optimizerState"])
    assert restoredOptimizer.state_dict()["state"][0]["step"] == optimizer.state_dict()["state"][0]["step"]


def testOneFullTrainingEpoch(tinyExperimentConfig):
    result = train(tinyExperimentConfig, verbose=False)
    assert len(result.trainingHistory) == 1
    row = result.trainingHistory[0]
    for component in LOSS_COMPONENTS:
        suffix = component[0].upper() + component[1:]
        assert torch.isfinite(torch.tensor(row["train" + suffix]))
        assert torch.isfinite(torch.tensor(row["val" + suffix]))
    assert row["epoch"] == 1 and row["learningRate"] == 1e-3 and row["epochSeconds"] > 0
    assert result.lastCheckpointPath.exists() and result.bestCheckpointPath.exists()

    outputDir = tinyExperimentConfig.outputDir
    assert readHistory(outputDir) == result.trainingHistory
    with open(f"{outputDir}/history.csv") as csvFile:
        csvRows = list(csv.DictReader(csvFile))
    assert len(csvRows) == 1 and float(csvRows[0]["valTotal"]) == row["valTotal"]


def testResumeMatchesUninterruptedRun(tmp_path, tinyExperimentConfig):
    tinyExperimentConfig.training.epochs = 2
    uninterrupted = train(tinyExperimentConfig, verbose=False)
    uninterruptedState = loadCheckpoint(uninterrupted.lastCheckpointPath)["modelState"]

    tinyExperimentConfig.checkpointDir = str(tmp_path / "resumedCheckpoints")
    tinyExperimentConfig.outputDir = str(tmp_path / "resumedResults")
    tinyExperimentConfig.training.epochs = 1
    firstHalf = train(tinyExperimentConfig, verbose=False)
    tinyExperimentConfig.training.epochs = 2
    resumed = train(tinyExperimentConfig, resumePath=firstHalf.lastCheckpointPath, verbose=False)

    assert [row["epoch"] for row in resumed.trainingHistory] == [1, 2]
    assert resumed.trainingHistory[1]["trainTotal"] == uninterrupted.trainingHistory[1]["trainTotal"]
    resumedState = loadCheckpoint(resumed.lastCheckpointPath)["modelState"]
    for name, value in uninterruptedState.items():
        torch.testing.assert_close(resumedState[name], value)


def testCommandLineEntryPoint(tmp_path):
    result = main([
        "--config", "configs/baseline.yaml",
        "--set", "device=cpu",
        "--set", f"outputDir={tmp_path / 'results'}",
        "--set", f"checkpointDir={tmp_path / 'checkpoints'}",
        "--set", f"data.synthetic.root={tmp_path / 'synthetic'}",
        "--set", "data.synthetic.numTrain=16",
        "--set", "data.synthetic.numVal=8",
        "--set", "training.epochs=1",
        "--set", "training.batchSize=8",
    ])
    assert result.lastCheckpointPath.exists()


def testInferenceAndDecodingInterfaces(tinyExperimentConfig):
    result = train(tinyExperimentConfig, verbose=False)
    model, checkpoint = loadModelFromCheckpoint(result.bestCheckpointPath)
    _, valLoader = buildDataLoaders(tinyExperimentConfig.data, 8, tinyExperimentConfig.seed)
    outputs = collectPredictions(model, valLoader)
    assert outputs.predictions.shape == (16, 7, 7, 13) and len(outputs.annotations) == 16

    decoded = decodePredictions(outputs.predictions, numBoxes=2)
    assert decoded.imageBoxes.shape == (16, 7, 7, 2, 4)
    assert decoded.classScores.shape == (16, 7, 7, 2, 3)
    detections = predictionsToDetections(outputs.predictions, numBoxes=2, scoreThreshold=0.0)
    assert len(detections) == 16 and all(d.shape[1] == 6 for d in detections)


def testDecodingRecoversKnownBox():
    # A perfect prediction for one object must decode back to the ground-truth box
    annotation = torch.tensor([[2.0, 0.3, 0.6, 0.25, 0.4]])
    target, _ = encodeTarget(annotation, 7, 3)
    predictions = torch.zeros(1, 7, 7, 13)
    predictions[0, 4, 2, 5:10] = torch.cat((target[4, 2, :4], torch.tensor([0.9])))
    predictions[0, 4, 2, 10:] = torch.tensor([0.0, 0.0, 1.0])

    detections = predictionsToDetections(predictions, numBoxes=2, scoreThreshold=0.5)[0]
    assert detections.shape == (1, 6)
    expectedCorners = cxcywhToXyxy(annotation[:, 1:])
    assert boxIou(detections[:, :4], expectedCorners).item() > 0.999
    assert detections[0, 4].item() > 0.89 and detections[0, 5].item() == 2.0


def testConfigOverridesAndValidation():
    config = loadExperimentConfig("configs/baseline.yaml", {"training.epochs": 3, "loss.lambdaNoobj": 1.0})
    assert config.training.epochs == 3 and config.loss.lambdaNoobj == 1.0
    assert config.model.numClasses == 3 and config.model.gridSize == 7
    try:
        loadExperimentConfig("configs/baseline.yaml", {"training.epochz": 3})
    except ValueError as error:
        assert "epochz" in str(error)
    else:
        raise AssertionError("unknown key was accepted")
