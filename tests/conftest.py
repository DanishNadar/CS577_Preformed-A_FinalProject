import pytest

from src.datasets.syntheticShapes import SyntheticConfig
from src.utils.config import loadExperimentConfig


@pytest.fixture
def tinySyntheticConfig(tmp_path):
    return SyntheticConfig(root=str(tmp_path / "synthetic"), numTrain=48, numVal=16)


@pytest.fixture
def tinyExperimentConfig(tmp_path):
    return loadExperimentConfig(
        "configs/baseline.yaml",
        {
            "device": "cpu",
            "outputDir": str(tmp_path / "results"),
            "checkpointDir": str(tmp_path / "checkpoints"),
            "data.synthetic.root": str(tmp_path / "synthetic"),
            "data.synthetic.numTrain": 48,
            "data.synthetic.numVal": 16,
            "training.batchSize": 16,
            "training.epochs": 1,
        },
    )
