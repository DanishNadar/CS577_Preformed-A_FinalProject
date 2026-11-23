import pytest
import torch

from src.models.yoloModel import ModelConfig, YoloModel, createModel


@pytest.mark.parametrize("batchSize", [1, 32])
def testOutputShape(batchSize):
    model = createModel(ModelConfig(), seed=0).eval()
    predictions = model(torch.rand(batchSize, 3, 112, 112))
    assert predictions.shape == (batchSize, 7, 7, 13)
    assert predictions.device.type == "cpu"


def testSigmoidOutputRanges():
    model = createModel(ModelConfig(), seed=0).eval()
    predictions = model(torch.rand(4, 3, 112, 112))
    assert predictions[..., :10].min() >= 0 and predictions[..., :10].max() <= 1
    torch.testing.assert_close(predictions[..., 10:].sum(-1), torch.ones(4, 7, 7))


def testLinearOutputMatchesRawHead():
    model = createModel(ModelConfig(outputActivation="linear"), seed=0).eval()
    images = torch.rand(2, 3, 112, 112)
    torch.testing.assert_close(model(images), model.rawPredictions(images))


def testSeededInitializationIsDeterministic():
    firstModel, secondModel = createModel(seed=123), createModel(seed=123)
    for first, second in zip(firstModel.parameters(), secondModel.parameters()):
        assert torch.equal(first, second)
    assert not torch.equal(next(createModel(seed=124).parameters()), next(firstModel.parameters()))


def testConfigurableDimensions():
    config = ModelConfig(gridSize=7, numBoxes=3, numClasses=5, convChannels=(8, 8, 16, 16), hiddenSize=64)
    assert YoloModel(config)(torch.rand(2, 3, 112, 112)).shape == (2, 7, 7, 3 * 5 + 5)


def testInconsistentGridRaises():
    with pytest.raises(ValueError):
        YoloModel(ModelConfig(gridSize=14))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def testForwardOnGpu():
    model = createModel(seed=0).cuda().eval()
    assert model(torch.rand(2, 3, 112, 112, device="cuda")).shape == (2, 7, 7, 13)
