# YOLOv1-style detector: (N, 3, 112, 112) -> (N, 7, 7, 13) = 2 x [x, y, w, h, conf] + 3 class scores per cell

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

OUTPUT_ACTIVATIONS = ("sigmoid", "linear")


@dataclass
class ModelConfig:
    imageSize: int = 112
    inputChannels: int = 3
    gridSize: int = 7
    numBoxes: int = 2
    numClasses: int = 3
    convChannels: Sequence[int] = (16, 32, 64, 128)
    hiddenSize: int = 512
    dropout: float = 0.5
    leakySlope: float = 0.1
    # "sigmoid" squashes box/confidence values and softmaxes class scores; "linear" keeps raw outputs like the paper
    outputActivation: str = "sigmoid"

    @property
    def predictionDepth(self) -> int:
        return self.numBoxes * 5 + self.numClasses


def convBlock(inChannels: int, outChannels: int, leakySlope: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inChannels, outChannels, kernel_size=3, padding=1),
        nn.LeakyReLU(leakySlope),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class YoloModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self._validateConfig()

        channelSizes = [cfg.inputChannels, *cfg.convChannels]
        self.features = nn.Sequential(
            *[convBlock(channelSizes[i], channelSizes[i + 1], cfg.leakySlope) for i in range(len(cfg.convChannels))]
        )
        flattenedSize = cfg.convChannels[-1] * cfg.gridSize * cfg.gridSize
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattenedSize, cfg.hiddenSize),
            nn.LeakyReLU(cfg.leakySlope),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hiddenSize, cfg.gridSize * cfg.gridSize * cfg.predictionDepth),
        )

    def _validateConfig(self) -> None:
        cfg = self.config
        downsampledSize = cfg.imageSize // (2 ** len(cfg.convChannels))
        if downsampledSize != cfg.gridSize or cfg.imageSize % (2 ** len(cfg.convChannels)) != 0:
            raise ValueError(
                f"{len(cfg.convChannels)} pooling stages map {cfg.imageSize}px to {downsampledSize}, "
                f"but gridSize is {cfg.gridSize}"
            )
        if cfg.outputActivation not in OUTPUT_ACTIVATIONS:
            raise ValueError(f"outputActivation must be one of {OUTPUT_ACTIVATIONS}")

    def rawPredictions(self, images: Tensor) -> Tensor:
        # Head output before activation, shape (N, S, S, B*5 + C)
        cfg = self.config
        featureMap = self.features(images)  # (N, convChannels[-1], S, S)
        return self.head(featureMap).view(-1, cfg.gridSize, cfg.gridSize, cfg.predictionDepth)

    def activatePredictions(self, rawPredictions: Tensor) -> Tensor:
        if self.config.outputActivation == "linear":
            return rawPredictions
        boxValueCount = self.config.numBoxes * 5
        boxValues = rawPredictions[..., :boxValueCount].sigmoid()
        classProbabilities = rawPredictions[..., boxValueCount:].softmax(dim=-1)
        return torch.cat((boxValues, classProbabilities), dim=-1)

    def forward(self, images: Tensor) -> Tensor:
        # images (N, 3, H, W) in [0, 1] -> predictions (N, S, S, B*5 + C)
        return self.activatePredictions(self.rawPredictions(images))


def createModel(config: ModelConfig | None = None, seed: int | None = None) -> YoloModel:
    # Passing a seed makes the weight initialization reproducible
    if seed is not None:
        torch.manual_seed(seed)
    return YoloModel(config)


def countParameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
