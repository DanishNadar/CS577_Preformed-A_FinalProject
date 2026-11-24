# Experiment configuration: YAML file -> nested dataclasses, with dotted command-line overrides

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import yaml

from src.datasets.loaders import DataConfig, numClassesFor
from src.losses.yoloLoss import YoloLossConfig
from src.models.yoloModel import ModelConfig


@dataclass
class TrainingConfig:
    batchSize: int = 32
    epochs: int = 30
    learningRate: float = 1e-3
    weightDecay: float = 0.0
    gradientClipNorm: float | None = None


@dataclass
class ExperimentConfig:
    experimentName: str = "baseline"
    seed: int = 577
    device: str = "auto"
    outputDir: str = "results/baseline"
    checkpointDir: str = "checkpoints/baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: YoloLossConfig = field(default_factory=YoloLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def dataclassFromDict(cls: type, values: dict):
    # Build a (possibly nested) dataclass, rejecting keys it does not define
    fieldNames = {f.name for f in fields(cls)}
    unknownKeys = set(values) - fieldNames
    if unknownKeys:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknownKeys)}")
    typeHints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in values.items():
        fieldType = typeHints[name]
        kwargs[name] = dataclassFromDict(fieldType, value) if is_dataclass(fieldType) and isinstance(value, dict) else value
    return cls(**kwargs)


def setDottedValue(values: dict, dottedKey: str, value) -> None:
    keys = dottedKey.split(".")
    for key in keys[:-1]:
        values = values.setdefault(key, {})
    values[keys[-1]] = value


def loadExperimentConfig(path: str | Path | None = None, overrides: dict | None = None) -> ExperimentConfig:
    # Overrides use dotted keys, e.g. {"training.epochs": 2}
    values = yaml.safe_load(Path(path).read_text()) if path else {}
    values = values or {}
    for dottedKey, value in (overrides or {}).items():
        setDottedValue(values, dottedKey, value)
    config = dataclassFromDict(ExperimentConfig, values)
    syncSharedDimensions(config)
    return config


def syncSharedDimensions(config: ExperimentConfig) -> None:
    # Image size, grid size and class count come from the data section and are copied into the model
    config.model.imageSize = config.data.imageSize
    config.model.gridSize = config.data.gridSize
    config.model.numClasses = numClassesFor(config.data)
    config.model.convChannels = list(config.model.convChannels)


def configToDict(config: ExperimentConfig) -> dict:
    return asdict(config)


def parseOverrideValue(text: str):
    # Interpret command-line override text with YAML rules (numbers, booleans, lists, null)
    return yaml.safe_load(text)
