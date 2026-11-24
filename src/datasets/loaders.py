# Dataset selection and DataLoader construction shared by training, smoke tests and evaluation

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.datasets.syntheticShapes import SyntheticConfig, SyntheticShapesDataset, ensureSyntheticDataset
from src.datasets.vocDataset import VocConfig, VocDetectionDataset
from src.utils.reproducibility import seedDataLoaderWorker

DATASET_NAMES = ("synthetic", "voc")


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    imageSize: int = 112
    gridSize: int = 7
    numWorkers: int = 0
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    voc: VocConfig = field(default_factory=VocConfig)


def collateDetections(batch: list[tuple[Tensor, Tensor, Tensor]]) -> tuple[Tensor, Tensor, list[Tensor]]:
    # Stack images and targets; annotations stay a list because object counts vary
    images, targets, annotations = zip(*batch)
    return torch.stack(images), torch.stack(targets), list(annotations)


def buildDatasets(config: DataConfig, seed: int) -> tuple[Dataset, Dataset]:
    if config.dataset == "synthetic":
        root = ensureSyntheticDataset(config.synthetic, seed, config.imageSize, config.gridSize)
        return (
            SyntheticShapesDataset(root / "train", config.gridSize),
            SyntheticShapesDataset(root / "val", config.gridSize),
        )
    if config.dataset == "voc":
        return (
            VocDetectionDataset(config.voc, "train", seed, config.imageSize, config.gridSize),
            VocDetectionDataset(config.voc, "val", seed, config.imageSize, config.gridSize),
        )
    raise ValueError(f"Unknown dataset '{config.dataset}', expected one of {DATASET_NAMES}")


def buildDataLoaders(
    config: DataConfig, batchSize: int, seed: int, pinMemory: bool = False
) -> tuple[DataLoader, DataLoader]:
    trainDataset, valDataset = buildDatasets(config, seed)
    shuffleGenerator = torch.Generator().manual_seed(seed)
    loaderOptions = dict(
        batch_size=batchSize,
        num_workers=config.numWorkers,
        collate_fn=collateDetections,
        worker_init_fn=seedDataLoaderWorker,
        pin_memory=pinMemory,
    )
    trainLoader = DataLoader(trainDataset, shuffle=True, generator=shuffleGenerator, **loaderOptions)
    valLoader = DataLoader(valDataset, shuffle=False, **loaderOptions)
    return trainLoader, valLoader


def numClassesFor(config: DataConfig) -> int:
    return len(SyntheticShapesDataset.classNames) if config.dataset == "synthetic" else len(config.voc.classes)
