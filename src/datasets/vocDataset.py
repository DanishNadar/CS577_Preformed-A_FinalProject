# PASCAL VOC 2007 three-class adapter; expects <root>/Annotations, JPEGImages and ImageSets/Main/<imageSet>.txt

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.targetEncoder import encodeTarget

DEFAULT_VOC_CLASSES = ("aeroplane", "car", "dog")


@dataclass
class VocConfig:
    root: str = "data/VOCdevkit/VOC2007"
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_VOC_CLASSES))
    imageSet: str = "trainval"
    valFraction: float = 0.2
    includeDifficult: bool = False
    maxImages: int | None = None


def parseVocAnnotation(xmlPath: Path, classNames: list[str], includeDifficult: bool) -> Tensor:
    # Parse one XML file into (K, 5) normalized [classId, centerX, centerY, width, height]
    root = ElementTree.parse(xmlPath).getroot()
    imageWidth = float(root.find("size/width").text)
    imageHeight = float(root.find("size/height").text)

    rows = []
    for objectNode in root.findall("object"):
        className = objectNode.find("name").text.strip()
        difficultNode = objectNode.find("difficult")
        isDifficult = difficultNode is not None and difficultNode.text.strip() == "1"
        if className not in classNames or (isDifficult and not includeDifficult):
            continue
        boxNode = objectNode.find("bndbox")
        # VOC pixel indices are 1-based and inclusive; convert to continuous edges
        left = float(boxNode.find("xmin").text) - 1
        top = float(boxNode.find("ymin").text) - 1
        right = float(boxNode.find("xmax").text)
        bottom = float(boxNode.find("ymax").text)
        left, right = np.clip([left, right], 0, imageWidth) / imageWidth
        top, bottom = np.clip([top, bottom], 0, imageHeight) / imageHeight
        if right <= left or bottom <= top:
            continue
        rows.append([classNames.index(className), (left + right) / 2, (top + bottom) / 2, right - left, bottom - top])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)


def loadVocSplit(config: VocConfig, split: str, seed: int) -> tuple[list[str], list[Tensor]]:
    # Seeded train/val split of the image set, keeping only images with a selected-class object
    root = Path(config.root)
    imageSetPath = root / "ImageSets" / "Main" / f"{config.imageSet}.txt"
    if not imageSetPath.exists():
        raise FileNotFoundError(
            f"VOC image set not found at {imageSetPath}. Download VOC2007 trainval and set data.voc.root (see README)."
        )
    imageIds = sorted(line.strip() for line in imageSetPath.read_text().splitlines() if line.strip())

    keptIds, keptAnnotations = [], []
    for imageId in imageIds:
        annotations = parseVocAnnotation(root / "Annotations" / f"{imageId}.xml", config.classes, config.includeDifficult)
        if len(annotations) > 0:
            keptIds.append(imageId)
            keptAnnotations.append(annotations)

    order = np.random.default_rng(seed).permutation(len(keptIds))
    if config.maxImages is not None:
        order = order[: config.maxImages]
    valCount = int(round(len(order) * config.valFraction))
    selected = order[:valCount] if split == "val" else order[valCount:]
    selected = sorted(selected.tolist())
    return [keptIds[i] for i in selected], [keptAnnotations[i] for i in selected]


class VocDetectionDataset(Dataset):
    # Same item interface as SyntheticShapesDataset: (image, target, annotations)

    def __init__(self, config: VocConfig, split: str, seed: int, imageSize: int = 112, gridSize: int = 7):
        if len(config.classes) < 1:
            raise ValueError("VOC adapter needs at least one class")
        self.config = config
        self.classNames = tuple(config.classes)
        self.imageSize = imageSize
        self.imageDir = Path(config.root) / "JPEGImages"
        self.imageIds, self.annotations = loadVocSplit(config, split, seed)

        encoded = [encodeTarget(annotations, gridSize, len(self.classNames)) for annotations in self.annotations]
        self.targets = torch.stack([target for target, _ in encoded]) if encoded else torch.empty(0)
        self.droppedObjects = sum(dropped for _, dropped in encoded)
        self.imagesWithCollisions = sum(dropped > 0 for _, dropped in encoded)

    def __len__(self) -> int:
        return len(self.imageIds)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        image = Image.open(self.imageDir / f"{self.imageIds[index]}.jpg").convert("RGB")
        image = image.resize((self.imageSize, self.imageSize), Image.BILINEAR)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)
        return pixels.float() / 255.0, self.targets[index], self.annotations[index]

    def collisionSummary(self) -> dict:
        return {
            "encoderDroppedObjects": self.droppedObjects,
            "imagesWithCollisions": self.imagesWithCollisions,
            "totalObjects": sum(len(annotations) for annotations in self.annotations),
        }
