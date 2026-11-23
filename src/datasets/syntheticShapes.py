"""Synthetic circles / squares / triangles detection dataset.

ETL stages:
  extract   generateSplit() renders images with exact boxes taken from each shape's drawn mask
  transform annotations are normalized to cxcywh in [0, 1] and encoded into YOLO targets
  load      writeSplit() stores PNGs + annotations.json; SyntheticShapesDataset serves tensors

On disk:
  <root>/manifest.json                generation parameters (used to detect stale data)
  <root>/<split>/annotations.json     per-image objects plus split statistics
  <root>/<split>/images/000000.png
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from src.datasets.targetEncoder import encodeTarget, responsibleCell

CLASS_NAMES = ("circle", "square", "triangle")
SPLIT_IDS = {"train": 0, "val": 1}


@dataclass
class SyntheticConfig:
    root: str = "data/synthetic"
    numTrain: int = 2000
    numVal: int = 500
    minObjects: int = 1
    maxObjects: int = 3
    minObjectSize: int = 8
    maxObjectSize: int = 80
    avoidCellCollisions: bool = True
    maxPlacementAttempts: int = 100
    minColorDistance: float = 120.0
    backgroundNoise: float = 6.0


@dataclass
class ShapeObject:
    classId: int
    boxPixels: tuple[int, int, int, int]  # left, top, right, bottom (right/bottom exclusive)

    def normalizedBox(self, imageSize: int) -> list[float]:
        left, top, right, bottom = self.boxPixels
        return [
            (left + right) / 2 / imageSize,
            (top + bottom) / 2 / imageSize,
            (right - left) / imageSize,
            (bottom - top) / imageSize,
        ]


def splitRng(seed: int, split: str) -> np.random.Generator:
    return np.random.default_rng([seed, SPLIT_IDS[split]])


def contrastingColor(rng: np.random.Generator, background: np.ndarray, minDistance: float) -> tuple[int, int, int]:
    for _ in range(100):
        color = rng.integers(0, 256, size=3)
        if np.linalg.norm(color - background) >= minDistance:
            return tuple(int(value) for value in color)
    return tuple(int(255 - value) for value in background)


def drawShapeMask(classId: int, left: int, top: int, width: int, height: int, imageSize: int) -> Image.Image:
    mask = Image.new("L", (imageSize, imageSize), 0)
    draw = ImageDraw.Draw(mask)
    right, bottom = left + width - 1, top + height - 1
    if CLASS_NAMES[classId] == "circle":
        draw.ellipse((left, top, right, bottom), fill=255)
    elif CLASS_NAMES[classId] == "square":
        draw.rectangle((left, top, right, bottom), fill=255)
    else:
        draw.polygon([(left, bottom), (right, bottom), ((left + right) / 2, top)], fill=255)
    return mask


def boxesOverlap(boxA: tuple[int, ...], boxB: tuple[int, ...], margin: int = 1) -> bool:
    return not (
        boxA[2] + margin <= boxB[0]
        or boxB[2] + margin <= boxA[0]
        or boxA[3] + margin <= boxB[1]
        or boxB[3] + margin <= boxA[1]
    )


def sampleShape(rng: np.random.Generator, config: SyntheticConfig, imageSize: int) -> tuple[int, int, int, int, int]:
    """Return classId, left, top, width, height for one candidate shape."""
    classId = int(rng.integers(0, len(CLASS_NAMES)))
    size = int(rng.integers(config.minObjectSize, config.maxObjectSize + 1))
    width = height = size
    if CLASS_NAMES[classId] == "triangle":
        height = max(config.minObjectSize, int(round(size * rng.uniform(0.75, 1.0))))
    left = int(rng.integers(0, imageSize - width + 1))
    top = int(rng.integers(0, imageSize - height + 1))
    return classId, left, top, width, height


def generateSample(
    rng: np.random.Generator, config: SyntheticConfig, imageSize: int, gridSize: int, stats: dict
) -> tuple[Image.Image, list[ShapeObject]]:
    """Render one image. Shapes never overlap; same-cell centers are rejected and resampled if configured."""
    background = rng.integers(0, 256, size=3)
    canvas = Image.new("RGB", (imageSize, imageSize), tuple(int(value) for value in background))
    requestedCount = int(rng.integers(config.minObjects, config.maxObjects + 1))

    placedObjects: list[ShapeObject] = []
    occupiedCells: set[tuple[int, int]] = set()
    for _ in range(requestedCount):
        for _ in range(config.maxPlacementAttempts):
            classId, left, top, width, height = sampleShape(rng, config, imageSize)
            mask = drawShapeMask(classId, left, top, width, height, imageSize)
            tightBox = mask.getbbox()  # exact pixel extent of the rasterized shape
            if tightBox is None:
                continue
            if any(boxesOverlap(tightBox, existing.boxPixels) for existing in placedObjects):
                stats["rejectedOverlap"] += 1
                continue
            shapeObject = ShapeObject(classId, tightBox)
            centerX, centerY = shapeObject.normalizedBox(imageSize)[:2]
            cell = responsibleCell(centerX, centerY, gridSize)
            if config.avoidCellCollisions and cell in occupiedCells:
                stats["rejectedCellCollision"] += 1
                continue
            canvas.paste(contrastingColor(rng, background, config.minColorDistance), mask=mask)
            placedObjects.append(shapeObject)
            occupiedCells.add(cell)
            break
        else:
            stats["objectShortfall"] += 1

    pixels = np.asarray(canvas, dtype=np.float32)
    if config.backgroundNoise > 0:
        pixels = pixels + rng.normal(0.0, config.backgroundNoise, size=pixels.shape)
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return image, placedObjects


def generateSplit(
    numImages: int, seed: int, split: str, config: SyntheticConfig, imageSize: int, gridSize: int
) -> tuple[list[Image.Image], list[list[ShapeObject]], dict]:
    rng = splitRng(seed, split)
    stats = {"rejectedOverlap": 0, "rejectedCellCollision": 0, "objectShortfall": 0}
    images, objectLists = [], []
    for _ in range(numImages):
        image, placedObjects = generateSample(rng, config, imageSize, gridSize, stats)
        images.append(image)
        objectLists.append(placedObjects)

    objectCounts = [len(objects) for objects in objectLists]
    stats.update(
        numImages=numImages,
        numObjects=sum(objectCounts),
        objectsPerImage={str(count): objectCounts.count(count) for count in sorted(set(objectCounts))},
        classCounts={
            name: sum(obj.classId == classId for objects in objectLists for obj in objects)
            for classId, name in enumerate(CLASS_NAMES)
        },
    )
    return images, objectLists, stats


def writeSplit(
    splitDir: Path, images: list[Image.Image], objectLists: list[list[ShapeObject]], stats: dict, imageSize: int
) -> None:
    imageDir = splitDir / "images"
    imageDir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (image, placedObjects) in enumerate(zip(images, objectLists)):
        fileName = f"{index:06d}.png"
        image.save(imageDir / fileName)
        records.append({
            "fileName": fileName,
            "objects": [
                {
                    "classId": obj.classId,
                    "className": CLASS_NAMES[obj.classId],
                    "box": obj.normalizedBox(imageSize),
                    "boxPixels": list(obj.boxPixels),
                }
                for obj in placedObjects
            ],
        })
    payload = {"classNames": list(CLASS_NAMES), "imageSize": imageSize, "boxFormat": "cxcywh-normalized",
               "stats": stats, "images": records}
    (splitDir / "annotations.json").write_text(json.dumps(payload, indent=1))


def generationManifest(config: SyntheticConfig, seed: int, imageSize: int, gridSize: int) -> dict:
    generationParams = asdict(config)
    generationParams.pop("root")
    return {"seed": seed, "imageSize": imageSize, "gridSize": gridSize, **generationParams}


def ensureSyntheticDataset(
    config: SyntheticConfig, seed: int, imageSize: int = 112, gridSize: int = 7, force: bool = False
) -> Path:
    """Generate train/val splits under config.root unless an identical generation already exists."""
    root = Path(config.root)
    manifestPath = root / "manifest.json"
    manifest = generationManifest(config, seed, imageSize, gridSize)
    if not force and manifestPath.exists() and json.loads(manifestPath.read_text()) == manifest:
        return root

    for split, numImages in (("train", config.numTrain), ("val", config.numVal)):
        splitDir = root / split
        if splitDir.exists():
            shutil.rmtree(splitDir)
        images, objectLists, stats = generateSplit(numImages, seed, split, config, imageSize, gridSize)
        writeSplit(splitDir, images, objectLists, stats, imageSize)
    manifestPath.write_text(json.dumps(manifest, indent=2))
    return root


def readSplitAnnotations(splitDir: Path) -> dict:
    return json.loads((Path(splitDir) / "annotations.json").read_text())


class SyntheticShapesDataset(Dataset):
    """Returns (image (3, H, W) float in [0, 1], target (S, S, 5 + C), annotations (K, 5))."""

    classNames = CLASS_NAMES

    def __init__(self, splitDir: str | Path, gridSize: int = 7):
        self.splitDir = Path(splitDir)
        self.gridSize = gridSize
        payload = readSplitAnnotations(self.splitDir)
        self.generationStats = payload["stats"]
        self.records = payload["images"]

        # The full split is tiny (2000 x 112 x 112 x 3 bytes), so images are decoded once
        self.images = torch.stack([self._loadImage(record["fileName"]) for record in self.records])
        self.annotations = [
            torch.tensor([[obj["classId"], *obj["box"]] for obj in record["objects"]], dtype=torch.float32).reshape(-1, 5)
            for record in self.records
        ]
        encoded = [encodeTarget(annotations, gridSize, len(CLASS_NAMES)) for annotations in self.annotations]
        self.targets = torch.stack([target for target, _ in encoded])
        self.droppedObjects = sum(dropped for _, dropped in encoded)

    def _loadImage(self, fileName: str) -> Tensor:
        pixels = np.asarray(Image.open(self.splitDir / "images" / fileName).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(pixels.copy()).permute(2, 0, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.images[index].float() / 255.0, self.targets[index], self.annotations[index]

    def collisionSummary(self) -> dict:
        return {
            "generationRejectedCellCollision": self.generationStats["rejectedCellCollision"],
            "generationObjectShortfall": self.generationStats["objectShortfall"],
            "encoderDroppedObjects": self.droppedObjects,
        }
