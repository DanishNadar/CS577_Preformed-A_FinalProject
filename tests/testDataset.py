import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.datasets.loaders import DataConfig, buildDataLoaders
from src.datasets.syntheticShapes import (
    SyntheticConfig,
    SyntheticShapesDataset,
    drawShapeMask,
    ensureSyntheticDataset,
    generateSplit,
)
from src.datasets.targetEncoder import responsibleCell
from src.datasets.vocDataset import VocConfig, VocDetectionDataset, parseVocAnnotation


def testGenerationIsReproducible():
    config = SyntheticConfig(numTrain=20, numVal=5)
    firstImages, firstObjects, firstStats = generateSplit(20, 7, "train", config, 112, 7)
    secondImages, secondObjects, secondStats = generateSplit(20, 7, "train", config, 112, 7)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(firstImages, secondImages))
    assert firstObjects == secondObjects and firstStats == secondStats

    _, otherSplitObjects, _ = generateSplit(20, 7, "val", config, 112, 7)
    assert otherSplitObjects != firstObjects


def testDiskDatasetIsReproducible(tmp_path):
    config = SyntheticConfig(root=str(tmp_path / "a"), numTrain=12, numVal=4)
    otherConfig = SyntheticConfig(root=str(tmp_path / "b"), numTrain=12, numVal=4)
    first = SyntheticShapesDataset(ensureSyntheticDataset(config, seed=3) / "train")
    second = SyntheticShapesDataset(ensureSyntheticDataset(otherConfig, seed=3) / "train")
    assert torch.equal(first.images, second.images) and torch.equal(first.targets, second.targets)


def testBoxesAreValid():
    config = SyntheticConfig()
    images, objectLists, stats = generateSplit(150, 11, "train", config, 112, 7)
    assert all(1 <= len(objects) <= 3 for objects in objectLists)
    assert stats["numObjects"] == sum(len(objects) for objects in objectLists)
    for objects in objectLists:
        for obj in objects:
            left, top, right, bottom = obj.boxPixels
            assert 0 <= left < right <= 112 and 0 <= top < bottom <= 112
            assert 8 <= max(right - left, bottom - top) <= 80
            assert 0 <= obj.classId < 3
            centerX, centerY, width, height = obj.normalizedBox(112)
            assert 0 < width <= 1 and 0 < height <= 1
            assert width / 2 <= centerX <= 1 - width / 2 + 1e-9
            assert height / 2 <= centerY <= 1 - height / 2 + 1e-9


@pytest.mark.parametrize("classId", [0, 1, 2])
def testBoxMatchesRenderedShape(classId):
    mask = drawShapeMask(classId, left=20, top=30, width=40, height=34 if classId == 2 else 40, imageSize=112)
    rows, columns = np.nonzero(np.asarray(mask))
    assert mask.getbbox() == (columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)


def testNoCellCollisionsWhenAvoided():
    config = SyntheticConfig(minObjectSize=8, maxObjectSize=16, minObjects=3, maxObjects=3)
    _, objectLists, _ = generateSplit(200, 5, "train", config, 112, 7)
    for objects in objectLists:
        cells = [responsibleCell(*obj.normalizedBox(112)[:2], 7) for obj in objects]
        assert len(cells) == len(set(cells))


def testDatasetItemsAndLoader(tinySyntheticConfig):
    dataConfig = DataConfig(synthetic=tinySyntheticConfig)
    trainLoader, valLoader = buildDataLoaders(dataConfig, batchSize=8, seed=1)
    images, targets, annotations = next(iter(trainLoader))
    assert images.shape == (8, 3, 112, 112) and images.dtype == torch.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    assert targets.shape == (8, 7, 7, 8)
    assert len(annotations) == 8 and all(a.shape[1] == 5 and len(a) >= 1 for a in annotations)
    # Encoded objectness count matches the raw annotations when there are no collisions
    assert int(targets[..., 4].sum()) == sum(len(a) for a in annotations)
    assert len(trainLoader.dataset) == 48 and len(valLoader.dataset) == 16


def testRegeneratesWhenParametersChange(tinySyntheticConfig):
    root = ensureSyntheticDataset(tinySyntheticConfig, seed=1)
    assert len(SyntheticShapesDataset(root / "train")) == 48
    tinySyntheticConfig.numTrain = 10
    ensureSyntheticDataset(tinySyntheticConfig, seed=1)
    assert len(SyntheticShapesDataset(root / "train")) == 10
    assert len(list((root / "train" / "images").glob("*.png"))) == 10


def writeFakeVoc(root: Path) -> None:
    # Minimal VOC2007-style tree: three images, one with only an unselected class
    for folder in ("Annotations", "JPEGImages", "ImageSets/Main"):
        (root / folder).mkdir(parents=True)
    imageObjects = {
        "000001": [("car", 1, 1, 100, 50, 0), ("person", 120, 10, 180, 90, 0)],
        "000002": [("dog", 51, 26, 150, 125, 0), ("aeroplane", 2, 2, 40, 30, 1)],
        "000003": [("person", 10, 10, 50, 50, 0)],
    }
    for imageId, objects in imageObjects.items():
        Image.new("RGB", (200, 100), (128, 128, 128)).save(root / "JPEGImages" / f"{imageId}.jpg")
        objectXml = "".join(
            f"<object><name>{name}</name><difficult>{difficult}</difficult><bndbox><xmin>{x0}</xmin>"
            f"<ymin>{y0}</ymin><xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
            for name, x0, y0, x1, y1, difficult in objects
        )
        (root / "Annotations" / f"{imageId}.xml").write_text(
            f"<annotation><size><width>200</width><height>100</height><depth>3</depth></size>{objectXml}</annotation>"
        )
    (root / "ImageSets" / "Main" / "trainval.txt").write_text("\n".join(imageObjects) + "\n")


def testVocAnnotationParsing(tmp_path):
    writeFakeVoc(tmp_path)
    annotations = parseVocAnnotation(tmp_path / "Annotations" / "000001.xml", ["aeroplane", "car", "dog"], False)
    # car box 1..100 x 1..50 -> continuous edges 0..100 x 0..50 on a 200x100 image
    torch.testing.assert_close(annotations, torch.tensor([[1.0, 0.25, 0.25, 0.5, 0.5]]))
    withDifficult = parseVocAnnotation(tmp_path / "Annotations" / "000002.xml", ["aeroplane", "car", "dog"], True)
    assert withDifficult[:, 0].tolist() == [2.0, 0.0]


def testVocDatasetInterface(tmp_path):
    writeFakeVoc(tmp_path)
    config = VocConfig(root=str(tmp_path), valFraction=0.5)
    trainSet = VocDetectionDataset(config, "train", seed=0)
    valSet = VocDetectionDataset(config, "val", seed=0)
    # Image 000003 has no selected class and is filtered out
    assert sorted(trainSet.imageIds + valSet.imageIds) == ["000001", "000002"]
    assert VocDetectionDataset(config, "train", seed=0).imageIds == trainSet.imageIds
    image, target, annotations = trainSet[0]
    assert image.shape == (3, 112, 112) and target.shape == (7, 7, 8) and annotations.shape[1] == 5


def testVocMissingRootRaisesHelpfulError(tmp_path):
    with pytest.raises(FileNotFoundError, match="VOC"):
        VocDetectionDataset(VocConfig(root=str(tmp_path / "missing")), "train", seed=0)


def testAnnotationsJsonRecordsStats(tinySyntheticConfig):
    root = ensureSyntheticDataset(tinySyntheticConfig, seed=2)
    payload = json.loads((root / "train" / "annotations.json").read_text())
    for key in ("rejectedOverlap", "rejectedCellCollision", "objectShortfall", "numObjects", "classCounts"):
        assert key in payload["stats"]
