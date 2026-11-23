import pytest
import torch

from src.datasets.targetEncoder import countCellCollisions, decodeTarget, encodeTarget, responsibleCell


@pytest.mark.parametrize(
    "center, expectedCell",
    [((0.0, 0.0), (0, 0)), ((0.5, 0.5), (3, 3)), ((0.99, 0.01), (0, 6)), ((1.0, 1.0), (6, 6)), ((1 / 7, 2 / 7), (2, 1))],
)
def testResponsibleCell(center, expectedCell):
    assert responsibleCell(*center, gridSize=7) == expectedCell


def testEncodeSingleObject():
    # Center (0.3, 0.6) -> column floor(2.1) = 2, row floor(4.2) = 4
    annotations = torch.tensor([[2.0, 0.3, 0.6, 0.25, 0.4]])
    target, dropped = encodeTarget(annotations, gridSize=7, numClasses=3)
    assert dropped == 0 and target.shape == (7, 7, 8)
    torch.testing.assert_close(target[4, 2], torch.tensor([0.1, 0.2, 0.25, 0.4, 1.0, 0.0, 0.0, 1.0]))
    assert target[..., 4].sum() == 1 and target.abs().sum() == target[4, 2].abs().sum()


def testEncodeDecodeRoundTrip():
    annotations = torch.tensor([[0.0, 0.1, 0.1, 0.1, 0.1], [1.0, 0.55, 0.45, 0.3, 0.2], [2.0, 0.9, 0.8, 0.15, 0.3]])
    target, dropped = encodeTarget(annotations, 7, 3)
    decoded = decodeTarget(target)
    order = decoded[:, 0].argsort()
    assert dropped == 0
    torch.testing.assert_close(decoded[order], annotations)


def testCollisionKeepsLargestObjectDeterministically():
    # Both centers fall in cell (3, 3); the larger box (index 1) must win regardless of order
    small = [0.0, 0.45, 0.45, 0.1, 0.1]
    large = [1.0, 0.5, 0.5, 0.4, 0.4]
    for annotations in (torch.tensor([small, large]), torch.tensor([large, small])):
        target, dropped = encodeTarget(annotations, 7, 3)
        assert dropped == 1
        assert target[..., 4].sum() == 1
        assert target[3, 3, 6] == 1.0 and target[3, 3, 2] == pytest.approx(0.4)
    assert countCellCollisions(torch.tensor([small, large]), 7) == 1


def testCollisionTieKeepsEarlierAnnotation():
    first = [0.0, 0.44, 0.44, 0.2, 0.2]
    second = [2.0, 0.47, 0.47, 0.2, 0.2]
    target, dropped = encodeTarget(torch.tensor([first, second]), 7, 3)
    assert dropped == 1 and target[3, 3, 5] == 1.0


def testEmptyAnnotations():
    target, dropped = encodeTarget(torch.zeros(0, 5), 7, 3)
    assert dropped == 0 and target.abs().sum() == 0
    assert decodeTarget(target).shape == (0, 5)
