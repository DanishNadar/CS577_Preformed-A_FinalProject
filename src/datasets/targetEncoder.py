# Encode (K, 5) [classId, cx, cy, w, h] annotations into (S, S, 5 + C) YOLOv1 targets [x, y, w, h, objectness, one-hot class]

from __future__ import annotations

import torch
from torch import Tensor

TARGET_BOX_FIELDS = ("x", "y", "w", "h", "objectness")


def responsibleCell(centerX: float, centerY: float, gridSize: int) -> tuple[int, int]:
    # (row, column) of the grid cell containing a normalized box center
    column = min(int(centerX * gridSize), gridSize - 1)
    row = min(int(centerY * gridSize), gridSize - 1)
    return row, column


def encodeTarget(annotations: Tensor, gridSize: int, numClasses: int) -> tuple[Tensor, int]:
    # Larger objects win same-cell collisions (ties keep the earlier one); returns (target, droppedObjectCount)
    target = torch.zeros(gridSize, gridSize, 5 + numClasses)
    areas = annotations[:, 3] * annotations[:, 4]
    encodingOrder = sorted(range(len(annotations)), key=lambda index: (-float(areas[index]), index))

    droppedObjectCount = 0
    for objectIndex in encodingOrder:
        classId, centerX, centerY, width, height = annotations[objectIndex].tolist()
        row, column = responsibleCell(centerX, centerY, gridSize)
        if target[row, column, 4] == 1:
            droppedObjectCount += 1
            continue
        target[row, column, 0] = centerX * gridSize - column
        target[row, column, 1] = centerY * gridSize - row
        target[row, column, 2] = width
        target[row, column, 3] = height
        target[row, column, 4] = 1.0
        target[row, column, 5 + int(classId)] = 1.0
    return target, droppedObjectCount


def countCellCollisions(annotations: Tensor, gridSize: int) -> int:
    # Objects that would be dropped because another object already occupies their cell
    occupiedCells = {responsibleCell(float(cx), float(cy), gridSize) for cx, cy in annotations[:, 1:3]}
    return len(annotations) - len(occupiedCells)


def decodeTarget(target: Tensor) -> Tensor:
    # Recover (K, 5) [classId, centerX, centerY, width, height] annotations from one encoded target
    gridSize = target.shape[0]
    rows, columns = torch.nonzero(target[..., 4] > 0.5, as_tuple=True)
    cellValues = target[rows, columns]
    centerX = (cellValues[:, 0] + columns.to(target.dtype)) / gridSize
    centerY = (cellValues[:, 1] + rows.to(target.dtype)) / gridSize
    classIds = cellValues[:, 5:].argmax(dim=-1).to(target.dtype)
    return torch.stack((classIds, centerX, centerY, cellValues[:, 2], cellValues[:, 3]), dim=-1)
