"""YOLOv1 target encoding.

Annotations (dataset side) are tensors of shape (K, 5): [classId, centerX, centerY, width, height],
with every coordinate normalized to [0, 1] by the image size.

Encoded target tensor, shape (S, S, 5 + C):

    index 0  x: box center offset inside the responsible cell, centerX * S - column
    index 1  y: box center offset inside the responsible cell, centerY * S - row
    index 2  w: box width relative to the image
    index 3  h: box height relative to the image
    index 4  objectness: 1 if an object is assigned to this cell, else 0
    index 5+ one-hot class vector for the cell (YOLOv1 keeps one class vector per cell)

The responsible cell is the one containing the box center: column = floor(centerX * S),
row = floor(centerY * S), clamped to S - 1 so a center exactly at 1.0 stays on the grid.
The target stores one ground-truth box per cell; the loss decides which of the B predictors
in that cell is responsible for it (highest IoU), so the target does not depend on B.

Collision policy: when several objects share a cell, the object with the largest area is
encoded and the rest are dropped from the target (ties keep the earlier annotation). Dropped
objects stay in the raw annotation list, and the count is returned so datasets can report it.
"""

from __future__ import annotations

import torch
from torch import Tensor

TARGET_BOX_FIELDS = ("x", "y", "w", "h", "objectness")


def responsibleCell(centerX: float, centerY: float, gridSize: int) -> tuple[int, int]:
    """Return (row, column) of the grid cell containing a normalized box center."""
    column = min(int(centerX * gridSize), gridSize - 1)
    row = min(int(centerY * gridSize), gridSize - 1)
    return row, column


def encodeTarget(annotations: Tensor, gridSize: int, numClasses: int) -> tuple[Tensor, int]:
    """Encode (K, 5) annotations into a (S, S, 5 + C) target. Returns (target, droppedObjectCount)."""
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
    """Number of objects that would be dropped because another object already occupies their cell."""
    occupiedCells = {responsibleCell(float(cx), float(cy), gridSize) for cx, cy in annotations[:, 1:3]}
    return len(annotations) - len(occupiedCells)


def decodeTarget(target: Tensor) -> Tensor:
    """Recover (K, 5) annotations [classId, centerX, centerY, width, height] from one encoded target."""
    gridSize = target.shape[0]
    rows, columns = torch.nonzero(target[..., 4] > 0.5, as_tuple=True)
    cellValues = target[rows, columns]
    centerX = (cellValues[:, 0] + columns.to(target.dtype)) / gridSize
    centerY = (cellValues[:, 1] + rows.to(target.dtype)) / gridSize
    classIds = cellValues[:, 5:].argmax(dim=-1).to(target.dtype)
    return torch.stack((classIds, centerX, centerY, cellValues[:, 2], cellValues[:, 3]), dim=-1)
