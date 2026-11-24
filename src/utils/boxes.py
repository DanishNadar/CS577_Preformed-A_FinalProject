# Box conversions, IoU and NMS; cxcywh and xyxy boxes are normalized to [0, 1] by image size

from __future__ import annotations

import torch
from torch import Tensor


def cxcywhToXyxy(boxes: Tensor) -> Tensor:
    centerX, centerY, width, height = boxes.unbind(-1)
    return torch.stack(
        (centerX - width / 2, centerY - height / 2, centerX + width / 2, centerY + height / 2), dim=-1
    )


def xyxyToCxcywh(boxes: Tensor) -> Tensor:
    left, top, right, bottom = boxes.unbind(-1)
    return torch.stack(((left + right) / 2, (top + bottom) / 2, right - left, bottom - top), dim=-1)


def boxIou(boxesA: Tensor, boxesB: Tensor, eps: float = 1e-9) -> Tensor:
    # Elementwise IoU of broadcastable xyxy boxes with shape (..., 4)
    left = torch.maximum(boxesA[..., 0], boxesB[..., 0])
    top = torch.maximum(boxesA[..., 1], boxesB[..., 1])
    right = torch.minimum(boxesA[..., 2], boxesB[..., 2])
    bottom = torch.minimum(boxesA[..., 3], boxesB[..., 3])
    intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)

    # Clamping keeps degenerate predictions (negative w/h from a linear head) at zero area
    areaA = (boxesA[..., 2] - boxesA[..., 0]).clamp(min=0) * (boxesA[..., 3] - boxesA[..., 1]).clamp(min=0)
    areaB = (boxesB[..., 2] - boxesB[..., 0]).clamp(min=0) * (boxesB[..., 3] - boxesB[..., 1]).clamp(min=0)
    union = areaA + areaB - intersection
    return intersection / union.clamp(min=eps)


def pairwiseIou(boxesA: Tensor, boxesB: Tensor) -> Tensor:
    # IoU matrix (M, K) between xyxy boxes (M, 4) and (K, 4)
    return boxIou(boxesA[:, None, :], boxesB[None, :, :])


def cellToImageBoxes(cellBoxes: Tensor, gridSize: int) -> Tensor:
    # Cell-relative boxes (N, S, S, ..., 4) to image-relative cxcywh; dim 1 is the grid row, dim 2 the column
    extraDims = cellBoxes.dim() - 4
    gridIndex = torch.arange(gridSize, device=cellBoxes.device, dtype=cellBoxes.dtype)
    columnOffset = gridIndex.view(1, 1, gridSize, *([1] * extraDims))
    rowOffset = gridIndex.view(1, gridSize, 1, *([1] * extraDims))
    centerX = (cellBoxes[..., 0] + columnOffset) / gridSize
    centerY = (cellBoxes[..., 1] + rowOffset) / gridSize
    return torch.stack((centerX, centerY, cellBoxes[..., 2], cellBoxes[..., 3]), dim=-1)


def nonMaxSuppression(boxes: Tensor, scores: Tensor, iouThreshold: float) -> Tensor:
    # Greedy NMS on xyxy boxes (K, 4). Returns kept indices sorted by descending score
    order = scores.argsort(descending=True)
    keptIndices = []
    while order.numel() > 0:
        bestIndex = order[0]
        keptIndices.append(bestIndex)
        if order.numel() == 1:
            break
        overlaps = pairwiseIou(boxes[bestIndex].unsqueeze(0), boxes[order[1:]])[0]
        order = order[1:][overlaps <= iouThreshold]
    return torch.stack(keptIndices) if keptIndices else torch.empty(0, dtype=torch.long, device=boxes.device)
