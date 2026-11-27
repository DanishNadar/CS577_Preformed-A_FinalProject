# Per-epoch ablation metrics: responsible-box IoU (overall and by object size), recall/precision after NMS,
# the paper loss as a common yardstick, per-term gradient norms on a fixed probe batch, and training-step grad norms

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.optim.optimizer import register_optimizer_step_pre_hook
from torch.utils.data import DataLoader

from src.losses.yoloLoss import YoloLoss, YoloLossConfig, assignResponsibleBoxes
from src.training.inference import collectPredictions
from src.utils.boxes import boxIou, cxcywhToXyxy
from src.utils.decoding import predictionsToDetections, splitPredictions

# Object size buckets by sqrt(area) in pixels on the 112 px image (synthetic objects are 8-80 px)
SIZE_BUCKETS = (("Small", 0.0, 24.0), ("Medium", 24.0, 48.0), ("Large", 48.0, math.inf))
# Loss terms whose parameter-gradient norms are tracked separately
GRADIENT_TERMS = ("coordinateXy", "coordinateWh", "objectConfidence", "noObjectConfidence", "classification")


def sizeBucketMasks(targetBoxes: Tensor, objectMask: Tensor, imageSize: int) -> dict[str, Tensor]:
    # (N, S, S) boolean masks of occupied cells, split by ground-truth sqrt(w * h) in pixels
    sidePixels = (targetBoxes[..., 2] * targetBoxes[..., 3]).clamp(min=0).sqrt() * imageSize
    occupied = objectMask > 0.5
    return {name: occupied & (sidePixels >= low) & (sidePixels < high) for name, low, high in SIZE_BUCKETS}


@torch.no_grad()
def responsibleBoxMetrics(predictions: Tensor, targets: Tensor, numBoxes: int, imageSize: int) -> dict[str, float]:
    # Proposal's main metric: mean IoU of the responsible (highest-IoU) predictor, plus per-size breakdowns
    predictedBoxes, _, predictedClasses = splitPredictions(predictions, numBoxes)
    targetBoxes, objectMask, targetClasses = targets[..., :4], targets[..., 4], targets[..., 5:]
    responsibleMask, boxIous = assignResponsibleBoxes(predictedBoxes, targetBoxes, objectMask)
    occupied = objectMask > 0.5
    if not occupied.any():
        return {}

    responsibleIou = (boxIous * responsibleMask).sum(dim=-1)  # (N, S, S): IoU of the responsible predictor
    responsibleIndex = responsibleMask.argmax(dim=-1, keepdim=True)
    responsibleBox = predictedBoxes.gather(-2, responsibleIndex.unsqueeze(-1).expand(*responsibleIndex.shape, 4))
    responsibleBox = responsibleBox.squeeze(-2)
    # Relative size error |w_hat - w| / w averaged over width and height
    relativeSizeError = ((responsibleBox[..., 2:4] - targetBoxes[..., 2:4]).abs()
                         / targetBoxes[..., 2:4].clamp(min=1e-6)).mean(dim=-1)
    classCorrect = (predictedClasses.argmax(dim=-1) == targetClasses.argmax(dim=-1)).float()

    metrics = {
        "valMeanIou": responsibleIou[occupied].mean().item(),
        "valIouAt50": (responsibleIou[occupied] >= 0.5).float().mean().item(),
        "valRelativeSizeError": relativeSizeError[occupied].mean().item(),
        "valClassAccuracy": classCorrect[occupied].mean().item(),
    }
    for name, mask in sizeBucketMasks(targetBoxes, objectMask, imageSize).items():
        count = int(mask.sum())
        metrics[f"valMeanIou{name}"] = responsibleIou[mask].mean().item() if count else float("nan")
        metrics[f"valRelativeSizeError{name}"] = relativeSizeError[mask].mean().item() if count else float("nan")
        metrics[f"valCount{name}"] = count
    return metrics


@torch.no_grad()
def detectionMetrics(
    predictions: Tensor,
    annotations: list[Tensor],
    numBoxes: int,
    scoreThreshold: float = 0.2,
    iouThreshold: float = 0.5,
) -> dict[str, float]:
    # Class-matched recall and precision at IoU >= 0.5 after NMS, greedy matching by descending score
    detections = predictionsToDetections(predictions, numBoxes, scoreThreshold)
    truePositives, numDetections, numObjects = 0, 0, 0
    for imageDetections, imageAnnotations in zip(detections, annotations):
        numDetections += imageDetections.shape[0]
        numObjects += imageAnnotations.shape[0]
        if imageDetections.shape[0] == 0 or imageAnnotations.shape[0] == 0:
            continue
        truthCorners = cxcywhToXyxy(imageAnnotations[:, 1:5])
        truthClasses = imageAnnotations[:, 0].long()
        matched = torch.zeros(imageAnnotations.shape[0], dtype=torch.bool)
        for detection in imageDetections:
            ious = boxIou(detection[:4].unsqueeze(0), truthCorners)
            ious[(truthClasses != int(detection[5])) | matched] = -1.0
            bestIou, bestIndex = ious.max(dim=0)
            if bestIou >= iouThreshold:
                matched[bestIndex] = True
                truePositives += 1
    return {
        "valRecallAt50": truePositives / max(numObjects, 1),
        "valPrecisionAt50": truePositives / max(numDetections, 1),
    }


def termGradientNorms(model: nn.Module, images: Tensor, targets: Tensor, lossFn: nn.Module) -> dict[str, float]:
    # L2 norm of d(term)/d(parameters) for each weighted loss term on a fixed batch, in eval mode (no dropout)
    wasTraining = model.training
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        with torch.enable_grad():
            breakdown = lossFn(model(images), targets)
            norms = {}
            for index, term in enumerate(GRADIENT_TERMS):
                gradients = torch.autograd.grad(
                    breakdown[term], parameters, retain_graph=index < len(GRADIENT_TERMS) - 1, allow_unused=True
                )
                squaredNorm = sum(gradient.pow(2).sum() for gradient in gradients if gradient is not None)
                norms[f"probeGradNorm{term[0].upper()}{term[1:]}"] = float(torch.as_tensor(squaredNorm).sqrt())
    finally:
        model.train(wasTraining)
    norms["probeNoobjToObjGradRatio"] = (
        norms["probeGradNormNoObjectConfidence"] / max(norms["probeGradNormObjectConfidence"], 1e-12)
    )
    return norms


class TrainingGradientTracker:
    # Records the global gradient norm at every optimizer step without modifying the training loop

    def __init__(self):
        self.stepNorms: list[float] = []
        self.handle = None

    def __enter__(self) -> "TrainingGradientTracker":
        self.handle = register_optimizer_step_pre_hook(self.recordStep)
        return self

    def __exit__(self, *exc) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def recordStep(self, optimizer: torch.optim.Optimizer, args, kwargs) -> None:
        squaredNorm = 0.0
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    squaredNorm += parameter.grad.detach().pow(2).sum().item()
        self.stepNorms.append(math.sqrt(squaredNorm))

    def popEpochStats(self) -> dict[str, float]:
        norms, self.stepNorms = self.stepNorms, []
        finite = [norm for norm in norms if math.isfinite(norm)]
        if not finite:
            return {"trainGradNormMean": float("nan"), "trainGradNormMax": float("nan"),
                    "trainGradNormStd": float("nan"), "trainNonFiniteSteps": len(norms)}
        mean = sum(finite) / len(finite)
        variance = sum((norm - mean) ** 2 for norm in finite) / len(finite)
        return {
            "trainGradNormMean": mean,
            "trainGradNormMax": max(finite),
            "trainGradNormStd": math.sqrt(variance),
            "trainNonFiniteSteps": len(norms) - len(finite),
        }


def fixedProbeBatch(valLoader: DataLoader, size: int = 64) -> tuple[Tensor, Tensor]:
    # The first `size` validation images (the val loader is not shuffled, so this is identical across runs)
    images, targets, seen = [], [], 0
    for batchImages, batchTargets, _ in valLoader:
        images.append(batchImages)
        targets.append(batchTargets)
        seen += batchImages.shape[0]
        if seen >= size:
            break
    return torch.cat(images)[:size], torch.cat(targets)[:size]


def makeAblationMetricsFn(
    lossConfig: YoloLossConfig,
    numBoxes: int,
    imageSize: int,
    gradientTracker: TrainingGradientTracker | None = None,
    probeSize: int = 64,
):
    # Builds the extraMetricsFn passed to train(); the variant's own loss drives the probe gradients,
    # while the paper loss is evaluated on val for every variant so loss curves are comparable
    variantLoss = YoloLoss(lossConfig, numBoxes)
    paperLoss = YoloLoss(YoloLossConfig(), numBoxes)
    probe: dict[str, Tensor] = {}

    def extraMetrics(model: nn.Module, valLoader: DataLoader, device: torch.device) -> dict:
        metrics = dict(gradientTracker.popEpochStats()) if gradientTracker is not None else {}
        outputs = collectPredictions(model, valLoader, device)
        with torch.no_grad():
            paperBreakdown = paperLoss(outputs.predictions, outputs.targets)
        metrics.update({f"valPaper{name[0].upper()}{name[1:]}": value.item() for name, value in paperBreakdown.items()})
        metrics.update(responsibleBoxMetrics(outputs.predictions, outputs.targets, numBoxes, imageSize))
        metrics.update(detectionMetrics(outputs.predictions, outputs.annotations, numBoxes))

        if not probe:
            probe["images"], probe["targets"] = fixedProbeBatch(valLoader, probeSize)
        metrics.update(termGradientNorms(
            model, probe["images"].to(device), probe["targets"].to(device), variantLoss.to(device)
        ))
        return metrics

    return extraMetrics
