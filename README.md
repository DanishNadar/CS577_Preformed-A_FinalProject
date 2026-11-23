# CS 577 — YOLOv1 Loss Design Study

Team Pre-formed-A (Danish Nadar, Omar Ashurbayov, Abdul Mueed Mohammed, Muhammad Ahmad Hasan), CS 577 Deep Learning, Illinois Institute of Technology, Fall 2026.

## Objective

Redmon et al. argue that the YOLOv1 loss trains stably and localizes well because of two choices:

- the weights `lambdaCoord = 5` and `lambdaNoobj = 0.5`, so the many empty cells do not swamp the gradient from cells that contain objects
- regressing `sqrt(w)` and `sqrt(h)`, so a pixel error on a large box costs less than the same error on a small box

We test that claim with a small PyTorch YOLOv1-style detector, a reference implementation of the paper's loss, and controlled ablations of those two choices.

> Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. https://arxiv.org/abs/1506.02640

This repository currently holds the baseline (model, data ETL, reference loss, training) and the interfaces the ablation, evaluation and verification work builds on.

## Installation

Python 3.10+ is required. Run every command from the repository root.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA wheel for your GPU
pip install -r requirements.txt
```

Dependencies are limited to `torch`, `numpy`, `pillow`, `pyyaml` and `pytest`. torchvision is not needed.

## Quick commands

| Task | Command |
|---|---|
| Unit tests (~15 s) | `python -m pytest` |
| End-to-end smoke test (~5 s, CPU, temporary folder) | `python scripts/smokeTest.py` |
| Generate the synthetic dataset | `python scripts/generateDataset.py` |
| Train the full baseline | `python scripts/trainBaseline.py --config configs/baseline.yaml` |
| Same, as a module | `python -m src.training.train --config configs/baseline.yaml` |
| Override any config value | `... --set training.epochs=5 --set device=cuda --set loss.lambdaNoobj=1.0` |
| Resume | `python scripts/trainBaseline.py --resume checkpoints/baseline/last.pt` |
| Train on VOC | `python scripts/trainBaseline.py --set data.dataset=voc` |

`--set` takes dotted keys from [configs/baseline.yaml](configs/baseline.yaml). Values are parsed as YAML, so `true`, `null`, `1e-4` and `[8,16,32,64]` all work. Unknown keys are rejected.

Training creates the synthetic dataset on first use. It is regenerated only when the generation parameters or seed change, which is detected through `data/synthetic/manifest.json`.

## Repository layout

```text
configs/baseline.yaml          all experiment settings
src/models/yoloModel.py        ModelConfig, YoloModel, createModel
src/datasets/syntheticShapes.py   synthetic generator + SyntheticShapesDataset
src/datasets/vocDataset.py     PASCAL VOC 2007 three-class adapter
src/datasets/targetEncoder.py  annotation -> YOLO target encoding, collision policy
src/datasets/loaders.py        DataConfig, dataset selection, DataLoaders, collate
src/losses/yoloLoss.py         reference YOLOv1 loss, one function per term
src/training/train.py          training loop, CLI entry point
src/training/checkpoint.py     save / load / rebuild model from checkpoint
src/training/history.py        history.json / history.csv
src/training/inference.py      batch inference helpers
src/utils/boxes.py             box conversions, IoU, NMS
src/utils/decoding.py          raw output -> boxes, confidences, class scores, detections
src/utils/config.py            YAML -> dataclasses, overrides
src/utils/reproducibility.py   seeding, device selection, RNG state capture
scripts/                       generateDataset.py, trainBaseline.py, smokeTest.py
tests/                         pytest suite
data/ checkpoints/ results/    generated artifacts (git-ignored)
```

## Architecture

```text
input            (N, 3, 112, 112)
conv block 1     Conv3x3(3->16)   + LeakyReLU(0.1) + MaxPool2 -> (N, 16, 56, 56)
conv block 2     Conv3x3(16->32)  + LeakyReLU(0.1) + MaxPool2 -> (N, 32, 28, 28)
conv block 3     Conv3x3(32->64)  + LeakyReLU(0.1) + MaxPool2 -> (N, 64, 14, 14)
conv block 4     Conv3x3(64->128) + LeakyReLU(0.1) + MaxPool2 -> (N, 128, 7, 7)
flatten                                                       -> (N, 6272)
fc1              Linear(6272->512) + LeakyReLU(0.1) + Dropout(0.5)
fc2              Linear(512->637)                             -> reshape (N, 7, 7, 13)
output activation (default "sigmoid")
```

The model has 3.64M parameters. Channel widths, hidden size, dropout, the number of boxes and the number of classes are all set in `model:` in the config. The constructor checks that `imageSize / 2^len(convChannels) == gridSize`.

### Output tensor `(N, S, S, B*5 + C)` = `(N, 7, 7, 13)`

| Index | Meaning |
|---|---|
| 0–4 | box 0: `x, y, w, h, confidence` |
| 5–9 | box 1: `x, y, w, h, confidence` |
| 10–12 | class scores for the cell: circle, square, triangle (shared by both boxes) |

- Dimension 1 is the grid **row** (y) and dimension 2 is the grid **column** (x).
- `x, y` are the box center's offset inside its cell, in [0, 1] (classic YOLOv1).
- `w, h` are relative to the full image.
- There is one class vector per cell, not one per box, as in YOLOv1.
- There are no anchors. This is not YOLOv2+, v5 or v8.

## Data and ETL

### Synthetic shapes (default development dataset)

`src/datasets/syntheticShapes.py`, with settings under `data.synthetic` in the config.

- **Extract:** `generateSplit` renders 112×112 RGB images with 1–3 objects, each a circle, square or triangle. Shapes are 8–80 px (triangles are 75–100% as tall as they are wide). Positions, sizes, counts and colors are random. Each shape gets a random background-contrasting color, and Gaussian pixel noise (σ = 6) is added afterwards.
- **Exact boxes:** each shape is drawn on its own mask, and the box is the mask's tight pixel extent (`mask.getbbox()`). The box is therefore exactly what was rendered and always lies inside the image.
- **Placement rules:** shapes may not overlap, with a 1 px margin. If `avoidCellCollisions` is true, a shape whose center falls in an already occupied grid cell is rejected and resampled. A shape that still has not been placed after `maxPlacementAttempts` tries is skipped and counted as `objectShortfall`. The first shape always fits, so no image is empty.
- **Reproducibility:** each split uses `numpy.random.default_rng([seed, splitId])`, so train and val are independent and both are fixed by `seed` (default 577).
- **Load:** files are written to `data/synthetic/{train,val}/images/*.png` and `annotations.json`. The JSON holds per-object `classId`, `className`, normalized `box` and `boxPixels`, plus split statistics. `SyntheticShapesDataset` decodes all PNGs into memory once, about 75 MB for the train split.
- **Default size:** 2,000 train and 500 val images, as in the proposal.

Statistics recorded per split in `annotations.json`: `rejectedOverlap`, `rejectedCellCollision`, `objectShortfall`, `numObjects`, `objectsPerImage`, `classCounts`.

Full-split statistics for the default config (seed 577):

| Split | Images | Objects | 1 / 2 / 3 objects per image | circle / square / triangle | Rejected (overlap / same cell) | Shortfall |
|---|---|---|---|---|---|---|
| train | 2000 | 3997 | 678 / 647 / 675 | 1332 / 1255 / 1410 | 10050 / 0 | 2 |
| val | 500 | 993 | 173 / 161 / 166 | 334 / 327 / 332 | 2629 / 1 | 0 |

### Dataset item interface (synthetic and VOC)

`dataset[i]` returns `(image, target, annotations)`:

| Field | Shape | Content |
|---|---|---|
| `image` | `(3, 112, 112)` float32 | RGB in [0, 1] |
| `target` | `(7, 7, 5 + C)` | encoded YOLO target (below) |
| `annotations` | `(K, 5)` | every object: `[classId, centerX, centerY, width, height]`, normalized to [0, 1] |

`collateDetections` stacks images and targets and keeps `annotations` as a list, since `K` varies.

### YOLO target encoding

`src/datasets/targetEncoder.py`. The target tensor is `(S, S, 5 + C)`:

| Index | Meaning |
|---|---|
| 0 | `x = centerX * S - column` (offset inside the responsible cell) |
| 1 | `y = centerY * S - row` |
| 2 | `w` relative to the image |
| 3 | `h` relative to the image |
| 4 | objectness, 1 if an object is assigned to the cell |
| 5.. | one-hot class vector for the cell |

- **Responsible cell:** `column = floor(centerX * S)` and `row = floor(centerY * S)`, clamped to `S - 1`.
- **Responsible box:** the target stores one ground-truth box per cell and does not depend on `B`. The loss picks the responsible predictor at training time, using the highest IoU.
- **Collision policy:** if several objects share a cell, the one with the largest area is encoded and ties go to the earlier annotation. The others are dropped from the target but stay in `annotations`, and `encodeTarget` returns the dropped count. `dataset.collisionSummary()` reports these counts, and training prints them at startup. On the synthetic data, collisions are prevented during generation, so the encoder drops 0 objects.

### PASCAL VOC 2007 adapter (optional)

`src/datasets/vocDataset.py`, with settings under `data.voc`.

- **Classes:** `aeroplane`, `car`, `dog` by default (`data.voc.classes`). These are visually distinct with moderate instance counts. Objects of other VOC classes are ignored and treated as background.
- **Split:** the `trainval` image set is parsed, only images with at least one selected-class object are kept, and the result is shuffled with `seed` and split by `valFraction` (default 0.2). `maxImages` caps the subset size.
- **Boxes:** VOC's 1-based inclusive pixel boxes are converted to continuous edges, clipped, and normalized by the original width and height. Images are stretched to 112×112, so the normalized boxes are unchanged. Difficult objects are excluded by default.
- **Interface:** items and targets are the same as for the synthetic dataset. Same-cell collisions do happen on VOC and are reported through `collisionSummary()`.

Nothing is downloaded automatically. To enable VOC:

1. Download `VOCtrainval_06-Nov-2007.tar` (about 450 MB) from the [PASCAL VOC 2007 page](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) or a mirror.
2. Extract it into `data/` so that `data/VOCdevkit/VOC2007/Annotations`, `JPEGImages` and `ImageSets/Main/trainval.txt` exist.
3. Train with `python scripts/trainBaseline.py --set data.dataset=voc --set outputDir=results/voc --set checkpointDir=checkpoints/voc`.

The VOC adapter's parsing, filtering, splitting and item interface are unit-tested against a small fake VOC tree. It has **not** yet been run on the real dataset.

### Which dataset for the final results — team decision needed

The proposal describes both a synthetic shapes task and a three-class VOC 2007 subset. Both are implemented, and `data.dataset` selects between them. **The team still has to agree which one the final reported experiments use.** Synthetic is the default because it is fast on CPU, has exact labels and no same-cell collisions, and is what the proposal's Section 3 specifies. Results from the two datasets are **not directly comparable**: VOC is much harder, has label noise and collisions, and uses different classes. Report them separately and do not mix them in one table.

## Loss (reference implementation)

`src/losses/yoloLoss.py` implements eq. 3 of the paper:

```text
L = λcoord Σ 1obj_ij [(x−x̂)² + (y−ŷ)²]
  + λcoord Σ 1obj_ij [(√w−√ŵ)² + (√h−√ĥ)²]
  +        Σ 1obj_ij (C−Ĉ)²
  + λnoobj Σ 1noobj_ij (C−Ĉ)²
  +        Σ 1obj_i Σ_c (p(c)−p̂(c))²
```

- **Responsibility (`assignResponsibleBoxes`):** in each cell that contains an object, the predictor with the highest IoU against the ground truth is responsible. The function returns `responsibleMask (N,S,S,B)` and `boxIous (N,S,S,B)`. Ties, such as both IoUs being 0 early in training, go to predictor 0.
- **Confidence target:** IoU(prediction, truth), as in the paper. `confidenceTarget: one` uses a target of 1 instead.
- **No-object term:** this applies to every predictor that is not responsible. That includes the second predictor in an occupied cell, because the paper's `1noobj_ij` is the complement of `1obj_ij`.
- **Where `no_grad` / detach is used:** only in `assignResponsibleBoxes`. Argmax is not differentiable, and the paper treats the IoU target as a constant. Every other term gets full autograd gradients.
- **Numerical safeguards:**
  - `safeSqrt(v) = sign(v)·sqrt(|v| + ε)` with ε = 1e-6 gives finite values and gradients for zero or negative widths, which a linear head can produce.
  - IoU clamps negative extents and divides by `max(union, 1e-9)`.
- **Normalization:** each term is summed over cells and predictors, then divided by the batch size.
- **Return value:** a dict of scalar tensors: `coordinate`, `coordinateXy`, `coordinateWh`, `objectConfidence`, `noObjectConfidence`, `classification` and `total`. Each value already includes its λ, so the components add up to `total`.

## Training

`src/training/train.py`:

- **Optimizer:** Adam, with `lr = 1e-3` by default. Weight decay and gradient clipping are optional.
- **Each epoch:** a training pass, then a validation pass without gradients.
- **Seeding:** Python, NumPy and torch are all seeded, and the shuffle order comes from a seeded `torch.Generator`.
- **Device:** `device: auto` uses CUDA if it is available and CPU otherwise. `device: cuda` falls back to CPU with a warning.

Run outputs:

| Path | Content |
|---|---|
| `checkpoints/<run>/last.pt` | written every epoch |
| `checkpoints/<run>/best.pt` | lowest validation total loss so far |
| `results/<run>/config.json` | resolved config |
| `results/<run>/history.json`, `history.csv` | one row per epoch |

Checkpoints contain: `modelState`, `optimizerState`, `epoch`, `seed`, `config` (the full resolved config), `bestValLoss`, `trainingHistory`, `rngState` and `shuffleGeneratorState`. Resuming restores all of these. A test checks that 1 epoch + resume + 1 epoch gives the same weights as 2 uninterrupted epochs.

History row schema: `epoch`, `learningRate`, `epochSeconds`, then `train<Component>` and `val<Component>` for every loss component (for example `trainTotal`, `valCoordinateWh`, `valNoObjectConfidence`). Columns added through `extraMetricsFn` are appended automatically.

## Implementation decisions and deviations from YOLOv1

| Paper | Here | Why |
|---|---|---|
| 448×448 input, 24 conv + 2 FC layers, ImageNet pretraining | 112×112 input, 4 conv blocks + 2 FC layers, from scratch | CPU budget (proposal Section 5) |
| 20 VOC classes | 3 classes | proposal |
| Linear final activation | default `sigmoid` on `x, y, w, h, conf` and `softmax` on class scores; `outputActivation: linear` restores the paper's behavior | keeps `w, h` positive so `sqrt` is well defined, and stabilizes small-data training. The loss is unchanged, so ablations still compare the loss terms on equal footing. |
| Network predicts `sqrt(w)` directly | network predicts `w`, and the loss takes `sqrt` | the output means the same thing whichever ablation is running, so decoding and evaluation never depend on the loss config |
| Sum over the batch | sum ÷ batch size | learning rate does not depend on batch size |
| SGD with momentum, LR warm-up schedule, data augmentation | Adam with constant LR, no augmentation | simple, reproducible baseline |
| Dropout 0.5 after the first FC layer | same | — |

## Known limitations

- The synthetic task is far easier than VOC. Conclusions about the loss on shapes may not carry over to natural images.
- The no-overlap rule rejects large shapes more often when they are placed second or third. As a result, images with several objects tend to contain smaller ones.
- Collision avoidance hides YOLOv1's same-cell weakness on synthetic data. On VOC the encoder keeps the largest object per cell and drops the rest.
- Detection mAP is not implemented here (Abdul's area). `predictionsToDetections` provides thresholded, NMS-filtered boxes to build on.
- The VOC adapter is tested on a fake directory tree, not yet on the real download.
- The GPU code path is supported, but only CPU was available during development, so the CUDA test was skipped.

## Baseline results

These numbers come from one full run: default config, seed 577, synthetic data with 2,000 train and 500 val images, 30 epochs, CPU only (no GPU). Command:

```bash
python scripts/trainBaseline.py --config configs/baseline.yaml --set device=cpu
```

| Epoch | Train total | Val total | Val coord | Val obj | Val noobj | Val cls |
|---|---|---|---|---|---|---|
| 1 | 4.906 | 3.624 | 1.702 | 0.458 | 0.091 | 1.372 |
| 10 | 3.390 | 3.516 | 1.619 | 0.490 | 0.060 | 1.346 |
| 20 | 2.220 | 2.965 | 1.495 | 0.418 | 0.095 | 0.957 |
| 30 | 1.006 | 2.193 | 1.000 | 0.358 | 0.074 | 0.761 |

Wall clock was about 16 s per epoch and 491 s in total. The best checkpoint is epoch 30.

Observations:

- **Plateau:** the loss barely moves from epoch 2 to epoch 16, then drops quickly.
- **Not converged:** validation loss was still falling at epoch 30.
- **Overfitting:** the gap between train and validation loss widens after epoch 20.
- **Spike:** validation `noObjectConfidence` jumped once, at epoch 7.

Ablations should therefore compare full training curves, not only final values. It is also worth testing more epochs or `dropout`/`weightDecay` settings, but those settings must be the same for every loss variant.

The following were measured on the val split with the epoch-30 checkpoint, using an ad-hoc script built only on the interfaces in the Team Integration Guide. They are not part of the repository's evaluation code, which is Abdul's task.

| Metric | Value |
|---|---|
| Mean IoU of the responsible box | 0.576 |
| Responsible boxes with IoU ≥ 0.5 | 64.6% |
| Per-cell class accuracy (occupied cells) | 71.2% |
| Class-matched recall@0.5 after NMS (score ≥ 0.2) | 36.5% |

## Team Integration Guide

You should be able to build on the baseline by importing these interfaces, without editing them. If a change does turn out to be needed, discuss it first, because the baseline checkpoints and tests depend on it.

### Omar — loss ablations

- **Configure the loss through config only.** Change `loss.lambdaCoord`, `loss.lambdaNoobj` or `loss.useSqrtWidthHeight`, either in a copied YAML file or with `--set`:
  ```bash
  python scripts/trainBaseline.py --set loss.lambdaCoord=1 --set loss.lambdaNoobj=1 \
      --set outputDir=results/unitWeights --set checkpointDir=checkpoints/unitWeights
  python scripts/trainBaseline.py --set loss.useSqrtWidthHeight=false \
      --set outputDir=results/directWh --set checkpointDir=checkpoints/directWh
  ```
- **From Python:**
  ```python
  from src.utils.config import loadExperimentConfig
  from src.training.train import train
  config = loadExperimentConfig("configs/baseline.yaml", {"loss.lambdaNoobj": 1.0, "outputDir": "results/x", "checkpointDir": "checkpoints/x"})
  result = train(config)                       # result.trainingHistory, result.bestCheckpointPath
  ```
- **Custom loss:** pass any `nn.Module` with the signature `(predictions, targets) -> {"total": ..., <components>}` as `train(config, lossFn=myLoss)`. The term functions (`centerLoss`, `sizeLoss`, `objectConfidenceLoss`, `noObjectConfidenceLoss`, `classificationLoss`, `assignResponsibleBoxes`) can be reused or replaced individually.
- **Fair comparison:** keep `seed`, `data.*`, `model.*` and `training.*` identical across runs, and give each run its own `outputDir` and `checkpointDir`.

### Abdul — evaluation

- **Load a checkpoint:** `model, checkpoint = loadModelFromCheckpoint("checkpoints/baseline/best.pt")` (in `src.training.checkpoint`). This returns the model in eval mode, plus `checkpoint["config"]` and `checkpoint["trainingHistory"]`.
- **Data:** `trainLoader, valLoader = buildDataLoaders(DataConfig(**...), batchSize, seed)`. Alternatively, call `loadExperimentConfig(...)` and pass `config.data`.
- **Inference:** `outputs = collectPredictions(model, valLoader)` (in `src.training.inference`) gives `outputs.predictions (N,7,7,13)`, `outputs.targets (N,7,7,8)` and `outputs.annotations` (the list of all ground-truth boxes).
- **Decoding:** `decodePredictions(predictions, numBoxes=2)` (in `src.utils.decoding`) gives `imageBoxes`, `cellBoxes`, `confidence`, `classProbabilities` and `classScores`. `predictionsToDetections(...)` gives per-image `[left, top, right, bottom, score, classId]` after NMS.
- **Responsible-box IoU (the proposal's main metric):** `responsibleMask, boxIous = assignResponsibleBoxes(predictedBoxes, targets[..., :4], targets[..., 4])`, using `splitPredictions` from `src.utils.decoding`. Mean IoU over the responsible boxes is `(boxIous * responsibleMask).sum() / responsibleMask.sum()`.
- **Box utilities:** `boxIou`, `pairwiseIou`, `cxcywhToXyxy` and `nonMaxSuppression` in `src.utils.boxes`.
- **Training logs:** `readHistory("results/baseline")` (in `src.training.history`), or `history.csv` directly.
- **Per-epoch metrics during training:** `train(config, extraMetricsFn=fn)`, where `fn(model, valLoader, device) -> dict`. The returned keys are added to `history.json`/`.csv`, which is how to add IoU, recall@0.5 and gradient norms.

### Mohammad — mathematical verification

- **Per-term gradients:** each loss term in `src/losses/yoloLoss.py` is a pure function of tensors. Call one with `requires_grad=True` inputs and compare `torch.autograd.grad` with your analytical derivative.
- **Fixed inputs:** `tests/testLoss.py` shows how to build a single-object target and a hand-specified prediction (`singleObjectTarget`, `predictionWithBoxes`), with IoU 1/3 and other known values. Copy that pattern for analytical-vs-autograd checks.
- **Responsibility assignment:** `assignResponsibleBoxes` can be tested in isolation, as in `testResponsibilityPicksHighestIou`.
- **Square-root term:** `safeSqrt` adds ε = 1e-6 inside the square root. Set `sqrtEpsilon=0.0` in `YoloLossConfig` for exact comparisons when every width is positive. The gradient of `(√w − √ŵ)²` with respect to `w` is `(√w − √ŵ)/√w`.
- **Gradients through the network:** with the default sigmoid head, gradients pass through the sigmoid. For a cleaner analytic derivation, feed prediction tensors straight into the loss, or use `outputActivation: linear`.

## Tests

The suite in `tests/` runs with `python -m pytest`. It covers:

- **Model:** output shape for batch sizes 1 and 32, the CPU forward pass, seeded initialization and output ranges.
- **Synthetic data:** reproducibility in memory and on disk, box validity against the rendered masks, and collision avoidance.
- **Target encoding:** encoding, cell assignment and decode round-trips, and the collision policy.
- **VOC adapter:** XML parsing and splitting on a fake VOC tree.
- **Loss:** a hand-computed example, responsibility with known IoUs, finite loss and gradients for every head/loss variant, and a fixed-batch optimizer step.
- **Training:** checkpoint round-trip, one full epoch, resume equivalence, the CLI, and the inference and decoding interfaces.
