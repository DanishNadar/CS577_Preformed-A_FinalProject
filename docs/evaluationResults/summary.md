# Evaluation summary

Checkpoint: `last.pt` of every run. Seeds: 1, 2, 577. Validation split of each seed. Values are mean ± std over seeds.

AP uses VOC all-point interpolation over detections with score ≥ 0.01 after class-wise NMS (IoU 0.5). Recall, precision and the error breakdown use the operating point score ≥ 0.2.

## Detection metrics

| Metric | `paper` | `unitWeights` | `directWh` | `unitCoord` | `unitNoobj` | `unitWeightsDirectWh` |
|---|---|---|---|---|---|---|
| mAP@0.5 | 0.380 ± 0.170 | 0.011 ± 0.020 | 0.493 ± 0.029 | 0.377 ± 0.121 | 0.356 ± 0.241 | 0.256 ± 0.222 |
| mAP@0.75 | 0.133 ± 0.096 | 0.001 ± 0.001 | 0.212 ± 0.008 | 0.110 ± 0.057 | 0.142 ± 0.112 | 0.083 ± 0.073 |
| mAP@[.5:.95] | 0.175 ± 0.093 | 0.003 ± 0.005 | 0.242 ± 0.014 | 0.159 ± 0.064 | 0.167 ± 0.121 | 0.111 ± 0.096 |
| Mean IoU (responsible box) | 0.561 ± 0.051 | 0.222 ± 0.223 | 0.574 ± 0.018 | 0.528 ± 0.037 | 0.565 ± 0.061 | 0.338 ± 0.290 |
| Recall@0.5 (score ≥ 0.2) | 0.306 ± 0.181 | 0.000 ± 0.001 | 0.425 ± 0.029 | 0.313 ± 0.127 | 0.239 ± 0.194 | 0.185 ± 0.160 |
| Precision@0.5 (score ≥ 0.2) | 0.770 ± 0.088 | 0.111 ± 0.192 | 0.830 ± 0.035 | 0.755 ± 0.050 | 0.767 ± 0.258 | 0.582 ± 0.504 |
| AP@0.5, circle | 0.399 ± 0.133 | 0.003 ± 0.006 | 0.505 ± 0.075 | 0.411 ± 0.064 | 0.374 ± 0.240 | 0.295 ± 0.261 |
| AP@0.5, square | 0.365 ± 0.161 | 0.015 ± 0.027 | 0.466 ± 0.045 | 0.319 ± 0.164 | 0.324 ± 0.198 | 0.193 ± 0.170 |
| AP@0.5, triangle | 0.376 ± 0.222 | 0.015 ± 0.026 | 0.508 ± 0.018 | 0.401 ± 0.145 | 0.369 ± 0.289 | 0.279 ± 0.243 |

## Paired difference vs `paper` (mean over seeds)

| Metric | `unitWeights` | `directWh` | `unitCoord` | `unitNoobj` | `unitWeightsDirectWh` |
|---|---|---|---|---|---|
| mAP@0.5 | -0.369 (0/3 ≥ paper) | +0.113 (2/3 ≥ paper) | -0.003 (2/3 ≥ paper) | -0.024 (1/3 ≥ paper) | -0.124 (1/3 ≥ paper) |
| mAP@0.75 | -0.132 (0/3 ≥ paper) | +0.079 (2/3 ≥ paper) | -0.023 (2/3 ≥ paper) | +0.009 (1/3 ≥ paper) | -0.050 (1/3 ≥ paper) |
| mAP@[.5:.95] | -0.172 (0/3 ≥ paper) | +0.067 (2/3 ≥ paper) | -0.016 (2/3 ≥ paper) | -0.008 (1/3 ≥ paper) | -0.064 (1/3 ≥ paper) |

## Error breakdown (share of ground-truth objects, all seeds pooled)

Each object gets its best outcome among detections with score ≥ 0.2: **correct** (right class, IoU ≥ 0.5), **localization** (right class, 0.1 ≤ IoU < 0.5), **classification** (wrong class, IoU ≥ 0.5), **missed** (none of these).

| Variant | Objects | all: correct | all: localization | all: classification | all: missed | small: correct | medium: correct | large: correct |
|---|---|---|---|---|---|---|---|---|
| `paper` | 2972 | 30.6% | 3.1% | 1.2% | 65.1% | 2.1% | 34.4% | 71.2% |
| `unitWeights` | 2972 | 0.0% | 0.0% | 0.1% | 99.9% | 0.0% | 0.0% | 0.1% |
| `directWh` | 2972 | 42.4% | 3.9% | 1.1% | 52.6% | 4.0% | 52.6% | 90.5% |
| `unitCoord` | 2972 | 31.4% | 5.0% | 0.7% | 62.9% | 2.2% | 33.2% | 75.7% |
| `unitNoobj` | 2972 | 23.9% | 1.0% | 0.7% | 74.5% | 1.3% | 26.1% | 56.9% |
| `unitWeightsDirectWh` | 2972 | 18.7% | 2.2% | 0.1% | 79.0% | 1.0% | 18.5% | 47.3% |
