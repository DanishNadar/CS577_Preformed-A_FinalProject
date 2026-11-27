# Loss ablation summary

Seeds: 1, 2, 577. Epochs: 30. Values are the mean of the last 3 epochs per run, then mean ± std over seeds. Differences are paired by seed against the paper loss.

## Variants

| Variant | λcoord | λnoobj | sqrt(w,h) | Description |
|---|---|---|---|---|
| `paper` | 5 | 0.5 | yes | Paper loss: lambdaCoord=5, lambdaNoobj=0.5, sqrt(w), sqrt(h) |
| `unitWeights` | 1 | 1 | yes | Ablation 1 (proposal): both weights set to 1 |
| `directWh` | 5 | 0.5 | no | Ablation 2 (proposal): regress w, h directly, no sqrt |
| `unitCoord` | 1 | 0.5 | yes | Factorial: only lambdaCoord set to 1 |
| `unitNoobj` | 5 | 1 | yes | Factorial: only lambdaNoobj set to 1 |
| `unitWeightsDirectWh` | 1 | 1 | no | Factorial: both ablations together |

## Final metrics

| Metric | `paper` | `unitWeights` | `directWh` | `unitCoord` | `unitNoobj` | `unitWeightsDirectWh` |
|---|---|---|---|---|---|---|
| Mean IoU (responsible box) ↑ | 0.559 ± 0.049 | 0.224 ± 0.227 | 0.575 ± 0.016 | 0.522 ± 0.033 | 0.560 ± 0.061 | 0.337 ± 0.286 |
| Responsible IoU >= 0.5 ↑ | 0.609 ± 0.087 | 0.159 ± 0.269 | 0.630 ± 0.033 | 0.544 ± 0.051 | 0.611 ± 0.105 | 0.347 ± 0.300 |
| Mean IoU, small objects (<24 px) ↑ | 0.386 ± 0.034 | 0.127 ± 0.207 | 0.382 ± 0.014 | 0.352 ± 0.026 | 0.397 ± 0.047 | 0.211 ± 0.182 |
| Mean IoU, medium objects (24-48 px) ↑ | 0.618 ± 0.065 | 0.213 ± 0.266 | 0.643 ± 0.011 | 0.574 ± 0.054 | 0.612 ± 0.077 | 0.373 ± 0.318 |
| Mean IoU, large objects (>=48 px) ↑ | 0.758 ± 0.052 | 0.395 ± 0.222 | 0.794 ± 0.005 | 0.726 ± 0.040 | 0.753 ± 0.066 | 0.501 ± 0.419 |
| Relative w/h error, small ↓ | 0.460 ± 0.048 | 3.056 ± 2.257 | 0.414 ± 0.004 | 0.515 ± 0.056 | 0.440 ± 0.078 | 0.899 ± 0.656 |
| Relative w/h error, large ↓ | 0.118 ± 0.026 | 0.617 ± 0.388 | 0.100 ± 0.004 | 0.142 ± 0.022 | 0.121 ± 0.036 | 0.414 ± 0.491 |
| Recall@0.5 after NMS ↑ | 0.286 ± 0.190 | 0.000 ± 0.000 | 0.427 ± 0.024 | 0.295 ± 0.131 | 0.234 ± 0.192 | 0.175 ± 0.151 |
| Precision@0.5 after NMS ↑ | 0.765 ± 0.075 | 0.093 ± 0.160 | 0.818 ± 0.023 | 0.774 ± 0.039 | 0.766 ± 0.244 | 0.570 ± 0.494 |
| Class accuracy (occupied cells) ↑ | 0.643 ± 0.134 | 0.347 ± 0.054 | 0.745 ± 0.018 | 0.733 ± 0.104 | 0.658 ± 0.237 | 0.666 ± 0.254 |
| Val loss, paper weights ↓ | 2.467 ± 0.736 | 9.938 ± 5.440 | 1.929 ± 0.099 | 2.417 ± 0.534 | 2.486 ± 1.021 | 6.144 ± 6.866 |
| Val no-object term, paper weights ↓ | 0.072 ± 0.022 | 0.003 ± 0.005 | 0.073 ± 0.005 | 0.053 ± 0.012 | 0.032 ± 0.016 | 0.016 ± 0.014 |

## Paired difference vs `paper` (mean over seeds)

| Metric | `paper` | `unitWeights` | `directWh` | `unitCoord` | `unitNoobj` | `unitWeightsDirectWh` |
|---|---|---|---|---|---|---|
| Mean IoU (responsible box) | — | -0.335 (0/3 ≥ paper) | +0.016 (1/3 ≥ paper) | -0.037 (0/3 ≥ paper) | +0.001 (1/3 ≥ paper) | -0.221 (0/3 ≥ paper) |
| Mean IoU, small objects (<24 px) | — | -0.259 (0/3 ≥ paper) | -0.004 (1/3 ≥ paper) | -0.034 (0/3 ≥ paper) | +0.011 (2/3 ≥ paper) | -0.175 (0/3 ≥ paper) |
| Mean IoU, large objects (>=48 px) | — | -0.363 (0/3 ≥ paper) | +0.036 (3/3 ≥ paper) | -0.032 (1/3 ≥ paper) | -0.005 (1/3 ≥ paper) | -0.257 (1/3 ≥ paper) |
| Recall@0.5 after NMS | — | -0.286 (0/3 ≥ paper) | +0.141 (3/3 ≥ paper) | +0.009 (2/3 ≥ paper) | -0.053 (0/3 ≥ paper) | -0.112 (1/3 ≥ paper) |
| Val loss, paper weights | — | +7.471 (3/3 ≥ paper) | -0.538 (1/3 ≥ paper) | -0.049 (1/3 ≥ paper) | +0.020 (2/3 ≥ paper) | +3.678 (2/3 ≥ paper) |

## Training dynamics and stability

| Metric | `paper` | `unitWeights` | `directWh` | `unitCoord` | `unitNoobj` | `unitWeightsDirectWh` |
|---|---|---|---|---|---|---|
| Epochs until mean IoU ≥ 0.5 | 2.0 ± 0.0 | 15.0 ± 0.0 | 1.3 ± 0.6 | 2.0 ± 0.0 | 3.3 ± 2.3 | 3.0 ± 0.0 |
| Best mean IoU (any epoch) | 0.568 ± 0.046 | 0.238 ± 0.232 | 0.581 ± 0.013 | 0.535 ± 0.025 | 0.574 ± 0.054 | 0.347 ± 0.283 |
| Val paper-loss spikes (>10% rise) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| Train grad norm, mean | 3.62 ± 0.06 | 6.23 ± 2.29 | 3.73 ± 0.39 | 2.62 ± 0.48 | 4.08 ± 0.25 | 3.74 ± 1.06 |
| Train grad norm, peak | 126.5 ± 123.7 | 2721.4 ± 1388.8 | 268.0 ± 210.2 | 287.3 ± 245.9 | 88.7 ± 50.8 | 219.0 ± 146.7 |
| Train grad norm std, epochs 16-30 | 1.01 ± 0.24 | 8.14 ± 9.22 | 0.76 ± 0.28 | 0.68 ± 0.22 | 0.93 ± 0.36 | 3.20 ± 4.08 |
| Non-finite steps | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| noobj / obj grad-norm ratio, epoch 1 | 1.93 ± 0.30 | 0.00 ± 0.00 | 2.03 ± 0.34 | 1.31 ± 0.50 | 2.70 ± 0.11 | 2.03 ± 2.95 |
| noobj / obj grad-norm ratio, mean | 1.13 ± 0.11 | 1.91 ± 3.30 | 0.97 ± 0.04 | 0.97 ± 0.12 | 1.18 ± 0.14 | 0.68 ± 0.60 |
| Seconds per epoch | 9.5 ± 0.4 | 9.4 ± 0.4 | 9.4 ± 0.4 | 9.4 ± 0.3 | 9.4 ± 0.4 | 9.4 ± 0.4 |
