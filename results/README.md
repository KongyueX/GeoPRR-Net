# ReMSTNet-v3 paper results

This directory is the public, code-only result surface for the manuscript
**“ReMSTNet: Relation-Encoded Moment-Exact Transport for Pointer-Gauge Reading
under Projective Distortion.”** Values are mirrored in machine-readable form in
[`remstnet_v3_tables.json`](remstnet_v3_tables.json). Lower NMAE is better.

## Dataset and protocol inventory

| Dataset / split | Images | Groups | Grouping unit | Role |
|---|---:|---:|---|---|
| SyncG Fit | 14,442 | 131 | scene stem | foundation fitting; parent correction roster |
| SyncG correction subset | 6,616 | 60 | scene stem | ReMST modules only |
| SyncG Scene-Holdout | 1,558 | 14 | scene stem | disjoint retrospective same-domain benchmark |
| FieldGauge-ROI Test-A | 434 | 11 | source directory | retrospective test only |
| FieldGauge-ROI Test-B | 814 | 20 | source directory | retrospective test only |
| External-ROI | 147 | 21 | provisional capture session | retrospective test only |
| RF100-derived | 151 | 35 | conservative source/evaluation group | retrospective test only |
| Natural-repeat | 1,203 | 31 | physical gauge | 86 repeat units; stability audit |

The SyncG foundation was fitted on 14,442 images from 131 scenes. ReMST modules
were fitted only on the 6,616-image, 60-scene correction subset. Scene-Holdout
is disjoint from fitting, but it was inspected repeatedly during historical
method development and is therefore reported as a retrospective benchmark,
not untouched confirmation.

## Main SyncG Scene-Holdout comparison

Mean ± sample standard deviation across three independently fitted models.

| Method | Clean | Blur-M | Blur-S | Perspective 25° | Perspective 45° | 45° + blur | All | Projective pooled |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SARN-v2 + Direct | 0.007910 ± 0.000411 | 0.007932 ± 0.000397 | 0.008207 ± 0.000420 | 0.014422 ± 0.000343 | 0.024302 ± 0.000398 | 0.026219 ± 0.000564 | 0.014832 ± 0.000346 | 0.021648 ± 0.000435 |
| SARN-v2 + EfficientNet-B0 | 0.008099 ± 0.000323 | 0.008064 ± 0.000338 | 0.008282 ± 0.000472 | 0.014177 ± 0.000607 | 0.021578 ± 0.000968 | 0.023411 ± 0.000995 | 0.013935 ± 0.000606 | 0.019722 ± 0.000852 |
| SARN-v2 + MobileNetV3-Large | 0.009355 ± 0.000435 | 0.009421 ± 0.000489 | 0.009807 ± 0.000385 | 0.016255 ± 0.000489 | 0.025584 ± 0.001962 | 0.027589 ± 0.001664 | 0.016335 ± 0.000716 | 0.023143 ± 0.001361 |
| **ReMSTNet-v3** | **0.008099 ± 0.000323** | **0.008064 ± 0.000338** | **0.008282 ± 0.000471** | **0.010330 ± 0.000476** | **0.013884 ± 0.000441** | **0.015564 ± 0.000299** | **0.010704 ± 0.000373** | **0.013259 ± 0.000391** |

Relative to SARN-v2 + EfficientNet-B0, ReMSTNet-v3 reduces all-condition NMAE
by **23.19%** and projective-pooled NMAE by **32.77%**. Clean and blur-only
endpoints are effectively preserved; the principal gain is concentrated in
projective conditions.

## Controlled transformations on real-source ROIs

Each row aggregates six controlled conditions. The paired difference is
ReMSTNet-v3 minus SARN-v2 + EfficientNet-B0 after rowwise averaging across the
three fitted seeds. Intervals are 95% group-bootstrap CIs with 20,000
replicates; they do **not** include uncertainty over retraining.

| Dataset | Images / groups | ReMSTNet-v3 | SARN-v2 + B0 | Paired Δ | 95% CI |
|---|---:|---:|---:|---:|---:|
| FieldGauge-ROI Test-A | 434 / 11 | 0.116689 ± 0.012464 | 0.116963 ± 0.012283 | -0.000275 | [-0.000438, -0.000116] |
| FieldGauge-ROI Test-B | 814 / 20 | 0.112915 ± 0.015050 | 0.113247 ± 0.014806 | -0.000332 | [-0.000487, -0.000177] |
| External-ROI | 147 / 21 | 0.201162 ± 0.057755 | 0.201135 ± 0.057342 | +0.000027 | [-0.000106, +0.000155] |
| RF100-derived | 151 / 35 | 0.050289 ± 0.015948 | 0.051548 ± 0.015739 | -0.001259 | [-0.001893, -0.000530] |

These experiments use controlled transformations of real cropped ROIs. They do
not establish performance under arbitrary natural camera tilt.

## Factorial mechanism ablation

This is a fixed-source-fit ablation using source seed 20262022. It isolates the
two v3 additions but does not estimate training-seed uncertainty.

| Progress mixing | Adaptive budget | Trainable params | All NMAE | Projective NMAE |
|:---:|:---:|---:|---:|---:|
| No | No | 252,183 | 0.010800 | 0.013858 |
| Yes | No | 261,527 | 0.010682 | 0.013623 |
| No | Yes | 252,184 | 0.011067 | 0.014394 |
| **Yes** | **Yes** | **261,528** | **0.010276** | **0.012811** |

The estimated interaction is -0.000674 (95% CI [-0.000792, -0.000550]).

## Perspective stress sweep

| Perspective | Raw | SARN | ReMSTNet-v3 | Relation available |
|---:|---:|---:|---:|---:|
| 0° | 0.007738 | 0.007738 | 0.007738 | 0.0% |
| 15° | 0.012094 | 0.010437 | 0.009111 | 95.0% |
| 30° | 0.019283 | 0.015121 | 0.010092 | 96.4% |
| 45° | 0.041001 | 0.019950 | 0.012521 | 96.1% |
| 60° | 0.141743 | 0.141743 | 0.141743 | 0.0% (raw fallback) |

The 60° row is an explicit fallback case: relation geometry is unavailable and
the final posterior equals the raw posterior.

## Efficiency

Windows, Python 3.11, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4060, BF16
autocast. Timing covers the model forward pass; image decoding and SARN
materialization are excluded.

| Method | Params (trainable) | Batch-1 mean / P50 / P95 | Batch-8 ms/sample | Batch-8 samples/s | Peak MiB B1 / B8 |
|---|---:|---:|---:|---:|---:|
| Raw foundation | 4.010M (0) | 7.973 / 7.834 / 9.635 ms | 1.123 | 890.82 | 30.9 / 81.7 |
| Raw + SARN twin endpoint | 4.010M (0) | 14.551 / 14.506 / 15.142 ms | 2.118 | 472.25 | 31.8 / 88.7 |
| **ReMSTNet-v3** | **4.272M (0.262M)** | **30.640 / 30.596 / 31.969 ms** | **4.138** | **241.67** | **34.2 / 94.9** |

## Natural-repeat stability

The retained cohort contains 1,203 samples, 31 physical gauges and 86 repeat
units. ReMSTNet-v3, Raw and SARN are identical on this cohort:

| Quantity | Mean ± sample SD |
|---|---:|
| Full-denominator NMAE | 0.1158669543 ± 0.0175594953 |
| Mean within-unit prediction range | 0.1431469954 ± 0.0132710207 |
| Mean within-unit population SD | 0.0407908131 ± 0.0032719166 |

Both paired differences and their intervals are exactly [0, 0]. This is
evidence of **preservation**, not an improvement claim.

## Interpretation boundaries

- “Moment-exact” means equality to a feasible first-moment target within FP32
  numerical tolerance. Boundary clipping may shorten a requested displacement,
  but it cannot reverse its direction.
- The foundation uses ImageNet initialization and task-specific SyncG fitting.
- Group-bootstrap intervals in the real-source table condition on the averaged
  predictions from three fitted seeds; they do not describe a retraining
  distribution.
- Model-level timing is not end-to-end service latency.
- The model predicts normalized progress for a cropped single-pointer ROI; it
  does not by itself detect a meter, read printed scale values or produce a
  physical-unit reading.
