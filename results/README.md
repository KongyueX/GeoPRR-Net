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
| Industrial Real-Photo Baseline | 1,395 | 52 | pooled real-photo source group | retrospective in-house industrial ROI benchmark |
| FieldGauge-ROI Test-A | 434 | 11 | source directory | industrial-baseline source subdivision |
| FieldGauge-ROI Test-B | 814 | 20 | source directory | industrial-baseline source subdivision |
| External-ROI | 147 | 21 | provisional capture session | industrial-baseline source subdivision |
| RF100-derived | 151 | 35 | conservative source/evaluation group | retrospective test only |
| Natural-repeat | 1,203 | 31 | physical gauge | 86 repeat units; stability audit |
| RPM-10K single-pointer test | 1,797 | 6 | official meter type | zero-shot public scalar-reading transfer |
| Industrial Full-Frame Diagnostic | 153 | 153 | deduplicated raw photo | detector-to-reading replay; 33 labeled |

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

## Secondary VDN annotation-assisted comparison

The ReMSTNet Scene-Holdout and the VDN grouped holdout were created by
different pre-existing protocols. Their full-cohort values are not directly
mergeable. This comparison therefore uses the set intersection only: 129
samples from 6 ReMSTNet scene groups, evaluated under all six identical
condition pixels (774 rows). Membership was selected without predictions or
errors. VDN's direction prediction is converted to normalized progress with
the annotation-derived pivot and ordered scale endpoints; this retains the
complete fixed intersection. ReMSTNet and B0 predict progress directly and do
not receive those annotations.

| Scope | Compared rows (coverage) | ReMSTNet-v3 NMAE | SARN-v2+B0 NMAE | VDN annotation-reference NMAE | ReMSTNet−VDN Δ (95% CI) |
|---|---:|---:|---:|---:|---:|
| All conditions | 774 / 774 (100%) | **0.008582 ± 0.000414** | 0.011673 ± 0.000638 | 0.016444 | -0.007861 [-0.012340, -0.005530] |
| Projective pooled | 387 / 387 (100%) | **0.010938 ± 0.000935** | 0.017119 ± 0.001396 | 0.021030 | -0.010092 [-0.014298, -0.006818] |

The VDN checkpoint is the local terminal epoch-200 SyncG retraining of the
upstream architecture, not an official pretrained weight. It is one
checkpoint; ReMSTNet and B0 values are mean ± sample SD over three fits. The
annotation-derived reference arc makes this a non-deployable component
comparison rather than an input-equivalent automatic-system comparison. The
paired intervals resample only six scene groups and do not include VDN
retraining uncertainty, so this remains supporting evidence rather than a
replacement for the primary 1,558-image comparison.

Condition-level results are:

| Condition | Rows (coverage) | ReMSTNet-v3 | SARN-v2+B0 | VDN annotation reference | ReMSTNet−VDN Δ (95% CI) |
|---|---:|---:|---:|---:|---:|
| Clean | 129 (100%) | **0.006099 ± 0.000170** | 0.006099 ± 0.000170 | 0.009534 | -0.003434 [-0.005162, -0.001968] |
| Blur-moderate | 129 (100%) | **0.006122 ± 0.000150** | 0.006122 ± 0.000150 | 0.010204 | -0.004082 [-0.005639, -0.002719] |
| Blur-severe | 129 (100%) | 0.006457 ± 0.000138 | **0.006457 ± 0.000137** | 0.015834 | -0.009377 [-0.022822, -0.002861] |
| Perspective-moderate | 129 (100%) | **0.008422 ± 0.001031** | 0.011986 ± 0.000966 | 0.009114 | -0.000692 [-0.002282, +0.001191] |
| Perspective-severe | 129 (100%) | **0.011555 ± 0.000946** | 0.018918 ± 0.001789 | 0.025390 | -0.013835 [-0.016721, -0.010404] |
| Perspective + blur severe | 129 (100%) | **0.012838 ± 0.000893** | 0.020455 ± 0.001507 | 0.028587 | -0.015748 [-0.025011, -0.009978] |

The aggregate reductions relative to the annotation-assisted VDN reference are
**47.81%** over all rows and **47.99%** in the projective pool. Only the
perspective-moderate condition is unresolved individually; the pooled
projective interval is below zero.

## Public external datasets

RPM-10K supplies ordered range and scalar-reading labels, so it supports a
zero-shot full-denominator reading test. The official single-pointer test
subset contains 1,797 images from six meter types. Existing detector crops were
used; 1,775 images passed detection and 22 failures received normalized error
1. Across three frozen ReMSTNet fits, detector/read coverage was 98.78%, NMAE
was **0.233945 ± 0.012331**, Acc@2% was **10.55 ± 2.01%**, and Acc@5% was
**24.43 ± 1.04%**. The rowwise three-seed mean NMAE had an image-within-type
bootstrap 95% CI of [0.225546, 0.242634]. The relation was inactive on this
single-view protocol because it supplies identical raw/normalized images, unit
support, an inactive relation mask and identity homography. RPM-10K is not a
clean-only dataset: 474 images are tagged blur, 993 tilted and 157 both. The
reported equality with B0 is structurally imposed by the evaluator, so this is
a raw-foundation transfer diagnostic and cannot measure ReMST relation gains.

## Industrial real-photo baseline

`C:/pointer_read/unified_real_photo_progress_v1` is the organized version of
the project's real deployment photographs. It contains 1,395 labeled ROIs from
52 groups. A decoded-pixel audit shows exact identity with the union of the
three existing FieldGauge sources: 1,395 unique pixels on each side and zero
rows in either difference. Model predictions were not used to form the set.

| Scope | ReMSTNet-v3 | SARN-v2 + B0 | Paired Δ | 95% group-bootstrap CI |
|---|---:|---:|---:|---:|
| Native clean | 0.123046 ± 0.021714 | 0.123046 ± 0.021714 | +0.000000002 | [+0.000000001, +0.000000003] |
| Perspective-moderate | **0.117033 ± 0.016771** | 0.117349 ± 0.016716 | -0.000316 | [-0.000513, -0.000090] |
| Perspective-severe | **0.125612 ± 0.017054** | 0.126244 ± 0.016491 | -0.000632 | [-0.000859, -0.000404] |
| Perspective + blur severe | **0.125125 ± 0.012872** | 0.125834 ± 0.012258 | -0.000709 | [-0.001011, -0.000420] |
| Six conditions | **0.123388 ± 0.018553** | 0.123664 ± 0.018317 | -0.000276 | [-0.000382, -0.000168] |
| Projective pooled | **0.122590 ± 0.015220** | 0.123142 ± 0.014802 | -0.000552 | [-0.000761, -0.000335] |

All three controlled projective-condition intervals and both pooled intervals
are below zero. Native clean and blur-only images preserve the raw endpoint
because the defined projective-support relation is inactive. Thus the result
supports selective transport on projectively stressed industrial photographs;
it does not claim automatic correction of arbitrary natural blur or viewpoint.

## Industrial full-frame diagnostic

The organized dataset also retains 153 unique full photographs derived from
the 503 files in repository `data/` and `data-717/`, comprising 33
scalar-labeled and 120 unlabeled frames. A cached best-confidence YOLO
detection was replayed on each frame and fed to the three frozen readers.
Detection/read coverage was **151/153 (98.69%)**. All 33 labeled frames passed;
their NMAE was **0.051930 ± 0.001031**, Acc@2% **34.34 ± 13.66%**, and Acc@5%
**70.71 ± 1.75%**. The rowwise mean NMAE image-bootstrap 95% CI was [0.032544,
0.079142].

This is a retrospective deployment-source full-frame-to-reading replay, not a
fresh detector benchmark: bounding-box ground truth is unavailable, so recall
and IoU cannot be reported, and unlabeled frames contribute to coverage only.
The relation-disabled route again invokes exact raw-foundation fallback. OCR
and automatic physical-range recovery are evaluated separately below.


## Independent existing-model OCR deployment replay

[remstnet_ocr_end_to_end_deployment.json](remstnet_ocr_end_to_end_deployment.json)
records a separate prediction-then-scoring experiment on the same 153 organized
deployment photographs. The label-free pass uses cached detector boxes, reruns
the three ReMSTNet checkpoints, and combines normalized progress with PP-OCRv4
through RapidOCR/ONNX Runtime plus the existing production point geometry. No
OCR weight, decoder threshold, or range rule was fitted to the 33 field labels.

| Stage / operating point | All-photo outputs | Fraction of 153 |
|---|---:|---:|
| Cached detector | 151 | 98.69% |
| Any numeric OCR token | 83 | 54.25% |
| Physical reading, decoder default | 27 | 17.65% |
| Physical reading, transferred 0.95 sensitivity | 5 | 3.27% |

On the 33 labeled photographs, the default decoder accepts 9 (27.27%). Its
conditional physical-unit NMAE is **0.036814 ± 0.004015**; the same nine
ReMSTNet predictions with the true ranges give **0.034336 ± 0.003848**, so the
matched incremental range-stage difference is **+0.002478 ± 0.002546**.
Conditional Acc@5% is **70.37 ± 6.42%**. Both recovered endpoints are within 5%
of the true span for 6/9 accepted photographs and within 10% for 9/9. Missing
outputs receive error 1 on the labeled full denominator, producing NMAE
**0.737313 ± 0.001095**.

The 0.95 operating point is a transferred sensitivity rather than a field-
calibrated threshold and retains only 2/33 labeled photographs. Cached boxes
exclude live detector latency and prevent detector recall/IoU claims. The result
therefore demonstrates feasible use of an existing OCR model on accepted
photographs, not an OCR architecture contribution or a high-coverage live
camera-to-reading system.


## Controlled transformations on real-source ROIs

Each row aggregates six controlled conditions. The paired difference is
ReMSTNet-v3 minus SARN-v2 + EfficientNet-B0 after rowwise averaging across the
three fitted seeds. Intervals are 95% group-bootstrap CIs with 20,000
replicates; they do **not** include uncertainty over retraining.

| Dataset | Images / groups | ReMSTNet-v3 | SARN-v2 + B0 | Paired Δ | 95% CI |
|---|---:|---:|---:|---:|---:|
| **Industrial Real-Photo Baseline** | **1,395 / 52** | **0.123388 ± 0.018553** | 0.123664 ± 0.018317 | **-0.000276** | **[-0.000382, -0.000168]** |
| FieldGauge-ROI Test-A | 434 / 11 | 0.116689 ± 0.012464 | 0.116963 ± 0.012283 | -0.000275 | [-0.000438, -0.000116] |
| FieldGauge-ROI Test-B | 814 / 20 | 0.112915 ± 0.015050 | 0.113247 ± 0.014806 | -0.000332 | [-0.000487, -0.000177] |
| External-ROI | 147 / 21 | 0.201162 ± 0.057755 | 0.201135 ± 0.057342 | +0.000027 | [-0.000106, +0.000155] |
| RF100-derived | 151 / 35 | 0.050289 ± 0.015948 | 0.051548 ± 0.015739 | -0.001259 | [-0.001893, -0.000530] |

The bold industrial row is the pooled benchmark; the next three rows are its
source subdivisions. These experiments use controlled transformations of real
cropped ROIs. They do not establish performance under arbitrary natural camera
tilt.

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
- The VDN comparison uses all 774 rows in a fixed cross-split intersection.
  VDN receives annotation-derived pivot and ordered scale endpoints for
  offline direction-to-progress conversion, so this is a component comparison
  rather than an input-equivalent automatic-system result.
- RPM-10K contains blur and tilted views, but its released single-view
  evaluator disables the relation path; the result is a raw-foundation
  diagnostic, not evidence about relation-module benefit.
- The Industrial Real-Photo Baseline contains 1,395 scored ROIs from 52 groups.
  The separate 153-photo full-frame diagnostic uses cached detections and only
  33 scalar labels, so it is not a detector recall or localization audit.
- Model-level timing is not end-to-end service latency.
- The model predicts normalized progress for a cropped single-pointer ROI; it
  does not by itself detect a meter, read printed scale values or produce a
  physical-unit reading.
