# GeoPRR-Net

### Geometry-Aware Polar-Relational Routing for Robust Analog Gauge Reading

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.11](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-CPU--safe-2EA44F.svg)](#verification-and-release-audit)
[![Manuscript](https://img.shields.io/badge/manuscript-MDPI%20Electronics-008A8A.svg)](https://github.com/KongyueX/GeoPRR-Net-Paper/blob/main/manuscript.tex)
[![License](https://img.shields.io/badge/license-not%20yet%20declared-lightgrey.svg)](#license-and-third-party-materials)

GeoPRR-Net is the final and only supported model identity in this repository.
This research artifact contains the model implementation, training and evaluation
entry points, matched VDN comparison, release audit, and focused tests used by
the accompanying paper. Manuscript source, figures, and compact aggregate figure
data are maintained in the separate
[GeoPRR-Net-Paper](https://github.com/KongyueX/GeoPRR-Net-Paper) repository.

> **Research-artifact status.** The public API and CPU-safe tests can be checked
> from this repository alone. Reproducing the reported GPU experiments requires
> separately obtained datasets and locally reproduced checkpoints. No model
> weights or restricted industrial images are distributed here.

## Scope

GeoPRR-Net estimates normalized pointer progress in <code>[0, 1]</code> from a
localized single-pointer analog-gauge ROI with known ordered scale endpoints.

| Contract | Included |
|---|---|
| Input | Raw ROI, support-normalized ROI, support mask, view-availability flag, and raw-to-normalized homography |
| Output | Progress posterior, normalized mean reading, uncertainty, candidate predictions, and routing weights |
| Core task | Geometry-aware progress estimation under appearance and projective variation |
| Out of scope | Full-image meter detection, OCR, automatic scale-range discovery, and multi-pointer disambiguation |

The out-of-scope modules may be connected around GeoPRR-Net, but their
performance must be evaluated separately. Reported component comparisons should
not be interpreted as automatic end-to-end meter-reading rankings.

## Method

GeoPRR-Net uses a shared EfficientNet-B0 foundation over raw and
support-normalized observations. It retains three candidate readings:

1. a geometry-aware base posterior;
2. polar evidence aligned with dial progression; and
3. relational moment transport between the two observations.

A conditional router assigns per-sample candidate weights. A final
moment-consistent projection returns one posterior whose first moment matches
the routed estimate. The paper evaluates executable interventions that remove
geometry-aware fusion, polar evidence, or relational transport, and that replace
adaptive routing with a fixed prior.

## Reported results

NMAE is reported as percentage of full scale (%FS); lower is better.

| Cohort | GeoPRR-Net | Comparison | Evaluation scope |
|---|---:|---:|---|
| SyncG scene-disjoint holdout | **1.0013 ± 0.0382** | best tested raw CNN: 2.0530 ± 0.0294 | 1,558 images, six conditions, three fits |
| SyncG matched VDN component | **1.0013 ± 0.0382** | VDN: 1.6620 ± 0.0878 | identical 9,348-row roster per seed, three terminal checkpoints |
| RF100-VL public transfer | **5.0098 ± 1.5344** | best tested raw CNN: 6.4809 ± 1.2255 | 151 images, six conditions, three fits |
| Industrial-1395 | **18.9% lower NMAE** | best tested raw CNN | 1,395 field ROIs, six conditions, three fits |

For the matched VDN experiment, GeoPRR-Net reduces pooled NMAE by 39.8%
relative to three independently retrained terminal VDN checkpoints. The paired
scene-bootstrap GeoPRR-Net-minus-VDN difference is <code>-0.6607%FS</code>
(95% CI <code>-0.8251</code> to <code>-0.4854</code>). VDN receives
annotation-derived pivot and ordered scale endpoints for offline
direction-to-progress conversion; this is therefore an annotation-assisted
component comparison, not an end-to-end system ranking.

These are manuscript-reported results from the completed formal runs. The
focused repository tests validate code behavior and release integrity; they do
not substitute for a full GPU retraining of those experiments.

## Repository layout

~~~text
geoprr/
  __init__.py                                   only supported public import surface
  model.py                                      GeoPRRNet identity and checkpoint loader
experiments/
  unified_pointer_reader.py                     GeoPRR-Net architecture
  train_unified_pointer_reader.py               staged training and ablations
  evaluate_unified_pointer_reader_syncg.py       SyncG holdout evaluation
  evaluate_unified_pointer_reader_rf100.py       public RF100-VL transfer
  evaluate_unified_pointer_reader_industrial.py  restricted field-cohort replay
  benchmark_unified_pointer_reader_paper_efficiency.py
  summarize_unified_pointer_reader_experiments.py
  evaluate_geoprr_geometry_routing_factorial.py
  evaluate_geoprr_perspective_scan.py
  train_vdn_syncg.py                            matched VDN training
  evaluate_geoprr_vdn_matched.py                 same-pixel VDN evaluation
  summarize_geoprr_vdn_matched.py                paired three-seed summary
  run_geoprr_vdn_matched.ps1                     run/resume wrapper
  train_deeplabv3plus_roi.py                     matched ROI segmentation comparison
  evaluate_deeplabv3plus_roi.py                  six-condition segmentation evaluation
  evaluate_deeplabv3plus_roi_rf100.py            annotation-assisted RF100 field evaluation
  train_yolo11s_pose4kp.py                       matched four-keypoint pose comparison
  evaluate_yolo11s_pose4kp.py                    six-condition pose evaluation
  evaluate_yolo11s_pose4kp_field.py              four-cohort real-photo ROI evaluation
  roi_geometry_field.py                          real-photo cohort/geometry registry
  summarize_roi_geometry_comparison.py           ROI comparison aggregation
  summarize_roi_geometry_full_experiment.py      three-seed synthetic/field aggregation
  roi_reference_geometry.py                     source-trained pivot/start/end adapter
  evaluate_missing_roi_direction_baselines.py    automatic-reference field evaluation
  benchmark_roi_comparison_efficiency.py         native and complete ROI timing
  summarize_roi_comparison_zero_shot.py          per-row transfer metric aggregation
  export_roi_comparison_public_results.py        aggregate-only public data export
test/
  test_geoprr_public_api.py
  test_unified_pointer_reader.py
  test_geoprr_figure3_experiments.py
  test_run_vdn_oracle_reference_component.py
docs/
  GEOPRR_REPRODUCIBILITY.md                      detailed reproduction guide
~~~

Earlier-named modules under <code>experiments/</code>,
<code>remstnet/</code>, and <code>utils/</code> are retained only when the
GeoPRR checkpoint-replay or reproduction path still imports them. They are
internal implementation dependencies, not supported standalone models.
Historical checkpoint protocol and CLI field names remain unchanged where
needed to replay completed experiments.

## Requirements and installation

The formally validated environment is Windows, Python 3.11, PyTorch 2.11, and
CUDA 12.8. CPU-only installations can run the focused tests, while formal
training and reported latency measurements require a compatible CUDA GPU.

Clone the repository and create the pinned environment:

~~~powershell
git clone https://github.com/KongyueX/GeoPRR-Net.git
Set-Location GeoPRR-Net
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
~~~

The lock file is
[<code>experiments/requirements-training.lock.txt</code>](experiments/requirements-training.lock.txt).
The repository does not currently provide a separately tested Linux lock or a
minimal inference-only dependency set.

## Data and checkpoints

| Resource | Availability | Repository policy |
|---|---|---|
| [SyncG](https://doi.org/10.1038/s41597-026-07308-x) | Public source | Images and generated manifests are not redistributed |
| [RF100-VL gauge subset](https://universe.roboflow.com/rf100-vl/needle-base-tip-min-max-u87vi-wzsrt-kjyu) | Public source | Evaluated as an annotation-derived transfer cohort |
| Industrial-1395 | Restricted | Images and per-sample ledgers are not public |
| GeoPRR-Net checkpoints | Reproduce locally | Weight files are ignored by Git |
| Source expert-bank and warm-polar checkpoints | Reproduce locally | Required by the current training/checkpoint lineage |
| External VDN checkout | Public GPL-3.0 source | Loaded from a separate pinned checkout; never vendored here |

Expected local inputs include a scene-disjoint outer split, label-bearing
training manifest, conditioned ROI manifest, reproduced source checkpoints, and
output directories. Exact schemas and cohort boundaries are documented in
[<code>docs/GEOPRR_REPRODUCIBILITY.md</code>](docs/GEOPRR_REPRODUCIBILITY.md).

## Quick verification

Inspect the public identity without loading a checkpoint:

~~~powershell
.\.venv\Scripts\python.exe -c "from geoprr import publication_model_identity; print(publication_model_identity())"
~~~

Run the focused CPU-safe suite:

~~~powershell
.\.venv\Scripts\python.exe -m pytest test\test_geoprr_public_api.py test\test_unified_pointer_reader.py test\test_geoprr_figure3_experiments.py test\test_run_vdn_oracle_reference_component.py
~~~

Load a reproduced checkpoint:

~~~python
from geoprr import GeoPRRNet, load_geoprr_net

model: GeoPRRNet
model, metadata = load_geoprr_net(
    "path/to/geoprr_checkpoint.pt",
    device="cuda:0",
)
~~~

The current checkpoint format preserves its source expert-bank lineage, so the
referenced source checkpoint must remain available when loading a reproduced
GeoPRR checkpoint.

## Reproducing the paper workflow

### 1. Train GeoPRR-Net

~~~powershell
.\.venv\Scripts\python.exe -m experiments.train_unified_pointer_reader --source-r2mt <path-to-source-expert-bank.pt> --warm-polar <path-to-warm-polar.pt> --fit-manifest <path-to-syncg-train-manifest.jsonl> --outer-split <path-to-scene-disjoint-split.json> --output-root artifacts\runs\unified_pointer_reader --device cuda:0 --seed 20262020
~~~

Repeat with seeds <code>20262021</code> and <code>20262022</code>. The
historical <code>--source-r2mt</code> option name is retained for checkpoint
compatibility; it does not expose R2MT as a separate supported model.

### 2. Evaluate a frozen SyncG checkpoint

~~~powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_unified_pointer_reader_syncg --checkpoint <path-to-geoprr-checkpoint.pt> --output artifacts\evaluation\syncg_seed_20262020.json --device cuda:0 --amp
~~~

The evaluator uses the fixed formal cohort definitions in the retained
reproduction modules. Place the required local manifests and reference files as
described in the detailed reproduction guide before running it.

### 3. Run external and efficiency evaluations

~~~powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_unified_pointer_reader_rf100 --checkpoint <path-to-geoprr-checkpoint.pt> --output artifacts\evaluation\rf100_seed_20262020.json --device cuda:0
.\.venv\Scripts\python.exe -m experiments.benchmark_unified_pointer_reader_paper_efficiency --checkpoint <path-to-geoprr-checkpoint.pt> --manifest <path-to-conditioned-roi-manifest.jsonl> --output-json artifacts\evaluation\efficiency_seed_20262020.json --device cuda:0
~~~

The Industrial-1395 evaluator is present for authorized local replay only.
External users cannot reproduce that cohort without permission from the data
owners.

### 4. Aggregate multiseed results

~~~powershell
.\.venv\Scripts\python.exe -m experiments.summarize_unified_pointer_reader_experiments --run-root artifacts\runs\unified_pointer_reader --output-dir artifacts\summary\geoprr --seeds 20262020 20262021 20262022
~~~

Use <code>--help</code> on each entry point to inspect all optional paths,
bootstrap settings, worker counts, and batch sizes before a formal run.

### 5. Reproduce the matched VDN comparison

The adapter uses the upstream
[VectorDetectionNetwork](https://github.com/DrawZeroPoint/VectorDetectionNetwork)
source at the commit recorded in
[<code>experiments/vdn_baseline.py</code>](experiments/vdn_baseline.py). Clone
it outside this repository or under an ignored local artifact directory.

Fresh run:

~~~powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile -File experiments\run_geoprr_vdn_matched.ps1 -OuterSplit <path-to-outer-split.json> -RoiManifest <path-to-conditioned-roi-manifest.jsonl> -PixelReference <path-to-geoprr-pixel-reference.json>
~~~

Resume from validated <code>last.pt</code> epoch boundaries:

~~~powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile -File experiments\run_geoprr_vdn_matched.ps1 -Resume -OuterSplit <path-to-outer-split.json> -RoiManifest <path-to-conditioned-roi-manifest.jsonl> -PixelReference <path-to-geoprr-pixel-reference.json>
~~~

The formal configuration uses three seeds and 200 epochs. Existing seed
directories resume; missing seed directories begin fresh. Checkpoints, logs,
per-sample ledgers, datasets, and private images remain ignored by Git.

### 6. Run the matched ROI geometry comparisons

DeepLabV3+-ROI predicts a pointer mask from pixels. On SyncG and RF100-VL,
its offline conversion uses annotated pivot and ordered scale endpoints and
is an annotation-assisted component comparison. Industrial-1395 uses the
source-trained automatic reference geometry described below.
YOLO11s-Pose-4KP predicts pivot, pointer tip, scale start, and scale end from
the same canonical ROI and does not use annotation geometry at inference.

~~~powershell
.\.venv\Scripts\python.exe -m experiments.train_deeplabv3plus_roi --outer-split <path-to-outer-split.json> --output-dir artifacts\runs\roi_comparison\seed_20262020\deeplabv3plus_roi --seed 20262020
.\.venv\Scripts\python.exe -m experiments.evaluate_deeplabv3plus_roi --checkpoint artifacts\runs\roi_comparison\seed_20262020\deeplabv3plus_roi\best.pt --roi-manifest <path-to-label-free-roi-manifest.jsonl> --output-dir artifacts\runs\roi_comparison\seed_20262020\deeplabv3plus_roi\evaluation

.\.venv\Scripts\python.exe -m experiments.train_yolo11s_pose4kp --outer-split <path-to-outer-split.json> --output-dir artifacts\runs\roi_comparison\seed_20262020\yolo11s_pose4kp --seed 20262020
.\.venv\Scripts\python.exe -m experiments.evaluate_yolo11s_pose4kp --checkpoint artifacts\runs\roi_comparison\seed_20262020\yolo11s_pose4kp\ultralytics\weights\best.pt --roi-manifest <path-to-label-free-roi-manifest.jsonl> --output-dir artifacts\runs\roi_comparison\seed_20262020\yolo11s_pose4kp\evaluation_square

.\.venv\Scripts\python.exe -m experiments.evaluate_deeplabv3plus_roi_rf100 --checkpoint artifacts\runs\roi_comparison\seed_20262020\deeplabv3plus_roi\best.pt --output-dir artifacts\runs\roi_comparison\seed_20262020\deeplabv3plus_roi\field\rf100
.\.venv\Scripts\python.exe -m experiments.evaluate_yolo11s_pose4kp_field --checkpoint artifacts\runs\roi_comparison\seed_20262020\yolo11s_pose4kp\ultralytics\weights\best.pt --output-dir artifacts\runs\roi_comparison\seed_20262020\yolo11s_pose4kp\field

.\.venv\Scripts\python.exe -m experiments.summarize_roi_geometry_full_experiment --run-root artifacts\runs\roi_comparison --geoprr-root artifacts\runs\unified_pointer_reader --output artifacts\runs\roi_comparison_three_seed\summary.json --report artifacts\runs\roi_comparison_three_seed\report.md
~~~

Use seeds <code>20262020</code>, <code>20262021</code>, and
<code>20262022</code> for the paper comparison. Every failed sample-condition
row remains in the denominator with normalized absolute error 1.0.

The completed three-seed comparison on the 1,558-sample, six-condition outer
holdout produced NMAE 1.0013 ± 0.0382%FS for GeoPRR-Net, 1.7076 ± 0.4078%FS
for annotation-assisted DeepLabV3+-ROI, 5.5666 ± 2.9705%FS for
YOLO11s-Pose-4KP, and 1.6620 ± 0.0878%FS for the VDN direction component.
Real-photo transfer is reported separately on Industrial-1395 (1,395 ROIs,
52 groups) and RF100-VL (151 ROIs, 35 groups). On Industrial-1395, YOLO,
DeepLab, and VDN obtain six-condition NMAE 31.0045 ± 7.6146%FS,
32.9356 ± 4.0842%FS, and 25.3840 ± 6.7070%FS, respectively. The latter two
Industrial results were rerun with source-trained YOLO reference geometry on
2026-09-07. On RF100-VL,
the corresponding values are 16.1632 ± 4.6280%FS, 15.8532 ± 4.0813%FS,
and 13.0129 ± 3.2761%FS. DeepLab and VDN use annotated geometry on RF100-VL
and the same-seed SyncG-trained YOLO11s-Pose model's pivot/start/end on
Industrial-1395. Its predicted pointer tip is excluded from detection selection
and decoding. This replaces the legacy detector with unresolved training-data
provenance; all three Industrial pipelines are evaluated within a provided ROI.

Matched RTX 4060 / FP32 / batch-1 measurements give P50 latency of 6.398 ms
for YOLO, 19.558 ms for DeepLab plus source reference detection, and 12.380 ms
for VDN plus source reference detection. Their full loaded parameter counts
are 9.715M, 50.062M, and 25.093M. See the updated results, native-component
timings, peak memory, supported operation counts, and measurement scope in
[<code>docs/ROI_GEOMETRY_COMPARISON_CN.md</code>](docs/ROI_GEOMETRY_COMPARISON_CN.md).

Reproduce the source-trained reference replacement using the commands in
[the detector audit](docs/ROI_REFERENCE_DETECTOR_AUDIT_CN.md), then export:

~~~powershell
.\.venv\Scripts\python.exe -m experiments.summarize_roi_comparison_zero_shot --missing-root artifacts/runs/roi_source_pose_reference_20260907
.\.venv\Scripts\python.exe -m experiments.export_roi_comparison_public_results
~~~

Published summaries are in [docs/data](docs/data). Raw local JSONs, private
input manifests, image-level field ledgers, and weights are not included in
the public export; field-data users must configure their own input paths.

### 7. Rebuild manuscript figures

Clone the paper repository beside this code repository, then rebuild figures
from its compact aggregate CSV files:

~~~powershell
git clone https://github.com/KongyueX/GeoPRR-Net-Paper.git ..\GeoPRR-Net-Paper
.\.venv\Scripts\python.exe ..\GeoPRR-Net-Paper\figures\build_geoprr_figures.py
~~~

The completed VDN aggregate is stored as
[<code>figures/vdn_supplement.csv</code>](https://github.com/KongyueX/GeoPRR-Net-Paper/blob/main/figures/vdn_supplement.csv)
in the companion repository.

## Verification and release audit

The repository separates lightweight code verification from full experiment
reproduction:

| Check | Command or status |
|---|---|
| Public API and structural behavior | Focused CPU-safe pytest suite |
| CLI construction | Entry points support <code>--help</code> without opening datasets |
| Dependency and release surface | <code>experiments/check_public_release.py</code> |
| Full three-seed GPU training | Requires local data and checkpoints; not run by unit tests |
| Industrial-1395 replay | Restricted-data verification only |

Audit the current working tree:

~~~powershell
.\.venv\Scripts\python.exe experiments\check_public_release.py
~~~

Before a public commit or release, use the stricter declared-surface check:

~~~powershell
.\.venv\Scripts\python.exe experiments\check_public_release.py --mode release
~~~

The declared release surface is
[<code>experiments/public_release_inventory.json</code>](experiments/public_release_inventory.json).
The audit checks dependency closure, pinned distributions, ignored private or
generated artifacts, broken local links, and accidental credential-like
content. It does not certify the scientific claims or replace an experiment
rerun.

## Limitations

- GeoPRR-Net assumes a localized single-pointer ROI and known ordered endpoints.
- The public artifact does not include an end-to-end detector, OCR stack, or
  automatic range estimator.
- RF100-VL targets and VDN progress conversion use annotation-derived geometry.
- Industrial-1395 is not externally reproducible because redistribution
  permission has not been granted.
- Model weights and third-party initialization files are not redistributed.
- The validated formal environment is Windows/CUDA; other platforms may require
  adaptation and should report their environment separately.

## Citation

If this code contributes to published work, cite the accompanying GeoPRR-Net
article. Final journal metadata has not yet been added; the repository provides
machine-readable provisional metadata in [<code>CITATION.cff</code>](CITATION.cff).
Please also record the repository revision used for an experiment.

## License and third-party materials

This repository currently has **no project-level software license**. Public
availability supports inspection and research reproducibility but does not, by
itself, grant permission to reuse or redistribute the code. Third-party code,
datasets, pretrained initializations, and externally reproduced weights remain
subject to their own terms.

Review
[<code>THIRD_PARTY_NOTICES.md</code>](THIRD_PARTY_NOTICES.md) before reuse,
redistribution, or artifact publication.
