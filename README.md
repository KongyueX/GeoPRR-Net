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

### 6. Rebuild manuscript figures

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
