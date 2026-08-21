# ReMSTNet

### Relation-Encoded Moment-Exact Transport for Pointer-Gauge Reading under Projective Distortion

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Release](https://img.shields.io/badge/release-paper%20%2B%20code%20%2B%20tables-2E7D32.svg)](paper/remstnet_electronics_overleaf/remstnet_electronics_manuscript.pdf)

This repository contains the current manuscript, paper-facing implementation,
evaluation code and reported tables for **ReMSTNet-v3**, a compact
relation-moment network for predicting normalized pointer progress from a
cropped, single-pointer gauge ROI. The final architecture identifier is:

```text
ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3
```

> [!IMPORTANT]
> ReMSTNet operates on cropped ROIs and predicts progress in `[0, 1]`. It is not
> an end-to-end meter detector, printed-scale recognizer or physical-unit
> reader. The physical-reading experiment is a separately reported diagnostic.
> This release contains the current manuscript and its compile assets, the final
> ReMSTNet-v3 paper model, the code required to fit and evaluate it, and the
> reported tables.

## Highlights

- **Relation-encoded dual-view backbone.** Raw and support-normalized (SARN)
  views share one frozen EfficientNet-B0 foundation; ReMST blocks at strides 8
  and 16 write bounded relation residuals into the raw feature stream.
- **Moment-exact posterior transport.** Each stage projects the progress
  posterior onto a feasible first-moment target within FP32 numerical
  tolerance.
- **Adaptive, coordinated correction.** The v3 coordinator combines local
  progress mixing with a learned bounded moment budget across the two spatial
  stages and the context stage.
- **Explicit conservative fallback.** If the required geometric relation is
  unavailable, the output posterior is exactly the raw-view posterior.
- **Small fitted surface.** The executable model has 4,271,638 unique
  parameters; only 261,528 are trainable during the ReMST fit and 4,010,110
  foundation parameters remain frozen.

## Abstract

Projective distortion changes the appearance and geometry of analog gauges in
ways that a raw cropped-image regressor may not represent reliably. ReMSTNet
uses paired raw and support-normalized ROI views, aligns their intermediate
features, encodes cross-view relations and converts bounded relation evidence
into sequential first-moment updates of a progress posterior. The final v3
model adds progress-aware local mixing and an adaptive correction budget while
retaining an exact raw-posterior fallback when geometric support is missing.
On the retrospective SyncG Scene-Holdout benchmark, three independently fitted
models reduce all-condition NMAE by 23.19% and projective-pooled NMAE by 32.77%
relative to the SARN-v2 + EfficientNet-B0 endpoint. The organized industrial
real-photo baseline contains 1,395 scored ROIs from 52 groups. Across its six
controlled conditions, ReMSTNet obtains 0.123388 NMAE versus 0.123664 for B0
(paired delta -0.000276, 95% CI [-0.000382, -0.000168]); all three projective
condition intervals are below zero. On the full 774-row VDN intersection,
ReMSTNet obtains 0.008582 versus 0.016444 NMAE for the annotation-assisted VDN
reference, a 47.81% reduction. RPM-10K is retained only as a raw-foundation
cross-domain diagnostic because that protocol disables relation inputs by
construction. A separate replay of 153 unique industrial full frames obtains
98.69% detector/read coverage and $0.051930\pm0.001031$ NMAE on 33 labeled
frames. These results support selective projective transport on synthetic and
industrial photographs while preserving the stated deployment boundary.

## Method at a glance

```text
 Raw ROI ───────────────┐
                        ├─ shared frozen encoder ─ stride-8 ─ stride-16 ─ context
 SARN ROI + support ────┘             │             │             │
                                      ▼             ▼             ▼
                                ReMST block 8  ReMST block 16  coordinator
                                      └─────────────┴─────────────┘
                                                    │
                                     bounded moment allocations
                                                    │
                                  sequential information projections
                                                    │
                                  normalized progress posterior / mean

 Missing relation geometry ───────────────────────► exact raw-posterior fallback
```

The validated implementation and stable public API are in
[`remstnet/model.py`](remstnet/model.py) and
[`remstnet/__init__.py`](remstnet/__init__.py). Detailed architecture and
evidence notes are available in
[`docs/REMSTNET_ARCHITECTURE_CN.md`](docs/REMSTNET_ARCHITECTURE_CN.md).

## Main results

The metric is normalized mean absolute error,
`NMAE = mean(|predicted_progress - target_progress|)`; lower is better. Values
below are mean ± sample standard deviation across three independently fitted
models on 1,558 images from 14 held-out SyncG scene groups.

| Method | Clean | Perspective 25° | Perspective 45° | 45° + blur | All | Projective pooled |
|---|---:|---:|---:|---:|---:|---:|
| SARN-v2 + Direct | 0.007910 ± 0.000411 | 0.014422 ± 0.000343 | 0.024302 ± 0.000398 | 0.026219 ± 0.000564 | 0.014832 ± 0.000346 | 0.021648 ± 0.000435 |
| SARN-v2 + EfficientNet-B0 | 0.008099 ± 0.000323 | 0.014177 ± 0.000607 | 0.021578 ± 0.000968 | 0.023411 ± 0.000995 | 0.013935 ± 0.000606 | 0.019722 ± 0.000852 |
| SARN-v2 + MobileNetV3-Large | 0.009355 ± 0.000435 | 0.016255 ± 0.000489 | 0.025584 ± 0.001962 | 0.027589 ± 0.001664 | 0.016335 ± 0.000716 | 0.023143 ± 0.001361 |
| **ReMSTNet-v3** | **0.008099 ± 0.000323** | **0.010330 ± 0.000476** | **0.013884 ± 0.000441** | **0.015564 ± 0.000299** | **0.010704 ± 0.000373** | **0.013259 ± 0.000391** |

The clean and blur-only endpoints are preserved; the main gain is concentrated
in projective conditions.

### Secondary VDN annotation-assisted comparison

The ReMSTNet and VDN experiments used different pre-existing holdout rosters,
so their full-cohort aggregates cannot be placed in one denominator. The
released comparison uses only their prediction-independent intersection: 129
SyncG samples from 6 ReMSTNet scene groups, each under the same six condition
pixels (774 rows). VDN's predicted direction is converted to normalized
progress with the annotation-derived pivot and ordered scale endpoints, so all
774 rows are retained. ReMSTNet and B0 predict progress directly and do not
receive those annotations.

| Scope | Compared rows (coverage) | ReMSTNet-v3 NMAE | SARN-v2+B0 NMAE | VDN annotation-reference NMAE | ReMSTNet−VDN Δ (95% CI) |
|---|---:|---:|---:|---:|---:|
| All conditions | 774 / 774 (100%) | **0.008582 ± 0.000414** | 0.011673 ± 0.000638 | 0.016444 | -0.007861 [-0.012340, -0.005530] |
| Projective pooled | 387 / 387 (100%) | **0.010938 ± 0.000935** | 0.017119 ± 0.001396 | 0.021030 | -0.010092 [-0.014298, -0.006818] |

The VDN checkpoint is a local SyncG retraining under the upstream 200-epoch
architecture/configuration, not an official pretrained end-to-end result.
The annotation reference makes this a full-coverage direction-component
comparison, not an input-equivalent deployable-system comparison. The
intersection contains only six scene groups and one VDN checkpoint, so it
remains supporting evidence and does not replace the 1,558-sample main table.

### Public external and industrial checks

- **RPM-10K:** on the 1,797-image official single-pointer test subset, frozen
  ReMSTNet has 98.78% detector/read coverage and full-denominator NMAE
  **0.233945 ± 0.012331**. RPM-10K includes 474 blur-tagged and 993
  tilted-tagged images (157 have both), but this evaluator supplies identical
  raw/normalized images, unit support, an inactive relation mask and identity
  homography. Equality with B0 is therefore imposed by the protocol; this is a
  raw-foundation transfer diagnostic, not a relation-module comparison.
- **Industrial Real-Photo Baseline:**
  `C:/pointer_read/unified_real_photo_progress_v1` contains 1,395 labeled ROIs
  from 52 groups. Its decoded-pixel set exactly equals the union of the three
  existing FieldGauge evaluation sources. ReMSTNet beats B0 on the pooled
  projective conditions (0.122590 vs 0.123142; paired delta -0.000552, 95% CI
  [-0.000761, -0.000335]), and every individual projective-condition interval
  is below zero.
- **Industrial full-frame scalar diagnostic:** the same organized dataset
  retains 153 unique deployment photographs derived from repository `data/` and
  `data-717/`. Cached YOLO detections plus the frozen readers cover **151/153
  (98.69%)**; all 33 scalar-labeled frames pass and yield NMAE **0.051930 ±
  0.001031**. The other 120 frames contribute to coverage only. This is a
  detector replay, not a detector recall/IoU benchmark.
- **Independent existing-model OCR replay:** PP-OCRv4 through RapidOCR plus the
  existing range decoder produces physical readings for **27/153 (17.65%)**
  deployment photographs and **9/33 (27.27%)** labeled photographs. Conditional
  NMAE on those nine is **0.036814 ± 0.004015**, versus **0.034336 ± 0.003848**
  with the true ranges on the identical subset. Full-denominator NMAE is
  **0.737313 ± 0.001095**, making OCR/range coverage the limiting stage. OCR
  choice and thresholds were fixed before opening field labels; cached boxes
  exclude live detector latency. The replay entrypoint is
  `experiments/evaluate_remstnet_ocr_end_to_end.py`; its deterministic paper
  asset exporter is `experiments/export_remstnet_ocr_paper_assets.py`.

Full condition tables, real-source results, factorial ablations, perspective
stress results, repeat stability and runtime measurements are in:

- [current compiled manuscript](paper/remstnet_electronics_overleaf/remstnet_electronics_manuscript.pdf)
- [LaTeX manuscript source](paper/remstnet_electronics_overleaf/manuscript.tex)
- [human-readable paper tables](results/README.md)
- [machine-readable paper tables](results/remstnet_v3_tables.json)
- [OCR-complete deployment aggregate](results/remstnet_ocr_end_to_end_deployment.json)

## Repository structure

```text
remstnet/
  __init__.py                              stable ReMSTNet-v3 import surface
  model.py                                 final architecture and initialization
experiments/
  train_remstnet.py                         final fit and mechanism ablations
  evaluate_remstnet_syncg.py                SyncG condition evaluation
  evaluate_remstnet_real_domains.py        real-source ROI evaluation
  evaluate_remstnet_stress_sweep.py        controlled stress sweep
  evaluate_remstnet_natural_repeat_stability.py
  evaluate_remstnet_clean_external.py      RPM-10K and field full-frame replay
  evaluate_remstnet_ocr_end_to_end.py      independent OCR-to-physical-reading replay
  export_remstnet_ocr_paper_assets.py      OCR figure source-data exporter
  summarize_remstnet_multiseed.py          three-fit aggregation
  summarize_remstnet_ablations.py          factorial mechanism analysis
  summarize_remstnet_real_domains.py       grouped real-source analysis
  summarize_remstnet_vdn_intersection.py   same-pixel VDN comparison
  benchmark_remstnet_efficiency.py         model-level GPU benchmark
  requirements-training.lock.txt           pinned research environment
results/
  README.md                                publication tables and caveats
  remstnet_v3_tables.json                  machine-readable values
  remstnet_ocr_end_to_end_deployment.json  OCR deployment aggregate
paper/remstnet_electronics_overleaf/
  manuscript.tex                          current LaTeX manuscript
  remstnet_electronics_manuscript.pdf      compiled author copy
  references.bib                          manuscript bibliography
  Definitions/                            MDPI LaTeX dependencies
  figures/                                referenced figures in PDF and PNG
test/
  test_remstnet_model.py                    model invariants and fallback
  test_remstnet_extended_evaluators.py     evaluator coverage
  test_remstnet_public_api.py              stable import surface
  test_remstnet_release_tables.py          table consistency
  test_summarize_remstnet_vdn_intersection.py
  test_evaluate_remstnet_clean_external.py
  test_evaluate_remstnet_ocr_end_to_end.py
THIRD_PARTY_NOTICES.md                     dependency/data provenance notes
```

Primary executable entrypoints are the
[trainer](experiments/train_remstnet.py),
[SyncG evaluator](experiments/evaluate_remstnet_syncg.py),
[real-source evaluator](experiments/evaluate_remstnet_real_domains.py),
[stress evaluator](experiments/evaluate_remstnet_stress_sweep.py),
[natural-repeat evaluator](experiments/evaluate_remstnet_natural_repeat_stability.py),
[single-view external/full-frame diagnostic evaluator](experiments/evaluate_remstnet_clean_external.py),
[multi-fit summarizer](experiments/summarize_remstnet_multiseed.py),
[VDN intersection summarizer](experiments/summarize_remstnet_vdn_intersection.py) and
[efficiency benchmark](experiments/benchmark_remstnet_efficiency.py).

## Installation

The paper workflow was validated with Python 3.11, CUDA 12.8 and PyTorch
2.11.0+cu128 on Windows with an NVIDIA GeForce RTX 4060. A CUDA-capable machine
is recommended for fitting and full evaluation; architecture and unit smoke
tests also run on CPU.

Using [`uv`](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
```

The lock file records the validated environment. If the exact CUDA wheel index
is unavailable on another platform, install the platform-appropriate PyTorch
build first and then install the remaining pinned dependencies.

## Quick architecture smoke test

This constructs the final architecture with unfitted weights; it does not
reproduce the reported predictions.

```powershell
.venv/Scripts/python.exe -c "from remstnet import ARCHITECTURE_ID, build_remstnet_v3, remstnet_parameter_counts; m = build_remstnet_v3(); print(ARCHITECTURE_ID); print(remstnet_parameter_counts(m))"
```

Expected parameter inventory:

```text
shared foundation: 4,010,110
trainable ReMST surface: 261,528
total unique: 4,271,638
```

## Data and checkpoint preparation

### SyncG

The synthetic source is the public
[SyncG dataset at its pinned revision](https://huggingface.co/datasets/YihengDeng/syncG/tree/44a0ea1c8e447f5cb4cbe3f46c131da8a44400a2)
([dataset DOI](https://doi.org/10.57967/hf/8201), CC BY 4.0). The companion
dataset paper is
[Deng et al., *Scientific Data* (2026)](https://doi.org/10.1038/s41597-026-07308-x).
Keep downloaded images and generated manifests outside Git; dataset and artifact
directories are intentionally ignored.

The paper protocol uses:

| Split | Images | Scene groups | Use |
|---|---:|---:|---|
| SyncG Fit | 14,442 | 131 | foundation fitting |
| Correction subset | 6,616 | 60 | ReMST modules only |
| Scene-Holdout | 1,558 | 14 | retrospective same-domain benchmark |
| Industrial Real-Photo Baseline | 1,395 | 52 | retrospective in-house real-photo ROI benchmark |
| RPM-10K single-pointer test | 1,797 | 6 meter types | public zero-shot scalar transfer |
| Industrial Full-Frame Diagnostic | 153 | 153 unique photos | detector-to-reading replay; 33 labeled |

### Real-source cohorts

The organized industrial benchmark is
`C:/pointer_read/unified_real_photo_progress_v1`: 1,395 labeled real-photo ROIs
from 52 groups, plus a separate 153-photo full-frame diagnostic track. Its ROI
pixel set exactly matches the union of FieldGauge-ROI Test-A, Test-B and
External-ROI (434 + 814 + 147). The raw photos originate from the repository's
`data/` and `data-717/` collections. These assets and Natural-repeat are not
redistributed in the paper release because their ownership and consent
boundaries are not uniformly public. RPM-10K and RF100-derived inputs must be
acquired under their source terms. The code accepts caller-supplied manifests
and prediction roots; see the respective evaluator `--help` output.

The external source is [RPM-10K / DialBench](https://github.com/Event-AHU/DialBench),
whose repository currently marks the dataset license as TBD. It is not
redistributed here.

### Checkpoints

Trained `.pt` files are intentionally excluded from this code-only release.
Reproduction therefore starts from a locally trained direct-reader foundation
checkpoint, followed by ReMST fitting. This avoids implying redistribution
rights for upstream weights or private data-derived artifacts. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before supplying external
assets.

## Reproducing ReMSTNet-v3

All commands are run from the repository root.

### 1. Fit the final architecture

```powershell
.venv/Scripts/python.exe -m experiments.train_remstnet `
  --correction-train-manifest <correction_train_manifest.json> `
  --direct-checkpoint <direct_foundation.pt> `
  --architecture-variant adaptive_budget_progress_mixing_v3 `
  --initialization-seed 20262213 `
  --sample-order-seed 20262217 `
  --output <remstnet_v3_seed1.pt> `
  --device cuda:0
```

The three reported fits pair source/foundation seeds 20262020–20262022 with
ReMST initialization seeds 20262213–20262215 and sample-order seeds
20262217–20262219, respectively. Use a separate output path for each fit.

### 2. Evaluate SyncG conditions

```powershell
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_syncg `
  --checkpoint <remstnet_v3_seed1.pt> `
  --manifest <scene_holdout_manifest.json> `
  --reference <reference_evaluation.json> `
  --output <seed1_syncg_evaluation.json> `
  --device cuda:0 `
  --amp
```

Repeat for all three fitted checkpoints, then aggregate:

```powershell
.venv/Scripts/python.exe -m experiments.summarize_remstnet_multiseed `
  --evaluation <seed1_syncg_evaluation.json> `
  --evaluation <seed2_syncg_evaluation.json> `
  --evaluation <seed3_syncg_evaluation.json> `
  --output <remstnet_v3_multiseed_summary.json>
```

To reproduce the secondary VDN table from the three ReMSTNet evaluations and
the precomputed VDN ledgers:

```powershell
.venv/Scripts/python.exe -m experiments.summarize_remstnet_vdn_intersection `
  --evaluation <seed1_syncg_evaluation.json> `
  --evaluation <seed2_syncg_evaluation.json> `
  --evaluation <seed3_syncg_evaluation.json> `
  --vdn-automatic <vdn_automatic_reference_predictions.jsonl> `
  --vdn-oracle <vdn_annotation_reference_predictions.jsonl> `
  --output <remstnet_vdn_intersection.json>
```

The utility forms the complete sample-by-condition intersection and verifies
that the existing ledgers refer to identical condition pixels. The paper-facing
table uses the annotation-reference conversion on every fixed-intersection
row. Automatic-reference and success-conditioned fields remain available only
as machine-readable stage diagnostics.

### 3. Run the supporting evaluations

```powershell
# Controlled transformations of real-source ROIs
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_real_domains `
  --checkpoint <remstnet_v3_seed1.pt> `
  --prediction-root <prepared_prediction_root> `
  --output <seed1_real_domains.json> `
  --device cuda:0

# Controlled perspective/degradation sweep
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_stress_sweep `
  --checkpoint <remstnet_v3_seed1.pt> `
  --manifest <stress_manifest.json> `
  --labels <stress_labels.json> `
  --split <stress_split.json> `
  --output <seed1_stress.json> `
  --device cuda:0 `
  --amp

# Natural-repeat preservation audit (three checkpoints together)
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_natural_repeat_stability `
  --checkpoint <remstnet_v3_seed1.pt> `
  --checkpoint <remstnet_v3_seed2.pt> `
  --checkpoint <remstnet_v3_seed3.pt> `
  --manifest <repeat_manifest.json> `
  --provenance <repeat_provenance.json> `
  --labels <repeat_labels.json> `
  --physical-capture-manifest <physical_capture_manifest.json> `
  --output <natural_repeat_summary.json> `
  --device cuda:0 `
  --amp
```

The two single-view diagnostic tracks use all three checkpoints together:

```powershell
# RPM-10K public scalar-reading transfer
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_clean_external `
  --dataset rpm10k `
  --checkpoint <remstnet_v3_seed1.pt> `
  --checkpoint <remstnet_v3_seed2.pt> `
  --checkpoint <remstnet_v3_seed3.pt> `
  --rpm-manifest <rpm10k_manifest.jsonl> `
  --rpm-materialized-manifest <rpm10k_detector_crops.jsonl> `
  --rpm-detector-sidecar <rpm10k_detector_sidecar.jsonl> `
  --output <rpm10k_clean_external.json> `
  --device cuda:0

# Organized industrial full-frame diagnostic
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_clean_external `
  --dataset field_full_frame `
  --checkpoint <remstnet_v3_seed1.pt> `
  --checkpoint <remstnet_v3_seed2.pt> `
  --checkpoint <remstnet_v3_seed3.pt> `
  --field-root <deduplicated_field_root> `
  --field-labels <full_frame_labels.jsonl> `
  --field-labeled-detections <labeled_detector_rows.jsonl> `
  --field-unlabeled-detections <unlabeled_detector_rows.jsonl> `
  --output <field_full_frame_clean_external.json> `
  --device cuda:0

# Label-free OCR-to-physical-reading prediction replay
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_ocr_end_to_end predict `
  --output-root <ocr_prediction_root> `
  --device cuda:0 `
  --ocr-view-mode original

# Open labels only after prediction and compute the paper metrics
.venv/Scripts/python.exe -m experiments.evaluate_remstnet_ocr_end_to_end score `
  --prediction-root <ocr_prediction_root> `
  --labels <full_frame_labels.jsonl> `
  --output <ocr_score.json>
```

For every script, use `--help` to inspect the complete argument surface. The
real-domain and ablation aggregators are
[`experiments/summarize_remstnet_real_domains.py`](experiments/summarize_remstnet_real_domains.py)
and [`experiments/summarize_remstnet_ablations.py`](experiments/summarize_remstnet_ablations.py).

### 4. Benchmark model-level efficiency

```powershell
.venv/Scripts/python.exe -m experiments.benchmark_remstnet_efficiency `
  --arm full_remstnet `
  --checkpoint <remstnet_v3_seed1.pt> `
  --correction-train-manifest <correction_train_manifest.json> `
  --output-json <full_remstnet_efficiency.json> `
  --output-markdown <full_remstnet_efficiency.md> `
  --device cuda:0
```

Available arms are `raw_foundation`, `twin_endpoint` and `full_remstnet`. The
benchmark excludes image decoding and SARN materialization, so its timings are
not end-to-end API latency.

## Verification

Run the focused paper-model suite:

```powershell
.venv/Scripts/python.exe -m unittest `
  test.test_remstnet_model `
  test.test_remstnet_extended_evaluators `
  test.test_summarize_remstnet_multiseed `
  test.test_summarize_remstnet_ablations `
  test.test_summarize_remstnet_real_domains `
  test.test_summarize_remstnet_vdn_intersection `
  test.test_evaluate_remstnet_clean_external `
  test.test_remstnet_public_api `
  test.test_remstnet_release_tables
```

Audit the declared public release surface without changing Git state:

```powershell
.venv/Scripts/python.exe -m experiments.check_public_release --mode workspace
```

The release inventory is
[`experiments/public_release_inventory.json`](experiments/public_release_inventory.json),
and the read-only audit implementation is
[`experiments/check_public_release.py`](experiments/check_public_release.py).

## Statistical reporting and interpretation

- Main SyncG entries are arithmetic mean ± sample SD across three independent
  fitted models, not confidence intervals.
- The VDN table covers all 774 rows in the fixed intersection by using
  annotation-derived pivot and ordered scale endpoints for offline conversion
  of VDN direction. It is a component comparison, not an input-equivalent
  automatic-system result, and VDN uses one terminal checkpoint.
- RPM-10K includes blur and tilted views, but the current evaluator disables
  relation inputs; it cannot establish either a ReMST gain or a lack of useful
  degradation.
- The industrial ROI benchmark contains 1,395 scored real-photo ROIs. Its
  controlled projective results are distinct from the 153-photo full-frame
  diagnostic, which uses cached detections and only 33 scalar labels; without
  bounding-box truth it does not estimate detector recall or IoU.
- Real-source paired CIs use 20,000 group-bootstrap replicates after rowwise
  averaging across the three seeds. They do not include retraining uncertainty.
- The factorial mechanism table uses one fixed source fit (seed 20262022), so it
  isolates the measured interaction but does not estimate training-seed
  uncertainty.
- “Moment-exact” refers to feasible target equality within FP32 numerical
  tolerance. Near 0 and 1, clipping can shorten a requested displacement but
  cannot reverse it.
- ImageNet initialization is part of the foundation provenance; task-specific
  fitting uses SyncG.
- Scene-Holdout is scene-disjoint from fitting but was reused during historical
  model comparison. It is retrospective evidence, not untouched confirmation.
- The natural-repeat result is preservation, not improvement: ReMSTNet, Raw and
  SARN are identical on that cohort.

## Citation

The current manuscript is available in this repository, but a permanent paper
identifier has not yet been finalized. Until the archival record is available,
cite the manuscript title and this repository revision:

```bibtex
@misc{remstnet2026,
  title        = {ReMSTNet: Relation-Encoded Moment-Exact Transport for
                  Pointer-Gauge Reading under Projective Distortion},
  year         = {2026},
  howpublished = {Software and results release},
  note         = {Manuscript; cite the exact repository revision used}
}
```

## License and third-party material

No repository-level license has been granted at this time. Public visibility
does not itself grant permission to copy, modify or redistribute the source.
Dataset, pretrained-weight and third-party-code terms remain those of their
respective owners; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
Before broader reuse, the repository owner should add an explicit license that
is compatible with all included material.

## Release status

The current manuscript source, compiled author copy, implementation and reported
tables are available as a transparent paper release. A final archival
publication still requires the authors to confirm the definitive
author/affiliation list, CRediT roles, funding and conflict statements,
AI-assistance disclosure as applicable, a permanent repository identifier/DOI,
and rights for any non-public FieldGauge assets.
