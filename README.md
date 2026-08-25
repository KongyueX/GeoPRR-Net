# R²MT-Net

### An OCR-Enabled End-to-End Pointer-Gauge Reader with Multi-Risk Moment Transport

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Evidence](https://img.shields.io/badge/evidence-retrospective-orange.svg)](paper/r2mt_net_electronics_overleaf/manuscript.tex)

This repository contains the paper, implementation, replay code, aggregate
results, and figure source data for **R²MT-Net** (Representation-Conditioned
Multi-Risk Moment Transport Network). R²MT-Net is the single progress-estimation
core in an OCR-enabled image-to-physical-reading pipeline:

```text
full image -> meter localization -> ROI -> R²MT-Net normalized progress
                                      \-> OCR range endpoints
normalized progress + validated range endpoints -> physical reading
```

The OCR branch completes the physical-reading graph but is not the paper's main
method contribution. Its reported accuracy is conditional on the labeled frames
accepted by the existing range-validity checks.

## Why R²MT-Net

A single projective correction is insufficient because the residual errors are
heterogeneous: mean-error, tail-error, and severe perspective-plus-blur
objectives emphasize different samples. R²MT-Net addresses this with four
linked mechanisms:

1. A **shared dual-observation ResNet-18** processes the original and
   support-normalized ROI without introducing a second image encoder.
2. **Multi-scale relation transport** aligns stride-8/16 features and encodes
   signed discrepancy, absolute discrepancy, agreement, support, and
   homography evidence.
3. **Three risk-specialized moment heads** model mean, tail, and combined
   severe-condition residuals; a representation-conditioned router makes only
   a conservative adjustment to a validated fixed prior.
4. **One final moment-exact posterior transport** combines moment residuals
   before probability transport. Invalid relation evidence returns the raw
   posterior exactly.

The older `ReMST` name appears only in internal checkpoint protocols and
low-level compatibility modules. It denotes the relation-transport foundation,
not a second publication model.

## Main retrospective results

NMAE is reported as percentage of full scale (%FS), lower is better.

| Evidence set | R²MT-Net | Comparator | Scope |
|---|---:|---:|---|
| SyncG, all six conditions | **1.0517 ± 0.0415** | EfficientNet-B0: 1.3935 ± 0.0606 | 1,558 images, 14 held-out scenes, 3 fits |
| SyncG, projective pool | **1.3099 ± 0.0501** | EfficientNet-B0: 1.9722 ± 0.0852 | same roster |
| Industrial ROIs, all conditions | 20.2817 ± 3.1514 | Direct-ResNet18: 23.5771 ± 2.4504; EfficientNet-B0: **15.2550 ± 2.3430** | paired reduction vs Direct 3.2954 pp, 95% CI 2.0864–4.4153 |
| Industrial ROIs, projective pool | 21.5524 ± 3.7915 | Direct-ResNet18: 28.1866 ± 2.7176; EfficientNet-B0: **18.0914 ± 2.9211** | paired reduction vs Direct 6.6342 pp, 95% CI 4.3215–8.8624 |
| Same-pixel VDN intersection, all | **0.9054 ± 0.0392** | annotation-assisted VDN: 1.6444 | 129 samples, 774 rows |
| Accepted OCR outputs | **Acc@5%FS: 74.07 ± 12.83%** | -- | 9 accepted labeled frames |

Model-only RTX 4060 BF16 latency is 22.32 ms/sample at batch 1 and
3.57 ms/sample at batch 8. Localization, support-normalization materialization,
OCR, and image decoding are excluded from this timing.

The industrial replay supports projective improvement over the matched
Direct-ResNet18 endpoint, while EfficientNet-B0 transfers better in aggregate;
it does not establish universal cross-backbone superiority. These are
retrospective development results. An untouched multi-site real-photo cohort is
required for a strong generalization claim. Backbone-controlled R²MT-Net
variants are the next mechanism test, and a live detector-reader-OCR replay is
required if automatic coverage or end-to-end latency is claimed.

## Public code surface

```text
r2mt/
  model.py                              stable R²MT-Net identity and loader
experiments/
  r2mt_net.py                          final risk-transport architecture
  train_r2mt_net.py                    final checkpoint training/loading
  evaluate_r2mt_net_syncg.py           six-condition SyncG evaluation
  run_r2mt_downstream_replays.py       field, VDN, OCR, repeat, efficiency replay
  summarize_r2mt_ablations.py          ablation aggregation
  r2mt_paired_statistics.py            paired cluster-bootstrap statistics
paper/r2mt_net_electronics_overleaf/
  manuscript.tex                       Electronics manuscript source
  figures/                              final figures and source CSVs
results/
  r2mt_net_tables.json                 compact machine-readable results
  r2mt_industrial_multimethod.json     four-model industrial summary and CIs
```

Release entrypoints and manifests are tracked at:

- `r2mt/__init__.py`
- `r2mt/model.py`
- `experiments/r2mt_net.py`
- `experiments/train_r2mt_net.py`
- `experiments/evaluate_r2mt_net_syncg.py`
- `experiments/run_r2mt_downstream_replays.py`
- `experiments/summarize_r2mt_ablations.py`
- `experiments/r2mt_paired_statistics.py`
- `results/README.md`
- `results/r2mt_net_tables.json`
- `results/r2mt_industrial_multimethod.json`
- `experiments/public_release_inventory.json`

Load a fitted publication checkpoint with the conservative settings used by the
paper:

```python
from r2mt import load_r2mt_net

anchor, model, metadata = load_r2mt_net(
    "path/to/r2mt_checkpoint.pt",
    device="cuda:0",
)
```

The loader deliberately accepts legacy checkpoint keys behind the API boundary
and returns an R²MT-Net publication identity. Model weights and private field
images are not redistributed.

## Environment and validation

The research environment is pinned in
[`experiments/requirements-training.lock.txt`](experiments/requirements-training.lock.txt).

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
python -m unittest test.test_r2mt_public_api test.test_r2mt_release_tables
python experiments/check_public_release.py --mode workspace
```

The manuscript is intended for Overleaf; this repository does not require a
locally compiled paper PDF. See
[`docs/R2MT_NET_ARCHITECTURE_CN.md`](docs/R2MT_NET_ARCHITECTURE_CN.md) for the
method boundary and
[`paper/r2mt_net_electronics_overleaf/R2MT_EXPERIMENT_GAP_AUDIT_CN.md`](paper/r2mt_net_electronics_overleaf/R2MT_EXPERIMENT_GAP_AUDIT_CN.md)
for the remaining experiment audit.
