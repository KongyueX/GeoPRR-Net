# GeoPRR-Net

### Geometry-Aware Polar-Relational Routing for Robust Analog Gauge Reading

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Manuscript](https://img.shields.io/badge/manuscript-MDPI%20Electronics-008A8A.svg)](paper/geoprr_net_electronics_overleaf/manuscript.tex)

This repository contains the manuscript source, model implementation,
experiment entry points, VDN comparison code, aggregate figure data, and
focused tests for **GeoPRR-Net**.

GeoPRR-Net estimates normalized pointer progress from a localized, single-pointer
analog-gauge ROI with known scale endpoints. It is not presented as a complete
full-image detector, OCR system, or automatic scale-range discovery pipeline.
Those modules can be connected outside the progress-estimation core and must be
evaluated separately.

## Method at a glance

GeoPRR-Net uses one shared EfficientNet-B0 encoder over raw and
support-normalized observations. It retains three candidate readings:

1. a geometry-aware base posterior;
2. polar evidence aligned with dial progression; and
3. relational moment transport between the two observations.

A conditional router assigns per-sample candidate weights. A final information
projection returns one posterior whose first moment matches the routed estimate.
The paper evaluates four executable interventions: removing geometry-aware
fusion, polar evidence, or relational transport, and replacing adaptive routing
with a fixed prior.

## Reported evidence

NMAE is reported as percentage of full scale (%FS); lower is better.

| Cohort | GeoPRR-Net | Comparison | Scope |
|---|---:|---:|---|
| SyncG scene-disjoint holdout | **1.0013 ± 0.0382** | best tested raw CNN: 2.0530 ± 0.0294 | 1,558 images, six conditions, three fits |
| RF100-VL public transfer | **5.0098 ± 1.5344** | best tested raw CNN: 6.4809 ± 1.2255 | 151 images, six conditions, three fits |
| Industrial-1395 | **18.9% lower NMAE** | best tested raw CNN | 1,395 field ROIs, six conditions, three fits |

The manuscript currently reports a bounded, annotation-assisted VDN component
comparison. A matched three-seed VDN run with a fixed 200-epoch budget is being
completed; its results must replace the provisional VDN row before any broader
comparison claim is made.

## Public code surface

```text
geoprr/
  model.py                                      stable model identity and loader
experiments/
  unified_pointer_reader.py                     GeoPRR-Net architecture
  train_unified_pointer_reader.py               staged model and ablation fitting
  evaluate_unified_pointer_reader_syncg.py       SyncG evaluation
  evaluate_unified_pointer_reader_rf100.py       public RF100-VL transfer
  evaluate_unified_pointer_reader_industrial.py  restricted field-cohort replay
  benchmark_unified_pointer_reader_paper_efficiency.py
  summarize_unified_pointer_reader_experiments.py
  train_vdn_syncg.py                            matched VDN training core
  evaluate_geoprr_vdn_matched.py                 same-pixel VDN evaluation
  summarize_geoprr_vdn_matched.py                three-seed paired summary
  run_geoprr_vdn_matched.ps1                     200-epoch run/resume wrapper
paper/geoprr_net_electronics_overleaf/
  manuscript.tex
  references.bib
  figures/                                      final figures and aggregate sources
```

The repository retains earlier research modules because the publication model
reuses their tested low-level geometry and expert-bank implementations. The
paper-facing API keeps those internal names behind one GeoPRR-Net identity.

## Environment

The validated Windows/CUDA environment is pinned in
[`experiments/requirements-training.lock.txt`](experiments/requirements-training.lock.txt).

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r experiments/requirements-training.lock.txt
```

Run the focused CPU-safe checks:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  test.test_geoprr_public_api `
  test.test_unified_pointer_reader `
  test.test_vdn_baseline `
  test.test_run_vdn_oracle_reference_component
```

## Loading a reproduced checkpoint

Weights are not redistributed in this repository. After reproducing the model
and its source expert-bank checkpoint:

```python
from geoprr import load_geoprr_net

model, metadata = load_geoprr_net(
    "path/to/geoprr_checkpoint.pt",
    device="cuda:0",
)
```

## VDN matched experiment

The comparison adapter uses the public GPL-3.0 VDN source at the pinned commit
recorded in [`experiments/vdn_baseline.py`](experiments/vdn_baseline.py). The
external checkout is loaded at runtime and is not copied into this repository.

The formal wrapper requires explicit paths to the frozen outer split, conditioned
ROI manifest, and same-pixel reference. A fresh run uses exactly three seeds and
defaults to 200 epochs:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_geoprr_vdn_matched.ps1 `
  -OuterSplit <path-to-outer-split.json> `
  -RoiManifest <path-to-conditioned-roi-manifest.jsonl> `
  -PixelReference <path-to-geoprr-pixel-reference.json>
```

Resume an interrupted seed from its validated `last.pt` epoch boundary:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_geoprr_vdn_matched.ps1 `
  -Resume `
  -OuterSplit <path-to-outer-split.json> `
  -RoiManifest <path-to-conditioned-roi-manifest.jsonl> `
  -PixelReference <path-to-geoprr-pixel-reference.json>
```

Existing seed directories are resumed; missing seed directories begin fresh.
Checkpoints, logs, per-sample ledgers, datasets, and private field images remain
ignored by Git.

## Manuscript and figures

The *Electronics* LaTeX source is
[`paper/geoprr_net_electronics_overleaf/manuscript.tex`](paper/geoprr_net_electronics_overleaf/manuscript.tex).
Rebuild the complete figure set from the included aggregate CSV files with:

```powershell
.\.venv\Scripts\python.exe `
  paper\geoprr_net_electronics_overleaf\figures\build_geoprr_figures.py
```

## Data and licensing boundaries

- SyncG and RF100-VL are referenced through their public sources; images are
  not redistributed here.
- Industrial-1395 is not public because redistribution permission has not been
  granted by the data owners.
- Model checkpoints and third-party weights are not redistributed.
- The repository currently has no project-level software license. Public access
  supports inspection and reproducibility but does not itself grant reuse or
  redistribution rights. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Detailed reproduction notes are available in
[`docs/GEOPRR_REPRODUCIBILITY.md`](docs/GEOPRR_REPRODUCIBILITY.md).
