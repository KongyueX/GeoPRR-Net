# GeoPRR-Net reproducibility guide

This document defines the public reproduction surface for the *Electronics*
manuscript. It distinguishes model training, frozen evaluation, figure
generation, and the supporting VDN comparison so that each result can be
replayed without exposing private field images or local checkpoints.

## 1. Scientific scope

GeoPRR-Net predicts normalized progress in `[0, 1]` from a localized
single-pointer gauge ROI with known scale endpoints. The released experiment
code does not turn this core into a full-image detector, OCR system, or automatic
range-discovery system. RF100-VL targets and the VDN progress conversion use
annotation-derived geometry and are identified as such in the manuscript.

## 2. Environment

The validated environment is Windows, Python 3.11, PyTorch 2.11, and CUDA 12.8.
Create it from the checked-in lock file:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe `
  -r experiments/requirements-training.lock.txt
```

CPU-only installations can run the focused unit tests, but formal training and
reported latency measurements require a compatible CUDA GPU.

## 3. Data boundaries

| Cohort | Public source | Repository contents |
|---|---|---|
| SyncG | `https://doi.org/10.1038/s41597-026-07308-x` | no images; scripts expect a local manifest |
| RF100-VL gauge subset | `https://universe.roboflow.com/rf100-vl/needle-base-tip-min-max-u87vi-wzsrt-kjyu` | one licensed illustrative ROI plus aggregate outputs |
| Industrial-1395 | restricted field-photo collections | aggregate statistics only; no images or per-sample ledger |

All manifests, images, checkpoints, JSONL predictions, and run logs are ignored
by Git. Paths supplied to the scripts are local inputs, not download promises.

## 4. GeoPRR-Net entry points

The stable import API is `geoprr.GeoPRRNet` together with
`geoprr.load_geoprr_net`. GeoPRR-Net is the only supported public model; older
module and checkpoint names below are retained strictly as internal provenance
and replay identifiers. The implementation and reproduction scripts are:

- `experiments/unified_pointer_reader.py`
- `experiments/train_unified_pointer_reader.py`
- `experiments/evaluate_unified_pointer_reader_syncg.py`
- `experiments/evaluate_unified_pointer_reader_rf100.py`
- `experiments/evaluate_unified_pointer_reader_industrial.py`
- `experiments/benchmark_unified_pointer_reader_paper_efficiency.py`
- `experiments/analyze_unified_pointer_reader_routing.py`
- `experiments/summarize_unified_pointer_reader_experiments.py`
- `experiments/summarize_unified_pointer_reader_rf100.py`

Inspect the complete CLI of any stage before running it:

```powershell
.\.venv\Scripts\python.exe -m experiments.train_unified_pointer_reader --help
.\.venv\Scripts\python.exe -m experiments.evaluate_unified_pointer_reader_syncg --help
.\.venv\Scripts\python.exe -m experiments.evaluate_unified_pointer_reader_rf100 --help
```

The trainer requires a locally reproduced source expert-bank checkpoint and a
warm polar checkpoint. Their paths are explicit inputs; neither weight file is
redistributed:

The `--source-r2mt` option name is a checkpoint-compatibility field. It does not
expose R2MT as a separate supported model in this repository.

```powershell
.\.venv\Scripts\python.exe -m experiments.train_unified_pointer_reader `
  --source-r2mt <path-to-source-expert-bank.pt> `
  --warm-polar <path-to-warm-polar.pt> `
  --fit-manifest <path-to-syncg-train-manifest.jsonl> `
  --outer-split <path-to-scene-disjoint-split.json> `
  --output-root artifacts\runs\unified_pointer_reader `
  --device cuda:0 `
  --seed 20262020
```

Repeat with seeds `20262021` and `20262022`. Do not select a seed using the
formal holdout.

Evaluate a frozen checkpoint on the formal SyncG holdout:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_unified_pointer_reader_syncg `
  --checkpoint <path-to-geoprr-checkpoint.pt> `
  --output <path-to-syncg-output.json> `
  --device cuda:0 `
  --amp
```

The RF100-VL and Industrial entry points use fixed local cohort definitions in
the underlying evaluation modules. Industrial-1395 cannot be reproduced by an
external user without data-owner permission.

## 5. Matched VDN comparison

The VDN comparison uses the upstream GPL-3.0 implementation without copying it
into this repository:

```powershell
git clone https://github.com/DrawZeroPoint/VectorDetectionNetwork.git `
  artifacts\vendor\VectorDetectionNetwork
git -C artifacts\vendor\VectorDetectionNetwork checkout `
  68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb
```

The formal matched run uses seeds `20262020`, `20262021`, and `20262022`, image
size 384, batch size 8, and exactly 200 epochs. At 200 epochs the learning-rate
milestones are 140 and 190. The VDN direction is converted offline to normalized
progress using the annotated pivot and ordered scale endpoints, so this is an
annotation-assisted component comparison rather than a deployable end-to-end
VDN system.

Fresh run:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_geoprr_vdn_matched.ps1 `
  -Epochs 200 `
  -OuterSplit <path-to-scene-disjoint-split.json> `
  -VdnSource artifacts\vendor\VectorDetectionNetwork `
  -RoiManifest <path-to-conditioned-roi-manifest.jsonl> `
  -PixelReference <path-to-geoprr-pixel-reference.json>
```

Resume after an interruption:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoLogo -NoProfile `
  -File experiments\run_geoprr_vdn_matched.ps1 `
  -Resume `
  -Epochs 200 `
  -OuterSplit <path-to-scene-disjoint-split.json> `
  -VdnSource artifacts\vendor\VectorDetectionNetwork `
  -RoiManifest <path-to-conditioned-roi-manifest.jsonl> `
  -PixelReference <path-to-geoprr-pixel-reference.json>
```

Resume validates the checkpoint signature and restores the model, optimizer,
scheduler, scaler, and training history from `last.pt`. Existing completed
prediction ledgers are reused; missing seed directories start fresh. The wrapper
then evaluates each terminal epoch-200 checkpoint and produces a paired,
scene-bootstrap three-seed summary.

The formal run is complete. Each terminal checkpoint was evaluated on all 1,558
held-out images under the six prespecified conditions (9,348 rows per seed; 14
scene clusters). GeoPRR-Net obtained 1.0013 ± 0.0382%FS NMAE and 91.11 ± 0.79%
Acc@2%FS, while VDN obtained 1.6620 ± 0.0878%FS and 74.49 ± 2.19%. The paired
GeoPRR-Net-minus-VDN NMAE difference was -0.6607%FS (95% scene-bootstrap CI
-0.8251 to -0.4854; 20,000 replicates), corresponding to a 39.8% relative NMAE
reduction. The compact publication aggregate is
[`figures/vdn_supplement.csv`](https://github.com/KongyueX/GeoPRR-Net-Paper/blob/main/figures/vdn_supplement.csv)
in the companion paper repository.

## 6. Focused validation

```powershell
.\.venv\Scripts\python.exe -m pytest `
  test\test_geoprr_public_api.py `
  test\test_unified_pointer_reader.py `
  test\test_geoprr_figure3_experiments.py `
  test\test_run_vdn_oracle_reference_component.py
```

Validate that the PowerShell wrapper parses without launching training:

```powershell
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  (Resolve-Path experiments\run_geoprr_vdn_matched.ps1),
  [ref]$tokens,
  [ref]$errors
) | Out-Null
if ($errors.Count -ne 0) { $errors | Format-List; exit 1 }
```

## 7. Manuscript figures

Quantitative figures are generated from compact aggregate CSV files stored with
the manuscript in
[GeoPRR-Net-Paper](https://github.com/KongyueX/GeoPRR-Net-Paper). Clone that
repository alongside this code repository before running:

```powershell
git clone https://github.com/KongyueX/GeoPRR-Net-Paper.git ..\GeoPRR-Net-Paper
.\.venv\Scripts\python.exe `
  ..\GeoPRR-Net-Paper\figures\build_geoprr_figures.py
```

The paper source targets the bundled MDPI *Electronics* LaTeX class. Compile it
in Overleaf or another TeX environment and inspect undefined references,
overfull boxes, float placement, and the final author declarations before
submission.

## 8. Release limitations

- The repository does not redistribute fitted checkpoints.
- Industrial-1395 remains restricted.
- The VDN comparison is an annotation-assisted direction-component evaluation,
  not a deployable end-to-end VDN system comparison.
- The repository has no project-level license at this release point; see
  `THIRD_PARTY_NOTICES.md` before reuse or redistribution.
