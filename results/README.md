# R²MT-Net reported results

All accuracy values below are normalized mean absolute error in percentage of
full scale (%FS), lower is better. The main machine-readable summary is
[`r2mt_net_tables.json`](r2mt_net_tables.json), and the four-model industrial
summary is [`r2mt_industrial_multimethod.json`](r2mt_industrial_multimethod.json).
Plotting inputs are the `source_r2mt_*.csv` files under the paper figure
directory.

## SyncG retrospective evaluation

The common roster contains 1,558 images from 14 held-out scene groups and three
independently fitted checkpoints.

| Method | All six | Projective pool |
|---|---:|---:|
| Direct-ResNet18 | 1.4832 ± 0.0346 | 2.1648 ± 0.0435 |
| EfficientNet-B0 | 1.3935 ± 0.0606 | 1.9722 ± 0.0852 |
| MobileNetV3-Large | 1.6335 ± 0.0716 | 2.3143 ± 0.1361 |
| Internal relation foundation | 1.1611 ± 0.0424 | 1.5287 ± 0.0569 |
| **R²MT-Net** | **1.0517 ± 0.0415** | **1.3099 ± 0.0501** |

The internal foundation is shown only for ablation traceability; it is not a
second manuscript model.

## Industrial four-model ROI replay

The cohort contains 1,395 ROIs from 52 provisional source groups. All methods
use identical condition pixels and three fitted seeds.

| Method | All six | Projective pool | Persp.-45 | 45 + blur |
|---|---:|---:|---:|---:|
| Direct-ResNet18 | 23.5771 ± 2.4504 | 28.1866 ± 2.7176 | 29.7125 ± 2.3260 | 31.1401 ± 3.1682 |
| **EfficientNet-B0** | **15.2550 ± 2.3430** | **18.0914 ± 2.9211** | **20.3975 ± 3.8195** | **20.8227 ± 3.8398** |
| MobileNetV3-Large | 18.7569 ± 2.8915 | 23.2777 ± 3.0215 | 25.6296 ± 2.7454 | 26.3488 ± 3.1612 |
| R²MT-Net | 20.2817 ± 3.1514 | 21.5524 ± 3.7915 | 21.2454 ± 3.5084 | 23.7732 ± 4.4467 |

Against Direct-ResNet18, R²MT-Net reduces all-condition NMAE by 3.2954 pp
(95% group-bootstrap CI 2.0864–4.4153) and projective NMAE by 6.6342 pp
(4.3215–8.8624). EfficientNet-B0 transfers better in aggregate, so the
industrial result supports a matched-backbone mechanism claim rather than
universal cross-backbone superiority.

## Same-pixel VDN intersection

This supporting comparison contains 129 samples, six conditions, and 774 rows.
VDN is one annotation-assisted direction checkpoint, not an input-equivalent
automatic-system baseline.

| Scope | R²MT-Net | EfficientNet-B0 | VDN annotation reference |
|---|---:|---:|---:|
| All six | **0.9054 ± 0.0392** | 1.1673 ± 0.0638 | 1.6444 |
| Projective pool | **1.2140 ± 0.0714** | 1.7119 ± 0.1396 | 2.1030 |

## OCR accepted-output accuracy

Only accepted labeled frames are scored: Acc@5%FS is **74.07 ± 12.83%**
(`n=9`) across three R²MT-Net fits. This value does not measure full-frame OCR
coverage or live detector latency.

## Model-only efficiency

On an RTX 4060 with BF16 and 256×256 inputs, the full model has 11.6892 million
unique parameters and 9.582 GFLOPs per sample. Mean latency is 22.3197 ms/sample
at batch 1 and 3.5657 ms/sample at batch 8. The timing excludes localization,
support-normalization materialization, OCR, and image decoding.
