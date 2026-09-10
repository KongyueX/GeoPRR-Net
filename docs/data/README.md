# Public experiment data

| Package | Contents |
| --- | --- |
| [Shared full-source GeoPRR](geoprr_shared_full_20260911/README.md) | Three-seed full-source downstream training, 99,828 predictions, source-group statistics, training histories and matched comparator summaries |
| [Official full-training experiments](official_syncg_fulltrain_20260908/README.md) | Original official-source protocol, aggregate results and detailed ledgers for all prescribed experiment families |

Each package records its own training and evaluation protocol. The shared
full-source study is a separate experiment version and keeps the original
results available for comparison.

## ROI comparison update: 2026-09-07

Updated 2026-09-07. These files contain cohort/seed aggregates only.

| File | Contents |
|---|---|
| `roi_comparison_zero_shot.csv` | 12 rows: three methods, two target domains, clean and six-condition scopes. NMAE, accuracy and coverage use percentage units. |
| `roi_comparison_zero_shot_public.json` | Per-seed/aggregate transfer metrics, source-reference provenance and evaluation scope. |
| `roi_comparison_efficiency.csv` | Five FP32 batch-1 native-component and complete ROI-reading efficiency measurements. |
| `roi_comparison_efficiency_public.json` | Timing boundaries, observed precision, parameters, supported operation counts, memory and failure statistics. |

Industrial DeepLab and VDN use same-seed SyncG-trained YOLO pivot/start/end,
excluding the predicted pointer tip. RF100 retains annotation-assisted
reference geometry. Full ROI efficiency includes the reference network;
native probability-map/direction timings are separate measurements.

See the [result report](../ROI_GEOMETRY_COMPARISON_CN.md) and
[reference detector audit](../ROI_REFERENCE_DETECTOR_AUDIT_CN.md).
The public tables are also available in the
[paper repository](https://github.com/KongyueX/GeoPRR-Net-Paper/tree/main/data).

Raw local result JSONs are inputs to
`python -m experiments.export_roi_comparison_public_results`; they are not
part of this release. Public files omit local absolute paths, raw images,
weights and per-image Industrial readings.
