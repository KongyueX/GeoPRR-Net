# Official SyncG full-training aggregate results

Verified aggregate results for `official_syncg_fulltrain_20260908`.

The experiment completed 21 source model-seed combinations, 65 optimization stages, 63 automatic main comparison units, 27 additional SARN units, 12 annotation-assisted diagnostics, 18 ablation evaluations, 63 sensitivity configurations, 54 angle-scan units, 60 adaptation heads, 12 OOF outputs, 8 ensembles and 18 full-path efficiency measurements.

## Contents

| File | Contents |
| --- | --- |
| `per_seed_metrics.csv` | Full-denominator metrics for each method, seed, dataset and scope |
| `seed_mean_sample_sd.csv` | Three-seed means and sample standard deviations |
| `paired_comparisons.csv` | Paired source-group bootstrap intervals and effect sizes |
| `paired_group_consistency.csv` | Aggregate counts of better/tied/worse source groups |
| `ensemble_metrics.csv` | Frozen and adapted fixed equal-weight ensemble metrics |
| `alpha_prior_sensitivity.csv` | All seven inference settings across three seeds/datasets |
| `perspective_scan*.csv`, `perspective_scan*.json` | Six-angle scan results and comparisons |
| `annotation_assisted_comparison.csv` | Separate GeoPRR versus annotation-assisted VDN diagnostic |
| `factorial_interactions.json` | Geometry-by-routing factorial effects |
| `geoprr_diagnostics.json` | Aggregate routing, posterior and geometry diagnostics |
| `error_distribution_summaries.json` | Aggregate tail errors, RMSE and Acc@1 results |
| `training_*.csv` | Training budgets, updates, epoch costs and source loss curves |
| `efficiency.csv` | Complete ROI-path latency, throughput, memory and parameters |
| `supported_operation_count.json` | THOP-supported operation lower bound |
| `protocol.json`, `completion_counts.json` | Protocol definitions and verified counts |

`family=main` is the automatic source-only comparison. `sarn_v2` is a preprocessing diagnostic; `annotation` is an annotation-assisted diagnostic; `ablation` contains prescribed interventions; `adapted` is supervised OOF adaptation. `frozen_fold_aligned` repeats frozen predictions with fold alignment and is not another independently fitted model.

Each result keeps its full denominator, including failed readings (normalized error 1.0). Three seeds are not treated as three times as many independent images. Negative effects and confidence intervals crossing zero are retained. The supplemental annotation-assisted comparison has unequal auxiliary information and is separate from the automatic ranking.

## Release scope

This folder contains aggregate tables and protocol metadata. Source images, individual labels/predictions, acquisition-group identifiers, weights, caches, local file inventories and machine-specific paths remain in the original run archive. The release follows the repository's existing aggregate-data publication scope and third-party notices.

## Limitations

The data have been used in historical development. Official SyncG training/test image IDs are disjoint, but their scene identities overlap; the source test is not a new scene-disjoint or blind cross-dataset evaluation. GeoPRR uses a fixed A/B division within the complete source-training pool, and both portions participate in training different stages.

Industrial results use 1,395 recovered native ROIs. Full original field frames are unavailable. Supervised adaptation is reported separately from frozen source-only transfer.

Two source YOLO loss records are nonfinite; the raw training observations are retained in the loss-curve export, while final weights are finite. DeepLab's recorded batch-six/batch-one paths produce two threshold-boundary mask differences; their actual execution records were retained. The final GeoPRR moment projection uses the recorded numerical tolerance and preserves source weights.

Supported GMACs include THOP-supported module operations only. Grid sampling, tensor-only moment transport, softmax and unsupported operations are omitted; this is a lower bound, not a complete pipeline FLOP count. Full-path latency, throughput and memory are the primary efficiency evidence.
