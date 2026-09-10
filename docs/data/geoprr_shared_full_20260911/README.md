# Shared full-source GeoPRR: three-seed results

Completed results for seeds 20262020, 20262021 and 20262022. Each reader reuses
its same-seed EfficientNet-B0 foundation trained on all 16,000 official SyncG
training images. ReMST, R2MT, polar and the final router are subsequently fitted
on all 16,000 source images. The complete reader is evaluated on the same
SyncG, RF100-VL and Industrial-1395 ROI populations as the original experiment.

## Results

Six-condition NMAE in %FS, mean ± sample standard deviation across three seeds;
lower is better. All 99,828 shared-reader predictions completed without an
inference failure.

| Dataset | Shared + full-source downstream | Original GeoPRR | RawEff |
| --- | ---: | ---: | ---: |
| SyncG | **0.741 ± 0.022** | 0.873 ± 0.041 | 1.659 ± 0.063 |
| RF100-VL | **3.205 ± 0.254** | 5.687 ± 1.366 | 4.606 ± 0.557 |
| Industrial-1395 | 11.624 ± 1.603 | **10.851 ± 0.809** | 13.691 ± 2.402 |

[RESULTS.md](RESULTS.md) includes clean-image results, individual seeds,
accuracy thresholds and paired group intervals. Original GeoPRR and RawEff
comparators are the saved source-only results from
`official_syncg_fulltrain_20260908`; their models are not fitted again here.

## Files

| File | Contents |
| --- | --- |
| `per_sample/syncg.csv.gz` | 72,000 predictions: 4,000 images × 6 conditions × 3 seeds |
| `per_sample/rf100.csv.gz` | 2,718 predictions: 151 images × 6 conditions × 3 seeds |
| `per_sample/industrial.csv.gz` | 25,110 predictions: 1,395 images × 6 conditions × 3 seeds |
| `group_metrics.csv.gz` | 8,352 per-group metric rows, including individual seeds and their arithmetic means |
| `group_comparisons.csv.gz` | 4,176 per-group differences against the original GeoPRR and RawEff |
| `full_three_seed_per_seed_metrics.csv` | 243 method/seed/dataset/scope metric rows |
| `full_three_seed_mean_sample_sd.csv` | 81 three-seed aggregate rows |
| `full_three_seed_paired_comparisons.csv` | 12 paired group-bootstrap comparisons for clean and all six conditions |
| `full_three_seed_summary.json` | Complete numerical summary and training-scope validation |
| `training_epochs.csv` | 1,005 newly trained optimization-epoch records |
| `training_stages.csv` | 15 stage records, distinguishing reused initialization and newly trained stages |
| `protocol.json`, `export_validation.json`, `manifest.json` | Dataset roles, export checks, file schemas and row counts |

## Prediction schema and identities

Each prediction has one `(dataset, seed, sample_id, condition)` key. Targets,
predictions and errors use normalized full-scale units in `[0, 1]`; multiply
errors by 100 for %FS. A failed prediction would have a blank prediction field
and error 1.0, remaining in the denominator. Coverage reports valid outputs,
not the percentage of correct readings.

Candidate fields describe the base, polar and relational outputs; the three
`weight_*` fields contain the corresponding router weights.
`relation_available` records image-derived geometric availability.
`original_geoprr_error` and `raw_full_error` retain matching reference errors.
No target-domain adaptation or prediction ensembling is applied.

Industrial samples use anonymous `industrial_000001` style IDs. All groups use
dataset-prefixed aliases. The exporter checks each sample, target and group
against the already published `official_syncg_fulltrain_20260908/details`
package: identifiers in these two packages can be joined directly. Other older
manuscript packages may use different group aliases.

SyncG has 145 scene groups, RF100 has 35 filename-derived groups, and Industrial
has 52 acquisition groups. Group statistics use those source groups rather than
counting conditions or seeds as independent images.

## Statistics

Regular scopes are each of the six conditions, `all_conditions`,
`perspective_pair` (moderate/severe perspective) and `projective_three`
(the two perspective conditions plus combined severe).

Fields ending in `_pct_fs` or `_pct` are already percentages. In the group
table, `aggregation=single_seed` describes one reader;
`mean_of_three_seed_metrics` averages those three metrics. The latter keeps the
per-seed image/row denominator and reports the mean failure count.

For paired comparisons, errors are averaged across seeds for each original
sample-condition row before aggregation within its source group. Negative
`candidate_minus_reference_pct_fs` favors the shared reader. Weight group
differences by `rows_per_seed` to recover the global image-weighted effect.
The 20,000-resample intervals concern group uncertainty in the fixed three-seed
average, while sample SD separately describes variation among training seeds.

## Training records

Each seed newly fits ReMST for 5 epochs; three R2MT experts and its internal
router for 40 epochs each; polar for 10 epochs; and the final router for
40 + 120 epochs. The final router uses 16,000 × 6 = 96,000 training records.
The shared warm initializer was previously trained on A=12,818 images using
seed 20262020 and is reused by all three trajectories. Full-source polar
training then updates that initialization separately for every seed.

The epoch table retains raw numerical observations. `metric_*` errors remain
normalized values rather than %FS. `epoch_elapsed_seconds` in the stage table
is the sum of epoch times; `wall_elapsed_seconds` includes that stage's cache
generation and other work, so the two durations must not be added. Reused
RawEff/warm stages have no new training time or duplicated epoch curves.

`experiments/export_shared_full_results.py` in the code repository exports the
prediction and group tables from the archived run without model inference.
It checks row alignment, error arithmetic, aggregate metrics and the existing
public aliases. The stage and epoch tables come from the completed training
histories. This package contains result data; source images, checkpoints,
original Industrial identities/physical readings and local paths are excluded.

## Limitations

The base, candidate modules and final router share source-training images.
This is a different training protocol from the original A/B-separated upstream
scheme. Official SyncG training/test image IDs are disjoint, but scene identities
overlap. Target domains have historical development visibility, so this is an
exploratory comparison rather than a newly established blind test. RF100
targets are geometric annotation-derived progress values, and its filename
groups are not independently verified distinct instruments. Industrial's
original full frames are unavailable; evaluation uses recovered native ROIs.
