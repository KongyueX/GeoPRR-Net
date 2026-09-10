# Per-sample and source-group results

These tables belong to `official_syncg_fulltrain_20260908`. They contain
2,762,352 prediction rows from 269 result units. They supplement the aggregate
tables in the parent folder. Earlier manuscript data outside this folder belong
to a different experiment version.

## Files

| File | Contents |
| --- | --- |
| `per_sample/*.csv.gz` | Every normalized target, prediction, error, success/failure flag and available outer fold; partitioned by dataset and result family |
| `group_metrics.csv.gz` | Metrics for each source group, condition scope and seed, plus arithmetic means of the three seed metrics |
| `group_comparisons.csv.gz` | Group-level candidate and reference errors and their signed differences for the 486 published paired comparisons |
| `scan_group_comparisons.csv.gz` | Group-level differences for the 12 published angle-scan comparisons |
| `manifest.json` | File row counts, columns, cohort sizes, source result units and duplicate relationships |

| Result family | Units | Prediction rows |
| --- | ---: | ---: |
| `main` | 63 | 698,796 |
| `sarn_v2` | 27 | 299,484 |
| `annotation` | 12 | 149,436 |
| `ablation` | 18 | 432,000 |
| `adapted` | 12 | 100,440 |
| `frozen_fold_aligned` | 12 | 100,440 |
| `sensitivity` | 63 | 698,796 |
| `perspective_scan` | 54 | 216,000 |
| `ensemble_adapted`, `ensemble_frozen_fold_aligned` | 8 | 66,960 |

`main` uses frozen source-only models. `sarn_v2` is a separate preprocessing
diagnostic; the main CNN rows are Raw inputs. `annotation` uses annotated
reference points and is separate from the automatic comparison.
`adapted` contains supervised target-domain out-of-fold predictions.

`frozen_fold_aligned` repeats the Industrial main predictions with the same
outer folds as the adaptation experiment. The sensitivity configuration
`alpha_100_default_prior` repeats the main GeoPRR setting. These rows support
paired analysis; they are not additional training runs or independent samples.
An ensemble averages three source-seed predictions before scoring. It is one
ensemble, not three separately fitted models.

## Identities and units

The result key is `(family, method, dataset, seed, configuration, angle_degrees,
sample_id, condition)`. Pair methods on the same dataset/sample/condition and
seed. Each dataset uses one consistent group roster across all files:
SyncG has 4,000 samples and 145 scene groups; RF100 has 151 samples and 35
filename-derived groups; Industrial has 1,395 samples and 52 acquisition groups.

Industrial samples are assigned `industrial_000001` style identifiers.
Groups are assigned dataset-prefixed aliases, such as `rf100_group_001`.
Aliases refer to this release and must not be joined to similarly numbered
aliases in older result packages. Original Industrial source names, partitions,
paths and physical readings are not included.

`normalized_target`, `normalized_prediction` and `normalized_absolute_error`
use full-scale units: multiply error by 100 for %FS. Successful error equals
the absolute prediction-target difference. A failed prediction is blank,
has status `failure`, and retains error 1.0 in the denominator.
`outer_fold` is blank when not applicable. It is the recorded fold label;
no renumbering is applied.

Fields ending in `_pct_fs` or `_pct` are already percentages. Lower NMAE is
better. Acc@2/5 includes successful predictions within 0.02/0.05 normalized
error. Coverage is the fraction producing a valid prediction, not the fraction
with a correct reading.

## Group statistics

The nine regular scopes are each of the six conditions, `all_conditions`,
`perspective_pair` (moderate and severe perspective), and `projective_three`
(the two perspective conditions plus combined severe). For a scan unit,
`all_conditions` means its single explicit angle.

`aggregation=single_seed` describes one seed. `mean_of_three_seed_metrics`
averages the corresponding three seed metrics; its seed cell is blank.
Images and rows remain the original per-seed denominator, while its failure
count is the mean across seeds and may be fractional. `aggregation=ensemble`
describes errors after fixed equal-weight prediction averaging.

Group comparisons use the same error sums and group counts as the published
paired bootstrap analysis. For the three-seed comparisons, errors are averaged
across seeds for each original row before taking group means. Negative
`candidate_minus_reference_pct_fs` favors the candidate. Group sizes differ:
the global image-weighted effect is obtained by weighting group differences by
`rows_per_seed`, not by taking an unweighted mean over groups. Existing global
confidence intervals remain in `../paired_comparisons.csv`; these group rows do
not assert individual-group significance. An ensemble comparison has one
scored prediction per row and `seed_count=1`.

## Reproduction

The code repository includes `experiments/export_official_result_details.py`.
It exports saved final arrays and prediction ledgers without model training or
inference, checks the full sample-condition roster and prediction/error
relationship, and compares every unit's metrics against the completed run's
summaries before writing the publication manifest. It also checks that group
error sums recover the published paired-comparison means.

```python
import pandas as pd

rows = pd.read_csv('per_sample/rf100_main.csv.gz')
raw = rows[rows['method'] == 'efficientnet_b0']
print(raw.groupby('seed')['normalized_absolute_error'].mean() * 100)
```

## Limitations

These are the completed experiment's results, including unfavorable outcomes
and failed readings. No samples were selected or removed based on model error.
RF100 targets are geometric annotation-derived progress values; its 35 groups
are not 35 independently verified instruments. The source/target roles,
historical development visibility, shared SyncG scene identities and Industrial
adaptation setting are described in the parent package.
