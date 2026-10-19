# Original MobileNetV4: frozen modal readout across three seeds

The fixed `modal_r8` readout lowers official-test NMAE for all three seeds: the mean changes from **0.208316 ± 0.011749%FS** to **0.190999 ± 0.004579%FS** (mean ± sample SD), an **8.31%** relative reduction. Errors above 5%FS fall from **16/16/14** to **9/9/8**. **The maximum official-test error worsens for every seed**, from **67.554/50.288/75.088%FS** to **73.534/76.453/81.429%FS**. This is a tradeoff between lower average error and fewer severe-error cases versus larger remaining extreme errors.

Both readouts use the same original MobileNetV4 checkpoint for each seed. `native` takes the original full-distribution expectation. `modal_r8` takes a continuous expectation within ±8 bins of the main peak, clips the window at the scale endpoints and retains the original invalid-field fallback. The radius was frozen using seed-20262020 source validation before these test comparisons; no test-dependent clipping, threshold or fusion weight is applied. The operator adds no trainable parameters.

The frozen training/source-validation/test populations contain 14,494/1,506/4,000 images. Each seed completes the original 65-epoch recipe (5 backbone-frozen warmup epochs and 60 joint epochs). Source-validation NMAE using the original readout selects checkpoints at epochs **61, 49 and 58** for seeds **20262020, 20262021 and 20262022**. The first reader is reused; the other two use the unchanged original architecture and losses.

## Source validation

| Seed | Readout | NMAE (%FS) | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS | Maximum (%FS) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20262020 | native | 0.191749 | 98.938 | 99.535 | 99.668 | 7 | 5 | 32.007 |
| 20262020 | modal_r8 | 0.164086 | 99.203 | 99.734 | 99.801 | 4 | 3 | 27.949 |
| 20262021 | native | 0.204480 | 99.004 | 99.402 | 99.602 | 9 | 6 | 42.682 |
| 20262021 | modal_r8 | 0.172328 | 99.336 | 99.734 | 99.801 | 4 | 3 | 34.920 |
| 20262022 | native | 0.176975 | 98.871 | 99.734 | 99.801 | 4 | 3 | 30.010 |
| 20262022 | modal_r8 | 0.144196 | 99.270 | 99.801 | 99.867 | 3 | 2 | 22.001 |

## Official native-image test

| Seed | Readout | NMAE (%FS) | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS | Maximum (%FS) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20262020 | native | 0.220749 | 98.925 | 99.400 | 99.600 | 24 | 16 | 67.554 |
| 20262020 | modal_r8 | 0.193042 | 99.200 | 99.700 | 99.775 | 12 | 9 | 73.534 |
| 20262021 | native | 0.197397 | 98.900 | 99.325 | 99.600 | 27 | 16 | 50.288 |
| 20262021 | modal_r8 | 0.185753 | 99.200 | 99.650 | 99.775 | 14 | 9 | 76.453 |
| 20262022 | native | 0.206803 | 99.075 | 99.400 | 99.650 | 24 | 14 | 75.088 |
| 20262022 | modal_r8 | 0.194201 | 99.400 | 99.650 | 99.800 | 14 | 8 | 81.429 |

## Three-seed summaries

The values below use arithmetic means and **sample standard deviations with denominator n−1** across three seeds. Tail10 is the mean of the worst ceil(10% × N) errors within each seed; it is not the maximum.

| Split | Readout | NMAE (%FS) | Acc@2 (%) | >5%FS count | Tail10 NMAE (%FS) | Maximum (%FS) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| source_validation | native | 0.191068 ± 0.013765 | 99.557 ± 0.167 | 4.667 ± 1.528 | 1.194863 ± 0.117196 | 34.900 ± 6.813 |
| source_validation | modal_r8 | 0.160203 ± 0.014463 | 99.757 ± 0.038 | 2.667 ± 0.577 | 0.892565 ± 0.124299 | 28.290 ± 6.466 |
| official_test | native | 0.208316 ± 0.011749 | 99.375 ± 0.043 | 15.333 ± 1.155 | 1.390274 ± 0.141551 | 64.310 ± 12.714 |
| official_test | modal_r8 | 0.190999 ± 0.004579 | 99.667 ± 0.029 | 8.667 ± 0.577 | 1.220970 ± 0.069620 | 77.138 ± 3.992 |

## Matched official-test changes

A rescue crosses from an original error strictly above the threshold to a modal error at or below it; a regression crosses in the opposite direction. These counts compare exactly matched IDs, normalized targets and group IDs.

| Seed | >2%FS rescues | >2%FS regressions | >5%FS rescues | >5%FS regressions | NMAE reduction (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20262020 | 12 | 0 | 7 | 0 | 12.551 |
| 20262021 | 14 | 1 | 7 | 0 | 5.899 |
| 20262022 | 11 | 1 | 6 | 0 | 6.094 |

Across the three seeds there are **37 rescues and 2 regressions at 2%FS**, and **20 rescues and 0 regressions at 5%FS**. These totals cover 12,000 matched seed/image comparisons of the same 4,000 images. The source-validation matched counts are retained separately in `summary.json` and the modal rows of `metrics.csv`.

## Parameters and measured execution cost

Every evaluation executes the full **7,695,954-parameter** reader, with **0 router parameters** and **0 added trainable parameters**. Actual expert selection is field=4,000, direct=0, mean=0 per official-test evaluation, and field=1,506, direct=0, mean=0 per source-validation evaluation. All evaluations retain their complete population; all recorded prediction-failure counts are zero.

The table reports observed official-test throughput on **NVIDIA RTX PRO 6000 Blackwell Server Edition**, using **batch 16**, **4 data workers** and AMP. Forward time includes synchronized model forward and the selected readout. The ROI pipeline additionally includes image reads, ROI crop/resize, normalization, transfer and prediction collection.

| Seed | Readout | Forward ms/image | Forward images/s | ROI-pipeline ms/image | ROI-pipeline images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 20262020 | native | 2.261 | 442.19 | 6.084 | 164.38 |
| 20262020 | modal_r8 | 2.277 | 439.12 | 5.808 | 172.18 |
| 20262021 | native | 2.265 | 441.56 | 5.937 | 168.45 |
| 20262021 | modal_r8 | 2.265 | 441.48 | 6.309 | 158.51 |
| 20262022 | native | 2.216 | 451.27 | 5.785 | 172.86 |
| 20262022 | modal_r8 | 2.502 | 399.64 | 5.951 | 168.04 |

Source-validation timing was collected separately and is preserved per evaluation. Training epoch durations, full parameter counts, peak allocated CUDA memory and the actual expert counts are also retained in `summary.json` and `metrics.csv`.

## Files

- `summary.json`: frozen protocol, checkpoint-selection records, per-seed metrics, mean/sample-SD summaries, paired rescues/regressions, execution boundaries and training cost.
- `metrics.csv`: the 12 full-population source-validation/test evaluations, with accuracy, severity, tail/maximum error, actual expert counts and timing. Modal rows include the matched change counts.
- `three_seed_metrics.csv`: means and sample SDs for all primary error/accuracy/severity metrics and recorded costs, grouped by split and readout.
- `predictions.csv.gz`: **33,036 normalized prediction rows**, with split, seed, readout, sample/group IDs, target, prediction, validity, absolute error and executed expert index. Errors are fractions of full scale; multiply by 100 for %FS. The archive contains 24,000 official-test rows (4,000 images × 3 seeds × 2 readouts) and 9,036 source-validation rows (1,506 × 3 × 2). Invalid predictions, if present, would retain error 1.0 in the full denominator.
- Raw protocol, exports, evaluation JSON/CSV files and small reader artifacts are retained under the local archive path recorded in `summary.json`. Source images and model weights are excluded from this public package.

## Limitations

The radius-selection seed shares its source-validation data with the reported validation comparison; those scores include selection evidence. The official test was viewed during historical development, and this phase evaluates native images only, so it does not establish a new blind-test, perturbation or real-domain improvement. The same images recur across seeds; the 12,000 matched official-test comparisons are not independent images, and no significance claim is made. Source-validation reevaluation may differ slightly from saved selection scores because of AMP/batch numeric paths; the original selected checkpoints remain fixed. All rows use the field path, so this phase provides no additional evidence about low-support direct fallback. Timing comes from single runs, includes startup/warmup batches and excludes checkpoint/model construction, detector inference and output serialization; it is batch throughput, not single-image or complete camera-to-reading latency.

The observed reduction in average and tail error supports retaining the fixed readout as a candidate. Its larger worst-case errors leave the original Q diffusion and local U calibration problems unresolved; the next training iteration described in [the tail plan](../../GEOPRR_V2_TAIL_ITERATION_CN.md) should be judged on source validation before expanding seeds. No radius or correction is selected from the test extremes.
