# Decoupled Q/U supervision: first-seed source-validation results

The decoupled candidate slightly lowers native source-validation NMAE from **0.191749420%FS** to **0.191394952%FS**, a **0.184860% relative reduction**. It does **not** reach the frozen requirement of at least 5% lower native NMAE, so the recorded screening verdict remains **`validation_rejected`**. Under the separately reported, already frozen `modal_r8` readout, source-validation NMAE changes from **0.164085502%FS** to **0.144276557%FS** (12.072331% lower), with >5%FS errors decreasing from **3 to 1**. That modal comparison does not replace the native screening rule.

The candidate uses the unchanged MobileNetV4 field architecture and original training recipe, with additional source-supervised losses `q_physical_cdf=1.0`, `u_local_cdf=1.0` and `q_gt_support=0.1`. These use source-training structural labels to constrain Q and local U readouts separately. Evaluation takes RGB ROI pixels only. Both readers contain **7,695,954 parameters**; the added losses introduce **0 trainable parameters**.

Training uses the frozen **14,494 source-training / 1,506 source-validation** images and seed **20262020**. The candidate completed **65 epochs** and selected **epoch 49** by original native source-validation NMAE. The original baseline checkpoint is epoch 61. `native` and `modal_r8` are evaluated on the same checkpoint within each reader; the ±8-bin readout window was frozen before candidate training. The comparisons below reuse the exact baseline evaluations from `iteration_v1/validation_evaluation`.

## Source-validation metrics

| Reader | Readout | NMAE (%FS) | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | native | 0.191749420 | 98.937583 | 99.535193 | 99.667995 | 7 | 5 |
| baseline | modal_r8 | 0.164085502 | 99.203187 | 99.734396 | 99.800797 | 4 | 3 |
| decoupled | native | 0.191394952 | 99.003984 | 99.535193 | 99.667995 | 7 | 5 |
| decoupled | modal_r8 | 0.144276557 | 99.203187 | 99.867198 | 99.933599 | 2 | 1 |

| Reader | Readout | Median (%FS) | P90 (%FS) | P95 (%FS) | Tail10 NMAE (%FS) | Maximum (%FS) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | native | 0.067364 | 0.309844 | 0.477729 | 1.204454 | 32.006876 |
| baseline | modal_r8 | 0.066868 | 0.291035 | 0.456348 | 0.935811 | 27.949108 |
| decoupled | native | 0.068597 | 0.307125 | 0.436540 | 1.180059 | 29.678018 |
| decoupled | modal_r8 | 0.068548 | 0.296729 | 0.423828 | 0.711313 | 25.225353 |

The native comparison lowers maximum error from **32.006876 to 29.678018%FS** and Tail10 NMAE from **1.204454 to 1.180059%FS**. Acc@1 improves by one image, while the >2%FS and >5%FS counts are unchanged. Its median error increases from **0.067364 to 0.068597%FS**. With the fixed modal readout, maximum error falls from **27.949108 to 25.225353%FS** and Tail10 NMAE from **0.935811 to 0.711313%FS**; the median increases from **0.066868 to 0.068548%FS**, and Acc@1 is unchanged.

## Frozen continuation decision

The prespecified rule requires native NMAE at most **0.182161949%FS** (95% of the same-evaluation baseline), no increase in the **>5%FS count**, and no increase in **native Tail10 NMAE**. The candidate has native NMAE **0.191394952%FS**, >5%FS count **5 versus 5**, and Tail10 NMAE **1.180059 versus 1.204454%FS**. The last two conditions pass, while the 5% NMAE improvement condition does not. The saved verdict is `validation_rejected`; automatic seed expansion is disabled.

## Matched rescues and regressions

All four evaluations contain the same 1,506 sample IDs, normalized targets and group IDs. A rescue crosses from a baseline error strictly above the threshold to an error at or below it; a regression crosses in the opposite direction.

| Comparison | >2%FS rescues | >2%FS regressions | >5%FS rescues | >5%FS regressions |
| --- | ---: | ---: | ---: | ---: |
| decoupled native vs baseline native | 1 | 1 | 1 | 1 |
| decoupled r8 vs baseline r8 | 2 | 0 | 2 | 0 |
| decoupled r8 vs decoupled native | 5 | 0 | 4 | 0 |
| baseline r8 vs baseline native | 3 | 0 | 2 | 0 |

The unchanged native severe-error counts include a case exchange: one rescue and one regression at each threshold. The same-readout modal comparison rescues two cases at each threshold with no regressions across those thresholds. Per-metric deltas are preserved in `summary.json`.

## Recorded cost

The candidate’s 65 epoch-duration records sum to **5,911.142442 seconds (98.519041 minutes)**; its mean joint-training epoch duration is **90.864394 seconds**. These are training-duration records, separate from inference timing.

All source-validation evaluations use **NVIDIA RTX PRO 6000 Blackwell Server Edition**, **batch 16**, **4 workers** and AMP. Forward timing includes synchronized model forward and the chosen readout. The ROI pipeline additionally includes image reads, ROI crop/resize, normalization, transfer and prediction collection. Each evaluation selected field=**1,506**, direct=**0**, mean=**0**, with **0 failures** and full denominator coverage.

| Reader | Readout | Forward ms/image | Forward images/s | ROI-pipeline ms/image | ROI-pipeline images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | native | 6.357 | 157.30 | 9.877 | 101.24 |
| baseline | modal_r8 | 6.432 | 155.47 | 10.430 | 95.87 |
| decoupled | native | 6.564 | 152.34 | 10.366 | 96.47 |
| decoupled | modal_r8 | 6.261 | 159.71 | 9.808 | 101.96 |

## Files

- `summary.json`: the unchanged screening verdict and its three checks, configurations, selected epochs, per-metric changes, matched rescues/regressions and timing boundaries.
- `metrics.csv`: the four complete source-validation evaluations, with NMAE, Acc@1/2/5, severity counts, quantiles, tail/maximum error, actual expert counts and measured execution cost.
- `predictions.csv.gz`: **6,024 normalized prediction rows** (1,506 images × 2 readers × 2 readouts), including configuration, split, seed, readout, sample/group IDs, target, prediction, validity, error and selected expert index. Multiply absolute errors by 100 for %FS. A failed prediction would retain normalized error 1.0 in the full denominator.
- Candidate protocol, queue state, raw decision, reader JSON/history records and the two existing evaluation CSV/JSON pairs are archived under `refine-logs/geoprr_mobilenet_tail_20261019/iteration_v2_decoupled`. Baseline reference records remain in the adjacent `iteration_v1` archive. Weights and source images are excluded.

## Limitations

This candidate has **one training seed**, so no sample SD, multi-seed reproducibility or statistical significance estimate is reported. These are **source-validation results only**; no candidate official-test, perturbation or real-domain result is available. The same validation population selects the checkpoint and previously selected the modal window, so the improvements are development evidence. The saved epoch-49 selection score is 0.191400892%FS and its batch-16 reevaluation is 0.191394952%FS; this small AMP/batch numeric difference does not change the checkpoint or the frozen decision. All rows use the field path and therefore provide no new low-support direct-fallback evidence. Timing includes startup/warmup and excludes checkpoint/model construction, detector inference and output serialization; it measures batch throughput rather than single-image or complete camera-to-reading latency.
