# Controlled multi-state training: complete first-seed comparison

**Status: complete — all three arms finished 65 epochs; the frozen native gate is `validation_rejected`.** Composition gives native source-validation NMAE **0.162261183%FS**, a **4.085522% reduction** from consistency, below the required 5%, and **9.692213% higher** than matched-exposure ordinary supervision. It reduces Tail10 and maximum native errors, while its median, P90, P95 and Acc@1 are worse than supervised. On the separate heldout-combination diagnostic it introduces two severe errors and has **0.085669521%FS** NMAE, versus **0.021373696%FS** for supervised. The results do not support an overall accuracy or heldout-composition benefit over matched-exposure ordinary supervision under this protocol.

All three controlled arms retain the original MobileNetV4 field architecture and **7,695,954 parameters**, seed **20262020**, the same image/pair lists, donor/recipient forwards and executed batch counts. Native source-validation NMAE selected epoch **63** for supervised, **58** for consistency and **34** for composition; the original baseline checkpoint is epoch 61. Within each model, native, frozen `modal_r8` and heldout evaluation use the same selected checkpoint. The ±8-bin window was frozen before candidate training. Original train/validation/test lists remain unchanged; a separately recorded 4,096-image supplement is included in all three controlled arms.

## Complete source validation

All eight evaluations contain the same **1,506 IDs, targets and group IDs**, covering 70 groups. Metrics were recomputed from archived predictions; error units are percent of full scale (%FS).

| Reader | Readout | NMAE | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | native | 0.191749420 | 98.937583 | 99.535193 | 99.667995 | 7 | 5 |
| baseline | modal_r8 | 0.164085502 | 99.203187 | 99.734396 | 99.800797 | 4 | 3 |
| supervised | native | 0.147924067 | 99.269588 | 99.667995 | 99.800797 | 5 | 3 |
| supervised | modal_r8 | 0.129950499 | 99.402390 | 99.667995 | 99.867198 | 5 | 2 |
| consistency | native | 0.169172775 | 99.203187 | 99.535193 | 99.800797 | 7 | 3 |
| consistency | modal_r8 | 0.159951672 | 99.269588 | 99.667995 | 99.800797 | 5 | 3 |
| composition | native | 0.162261183 | 98.937583 | 99.667995 | 99.800797 | 5 | 3 |
| composition | modal_r8 | 0.156721114 | 98.937583 | 99.601594 | 99.867198 | 6 | 2 |

| Reader | Readout | Median | P90 | P95 | Tail10 NMAE | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | native | 0.067364 | 0.309844 | 0.477729 | 1.204454 | 32.006876 |
| baseline | modal_r8 | 0.066868 | 0.291035 | 0.456348 | 0.935811 | 27.949108 |
| supervised | native | 0.057125 | 0.239308 | 0.389268 | 0.897473 | 26.627396 |
| supervised | modal_r8 | 0.057386 | 0.237127 | 0.381035 | 0.720578 | 19.557136 |
| consistency | native | 0.057375 | 0.252053 | 0.383446 | 1.074270 | 44.822315 |
| consistency | modal_r8 | 0.057375 | 0.251916 | 0.381065 | 0.983087 | 44.822315 |
| composition | native | 0.071853 | 0.314324 | 0.506617 | 0.862841 | 20.013927 |
| composition | modal_r8 | 0.071695 | 0.309066 | 0.512059 | 0.816990 | 20.489101 |

Supervised improves native NMAE by **22.855533%** relative to the original native baseline. With the same frozen modal readout it improves by **20.803181%**, but >2%FS errors rise from 4 to 5. Adding U/M consistency increases matched-exposure NMAE by **14.364605%** for native and **23.086617%** for r8. Composition improves on consistency by **4.085522%** for native and **2.019709%** for r8, while remaining worse than supervised by **9.692213%** and **20.600626%**, respectively. Its native Tail10 and maximum are lower than supervised, but the same five >2%FS errors and three >5%FS errors are not the same individual cases.

## Frozen native decision

All five conditions were required. The decision uses complete formal batch-16 native reevaluation scores, not the slightly different in-training selection score, the r8 scores or the heldout diagnostic.

| Condition | Observed comparison | Passed |
| --- | --- | --- |
| Native NMAE reduced at least 5% vs consistency | 4.085522% reduction | no |
| >5%FS count does not increase vs consistency | 3 vs 3 | yes |
| Tail10 does not increase vs consistency | 0.862841 vs 1.074270%FS | yes |
| Native NMAE strictly better than supervised | 0.162261183 vs 0.147924067%FS | no |
| Native NMAE strictly better than original | 0.162261183 vs 0.191749420%FS | yes |

The outcome is **validation_rejected**. This development screen is not a significance test. The user prohibited additional seeds and candidate training; no seed expansion, official-test evaluation or new candidate training was started.

## Low-support and readout observations

Both consistency evaluations select **field=1,505, direct=1, mean=0**; all other evaluations, including composition, select **field=1,506, direct=0, mean=0**. All predictions remain in the full denominator, with zero prediction failures. For consistency `sync_3663`, support **0.023441257** triggers the direct prediction **0.501960754** against target **0.053737599**, producing **44.822315%FS** error. Composition predicts native **0.042028882** with support **0.998379707**, producing **1.170872%FS** error and no fallback. This is a case-level recovery; it does not identify which added term caused it or demonstrate a general repair.

The frozen r8 readout has adverse case-level responses on the composition checkpoint. It changes `sync_3663` error from **1.170872 to 3.071022%FS** and `sync_12540` from **0.185541 to 2.070816%FS**, making both new >2%FS errors. Across the full 1,506 images, r8 rescues one >2%FS error and regresses two, while reducing >5%FS errors from 3 to 2 and increasing maximum error from **20.013927 to 20.489101%FS**. Field coverage is **99.933599%** for consistency and 100% for the other checkpoints; prediction coverage is 100% throughout.

## Matched rescues and regressions

A rescue moves from error strictly above a threshold to at or below it; a regression moves in the opposite direction. Counts are paired image-level comparisons, not additional independent samples.

| Comparison | >2%FS rescues | >2%FS regressions | >5%FS rescues | >5%FS regressions |
| --- | ---: | ---: | ---: | ---: |
| supervised_native vs baseline_native | 3 | 1 | 2 | 0 |
| supervised_modal_r8 vs baseline_modal_r8 | 1 | 2 | 1 | 0 |
| supervised_modal_r8 vs supervised_native | 1 | 1 | 1 | 0 |
| consistency_native vs baseline_native | 2 | 2 | 2 | 0 |
| consistency_modal_r8 vs baseline_modal_r8 | 1 | 2 | 0 | 0 |
| consistency_modal_r8 vs consistency_native | 2 | 0 | 0 | 0 |
| composition_native vs baseline_native | 5 | 3 | 3 | 1 |
| composition_modal_r8 vs baseline_modal_r8 | 1 | 3 | 1 | 0 |
| composition_modal_r8 vs composition_native | 1 | 2 | 1 | 0 |
| consistency_native vs supervised_native | 1 | 3 | 1 | 1 |
| consistency_modal_r8 vs supervised_modal_r8 | 0 | 0 | 0 | 1 |
| supervised_native vs baseline_modal_r8 | 0 | 1 | 1 | 1 |
| composition_native vs supervised_native | 2 | 2 | 1 | 1 |
| composition_modal_r8 vs supervised_modal_r8 | 0 | 1 | 0 | 0 |
| composition_native vs consistency_native | 4 | 2 | 1 | 1 |
| composition_modal_r8 vs consistency_modal_r8 | 0 | 1 | 1 | 0 |

## Separate heldout-combination diagnostic

Each selected checkpoint reads the same **4,096 heldout directed edges between 4,096 seen source images**, in **1,024 layout groups**. These edges have zero overlap with the 8,192 training edges; every image, state and layout appeared in training. The three models contribute **12,288 prediction rows** on these same 4,096 edges. This population is separate from source validation and is not pooled with it.

| Arm | NMAE (%FS) | Acc@1/2/5 (%) | >2%FS / >5%FS | Median (%FS) | P90 (%FS) | P95 (%FS) | Tail10 (%FS) | Maximum (%FS) | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised | 0.021373696 | 100.000000 / 100.000000 / 100.000000 | 0 / 0 | 0.012045 | 0.047918 | 0.083366 | 0.097164 | 0.327731 | 0 |
| consistency | 0.023266617 | 100.000000 / 100.000000 / 100.000000 | 0 / 0 | 0.014336 | 0.048073 | 0.074875 | 0.097448 | 0.378931 | 0 |
| composition | 0.085669521 | 99.804688 / 99.951172 / 99.951172 | 2 / 2 | 0.041151 | 0.160843 | 0.273483 | 0.442885 | 24.143932 | 0 |

Inference combines predicted donor U/M with predicted recipient Q/M using native expectation, with no GT spatial mask, warp or fallback; targets are recipient physical readings. Mass below 0.15, invalid inputs or nonfinite/out-of-range readings retain normalized error 1.0. All edges pass these validity checks, but validity does not guarantee accuracy. Composition produces **24.143932%FS** error on `g0336_s03 → g0336_s00` and **19.309342%FS** on `g0825_s01 → g0825_s03`; both supports exceed **0.99999**. These are valid, severely wrong readings and cannot be attributed to low support. Both are >2%FS and >5%FS regressions relative to either control, with zero rescues at these thresholds.

Composition's heldout NMAE is **4.008175 times** supervised and **3.682079 times** consistency. Ordinary supervision already composes accurately on these edges. The same native-selected checkpoint is used throughout; no heldout-selected checkpoint or second model is substituted. Low combination error in a control does not prove necessity of explicit composition supervision, and the composition arm's unfavorable diagnostic is retained.

## Training and execution cost

Each epoch has **18,590 unique RGB images**, expanded to **22,686 donor items** plus **8,192 recipient forwards**, for **30,878 image forwards** and **709 executed batches**. All 65 history records in all three arms match. Each arm executes **2,007,070 image forwards** and **46,085 batches**. Supervised and consistency each record **46,062 effective optimizer updates** and **23 AMP skips**; composition records **46,063 effective updates** and **22 skips**. Thus executed batches and image exposure match, while effective updates differ by one. Skipped updates remain in batch and exposure counts. The original baseline has 14,494 unique images, 453 batches/epoch and 29,430 effective updates over 65 epochs.

Supervised has all four state weights zero. After five warmup epochs, consistency enables `state_field_l1=0.5` and `state_support_l1=0.1`; composition additionally enables `state_reading_cdf=1.0` and `state_mass_l1=0.1`. Raw losses and `supervised_cross_readouts=8192` eligibility counts do not imply that zero-weight objectives supervise a control. Both donor and recipient receive the same original single-image losses. Comparisons with the original baseline combine added images, repeated exposure and extra executed batches; comparisons among the three new arms match those quantities.

| Arm | Recorded epoch seconds | Whole training-process seconds | Peak recorded CUDA bytes | Selected epoch | Effective updates / AMP skips |
| --- | ---: | ---: | ---: | ---: | ---: |
| supervised | 14,939.500141 | 15,004.380793 | 31,136,305,152 | 63 | 46,062 / 23 |
| consistency | 14,137.293540 | 14,198.642869 | 31,133,880,320 | 58 | 46,062 / 23 |
| composition | 14,290.725327 | 14,346.154169 | 31,135,903,744 | 34 | 46,063 / 22 |

Epoch durations include each epoch's source validation and are recorded before the subsequent checkpoint write. Process wall additionally includes setup, writes and shutdown. Supervised ran **10:13:45–14:23:49**, consistency **14:26:04–18:22:43** and composition **18:24:51–22:23:58**, on **2026-10-20 (Asia/Shanghai)**. Composition's joint-epoch mean is **217.608089 seconds**. The original baseline's epoch-duration sum is **5,729.045714 seconds** under its smaller exposure protocol.

The three training processes total **43,549.177831 seconds**; their nine evaluations total **390.760933 seconds**. From the first training start through the final heldout process finish, sequential queue wall is **43,940.119930 seconds (12h 12m 20s)**. These process wall figures exclude source generation and are not GPU-kernel timing. The separately archived source supplement required **16,876.000385 seconds (4h 41m 16s)**; its rendering/preparation breakdown is in the linked source package.

The following observations use **RTX PRO 6000 Blackwell Server Edition, batch 16, 4 workers and AMP**. Source-validation forward timing includes synchronized model forward and selected readout. ROI pipeline timing additionally includes image reads, annotation-defined ROI crop/resize, normalization, transfer and prediction collection.

| Reader | Readout | Forward ms/image | Forward images/s | ROI pipeline ms/image | ROI pipeline images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | native | 6.357 | 157.30 | 9.877 | 101.24 |
| baseline | modal_r8 | 6.432 | 155.47 | 10.430 | 95.87 |
| supervised | native | 6.778 | 147.54 | 11.013 | 90.80 |
| supervised | modal_r8 | 6.792 | 147.24 | 10.933 | 91.47 |
| consistency | native | 6.552 | 152.63 | 10.333 | 96.78 |
| consistency | modal_r8 | 7.713 | 129.66 | 11.641 | 85.90 |
| composition | native | 6.467 | 154.63 | 10.137 | 98.65 |
| composition | modal_r8 | 6.325 | 158.11 | 10.147 | 98.55 |

Heldout costs are **per two-image edge**, with the shared reader's parameters counted once. Each model executes 8,192 image forwards.

| Arm | Reader forward ms/edge | Composition ms/edge | ROI pipeline ms/edge | Whole evaluation-process seconds: native / r8 / heldout |
| --- | ---: | ---: | ---: | ---: |
| supervised | 3.480488 | 0.091563 | 12.007195 | 20.220129 / 20.115226 / 94.035810 |
| consistency | 3.299575 | 0.090765 | 11.437717 | 18.891157 / 20.937055 / 89.083097 |
| composition | 3.455150 | 0.091789 | 11.814592 | 18.448821 / 18.384900 / 90.644738 |

Model/checkpoint setup, detector inference and export are excluded from the timed ROI segment; process wall includes setup and export. These observations are batch throughput, not single-image latency or a controlled timing benchmark.

## Files

- `summary.json`: complete status, formal native decision, metrics, paired deltas and affected IDs, fallback/readout/heldout severe cases, per-arm configurations/exposure/effective updates, cost and limitations.
- `metrics.csv`: eight source-validation rows and three separately labeled heldout-combination rows.
- `predictions.csv.gz`: **12,048 source-validation rows** (1,506 images × 4 checkpoints × 2 readouts), including normalized target, prediction/error, IDs, validity, support and selected expert. All existing baseline/supervised/consistency rows are preserved.
- `heldout_predictions.csv.gz`: **12,288 edge rows** (the same 4,096 edges × 3 checkpoints), including donor/recipient/layout identity and complete error denominators. Existing two-control rows are preserved.
- Raw protocol/queue snapshots, reader config/run/history/best/completed records, original evaluation summaries/CSVs and four completed step logs per arm are archived under `refine-logs/geoprr_mobilenet_tail_20261019/iteration_v3_compositional/{supervised_completed,consistency_completed,composition_completed}`. The composition archive also contains the final native decision. Baseline raw records remain in the adjacent `iteration_v1` archive. Images, weights and vendor assets are excluded from the public package.
- [Controlled source generation](../geoprr_v2_compositional_source_20261020/README.md) and [separate low-support diagnosis](../geoprr_v2_low_support_20261020/README.md) document data provenance and diagnostic counterfactuals.

## Limitations

One seed provides no seed SD, significance or reproducibility estimate. Source validation serves development and checkpoint selection; these arms have no candidate official-test, perturbation or real-domain result here. Comparisons with the original baseline combine data and exposure changes. Heldout combinations use seen images/states/layouts and do not establish unseen-layout transfer. Case-level support recovery does not establish an optimization mechanism or general fallback reliability. Saved selection and formal batch-16 reevaluation scores differ slightly with AMP/batch numerical paths; selected checkpoints remain fixed and main metrics use formal reevaluation. Timings include startup/warmup, measure batch throughput and exclude detector inference. A favorable tail or r8 value cannot replace the frozen native continuation rule.
