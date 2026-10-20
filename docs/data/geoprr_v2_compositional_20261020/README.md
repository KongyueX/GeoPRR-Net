# Controlled multi-state training: two completed arms

**Status: partial — supervised and consistency are complete; 2 of 3 first-seed arms.** Ordinary supervision gives native source-validation NMAE **0.147924067%FS**. Adding U/M consistency gives **0.169172775%FS**, a **14.364605% increase**, with >2%FS errors rising from **5 to 7** and maximum error from **26.627396 to 44.822315%FS**. The composition arm remains pending. These results do not establish a benefit from explicit composition supervision or supply the final five-condition continuation verdict.

Both completed arms retain the original MobileNetV4 field architecture and **7,695,954 parameters**, seed **20262020**, the same image/pair lists, donor/recipient forwards and batch counts. Each completed **65 epochs**; native source-validation NMAE selected epoch **63** for supervised and **58** for consistency. The original baseline checkpoint is epoch 61. Within each model, native and `modal_r8` use the same selected checkpoint. The ±8-bin window was frozen before candidate training. Original train/validation/test lists remain unchanged; a separately recorded 4,096-image supplement is included in both new arms.

## Complete source validation

All six evaluations contain the same **1,506 IDs, targets and group IDs**, covering 70 groups. Metrics were recomputed from archived predictions; error units are percent of full scale (%FS).

| Reader | Readout | NMAE | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original baseline | native | 0.191749420 | 98.937583 | 99.535193 | 99.667995 | 7 | 5 |
| original baseline | modal_r8 | 0.164085502 | 99.203187 | 99.734396 | 99.800797 | 4 | 3 |
| supervised | native | 0.147924067 | 99.269588 | 99.667995 | 99.800797 | 5 | 3 |
| supervised | modal_r8 | 0.129950499 | 99.402390 | 99.667995 | 99.867198 | 5 | 2 |
| consistency | native | 0.169172775 | 99.203187 | 99.535193 | 99.800797 | 7 | 3 |
| consistency | modal_r8 | 0.159951672 | 99.269588 | 99.667995 | 99.800797 | 5 | 3 |

| Reader | Readout | Median | P90 | P95 | Tail10 NMAE | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original baseline | native | 0.067364 | 0.309844 | 0.477729 | 1.204454 | 32.006876 |
| original baseline | modal_r8 | 0.066868 | 0.291035 | 0.456348 | 0.935811 | 27.949108 |
| supervised | native | 0.057125 | 0.239308 | 0.389268 | 0.897473 | 26.627396 |
| supervised | modal_r8 | 0.057386 | 0.237127 | 0.381035 | 0.720578 | 19.557136 |
| consistency | native | 0.057375 | 0.252053 | 0.383446 | 1.074270 | 44.822315 |
| consistency | modal_r8 | 0.057375 | 0.251916 | 0.381065 | 0.983087 | 44.822315 |

Supervised improves native NMAE by **22.855533%** relative to the original native baseline. With the same frozen modal readout it improves by **20.803181%**, but >2%FS errors rise from 4 to 5. On its own checkpoint, modal readout reduces NMAE while median error rises slightly. The consistency arm remains better than the original native baseline on average, but its matched-exposure native NMAE, Tail10 and maximum errors are worse than supervised. Its r8 NMAE is **23.086617% higher** than supervised r8, and >5%FS errors rise from 2 to 3.

Both consistency evaluations select **field=1,505, direct=1, mean=0**; all other evaluations select **field=1,506, direct=0, mean=0**. All predictions remain in the full denominator, with zero prediction failures. For `sync_3663`, predicted field support is **0.023441257**, below 0.15; the native direct fallback predicts **0.501960754** against target **0.053737599**, yielding the **44.822315%FS** maximum error in both readouts. It is retained rather than removed or reclassified as a failed prediction. Field coverage is **99.933599%** for consistency and 100% for the other models; prediction coverage is 100% throughout.

## Matched rescues and regressions

A rescue moves from error strictly above a threshold to at or below it; a regression moves in the opposite direction. Counts are paired image-level comparisons, not additional independent samples.

| Comparison | >2%FS rescues | >2%FS regressions | >5%FS rescues | >5%FS regressions |
| --- | ---: | ---: | ---: | ---: |
| supervised native vs original native | 3 | 1 | 2 | 0 |
| supervised r8 vs original r8 | 1 | 2 | 1 | 0 |
| supervised r8 vs supervised native | 1 | 1 | 1 | 0 |
| consistency native vs original native | 2 | 2 | 2 | 0 |
| consistency r8 vs original r8 | 1 | 2 | 0 | 0 |
| consistency r8 vs consistency native | 2 | 0 | 0 | 0 |
| consistency native vs supervised native | 1 | 3 | 1 | 1 |
| consistency r8 vs supervised r8 | 0 | 0 | 0 | 1 |
| supervised native vs original r8 | 0 | 1 | 1 | 1 |

## Separate heldout-combination diagnostic

Each selected checkpoint reads the same **4,096 heldout directed edges between 4,096 seen source images**, in **1,024 layout groups**. These edges have zero overlap with the 8,192 training edges; every image, state and layout appeared in training. The two models contribute 8,192 prediction rows on these same 4,096 edges. This population is separate from the original source validation and is not pooled with it.

| Arm | NMAE (%FS) | Acc@1/2/5 (%) | >2%FS / >5%FS | Median (%FS) | P90 (%FS) | P95 (%FS) | Tail10 (%FS) | Maximum (%FS) | Failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| supervised | 0.021373696 | 100 / 100 / 100 | 0 / 0 | 0.012045 | 0.047918 | 0.083366 | 0.097164 | 0.327731 | 0 |
| consistency | 0.023266617 | 100 / 100 / 100 | 0 / 0 | 0.014336 | 0.048073 | 0.074875 | 0.097448 | 0.378931 | 0 |

Inference combines predicted donor U/M with predicted recipient Q/M using native expectation, with no GT spatial mask, warp or fallback; targets are recipient physical readings. Mass below 0.15, invalid inputs or nonfinite/out-of-range readings would retain normalized error 1.0. All edges are valid here. Ordinary supervision already composes accurately on this diagnostic. Consistency's slightly higher mean error and lower P95 are both retained; low heldout error alone cannot demonstrate the necessity of explicit composition supervision.

## Training and execution cost

Each epoch has **18,590 unique RGB images**, expanded to **22,686 donor items** plus **8,192 recipient forwards**, for **30,878 image forwards** and **709 batches**. All 65 history records in both arms match. Each arm executes **2,007,070 image forwards**, **46,085 batches** and **46,062 effective optimizer updates**. AMP skipped 23 updates in each arm; skipped updates remain in batch and exposure counts. The original baseline has 14,494 unique images, 453 batches/epoch and 29,430 effective updates over 65 epochs.

Supervised has all four state weights zero. Consistency enables `state_field_l1=0.5` and `state_support_l1=0.1` after five warmup epochs, while `state_reading_cdf` and `state_mass_l1` remain zero. Raw losses and `supervised_cross_readouts=8192` eligibility counts do not imply that zero-weight objectives supervise either control. Both donor and recipient receive the same original single-image losses. Comparisons with the original baseline combine added images, repeated exposure and extra optimization batches; the supervised-versus-consistency comparison matches these quantities.

| Arm | Recorded epoch seconds | Whole training-process seconds | Peak recorded CUDA bytes | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| supervised | 14,939.500141 | 15,004.380793 | 31,136,305,152 | 63 |
| consistency | 14,137.293540 | 14,198.642869 | 31,133,880,320 | 58 |

Epoch durations include each epoch's source validation and are recorded before the subsequent checkpoint write. Process wall additionally includes setup, writes and shutdown. Supervised ran **10:13:45–14:23:49**, consistency **14:26:04–18:22:43**, on **2026-10-20 (Asia/Shanghai)**. The original baseline's epoch-duration sum is 5,729.045714 seconds under its smaller exposure protocol.

The following observations use **RTX PRO 6000 Blackwell Server Edition, batch 16, 4 workers and AMP**. Source-validation forward timing includes synchronized model forward and the selected readout. The ROI pipeline additionally includes image reads, annotation-defined ROI crop/resize, normalization, transfer and prediction collection.

| Reader | Readout | Forward ms/image | Forward images/s | ROI pipeline ms/image | ROI pipeline images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| original baseline | native | 6.357 | 157.30 | 9.877 | 101.24 |
| original baseline | modal_r8 | 6.432 | 155.47 | 10.430 | 95.87 |
| supervised | native | 6.778 | 147.54 | 11.013 | 90.80 |
| supervised | modal_r8 | 6.792 | 147.24 | 10.933 | 91.47 |
| consistency | native | 6.552 | 152.63 | 10.333 | 96.78 |
| consistency | modal_r8 | 7.713 | 129.66 | 11.641 | 85.90 |

Heldout costs are **per two-image edge**, with the shared reader's parameters counted once. Each model executes 8,192 image forwards. Supervised records **3.480488 ms/edge** reader forward, **0.091563 ms/edge** composition and **12.007195 ms/edge** ROI pipeline; consistency records **3.299575**, **0.090765** and **11.437717 ms/edge** respectively. Whole process wall seconds for native validation / r8 validation / heldout are **20.220129 / 20.115226 / 94.035810** for supervised and **18.891157 / 20.937055 / 89.083097** for consistency. Model/checkpoint setup, detector inference and export are excluded from the timed ROI segment; process wall includes setup and export. These observations are batch throughput, not single-image latency or a controlled timing benchmark.

## Files

- `summary.json`: partial status, complete metrics, paired deltas and affected IDs, actual fallback rows, per-arm configuration/exposure/effective updates, cost and limitations.
- `metrics.csv`: six source-validation rows and two separately labeled heldout-combination rows.
- `predictions.csv.gz`: **9,036 source-validation rows** (1,506 images × 3 checkpoints × 2 readouts), including normalized target, prediction/error, IDs, validity, support and selected expert.
- `heldout_predictions.csv.gz`: **8,192 edge rows** (the same 4,096 edges × 2 checkpoints), including donor/recipient/layout identity and complete error denominators.
- Raw protocol/queue snapshots, reader config/run/history/best/completed records, original evaluation summaries/CSVs and four completed step logs per arm are archived under `refine-logs/geoprr_mobilenet_tail_20261019/iteration_v3_compositional/{supervised_completed,consistency_completed}`. Baseline raw records remain in the adjacent `iteration_v1` archive. Images, weights and vendor assets are excluded from the public package.

## Limitations

Two of three first-seed arms are complete; composition results and the final native continuation decision remain pending. One seed provides no seed SD, significance or reproducibility estimate. Source validation serves development and checkpoint selection; neither new arm has an official-test, perturbation or real-domain result here. Heldout combinations use seen images/states/layouts and do not establish unseen-layout transfer. The one observed consistency fallback is a retained severe error, not evidence of general fallback reliability. Saved selection scores and batch-16 reevaluation can differ slightly with AMP/batch numerical paths; selected checkpoints remain fixed. Timings include startup/warmup, measure batch throughput and exclude detector inference. A favorable r8 result does not replace the frozen native continuation rule.
