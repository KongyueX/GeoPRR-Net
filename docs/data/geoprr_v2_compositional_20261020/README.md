# Controlled multi-state training: first completed arm

**Status: partial — supervised is complete; 1 of 3 first-seed arms.** The ordinary supervised control lowers native source-validation NMAE from **0.191749420 to 0.147924067%FS** (22.855533% relative reduction), with >5%FS errors decreasing from **5 to 3**. Its frozen `modal_r8` result is **0.129950499%FS**, compared with **0.164085502%FS** for the original baseline. These results establish a stronger control for the remaining consistency and composition arms; they do not establish a benefit from either additional objective.

The model retains the original MobileNetV4 field architecture and **7,695,954 parameters**. Seed **20262020** completed **65 epochs** and selected **epoch 63** by native NMAE on the original 1,506-image source validation. Both readouts use that same checkpoint; the original baseline uses epoch 61. The ±8-bin window was frozen before candidate training. The original train/validation/test lists remain unchanged. A separately recorded 4,096-image supplement is included in this arm's training.

## Complete source validation

All four evaluations contain the same **1,506 IDs, targets and group IDs**, covering 70 groups. Metrics below were recomputed from the archived prediction rows. Error units are percent of full scale (%FS); smaller errors are better.

| Reader | Readout | NMAE | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | >2%FS | >5%FS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original baseline | native | 0.191749420 | 98.937583 | 99.535193 | 99.667995 | 7 | 5 |
| original baseline | modal_r8 | 0.164085502 | 99.203187 | 99.734396 | 99.800797 | 4 | 3 |
| supervised | native | 0.147924067 | 99.269588 | 99.667995 | 99.800797 | 5 | 3 |
| supervised | modal_r8 | 0.129950499 | 99.402390 | 99.667995 | 99.867198 | 5 | 2 |

| Reader | Readout | Median | P90 | P95 | Tail10 NMAE | Maximum |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original baseline | native | 0.067364 | 0.309844 | 0.477729 | 1.204454 | 32.006876 |
| original baseline | modal_r8 | 0.066868 | 0.291035 | 0.456348 | 0.935811 | 27.949108 |
| supervised | native | 0.057125 | 0.239308 | 0.389268 | 0.897473 | 26.627396 |
| supervised | modal_r8 | 0.057386 | 0.237127 | 0.381035 | 0.720578 | 19.557136 |

The native comparison improves every aggregate error metric listed. The same-readout modal comparison improves NMAE by **20.803181%** and reduces >5%FS errors from 3 to 2, but **>2%FS errors increase from 4 to 5**. On the supervised checkpoint, modal readout lowers NMAE by **12.150537%** and maximum error, while the median increases slightly from **0.057125 to 0.057386%FS**. These tradeoffs are retained in the paired records.

## Matched rescues and regressions

A rescue moves from error strictly above the stated threshold to at or below it; a regression moves in the opposite direction. These are paired image-level counts, not additional independent samples.

| Comparison | >2%FS rescues | >2%FS regressions | >5%FS rescues | >5%FS regressions |
| --- | ---: | ---: | ---: | ---: |
| supervised native vs original native | 3 | 1 | 2 | 0 |
| supervised r8 vs original r8 | 1 | 2 | 1 | 0 |
| supervised r8 vs supervised native | 1 | 1 | 1 | 0 |
| supervised native vs original r8 | 0 | 1 | 1 | 1 |

## Separate heldout-combination diagnostic

The selected ordinary supervised checkpoint also reads **4,096 heldout directed edges between 4,096 seen source images**, in **1,024 layout groups**. These edges have zero overlap with the 8,192 training edges; all their images, states and layouts appeared in training. This population is separate from the original source validation and is not pooled with it.

| Edges | Groups | NMAE (%FS) | Acc@1/2/5 (%) | >2%FS / >5%FS | Median (%FS) | P90 (%FS) | P95 (%FS) | Tail10 (%FS) | Maximum (%FS) | Failures |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,024 | 0.021373696 | 100 / 100 / 100 | 0 / 0 | 0.012045 | 0.047918 | 0.083366 | 0.097164 | 0.327731 | 0 |

Inference combines **predicted donor U/M with predicted recipient Q/M**, using full-distribution native expectation. It uses no GT spatial mask, warp or fallback; the target is the recipient's physical reading. Mass below 0.15, invalid input, nonfinite or out-of-range readout would retain normalized error 1.0 in the complete denominator. All evaluated edges are valid here. The ordinary supervised control already composes accurately on this diagnostic, so a low heldout error alone cannot demonstrate that explicit composition supervision is necessary.

## Training and execution cost

Each epoch contains **18,590 unique RGB images**, expanded to **22,686 donor items** plus **8,192 recipient forwards**, for **30,878 image forwards** and **709 batches**. All 65 history records match those counts. Across training this is **2,007,070 image forwards**, **46,085 executed batches** and **46,062 effective optimizer updates**. AMP skipped 23 updates; those batches remain in the image-exposure counts. The original baseline has 14,494 unique images and 453 batches per epoch, with 29,430 effective updates over 65 epochs.

All four state loss weights (`state_field_l1`, `state_support_l1`, `state_reading_cdf`, `state_mass_l1`) are **zero** in this control. Raw state losses and an 8,192-edge `supervised_cross_readouts` eligibility count are recorded after warmup; they do not mean that this arm receives a weighted cross-state objective. Donor and recipient both receive the original single-image supervision. Comparisons with the original baseline combine **added images, repeated exposure and additional optimization batches**, and cannot isolate one factor.

The recorded epoch durations sum to **14,939.500141 seconds (4 h 8 min 59.5 s)**, including each epoch's source validation and excluding the subsequent checkpoint write. Mean joint-training duration is **226.602929 seconds/epoch**. Full training-process wall time, from **10:13:45 to 14:23:49 on 2026-10-20 (Asia/Shanghai)**, is **15,004.380793 seconds (4 h 10 min 4.4 s)**. Peak recorded PyTorch CUDA memory is **31,136,305,152 bytes**. The original baseline's epoch-duration sum is **5,729.045714 seconds**, under its smaller exposure protocol.

The following observations use **RTX PRO 6000 Blackwell Server Edition, batch 16, 4 workers and AMP**. Source-validation forward timing includes synchronized model forward plus the selected readout; the ROI pipeline additionally includes image reads, annotation-defined ROI crop/resize, normalization, transfer and prediction collection. All four source-validation paths select field=1,506, direct=0, mean=0, with no failures.

| Reader | Readout | Forward ms/image | Forward images/s | ROI pipeline ms/image | ROI pipeline images/s |
| --- | --- | ---: | ---: | ---: | ---: |
| original baseline | native | 6.357 | 157.30 | 9.877 | 101.24 |
| original baseline | modal_r8 | 6.432 | 155.47 | 10.430 | 95.87 |
| supervised | native | 6.778 | 147.54 | 11.013 | 90.80 |
| supervised | modal_r8 | 6.792 | 147.24 | 10.933 | 91.47 |

The heldout diagnostic executes the shared reader twice per edge: **8,192 reader image forwards**, **3.480488 ms/edge** for reader forward, **0.091563 ms/edge** for composition and **12.007195 ms/edge** for the two-image ROI pipeline. Shared-reader parameters are counted once. Whole evaluation-process wall times are **20.220129 s** (native validation), **20.115226 s** (r8 validation) and **94.035810 s** (heldout); the timed ROI segments are respectively **16.585154 s**, **16.464683 s** and **49.181470 s**. Process wall also includes setup and export, so it differs from the timed segment.

## Files

- `summary.json`: partial-stage status, frozen protocol, complete metrics, paired deltas and affected IDs, actual training exposure, effective updates, timing boundaries and limitations.
- `metrics.csv`: four source-validation rows and one separately labeled heldout-combination row.
- `predictions.csv.gz`: **6,024 source-validation prediction rows** (1,506 images × 2 checkpoints × 2 readouts), with normalized target, prediction, absolute error, IDs, validity, support and selected expert.
- `heldout_predictions.csv.gz`: **4,096 edge prediction rows**, with donor/recipient/layout identity and the complete validity/error denominator.
- Raw protocol, queue snapshot, reader config/run/history/best/completed records, original summaries, prediction CSVs and step logs are archived under `refine-logs/geoprr_mobilenet_tail_20261019/iteration_v3_compositional/supervised_completed`. Baseline raw records remain under the adjacent `iteration_v1` archive. No images or weights are included in the public package.

## Limitations

Only one of the three first-seed arms is complete. There is no result for explicit consistency or composition training in this package, no multi-seed SD or significance estimate, and no official-test, perturbation or real-domain evaluation. Source validation serves development and checkpoint selection. The saved epoch-63 selection score and the batch-16 reevaluation differ slightly with AMP/batch numerical paths; the selected checkpoint is unchanged. Heldout combinations use seen source images/states/layouts and do not establish unseen-layout or real-domain generalization. All source-validation rows select the field path, so fallback reliability remains untested. Timings include startup/warmup, exclude detector inference, and measure batched throughput rather than batch-one or camera-to-reading latency. The five-condition native continuation rule remains pending the other two arms; the favorable modal result does not replace that rule.
