# MobileNetV4 expert/detail iteration: validation decision

The completed `detail_ce` model did not improve the original MobileNetV4. Its best source-validation NMAE is **0.419775%FS**, versus **0.191741%FS** for the original reader: an increase of 0.228035%FS, or 118.93% relative. The source-validation continuation gate therefore selected `return_to_original_mobilenet`. OOF training, routing and the remaining previous experiment matrix are deferred. The next iteration addresses severe errors in the original reader; its protocol is in [GEOPRR_V2_TAIL_ITERATION_CN.md](../../GEOPRR_V2_TAIL_ITERATION_CN.md).

All three readers completed 65 epochs with seed `20262020`, using the frozen 14,494-image source-training and 1,506-image source-validation sets. Checkpoints minimize full-denominator source-validation NMAE. The continuation decision uses the saved original baseline score and the candidate source-validation score only. The controller recorded `validation_rejected` as a normal research decision at 2026-10-19 16:04:19 UTC.

| Reader | Selected epoch | Source-validation NMAE (%FS) | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | Errors >2%FS | Errors >5%FS | Maximum error (%FS) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original MobileNetV4 | 61 | 0.191741 | 98.938 | 99.535 | 99.668 | 7 | 5 | 32.007 |
| global_ce | 60 | 0.351582 | 94.356 | 98.738 | 99.469 | 19 | 8 | 34.853 |
| detail_ce | 64 | 0.419775 | 94.555 | 98.539 | 99.402 | 22 | 9 | 69.628 |

`global_ce` adds stable two-hot cross-entropy to the original direct head. `detail_ce` additionally uses spatial direct pooling and coarse-to-fine Q refinement. At their selected field checkpoints, direct-head source-validation NMAE is 1.218451%FS and 3.036907%FS, with normalized prediction standard deviations of 0.282832 and 0.274454. These distributions are nonconstant, but the complete field reader is less accurate than the original model.

The original and `global_ce` official-test results had already been evaluated on the same 4,000 native SyncG images. `detail_ce` was not evaluated on the official test.

| Reader | Official-test NMAE (%FS) | Acc@1 (%) | Acc@2 (%) | Acc@5 (%) | Errors >2%FS | Errors >5%FS | Maximum error (%FS) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original MobileNetV4 | 0.220749 | 98.925 | 99.400 | 99.600 | 24 | 16 | 67.554 |
| global_ce | 0.345733 | 95.075 | 98.950 | 99.475 | 42 | 21 | 59.628 |
| detail_ce | Not evaluated | — | — | — | — | — | — |

`detail_ce` contains 7,919,575 parameters; the original and `global_ce` readers contain 7,695,954. Its 65 recorded epoch durations sum to 6,246.65 seconds, with a mean joint-training epoch duration of 96.44 seconds.

## Files

- `summary.json`: source-validation decision, configurations, full metrics, direct-head diagnostics and recorded training cost. Errors stored in normalized units also have explicitly named %FS fields.
- `metrics.csv`: one row per model and split; the unevaluated `detail_ce` test row has empty metric cells.
- The original MobileNetV4 test predictions and baseline configuration are archived in [geoprr_v2_syncg_20261019](../geoprr_v2_syncg_20261019/README.md). Raw `global_ce` and `detail_ce` JSON/history records and the complete gate decision are retained in the local experiment archive named in `summary.json`.

## Limitations

These configurations have one completed seed, so no three-seed mean, sample standard deviation or statistical significance claim is available. The official test had been inspected during historical development; its existing native-image scores are exploratory comparisons and do not establish a new blind-test, perturbation or real-domain result. The rejected candidate's official-test score and the deferred OOF, routing, dynamic-convolution and wider-control results are unavailable. Recorded epoch durations measure training compute; complete inference latency was not measured for this phase. A nonconstant direct head does not explain the cause of the field-reader regression.
