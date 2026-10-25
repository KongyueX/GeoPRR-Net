# Fixed image-structure source-control audit

**Scientific verdict: NOT_RETAINED.** Integrity checks: PASS_WITH_QUALIFICATIONS; 0 numeric mismatches across 11698 comparisons, maximum absolute difference 1.4210854715202004e-14.

The fixed 30E representation fails both native learning predicates and all five native joint predicates. The run stopped normally; candidate CF was not eligible and has null results, zero calls/images and no files.

| Metric | identity15 native | image-structure30 native | constant |
|---|---:|---:|---:|
| Mean %FS | 2.54312182437 | 23.1894797459 | 25.0089274061 |
| Median %FS | 1.54945105314 | 19.3514585495 | 24.896581471 |
| P90 %FS | 4.09151151776 | 48.632774353 | 44.1684859991 |
| P95 %FS | 5.40433466434 | 54.5679415762 | 47.6235611923 |
| Worst ceil(292*0.1)=30 mean %FS | 11.8009553314 | 55.7430449128 | 47.3844161057 |
| Maximum %FS | 53.7970364094 | 68.6384409666 | 49.9422221968 |
| Acc@1 count /292 | 102 | 13 | 7 |
| Acc@2 count /292 | 182 | 28 | 14 |
| Acc@5 count /292 | 276 | 53 | 24 |
| Errors >1%FS | 190 | 279 | 285 |
| Errors >2%FS | 110 | 264 | 278 |
| Errors >5%FS | 16 | 239 | 268 |

The final training epoch saved SmoothL1 is 0.006354563796670438 and training NMAE is 1.760571110543552%FS. These training-set numbers do not pass the held-out native learning gate.

Learning limits are mean <= 12.504463703%FS and tail <= 23.6922080529%FS. Both fail. Native paired rows: improved 26, worsened 266, unchanged 0; groups: improved 0, worsened 73, unchanged 0. Of the reference native 102 rows at <=1%FS, 83 become >5%FS.

| Threshold %FS | Newly above | Rescued | Reference above | Candidate above |
|---|---:|---:|---:|---:|
| 1 | 97 | 8 | 190 | 279 |
| 2 | 165 | 11 | 110 | 264 |
| 5 | 224 | 1 | 16 | 239 |

All four saved scalar states retain 292 exact IDs and 73 groups of four, aligned saved targets, full coverage and zero input/output failures. Prediction 0 or1 is valid under the inclusive range rule; failure error remains1 in the denominator. Float checks use absolute tolerance1e-12; IDs/groups/counts use exact equality. Reference CF is separately verified as an existing reference with its own104-row accurate set; it is not a candidate observation.

The independently recomputed saved numeric summaries, per-group numbers, per-image errors, threshold pairs, group differences, ordinary guard, execution counts and declared reference statistics agree. Raw scalar values are retained in the collected CSVs. No target label was created or re-normalized.

Formal training: 164880 image presentations and 2580 encoder/point/filter calls each; 2580 backward attempts = 2577 updates + 3 AMP skips. Native adds 292 images / 5 calls each. Formal totals: 165172 images and 2585 encoder/point/filter calls each. No CF, source03, Industrial or RF evaluation was performed.

Saved native full-model forward_seconds is 3.5219172462821007s. Training phase wall: 885.743702851s. Native pipeline/phase wall: 5.0427265726s / 5.04454496503s. Filter event time: training 16.5573310032s +native 0.0288711681366s =16.5862021713s. Formal full process wall: 900.88052877s; supervisor source stage:902.043468833s; supervisor whole pipeline:923.498824764s. Formal CUDA peak allocated/reserved:8398941696/11058282496bytes. Isolated point time:null.

Preflight is separate:8 unique RGB images,3 encoder/point/filter calls,1 combined backward,0 optimizer updates. Four ID images were additionally re-decoded for the CPU native-input identity witness; those CPU reads/checks are included in initialization and full wall, not represented as eight decode operations only. Preflight full wall:18.6658721007s; supervisor stage:19.5334707238s; filter event time:0.101651362896s; CUDA peak allocated/reserved:631291392/734003200bytes. No isolated CPU identity-check time is saved.

Static new-code review confirms original batch-mean SmoothL1 beta.05, fixed last30, no intermediate validation/scheduler/best/extension, one fixed measurement pass before the encoder, F32 operator and restoring fixed-filter-only TF32 scope. Saved witnesses cover462 cached encoder states,233 trainable parameter tensors/8435793 parameters,465 full states, one fixed kernel, source BN and strict reload. Checkpoints were not reopened.

Two serialized stages completed with exit0. Source stage ended2026-10-25T21:51:07.564473+00:00. The root completion/idle record at2026-10-25T22:05:46.807129+00:00 has all three processes gone and empty GPU compute rows. Scientific revision: d6b2db442a1d89799dff4024cd6baa26c0f2d28c; parent-supplied runtime revision: 3263eaf90c4b985f78c72ab7484e1dcac6e3c158.

## Known Issues

- **WARN / inherited_CF_condition_last15_text**: Frozen public/runtime config counterfactual_gate.condition says same last15 only; top-level last30 contract and implementation use last30. Candidate CF was not executed, so this inherited diagnostic wording has no observed numeric effect. Keep the frozen config unchanged.
- **QUALIFICATION / provided_label_scope**: Provided-label saved-number comparison only. The prior C3 audit remains149checks148PASS1FAIL: the normalized_progress/y field check was unavailable in the physical-label-triplet source contract. This is inherited raw-label-derivation coverage, not a discovered label conflict; no old audit or labels are recomputed.
- **QUALIFICATION / saved_witness_scope**: Saved scalar/metadata and new-code static audit only; no model/torch/weights/RGB/GPU/old tests/raw-label derivation rerun. Historical pixel/full-weight provenance is not newly certified.
- **QUALIFICATION / summary_column_scope**: Numeric/null summaries and collected numeric/null group columns are checked. Derived maximum-error row IDs come from retained rows; saved maximum_error_ids text and redundant groups column were excluded from collection.
- **QUALIFICATION / budget_and_generalization**: One seed; source layouts and an artificial RG-swap reference view. 30E vs15E is a 2x scheduled training budget. No causal representation effect, significance, convergence, real-domain generalization or new blind-test claim.

The result closes this frozen representation/control under its fixed30E schedule. It does not support promotion, budget/kernel/scale/seed tuning, causal attribution, real-domain generalization or a new blind-test claim. The original frozen config must be archived unchanged with the inherited CF text qualification stated in the archive README.

Artifacts: audit_result.json, checks.csv and numeric_comparisons.csv contain check-level evidence; metric_comparisons.csv, native_paired_rows.csv, native_paired_groups.csv and native_threshold_comparisons.csv retain unfavorable results and exact pair counts. Collected metadata and CSVs contain only permitted scalar observations and normalized-safe metadata.
