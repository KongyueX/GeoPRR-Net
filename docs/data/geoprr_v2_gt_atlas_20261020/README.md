# Full source-validation GT-atlas / frozen-Q feasibility

The complete 1,506-image diagnostic shows average-error room for better calibration, but **does not solve the maximum-error objective**. With the ordinary-supervision checkpoint's unchanged Q, GT-atlas NMAE is **0.119645005%FS**, 19.1173% below native and 7.9303% below frozen r8. It reduces native >5%FS errors from 3 to 2 without introducing a new >5%FS case relative to native. However, maximum error increases from **26.627396 to 28.588665%FS**, and one severe error is introduced relative to r8. Continuous GT-H gives nearly identical results; the experiment supplies no evidence that 64 free ray gaps are needed.

This is an annotation-assisted conditional diagnostic, not a trained atlas model or a mathematical performance upper bound. No training, adaptation, new seed, official-test evaluation or real-domain evaluation was performed.

## Fixed protocol

The checkpoint is the completed ordinary-supervision MobileNetV4, seed 20262020, selected epoch 63, with 7,695,954 parameters. All original 1,506 validation IDs, float32 targets and groups are checked against the frozen manifest and formal native/r8 CSVs. One GPU batch-16 AMP pass, 95 batches and 1,506 image forwards, extracts the original `point_probs` Q and reproduces native/r8. Both replays match the archived formal predictions exactly, including every threshold comparison; maximum prediction difference is zero.

GT ROI-to-canonical H is computed in float64 from the existing annotations with the original crop and half-pixel coordinates. Its inverse projects the canonical center and the 65 fixed progress anchors j/64. The chart uses strictly ordered common-center ray bearings, image-arc-midpoint unwrapping, linear inverse interpolation and fixed endpoint clipping. Full normalized Q is read on this chart without predicted M, GT spatial masking, a window, argmax selection or direct fallback. Continuous GT-H/full-Q is a separate representation reference using the same Q. Every input, geometry or readout failure would retain error 1 in the original denominator; all 1,506 inputs and geometries are valid here.

Replacing U/M with a full-Q chart changes both calibration and M conditioning. Therefore the comparison evaluates this particular structural interface; it does not identify a pure causal effect of calibration alone. Targets enter scoring and annotation consistency checks, not Q inference or a sample-specific decoder choice.

## Complete paired results

All error metrics use percent of full scale (%FS). Tail10 contains the worst ceil(0.1×1506)=151 errors.

| Readout on the same checkpoint | NMAE | Acc@1 / @2 / @5 (%) | >2 / >5 | Median | P90 | P95 | Tail10 | Maximum |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary supervision native | 0.147924067 | 99.269588 / 99.667995 / 99.800797 | 5 / 3 | 0.057125 | 0.239308 | 0.389268 | 0.897473 | 26.627396 |
| Ordinary supervision frozen r8 | 0.129950499 | 99.402390 / 99.667995 / 99.867198 | 5 / 2 | 0.057386 | 0.237127 | 0.381035 | 0.720578 | 19.557136 |
| GT-atlas / full Q | 0.119645005 | 99.734396 / 99.800797 / 99.867198 | 3 / 2 | 0.047875 | 0.190001 | 0.275262 | 0.710910 | 28.588665 |
| Continuous GT-H / full Q | 0.119658128 | 99.734396 / 99.800797 / 99.867198 | 3 / 2 | 0.047875 | 0.190137 | 0.275262 | 0.710988 | 28.588665 |

All four readouts have zero failures and 100% valid-prediction coverage. Atlas error decreases on 873 images and increases on 633 relative to native. This is one population viewed through alternative readouts, not four independent samples.

| GT-atlas reference | >2%FS rescued IDs | >2%FS new IDs | >5%FS rescued IDs | >5%FS new IDs |
| --- | --- | --- | --- | --- |
| native | sync_10635, sync_338 | none | sync_338 | none |
| frozen r8 | sync_10635, sync_12540, sync_338 | sync_4515 | sync_338 | sync_4515 |

A rescue moves from strictly above the threshold to at or below it. The following is the union of images exceeding 2%FS in native, r8 or GT-atlas; no severe image is omitted.

| ID | Native error | r8 error | GT-atlas error | Observation |
| --- | ---: | ---: | ---: | --- |
| sync_10202 | 26.627396 | 16.561826 | 28.588665 | Maximum error worsens |
| sync_10635 | 2.455890 | 2.455461 | 0.053573 | Rescued at 2%FS |
| sync_12540 | 1.364576 | 2.511204 | 0.789982 | Rescued relative to r8 |
| sync_338 | 19.577676 | 19.557136 | 0.586411 | Rescued at 5%FS |
| sync_3663 | 2.311166 | 3.186789 | 4.127781 | Existing >2%FS error worsens |
| sync_4515 | 17.546070 | 0.904751 | 22.726863 | New >5%FS error relative to r8 |

## Remaining pointer-evidence errors

Independent analysis of the saved Q and GT charts separates two residual failure types. These are component diagnostics, not scores from a newly selected point decoder.

| ID | Q mass within true ±2%FS | Within true ±5%FS | Progress at highest-Q pixel / target | Interpretation |
| --- | ---: | ---: | --- | --- |
| sync_10202 | 2.1273% | 4.6119% | 0.345797 / 0.181493 | Highest-Q position is already wrong; 81.24% of Q lies above target+5%FS |
| sync_4515 | 13.9984% | 27.5112% | 0.818588 / 0.831575 | Peak position is closer, but 51.71% of Q lies below target−5%FS |

Both images have broad wrong mass, not merely a tiny tail. Their atlas−continuous-H reading differences are about 1e−12 in normalized progress, so their severe errors are not explained by K65 approximation or image/canonical endpoint seams. In contrast, sync_338 places 99.7282% of Q within true ±5%FS; GT calibration rescues its large original error. Sync_3663 has an accurate highest-Q position but only 68.39% of Q within ±5%FS, with 22.07% clipped to the first endpoint. The four cases establish possible failure forms, not their frequency in another dataset.

## Representation and independent verification

All 1,506 GT charts are finite and strictly ordered. Minimum ray-gap fraction is 0.0149189704; no output cell coincides exactly with its projected center. The fixed center convention is atan2(0,0)=0 with full Q retained. Whole-grid arc-membership mismatches are zero. The largest within-arc coordinate approximation error is 0.00135433%FS. The largest atlas−H reading difference is 0.0108652%FS on sync_3663, mainly from outside-arc endpoint selection; it must not be called within-arc interpolation error.

An independent NumPy implementation reconstructed every chart and prediction from saved Q/H, without importing the evaluator, loading dataset items or making model forwards. The largest per-image prediction difference is 2.24e−14. All Q mass columns, paired IDs, metrics and decompositions were verified. The independent pass took 5.079071 seconds with one CPU thread.

## Cost and artifacts

The frozen model forward took 10.391624 seconds, contained within the 16.227480-second inference pipeline including IO, crop, transfer, native/r8 and Q collection. The subsequent CPU chart calculation took 4.899588 seconds, followed by 8.185971 seconds of private cache compression. Whole evaluator wall before summary writing was 31.079176 seconds, including setup; forward must not be added again to its enclosing pipeline or whole-process boundary. Peak CUDA allocated memory was 3,101,364,736 bytes, measured with the model resident during the inference segment. These are batch-throughput observations, not camera-to-reading latency. No extra parameters were trained.

- `predictions.csv.gz`: all 1,506 paired rows, including native/r8 replay and formal references, atlas/H predictions, errors, validity, clipping and seam diagnostics.
- `metrics.csv`: four distinct readout summaries.
- `severe_cases.csv`: complete >2%FS union for native/r8/atlas.
- `summary.json`: full protocol, scores, all paired IDs, replay agreement and resources.
- `independent_verification.json`: independent full-population numerical verification and Q diagnostics.
- Reproduction entry: `python -m experiments.evaluate_geoprr_v2_gt_atlas` with the fixed supervised checkpoint, original manifest, native/r8 reference CSVs and a new output directory.
- Original outputs, invocation and the 69,799,433-byte Q/H cache remain private under `refine-logs/geoprr_method_value_20261020/gt_atlas_full1506_v1/` and the corresponding server diagnostic directory. RGB, weights and the Q cache are excluded from this public package.

In the per-image CSVs, targets, predictions, `*_absolute_error` and reading deltas retain normalized-progress units; Q masses are fractions. Multiply normalized errors by 100 for %FS. `metrics.csv`, explicitly named `*_percent_fs` fields and the tables above already use %FS.

## Limitations

The oracle uses unavailable-at-inference GT geometry and removes M conditioning, so its score is neither an automatic model result nor a guaranteed upper bound. Only one frozen checkpoint and source-development population were used. Lower mean error and fewer threshold exceedances do not establish better maximum-error robustness. K65 and continuous H are numerically equivalent for the important failures, leaving no current evidence that 64 learned free gaps are useful. Predicted calibration, a new pointer-evidence head, independent data and real-domain transfer remain untested. This diagnostic does not authorize or trigger model training or seed expansion.
