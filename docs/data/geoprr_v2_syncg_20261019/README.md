# Lightweight GeoPRR: single-seed SyncG results

Both readers completed 65 epochs using seed `20262020`. Their best checkpoints were selected at epoch 61 using the same 1,506-image source validation set. Frozen evaluation uses the 4,000 official SyncG test images in their native condition. Test images and labels do not participate in training or checkpoint selection.

| Model | NMAE (%FS) | Acc@2 (%) | Acc@5 (%) | Images above 5%FS | Maximum error (%FS) |
| --- | ---: | ---: | ---: | ---: | ---: |
| original_geoprr | 0.6821 | 95.250 | 99.225 | 31 | 35.08 |
| shared_full_geoprr | 0.6080 | 96.475 | 99.450 | 22 | 31.71 |
| femto | 0.2422 | 99.325 | 99.525 | 19 | 43.55 |
| mobilenetv4 | 0.2207 | 99.400 | 99.600 | 16 | 67.55 |

MobileNetV4 lowers NMAE by 8.84% relative to Femto and by 63.69% relative to the historical shared-full GeoPRR. Its maximum error is higher. These observations concern the complete readers and do not isolate a backbone effect.

## Files

- `metrics.csv`: full-population native-image metrics for two historical readers and two new readers.
- `predictions.csv.gz`: 8,000 new predictions with matched historical errors, normalized targets, support and validity. Multiply errors by 100 for %FS.
- `comparison.json`: configurations, source/validation populations, selected epochs, timing records, pairing checks and numerical comparisons.

Historical errors are joined from `../geoprr_shared_full_20260911/per_sample/syncg.csv.gz` using sample ID, seed and condition. All 4,000 IDs match each new reader. Target differences are at most 2.99e-8 in normalized units, from float32 conversion. `source_group_id` retains the historical 145-group alias mapping; `model_group_id` retains the new reader's 723 gauge-and-scene groups. No confidence interval is reported.

## Limitations

This package contains one seed and native images. Training partitions, input resolution, structural supervision and checkpoint selection differ from the historical protocols. New readers train on 14,494 images and select on 1,506 held-out source images; the historical shared-full reader trains its stages on all 16,000 source images. New inference uses RGB ROI pixels only. The results do not establish perturbation or cross-domain improvement. All new test samples use the field path, while appearance auxiliary losses remain near 0.25, so low-support fallback reliability is unverified. Worst-sample errors are worse for both new readers. Recorded training durations do not measure deployment latency.
