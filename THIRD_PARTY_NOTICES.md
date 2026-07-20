# Third-party data and model notice

This repository does not currently declare a project-level software license.
Before a public release, the maintainer must choose a license compatible with
every redistributed dependency and model artifact. This file records observed
metadata; it does not grant any additional rights.

## Local model artifacts (not tracked by Git)

| Artifact | SHA-256 | Observed provenance/license metadata |
|---|---|---|
| `utils/angleDetect/pointerSeg/resultSeg/best.pt` | `27a48bd42cdce19949f2b2fc61746b801aec206d98320e338571d0c0545a34f8` | Raw U2NetP state dictionary; no provenance or license is embedded. Confirm the original training data and redistribution rights before publication. |
| `utils/angleDetect/vitTranforms/result/best.pt` | `e7592c7f0d782674b645d4b331845e313390d0b1aa756a637de7d798cab56360` | Raw transformer state dictionary; no provenance or license is embedded. Confirm the original training data and redistribution rights before publication. |
| `utils/angleDetect/yoloDetection/result/yolo_findMeter.pt` | `98b8f40cba170b40f828bd0579e6a8a5feba500fd83261a13651ab428c0956ab` | Ultralytics 8.3.82 checkpoint metadata declares `AGPL-3.0 (https://ultralytics.com/license)`. |
| `utils/angleDetect/yoloDetection/result/yolo_pointbest.pt` | `2cb5c2523e364063ccdfd5c047390f17986ebcb622cef09d8604dfdaf038bdd6` | Ultralytics 8.3.193 checkpoint metadata declares `AGPL-3.0 (https://ultralytics.com/license)`. |

The SyncG-fine-tuned segmentation checkpoint is derived from the first
artifact. Do not publish that derived checkpoint until the base artifact's
license and training-data provenance have been established.

## Public datasets

- SyncG is downloaded from the pinned Hugging Face commit recorded by the
  experiment scripts. Its data card declares CC BY 4.0. The images are ignored
  by Git and are not redistributed here.
- RPM-10K is downloaded from the official DialBench release. The upstream
  repository currently marks the dataset license as TBD. Its images are
  ignored by Git and must not be redistributed from this repository without
  additional permission.

Generated manifests, prediction caches, fitted calibrators, and raw result
tables are also ignored by Git by default. A small, manually reviewed snapshot
of aggregate formal results is retained in `docs/FORMAL_RESULTS_CN.md`; it
contains no images, per-sample predictions, labels, or model parameters. Review
the target venue's artifact policy before publishing any generated model or
data-derived artifact.
