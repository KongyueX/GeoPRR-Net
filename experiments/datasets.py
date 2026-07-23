"""Build one common JSONL manifest for SyncG, RPM-10K, and RealGauges.

The heavy inference code consumes only this manifest. Dataset-specific parsing
therefore happens once, and all methods are evaluated on exactly the same
sample list.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from experiments.extract_pointer10k_test import (
    POINTER10K_ARCHIVE_SHA256,
    POINTER10K_TEST_ANNOTATION_SHA256,
    POINTER10K_TEST_IMAGES,
    POINTER10K_TEST_POINTERS,
    POINTER10K_TEST_SINGLE_POINTER_IMAGES,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
RPM10K_PRIMARY_METER_TYPES = frozenset(
    {"biao1", "biao2", "biao3", "biao4", "biao5", "biao6"}
)
RPM10K_TEST_SHA256 = (
    "ddfe8a7cee13fba71639ced8a55ccb362851d91b30b4dd7cb22718b9236d49dc"
)
RPM10K_TEST_ROWS = 2000
RPM10K_SINGLE_POINTER_ROWS = 1797
SYNCG_TRAIN_ROWS = 16_000
SYNCG_TEST_ROWS = 4_000
SYNCG_PINNED_COMMIT = "14204c3f5b35d160fafa39ad195cd5a63e6e9c12"
SYNCG_SAMPLE_IDS_SHA256 = {
    "train": "6c1bcfd7a6a83c07e6a8d6c133f0fc48abed543ca02d02c6ee46804240216e75",
    "test": "89387845185c4dc8d653f411f9b011cbf99672c254b2ed35be55c71ef178efbe",
}
POINTER10K_PINNED_COMMIT = "68afe1efbdb35d3196d9a6243bfac8e5c9de5ceb"
POINTER10K_TEST_IDS_SHA256 = (
    "41b7d63c194aeaee859b388ecbdc933ff1fd0938cf3582a9859909fc7d0c5001"
)
POINTER10K_SINGLE_POINTER_IDS_SHA256 = (
    "72e9421ec0778445a367939adbb4672190b4649e24384e2ef0f6b37004b7d76b"
)


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def _pick(row: dict[str, Any], aliases: Iterable[str], default: Any = None) -> Any:
    for key in aliases:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
        return rows
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("samples", value.get("rows", value.get("data", value)))
        if not isinstance(value, list):
            raise ValueError(f"{path} must contain a JSON list or a samples/rows/data list")
        if not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{path} contains non-object rows")
        return value
    raise ValueError(f"unsupported label format: {path.suffix}")


def _write_jsonl(rows: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Hash a split identity independently of filesystem order."""
    payload = json.dumps(
        sorted(str(sample_id) for sample_id in sample_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def syncg_sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Backward-compatible name for the pinned SyncG split identity hash."""
    return sample_ids_sha256(sample_ids)


def _build_media_index(root: Path, suffixes: set[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        resolved = path.resolve()
        for key in (path.name.lower(), path.stem.lower(), path.as_posix().lower()):
            index.setdefault(key, resolved)
        try:
            relative = path.relative_to(root).as_posix().lower()
            index.setdefault(relative, resolved)
        except ValueError:
            pass
    return index


def _resolve_media(
    root: Path,
    value: Any,
    index: dict[str, Path],
    *,
    expected: str,
) -> Path:
    if value in (None, ""):
        raise ValueError(f"missing {expected} path")
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [root / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    keys = [str(value).replace("\\", "/").lower(), raw.name.lower(), raw.stem.lower()]
    for key in keys:
        if key in index:
            return index[key]
    raise FileNotFoundError(f"cannot resolve {expected} {value!r} below {root}")


def _find_syncg_root(root: Path) -> Path:
    candidates = [root, root / "syncG", root / "SyncG"]
    for candidate in candidates:
        if (candidate / "annotations").is_dir() and (candidate / "images").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"SyncG root not found below {root}; expected annotations/<split> and images/<split>"
    )


def build_syncg_manifest(
    root: Path,
    split: str,
    output: Path,
    limit: int | None = None,
    *,
    strict_release: bool = True,
) -> int:
    if split not in {"train", "test"}:
        raise ValueError(f"unsupported SyncG split: {split!r}")
    root = _find_syncg_root(root)
    annotation_dir = root / "annotations" / split
    image_dir = root / "images" / split
    if not annotation_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError(f"SyncG split {split!r} is incomplete below {root}")

    image_index = _build_media_index(image_dir, IMAGE_SUFFIXES)
    annotation_paths = sorted(annotation_dir.glob("*.json"))
    expected_rows = SYNCG_TRAIN_ROWS if split == "train" else SYNCG_TEST_ROWS
    if strict_release and limit is None and len(annotation_paths) != expected_rows:
        raise ValueError(
            f"SyncG {split} has {len(annotation_paths)} annotations; "
            f"expected {expected_rows}. Pass --allow-protocol-drift only "
            "for a clearly labelled diagnostic."
        )
    release_ids_sha256 = syncg_sample_ids_sha256(
        path.stem for path in annotation_paths
    )
    expected_ids_sha256 = SYNCG_SAMPLE_IDS_SHA256[split]
    if (
        strict_release
        and limit is None
        and release_ids_sha256 != expected_ids_sha256
    ):
        raise ValueError(
            f"SyncG {split} sample identifiers do not match pinned release "
            f"{SYNCG_PINNED_COMMIT}: expected {expected_ids_sha256}, "
            f"got {release_ids_sha256}. Pass --allow-protocol-drift only "
            "for a clearly labelled diagnostic."
        )
    if limit is not None:
        annotation_paths = annotation_paths[: max(0, limit)]

    def rows():
        for annotation_path in annotation_paths:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not isinstance(annotation, dict):
                raise ValueError(f"{annotation_path} is not a JSON object")

            file_name = annotation.get("file_name") or f"{annotation_path.stem}.jpg"
            image_path = _resolve_media(image_dir, file_name, image_index, expected="image")
            ground_truth = _as_float(annotation.get("ground_truth"), "ground_truth")
            scale_start = _as_float(annotation.get("start_value"), "start_value")
            interval_value = _as_float(
                annotation.get("long_interval_value"),
                "long_interval_value",
            )
            long_num = int(_as_float(annotation.get("long_num"), "long_num"))
            if long_num < 2:
                raise ValueError(f"{annotation_path}: long_num must be at least 2")
            # SyncG stores the number of major ticks, including both endpoints.
            scale_end = scale_start + (long_num - 1) * interval_value

            scene_name = str(annotation.get("scene_name") or "unknown_scene")
            gauge_type = str(annotation.get("gauge_type") or "unknown_type")
            yield {
                "dataset": "SyncG",
                "split": split,
                "sample_id": annotation_path.stem,
                "group_id": f"{gauge_type}::{Path(scene_name).stem}",
                "meter_id": gauge_type,
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "scale_start": scale_start,
                "scale_end": scale_end,
                "metadata": {
                    "annotation_path": str(annotation_path.resolve()),
                    "scene_name": scene_name,
                    "gauge_type": gauge_type,
                    "pointer_angle": annotation.get("pointer_rotate_degree"),
                    "long_interval_degree": annotation.get("long_interval_degree"),
                    "long_interval_value": interval_value,
                    "long_num": long_num,
                    "dial_bbox": annotation.get("dial_bbox_annotations"),
                    "keypoints": annotation.get("keypoints_annotations"),
                    "homography": annotation.get("homo_matrix"),
                },
            }

    count = _write_jsonl(rows(), output)
    sample_ids = [path.stem for path in annotation_paths]
    sample_ids_sha256 = syncg_sample_ids_sha256(sample_ids)
    release_identity_verified = bool(
        strict_release
        and limit is None
        and count == expected_rows
        and sample_ids_sha256 == expected_ids_sha256
    )
    protocol = {
        "protocol": "syncg_official_split_v1",
        "dataset": "SyncG",
        "split": split,
        "syncg_root": str(root),
        "reference_huggingface_commit": SYNCG_PINNED_COMMIT,
        "release_identity_verified": release_identity_verified,
        "dataset_license": "CC-BY-4.0",
        "strict_release": bool(strict_release and limit is None),
        "expected_rows": expected_rows,
        "emitted_rows": count,
        "sample_ids_sha256": sample_ids_sha256,
        "expected_sample_ids_sha256": expected_ids_sha256,
        "scale_policy": (
            "scale_start=start_value; "
            "scale_end=start_value+(long_num-1)*long_interval_value"
        ),
    }
    output.with_name(output.name + ".protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def _find_pointer10k_root(root: Path) -> Path:
    candidates = [
        root,
        root / "pointer_10k",
        root / "Database" / "Done" / "pointer_10k",
    ]
    for candidate in candidates:
        if (
            (candidate / "annotations" / "ann_test_pointer.json").is_file()
            and (candidate / "images" / "test_pointer").is_dir()
        ):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Pointer-10K test root not found below {root}; expected "
        "annotations/ann_test_pointer.json and images/test_pointer"
    )


def select_pointer10k_single_pointer_rows(
    coco: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]]:
    """Select the label-only single-pointer applicability domain.

    Pointer-10K contains one to three pointers per dial, whereas the current
    reader emits exactly one direction.  The subset is therefore fixed from
    the official annotation count before any model is evaluated.
    """

    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("Pointer-10K annotation lacks COCO images/annotations lists")
    if not all(isinstance(item, dict) for item in images + annotations):
        raise ValueError("Pointer-10K COCO images/annotations contain non-object rows")

    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        by_image.setdefault(image_id, []).append(annotation)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pointer_count_distribution: Counter[int] = Counter()
    seen_image_ids: set[int] = set()
    for image in sorted(images, key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        if image_id in seen_image_ids:
            raise ValueError(f"duplicate Pointer-10K image id: {image_id}")
        seen_image_ids.add(image_id)
        image_annotations = by_image.get(image_id, [])
        pointer_count_distribution[len(image_annotations)] += 1
        if len(image_annotations) == 1:
            selected.append((image, image_annotations[0]))

    unknown_image_ids = set(by_image) - seen_image_ids
    if unknown_image_ids:
        raise ValueError(
            "Pointer-10K annotations reference unknown image ids; "
            f"first: {min(unknown_image_ids)}"
        )

    audit = {
        "protocol": "pointer10k_official_test_single_pointer_v1",
        "derived_subset": True,
        "selection_uses_predictions": False,
        "selection_rule": "retain official test images with exactly one pointer annotation",
        "source_images": len(images),
        "source_pointer_instances": len(annotations),
        "included_images": len(selected),
        "excluded_multi_pointer_images": len(images) - len(selected),
        "pointer_count_distribution": {
            str(key): int(value)
            for key, value in sorted(pointer_count_distribution.items())
        },
    }
    return selected, audit


def _read_pointer10k_image(path: Path) -> np.ndarray:
    payload = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"cannot decode Pointer-10K image: {path}")
    return image


def _pointer10k_quality_features(
    image_path: Path,
    bbox_xyxy: tuple[float, float, float, float],
) -> dict[str, float]:
    """Compute prediction-independent natural-image quality descriptors."""

    image = _read_pointer10k_image(image_path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = int(np.clip(math.floor(x1), 0, max(0, width - 1)))
    top = int(np.clip(math.floor(y1), 0, max(0, height - 1)))
    right = int(np.clip(math.ceil(x2), left + 1, width))
    bottom = int(np.clip(math.ceil(y2), top + 1, height))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"Pointer-10K bbox produced an empty crop: {image_path}")
    resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    bbox_width = max(float(x2 - x1), 1.0)
    bbox_height = max(float(y2 - y1), 1.0)
    return {
        "laplacian_variance_256": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luminance_mean_256": float(np.mean(gray)),
        "luminance_std_256": float(np.std(gray)),
        "dial_area_ratio": float(
            (bbox_width * bbox_height) / max(float(width * height), 1.0)
        ),
        "dial_short_side_ratio": float(
            min(bbox_width, bbox_height) / max(float(min(width, height)), 1.0)
        ),
    }


def _freeze_pointer10k_quality_groups(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    if not rows:
        raise ValueError("Pointer-10K single-pointer subset is empty")
    definitions = {
        "natural_blur_q1": "laplacian_variance_256",
        "natural_low_light_q1": "luminance_mean_256",
        "natural_low_contrast_q1": "luminance_std_256",
        "natural_small_dial_q1": "dial_short_side_ratio",
    }
    thresholds = {
        group: float(
            np.quantile(
                [float(row["metadata"]["quality"][field]) for row in rows],
                0.25,
            )
        )
        for group, field in definitions.items()
    }
    group_counts: Counter[str] = Counter()
    for row in rows:
        quality = row["metadata"]["quality"]
        groups = [
            group
            for group, field in definitions.items()
            if float(quality[field]) <= thresholds[group]
        ]
        quality["challenge_count_q1"] = len(groups)
        if len(groups) >= 2:
            groups.append("natural_low_quality_2of4")
        row["metadata"]["quality_groups"] = groups
        for group in groups:
            group_counts[group] += 1
    return thresholds, dict(sorted(group_counts.items()))


def build_pointer10k_manifest(
    root: Path,
    output: Path,
    *,
    limit: int | None = None,
    strict_release: bool = True,
) -> int:
    """Build the frozen, zero-shot Pointer-10K single-pointer test manifest."""

    root = _find_pointer10k_root(root)
    annotation_path = root / "annotations" / "ann_test_pointer.json"
    extraction_audit_path = root / "extraction.json"
    extraction_audit: dict[str, Any] | None = None
    if extraction_audit_path.is_file():
        candidate = json.loads(extraction_audit_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise ValueError("Pointer-10K extraction audit must be a JSON object")
        extraction_audit = candidate
    annotation_sha256 = _sha256(annotation_path)
    coco = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(coco, dict):
        raise ValueError("Pointer-10K test annotation must be a COCO JSON object")
    selected, audit = select_pointer10k_single_pointer_rows(coco)

    all_ids_hash = sample_ids_sha256(str(item["id"]) for item in coco["images"])
    single_ids_hash = sample_ids_sha256(str(item["id"]) for item, _ in selected)
    if strict_release:
        problems = []
        if extraction_audit is None:
            problems.append(
                "test-only extraction.json is missing; use "
                "experiments.extract_pointer10k_test"
            )
        else:
            expected_audit_values = {
                "protocol": "pointer10k_official_test_extraction_v1",
                "archive_sha256": POINTER10K_ARCHIVE_SHA256,
                "archive_identity_verified": True,
                "train_or_validation_extracted": False,
                "test_images": POINTER10K_TEST_IMAGES,
                "test_pointer_instances": POINTER10K_TEST_POINTERS,
                "single_pointer_images": POINTER10K_TEST_SINGLE_POINTER_IMAGES,
            }
            for field, expected in expected_audit_values.items():
                if extraction_audit.get(field) != expected:
                    problems.append(
                        f"extraction audit {field} is "
                        f"{extraction_audit.get(field)!r}, expected {expected!r}"
                    )
            audited_root = extraction_audit.get("output_root")
            if not isinstance(audited_root, str) or Path(audited_root).resolve() != root:
                problems.append(
                    f"extraction audit output_root is {audited_root!r}, "
                    f"expected {str(root)!r}"
                )
        if annotation_sha256 != POINTER10K_TEST_ANNOTATION_SHA256:
            problems.append(
                f"annotation SHA-256 is {annotation_sha256}, "
                f"expected {POINTER10K_TEST_ANNOTATION_SHA256}"
            )
        if len(coco["images"]) != POINTER10K_TEST_IMAGES:
            problems.append(
                f"test split has {len(coco['images'])} images, "
                f"expected {POINTER10K_TEST_IMAGES}"
            )
        if len(coco["annotations"]) != POINTER10K_TEST_POINTERS:
            problems.append(
                f"test split has {len(coco['annotations'])} pointers, "
                f"expected {POINTER10K_TEST_POINTERS}"
            )
        if len(selected) != POINTER10K_TEST_SINGLE_POINTER_IMAGES:
            problems.append(
                f"single-pointer protocol selected {len(selected)} images, "
                f"expected {POINTER10K_TEST_SINGLE_POINTER_IMAGES}"
            )
        if all_ids_hash != POINTER10K_TEST_IDS_SHA256:
            problems.append(
                f"test image-id hash is {all_ids_hash}, "
                f"expected {POINTER10K_TEST_IDS_SHA256}"
            )
        if single_ids_hash != POINTER10K_SINGLE_POINTER_IDS_SHA256:
            problems.append(
                f"single-pointer image-id hash is {single_ids_hash}, "
                f"expected {POINTER10K_SINGLE_POINTER_IDS_SHA256}"
            )
        if problems:
            raise ValueError(
                "Pointer-10K release/protocol drift detected: " + "; ".join(problems)
            )

    image_dir = root / "images" / "test_pointer"
    rows: list[dict[str, Any]] = []
    for image, annotation in selected:
        sample_id = f"{int(image['id']):012d}"
        file_name = str(image["file_name"])
        image_path = (image_dir / file_name).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Pointer-10K test image is missing: {image_path}"
            )
        bbox = annotation.get("bbox")
        keypoints = annotation.get("keypoints")
        if not (
            isinstance(bbox, list)
            and len(bbox) >= 4
            and isinstance(keypoints, list)
            and len(keypoints) >= 9
        ):
            raise ValueError(
                f"Pointer-10K sample {sample_id} lacks bbox or three keypoints"
            )
        x, y, width, height = map(float, bbox[:4])
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Pointer-10K sample {sample_id} has an invalid bbox")
        bbox_xyxy = (x, y, x + width, y + height)
        tip = [float(keypoints[0]), float(keypoints[1])]
        midpoint = [float(keypoints[3]), float(keypoints[4])]
        tail = [float(keypoints[6]), float(keypoints[7])]
        if math.hypot(tip[0] - tail[0], tip[1] - tail[1]) <= 1e-8:
            raise ValueError(
                f"Pointer-10K sample {sample_id} has coincident tip and tail"
            )
        rows.append(
            {
                "dataset": "Pointer-10K",
                "split": "official_test_single_pointer",
                "sample_id": sample_id,
                "group_id": sample_id,
                "meter_id": "unknown",
                "image_path": str(image_path),
                "ground_truth": None,
                "scale_start": None,
                "scale_end": None,
                "metadata": {
                    "source_annotation": str(annotation_path.resolve()),
                    "annotation_id": int(annotation["id"]),
                    "image_id": int(image["id"]),
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "pointer_count": 1,
                    "dial_bbox": list(bbox_xyxy),
                    "dial_bbox_xywh": [x, y, width, height],
                    "keypoints": [
                        {
                            "type": "pointer",
                            "outside_kp": tip,
                            "midpoint_kp": midpoint,
                            "origin_kp": tail,
                        }
                    ],
                    "quality": _pointer10k_quality_features(
                        image_path,
                        bbox_xyxy,
                    ),
                },
            }
        )

    quality_thresholds, quality_group_counts = _freeze_pointer10k_quality_groups(rows)
    emitted_rows = rows if limit is None else rows[: max(0, limit)]
    count = _write_jsonl(emitted_rows, output)
    audit.update(
        {
            "dataset": "Pointer-10K",
            "split": "official_test_single_pointer",
            "pointer10k_root": str(root),
            "source_annotation": str(annotation_path.resolve()),
            "source_annotation_sha256": annotation_sha256,
            "source_archive_sha256": (
                extraction_audit.get("archive_sha256")
                if extraction_audit is not None
                else None
            ),
            "extraction_audit": (
                str(extraction_audit_path.resolve())
                if extraction_audit is not None
                else None
            ),
            "extraction_audit_sha256": (
                _sha256(extraction_audit_path)
                if extraction_audit is not None
                else None
            ),
            "reference_vdn_commit": POINTER10K_PINNED_COMMIT,
            "dataset_license": "CC-BY-NC-SA-4.0",
            "strict_release": bool(strict_release),
            "release_identity_verified": bool(strict_release),
            "expected_test_images": POINTER10K_TEST_IMAGES,
            "expected_test_pointer_instances": POINTER10K_TEST_POINTERS,
            "expected_single_pointer_images": POINTER10K_TEST_SINGLE_POINTER_IMAGES,
            "all_test_image_ids_sha256": all_ids_hash,
            "single_pointer_image_ids_sha256": single_ids_hash,
            "emitted_rows": count,
            "diagnostic_limit": limit,
            "training_or_fine_tuning_allowed": False,
            "pointer10k_training_images_used": 0,
            "evaluation_role": (
                "auxiliary zero-shot pointer direction benchmark; "
                "not a final scalar-reading benchmark"
            ),
            "crop_policy": "official ground-truth dial bbox, converted xywh to xyxy",
            "quality_strata_policy": (
                "prediction-independent bottom quartile on the complete 438-image "
                "single-pointer subset; combined group requires at least 2 of 4 flags"
            ),
            "quality_feature_policy": (
                "official dial bbox resized to 256x256; Laplacian variance, mean "
                "luminance, luminance std, and dial short-side ratio"
            ),
            "quality_thresholds_q25": quality_thresholds,
            "quality_group_counts": quality_group_counts,
        }
    )
    output.with_name(output.name + ".protocol.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def select_rpm10k_single_pointer_rows(
    label_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the predeclared RPM-10K single-pointer evaluation protocol.

    RPM-10K's test split contains free-form multi-dial answers and web-sourced
    ``others`` meters in addition to its six primary, zero-based dial types.
    The current reader emits one scalar for one pointer, so those samples are
    outside its task definition. Filtering happens only from label schema
    fields and never from model predictions.
    """

    selected: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for row_number, row in enumerate(label_rows, 1):
        meter_type = str(row.get("meter_type") or "").strip()
        if meter_type not in RPM10K_PRIMARY_METER_TYPES:
            exclusions["unsupported_meter_type"] += 1
            continue

        try:
            ground_truth = _as_float(row.get("reading"), "reading")
            scale_end = _as_float(row.get("range"), "range")
        except ValueError:
            exclusions["non_scalar_reading"] += 1
            continue
        if scale_end <= 0.0 or not 0.0 <= ground_truth <= scale_end:
            exclusions["outside_zero_based_range"] += 1
            continue

        normalized = dict(row)
        normalized["_source_row"] = row_number
        normalized["_ground_truth"] = ground_truth
        normalized["_scale_end"] = scale_end
        selected.append(normalized)
        type_counts[meter_type] += 1

    audit = {
        "protocol": "rpm10k_single_pointer_six_primary_types_v1",
        "derived_subset": True,
        "selection_uses_predictions": False,
        "source_rows": len(label_rows),
        "included_rows": len(selected),
        "excluded_rows": int(sum(exclusions.values())),
        "exclusions": dict(sorted(exclusions.items())),
        "included_by_meter_type": dict(sorted(type_counts.items())),
        "scale_policy": "scale_start=0; scale_end=official range field",
        "supported_meter_types": sorted(RPM10K_PRIMARY_METER_TYPES),
    }
    return selected, audit


def build_rpm10k_manifest(
    root: Path,
    labels: Path,
    output: Path,
    *,
    split: str = "external_test",
    limit: int | None = None,
    strict_release: bool = True,
) -> int:
    """Build the frozen RPM-10K single-pointer external-test manifest."""

    root = root.resolve()
    labels = labels.resolve()
    label_rows = _load_rows(labels)
    selected, audit = select_rpm10k_single_pointer_rows(label_rows)
    labels_sha256 = _sha256(labels)
    audit["source_labels"] = str(labels)
    audit["source_labels_sha256"] = labels_sha256

    if strict_release:
        problems = []
        if labels_sha256 != RPM10K_TEST_SHA256:
            problems.append(
                f"test.json SHA-256 is {labels_sha256}, expected {RPM10K_TEST_SHA256}"
            )
        if len(label_rows) != RPM10K_TEST_ROWS:
            problems.append(
                f"test.json has {len(label_rows)} rows, expected {RPM10K_TEST_ROWS}"
            )
        if len(selected) != RPM10K_SINGLE_POINTER_ROWS:
            problems.append(
                f"protocol selected {len(selected)} rows, "
                f"expected {RPM10K_SINGLE_POINTER_ROWS}"
            )
        if problems:
            raise ValueError(
                "RPM-10K release/protocol drift detected: " + "; ".join(problems)
            )

    if limit is not None:
        selected = selected[: max(0, limit)]
    audit["emitted_rows"] = len(selected)
    image_index = _build_media_index(root, IMAGE_SUFFIXES)

    def rows():
        seen_ids: set[str] = set()
        for row in selected:
            image_value = row.get("image")
            image_path = _resolve_media(root, image_value, image_index, expected="image")
            sample_id = Path(str(image_value)).stem
            if sample_id in seen_ids:
                raise ValueError(f"duplicate RPM-10K sample_id: {sample_id}")
            seen_ids.add(sample_id)

            meter_type = str(row["meter_type"]).strip()
            environment = [
                value.strip()
                for value in str(row.get("environment_conditions") or "").split(",")
                if value.strip()
            ]
            yield {
                "dataset": "RPM-10K",
                "split": split,
                "sample_id": sample_id,
                # The release exposes meter type but no sequence/instance ID.
                "group_id": meter_type,
                "meter_id": meter_type,
                "image_path": str(image_path),
                "ground_truth": float(row["_ground_truth"]),
                "scale_start": 0.0,
                "scale_end": float(row["_scale_end"]),
                "metadata": {
                    "source_labels": str(labels),
                    "source_row": int(row["_source_row"]),
                    "query": row.get("query"),
                    "meter_type": meter_type,
                    "environment_conditions": environment,
                    "official_range": row.get("range"),
                    "protocol": audit["protocol"],
                },
            }

    count = _write_jsonl(rows(), output)
    protocol_path = output.with_name(output.name + ".protocol.json")
    protocol_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return count


def _load_meter_config(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("meter config must be a JSON object keyed by meter_id")
    return {str(key): dict(item) for key, item in value.items() if isinstance(item, dict)}


def build_realgauges_manifest(
    root: Path,
    labels: Path,
    output: Path,
    meter_config_path: Path | None = None,
    split: str = "external_test",
    limit: int | None = None,
) -> int:
    root = root.resolve()
    label_rows = _load_rows(labels)
    if limit is not None:
        label_rows = label_rows[: max(0, limit)]
    meter_config = _load_meter_config(meter_config_path)
    image_index = _build_media_index(root, IMAGE_SUFFIXES)
    video_index = _build_media_index(root, VIDEO_SUFFIXES)

    def rows():
        for row_number, row in enumerate(label_rows, 1):
            meter_id = str(
                _pick(row, ("meter_id", "gauge_id", "gauge", "meter", "instrument_id"), "unknown_meter")
            )
            per_meter = meter_config.get(meter_id, {})
            scale_start = _pick(
                row,
                ("scale_start", "start_value", "min_value", "minimum", "min"),
                _pick(per_meter, ("scale_start", "start_value", "min_value", "min")),
            )
            scale_end = _pick(
                row,
                ("scale_end", "end_value", "max_value", "maximum", "max"),
                _pick(per_meter, ("scale_end", "end_value", "max_value", "max")),
            )
            ground_truth = _pick(row, ("ground_truth", "reading", "value", "label", "gt"))
            scale_start = _as_float(scale_start, f"row {row_number} scale_start")
            scale_end = _as_float(scale_end, f"row {row_number} scale_end")
            ground_truth = _as_float(ground_truth, f"row {row_number} ground_truth")
            if abs(scale_end - scale_start) <= 1e-12:
                raise ValueError(f"row {row_number}: scale_start equals scale_end")

            image_value = _pick(row, ("image_path", "image", "file_name", "filename", "path"))
            video_value = _pick(row, ("video_path", "video", "video_file", "clip"))
            frame_value = _pick(row, ("frame_index", "frame_id", "frame", "index"))
            timestamp_value = _pick(
                row,
                ("timestamp_seconds", "time_seconds", "timestamp", "time_sec"),
            )
            media: dict[str, Any]
            if image_value not in (None, ""):
                image_path = _resolve_media(root, image_value, image_index, expected="image")
                media = {"image_path": str(image_path)}
                default_id = image_path.stem
            elif video_value not in (None, "") and frame_value not in (None, ""):
                video_path = _resolve_media(root, video_value, video_index, expected="video")
                frame_index = int(_as_float(frame_value, f"row {row_number} frame_index"))
                if frame_index < 0:
                    raise ValueError(f"row {row_number}: frame_index must be non-negative")
                media = {"video_path": str(video_path), "frame_index": frame_index}
                default_id = f"{video_path.stem}_{frame_index:08d}"
            elif video_value not in (None, "") and timestamp_value not in (None, ""):
                video_path = _resolve_media(root, video_value, video_index, expected="video")
                timestamp_seconds = _as_float(
                    timestamp_value,
                    f"row {row_number} timestamp_seconds",
                )
                if timestamp_seconds < 0:
                    raise ValueError(
                        f"row {row_number}: timestamp_seconds must be non-negative"
                    )
                media = {
                    "video_path": str(video_path),
                    "timestamp_seconds": timestamp_seconds,
                }
                default_id = f"{video_path.stem}_{round(timestamp_seconds * 1000):010d}ms"
            else:
                raise ValueError(
                    f"row {row_number}: provide image_path, video_path+frame_index, "
                    "or video_path+timestamp_seconds"
                )

            video_id = str(
                _pick(
                    row,
                    ("video_id", "sequence_id", "sequence", "clip_id"),
                    Path(str(video_value)).stem if video_value not in (None, "") else "",
                )
            )
            sample_id = str(_pick(row, ("sample_id", "id", "uid"), default_id))
            group_id = str(_pick(row, ("group_id",), video_id or meter_id))
            yield {
                "dataset": "RealGauges",
                "split": split,
                "sample_id": sample_id,
                "group_id": group_id,
                "meter_id": meter_id,
                **media,
                "ground_truth": ground_truth,
                "scale_start": scale_start,
                "scale_end": scale_end,
                "metadata": {
                    "source_labels": str(labels.resolve()),
                    "source_row": row_number,
                    "video_id": video_id or None,
                    "pointer_angle": _pick(row, ("pointer_angle", "angle", "angle_label")),
                },
            }

    return _write_jsonl(rows(), output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    syncg = subparsers.add_parser("syncg", help="convert an official SyncG split")
    syncg.add_argument("--root", type=Path, required=True)
    syncg.add_argument("--split", choices=("train", "test"), required=True)
    syncg.add_argument("--output", type=Path, required=True)
    syncg.add_argument("--limit", type=int)
    syncg.add_argument(
        "--allow-protocol-drift",
        action="store_true",
        help="allow non-official counts (diagnostics only, not paper results)",
    )

    real = subparsers.add_parser(
        "realgauges",
        help="convert RealGauges labels (CSV/JSON/JSONL) and optional videos",
    )
    real.add_argument("--root", type=Path, required=True)
    real.add_argument("--labels", type=Path, required=True)
    real.add_argument("--meter-config", type=Path)
    real.add_argument("--split", default="external_test")
    real.add_argument("--output", type=Path, required=True)
    real.add_argument("--limit", type=int)

    rpm = subparsers.add_parser(
        "rpm10k",
        help="convert the pinned RPM-10K test split to the single-pointer protocol",
    )
    rpm.add_argument("--root", type=Path, required=True)
    rpm.add_argument("--labels", type=Path, required=True)
    rpm.add_argument("--split", default="external_test")
    rpm.add_argument("--output", type=Path, required=True)
    rpm.add_argument("--limit", type=int)
    rpm.add_argument(
        "--allow-protocol-drift",
        action="store_true",
        help="allow non-pinned labels/counts (diagnostics only, not paper results)",
    )

    pointer10k = subparsers.add_parser(
        "pointer10k",
        help="convert the official Pointer-10K test-only single-pointer subset",
    )
    pointer10k.add_argument("--root", type=Path, required=True)
    pointer10k.add_argument("--output", type=Path, required=True)
    pointer10k.add_argument("--limit", type=int)
    pointer10k.add_argument(
        "--allow-protocol-drift",
        action="store_true",
        help="allow non-pinned labels/counts (diagnostics only, not paper results)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "syncg":
        count = build_syncg_manifest(
            args.root,
            args.split,
            args.output,
            args.limit,
            strict_release=not args.allow_protocol_drift,
        )
    elif args.dataset == "realgauges":
        count = build_realgauges_manifest(
            args.root,
            args.labels,
            args.output,
            meter_config_path=args.meter_config,
            split=args.split,
            limit=args.limit,
        )
    elif args.dataset == "rpm10k":
        count = build_rpm10k_manifest(
            args.root,
            args.labels,
            args.output,
            split=args.split,
            limit=args.limit,
            strict_release=not args.allow_protocol_drift,
        )
    else:
        count = build_pointer10k_manifest(
            args.root,
            args.output,
            limit=args.limit,
            strict_release=not args.allow_protocol_drift,
        )
    print(f"wrote {count} samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
