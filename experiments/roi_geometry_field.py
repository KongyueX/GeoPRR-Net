"""Field-ROI cohorts and RF100 geometry for the ROI comparison experiment."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments.roi_geometry_comparison import decode_progress_from_keypoints


@dataclass(frozen=True, slots=True)
class FieldDataset:
    slug: str
    paper_name: str
    manifest: Path
    labels: Path
    expected_samples: int
    expected_groups: int
    group_unit: str
    geometry_role: str


@dataclass(frozen=True, slots=True)
class FieldTarget:
    sample_id: str
    normalized_target: float
    group_id: str


FIELD_DATASETS: Final[dict[str, FieldDataset]] = {
    "field_gauge_roi_test_a": FieldDataset(
        slug="field_gauge_roi_test_a",
        paper_name="FieldGauge-ROI Test-A",
        manifest=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_development_v1/input_manifest.jsonl"
        ),
        labels=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_development_v1/labels.jsonl"
        ),
        expected_samples=434,
        expected_groups=11,
        group_unit="source-directory group",
        geometry_role="normalized-progress labels only",
    ),
    "field_gauge_roi_test_b": FieldDataset(
        slug="field_gauge_roi_test_b",
        paper_name="FieldGauge-ROI Test-B",
        manifest=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_confirmatory_v1/input_manifest.jsonl"
        ),
        labels=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_confirmatory_v1/labels.jsonl"
        ),
        expected_samples=814,
        expected_groups=20,
        group_unit="source-directory group",
        geometry_role="normalized-progress labels only",
    ),
    "field_gauge_external_roi": FieldDataset(
        slug="field_gauge_external_roi",
        paper_name="FieldGauge-External-ROI",
        manifest=Path(
            "C:/pointer_read/cagh_resnet_xiangmu1_final_v1/input_manifest.jsonl"
        ),
        labels=Path("C:/pointer_read/cagh_resnet_xiangmu1_final_v1/labels.jsonl"),
        expected_samples=147,
        expected_groups=21,
        group_unit="provisional capture session",
        geometry_role="normalized-progress labels only",
    ),
    "rf100": FieldDataset(
        slug="rf100",
        paper_name="RF100-VL",
        manifest=Path(
            "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/input_manifest.jsonl"
        ),
        labels=Path(
            "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/labels.jsonl"
        ),
        expected_samples=151,
        expected_groups=35,
        group_unit="RF100 source/evaluation group",
        geometry_role="annotated center, pointer tip, minimum, and maximum",
    ),
}

RF100_IDENTITY: Final[Path] = Path(
    "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/identity.jsonl"
)
RF100_COCO: Final[Path] = Path(
    "C:/pointer_read/db_gar18_rf100_external_transfer_v1/source/needle-base-tip-min-max/test/_annotations.coco.json"
)


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    with Path(path).resolve().open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            values.append(value)
    if not values:
        raise ValueError(f"empty JSONL input: {path}")
    return tuple(values)


def load_field_targets(dataset: FieldDataset) -> dict[str, FieldTarget]:
    targets: dict[str, FieldTarget] = {}
    for row in _jsonl(dataset.labels):
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        value = float(row.get("normalized_progress"))
        if not sample_id or not group_id or not math.isfinite(value):
            raise ValueError(f"{dataset.slug}: invalid target row for {sample_id!r}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{dataset.slug}: target outside [0,1] for {sample_id}")
        if sample_id in targets:
            raise ValueError(f"{dataset.slug}: duplicate target {sample_id}")
        targets[sample_id] = FieldTarget(sample_id, value, group_id)
    return targets


def load_field_cohort(
    slug: str,
) -> tuple[FieldDataset, tuple[Any, ...], dict[str, FieldTarget]]:
    try:
        dataset = FIELD_DATASETS[str(slug)]
    except KeyError as exc:
        raise ValueError(f"unknown field dataset: {slug}") from exc
    manifest_rows = primary_batch.load_manifest(dataset.manifest)
    targets = load_field_targets(dataset)
    manifest_ids = {row.sample_id for row in manifest_rows}
    if manifest_ids != set(targets):
        raise ValueError(f"{dataset.slug}: manifest and target sample IDs differ")
    groups = {target.group_id for target in targets.values()}
    if len(manifest_rows) != dataset.expected_samples:
        raise ValueError(f"{dataset.slug}: unexpected sample count")
    if len(groups) != dataset.expected_groups:
        raise ValueError(f"{dataset.slug}: unexpected group count")
    return dataset, manifest_rows, targets


def _bbox_center(annotation: Mapping[str, Any]) -> np.ndarray:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        raise ValueError("RF100 annotation bbox is invalid")
    x, y, width, height = map(float, bbox)
    return np.asarray([x + 0.5 * width, y + 0.5 * height], dtype=np.float32)


def load_rf100_keypoints(
    *,
    coco_path: Path = RF100_COCO,
    identity_path: Path = RF100_IDENTITY,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return RF100 points ordered as center, tip, minimum, maximum in ROI pixels."""

    coco = json.loads(Path(coco_path).resolve().read_text(encoding="utf-8"))
    categories = {
        int(row["id"]): str(row["name"])
        for row in coco.get("categories", ())
        if isinstance(row, Mapping)
    }
    required = {"center", "pointer tip", "min", "max"}
    if not required <= set(categories.values()):
        raise ValueError("RF100 COCO categories lack the four required points")
    images = {
        str(row["file_name"]): int(row["id"])
        for row in coco.get("images", ())
        if isinstance(row, Mapping)
    }
    annotations: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in coco.get("annotations", ()):
        if not isinstance(row, Mapping):
            continue
        image_id = int(row["image_id"])
        name = categories[int(row["category_id"])]
        if name in annotations[image_id]:
            raise ValueError(f"RF100 image {image_id}: duplicate {name} annotation")
        annotations[image_id][name] = row

    keypoints: dict[str, np.ndarray] = {}
    for row in _jsonl(identity_path):
        sample_id = str(row.get("sample_id") or "")
        file_name = str(row.get("source_file_name") or "")
        crop = row.get("crop_xyxy")
        if not sample_id or file_name not in images:
            raise ValueError(f"RF100 identity row is invalid for {sample_id!r}")
        if not isinstance(crop, Sequence) or len(crop) != 4:
            raise ValueError(f"{sample_id}: RF100 crop is invalid")
        image_annotations = annotations[images[file_name]]
        if not required <= set(image_annotations):
            raise ValueError(f"{sample_id}: incomplete RF100 point annotations")
        origin = np.asarray([float(crop[0]), float(crop[1])], dtype=np.float32)
        points = np.stack(
            (
                _bbox_center(image_annotations["center"]),
                _bbox_center(image_annotations["pointer tip"]),
                _bbox_center(image_annotations["min"]),
                _bbox_center(image_annotations["max"]),
            )
        ) - origin
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"{sample_id}: RF100 keypoints are invalid")
        keypoints[sample_id] = points.astype(np.float32)

    dataset, _rows, targets = load_field_cohort("rf100")
    if set(keypoints) != set(targets):
        raise ValueError("RF100 geometry and target rosters differ")
    maximum_target_delta = 0.0
    for sample_id, points in keypoints.items():
        progress, failure, _telemetry = decode_progress_from_keypoints(points)
        if progress is None or failure is not None:
            raise ValueError(f"{sample_id}: RF100 annotated geometry is undecodable")
        maximum_target_delta = max(
            maximum_target_delta,
            abs(float(progress) - targets[sample_id].normalized_target),
        )
    if maximum_target_delta > 1.0e-5:
        raise ValueError("RF100 annotated geometry does not reproduce prepared targets")
    return keypoints, {
        "dataset": dataset.paper_name,
        "samples": len(keypoints),
        "keypoint_order": ["pivot", "pointer_tip", "reference_start", "reference_end"],
        "maximum_target_replay_delta": maximum_target_delta,
        "geometry_source": str(Path(coco_path).resolve()),
    }


__all__ = [
    "FIELD_DATASETS",
    "RF100_COCO",
    "RF100_IDENTITY",
    "FieldDataset",
    "FieldTarget",
    "load_field_cohort",
    "load_field_targets",
    "load_rf100_keypoints",
]
