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


def syncg_sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    """Hash the official split identity independently of filesystem order."""
    payload = json.dumps(
        sorted(str(sample_id) for sample_id in sample_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    else:
        count = build_rpm10k_manifest(
            args.root,
            args.labels,
            args.output,
            split=args.split,
            limit=args.limit,
            strict_release=not args.allow_protocol_drift,
        )
    print(f"wrote {count} samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
