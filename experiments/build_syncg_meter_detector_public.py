"""Build a sealed YOLO meter-detection corpus from public SyncG/train only.

The public images are never copied.  A directory symlink exposes the pinned
SyncG/train image directory below the output root so Ultralytics' standard
``images -> labels`` lookup remains valid.  Train, calibration, and independent
validation inventories are split by physical ``group_id``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.syncg_numeric_ocr import grouped_three_way_split


PROTOCOL = "syncg_public_meter_detector_corpus_v1"
SEAL_PROTOCOL = "syncg_public_meter_detector_corpus_seal_v1"
MANIFEST = PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
MANIFEST_PROTOCOL = MANIFEST.with_name(MANIFEST.name + ".protocol.json")
EXPERIMENT_PROTOCOL_PATH = PROJECT_ROOT / "experiments/syncg_meter_detector_protocol.json"
PUBLIC_IMAGE_ROOT = (PROJECT_ROOT / "datasets/SyncG/syncG/images/train").resolve()
PUBLIC_ANNOTATION_ROOT = (
    PROJECT_ROOT / "datasets/SyncG/syncG/annotations/train"
).resolve()
SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()
DEFAULT_OUTPUT = SAFE_OUTPUT_ROOT / "syncg_meter_detector_public_v1"
BUILDER_PATH = Path(__file__).resolve()
SPLIT_IMPLEMENTATION_PATH = PROJECT_ROOT / "experiments/syncg_numeric_ocr.py"
EXPECTED_SAMPLES = 16_000
EXPECTED_GROUPS = 725


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(Path(root).resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes the allowed public root: {resolved}") from error
    return resolved


def _load_manifest() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(MANIFEST.is_file(), f"missing pinned public manifest: {MANIFEST}")
    _require(MANIFEST_PROTOCOL.is_file(), "missing public manifest protocol")
    protocol = json.loads(MANIFEST_PROTOCOL.read_text(encoding="utf-8"))
    _require(protocol.get("protocol") == "syncg_official_split_v1", "public manifest protocol drift")
    _require(protocol.get("dataset") == "SyncG", "public manifest dataset drift")
    _require(protocol.get("split") == "train", "only SyncG/train is permitted")
    _require(protocol.get("strict_release") is True, "public manifest is not strict")
    _require(protocol.get("release_identity_verified") is True, "public release is unverified")
    _require(
        int(protocol.get("expected_rows", -1)) == EXPECTED_SAMPLES
        and int(protocol.get("emitted_rows", -1)) == EXPECTED_SAMPLES,
        "public manifest protocol inventory drift",
    )
    _require(
        protocol.get("sample_ids_sha256") == protocol.get("expected_sample_ids_sha256"),
        "public manifest sample identity was not verified",
    )
    _require(
        Path(str(protocol.get("syncg_root") or "")).resolve(strict=True)
        == PUBLIC_IMAGE_ROOT.parents[1],
        "public manifest SyncG root drift",
    )
    rows: list[dict[str, Any]] = []
    with MANIFEST.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            _require(isinstance(row, dict), f"manifest line {line_number} is not an object")
            _require(
                row.get("dataset") == "SyncG" and row.get("split") == "train",
                f"manifest line {line_number} escapes SyncG/train",
            )
            rows.append(row)
    _require(len(rows) == EXPECTED_SAMPLES, "SyncG/train image inventory drift")
    _require(len({str(row.get("sample_id")) for row in rows}) == len(rows), "duplicate sample id")
    _require(
        canonical_sha256(sorted(str(row.get("sample_id")) for row in rows))
        == protocol.get("sample_ids_sha256"),
        "public manifest sample-id content drift",
    )
    _require(
        len({str(row.get("group_id")) for row in rows}) == EXPECTED_GROUPS,
        "SyncG/train physical-group inventory drift",
    )
    return rows, protocol


def _load_experiment_protocol() -> dict[str, Any]:
    _require(EXPERIMENT_PROTOCOL_PATH.is_file(), "missing frozen detector experiment protocol")
    value = json.loads(EXPERIMENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "detector experiment protocol is not an object")
    _require(
        value.get("protocol") == "syncg_public_meter_detector_experiment_v1"
        and value.get("status") == "frozen_before_training",
        "detector experiment protocol drift",
    )
    scope = value.get("data_scope") or {}
    _require(
        scope.get("dataset") == "SyncG"
        and scope.get("allowed_split") == "train"
        and int(scope.get("expected_images", -1)) == EXPECTED_SAMPLES
        and int(scope.get("expected_physical_groups", -1)) == EXPECTED_GROUPS,
        "detector experiment data scope drift",
    )
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(path, b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows))


def _write_text(path: Path, lines: Sequence[str]) -> None:
    _atomic_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _dataset_yaml(root: Path, *, validation_partition: str) -> str:
    root_text = root.as_posix()
    return (
        f'path: "{root_text}"\n'
        'train: "splits/train.txt"\n'
        f'val: "splits/{validation_partition}.txt"\n'
        "names:\n"
        "  0: meter\n"
    )


def _normalized_box(
    bbox: Sequence[Any], *, width: int, height: int, sample_id: str
) -> tuple[list[float], str, list[float]]:
    _require(len(bbox) >= 4, f"{sample_id}: dial bbox is incomplete")
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    _require(np.isfinite([x1, y1, x2, y2]).all(), f"{sample_id}: non-finite dial bbox")
    # SyncG contains a small number of valid partially visible dials whose
    # generated annotation extends a few pixels beyond the image.  Detection
    # supervision is necessarily the visible intersection, exactly as the
    # inference runtime clamps predicted boxes to the frame.
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    _require(x1 < x2 and y1 < y2, f"{sample_id}: dial bbox has no visible intersection")
    normalized = [
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    _require(all(0.0 < value <= 1.0 for value in normalized), f"{sample_id}: invalid YOLO box")
    line = "0 " + " ".join(f"{value:.10f}" for value in normalized)
    return normalized, line, [x1, y1, x2, y2]


def _safe_output(path: Path) -> Path:
    output = Path(path).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"detector corpus must stay below {SAFE_OUTPUT_ROOT}") from error
    _require(output != SAFE_OUTPUT_ROOT, "refusing to use broad output root")
    return output


def build_corpus(
    *,
    output_dir: Path,
    split_seed: int,
    calibration_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    output = _safe_output(output_dir)
    _require(not output.exists(), f"refusing to overwrite detector corpus: {output}")
    experiment_protocol = _load_experiment_protocol()
    frozen_partition = experiment_protocol.get("partition") or {}
    _require(
        int(split_seed) == int(frozen_partition.get("seed", -1)),
        "split seed differs from the frozen detector protocol",
    )
    _require(
        math.isclose(
            float(calibration_fraction),
            float(frozen_partition.get("calibration_fraction", -1.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(validation_fraction),
            float(frozen_partition.get("validation_fraction", -1.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "split fractions differ from the frozen detector protocol",
    )
    rows, manifest_protocol = _load_manifest()
    assignment = grouped_three_way_split(
        rows,
        seed=split_seed,
        calibration_fraction=calibration_fraction,
        validation_fraction=validation_fraction,
    )
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    _require(not temporary.exists(), f"temporary build path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        images_link = temporary / "images/public_train"
        images_link.parent.mkdir(parents=True)
        try:
            os.symlink(str(PUBLIC_IMAGE_ROOT), str(images_link), target_is_directory=True)
        except OSError as error:
            raise RuntimeError(
                "Windows directory-symlink creation failed. Run the builder from the configured "
                "administrator PowerShell 7 session; image copying is intentionally disabled."
            ) from error
        _require(images_link.is_symlink(), "public image indirection is not a directory symlink")
        _require(images_link.resolve(strict=True) == PUBLIC_IMAGE_ROOT, "public image symlink target drift")

        label_root = temporary / "labels/public_train"
        label_root.mkdir(parents=True)
        split_rows: dict[str, list[dict[str, Any]]] = {
            "train": [],
            "calibration": [],
            "validation": [],
        }
        annotation_identities: list[dict[str, Any]] = []
        image_content_identities: list[dict[str, Any]] = []
        label_identities: list[dict[str, Any]] = []
        seen_image_paths: set[Path] = set()
        seen_annotation_paths: set[Path] = set()
        clipped_boxes = 0
        for manifest_row in sorted(rows, key=lambda value: str(value["sample_id"])):
            sample_id = str(manifest_row["sample_id"])
            group_id = str(manifest_row["group_id"])
            _require(bool(sample_id and group_id), f"{sample_id}: missing identity")
            metadata = manifest_row.get("metadata") or {}
            image_path = _require_under(
                Path(str(manifest_row.get("image_path") or "")),
                PUBLIC_IMAGE_ROOT,
                label=f"{sample_id}.image",
            )
            annotation_path = _require_under(
                Path(str(metadata.get("annotation_path") or "")),
                PUBLIC_ANNOTATION_ROOT,
                label=f"{sample_id}.annotation",
            )
            _require(image_path not in seen_image_paths, f"{sample_id}: duplicate public image path")
            _require(
                annotation_path not in seen_annotation_paths,
                f"{sample_id}: duplicate public annotation path",
            )
            seen_image_paths.add(image_path)
            seen_annotation_paths.add(annotation_path)
            _require(image_path.stem == sample_id, f"{sample_id}: public image identity drift")
            _require(annotation_path.stem == sample_id, f"{sample_id}: annotation filename drift")
            raw = annotation_path.read_bytes()
            annotation = json.loads(raw.decode("utf-8"))
            _require(isinstance(annotation, dict), f"{sample_id}: annotation is not an object")
            _require(str(annotation.get("file_name")) == sample_id, f"{sample_id}: annotation id drift")
            width, height = int(annotation["width"]), int(annotation["height"])
            _require(width >= 32 and height >= 32, f"{sample_id}: invalid image dimensions")
            with Image.open(image_path) as public_image:
                _require(
                    tuple(public_image.size) == (width, height),
                    f"{sample_id}: public image/header dimension drift",
                )
                _require(bool(public_image.format), f"{sample_id}: public image format is unknown")
            image_bytes = int(image_path.stat().st_size)
            image_sha256 = sha256_file(image_path)
            annotation_bbox = [float(value) for value in annotation["dial_bbox_annotations"][:4]]
            manifest_bbox = [float(value) for value in metadata["dial_bbox"][:4]]
            _require(np.allclose(annotation_bbox, manifest_bbox, atol=1e-6), f"{sample_id}: dial bbox drift")
            normalized, label_line, visible_bbox = _normalized_box(
                annotation_bbox, width=width, height=height, sample_id=sample_id
            )
            clipped_boxes += not np.allclose(annotation_bbox, visible_bbox, atol=1e-9)
            label_path = label_root / f"{sample_id}.txt"
            _write_text(label_path, [label_line])
            partition = assignment[sample_id]
            corpus_image = images_link / image_path.name
            _require(corpus_image.is_file(), f"{sample_id}: public image missing through symlink")
            sample = {
                "sample_id": sample_id,
                "group_id": group_id,
                "partition": partition,
                "image_path": f"images/public_train/{image_path.name}",
                "label_path": f"labels/public_train/{sample_id}.txt",
                "image_width": width,
                "image_height": height,
                "image_bytes": image_bytes,
                "image_sha256": image_sha256,
                "source_annotation_path": annotation_path.name,
                "source_annotation_bytes": len(raw),
                "source_annotation_sha256": hashlib.sha256(raw).hexdigest(),
                "dial_bbox_xyxy": visible_bbox,
                "dial_bbox_xyxy_unclipped_public_annotation": annotation_bbox,
                "dial_bbox_yolo": normalized,
            }
            split_rows[partition].append(sample)
            annotation_identities.append(
                {
                    "sample_id": sample_id,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            image_content_identities.append(
                {"sample_id": sample_id, "bytes": image_bytes, "sha256": image_sha256}
            )
            label_identities.append(
                {"sample_id": sample_id, "sha256": hashlib.sha256((label_line + "\n").encode()).hexdigest()}
            )

        partition_stats: dict[str, Any] = {}
        group_sets: dict[str, set[str]] = {}
        for partition, values in split_rows.items():
            values.sort(key=lambda value: str(value["sample_id"]))
            groups = {str(value["group_id"]) for value in values}
            group_sets[partition] = groups
            samples_name = f"samples_{partition}.jsonl"
            _write_jsonl(temporary / samples_name, values)
            image_lines = [str((temporary / str(value["image_path"])).resolve(strict=True)) for value in values]
            # Replace the temporary prefix with the immutable final corpus prefix.
            image_lines = [
                str(output / Path(str(value["image_path"]))).replace("\\", "/") for value in values
            ]
            _write_text(temporary / f"splits/{partition}.txt", image_lines)
            partition_ids = {str(value["sample_id"]) for value in values}
            partition_label_identities = sorted(
                (
                    identity
                    for identity in label_identities
                    if str(identity["sample_id"]) in partition_ids
                ),
                key=lambda identity: str(identity["sample_id"]),
            )
            partition_image_identities = sorted(
                (
                    identity
                    for identity in image_content_identities
                    if str(identity["sample_id"]) in partition_ids
                ),
                key=lambda identity: str(identity["sample_id"]),
            )
            partition_annotation_identities = sorted(
                (
                    identity
                    for identity in annotation_identities
                    if str(identity["sample_id"]) in partition_ids
                ),
                key=lambda identity: str(identity["sample_id"]),
            )
            _require(
                len(partition_label_identities) == len(values),
                f"{partition} label identity inventory drift",
            )
            _require(
                len(partition_image_identities) == len(values)
                and len(partition_annotation_identities) == len(values),
                f"{partition} public source identity inventory drift",
            )
            partition_stats[partition] = {
                "samples": len(values),
                "groups": len(groups),
                "samples_artifact": samples_name,
                "samples_sha256": sha256_file(temporary / samples_name),
                "image_list": f"splits/{partition}.txt",
                "image_list_sha256": sha256_file(temporary / f"splits/{partition}.txt"),
                "sample_ids_sha256": canonical_sha256([str(value["sample_id"]) for value in values]),
                "group_ids_sha256": canonical_sha256(sorted(groups)),
                "label_inventory_sha256": canonical_sha256(partition_label_identities),
                "image_content_inventory_sha256": canonical_sha256(
                    partition_image_identities
                ),
                "annotation_content_inventory_sha256": canonical_sha256(
                    partition_annotation_identities
                ),
            }
        _require(
            not group_sets["train"].intersection(group_sets["calibration"] | group_sets["validation"])
            and not group_sets["calibration"].intersection(group_sets["validation"]),
            "physical-group leakage across detector partitions",
        )
        _require(sum(value["samples"] for value in partition_stats.values()) == EXPECTED_SAMPLES, "partition inventory drift")

        train_cal_yaml = temporary / "dataset_train_cal.yaml"
        validation_yaml = temporary / "dataset_validation.yaml"
        _atomic_bytes(train_cal_yaml, _dataset_yaml(output, validation_partition="calibration").encode())
        _atomic_bytes(validation_yaml, _dataset_yaml(output, validation_partition="validation").encode())
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "scope": {
                "dataset": "SyncG",
                "split": "train",
                "annotation": "dial_bbox_annotations",
                "public_data_only": True,
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
            },
            "source": {
                "manifest": str(MANIFEST.relative_to(PROJECT_ROOT).as_posix()),
                "manifest_sha256": sha256_file(MANIFEST),
                "manifest_protocol": str(MANIFEST_PROTOCOL.relative_to(PROJECT_ROOT).as_posix()),
                "manifest_protocol_sha256": sha256_file(MANIFEST_PROTOCOL),
                "experiment_protocol": str(
                    EXPERIMENT_PROTOCOL_PATH.relative_to(PROJECT_ROOT).as_posix()
                ),
                "experiment_protocol_sha256": sha256_file(EXPERIMENT_PROTOCOL_PATH),
                "reference_huggingface_commit": manifest_protocol.get("reference_huggingface_commit"),
                "public_image_root": str(PUBLIC_IMAGE_ROOT),
                "public_annotation_root": str(PUBLIC_ANNOTATION_ROOT),
                "image_indirection": "Windows directory symlink; no image bytes copied",
                "annotation_content_inventory_sha256": canonical_sha256(annotation_identities),
                "image_content_inventory_sha256": canonical_sha256(
                    image_content_identities
                ),
            },
            "implementation": {
                "builder": {
                    "path": str(BUILDER_PATH),
                    "sha256": sha256_file(BUILDER_PATH),
                },
                "group_split": {
                    "path": str(SPLIT_IMPLEMENTATION_PATH.resolve(strict=True)),
                    "sha256": sha256_file(SPLIT_IMPLEMENTATION_PATH),
                    "function": "grouped_three_way_split",
                },
            },
            "split": {
                "unit": "physical_group",
                "method": "two-stage SHA256 physical-group ordering",
                "seed": int(split_seed),
                "calibration_fraction_target": float(calibration_fraction),
                "validation_fraction_target": float(validation_fraction),
                "partitions": partition_stats,
                "group_disjoint": True,
            },
            "inventory": {
                "images": EXPECTED_SAMPLES,
                "physical_groups": EXPECTED_GROUPS,
                "boxes": EXPECTED_SAMPLES,
                "boxes_clipped_to_visible_frame": clipped_boxes,
                "classes": {"0": "meter"},
                "label_inventory_sha256": canonical_sha256(label_identities),
                "image_content_inventory_sha256": canonical_sha256(
                    image_content_identities
                ),
                "annotation_content_inventory_sha256": canonical_sha256(
                    annotation_identities
                ),
            },
            "artifacts": {
                "train_calibration_yaml": train_cal_yaml.name,
                "train_calibration_yaml_sha256": sha256_file(train_cal_yaml),
                "validation_yaml": validation_yaml.name,
                "validation_yaml_sha256": sha256_file(validation_yaml),
            },
        }
        _write_json(temporary / "summary.json", summary)
        seal = {
            "schema_version": 1,
            "protocol": SEAL_PROTOCOL,
            "status": "sealed",
            "summary_sha256": sha256_file(temporary / "summary.json"),
            "manifest_sha256": summary["source"]["manifest_sha256"],
            "experiment_protocol_sha256": summary["source"][
                "experiment_protocol_sha256"
            ],
            "label_inventory_sha256": summary["inventory"]["label_inventory_sha256"],
            "image_content_inventory_sha256": summary["inventory"][
                "image_content_inventory_sha256"
            ],
            "annotation_content_inventory_sha256": summary["inventory"][
                "annotation_content_inventory_sha256"
            ],
            "split_sha256": canonical_sha256(summary["split"]),
            "train_calibration_yaml_sha256": summary["artifacts"]["train_calibration_yaml_sha256"],
            "validation_yaml_sha256": summary["artifacts"]["validation_yaml_sha256"],
            "builder_sha256": summary["implementation"]["builder"]["sha256"],
            "group_split_source_sha256": summary["implementation"]["group_split"]["sha256"],
        }
        _write_json(temporary / "seal.json", seal)
        os.replace(temporary, output)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_corpus(root: Path) -> dict[str, Any]:
    corpus = _safe_output(root).resolve(strict=True)
    experiment_protocol = _load_experiment_protocol()
    summary_path = corpus / "summary.json"
    seal_path = corpus / "seal.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(summary.get("protocol") == PROTOCOL and summary.get("status") == "complete", "corpus summary drift")
    _require(seal.get("protocol") == SEAL_PROTOCOL and seal.get("status") == "sealed", "corpus seal drift")
    _require(seal.get("summary_sha256") == sha256_file(summary_path), "corpus summary hash drift")
    source = summary.get("source") or {}
    _require(
        source.get("experiment_protocol_sha256")
        == sha256_file(EXPERIMENT_PROTOCOL_PATH)
        == seal.get("experiment_protocol_sha256"),
        "corpus experiment-protocol binding drift",
    )
    implementation = summary.get("implementation") or {}
    builder_binding = implementation.get("builder") or {}
    split_binding = implementation.get("group_split") or {}
    _require(
        Path(str(builder_binding.get("path") or "")).resolve(strict=True) == BUILDER_PATH
        and builder_binding.get("sha256") == sha256_file(BUILDER_PATH),
        "corpus builder implementation drift",
    )
    _require(
        Path(str(split_binding.get("path") or "")).resolve(strict=True)
        == SPLIT_IMPLEMENTATION_PATH.resolve(strict=True)
        and split_binding.get("sha256") == sha256_file(SPLIT_IMPLEMENTATION_PATH)
        and split_binding.get("function") == "grouped_three_way_split",
        "corpus group-split implementation drift",
    )
    _require(seal.get("builder_sha256") == builder_binding.get("sha256"), "corpus seal/builder drift")
    _require(
        seal.get("group_split_source_sha256") == split_binding.get("sha256"),
        "corpus seal/group-split drift",
    )
    scope = summary.get("scope") or {}
    _require(scope.get("dataset") == "SyncG" and scope.get("split") == "train", "corpus scope drift")
    _require(scope.get("public_data_only") is True, "corpus is not public-only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(scope.get(key) is False, f"corpus audit violation: {key}")
    inventory = summary.get("inventory") or {}
    _require((int(inventory.get("images", -1)), int(inventory.get("physical_groups", -1))) == (EXPECTED_SAMPLES, EXPECTED_GROUPS), "corpus inventory drift")
    image_link = corpus / "images/public_train"
    _require(image_link.is_symlink(), "corpus public image path is not a symlink")
    _require(image_link.resolve(strict=True) == PUBLIC_IMAGE_ROOT, "corpus image source drift")
    partitions = (summary.get("split") or {}).get("partitions") or {}
    _require(set(partitions) == {"train", "calibration", "validation"}, "partition roster drift")
    frozen_partition = experiment_protocol.get("partition") or {}
    split = summary.get("split") or {}
    _require(
        split.get("unit") == "physical_group"
        and split.get("method") == "two-stage SHA256 physical-group ordering"
        and int(split.get("seed", -1)) == int(frozen_partition.get("seed", -2)),
        "corpus split protocol drift",
    )
    _require(
        math.isclose(
            float(split.get("calibration_fraction_target", -1.0)),
            float(frozen_partition.get("calibration_fraction", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(split.get("validation_fraction_target", -1.0)),
            float(frozen_partition.get("validation_fraction", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "corpus split fractions drift",
    )
    _require(summary.get("source", {}).get("manifest_sha256") == sha256_file(MANIFEST), "current public manifest hash drift")
    _require(
        summary.get("source", {}).get("manifest_protocol_sha256")
        == sha256_file(MANIFEST_PROTOCOL),
        "current public manifest protocol hash drift",
    )
    groups: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    label_identities: list[dict[str, Any]] = []
    image_identities: list[dict[str, Any]] = []
    annotation_identities: list[dict[str, Any]] = []
    total = 0
    for partition in ("train", "calibration", "validation"):
        rows, partition_groups, partition_labels = verify_corpus_partition(
            corpus, summary, partition
        )
        groups[partition] = partition_groups
        current_ids = {str(row["sample_id"]) for row in rows}
        _require(not sample_ids.intersection(current_ids), "sample overlap across detector partitions")
        sample_ids.update(current_ids)
        label_identities.extend(partition_labels)
        image_identities.extend(
            {
                "sample_id": str(row["sample_id"]),
                "bytes": int(row["image_bytes"]),
                "sha256": str(row["image_sha256"]),
            }
            for row in rows
        )
        annotation_identities.extend(
            {
                "sample_id": str(row["sample_id"]),
                "bytes": int(row["source_annotation_bytes"]),
                "sha256": str(row["source_annotation_sha256"]),
            }
            for row in rows
        )
        total += len(rows)
    _require(total == EXPECTED_SAMPLES, "verified sample inventory drift")
    _require(not groups["train"].intersection(groups["calibration"] | groups["validation"]) and not groups["calibration"].intersection(groups["validation"]), "verified physical-group leakage")
    label_identities.sort(key=lambda value: str(value["sample_id"]))
    _require(
        canonical_sha256(label_identities) == inventory.get("label_inventory_sha256"),
        "verified YOLO label inventory hash drift",
    )
    image_identities.sort(key=lambda value: str(value["sample_id"]))
    annotation_identities.sort(key=lambda value: str(value["sample_id"]))
    _require(
        canonical_sha256(image_identities)
        == inventory.get("image_content_inventory_sha256")
        == source.get("image_content_inventory_sha256")
        == seal.get("image_content_inventory_sha256"),
        "verified public image content inventory hash drift",
    )
    _require(
        canonical_sha256(annotation_identities)
        == inventory.get("annotation_content_inventory_sha256")
        == source.get("annotation_content_inventory_sha256")
        == seal.get("annotation_content_inventory_sha256"),
        "verified public annotation content inventory hash drift",
    )
    for key in ("train_calibration_yaml", "validation_yaml"):
        artifact = corpus / str(summary["artifacts"][key])
        _require(sha256_file(artifact) == summary["artifacts"][f"{key}_sha256"], f"{key} hash drift")
    return summary


def verify_corpus_partition(
    corpus_root: Path, summary: Mapping[str, Any], partition: str
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    """Authenticate exactly one public partition without opening the others."""

    _require(partition in {"train", "calibration", "validation"}, "unknown partition")
    corpus = _safe_output(corpus_root).resolve(strict=True)
    info = summary["split"]["partitions"][partition]
    samples_path = (corpus / str(info["samples_artifact"])).resolve(strict=True)
    list_path = (corpus / str(info["image_list"])).resolve(strict=True)
    _require(sha256_file(samples_path) == info["samples_sha256"], f"{partition} metadata hash drift")
    _require(sha256_file(list_path) == info["image_list_sha256"], f"{partition} list hash drift")
    rows = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line]
    _require(len(rows) == int(info["samples"]), f"{partition} row count drift")
    _require(all(row.get("partition") == partition for row in rows), f"{partition} assignment drift")
    ids = [str(row.get("sample_id") or "") for row in rows]
    _require(all(ids) and len(set(ids)) == len(ids), f"{partition} sample identity drift")
    _require(canonical_sha256(ids) == info["sample_ids_sha256"], f"{partition} sample hash drift")
    groups = {str(row["group_id"]) for row in rows}
    _require(canonical_sha256(sorted(groups)) == info["group_ids_sha256"], f"{partition} group hash drift")
    expected_images = [
        str(corpus / Path(str(row["image_path"]))).replace("\\", "/") for row in rows
    ]
    observed_images = list_path.read_text(encoding="utf-8").splitlines()
    _require(observed_images == expected_images, f"{partition} Ultralytics image list/content drift")
    labels: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        image_path = (corpus / str(row["image_path"])).resolve(strict=True)
        _require_under(image_path, PUBLIC_IMAGE_ROOT, label=f"{sample_id}.linked_image")
        image_identity = {
            "sample_id": sample_id,
            "bytes": int(image_path.stat().st_size),
            "sha256": sha256_file(image_path),
        }
        _require(
            image_identity["bytes"] == int(row.get("image_bytes", -1))
            and image_identity["sha256"] == row.get("image_sha256"),
            f"{sample_id}: public image content drift",
        )
        images.append(image_identity)
        annotation_path = _require_under(
            PUBLIC_ANNOTATION_ROOT / str(row.get("source_annotation_path") or ""),
            PUBLIC_ANNOTATION_ROOT,
            label=f"{sample_id}.source_annotation",
        )
        annotation_identity = {
            "sample_id": sample_id,
            "bytes": int(annotation_path.stat().st_size),
            "sha256": sha256_file(annotation_path),
        }
        _require(
            annotation_identity["bytes"]
            == int(row.get("source_annotation_bytes", -1))
            and annotation_identity["sha256"]
            == row.get("source_annotation_sha256"),
            f"{sample_id}: public annotation content drift",
        )
        annotations.append(annotation_identity)
        label_path = (corpus / str(row["label_path"])).resolve(strict=True)
        try:
            label_path.relative_to((corpus / "labels/public_train").resolve(strict=True))
        except ValueError as error:
            raise ValueError(f"{sample_id}: label escapes frozen corpus") from error
        values = [float(value) for value in row["dial_bbox_yolo"]]
        _require(len(values) == 4 and np.isfinite(values).all(), f"{sample_id}: invalid stored YOLO box")
        expected_label = "0 " + " ".join(f"{value:.10f}" for value in values) + "\n"
        observed = label_path.read_text(encoding="utf-8")
        _require(observed == expected_label, f"{sample_id}: YOLO label content drift")
        labels.append(
            {"sample_id": sample_id, "sha256": hashlib.sha256(observed.encode("utf-8")).hexdigest()}
        )
    labels.sort(key=lambda value: str(value["sample_id"]))
    images.sort(key=lambda value: str(value["sample_id"]))
    annotations.sort(key=lambda value: str(value["sample_id"]))
    _require(
        canonical_sha256(labels) == info.get("label_inventory_sha256"),
        f"{partition} YOLO label inventory hash drift",
    )
    _require(
        canonical_sha256(images) == info.get("image_content_inventory_sha256"),
        f"{partition} public image content inventory hash drift",
    )
    _require(
        canonical_sha256(annotations)
        == info.get("annotation_content_inventory_sha256"),
        f"{partition} public annotation content inventory hash drift",
    )
    return rows, groups, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        result = verify_corpus(args.output_dir)
    else:
        result = build_corpus(
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            calibration_fraction=args.calibration_fraction,
            validation_fraction=args.validation_fraction,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
