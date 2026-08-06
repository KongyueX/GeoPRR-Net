"""Train, calibrate, and independently validate the public SyncG meter detector.

Formal selection uses only the group-disjoint training and calibration
partitions.  An immutable selection claim is written before the independent
validation sample metadata is opened.  Ultralytics ``last.pt`` runs may be
resumed explicitly; completed summaries are immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.build_syncg_meter_detector_public import (
    BUILDER_PATH,
    MANIFEST,
    MANIFEST_PROTOCOL,
    PUBLIC_IMAGE_ROOT,
    PROTOCOL as CORPUS_PROTOCOL,
    SEAL_PROTOCOL as CORPUS_SEAL_PROTOCOL,
    SPLIT_IMPLEMENTATION_PATH,
    canonical_sha256,
    sha256_file,
    verify_corpus,
    verify_corpus_partition,
)


TRAINING_PROTOCOL = "syncg_public_meter_detector_training_v1"
TRAINING_SEAL_PROTOCOL = "syncg_public_meter_detector_training_seal_v1"
RUN_INTENT_PROTOCOL = "syncg_public_meter_detector_run_intent_v1"
SELECTION_CLAIM_PROTOCOL = "syncg_public_meter_detector_selection_claim_v1"
SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()
DEFAULT_CORPUS = SAFE_OUTPUT_ROOT / "syncg_meter_detector_public_v1"
DEFAULT_OUTPUT = SAFE_OUTPUT_ROOT / "syncg_meter_detector_runs"
DEFAULT_PRETRAINED = (
    SAFE_OUTPUT_ROOT / "public_pretrained/yolo11n-ultralytics-assets-v8.3.0.pt"
)
EXPERIMENT_PROTOCOL_PATH = PROJECT_ROOT / "experiments/syncg_meter_detector_protocol.json"
TRAINER_PATH = Path(__file__).resolve()
EXPECTED_ULTRALYTICS = "8.4.102"
EXPECTED_PRETRAINED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
REQUIREMENTS_LOCK_PATH = PROJECT_ROOT / "experiments/requirements-training.lock.txt"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _safe_root(path: Path, *, label: str) -> Path:
    value = Path(path).resolve()
    try:
        value.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_OUTPUT_ROOT}") from error
    _require(value != SAFE_OUTPUT_ROOT, f"refusing broad {label}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a chronology-sensitive claim exactly once."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"immutable artifact already exists: {path}")
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    payload = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in materialized
    )
    if path.exists():
        _require(path.read_bytes() == payload, f"immutable JSONL artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _binding(path: Path) -> dict[str, Any]:
    value = Path(path).resolve(strict=True)
    return {"path": str(value), "sha256": sha256_file(value)}


def _runtime_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the installed Ultralytics implementation, not only its version string."""

    import ultralytics

    runtime = (protocol.get("model") or {}).get("runtime") or {}
    _require(runtime.get("ultralytics_version") == EXPECTED_ULTRALYTICS, "protocol runtime version drift")
    _require(ultralytics.__version__ == EXPECTED_ULTRALYTICS, "Ultralytics version drift")
    lock_path = (PROJECT_ROOT / str(runtime.get("requirements_lock_path") or "")).resolve(
        strict=True
    )
    _require(lock_path == REQUIREMENTS_LOCK_PATH.resolve(strict=True), "requirements lock path drift")
    _require(
        sha256_file(lock_path) == runtime.get("requirements_lock_sha256"),
        "requirements lock hash drift",
    )
    distribution = importlib.metadata.distribution("ultralytics")
    _require(distribution.version == EXPECTED_ULTRALYTICS, "Ultralytics distribution metadata drift")
    record_path = (Path(distribution._path) / "RECORD").resolve(strict=True)  # type: ignore[attr-defined]
    _require(
        sha256_file(record_path) == runtime.get("ultralytics_distribution_record_sha256"),
        "Ultralytics distribution RECORD hash drift",
    )
    package_root = Path(ultralytics.__file__).resolve(strict=True).parent
    declared_sources = runtime.get("critical_source_sha256") or {}
    _require(isinstance(declared_sources, Mapping) and bool(declared_sources), "critical runtime sources absent")
    critical_sources: dict[str, Any] = {}
    for relative, expected in sorted(declared_sources.items()):
        source = (package_root / str(relative)).resolve(strict=True)
        try:
            source.relative_to(package_root)
        except ValueError as error:
            raise ValueError(f"Ultralytics critical source escapes package: {relative}") from error
        _require(sha256_file(source) == expected, f"Ultralytics source hash drift: {relative}")
        critical_sources[str(relative)] = _binding(source)
    return {
        "ultralytics_version": ultralytics.__version__,
        "distribution_record": _binding(record_path),
        "requirements_lock": _binding(lock_path),
        "critical_sources": critical_sources,
    }


def _load_experiment_protocol() -> dict[str, Any]:
    protocol = _json(EXPERIMENT_PROTOCOL_PATH.resolve(strict=True))
    _require(
        protocol.get("protocol") == "syncg_public_meter_detector_experiment_v1"
        and protocol.get("status") == "frozen_before_training",
        "detector experiment protocol drift",
    )
    scope = protocol.get("data_scope") or {}
    _require(scope.get("dataset") == "SyncG" and scope.get("allowed_split") == "train", "protocol data scope drift")
    _require((int(scope.get("expected_images", -1)), int(scope.get("expected_physical_groups", -1))) == (16_000, 725), "protocol inventory drift")
    partition = protocol.get("partition") or {}
    _require(
        partition.get("unit") == "physical_group"
        and partition.get("method") == "two-stage SHA256 physical-group ordering"
        and int(partition.get("seed", -1)) == 20260819,
        "detector partition protocol drift",
    )
    selection = protocol.get("selection") or {}
    _require(
        selection.get("checkpoint_rule")
        == "maximum calibration mAP50-95; Ultralytics latest epoch attaining the maximum is retained on exact ties",
        "detector checkpoint-selection rule drift",
    )
    _require(
        math.isclose(float(selection.get("prediction_confidence_floor", -1.0)), 0.001)
        and math.isclose(float(selection.get("prediction_nms_iou", -1.0)), 0.70),
        "detector prediction settings drift",
    )
    return protocol


def _preselection_corpus(
    corpus_root: Path, protocol: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Authenticate summary plus train/calibration artifacts without opening validation rows."""

    root = _safe_root(corpus_root, label="corpus").resolve(strict=True)
    summary_path, seal_path = root / "summary.json", root / "seal.json"
    summary, seal = _json(summary_path), _json(seal_path)
    _require(summary.get("protocol") == CORPUS_PROTOCOL and summary.get("status") == "complete", "corpus summary drift")
    _require(seal.get("protocol") == CORPUS_SEAL_PROTOCOL and seal.get("status") == "sealed", "corpus seal drift")
    _require(seal.get("summary_sha256") == sha256_file(summary_path), "corpus seal/summary drift")
    scope = summary.get("scope") or {}
    _require(scope.get("dataset") == "SyncG" and scope.get("split") == "train", "corpus scope drift")
    _require(scope.get("public_data_only") is True, "corpus is not public-only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(scope.get(key) is False, f"corpus audit violation: {key}")
    source = summary.get("source") or {}
    _require(source.get("manifest_sha256") == sha256_file(MANIFEST), "current public manifest drift")
    _require(
        source.get("manifest_protocol_sha256") == sha256_file(MANIFEST_PROTOCOL),
        "current public manifest protocol drift",
    )
    image_link = root / "images/public_train"
    _require(
        image_link.is_symlink() and image_link.resolve(strict=True) == PUBLIC_IMAGE_ROOT,
        "public training image symlink drift",
    )
    implementation = summary.get("implementation") or {}
    builder = implementation.get("builder") or {}
    split_source = implementation.get("group_split") or {}
    _require(
        Path(str(builder.get("path") or "")).resolve(strict=True) == BUILDER_PATH
        and builder.get("sha256") == sha256_file(BUILDER_PATH),
        "corpus builder drift",
    )
    _require(
        Path(str(split_source.get("path") or "")).resolve(strict=True)
        == SPLIT_IMPLEMENTATION_PATH.resolve(strict=True)
        and split_source.get("sha256") == sha256_file(SPLIT_IMPLEMENTATION_PATH)
        and split_source.get("function") == "grouped_three_way_split",
        "corpus group-split implementation drift",
    )
    _require(seal.get("builder_sha256") == builder.get("sha256"), "corpus seal/builder drift")
    _require(
        seal.get("group_split_source_sha256") == split_source.get("sha256"),
        "corpus seal/group-split drift",
    )
    partitions = (summary.get("split") or {}).get("partitions") or {}
    _require(set(partitions) == {"train", "calibration", "validation"}, "corpus partition roster drift")
    inventory = summary.get("inventory") or {}
    scope_protocol = protocol.get("data_scope") or {}
    _require(
        int(inventory.get("images", -1)) == int(scope_protocol.get("expected_images", -2))
        and int(inventory.get("physical_groups", -1))
        == int(scope_protocol.get("expected_physical_groups", -2)),
        "corpus inventory differs from detector protocol",
    )
    split = summary.get("split") or {}
    frozen_split = protocol.get("partition") or {}
    _require(
        split.get("unit") == frozen_split.get("unit")
        and split.get("method") == frozen_split.get("method")
        and int(split.get("seed", -1)) == int(frozen_split.get("seed", -2)),
        "corpus split identity differs from detector protocol",
    )
    _require(
        math.isclose(
            float(split.get("calibration_fraction_target", -1.0)),
            float(frozen_split.get("calibration_fraction", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(split.get("validation_fraction_target", -1.0)),
            float(frozen_split.get("validation_fraction", -2.0)),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "corpus split fractions differ from detector protocol",
    )
    _require(
        seal.get("split_sha256") == canonical_sha256(split)
        and seal.get("label_inventory_sha256") == inventory.get("label_inventory_sha256")
        and seal.get("image_content_inventory_sha256")
        == inventory.get("image_content_inventory_sha256")
        and seal.get("annotation_content_inventory_sha256")
        == inventory.get("annotation_content_inventory_sha256"),
        "corpus seal inventory/split binding drift",
    )
    verified: dict[str, tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]] = {}
    for partition in ("train", "calibration"):
        verified[partition] = verify_corpus_partition(root, summary, partition)
    train_rows, train_groups, _ = verified["train"]
    calibration_rows, calibration_groups, _ = verified["calibration"]
    _require(not train_groups.intersection(calibration_groups), "train/calibration physical-group leakage")
    _require(
        not {str(row["sample_id"]) for row in train_rows}.intersection(
            str(row["sample_id"]) for row in calibration_rows
        ),
        "train/calibration sample leakage",
    )
    yaml_path = root / str(summary["artifacts"]["train_calibration_yaml"])
    _require(
        sha256_file(yaml_path) == summary["artifacts"]["train_calibration_yaml_sha256"],
        "training/calibration YAML hash drift",
    )
    return root, summary, seal


def _load_partition(root: Path, summary: Mapping[str, Any], partition: str) -> list[dict[str, Any]]:
    info = summary["split"]["partitions"][partition]
    path = root / str(info["samples_artifact"])
    _require(sha256_file(path) == info["samples_sha256"], f"{partition} metadata hash drift")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    _require(len(rows) == int(info["samples"]), f"{partition} row count drift")
    _require(all(row.get("partition") == partition for row in rows), f"{partition} assignment drift")
    return rows


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1) + max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1) - intersection
    return intersection / union if union > 0.0 else 0.0


def _prediction_rows(
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    corpus_root: Path,
    image_size: int,
    batch_size: int,
    device: str,
    confidence_floor: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    sources = [str((corpus_root / str(row["image_path"])).resolve(strict=True)) for row in samples]
    stream = model.predict(
        source=sources,
        stream=True,
        conf=float(confidence_floor),
        iou=float(nms_iou),
        imgsz=image_size,
        batch=batch_size,
        device=device,
        verbose=False,
        save=False,
    )
    result: list[dict[str, Any]] = []
    for sample, prediction in zip(samples, stream, strict=True):
        candidates: list[dict[str, Any]] = []
        boxes = prediction.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidence = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            for coordinates, score, class_id in zip(xyxy, confidence, classes, strict=True):
                if int(class_id) != 0:
                    continue
                raw_values = [float(value) for value in coordinates[:4]]
                if not (math.isfinite(float(score)) and np.isfinite(raw_values).all()):
                    continue
                # Match targetDetectModel.target_detection exactly: Python int
                # truncation first, then clamp to the source frame.
                width = int(sample["image_width"])
                height = int(sample["image_height"])
                x1, y1, x2, y2 = (int(value) for value in raw_values)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                values = [float(x1), float(y1), float(x2), float(y2)]
                candidates.append({"confidence": float(score), "xyxy": values, "class_id": 0})
        candidates.sort(key=lambda value: (-float(value["confidence"]), *value["xyxy"]))
        result.append({"sample_id": str(sample["sample_id"]), "candidates": candidates})
    _require(len(result) == len(samples), "prediction inventory drift")
    return result


def selected_box_metrics(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    _require(len(samples) == len(predictions) and bool(samples), "metric inventory mismatch")
    true_positive = selected_count = 0
    selected_ious: list[float] = []
    for sample, prediction in zip(samples, predictions, strict=True):
        _require(str(sample["sample_id"]) == str(prediction["sample_id"]), "prediction order drift")
        eligible = [
            value
            for value in prediction.get("candidates", [])
            if float(value["confidence"]) >= float(threshold)
        ]
        if not eligible:
            continue
        selected_count += 1
        iou = _iou(eligible[0]["xyxy"], sample["dial_bbox_xyxy"])
        selected_ious.append(iou)
        true_positive += iou >= 0.50
    total = len(samples)
    recall = true_positive / total
    precision = true_positive / selected_count if selected_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": total,
        "confidence_threshold": float(threshold),
        "selected_boxes": selected_count,
        "selected_box_coverage": selected_count / total,
        "selected_box_iou50_true_positives": true_positive,
        "selected_box_iou50_recall": recall,
        "selected_box_precision": precision,
        "selected_box_f1": f1,
        "mean_selected_iou": float(np.mean(selected_ious)) if selected_ious else 0.0,
        "p05_selected_iou": float(np.percentile(selected_ious, 5)) if selected_ious else 0.0,
    }


def choose_confidence_threshold(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidates = [
        selected_box_metrics(samples, predictions, threshold=round(0.05 + index * 0.01, 2))
        for index in range(91)
    ]
    feasible = [row for row in candidates if float(row["selected_box_iou50_recall"]) >= 0.995]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["selected_box_f1"]),
                float(row["mean_selected_iou"]),
                float(row["confidence_threshold"]),
            ),
        )
        branch = "recall_constraint_feasible"
    else:
        selected = max(
            candidates,
            key=lambda row: (
                float(row["selected_box_iou50_recall"]),
                float(row["selected_box_f1"]),
                float(row["mean_selected_iou"]),
                -float(row["confidence_threshold"]),
            ),
        )
        branch = "maximum_recall_fallback"
    return {
        "rule": "0.05..0.95 step 0.01; recall>=0.995, then maximum F1/mean-IoU/threshold",
        "branch": branch,
        "grid_size": len(candidates),
        "selected": selected,
        "grid_sha256": canonical_sha256(candidates),
    }


def _ultralytics_metrics(value: Any) -> dict[str, float]:
    raw = getattr(value, "results_dict", None)
    _require(isinstance(raw, Mapping), "Ultralytics validation returned no results_dict")
    result: dict[str, float] = {}
    for key, item in raw.items():
        number = float(item)
        if math.isfinite(number):
            result[str(key)] = number
    _require(bool(result), "Ultralytics validation metrics are empty")
    return result


def _selected_model_identity(model: Any) -> dict[str, Any]:
    task = str(getattr(model, "task", "") or "")
    names_value = getattr(model, "names", None)
    _require(task == "detect", f"selected checkpoint task is not detection: {task!r}")
    _require(isinstance(names_value, Mapping), "selected checkpoint class names absent")
    names = {str(int(key)): str(value) for key, value in names_value.items()}
    _require(names == {"0": "meter"}, f"selected checkpoint class identity drift: {names}")
    torch_model = getattr(model, "model", None)
    parameters = (
        sum(int(value.numel()) for value in torch_model.parameters())
        if torch_model is not None
        else 0
    )
    _require(parameters > 0, "selected checkpoint contains no parameters")
    return {"task": task, "names": names, "parameters": parameters}


def _metric(metrics: Mapping[str, float], suffix: str) -> float:
    matches = [float(value) for key, value in metrics.items() if key.casefold().endswith(suffix.casefold())]
    _require(len(matches) == 1, f"Ultralytics metric {suffix!r} is absent or ambiguous: {sorted(metrics)}")
    return matches[0]


def _validate_pretrained(path: Path, protocol: Mapping[str, Any]) -> Path:
    checkpoint = _safe_root(path, label="pretrained checkpoint").resolve(strict=True)
    expected = protocol["model"]["initial_checkpoint"]
    _require(expected.get("sha256") == EXPECTED_PRETRAINED_SHA256, "protocol pretrained hash drift")
    _require(int(expected.get("bytes", -1)) == checkpoint.stat().st_size, "pretrained byte-size drift")
    _require(sha256_file(checkpoint) == EXPECTED_PRETRAINED_SHA256, "pretrained checkpoint hash drift")
    return checkpoint


def _resolve_binding(value: Any, *, label: str) -> Path:
    _require(isinstance(value, Mapping), f"{label} binding absent")
    path = Path(str(value.get("path") or ""))
    _require(path.is_absolute(), f"{label}.path must be absolute")
    path = path.resolve(strict=True)
    digest = str(value.get("sha256") or "").casefold()
    _require(len(digest) == 64 and set(digest) <= set("0123456789abcdef"), f"{label}.sha256 invalid")
    _require(sha256_file(path) == digest, f"{label} hash drift")
    return path


def _frozen_train_configuration(
    *,
    protocol: Mapping[str, Any],
    corpus_root: Path,
    output_root: Path,
    pretrained: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model = protocol["model"]
    return {
        "seed": int(args.seed),
        "data": str((corpus_root / "dataset_train_cal.yaml").resolve(strict=True)),
        "output_root": str(output_root),
        "run_name": f"seed_{args.seed}",
        "pretrained": str(pretrained),
        "device": str(args.device),
        "workers": int(args.workers),
        "image_size": int(model["image_size"]),
        "batch_size": int(model["batch_size"]),
        "epochs_maximum": int(model["epochs_maximum"]),
        "early_stopping_patience": int(model["early_stopping_patience"]),
        "optimizer": str(model["optimizer"]),
        "initial_learning_rate": float(model["initial_learning_rate"]),
        "weight_decay": float(model["weight_decay"]),
        "deterministic": bool(model["deterministic"]),
        "amp": bool(model["amp"]),
        "close_mosaic": 10,
        "cache": False,
        "plots": False,
    }


def _run_intent(
    *,
    path: Path,
    protocol: Mapping[str, Any],
    corpus_root: Path,
    output_root: Path,
    pretrained: Path,
    runtime_identity: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "protocol": RUN_INTENT_PROTOCOL,
        "status": "frozen_before_first_training_process",
        "seed": int(args.seed),
        "corpus_root": str(corpus_root),
        "corpus_summary": _binding(corpus_root / "summary.json"),
        "corpus_seal": _binding(corpus_root / "seal.json"),
        "pretrained_checkpoint": _binding(pretrained),
        "configuration": _frozen_train_configuration(
            protocol=protocol,
            corpus_root=corpus_root,
            output_root=output_root,
            pretrained=pretrained,
            args=args,
        ),
        "runtime_identity": dict(runtime_identity),
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "experiment_protocol_sha256": sha256_file(EXPERIMENT_PROTOCOL_PATH),
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
    }
    if path.exists():
        observed = _json(path)
        _require(observed == expected, "existing detector run intent differs from current frozen run")
        return observed
    _require(not args.resume, "resume requires an existing immutable detector run intent")
    _atomic_new_json(path, expected)
    return expected


def _require_training_args(
    train_args: Mapping[str, Any], configuration: Mapping[str, Any], *, resumable: bool
) -> None:
    data_path = Path(str(train_args.get("data") or "")).resolve(strict=True)
    _require(data_path == Path(str(configuration["data"])).resolve(strict=True), "checkpoint data/corpus drift")
    _require(int(train_args.get("seed", -1)) == int(configuration["seed"]), "checkpoint seed drift")
    exact = {
        "imgsz": int(configuration["image_size"]),
        "batch": int(configuration["batch_size"]),
        "epochs": int(configuration["epochs_maximum"]),
        "patience": int(configuration["early_stopping_patience"]),
        "workers": int(configuration["workers"]),
        "close_mosaic": int(configuration["close_mosaic"]),
    }
    for key, expected in exact.items():
        _require(int(train_args.get(key, -1)) == expected, f"checkpoint training argument drift: {key}")
    _require(str(train_args.get("optimizer")) == str(configuration["optimizer"]), "checkpoint optimizer drift")
    for key, configuration_key in (("lr0", "initial_learning_rate"), ("weight_decay", "weight_decay")):
        _require(
            math.isclose(
                float(train_args.get(key, float("nan"))),
                float(configuration[configuration_key]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"checkpoint training argument drift: {key}",
        )
    for key in ("deterministic", "amp", "cache", "plots"):
        _require(bool(train_args.get(key)) is bool(configuration[key]), f"checkpoint training argument drift: {key}")
    _require(str(train_args.get("task") or "detect") == "detect", "checkpoint task drift")
    if resumable:
        _require(str(train_args.get("mode") or "train") == "train", "checkpoint mode drift")


def _validate_resume_checkpoint(model: Any, intent: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = getattr(model, "ckpt", None)
    _require(isinstance(checkpoint, Mapping), "resume checkpoint metadata absent")
    _require(checkpoint.get("version") == EXPECTED_ULTRALYTICS, "resume checkpoint Ultralytics version drift")
    _require(int(checkpoint.get("epoch", -1)) >= 0, "resume checkpoint was stripped or already terminal")
    _require(checkpoint.get("optimizer") is not None, "resume checkpoint optimizer state absent")
    train_args = checkpoint.get("train_args")
    _require(isinstance(train_args, Mapping), "resume checkpoint train_args absent")
    _require_training_args(train_args, intent["configuration"], resumable=True)
    return {
        "epoch_zero_based": int(checkpoint["epoch"]),
        "version": str(checkpoint["version"]),
        "train_args_sha256": canonical_sha256(dict(train_args)),
    }


def _training_evidence(
    *, run_dir: Path, selected_model: Any, intent: Mapping[str, Any]
) -> dict[str, Any]:
    args_path = (run_dir / "args.yaml").resolve(strict=True)
    results_path = (run_dir / "results.csv").resolve(strict=True)
    args_value = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    _require(isinstance(args_value, Mapping), "Ultralytics args.yaml is invalid")
    _require_training_args(args_value, intent["configuration"], resumable=False)
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    _require(bool(rows), "Ultralytics results.csv contains no epochs")
    metric_keys = [key for key in rows[0] if key.casefold() == "metrics/map50-95(b)".casefold()]
    _require(len(metric_keys) == 1, "results.csv calibration mAP50-95 column absent or ambiguous")
    metric_key = metric_keys[0]
    values = [float(row[metric_key]) for row in rows]
    _require(np.isfinite(values).all(), "results.csv contains non-finite calibration mAP50-95")
    best_value = max(values)
    best_epoch = max(
        int(float(row["epoch"]))
        for row, value in zip(rows, values, strict=True)
        if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
    )
    checkpoint = getattr(selected_model, "ckpt", None)
    _require(isinstance(checkpoint, Mapping), "selected best.pt checkpoint metadata absent")
    _require(checkpoint.get("version") == EXPECTED_ULTRALYTICS, "selected best.pt version drift")
    train_args = checkpoint.get("train_args")
    _require(isinstance(train_args, Mapping), "selected best.pt train_args absent")
    _require_training_args(train_args, intent["configuration"], resumable=False)
    train_metrics = checkpoint.get("train_metrics")
    _require(isinstance(train_metrics, Mapping), "selected best.pt train_metrics absent")
    checkpoint_fitness = float(train_metrics.get("fitness", float("nan")))
    _require(
        math.isfinite(checkpoint_fitness)
        and math.isclose(checkpoint_fitness, best_value, rel_tol=1e-5, abs_tol=5e-6),
        "best.pt fitness does not match maximum calibration mAP50-95",
    )
    embedded_results = checkpoint.get("train_results")
    _require(isinstance(embedded_results, Mapping), "selected best.pt embedded train_results absent")
    embedded_epochs = embedded_results.get("epoch")
    _require(
        isinstance(embedded_epochs, Sequence)
        and len(embedded_epochs) == len(rows),
        "selected best.pt embedded training history length drift",
    )
    return {
        "args_yaml": _binding(args_path),
        "results_csv": _binding(results_path),
        "epochs_completed": len(rows),
        "checkpoint_rule": "maximum calibration mAP50-95; latest exact tie retained",
        "selected_epoch_one_based": best_epoch,
        "maximum_calibration_map50_95": best_value,
        "checkpoint_train_metrics_fitness": checkpoint_fitness,
        "embedded_history_epochs": len(embedded_epochs),
    }


def _selection_claim(
    *,
    run_dir: Path,
    corpus_root: Path,
    corpus_summary: Mapping[str, Any],
    checkpoint: Path,
    calibration_ultralytics: Mapping[str, float],
    calibration_threshold: Mapping[str, Any],
    predictions_path: Path,
    model_identity: Mapping[str, Any],
    run_intent_path: Path,
    runtime_identity: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    claim = {
        "schema_version": 1,
        "protocol": SELECTION_CLAIM_PROTOCOL,
        "status": "selection_frozen_before_independent_validation",
        "seed": int(args.seed),
        "run_intent": _binding(run_intent_path),
        "corpus_summary": _binding(corpus_root / "summary.json"),
        "corpus_seal": _binding(corpus_root / "seal.json"),
        "checkpoint": _binding(checkpoint),
        "calibration_predictions": _binding(predictions_path),
        "calibration_partition": {
            "samples": int(corpus_summary["split"]["partitions"]["calibration"]["samples"]),
            "groups": int(corpus_summary["split"]["partitions"]["calibration"]["groups"]),
            "samples_sha256": corpus_summary["split"]["partitions"]["calibration"]["samples_sha256"],
            "group_ids_sha256": corpus_summary["split"]["partitions"]["calibration"]["group_ids_sha256"],
        },
        "checkpoint_selection": "maximum calibration mAP50-95; Ultralytics latest epoch attaining the maximum is retained on exact ties",
        "calibration_ultralytics": dict(calibration_ultralytics),
        "confidence_selection": dict(calibration_threshold),
        "fixed_padding_fraction": 0.05,
        "selected_model_identity": dict(model_identity),
        "runtime_identity": dict(runtime_identity),
        "training_evidence": dict(training_evidence),
        "independent_validation_model_inference_before_claim": False,
        "independent_validation_performance_metrics_before_claim": False,
        "independent_validation_format_integrity_preflight_permitted": True,
        "field_manifest_opened": False,
        "field_images_opened": False,
        "field_labels_opened": False,
        "trainer_sha256": sha256_file(TRAINER_PATH),
        "experiment_protocol_sha256": sha256_file(EXPERIMENT_PROTOCOL_PATH),
    }
    _atomic_new_json(run_dir / "selection_claim.json", claim)
    return claim


def _validate_existing_selection_claim(
    *,
    claim_path: Path,
    checkpoint: Path,
    selected_model: Any,
    corpus_root: Path,
    corpus_summary: Mapping[str, Any],
    calibration_samples: Sequence[Mapping[str, Any]],
    run_intent_path: Path,
    run_intent: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    claim = _json(claim_path)
    _require(
        claim.get("protocol") == SELECTION_CLAIM_PROTOCOL
        and claim.get("status") == "selection_frozen_before_independent_validation",
        "selection claim drift",
    )
    _require(int(claim.get("seed", -1)) == int(args.seed), "selection claim seed drift")
    _require(
        _resolve_binding(claim.get("run_intent"), label="selection run intent")
        == run_intent_path.resolve(strict=True),
        "selection claim run-intent path drift",
    )
    _require(
        claim.get("run_intent", {}).get("sha256") == sha256_file(run_intent_path),
        "selection claim run-intent hash drift",
    )
    _require(
        _resolve_binding(claim.get("corpus_summary"), label="selection corpus summary")
        == (corpus_root / "summary.json").resolve(strict=True)
        and _resolve_binding(claim.get("corpus_seal"), label="selection corpus seal")
        == (corpus_root / "seal.json").resolve(strict=True),
        "selection claim corpus binding drift",
    )
    _require(
        _resolve_binding(claim.get("checkpoint"), label="selection checkpoint")
        == checkpoint.resolve(strict=True),
        "selection claim checkpoint drift",
    )
    expected_partition = corpus_summary["split"]["partitions"]["calibration"]
    _require(
        claim.get("calibration_partition")
        == {
            "samples": int(expected_partition["samples"]),
            "groups": int(expected_partition["groups"]),
            "samples_sha256": expected_partition["samples_sha256"],
            "group_ids_sha256": expected_partition["group_ids_sha256"],
        },
        "selection claim calibration partition drift",
    )
    _require(
        claim.get("checkpoint_selection")
        == "maximum calibration mAP50-95; Ultralytics latest epoch attaining the maximum is retained on exact ties",
        "selection claim checkpoint rule drift",
    )
    predictions_path = _resolve_binding(
        claim.get("calibration_predictions"), label="selection calibration predictions"
    )
    _require(predictions_path.parent == claim_path.parent, "calibration predictions escape run directory")
    prediction_rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    recomputed = choose_confidence_threshold(calibration_samples, prediction_rows)
    _require(recomputed == claim.get("confidence_selection"), "selection threshold claim is not reproducible")
    _require(
        claim.get("selected_model_identity") == _selected_model_identity(selected_model),
        "selection claim model identity drift",
    )
    evidence = _training_evidence(
        run_dir=claim_path.parent, selected_model=selected_model, intent=run_intent
    )
    _require(evidence == claim.get("training_evidence"), "selection claim training evidence drift")
    _require(dict(runtime_identity) == claim.get("runtime_identity"), "selection claim runtime drift")
    for key in (
        "independent_validation_model_inference_before_claim",
        "independent_validation_performance_metrics_before_claim",
        "field_manifest_opened",
        "field_images_opened",
        "field_labels_opened",
    ):
        _require(claim.get(key) is False, f"selection claim chronology/scope drift: {key}")
    _require(claim.get("trainer_sha256") == sha256_file(TRAINER_PATH), "selection claim/trainer drift")
    _require(
        claim.get("experiment_protocol_sha256") == sha256_file(EXPERIMENT_PROTOCOL_PATH),
        "selection claim/protocol drift",
    )
    return claim, evidence


def train_formal(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    protocol = _load_experiment_protocol()
    corpus_root, corpus_summary, corpus_seal = _preselection_corpus(args.corpus, protocol)
    pretrained = _validate_pretrained(args.pretrained, protocol)

    import torch
    import ultralytics
    from ultralytics import YOLO

    runtime_identity = _runtime_identity(protocol)
    _require(torch.cuda.is_available(), "formal detector training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_root = _safe_root(args.output_dir, label="training output")
    run_dir = output_root / f"seed_{args.seed}"
    summary_path = run_dir / "summary.json"
    seal_path = run_dir / "seal.json"
    _require(not summary_path.exists(), f"completed detector summary exists: {summary_path}")
    _require(not seal_path.exists(), f"terminal detector seal exists without a reusable summary: {seal_path}")
    selection_path = run_dir / "selection_claim.json"
    run_intent_path = output_root / f"seed_{args.seed}.run_intent.json"

    if args.resume:
        _require(run_dir.is_dir(), f"resume run directory absent: {run_dir}")
    else:
        _require(not run_dir.exists(), f"refusing to overwrite incomplete run without --resume: {run_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_intent = _run_intent(
        path=run_intent_path,
        protocol=protocol,
        corpus_root=corpus_root,
        output_root=output_root,
        pretrained=pretrained,
        runtime_identity=runtime_identity,
        args=args,
    )

    image_size = int(protocol["model"]["image_size"])
    batch_size = int(protocol["model"]["batch_size"])
    train_yaml = corpus_root / str(corpus_summary["artifacts"]["train_calibration_yaml"])
    selection_protocol = protocol["selection"]
    confidence_floor = float(selection_protocol["prediction_confidence_floor"])
    nms_iou = float(selection_protocol["prediction_nms_iou"])
    padding_fraction = float(selection_protocol["fixed_padding_fraction"])
    calibration_samples = _load_partition(corpus_root, corpus_summary, "calibration")
    if selection_path.exists():
        _require(args.resume, "selection claim exists; explicit --resume is required")
        preliminary = _json(selection_path)
        checkpoint = _resolve_binding(
            preliminary.get("checkpoint"), label="selected checkpoint"
        )
        _require(
            checkpoint == (run_dir / "weights/best.pt").resolve(strict=True),
            "selection claim does not bind this run's best.pt",
        )
        model = YOLO(str(checkpoint))
        model_identity = _selected_model_identity(model)
        claim, training_evidence = _validate_existing_selection_claim(
            claim_path=selection_path,
            checkpoint=checkpoint,
            selected_model=model,
            corpus_root=corpus_root,
            corpus_summary=corpus_summary,
            calibration_samples=calibration_samples,
            run_intent_path=run_intent_path,
            run_intent=run_intent,
            runtime_identity=runtime_identity,
            args=args,
        )
        calibration_ultralytics = dict(claim["calibration_ultralytics"])
        calibration_threshold = dict(claim["confidence_selection"])
        calibration_predictions_path = _resolve_binding(
            claim["calibration_predictions"], label="calibration predictions"
        )
    else:
        if args.resume:
            last = run_dir / "weights/last.pt"
            _require(last.is_file(), f"resume checkpoint absent: {last}")
            resume_model = YOLO(str(last))
            checkpoint_metadata = getattr(resume_model, "ckpt", None)
            if isinstance(checkpoint_metadata, Mapping) and int(
                checkpoint_metadata.get("epoch", -1)
            ) >= 0:
                _validate_resume_checkpoint(resume_model, run_intent)
                resume_model.train(resume=True, device=args.device)
            else:
                # The optimizer-stripped last.pt proves training reached its
                # terminal finalization.  Resume the post-training selection
                # stage instead of trying to train a stripped checkpoint.
                best = run_dir / "weights/best.pt"
                _require(best.is_file(), "terminal last.pt exists but best.pt is absent")
        else:
            model = YOLO(str(pretrained))
            model.train(
                data=str(train_yaml),
                project=str(output_root),
                name=f"seed_{args.seed}",
                exist_ok=False,
                device=args.device,
                seed=args.seed,
                deterministic=True,
                epochs=int(protocol["model"]["epochs_maximum"]),
                patience=int(protocol["model"]["early_stopping_patience"]),
                batch=batch_size,
                imgsz=image_size,
                workers=args.workers,
                optimizer=str(protocol["model"]["optimizer"]),
                lr0=float(protocol["model"]["initial_learning_rate"]),
                weight_decay=float(protocol["model"]["weight_decay"]),
                amp=bool(protocol["model"]["amp"]),
                close_mosaic=10,
                cache=False,
                plots=False,
                verbose=True,
            )
        checkpoint = run_dir / "weights/best.pt"
        _require(checkpoint.is_file(), "training ended without best.pt")
        model = YOLO(str(checkpoint))
        model_identity = _selected_model_identity(model)
        training_evidence = _training_evidence(
            run_dir=run_dir, selected_model=model, intent=run_intent
        )
        calibration_result = model.val(
            data=str(train_yaml),
            split="val",
            imgsz=image_size,
            batch=batch_size,
            device=args.device,
            project=str(run_dir),
            name="calibration_eval",
            exist_ok=True,
            plots=False,
            save_json=False,
            verbose=False,
        )
        calibration_ultralytics = _ultralytics_metrics(calibration_result)
        calibration_predictions = _prediction_rows(
            model,
            calibration_samples,
            corpus_root=corpus_root,
            image_size=image_size,
            batch_size=batch_size,
            device=args.device,
            confidence_floor=confidence_floor,
            nms_iou=nms_iou,
        )
        calibration_predictions_path = run_dir / "calibration_predictions.jsonl"
        _atomic_jsonl(calibration_predictions_path, calibration_predictions)
        calibration_threshold = choose_confidence_threshold(
            calibration_samples, calibration_predictions
        )
        claim = _selection_claim(
            run_dir=run_dir,
            corpus_root=corpus_root,
            corpus_summary=corpus_summary,
            checkpoint=checkpoint,
            calibration_ultralytics=calibration_ultralytics,
            calibration_threshold=calibration_threshold,
            predictions_path=calibration_predictions_path,
            model_identity=model_identity,
            run_intent_path=run_intent_path,
            runtime_identity=runtime_identity,
            training_evidence=training_evidence,
            args=args,
        )

    # This full corpus verification and validation-row read occur only after the
    # immutable selection claim exists.
    verify_corpus(corpus_root)
    validation_samples = _load_partition(corpus_root, corpus_summary, "validation")
    validation_yaml = corpus_root / str(corpus_summary["artifacts"]["validation_yaml"])
    validation_result = model.val(
        data=str(validation_yaml),
        split="val",
        imgsz=image_size,
        batch=batch_size,
        device=args.device,
        project=str(run_dir),
        name="independent_validation_eval",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    validation_ultralytics = _ultralytics_metrics(validation_result)
    validation_predictions = _prediction_rows(
        model,
        validation_samples,
        corpus_root=corpus_root,
        image_size=image_size,
        batch_size=batch_size,
        device=args.device,
        confidence_floor=confidence_floor,
        nms_iou=nms_iou,
    )
    validation_predictions_path = run_dir / "validation_predictions.jsonl"
    _atomic_jsonl(validation_predictions_path, validation_predictions)
    threshold = float(calibration_threshold["selected"]["confidence_threshold"])
    validation_selected = selected_box_metrics(
        validation_samples, validation_predictions, threshold=threshold
    )
    gate_spec = protocol["independent_validation_gate"]
    gate_checks = {
        "selected_box_iou50_recall": validation_selected["selected_box_iou50_recall"] >= float(gate_spec["selected_box_iou50_recall_minimum"]),
        "selected_box_precision": validation_selected["selected_box_precision"] >= float(gate_spec["selected_box_precision_minimum"]),
        "mean_selected_iou": validation_selected["mean_selected_iou"] >= float(gate_spec["mean_selected_iou_minimum"]),
        "ultralytics_map50": _metric(validation_ultralytics, "mAP50(B)") >= float(gate_spec["ultralytics_map50_minimum"]),
        "ultralytics_map50_95": _metric(validation_ultralytics, "mAP50-95(B)") >= float(gate_spec["ultralytics_map50_95_minimum"]),
    }
    gate_pass = all(gate_checks.values())
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "complete" if gate_pass else "complete_gate_failed",
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "public_data_only": True,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
        },
        "corpus": {
            "root": str(corpus_root),
            "summary": _binding(corpus_root / "summary.json"),
            "seal": _binding(corpus_root / "seal.json"),
            "corpus_protocol": CORPUS_PROTOCOL,
            "corpus_seal_protocol": CORPUS_SEAL_PROTOCOL,
        },
        "pretrained": {
            "official_url": protocol["model"]["initial_checkpoint"]["public_url"],
            "asset_release": protocol["model"]["initial_checkpoint"]["asset_release"],
            "checkpoint": _binding(pretrained),
        },
        "implementation": {
            "trainer": _binding(TRAINER_PATH),
            "experiment_protocol": _binding(EXPERIMENT_PROTOCOL_PATH),
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "runtime_identity": runtime_identity,
        },
        "run": {
            "seed": int(args.seed),
            "device": str(args.device),
            "workers": int(args.workers),
            "image_size": image_size,
            "batch_size": batch_size,
            "elapsed_seconds": time.time() - started,
            "resume": bool(args.resume),
            "intent": _binding(run_intent_path),
            "training_evidence": training_evidence,
        },
        "selection": {
            "partition": "calibration",
            "validation_model_inference_and_metrics_after_selection_claim": True,
            "claim": _binding(run_dir / "selection_claim.json"),
            "checkpoint_rule": "maximum calibration mAP50-95; Ultralytics latest epoch attaining the maximum is retained on exact ties",
            "ultralytics_metrics": calibration_ultralytics,
            "confidence": calibration_threshold,
            "padding_fraction": padding_fraction,
            "prediction_confidence_floor": confidence_floor,
            "prediction_nms_iou": nms_iou,
            "runtime_integer_box_quantization": selection_protocol[
                "runtime_integer_box_quantization"
            ],
            "selected_model_identity": model_identity,
        },
        "validation": {
            "partition": "independent_validation",
            "model_inference_and_metrics_after_selection_claim": True,
            "samples": len(validation_samples),
            "groups": int(corpus_summary["split"]["partitions"]["validation"]["groups"]),
            "ultralytics_metrics": validation_ultralytics,
            "selected_box_metrics": validation_selected,
            "gate_specification": gate_spec,
            "gate_checks": gate_checks,
            "gate_pass": gate_pass,
        },
        "artifacts": {
            "best_checkpoint": _binding(checkpoint),
            "last_checkpoint": _binding(run_dir / "weights/last.pt"),
            "calibration_predictions": _binding(calibration_predictions_path),
            "validation_predictions": _binding(validation_predictions_path),
        },
    }
    _atomic_json(summary_path, summary)
    seal = {
        "schema_version": 1,
        "protocol": TRAINING_SEAL_PROTOCOL,
        "status": "sealed" if gate_pass else "sealed_gate_failed",
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint_sha256": summary["artifacts"]["best_checkpoint"]["sha256"],
        "selection_claim_sha256": summary["selection"]["claim"]["sha256"],
        "corpus_summary_sha256": summary["corpus"]["summary"]["sha256"],
        "corpus_seal_sha256": summary["corpus"]["seal"]["sha256"],
        "experiment_protocol_sha256": summary["implementation"]["experiment_protocol"]["sha256"],
        "trainer_sha256": summary["implementation"]["trainer"]["sha256"],
        "run_intent_sha256": summary["run"]["intent"]["sha256"],
        "training_results_sha256": training_evidence["results_csv"]["sha256"],
        "validation_gate_pass": gate_pass,
    }
    _atomic_json(run_dir / "seal.json", seal)
    return summary


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_experiment_protocol()
    corpus_root, corpus_summary, _ = _preselection_corpus(args.corpus, protocol)
    pretrained = _validate_pretrained(args.pretrained, protocol)
    runtime_identity = _runtime_identity(protocol)
    return {
        "status": "ready",
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(EXPERIMENT_PROTOCOL_PATH),
        "corpus": str(corpus_root),
        "corpus_summary_sha256": sha256_file(corpus_root / "summary.json"),
        "train_samples": corpus_summary["split"]["partitions"]["train"]["samples"],
        "calibration_samples": corpus_summary["split"]["partitions"]["calibration"]["samples"],
        "validation_samples_declared": corpus_summary["split"]["partitions"]["validation"]["samples"],
        "pretrained": str(pretrained),
        "pretrained_sha256": sha256_file(pretrained),
        "ultralytics_version": runtime_identity["ultralytics_version"],
        "runtime_identity": runtime_identity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = validate_inputs(args) if args.validate_only else train_formal(args)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
