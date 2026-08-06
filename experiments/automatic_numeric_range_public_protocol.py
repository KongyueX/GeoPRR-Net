"""Shared, public-only protocol utilities for automatic numeric-range evidence.

The module deliberately separates label-free inference rosters from the
SyncG range labels.  It never opens an image by itself and rejects paths that
enter field, test, confirmatory, or sealed namespaces.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PUBLIC_DATA_ROOT: Final[Path] = (PROJECT_ROOT / "datasets/SyncG/syncG").resolve()
PUBLIC_IMAGE_ROOT: Final[Path] = (PUBLIC_DATA_ROOT / "images/train").resolve()
SOURCE_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "artifacts/manifests/syncg_train.jsonl"
).resolve()
SOURCE_MANIFEST_PROTOCOL: Final[Path] = SOURCE_MANIFEST.with_name(
    SOURCE_MANIFEST.name + ".protocol.json"
)
V5_PROTOCOL_PATH: Final[Path] = (
    PROJECT_ROOT / "experiments/cagh_scalemark_reference_public_v5_protocol.json"
).resolve()
DEFAULT_DEVELOPMENT_SUMMARY: Final[Path] = Path(
    r"C:\pointer_read\automatic_numeric_range_public_screen_20260806_v4\summary.json"
)
DEFAULT_PROTOCOL_ROOT: Final[Path] = Path(
    r"C:\pointer_read\automatic_numeric_range_public_protocol_20260806_v1"
)
PROTOCOL: Final[str] = "automatic_numeric_range_public_group_holdout_v1"
ROW_PROTOCOL: Final[str] = "automatic_numeric_range_public_label_free_roster_v1"
PREDICTION_PROTOCOL: Final[str] = "automatic_numeric_range_public_predictions_v1"
CALIBRATION_PROTOCOL: Final[str] = "automatic_numeric_range_public_acceptance_v1"
VALIDATION_PROTOCOL: Final[str] = "automatic_numeric_range_public_validation_v1"
PARTITIONS: Final[tuple[str, ...]] = (
    "algorithm_fit",
    "calibration",
    "independent_validation",
    "development_excluded",
)
FORBIDDEN_PATH_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "field",
        "test",
        "sealed",
        "confirmatory",
        "confirmation",
        "xiangmu1",
        "xiangmu2",
    }
)
FORBIDDEN_RANGE_LABEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "actual",
        "actual_reading",
        "actual_value",
        "error",
        "errors",
        "expected",
        "ground_truth",
        "groundtruth",
        "gt",
        "label",
        "labels",
        "scale_end",
        "scale_start",
        "target",
        "target_reading",
        "truth",
        "truth_end",
        "truth_start",
    }
)
ROSTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "partition",
        "sample_id",
        "group_id",
        "image_relpath",
        "dial_bbox",
    }
)
PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "partition",
        "sample_id",
        "group_id",
        "canonical_roi_sha256",
        "status",
        "pred_start",
        "pred_end",
        "confidence",
        "failure_reason",
        "range_prediction",
        "sample_seconds",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value).rstrip(b"\n")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def strict_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
            value = json.loads(
                line,
                parse_constant=_reject_constant,
                object_pairs_hook=_strict_object,
            )
            require(isinstance(value, dict), f"non-object row at {path}:{line_number}")
            rows.append(value)
    require(bool(rows), f"empty JSONL file: {path}")
    return rows


def atomic_new(path: Path, payload: bytes) -> None:
    target = Path(path).resolve()
    require(not target.exists(), f"refusing to overwrite frozen artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    require(not temporary.exists(), f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_new(path, canonical_bytes(dict(value), pretty=True))


def atomic_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_new(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def _path_tokens(path: Path) -> set[str]:
    tokens: set[str] = set()
    for part in path.parts:
        normalized = part.casefold()
        for separator in ("-", ".", " "):
            normalized = normalized.replace(separator, "_")
        tokens.update(value for value in normalized.split("_") if value)
    return tokens


def guard_public_path(
    path: Path,
    *,
    label: str,
    must_exist: bool = True,
    allowed_root: Path | None = None,
    expect_file: bool = True,
) -> Path:
    candidate = Path(path)
    require(
        not (_path_tokens(candidate) & FORBIDDEN_PATH_TOKENS),
        f"{label} enters a restricted namespace: {candidate}",
    )
    resolved = candidate.resolve(strict=must_exist)
    require(
        not (_path_tokens(resolved) & FORBIDDEN_PATH_TOKENS),
        f"{label} resolves into a restricted namespace: {resolved}",
    )
    require(resolved != Path(resolved.anchor), f"{label} cannot be a drive root")
    if allowed_root is not None:
        require(
            resolved.is_relative_to(Path(allowed_root).resolve()),
            f"{label} is outside allowed public root {allowed_root}: {resolved}",
        )
    if must_exist and expect_file:
        require(resolved.is_file(), f"{label} is not a file: {resolved}")
    return resolved


def assert_range_label_free(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            require(
                normalized not in FORBIDDEN_RANGE_LABEL_KEYS,
                f"{location}: range-label key {key!r} is forbidden",
            )
            assert_range_label_free(nested, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            assert_range_label_free(nested, location=f"{location}[{index}]")


def validate_roster_rows(
    rows: Sequence[Mapping[str, Any]], *, partition: str
) -> dict[str, Any]:
    require(partition in PARTITIONS, f"unknown partition: {partition}")
    require(bool(rows), f"{partition} roster is empty")
    sample_ids: set[str] = set()
    groups: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        require(set(row) == ROSTER_KEYS, f"{partition}[{index}] roster schema drift")
        assert_range_label_free(row, location=f"{partition}[{index}]")
        require(row["schema_version"] == 1, "roster schema version drift")
        require(row["protocol"] == ROW_PROTOCOL, "roster protocol drift")
        require(row["partition"] == partition, "roster partition drift")
        sample_id = str(row["sample_id"] or "")
        group_id = str(row["group_id"] or "")
        require(bool(sample_id) and bool(group_id), "empty sample/group identity")
        require(sample_id not in sample_ids, f"duplicate sample_id: {sample_id}")
        relative = Path(str(row["image_relpath"] or ""))
        require(not relative.is_absolute(), f"absolute image path in roster: {relative}")
        require(".." not in relative.parts, f"image path escapes public root: {relative}")
        bbox = row["dial_bbox"]
        require(
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in bbox
            ),
            f"{sample_id}: invalid public dial bbox",
        )
        require(float(bbox[2]) > float(bbox[0]), f"{sample_id}: collapsed bbox x")
        require(float(bbox[3]) > float(bbox[1]), f"{sample_id}: collapsed bbox y")
        sample_ids.add(sample_id)
        groups.add(group_id)
        normalized_rows.append(row)
    return {
        "samples": len(sample_ids),
        "groups": len(groups),
        "sample_ids_sha256": canonical_sha256(sorted(sample_ids)),
        "group_ids_sha256": canonical_sha256(sorted(groups)),
        "sample_group_sha256": canonical_sha256(
            sorted((row["sample_id"], row["group_id"]) for row in normalized_rows)
        ),
        "roster_sha256": canonical_sha256(normalized_rows),
        "sample_ids": sample_ids,
        "group_ids": groups,
    }


def load_frozen_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = guard_public_path(path, label="range protocol")
    protocol = strict_json(protocol_path)
    require(protocol.get("protocol") == PROTOCOL, "range protocol identity drift")
    require(protocol.get("status") == "frozen_before_range_inference", "protocol not frozen")
    require(protocol.get("scope", {}).get("dataset") == "SyncG", "dataset drift")
    require(protocol.get("scope", {}).get("split") == "train", "split drift")
    return protocol_path, protocol


def verify_bound_file(
    protocol: Mapping[str, Any], section: str, key: str
) -> Path:
    binding = protocol.get(section, {}).get(key)
    require(isinstance(binding, Mapping), f"missing frozen binding: {section}.{key}")
    path = guard_public_path(
        Path(str(binding.get("path") or "")), label=f"{section}.{key}"
    )
    require(sha256_file(path) == binding.get("sha256"), f"{section}.{key} hash drift")
    return path


def load_partition_roster(
    protocol_path: Path, partition: str
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, Any]]:
    protocol_file, protocol = load_frozen_protocol(protocol_path)
    require(partition in PARTITIONS, f"unknown partition: {partition}")
    binding = protocol.get("partitions", {}).get(partition)
    require(isinstance(binding, Mapping), f"missing partition binding: {partition}")
    manifest_path = Path(str(binding.get("path") or ""))
    if not manifest_path.is_absolute():
        manifest_path = protocol_file.parent / manifest_path
    manifest_path = guard_public_path(manifest_path, label=f"{partition} manifest")
    require(sha256_file(manifest_path) == binding.get("sha256"), f"{partition} hash drift")
    rows = strict_jsonl(manifest_path)
    audit = validate_roster_rows(rows, partition=partition)
    for key in ("samples", "groups", "sample_ids_sha256", "group_ids_sha256"):
        require(audit[key] == binding.get(key), f"{partition}.{key} drift")
    return protocol, manifest_path, rows, audit


def validate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    roster_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    require(bool(rows), "prediction table is empty")
    observed: set[str] = set()
    groups: set[str] = set()
    for index, raw in enumerate(rows):
        row = dict(raw)
        require(set(row) == PREDICTION_KEYS, f"prediction[{index}] schema drift")
        assert_range_label_free(row, location=f"prediction[{index}]")
        require(row["schema_version"] == 1, "prediction schema version drift")
        require(row["protocol"] == PREDICTION_PROTOCOL, "prediction protocol drift")
        require(row["partition"] == partition, "prediction partition drift")
        sample_id = str(row["sample_id"] or "")
        group_id = str(row["group_id"] or "")
        require(sample_id in roster_by_id, f"prediction sample outside roster: {sample_id}")
        require(sample_id not in observed, f"duplicate prediction: {sample_id}")
        require(
            group_id == str(roster_by_id[sample_id]["group_id"]),
            f"prediction group drift: {sample_id}",
        )
        digest = str(row["canonical_roi_sha256"] or "").casefold()
        require(
            len(digest) == 64 and all(value in "0123456789abcdef" for value in digest),
            f"invalid ROI digest: {sample_id}",
        )
        require(isinstance(row["status"], bool), f"non-boolean status: {sample_id}")
        confidence = row["confidence"]
        require(
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and 0.0 <= float(confidence) <= 1.0,
            f"invalid confidence: {sample_id}",
        )
        for key in ("pred_start", "pred_end"):
            value = row[key]
            require(
                value is None
                or (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ),
                f"invalid {key}: {sample_id}",
            )
        if row["status"]:
            require(
                row["pred_start"] is not None and row["pred_end"] is not None,
                f"successful prediction has null range: {sample_id}",
            )
        seconds = row["sample_seconds"]
        require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) >= 0.0,
            f"invalid sample time: {sample_id}",
        )
        observed.add(sample_id)
        groups.add(group_id)
    return {
        "samples": len(observed),
        "groups": len(groups),
        "sample_ids": observed,
        "group_ids": groups,
        "sample_ids_sha256": canonical_sha256(sorted(observed)),
        "group_ids_sha256": canonical_sha256(sorted(groups)),
    }


def load_prediction_bundle(
    protocol_path: Path,
    prediction_root: Path,
    *,
    expected_partition: str,
    allow_smoke: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol, manifest_path, roster, roster_audit = load_partition_roster(
        protocol_path, expected_partition
    )
    root = guard_public_path(
        prediction_root,
        label="prediction bundle",
        expect_file=False,
    )
    require(root.is_dir(), f"prediction bundle is not a directory: {root}")
    summary_path = guard_public_path(root / "summary.json", label="prediction summary")
    seal_path = guard_public_path(root / "seal.json", label="prediction seal")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == PREDICTION_PROTOCOL, "prediction summary drift")
    require(summary.get("status") == "predictions_sealed", "predictions are not sealed")
    require(summary.get("partition") == expected_partition, "prediction partition drift")
    require(summary.get("mode") in ("formal", "smoke"), "prediction mode drift")
    require(allow_smoke or summary.get("mode") == "formal", "smoke predictions are ineligible")
    require(
        summary.get("parent_protocol", {}).get("sha256") == sha256_file(protocol_path),
        "parent protocol hash drift",
    )
    require(
        summary.get("partition_manifest", {}).get("sha256") == sha256_file(manifest_path),
        "partition manifest binding drift",
    )
    require(seal.get("protocol") == PREDICTION_PROTOCOL, "prediction seal drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "summary seal drift")
    prediction_path = Path(str(summary.get("artifacts", {}).get("predictions", {}).get("path") or ""))
    if not prediction_path.is_absolute():
        prediction_path = root / prediction_path
    prediction_path = guard_public_path(prediction_path, label="label-free predictions")
    prediction_hash = sha256_file(prediction_path)
    require(
        prediction_hash == summary["artifacts"]["predictions"].get("sha256"),
        "prediction artifact hash drift",
    )
    require(prediction_hash == seal.get("predictions_sha256"), "prediction seal hash drift")
    rows = strict_jsonl(prediction_path)
    roster_by_id = {str(row["sample_id"]): row for row in roster}
    audit = validate_prediction_rows(
        rows, partition=expected_partition, roster_by_id=roster_by_id
    )
    require(audit["samples"] == summary.get("samples"), "prediction sample count drift")
    require(audit["groups"] == summary.get("groups"), "prediction group count drift")
    require(
        audit["sample_ids_sha256"] == summary.get("sample_ids_sha256"),
        "prediction roster hash drift",
    )
    if summary["mode"] == "formal":
        require(audit["sample_ids"] == roster_audit["sample_ids"], "formal predictions incomplete")
        require(audit["group_ids"] == roster_audit["group_ids"], "formal groups incomplete")
    else:
        require(audit["sample_ids"].issubset(roster_audit["sample_ids"]), "smoke roster escaped")
    require(
        summary.get("audit", {}).get("range_labels_opened") == 0,
        "prediction runner reports label access",
    )
    return summary, rows, roster


def resolve_public_image(image_relpath: str) -> Path:
    return guard_public_path(
        PUBLIC_IMAGE_ROOT / image_relpath,
        label="public SyncG/train image",
        allowed_root=PUBLIC_IMAGE_ROOT,
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    require(0 <= successes <= total, "invalid binomial counts")
    if total == 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
