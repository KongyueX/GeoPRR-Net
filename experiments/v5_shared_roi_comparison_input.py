"""Shared, label-free canonical-ROI boundary for matched V5 comparisons.

The source-side materializer is allowed to use the public SyncG/train dial box
to construct one canonical meter ROI.  This module defines the *other* side of
that firewall: every comparator receives the exact same lossless PNG bytes and
decoded BGR pixels, and receives no source row, crop box, keypoint, ScaleMark,
physical range, target, error, or annotation path.

This is deliberately an input/attestation layer, not an evaluator.  In
particular, importing or using it never opens a field, test, sealed, or
confirmatory dataset and never starts model inference on its own.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from experiments.v5_unified_two_stage_retest import (
    CANONICAL_TIGHT_ROI_CONTRACT,
    canonical_json_sha256 as unified_contract_sha256,
)


PROTOCOL: Final[str] = "v5_shared_roi_comparison_input_v1"
PROOF_PROTOCOL: Final[str] = "v5_shared_roi_comparison_no_gt_proof_v1"
PACKAGE_PROTOCOL: Final[str] = "v5_shared_roi_comparison_preflight_v1"
PUBLIC_ROSTER_PROTOCOL: Final[str] = "syncg_public_group_roster_v1"
COMPARATOR_EXECUTION_PROTOCOL: Final[str] = (
    "v5_shared_roi_comparator_execution_v1"
)
FAILURE_PENALTY: Final[float] = 1.0

PACKAGE_DESCRIPTOR_NAME: Final[str] = "package.json"
PUBLIC_ROSTER_NAME: Final[str] = "public_roster.jsonl"
CONSUMER_MANIFEST_NAME: Final[str] = "consumer_manifest.jsonl"
PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"

MATCHED_COMPARATORS: Final[tuple[str, ...]] = (
    "v5",
    "base",
    "pepd",
    "fadr",
    "vdn",
    "original_transformer",
)

COMPARATOR_INTERFACES: Final[dict[str, str]] = {
    "v5": "canonical_roi_provider",
    "base": "canonical_roi_provider",
    "pepd": "canonical_roi_provider_or_label_free_direction_adapter",
    "fadr": "canonical_roi_composite_provider",
    "vdn": "canonical_roi_provider_or_label_free_direction_adapter",
    "original_transformer": "canonical_roi_provider",
}

# The unified evaluator already freezes this downstream whole-ROI contract.
# Keeping its hash makes rows directly convertible to the existing PEPD/VDN
# label-free adapter schema.
ROI_CONTRACT: Final[dict[str, Any]] = dict(CANONICAL_TIGHT_ROI_CONTRACT)
ROI_CONTRACT_SHA256: Final[str] = unified_contract_sha256(ROI_CONTRACT)

# This additional contract makes the previously implicit source-side operation
# explicit.  The native-resolution slice is materialized once; each method may
# then perform only its documented native resize on the same supplied pixels.
MATERIALIZATION_CONTRACT: Final[dict[str, Any]] = {
    "schema_version": 1,
    "name": "v5_shared_native_tight_roi_png_v1",
    "source_scope": "verified SyncG/train only",
    "offline_selector": "public dial_bbox only",
    "bbox_rule": "finite xyxy; floor left/top; ceil right/bottom; clip in bounds",
    "crop_rule": "direct native-resolution BGR slice",
    "encoding": "lossless PNG via cv2.imencode",
    "png_compression": 3,
    "bbox_expansion": 1.0,
    "resize_before_materialization": False,
    "letterbox": False,
    "padding": False,
    "constant_black_border": False,
    "geometric_correction": False,
    "consumer_rule": "whole supplied ROI; bbox=None; no second crop",
    "native_resize_allowed": True,
    "stage_order": (
        "materialize one native tight ROI once; authenticate PNG and decoded "
        "pixels; then each comparator directly resizes the whole supplied ROI "
        "to its checkpoint-native size"
    ),
    "downstream_256_resize_interpretation": (
        "the inherited canonical_tight_roi_v1 public 256x256 resize is the V5 "
        "native consumer resize, not a resize before shared PNG materialization"
    ),
    "source_manifest_forwarded": False,
    "bbox_coordinates_forwarded": False,
    "annotation_path_forwarded": False,
    "physical_scale_forwarded": False,
    "scalemark_geometry_forwarded": False,
    "ground_truth_reading_forwarded": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


MATERIALIZATION_CONTRACT_SHA256: Final[str] = canonical_sha256(
    MATERIALIZATION_CONTRACT
)

# This identity is recomputable from the checked public SyncG/train manifest by
# grouping on group_id and ordering groups by SHA256(f"20260720:{group_id}").
# It freezes the only common source-domain holdout authorized for comparison.
PUBLIC_COHORT_CONTRACT: Final[dict[str, Any]] = {
    "schema_version": 1,
    "protocol": "syncg_group_disjoint_common_holdout_v1",
    "dataset": "SyncG",
    "source_split": "train",
    "comparison_partition": "common_holdout",
    "split_seed": 20260720,
    "validation_fraction": 0.10,
    "source_manifest_sha256": (
        "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
    ),
    "source_manifest_protocol_sha256": (
        "315e17ac8aba46d00f84a0060dba145d7e036fa096bd26423dbd34a170600c59"
    ),
    "public": {
        "samples": 16_000,
        "groups": 725,
        "sample_ids_sha256": (
            "6c1bcfd7a6a83c07e6a8d6c133f0fc48abed543ca02d02c6ee46804240216e75"
        ),
        "group_ids_sha256": (
            "3e2567eb4a9a2707a31ae7c0d57f5902c6f015b46d7abcaa7b07db885a976e28"
        ),
        "sample_group_pairs_sha256": (
            "38a9dc28db15c3a3ed8e54fd32f237fa9ab151eeea94c5d9b18beca31fa671b1"
        ),
    },
    "fit": {
        "samples": 14_375,
        "groups": 652,
        "sample_ids_sha256": (
            "b38e477fdf8667bc012023731756aa4fa275fd6e931a87bc065fec9d8ef88979"
        ),
        "group_ids_sha256": (
            "e936e02292f7b550679026499ee106db974f48ba25a0278131b0335c4141b87e"
        ),
        "sample_group_pairs_sha256": (
            "037f1a18c98869fd3acb52f83fc44efbd0057993a6c6549b930e6a29d7f0a51b"
        ),
    },
    "holdout": {
        "samples": 1_625,
        "groups": 73,
        "sample_ids_sha256": (
            "7550cf807f6669723c8a58cf80c7e1046af4aaea8bacfebc781dcb70d899fca2"
        ),
        "group_ids_sha256": (
            "16e950f77e2f7929a6dc9a0abd06fbc454e6dba5a5f40c6f4003c0c0a2b67bbc"
        ),
        "sample_group_pairs_sha256": (
            "34add28d7e3379e62978e9f26980930d63d8297484f0dcafbf395767145a9ce2"
        ),
    },
    "sample_overlap": 0,
    "group_overlap": 0,
}
PUBLIC_COHORT_CONTRACT_SHA256: Final[str] = canonical_sha256(
    PUBLIC_COHORT_CONTRACT
)

PUBLIC_ROSTER_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "protocol", "sample_id", "group_id", "dataset", "split"}
)

CONSUMER_ROW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "group_id",
        "dataset",
        "split",
        "partition",
        "image_path",
        "image_sha256",
        "canonical_roi_sha256",
        "canonical_roi_pixel_sha256",
        "frame_sha256",
        "roi_shape",
        "roi_contract_sha256",
        "materialization_contract_sha256",
        "input_schema_sha256",
    }
)
INPUT_SCHEMA_SHA256: Final[str] = canonical_sha256(sorted(CONSUMER_ROW_KEYS))

PACKAGE_DESCRIPTOR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "public_cohort_contract_sha256",
        "materialization_contract_sha256",
        "roi_contract_sha256",
        "input_schema_sha256",
        "failure_penalty",
        "public_roster_path",
        "public_roster_sha256",
        "consumer_manifest_path",
        "consumer_manifest_sha256",
    }
)

CONSUMER_VISIBLE_FIELDS: Final[tuple[str, ...]] = (
    "sample_id",
    "group_id",
    "roi_bytes",
    "image_bgr",
    "roi_file_sha256",
    "roi_pixel_sha256",
    "roi_contract_sha256",
    "input_schema_sha256",
)

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_EXACT: Final[frozenset[str]] = frozenset(
    {
        "actual",
        "actualreading",
        "annotation",
        "annotationpath",
        "bbox",
        "dialbbox",
        "error",
        "errors",
        "groundtruth",
        "gt",
        "keypoint",
        "keypoints",
        "label",
        "labels",
        "manualreference",
        "metadata",
        "meterbbox",
        "pointerangle",
        "rangeangle",
        "referencepacket",
        "scaleend",
        "scalemark",
        "scalemarkpositions",
        "scalestart",
        "startangle",
        "target",
        "targets",
        "truth",
    }
)
_FORBIDDEN_KEY_FRAGMENTS: Final[tuple[str, ...]] = (
    "groundtruth",
    "scalemark",
    "keypoint",
    "annotation",
    "manualreference",
    "meterbbox",
    "dialbbox",
    "targetprogress",
    "targetendpoint",
    "physicalscale",
    "absoluteerror",
)


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} is not lowercase SHA-256")
    digest = value.strip()
    _require(
        digest == value and bool(_SHA256_RE.fullmatch(digest)),
        f"{label} is not lowercase SHA-256",
    )
    return digest


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path))))


def _validated_package_root(package_root: Path) -> Path:
    root_value = Path(package_root)
    _require(root_value.is_absolute(), "package_root must be absolute")
    root = _absolute_without_symlink_resolution(root_value)
    _require(root.exists() and root.is_dir(), "package_root is not a directory")
    _require(not root.is_symlink(), "package_root must not be a symbolic link")
    return root.resolve(strict=True)


def _resolve_package_path(
    path: Path,
    *,
    package_root: Path,
    expected_parent: str | None = None,
    expected_name: str | None = None,
    require_file: bool = True,
) -> Path:
    """Resolve one package member while rejecting lexical and symlink escapes."""

    root = _validated_package_root(package_root)
    raw = Path(path)
    _require(raw.is_absolute(), "package member path must be absolute")
    lexical = _absolute_without_symlink_resolution(raw)
    expected_base = root if expected_parent is None else root / expected_parent
    _require(
        lexical.is_relative_to(expected_base),
        "package member path escapes its expected package directory",
    )
    if expected_name is not None:
        _require(lexical.name == expected_name, f"expected package member {expected_name}")

    relative = lexical.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"symbolic link forbidden in package path: {cursor}")

    resolved = lexical.resolve(strict=True)
    _require(resolved.is_relative_to(root), "resolved package member escapes package_root")
    if require_file:
        _require(resolved.is_file(), f"package member is not a file: {resolved}")
    return resolved


def canonical_roi_pixel_sha256(image_bgr: np.ndarray) -> str:
    """Hash decoded pixels exactly like the strict legacy ROI adapters."""

    image = _validated_bgr(image_bgr)
    header = canonical_json_bytes(
        {
            "protocol": "canonical_meter_roi_bgr_uint8_v1",
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "channel_order": "BGR",
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(image.tobytes())
    return digest.hexdigest()


def _validated_bgr(image_bgr: np.ndarray) -> np.ndarray:
    _require(isinstance(image_bgr, np.ndarray), "canonical ROI is not a NumPy array")
    _require(
        image_bgr.dtype == np.uint8
        and image_bgr.ndim == 3
        and image_bgr.shape[2] == 3,
        "canonical ROI must be uint8 BGR with shape [H,W,3]",
    )
    _require(
        image_bgr.shape[0] >= 2 and image_bgr.shape[1] >= 2,
        "canonical ROI is too small",
    )
    return np.ascontiguousarray(image_bgr)


def canonical_tight_roi_native(
    image_bgr: np.ndarray,
    bbox_xyxy: Sequence[float],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Apply the frozen floor/ceil/clip crop without resize, padding or border."""

    image = _validated_bgr(image_bgr)
    _require(len(bbox_xyxy) >= 4, "dial bbox must contain xyxy")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy[:4])
    _require(
        all(math.isfinite(value) for value in (x1, y1, x2, y2))
        and x2 > x1
        and y2 > y1,
        "invalid canonical bbox",
    )
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(math.floor(x1))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    right = max(left + 1, min(width, int(math.ceil(x2))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    crop = _validated_bgr(image[top:bottom, left:right])
    return crop.copy(), (left, top, right, bottom)


def direct_resize_whole_roi(image_bgr: np.ndarray, *, size: int) -> np.ndarray:
    """Resize every supplied ROI pixel exactly like the frozen V5 input path."""

    image = _validated_bgr(image_bgr)
    _require(isinstance(size, int) and not isinstance(size, bool) and size >= 2,
             "native resize size must be an integer >= 2")
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def encode_lossless_png(image_bgr: np.ndarray) -> bytes:
    """Encode the authenticated native ROI using the frozen lossless setting."""

    image = _validated_bgr(image_bgr)
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    _require(bool(ok), "canonical ROI PNG encoding failed")
    payload = bytes(encoded)
    _require(payload.startswith(PNG_SIGNATURE), "canonical ROI encoder did not emit PNG")
    return payload


def assert_no_supervision_fields(value: Any, *, location: str = "consumer_input") -> None:
    """Reject supervision/crop geometry at the comparator boundary.

    This examines keys recursively.  The exact top-level schema is checked
    separately, so an opaque metadata container cannot be used to hide labels.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if normalized in _FORBIDDEN_KEY_EXACT or any(
                fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS
            ):
                raise ValueError(f"forbidden supervision/crop key {key!r} at {location}")
            assert_no_supervision_fields(nested, location=f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            assert_no_supervision_fields(nested, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {location}")


def validate_consumer_row(
    row: Mapping[str, Any],
    *,
    package_root: Path,
) -> dict[str, Any]:
    """Validate one exact, label-free row without opening the ROI."""

    _require(isinstance(row, Mapping), "consumer row is not an object")
    unexpected = set(row) - CONSUMER_ROW_KEYS
    missing = CONSUMER_ROW_KEYS - set(row)
    _require(not unexpected, f"unexpected consumer fields: {sorted(unexpected)}")
    _require(not missing, f"missing consumer fields: {sorted(missing)}")
    assert_no_supervision_fields(row)
    _require(row.get("schema_version") == 1, "consumer schema version drift")
    _require(row.get("protocol") == PROTOCOL, "consumer protocol drift")
    sample_id = str(row.get("sample_id") or "")
    group_id = str(row.get("group_id") or "")
    _require(bool(sample_id), "consumer sample_id is empty")
    _require(bool(group_id), f"{sample_id}: consumer group_id is empty")
    _require(row.get("dataset") == "SyncG", f"{sample_id}: dataset is not SyncG")
    _require(row.get("split") == "train", f"{sample_id}: split is not train")
    _require(
        row.get("partition") == "common_holdout",
        f"{sample_id}: partition is not the frozen common holdout",
    )
    image_hash = _require_sha256(row.get("image_sha256"), label=f"{sample_id}.image")
    roi_hash = _require_sha256(
        row.get("canonical_roi_sha256"), label=f"{sample_id}.canonical_roi"
    )
    _require(image_hash == roi_hash, f"{sample_id}: image/ROI byte hashes differ")
    _require_sha256(
        row.get("canonical_roi_pixel_sha256"), label=f"{sample_id}.pixels"
    )
    _require_sha256(row.get("frame_sha256"), label=f"{sample_id}.frame")
    _require(
        row.get("roi_contract_sha256") == ROI_CONTRACT_SHA256,
        f"{sample_id}: downstream ROI contract drift",
    )
    _require(
        row.get("materialization_contract_sha256")
        == MATERIALIZATION_CONTRACT_SHA256,
        f"{sample_id}: materialization contract drift",
    )
    _require(
        row.get("input_schema_sha256") == INPUT_SCHEMA_SHA256,
        f"{sample_id}: input schema drift",
    )
    shape = row.get("roi_shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(value, int) for value in shape)
        and shape[0] >= 2
        and shape[1] >= 2
        and shape[2] == 3,
        f"{sample_id}: invalid ROI shape",
    )
    image_path = Path(str(row.get("image_path") or ""))
    _require(image_path.suffix == ".png", f"{sample_id}: ROI is not lowercase .png")
    _resolve_package_path(
        image_path,
        package_root=package_root,
        expected_parent="rois",
    )
    return dict(row)


def strict_jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict),
                f"{path}:{line_number}: JSONL row is not an object",
            )
            rows.append(value)
    return rows


def strict_json_load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"{path}: JSON root is not an object")
    return value


@dataclass(frozen=True)
class PublicRosterPartition:
    public_pairs: tuple[tuple[str, str], ...]
    fit_pairs: tuple[tuple[str, str], ...]
    holdout_pairs: tuple[tuple[str, str], ...]
    fit_groups: frozenset[str]
    holdout_groups: frozenset[str]
    identity: Mapping[str, Any]


def _partition_identity(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    ordered_pairs = sorted((str(sample_id), str(group_id)) for sample_id, group_id in pairs)
    sample_ids = sorted(sample_id for sample_id, _ in ordered_pairs)
    group_ids = sorted({group_id for _, group_id in ordered_pairs})
    return {
        "samples": len(ordered_pairs),
        "groups": len(group_ids),
        "sample_ids_sha256": canonical_sha256(sample_ids),
        "group_ids_sha256": canonical_sha256(group_ids),
        "sample_group_pairs_sha256": canonical_sha256(
            [[sample_id, group_id] for sample_id, group_id in ordered_pairs]
        ),
    }


def partition_public_roster(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_seed: int,
    validation_fraction: float,
) -> PublicRosterPartition:
    """Recompute the frozen grouped split from a supervision-free public roster."""

    _require(isinstance(split_seed, int) and not isinstance(split_seed, bool),
             "split_seed must be an integer")
    _require(
        isinstance(validation_fraction, (int, float))
        and not isinstance(validation_fraction, bool)
        and math.isfinite(float(validation_fraction))
        and 0.0 < float(validation_fraction) < 1.0,
        "validation_fraction must be between zero and one",
    )
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    by_group: dict[str, list[str]] = {}
    for index, row in enumerate(rows, 1):
        _require(isinstance(row, Mapping), f"public roster row {index} is not an object")
        _require(
            set(row) == PUBLIC_ROSTER_KEYS,
            f"public roster row {index} schema drift",
        )
        assert_no_supervision_fields(row, location=f"public_roster[{index}]")
        _require(row.get("schema_version") == 1, f"public roster row {index} version drift")
        _require(row.get("protocol") == PUBLIC_ROSTER_PROTOCOL,
                 f"public roster row {index} protocol drift")
        _require(row.get("dataset") == "SyncG", f"public roster row {index} dataset drift")
        _require(row.get("split") == "train", f"public roster row {index} split drift")
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id) and bool(group_id), f"public roster row {index} lacks identity")
        _require(sample_id not in seen, f"duplicate public sample_id {sample_id}")
        seen.add(sample_id)
        pairs.append((sample_id, group_id))
        by_group.setdefault(group_id, []).append(sample_id)
    _require(len(by_group) >= 2, "public roster requires at least two groups")

    ordered_groups = sorted(
        by_group,
        key=lambda group_id: hashlib.sha256(
            f"{split_seed}:{group_id}".encode("utf-8")
        ).digest(),
    )
    target = max(1, round(len(pairs) * float(validation_fraction)))
    holdout_groups: set[str] = set()
    holdout_count = 0
    for group_id in ordered_groups:
        if holdout_count >= target and holdout_groups:
            break
        if len(by_group) - len(holdout_groups) <= 1:
            break
        holdout_groups.add(group_id)
        holdout_count += len(by_group[group_id])
    fit_groups = set(by_group) - holdout_groups
    _require(bool(fit_groups) and bool(holdout_groups), "grouped public split is empty")
    fit_pairs = sorted(pair for pair in pairs if pair[1] in fit_groups)
    holdout_pairs = sorted(pair for pair in pairs if pair[1] in holdout_groups)
    public_pairs = sorted(pairs)
    identity = {
        "public": _partition_identity(public_pairs),
        "fit": _partition_identity(fit_pairs),
        "holdout": _partition_identity(holdout_pairs),
        "sample_overlap": len(
            {sample_id for sample_id, _ in fit_pairs}
            & {sample_id for sample_id, _ in holdout_pairs}
        ),
        "group_overlap": len(fit_groups & holdout_groups),
    }
    return PublicRosterPartition(
        public_pairs=tuple(public_pairs),
        fit_pairs=tuple(fit_pairs),
        holdout_pairs=tuple(holdout_pairs),
        fit_groups=frozenset(fit_groups),
        holdout_groups=frozenset(holdout_groups),
        identity=identity,
    )


def validate_public_roster(
    rows: Sequence[Mapping[str, Any]],
    *,
    cohort_contract: Mapping[str, Any] = PUBLIC_COHORT_CONTRACT,
) -> PublicRosterPartition:
    required_contract_keys = {
        "schema_version",
        "protocol",
        "dataset",
        "source_split",
        "comparison_partition",
        "split_seed",
        "validation_fraction",
        "source_manifest_sha256",
        "source_manifest_protocol_sha256",
        "public",
        "fit",
        "holdout",
        "sample_overlap",
        "group_overlap",
    }
    _require(set(cohort_contract) == required_contract_keys, "public cohort contract schema drift")
    _require(cohort_contract.get("schema_version") == 1, "public cohort contract version drift")
    _require(cohort_contract.get("dataset") == "SyncG", "public cohort dataset drift")
    _require(cohort_contract.get("source_split") == "train", "public cohort split drift")
    _require(
        cohort_contract.get("comparison_partition") == "common_holdout",
        "public comparison partition drift",
    )
    _require_sha256(cohort_contract.get("source_manifest_sha256"), label="source manifest")
    _require_sha256(
        cohort_contract.get("source_manifest_protocol_sha256"),
        label="source manifest protocol",
    )
    result = partition_public_roster(
        rows,
        split_seed=int(cohort_contract["split_seed"]),
        validation_fraction=float(cohort_contract["validation_fraction"]),
    )
    expected_identity = {
        "public": cohort_contract["public"],
        "fit": cohort_contract["fit"],
        "holdout": cohort_contract["holdout"],
        "sample_overlap": cohort_contract["sample_overlap"],
        "group_overlap": cohort_contract["group_overlap"],
    }
    _require(result.identity == expected_identity, "public cohort/split identity drift")
    _require(result.identity["sample_overlap"] == 0, "fit/holdout sample overlap")
    _require(result.identity["group_overlap"] == 0, "fit/holdout group overlap")
    return result


def _validate_package_descriptor(
    descriptor: Mapping[str, Any],
    *,
    cohort_contract: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(descriptor) == PACKAGE_DESCRIPTOR_KEYS, "package descriptor schema drift")
    _require(descriptor.get("schema_version") == 1, "package descriptor version drift")
    _require(descriptor.get("protocol") == PACKAGE_PROTOCOL, "package protocol drift")
    _require(
        descriptor.get("public_cohort_contract_sha256")
        == canonical_sha256(cohort_contract),
        "public cohort contract hash drift",
    )
    _require(
        descriptor.get("materialization_contract_sha256")
        == MATERIALIZATION_CONTRACT_SHA256,
        "package materialization contract drift",
    )
    _require(descriptor.get("roi_contract_sha256") == ROI_CONTRACT_SHA256,
             "package ROI contract drift")
    _require(descriptor.get("input_schema_sha256") == INPUT_SCHEMA_SHA256,
             "package input schema drift")
    _require(descriptor.get("failure_penalty") == FAILURE_PENALTY,
             "package failure penalty drift")
    _require(descriptor.get("public_roster_path") == PUBLIC_ROSTER_NAME,
             "package public roster path drift")
    _require(descriptor.get("consumer_manifest_path") == CONSUMER_MANIFEST_NAME,
             "package consumer manifest path drift")
    _require_sha256(descriptor.get("public_roster_sha256"), label="public roster")
    _require_sha256(descriptor.get("consumer_manifest_sha256"), label="consumer manifest")
    return dict(descriptor)


def load_consumer_manifest(
    path: Path,
    *,
    package_root: Path,
) -> list[dict[str, Any]]:
    cohort_contract = PUBLIC_COHORT_CONTRACT
    root = _validated_package_root(package_root)
    manifest_path = _resolve_package_path(
        Path(path),
        package_root=root,
        expected_name=CONSUMER_MANIFEST_NAME,
    )
    _require(
        manifest_path == (root / CONSUMER_MANIFEST_NAME).resolve(strict=True),
        "consumer manifest must be the package-root manifest",
    )
    descriptor_path = _resolve_package_path(
        root / PACKAGE_DESCRIPTOR_NAME,
        package_root=root,
        expected_name=PACKAGE_DESCRIPTOR_NAME,
    )
    roster_path = _resolve_package_path(
        root / PUBLIC_ROSTER_NAME,
        package_root=root,
        expected_name=PUBLIC_ROSTER_NAME,
    )
    descriptor = _validate_package_descriptor(
        strict_json_load(descriptor_path), cohort_contract=cohort_contract
    )
    _require(
        sha256_file(roster_path) == descriptor["public_roster_sha256"],
        "public roster file hash drift",
    )
    _require(
        sha256_file(manifest_path) == descriptor["consumer_manifest_sha256"],
        "consumer manifest file hash drift",
    )
    roster = validate_public_roster(
        strict_jsonl_load(roster_path), cohort_contract=cohort_contract
    )
    rows = strict_jsonl_load(manifest_path)
    _require(bool(rows), "consumer manifest is empty")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        value = validate_consumer_row(row, package_root=root)
        sample_id = str(value["sample_id"])
        _require(sample_id not in seen, f"duplicate sample_id {sample_id}")
        seen.add(sample_id)
        validated.append(value)
    observed_pairs = tuple(
        sorted((str(row["sample_id"]), str(row["group_id"])) for row in validated)
    )
    _require(
        observed_pairs == roster.holdout_pairs,
        "consumer manifest is not the complete frozen common holdout",
    )
    return validated


@dataclass(frozen=True)
class SharedROIInput:
    """The complete and only object exposed to one comparator callback."""

    sample_id: str
    group_id: str
    roi_bytes: bytes = field(repr=False)
    image_bgr: np.ndarray = field(repr=False, compare=False)
    roi_file_sha256: str
    roi_pixel_sha256: str
    roi_contract_sha256: str
    input_schema_sha256: str


def read_shared_roi(
    row: Mapping[str, Any],
    *,
    package_root: Path,
) -> SharedROIInput:
    """Open and authenticate one ROI; no source image or label file is opened."""

    value = validate_consumer_row(row, package_root=package_root)
    sample_id = str(value["sample_id"])
    path = _resolve_package_path(
        Path(str(value["image_path"])),
        package_root=package_root,
        expected_parent="rois",
    )
    payload = path.read_bytes()
    _require(payload.startswith(PNG_SIGNATURE), f"{sample_id}: canonical ROI is not PNG")
    payload_hash = sha256_bytes(payload)
    _require(
        payload_hash == value["canonical_roi_sha256"],
        f"{sample_id}: canonical ROI file hash drift",
    )
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    _require(image is not None, f"{sample_id}: canonical ROI cannot be decoded")
    image = _validated_bgr(image)
    _require(
        list(image.shape) == value["roi_shape"],
        f"{sample_id}: decoded ROI shape drift",
    )
    pixel_hash = canonical_roi_pixel_sha256(image)
    _require(
        pixel_hash == value["canonical_roi_pixel_sha256"],
        f"{sample_id}: canonical ROI pixel hash drift",
    )
    image.setflags(write=False)
    return SharedROIInput(
        sample_id=sample_id,
        group_id=str(value["group_id"]),
        roi_bytes=payload,
        image_bgr=image,
        roi_file_sha256=payload_hash,
        roi_pixel_sha256=pixel_hash,
        roi_contract_sha256=str(value["roi_contract_sha256"]),
        input_schema_sha256=str(value["input_schema_sha256"]),
    )


def _isolated_method_input(source: SharedROIInput) -> SharedROIInput:
    # Independent arrays prevent one implementation from affecting the next;
    # encoded bytes remain the same immutable bytes object for all methods.
    image = np.ascontiguousarray(source.image_bgr.copy())
    image.setflags(write=False)
    return SharedROIInput(
        sample_id=source.sample_id,
        group_id=source.group_id,
        roi_bytes=source.roi_bytes,
        image_bgr=image,
        roi_file_sha256=source.roi_file_sha256,
        roi_pixel_sha256=source.roi_pixel_sha256,
        roi_contract_sha256=source.roi_contract_sha256,
        input_schema_sha256=source.input_schema_sha256,
    )


def input_receipt(method: str, value: SharedROIInput) -> dict[str, Any]:
    _require(method in MATCHED_COMPARATORS, f"unsupported comparator {method}")
    core = {
        "schema_version": 1,
        "protocol": "v5_shared_roi_runtime_receipt_v1",
        "method": method,
        "sample_id": value.sample_id,
        "group_id": value.group_id,
        "canonical_roi_sha256": value.roi_file_sha256,
        "canonical_roi_pixel_sha256": value.roi_pixel_sha256,
        "roi_contract_sha256": value.roi_contract_sha256,
        "input_schema_sha256": value.input_schema_sha256,
        "consumer_visible_fields": list(CONSUMER_VISIBLE_FIELDS),
        "source_manifest_exposed": False,
        "crop_box_exposed": False,
        "annotation_path_exposed": False,
        "ground_truth_reading_exposed": False,
        "physical_scale_exposed": False,
        "scalemark_geometry_exposed": False,
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


Comparator = Callable[[SharedROIInput], Any]


def _comparator_execution_record(
    method: str,
    *,
    payload: Any = None,
    failure_code: str | None = None,
) -> dict[str, Any]:
    failed = failure_code is not None
    return {
        "schema_version": 1,
        "protocol": COMPARATOR_EXECUTION_PROTOCOL,
        "method": method,
        "status": not failed,
        "failure_code": failure_code,
        "failure_penalty": FAILURE_PENALTY if failed else 0.0,
        "payload": payload,
    }


def _normalize_comparator_output(method: str, value: Any) -> dict[str, Any]:
    if value is None:
        return _comparator_execution_record(
            method, failure_code="consumer_returned_none"
        )
    if isinstance(value, Mapping):
        if "status" in value:
            status = value.get("status")
            declared_success = status is True or (
                isinstance(status, str)
                and status.casefold() in {"ok", "success"}
            )
        else:
            declared_success = True
        if not declared_success:
            code = str(value.get("failure_code") or "consumer_declared_failure")
            return _comparator_execution_record(
                method,
                payload=dict(value),
                failure_code=code,
            )
    return _comparator_execution_record(method, payload=value)


def dispatch_matched_comparators(
    row: Mapping[str, Any],
    consumers: Mapping[str, Comparator],
    *,
    package_root: Path,
    require_all: bool = True,
) -> dict[str, Any]:
    """Call comparators through one immutable, supervision-free ROI boundary."""

    methods = set(consumers)
    unknown = methods - set(MATCHED_COMPARATORS)
    _require(not unknown, f"unknown comparators: {sorted(unknown)}")
    if require_all:
        missing = set(MATCHED_COMPARATORS) - methods
        _require(not missing, f"missing matched comparators: {sorted(missing)}")
    source = read_shared_roi(row, package_root=package_root)
    outputs: dict[str, Any] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for method in MATCHED_COMPARATORS:
        if method not in consumers:
            continue
        consumer = consumers[method]
        _require(callable(consumer), f"{method}: consumer is not callable")
        value = _isolated_method_input(source)
        _require(
            sha256_bytes(value.roi_bytes) == source.roi_file_sha256,
            f"{method}: encoded ROI drift before dispatch",
        )
        _require(
            canonical_roi_pixel_sha256(value.image_bgr) == source.roi_pixel_sha256,
            f"{method}: pixel ROI drift before dispatch",
        )
        try:
            raw_output = consumer(value)
            output = _normalize_comparator_output(method, raw_output)
        except Exception as exc:
            output = _comparator_execution_record(
                method,
                failure_code=f"consumer_exception:{type(exc).__name__}",
            )
        _require(
            sha256_bytes(value.roi_bytes) == source.roi_file_sha256,
            f"{method}: encoded ROI mutated during inference",
        )
        _require(
            canonical_roi_pixel_sha256(value.image_bgr) == source.roi_pixel_sha256,
            f"{method}: pixel ROI mutated during inference",
        )
        outputs[method] = output
        receipts[method] = input_receipt(method, value)
    receipt_hashes = {
        receipt["canonical_roi_sha256"] for receipt in receipts.values()
    }
    pixel_hashes = {
        receipt["canonical_roi_pixel_sha256"] for receipt in receipts.values()
    }
    _require(len(receipt_hashes) == 1, "comparators received different ROI files")
    _require(len(pixel_hashes) == 1, "comparators received different ROI pixels")
    return {
        "sample_id": source.sample_id,
        "group_id": source.group_id,
        "outputs": outputs,
        "input_receipts": receipts,
        "all_methods_same_roi_bytes": True,
        "all_methods_same_roi_pixels": True,
        "failure_penalty": FAILURE_PENALTY,
    }


def canonical_roi_provider_consumer(provider: Any) -> Comparator:
    """Adapt V5/Base/PEPD/FADR/VDN/Transformer canonical-ROI providers.

    A FADR implementation should be bound as a composite provider whose own
    Base/PEPD branches both originate from this supplied array.  No labels or
    reference packet are available through this wrapper.
    """

    predict = getattr(provider, "predict", None)
    _require(callable(predict), "provider has no callable predict method")

    def consume(value: SharedROIInput) -> Any:
        return predict(
            value.image_bgr,
            input_is_canonical_meter_roi=True,
        )

    return consume


def to_direction_adapter_row(
    row: Mapping[str, Any],
    *,
    package_root: Path,
    reference_contract_sha256: str,
) -> dict[str, Any]:
    """Project a shared row to the existing PEPD/VDN adapter schema."""

    value = validate_consumer_row(row, package_root=package_root)
    reference_hash = _require_sha256(
        reference_contract_sha256, label="reference contract"
    )
    return {
        "sample_id": value["sample_id"],
        "group_id": value["group_id"],
        "image_path": value["image_path"],
        "image_sha256": value["image_sha256"],
        "canonical_roi_sha256": value["canonical_roi_sha256"],
        "frame_sha256": value["frame_sha256"],
        "roi_contract_sha256": value["roi_contract_sha256"],
        "reference_contract_sha256": reference_hash,
    }


__all__ = [
    "COMPARATOR_EXECUTION_PROTOCOL",
    "COMPARATOR_INTERFACES",
    "CONSUMER_MANIFEST_NAME",
    "CONSUMER_ROW_KEYS",
    "CONSUMER_VISIBLE_FIELDS",
    "FAILURE_PENALTY",
    "INPUT_SCHEMA_SHA256",
    "MATCHED_COMPARATORS",
    "MATERIALIZATION_CONTRACT",
    "MATERIALIZATION_CONTRACT_SHA256",
    "PACKAGE_DESCRIPTOR_KEYS",
    "PACKAGE_DESCRIPTOR_NAME",
    "PACKAGE_PROTOCOL",
    "PNG_SIGNATURE",
    "PROOF_PROTOCOL",
    "PROTOCOL",
    "PUBLIC_COHORT_CONTRACT",
    "PUBLIC_COHORT_CONTRACT_SHA256",
    "PUBLIC_ROSTER_KEYS",
    "PUBLIC_ROSTER_NAME",
    "PUBLIC_ROSTER_PROTOCOL",
    "ROI_CONTRACT",
    "ROI_CONTRACT_SHA256",
    "PublicRosterPartition",
    "SharedROIInput",
    "assert_no_supervision_fields",
    "canonical_json_bytes",
    "canonical_roi_pixel_sha256",
    "canonical_roi_provider_consumer",
    "canonical_sha256",
    "canonical_tight_roi_native",
    "direct_resize_whole_roi",
    "dispatch_matched_comparators",
    "encode_lossless_png",
    "input_receipt",
    "load_consumer_manifest",
    "partition_public_roster",
    "read_shared_roi",
    "sha256_bytes",
    "sha256_file",
    "strict_json_load",
    "strict_jsonl_load",
    "to_direction_adapter_row",
    "validate_consumer_row",
    "validate_public_roster",
]
