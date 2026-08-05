"""Freeze and freshly verify the complete VDN Phase-2 SyncG-train input identity.

The implementation is intentionally train-only.  It never enumerates the SyncG
dataset root: only the exact ``images/train`` and ``annotations/train`` trees
declared by the audited manifest protocol are traversed.  Manifest paths are
validated lexically before any target file is opened, then resolved again to
reject symlink/reparse-point escapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


INVENTORY_PROTOCOL = "vdn_phase2_syncg_train_content_inventory_v1"
INVENTORY_VERIFICATION_PROTOCOL = (
    "vdn_phase2_syncg_train_content_inventory_verification_v1"
)
INVENTORY_SCHEMA_VERSION = 1
CANONICALIZATION_PROTOCOL = "canonical_json_utf8_sort_keys_compact_v1"
SOURCE_HASH_PROTOCOL = "utf8_source_newlines_lf_v1"
SYNCG_MANIFEST_PROTOCOL = "syncg_official_split_v1"
SYNCG_TRAIN_EXPECTED_ROWS = 16_000

PINNED_MANIFEST_SHA256 = (
    "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
)
PINNED_MANIFEST_PROTOCOL_SHA256 = (
    "315e17ac8aba46d00f84a0060dba145d7e036fa096bd26423dbd34a170600c59"
)
PINNED_SAMPLE_IDS_SHA256 = (
    "6c1bcfd7a6a83c07e6a8d6c133f0fc48abed543ca02d02c6ee46804240216e75"
)

DEFAULT_MANIFEST = Path("artifacts/manifests/syncg_train.jsonl")
DEFAULT_VDN_PROTOCOL_DOCUMENT = Path("docs/VDN_CONVERGENCE_PROTOCOL_CN.md")
DEFAULT_VDN_PROTOCOL_SOURCE = Path("experiments/vdn_phase2_protocol.py")
DEFAULT_OUTPUT = Path(
    "artifacts/protocols/vdn_phase2_syncg_train_content_inventory_v1.json"
)

_SAMPLE_ID_RE = re.compile(r"^sync_([0-9]+)$")
_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_FORBIDDEN_SCOPE_PARTS = {
    "confirmatory",
    "field",
    "pointer-10k",
    "pointer10k",
    "rpm",
    "rpm-10k",
    "rpm10k",
    "sealed",
    "test",
    "testing",
    "val",
    "validation",
}


@dataclass(frozen=True)
class _RowSpec:
    line_number: int
    sample_id: str
    row: dict[str, Any]
    image_path: Path
    annotation_path: Path
    image_project_path: str
    annotation_project_path: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_file_raw(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_file(path: Path) -> str:
    path = Path(path)
    if path.suffix.casefold() not in {".md", ".py"}:
        raise ValueError(f"{path}: unsupported source suffix for {SOURCE_HASH_PROTOCOL}")
    payload = path.read_bytes()
    try:
        payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: source is not UTF-8") from exc
    canonical = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def _strict_json_loads(payload: bytes | str, *, label: str) -> Any:
    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label}: not UTF-8 JSON") from exc
    else:
        text = payload

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label}: non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: malformed JSON: {exc}") from exc


def _sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    return canonical_json_sha256(sorted(str(sample_id) for sample_id in sample_ids))


def _absolute_without_resolving(path: Path, *, base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _relative_to(path: Path, root: Path, *, role: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{role} escapes required root {root}") from exc


def _reject_forbidden_scope_parts(relative: Path, *, role: str) -> None:
    forbidden = [
        part
        for part in relative.parts
        if str(part).strip().casefold() in _FORBIDDEN_SCOPE_PARTS
    ]
    if forbidden:
        raise ValueError(f"{role} contains forbidden non-train scope {forbidden!r}")


def _is_reparse_point(path: Path) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if path.is_symlink():
        return True
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(marker and attributes & marker)


def _resolve_project_root(project_root: Path) -> Path:
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return root


def _resolve_project_file(
    path: Path,
    *,
    project_root: Path,
    role: str,
) -> tuple[Path, str]:
    lexical = _absolute_without_resolving(Path(path), base=project_root)
    _relative_to(lexical, project_root, role=role)
    if not lexical.is_file():
        raise FileNotFoundError(f"{role} is missing: {lexical}")
    if _is_reparse_point(lexical):
        raise ValueError(f"{role} is a symlink or reparse point: {lexical}")
    resolved = lexical.resolve(strict=True)
    relative = _relative_to(resolved, project_root, role=role)
    return resolved, relative.as_posix()


def _resolve_declared_root(
    value: Any,
    *,
    project_root: Path,
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("manifest protocol syncg_root must be a non-empty string")
    lexical = _absolute_without_resolving(Path(value), base=project_root)
    _relative_to(lexical, project_root, role="SyncG root")
    if not lexical.is_dir():
        raise FileNotFoundError(f"SyncG root is missing: {lexical}")
    if _is_reparse_point(lexical):
        raise ValueError(f"SyncG root is a symlink or reparse point: {lexical}")
    resolved = lexical.resolve(strict=True)
    relative = _relative_to(resolved, project_root, role="SyncG root")
    return resolved, relative.as_posix()


def _resolve_train_file(
    value: Any,
    *,
    project_root: Path,
    train_root: Path,
    role: str,
) -> tuple[Path, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} path must be a non-empty string")
    lexical = _absolute_without_resolving(Path(value), base=project_root)
    lexical_relative = _relative_to(lexical, train_root, role=role)
    _reject_forbidden_scope_parts(lexical_relative, role=role)
    if not lexical.is_file():
        raise FileNotFoundError(f"{role} is missing: {lexical}")
    if _is_reparse_point(lexical):
        raise ValueError(f"{role} is a symlink or reparse point: {lexical}")
    resolved = lexical.resolve(strict=True)
    _relative_to(resolved, train_root.resolve(strict=True), role=role)
    project_relative = _relative_to(resolved, project_root, role=role)
    return resolved, project_relative.as_posix()


def _enumerate_exact_train_files(root: Path, *, role: str) -> set[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{role} root is missing: {root}")
    if _is_reparse_point(root):
        raise ValueError(f"{role} root is a symlink or reparse point: {root}")
    files: set[Path] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in list(directory_names):
            child = directory_path / name
            if _is_reparse_point(child):
                raise ValueError(
                    f"{role} tree contains a symlink or reparse directory: {child}"
                )
        for name in file_names:
            child = directory_path / name
            if _is_reparse_point(child):
                raise ValueError(
                    f"{role} tree contains a symlink or reparse file: {child}"
                )
            if not child.is_file():
                raise ValueError(f"{role} tree contains a non-file entry: {child}")
            resolved = child.resolve(strict=True)
            _relative_to(resolved, root.resolve(strict=True), role=f"{role} file")
            files.add(resolved)
    return files


def _read_manifest(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = _strict_json_loads(line, label=f"{path}:{line_number}")
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}:{line_number}: row is not a JSON object"
                    )
                rows.append((line_number, value))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: manifest is not UTF-8") from exc
    return rows


def _require_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_rows: int,
    sample_ids_sha256: str,
) -> None:
    expected = {
        "protocol": SYNCG_MANIFEST_PROTOCOL,
        "dataset": "SyncG",
        "split": "train",
        "expected_rows": expected_rows,
        "emitted_rows": expected_rows,
        "release_identity_verified": True,
        "strict_release": True,
        "expected_sample_ids_sha256": sample_ids_sha256,
        "sample_ids_sha256": sample_ids_sha256,
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"manifest protocol {key}={protocol.get(key)!r}; "
                f"expected {expected_value!r}"
            )


def _casefold_path_key(path: Path) -> str:
    return path.as_posix().casefold()


def _validate_manifest_rows(
    rows: Sequence[tuple[int, dict[str, Any]]],
    *,
    project_root: Path,
    image_root: Path,
    annotation_root: Path,
) -> list[_RowSpec]:
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    annotation_paths: set[str] = set()
    specs: list[_RowSpec] = []
    for line_number, row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not _SAMPLE_ID_RE.fullmatch(sample_id):
            raise ValueError(
                f"manifest line {line_number}: invalid official sample_id {sample_id!r}"
            )
        if sample_id in sample_ids:
            raise ValueError(f"manifest line {line_number}: duplicate sample_id {sample_id}")
        sample_ids.add(sample_id)
        if row.get("dataset") != "SyncG":
            raise ValueError(f"{sample_id}: dataset must be exactly 'SyncG'")
        if row.get("split") != "train":
            raise ValueError(f"{sample_id}: split must be exactly 'train'")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{sample_id}: metadata must be a JSON object")

        image_path, image_project_path = _resolve_train_file(
            row.get("image_path"),
            project_root=project_root,
            train_root=image_root,
            role=f"{sample_id} image",
        )
        annotation_path, annotation_project_path = _resolve_train_file(
            metadata.get("annotation_path"),
            project_root=project_root,
            train_root=annotation_root,
            role=f"{sample_id} annotation",
        )
        image_key = _casefold_path_key(image_path)
        annotation_key = _casefold_path_key(annotation_path)
        if image_key in image_paths:
            raise ValueError(f"{sample_id}: duplicate image path")
        if annotation_key in annotation_paths:
            raise ValueError(f"{sample_id}: duplicate annotation path")
        image_paths.add(image_key)
        annotation_paths.add(annotation_key)
        if image_path.suffix.casefold() not in _IMAGE_SUFFIXES:
            raise ValueError(f"{sample_id}: unsupported image suffix {image_path.suffix!r}")
        if annotation_path.suffix.casefold() != ".json":
            raise ValueError(f"{sample_id}: annotation must be a .json file")
        if image_path.stem != sample_id:
            raise ValueError(f"{sample_id}: image stem does not match sample_id")
        if annotation_path.stem != sample_id:
            raise ValueError(f"{sample_id}: annotation stem does not match sample_id")
        specs.append(
            _RowSpec(
                line_number=line_number,
                sample_id=sample_id,
                row=row,
                image_path=image_path,
                annotation_path=annotation_path,
                image_project_path=image_project_path,
                annotation_project_path=annotation_project_path,
            )
        )
    return specs


def _raise_if_tree_inventory_differs(
    *,
    actual: set[Path],
    expected: set[Path],
    root: Path,
    role: str,
) -> None:
    missing = sorted(
        (_relative_to(path, root, role=role).as_posix() for path in expected - actual)
    )
    extra = sorted(
        (_relative_to(path, root, role=role).as_posix() for path in actual - expected)
    )
    if missing or extra:
        raise ValueError(
            f"{role} train tree inventory mismatch: "
            f"missing={missing[:10]!r}, extra={extra[:10]!r}, "
            f"missing_count={len(missing)}, extra_count={len(extra)}"
        )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_ino),
    )


def _hash_stream_with_size(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after) or total != int(after.st_size):
        raise RuntimeError(f"{path}: file changed while hashing")
    return total, digest.hexdigest()


def _read_bytes_with_hash(path: Path) -> tuple[bytes, str]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after) or len(payload) != int(after.st_size):
        raise RuntimeError(f"{path}: file changed while reading")
    return payload, hashlib.sha256(payload).hexdigest()


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _semantic_equal(left: Any, right: Any) -> bool:
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        left_value = float(left)
        right_value = float(right)
        return (
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and math.isclose(
                left_value,
                right_value,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _semantic_equal(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes, bytearray))
    ):
        return len(left) == len(right) and all(
            _semantic_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _require_semantic_equal(
    actual: Any,
    expected: Any,
    *,
    sample_id: str,
    relation: str,
) -> None:
    if not _semantic_equal(actual, expected):
        raise ValueError(
            f"{sample_id}: annotation/manifest semantic mismatch for {relation}"
        )


def _validate_annotation_semantics(
    spec: _RowSpec,
    annotation: Mapping[str, Any],
) -> None:
    row = spec.row
    metadata = row["metadata"]
    sample_id = spec.sample_id
    match = _SAMPLE_ID_RE.fullmatch(sample_id)
    assert match is not None

    relation_fields = (
        ("file_name", annotation.get("file_name"), sample_id),
        ("seed", annotation.get("seed"), int(match.group(1))),
        ("ground_truth", annotation.get("ground_truth"), row.get("ground_truth")),
        ("gauge_type", annotation.get("gauge_type"), metadata.get("gauge_type")),
        ("meter_id", annotation.get("gauge_type"), row.get("meter_id")),
        ("scene_name", annotation.get("scene_name"), metadata.get("scene_name")),
        ("start_value", annotation.get("start_value"), row.get("scale_start")),
        ("long_num", annotation.get("long_num"), metadata.get("long_num")),
        (
            "long_interval_value",
            annotation.get("long_interval_value"),
            metadata.get("long_interval_value"),
        ),
        (
            "long_interval_degree",
            annotation.get("long_interval_degree"),
            metadata.get("long_interval_degree"),
        ),
        (
            "pointer_rotate_degree",
            annotation.get("pointer_rotate_degree"),
            metadata.get("pointer_angle"),
        ),
        (
            "dial_bbox_annotations",
            annotation.get("dial_bbox_annotations"),
            metadata.get("dial_bbox"),
        ),
        ("homo_matrix", annotation.get("homo_matrix"), metadata.get("homography")),
        (
            "keypoints_annotations",
            annotation.get("keypoints_annotations"),
            metadata.get("keypoints"),
        ),
    )
    for relation, actual, expected in relation_fields:
        if actual is None or expected is None:
            raise ValueError(f"{sample_id}: missing semantic relation {relation}")
        _require_semantic_equal(
            actual,
            expected,
            sample_id=sample_id,
            relation=relation,
        )

    gauge_type = annotation.get("gauge_type")
    scene_name = annotation.get("scene_name")
    if not isinstance(gauge_type, str) or not gauge_type:
        raise ValueError(f"{sample_id}: annotation gauge_type is invalid")
    if not isinstance(scene_name, str) or not scene_name:
        raise ValueError(f"{sample_id}: annotation scene_name is invalid")
    expected_group = f"{gauge_type}::{Path(scene_name).stem}"
    if row.get("group_id") != expected_group:
        raise ValueError(
            f"{sample_id}: group_id does not match annotation gauge/scene semantics"
        )

    scale_start = _number(annotation.get("start_value"), label=f"{sample_id} start")
    long_num = _number(annotation.get("long_num"), label=f"{sample_id} long_num")
    interval = _number(
        annotation.get("long_interval_value"),
        label=f"{sample_id} long_interval_value",
    )
    expected_scale_end = scale_start + (long_num - 1.0) * interval
    actual_scale_end = _number(row.get("scale_end"), label=f"{sample_id} scale_end")
    if not math.isclose(
        actual_scale_end,
        expected_scale_end,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{sample_id}: manifest scale_end contradicts raw annotation semantics"
        )


def _materialize_record(spec: _RowSpec) -> dict[str, Any]:
    image_size, image_sha256 = _hash_stream_with_size(spec.image_path)
    annotation_bytes, annotation_sha256 = _read_bytes_with_hash(spec.annotation_path)
    annotation = _strict_json_loads(
        annotation_bytes,
        label=f"{spec.sample_id} annotation",
    )
    if not isinstance(annotation, dict):
        raise ValueError(f"{spec.sample_id}: annotation is not a JSON object")
    _validate_annotation_semantics(spec, annotation)
    return {
        "sample_id": spec.sample_id,
        "image_path": spec.image_project_path,
        "image_size_bytes": image_size,
        "image_sha256": image_sha256,
        "annotation_path": spec.annotation_project_path,
        "annotation_size_bytes": len(annotation_bytes),
        "annotation_sha256": annotation_sha256,
    }


def _validate_formal_identity(
    *,
    expected_rows: int,
    manifest_sha256: str,
    manifest_protocol_sha256: str,
    sample_ids_sha256: str,
) -> None:
    if expected_rows != SYNCG_TRAIN_EXPECTED_ROWS:
        raise ValueError(
            f"formal inventory requires {SYNCG_TRAIN_EXPECTED_ROWS} rows, "
            f"got {expected_rows}"
        )
    expected = {
        "manifest SHA-256": (manifest_sha256, PINNED_MANIFEST_SHA256),
        "manifest protocol SHA-256": (
            manifest_protocol_sha256,
            PINNED_MANIFEST_PROTOCOL_SHA256,
        ),
        "sample IDs SHA-256": (sample_ids_sha256, PINNED_SAMPLE_IDS_SHA256),
    }
    for label, (actual, pinned) in expected.items():
        if actual != pinned:
            raise ValueError(f"formal {label} mismatch: {actual} != {pinned}")


def build_inventory(
    *,
    project_root: Path,
    manifest: Path,
    manifest_protocol: Path | None = None,
    vdn_protocol_document: Path,
    vdn_protocol_source: Path,
    inventory_tool_source: Path | None = None,
    expected_rows: int = SYNCG_TRAIN_EXPECTED_ROWS,
    formal_identity: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Build a deterministic report after freshly validating all train content."""

    project_root = _resolve_project_root(project_root)
    manifest, manifest_project_path = _resolve_project_file(
        manifest,
        project_root=project_root,
        role="SyncG train manifest",
    )
    if manifest_protocol is None:
        manifest_protocol = manifest.with_name(manifest.name + ".protocol.json")
    manifest_protocol, manifest_protocol_project_path = _resolve_project_file(
        manifest_protocol,
        project_root=project_root,
        role="SyncG train manifest protocol",
    )
    vdn_protocol_document, vdn_protocol_document_project_path = (
        _resolve_project_file(
            vdn_protocol_document,
            project_root=project_root,
            role="VDN Phase-2 protocol document",
        )
    )
    vdn_protocol_source, vdn_protocol_source_project_path = _resolve_project_file(
        vdn_protocol_source,
        project_root=project_root,
        role="VDN Phase-2 protocol source",
    )
    if inventory_tool_source is None:
        inventory_tool_source = Path(__file__)
    inventory_tool_source, inventory_tool_source_project_path = _resolve_project_file(
        inventory_tool_source,
        project_root=project_root,
        role="inventory tool source",
    )

    manifest_sha256 = sha256_file_raw(manifest)
    manifest_protocol_sha256 = sha256_file_raw(manifest_protocol)
    protocol_value = _strict_json_loads(
        manifest_protocol.read_bytes(),
        label=str(manifest_protocol),
    )
    if not isinstance(protocol_value, dict):
        raise ValueError("manifest protocol is not a JSON object")
    syncg_root, syncg_root_project_path = _resolve_declared_root(
        protocol_value.get("syncg_root"),
        project_root=project_root,
    )
    image_root = syncg_root / "images" / "train"
    annotation_root = syncg_root / "annotations" / "train"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(
            "exact SyncG train roots are missing: "
            f"{image_root}, {annotation_root}"
        )

    rows = _read_manifest(manifest)
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"SyncG train manifest has {len(rows)} rows; expected {expected_rows}"
        )
    raw_sample_ids = [row.get("sample_id") for _, row in rows]
    if any(not isinstance(sample_id, str) for sample_id in raw_sample_ids):
        raise ValueError("manifest contains a non-string sample_id")
    sample_ids_sha256 = _sample_ids_sha256(raw_sample_ids)
    _require_protocol(
        protocol_value,
        expected_rows=int(expected_rows),
        sample_ids_sha256=sample_ids_sha256,
    )
    if formal_identity:
        _validate_formal_identity(
            expected_rows=int(expected_rows),
            manifest_sha256=manifest_sha256,
            manifest_protocol_sha256=manifest_protocol_sha256,
            sample_ids_sha256=sample_ids_sha256,
        )

    specs = _validate_manifest_rows(
        rows,
        project_root=project_root,
        image_root=image_root,
        annotation_root=annotation_root,
    )
    expected_images = {spec.image_path for spec in specs}
    expected_annotations = {spec.annotation_path for spec in specs}
    actual_images_before = _enumerate_exact_train_files(
        image_root,
        role="image",
    )
    actual_annotations_before = _enumerate_exact_train_files(
        annotation_root,
        role="annotation",
    )
    _raise_if_tree_inventory_differs(
        actual=actual_images_before,
        expected=expected_images,
        root=image_root,
        role="image",
    )
    _raise_if_tree_inventory_differs(
        actual=actual_annotations_before,
        expected=expected_annotations,
        root=annotation_root,
        role="annotation",
    )

    ordered_specs = sorted(specs, key=lambda value: value.sample_id)
    worker_count = max(1, int(workers))
    if worker_count == 1:
        records = [_materialize_record(spec) for spec in ordered_specs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(executor.map(_materialize_record, ordered_specs))

    actual_images_after = _enumerate_exact_train_files(image_root, role="image")
    actual_annotations_after = _enumerate_exact_train_files(
        annotation_root,
        role="annotation",
    )
    if (
        actual_images_after != actual_images_before
        or actual_annotations_after != actual_annotations_before
    ):
        raise RuntimeError("SyncG train tree changed while inventory was built")
    _raise_if_tree_inventory_differs(
        actual=actual_images_after,
        expected=expected_images,
        root=image_root,
        role="image",
    )
    _raise_if_tree_inventory_differs(
        actual=actual_annotations_after,
        expected=expected_annotations,
        root=annotation_root,
        role="annotation",
    )

    report: dict[str, Any] = {
        "protocol": INVENTORY_PROTOCOL,
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "status": "passed",
        "scope": "pinned_syncg_official_train_only",
        "canonicalization": CANONICALIZATION_PROTOCOL,
        "source_hash_protocol": SOURCE_HASH_PROTOCOL,
        "formal_identity_enforced": bool(formal_identity),
        "forbidden_scopes_not_accessed": [
            "syncg_test",
            "rpm",
            "pointer",
            "field",
            "sealed",
            "confirmatory",
        ],
        "manifest": {
            "path": manifest_project_path,
            "sha256": manifest_sha256,
            "protocol_path": manifest_protocol_project_path,
            "protocol_sha256": manifest_protocol_sha256,
            "rows": len(records),
            "sample_ids_sha256": sample_ids_sha256,
            "reference_huggingface_commit": protocol_value.get(
                "reference_huggingface_commit"
            ),
        },
        "vdn_phase2_protocol": {
            "document_path": vdn_protocol_document_project_path,
            "document_sha256": sha256_source_file(vdn_protocol_document),
            "source_path": vdn_protocol_source_project_path,
            "source_sha256": sha256_source_file(vdn_protocol_source),
        },
        "inventory_tool": {
            "source_path": inventory_tool_source_project_path,
            "source_sha256": sha256_source_file(inventory_tool_source),
        },
        "dataset_layout": {
            "syncg_root": syncg_root_project_path,
            "image_root": _relative_to(
                image_root.resolve(strict=True),
                project_root,
                role="image root",
            ).as_posix(),
            "annotation_root": _relative_to(
                annotation_root.resolve(strict=True),
                project_root,
                role="annotation root",
            ).as_posix(),
        },
        "counts": {
            "rows": len(records),
            "unique_sample_ids": len({record["sample_id"] for record in records}),
            "image_files": len(actual_images_after),
            "annotation_files": len(actual_annotations_after),
            "image_bytes": sum(int(record["image_size_bytes"]) for record in records),
            "annotation_bytes": sum(
                int(record["annotation_size_bytes"]) for record in records
            ),
            "extra_image_files": 0,
            "extra_annotation_files": 0,
        },
        "records": records,
        "canonical_inventory_sha256": canonical_json_sha256(records),
    }
    report["canonical_report_payload_sha256"] = canonical_json_sha256(report)
    return report


def _parse_report(path: Path) -> dict[str, Any]:
    value = _strict_json_loads(path.read_bytes(), label=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: inventory report is not a JSON object")
    if value.get("protocol") != INVENTORY_PROTOCOL:
        raise ValueError(f"{path}: inventory protocol mismatch")
    if value.get("schema_version") != INVENTORY_SCHEMA_VERSION:
        raise ValueError(f"{path}: inventory schema version mismatch")
    if value.get("status") != "passed":
        raise ValueError(f"{path}: inventory status is not passed")
    stored_payload_digest = value.get("canonical_report_payload_sha256")
    if not isinstance(stored_payload_digest, str):
        raise ValueError(f"{path}: canonical report payload digest is missing")
    payload = dict(value)
    del payload["canonical_report_payload_sha256"]
    actual_payload_digest = canonical_json_sha256(payload)
    if actual_payload_digest != stored_payload_digest:
        raise ValueError(f"{path}: canonical report payload digest mismatch")
    records = value.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records inventory is invalid")
    if canonical_json_sha256(records) != value.get("canonical_inventory_sha256"):
        raise ValueError(f"{path}: canonical inventory digest mismatch")
    return value


def verify_inventory_artifact(
    report_path: Path,
    *,
    project_root: Path,
    manifest: Path,
    manifest_protocol: Path | None = None,
    vdn_protocol_document: Path,
    vdn_protocol_source: Path,
    inventory_tool_source: Path | None = None,
    expected_rows: int = SYNCG_TRAIN_EXPECTED_ROWS,
    formal_identity: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    """Freshly rehash every train file and require exact report equality."""

    project_root = _resolve_project_root(project_root)
    report_path, report_project_path = _resolve_project_file(
        report_path,
        project_root=project_root,
        role="SyncG train inventory report",
    )
    stored = _parse_report(report_path)
    fresh = build_inventory(
        project_root=project_root,
        manifest=manifest,
        manifest_protocol=manifest_protocol,
        vdn_protocol_document=vdn_protocol_document,
        vdn_protocol_source=vdn_protocol_source,
        inventory_tool_source=inventory_tool_source,
        expected_rows=expected_rows,
        formal_identity=formal_identity,
        workers=workers,
    )
    if stored != fresh:
        raise ValueError(
            "stored SyncG train inventory differs from fresh content identity"
        )
    return {
        "protocol": INVENTORY_VERIFICATION_PROTOCOL,
        "verified": True,
        "content_rehashed": True,
        "report_path": report_project_path,
        "inventory_report_sha256": sha256_file_raw(report_path),
        "canonical_inventory_sha256": stored["canonical_inventory_sha256"],
        "canonical_report_payload_sha256": stored[
            "canonical_report_payload_sha256"
        ],
        "manifest_sha256": stored["manifest"]["sha256"],
        "manifest_protocol_sha256": stored["manifest"]["protocol_sha256"],
        "vdn_protocol_document_sha256": stored["vdn_phase2_protocol"][
            "document_sha256"
        ],
        "vdn_protocol_source_sha256": stored["vdn_phase2_protocol"][
            "source_sha256"
        ],
        "rows": stored["counts"]["rows"],
    }


def write_inventory_no_clobber(report: Mapping[str, Any], output: Path) -> Path:
    """Atomically publish a deterministic JSON report without overwriting."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, output)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite inventory: {output}") from exc
        return output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_cli_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("freeze", "verify-existing"),
    )
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-protocol", type=Path)
    parser.add_argument(
        "--vdn-protocol-document",
        type=Path,
        default=DEFAULT_VDN_PROTOCOL_DOCUMENT,
    )
    parser.add_argument(
        "--vdn-protocol-source",
        type=Path,
        default=DEFAULT_VDN_PROTOCOL_SOURCE,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, os.cpu_count() or 1)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve(strict=True)
    manifest = _resolve_cli_path(args.manifest, project_root)
    manifest_protocol = (
        _resolve_cli_path(args.manifest_protocol, project_root)
        if args.manifest_protocol is not None
        else None
    )
    vdn_protocol_document = _resolve_cli_path(
        args.vdn_protocol_document,
        project_root,
    )
    vdn_protocol_source = _resolve_cli_path(
        args.vdn_protocol_source,
        project_root,
    )
    output = _resolve_cli_path(args.output, project_root)
    if args.mode == "freeze":
        report = build_inventory(
            project_root=project_root,
            manifest=manifest,
            manifest_protocol=manifest_protocol,
            vdn_protocol_document=vdn_protocol_document,
            vdn_protocol_source=vdn_protocol_source,
            expected_rows=SYNCG_TRAIN_EXPECTED_ROWS,
            formal_identity=True,
            workers=args.workers,
        )
        write_inventory_no_clobber(report, output)
        result = {
            "status": "written",
            "path": _relative_to(
                output.resolve(strict=True),
                project_root,
                role="inventory output",
            ).as_posix(),
            "sha256": sha256_file_raw(output),
            "canonical_inventory_sha256": report[
                "canonical_inventory_sha256"
            ],
            "canonical_report_payload_sha256": report[
                "canonical_report_payload_sha256"
            ],
        }
    else:
        result = verify_inventory_artifact(
            output,
            project_root=project_root,
            manifest=manifest,
            manifest_protocol=manifest_protocol,
            vdn_protocol_document=vdn_protocol_document,
            vdn_protocol_source=vdn_protocol_source,
            expected_rows=SYNCG_TRAIN_EXPECTED_ROWS,
            formal_identity=True,
            workers=args.workers,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
