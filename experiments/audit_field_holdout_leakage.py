"""Audit field manifests for exact leakage before any model is evaluated.

This utility is deliberately prediction- and reading-label-independent.  It
uses only manifest identity fields, decoded-pixel hashes, perceptual hashes,
raw ``data/img_*.png`` files, and embedded ``xl/media`` images from the legacy
Xiangmu1 workbook.  Perceptual-hash matches are review candidates only; they
never fail the audit by themselves.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from experiments.prepare_field_holdout_xlsx import (
    _perceptual_hash_64,
    _pixel_sha256,
    sha256_file,
)


AUDIT_PROTOCOL = "prediction_independent_field_holdout_leakage_audit_v1"
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX_16_RE = re.compile(r"^[0-9a-f]{16}$")


def _issue(
    code: str,
    message: str,
    **context: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        **{
            key: value
            for key, value in sorted(context.items())
            if value is not None
        },
    }


def _normalized_path(path_value: str, manifest: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return os.path.normcase(str(path.resolve(strict=False)))


def _read_manifest(
    path: Path,
    label: str,
    *,
    expected_split: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    path = path.resolve()
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    if not path.is_file():
        return [], [
            _issue(
                "manifest_missing",
                "required manifest does not exist",
                manifest=label,
                path=str(path),
            )
        ]

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                issues.append(
                    _issue(
                        "manifest_invalid_json",
                        "manifest line is not valid JSON",
                        manifest=label,
                        line=line_number,
                        detail=str(error),
                    )
                )
                continue
            if not isinstance(value, dict):
                issues.append(
                    _issue(
                        "manifest_row_not_object",
                        "manifest line is not a JSON object",
                        manifest=label,
                        line=line_number,
                    )
                )
                continue

            sample_id = str(value.get("sample_id") or "").strip()
            group_id = str(value.get("group_id") or "").strip()
            image_path = str(value.get("image_path") or "").strip()
            split = str(value.get("split") or "").strip()
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                issues.append(
                    _issue(
                        "metadata_missing",
                        "manifest row lacks a metadata object",
                        manifest=label,
                        line=line_number,
                        sample_id=sample_id or None,
                    )
                )
            pixel_hash = str(
                metadata.get("decoded_pixel_sha256") or ""
            ).strip().lower()
            phash = str(
                metadata.get("perceptual_hash_phash64") or ""
            ).strip().lower()

            required = {
                "sample_id": sample_id,
                "group_id": group_id,
                "image_path": image_path,
            }
            for field, field_value in required.items():
                if not field_value:
                    issues.append(
                        _issue(
                            "required_identity_missing",
                            f"manifest row lacks {field}",
                            manifest=label,
                            line=line_number,
                            sample_id=sample_id or None,
                            field=field,
                        )
                    )
            if not HEX_64_RE.fullmatch(pixel_hash):
                issues.append(
                    _issue(
                        "decoded_pixel_sha256_invalid",
                        "decoded_pixel_sha256 must be 64 lowercase hex digits",
                        manifest=label,
                        line=line_number,
                        sample_id=sample_id or None,
                    )
                )
            if not HEX_16_RE.fullmatch(phash):
                issues.append(
                    _issue(
                        "perceptual_hash_phash64_invalid",
                        "perceptual_hash_phash64 must be 16 lowercase hex digits",
                        manifest=label,
                        line=line_number,
                        sample_id=sample_id or None,
                    )
                )
            if expected_split is not None and split != expected_split:
                issues.append(
                    _issue(
                        "unexpected_split",
                        "manifest row has an unexpected split",
                        manifest=label,
                        line=line_number,
                        sample_id=sample_id or None,
                        expected=expected_split,
                        observed=split,
                    )
                )

            rows.append(
                {
                    "manifest": label,
                    "line": str(line_number),
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "image_path": image_path,
                    "normalized_image_path": (
                        _normalized_path(image_path, path)
                        if image_path
                        else ""
                    ),
                    "decoded_pixel_sha256": pixel_hash,
                    "perceptual_hash_phash64": phash,
                    "split": split,
                }
            )

    duplicate_ids = sorted(
        sample_id
        for sample_id, count in _counts(
            row["sample_id"] for row in rows if row["sample_id"]
        ).items()
        if count > 1
    )
    if duplicate_ids:
        issues.append(
            _issue(
                "duplicate_sample_ids_within_manifest",
                "sample_id must be unique within each manifest",
                manifest=label,
                sample_ids=duplicate_ids,
            )
        )
    return rows, issues


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[value] += 1
    return dict(result)


def _index_unique(
    rows: list[dict[str, str]],
    key: str,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row[key]
        if value and value not in result:
            result[value] = row
    return result


def _sample_summary(row: dict[str, str]) -> dict[str, str]:
    return {
        "sample_id": row["sample_id"],
        "group_id": row["group_id"],
        "image_path": row["image_path"],
    }


def _pair_summary(
    left: dict[str, str],
    right: dict[str, str],
    *,
    distance: int | None = None,
    left_name: str = "development",
    right_name: str = "confirmatory",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        left_name: _sample_summary(left),
        right_name: _sample_summary(right),
    }
    if distance is not None:
        result["phash_hamming_distance"] = distance
    return result


def _exact_intersection(
    left: list[dict[str, str]],
    right: list[dict[str, str]],
    key: str,
) -> dict[str, Any]:
    left_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    right_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in left:
        if row[key]:
            left_index[row[key]].append(row)
    for row in right:
        if row[key]:
            right_index[row[key]].append(row)
    shared = sorted(set(left_index) & set(right_index))
    matches = []
    for value in shared:
        for left_row in sorted(
            left_index[value], key=lambda item: item["sample_id"]
        ):
            for right_row in sorted(
                right_index[value], key=lambda item: item["sample_id"]
            ):
                matches.append(
                    {
                        key: value,
                        **_pair_summary(left_row, right_row),
                    }
                )
    return {"shared_values": len(shared), "matches": matches}


def phash_hamming_distance(left: str, right: str) -> int:
    if not HEX_16_RE.fullmatch(left) or not HEX_16_RE.fullmatch(right):
        raise ValueError("pHash values must be 16 lowercase hex digits")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _cross_set_phash(
    development: list[dict[str, str]],
    confirmatory: list[dict[str, str]],
) -> dict[str, Any]:
    valid_development = [
        row
        for row in development
        if HEX_16_RE.fullmatch(row["perceptual_hash_phash64"])
    ]
    valid_confirmatory = [
        row
        for row in confirmatory
        if HEX_16_RE.fullmatch(row["perceptual_hash_phash64"])
    ]
    all_pairs: list[tuple[int, dict[str, str], dict[str, str]]] = []
    nearest: list[dict[str, Any]] = []
    for development_row in sorted(
        valid_development, key=lambda item: item["sample_id"]
    ):
        distances = [
            (
                phash_hamming_distance(
                    development_row["perceptual_hash_phash64"],
                    confirmatory_row["perceptual_hash_phash64"],
                ),
                confirmatory_row,
            )
            for confirmatory_row in valid_confirmatory
        ]
        if distances:
            distance, confirmatory_row = min(
                distances,
                key=lambda item: (item[0], item[1]["sample_id"]),
            )
            nearest.append(
                _pair_summary(
                    development_row,
                    confirmatory_row,
                    distance=distance,
                )
            )
            all_pairs.extend(
                (pair_distance, development_row, paired_confirmatory)
                for pair_distance, paired_confirmatory in distances
            )

    ordered = sorted(
        all_pairs,
        key=lambda item: (
            item[0],
            item[1]["sample_id"],
            item[2]["sample_id"],
        ),
    )
    minimum = ordered[0][0] if ordered else None
    global_minimum_pairs = [
        _pair_summary(left, right, distance=distance)
        for distance, left, right in ordered
        if distance == minimum
    ]

    def threshold_pairs(threshold: int) -> list[dict[str, Any]]:
        return [
            _pair_summary(left, right, distance=distance)
            for distance, left, right in ordered
            if distance <= threshold
        ]

    return {
        "metric": "64-bit pHash Hamming distance",
        "interpretation": (
            "Candidates only. A low pHash distance is not proof of leakage "
            "and never changes the process exit code."
        ),
        "minimum_hamming_distance": minimum,
        "global_minimum_pairs": global_minimum_pairs,
        "nearest_confirmatory_for_each_development_sample": nearest,
        "candidates_le_2": threshold_pairs(2),
        "candidates_le_4": threshold_pairs(4),
    }


def _decode_historical_image(
    payload: bytes,
    source_id: str,
) -> dict[str, str]:
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None or image.size == 0:
        raise ValueError(f"cannot decode historical image: {source_id}")
    return {
        "source_id": source_id,
        "decoded_pixel_sha256": _pixel_sha256(image),
        "perceptual_hash_phash64": _perceptual_hash_64(image),
    }


def _scan_historical_sources(
    data_root: Path,
    xiangmu1_workbook: Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    data_root = data_root.resolve()
    xiangmu1_workbook = xiangmu1_workbook.resolve()
    records: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    if not data_root.is_dir():
        errors.append(
            _issue(
                "historical_data_root_missing",
                "historical data root does not exist or is not a directory",
                path=str(data_root),
            )
        )
    else:
        raw_paths = sorted(
            (
                path
                for path in data_root.rglob("*")
                if path.is_file()
                and path.name.lower().startswith("img_")
                and path.suffix.lower() == ".png"
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    for path in raw_paths:
        source_id = f"data:{path.relative_to(data_root).as_posix()}"
        try:
            records.append(_decode_historical_image(path.read_bytes(), source_id))
        except (OSError, ValueError) as error:
            errors.append(
                _issue(
                    "historical_image_decode_failed",
                    "historical data image could not be decoded",
                    source_id=source_id,
                    detail=str(error),
                )
            )

    media_members: list[str] = []
    if not xiangmu1_workbook.is_file():
        errors.append(
            _issue(
                "xiangmu1_workbook_missing",
                "legacy Xiangmu1 workbook does not exist",
                path=str(xiangmu1_workbook),
            )
        )
    else:
        try:
            with zipfile.ZipFile(xiangmu1_workbook) as archive:
                media_members = sorted(
                    member
                    for member in archive.namelist()
                    if member.casefold().startswith("xl/media/")
                    and not member.endswith("/")
                )
                for member in media_members:
                    source_id = (
                        f"xlsx:{xiangmu1_workbook.name}!/{member}"
                    )
                    try:
                        records.append(
                            _decode_historical_image(
                                archive.read(member),
                                source_id,
                            )
                        )
                    except (KeyError, OSError, ValueError) as error:
                        errors.append(
                            _issue(
                                "xlsx_media_decode_failed",
                                "Xiangmu1 xl/media member could not be decoded",
                                source_id=source_id,
                                detail=str(error),
                            )
                        )
        except (OSError, zipfile.BadZipFile) as error:
            errors.append(
                _issue(
                    "xiangmu1_workbook_unreadable",
                    "legacy Xiangmu1 workbook is not a readable XLSX package",
                    path=str(xiangmu1_workbook),
                    detail=str(error),
                )
            )

    records.sort(key=lambda item: item["source_id"])
    summary = {
        "data_root": str(data_root),
        "recursive_img_png_files": len(raw_paths),
        "xiangmu1_workbook": str(xiangmu1_workbook),
        "xiangmu1_workbook_sha256": (
            sha256_file(xiangmu1_workbook)
            if xiangmu1_workbook.is_file()
            else None
        ),
        "xiangmu1_xl_media_members": len(media_members),
        "decoded_historical_images": len(records),
        "scan_errors": len(errors),
    }
    return records, errors, summary


def _holdout_history_matches(
    holdout: list[dict[str, str]],
    history: list[dict[str, str]],
) -> dict[str, Any]:
    history_by_pixel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in history:
        history_by_pixel[item["decoded_pixel_sha256"]].append(item)

    exact: list[dict[str, Any]] = []
    near: list[tuple[int, dict[str, str], dict[str, str]]] = []
    for row in sorted(holdout, key=lambda item: item["sample_id"]):
        pixel_hash = row["decoded_pixel_sha256"]
        if HEX_64_RE.fullmatch(pixel_hash):
            for historical in history_by_pixel.get(pixel_hash, []):
                exact.append(
                    {
                        "decoded_pixel_sha256": pixel_hash,
                        "holdout": _sample_summary(row),
                        "historical_source_id": historical["source_id"],
                    }
                )
        phash = row["perceptual_hash_phash64"]
        if not HEX_16_RE.fullmatch(phash):
            continue
        for historical in history:
            distance = phash_hamming_distance(
                phash,
                historical["perceptual_hash_phash64"],
            )
            # A pHash distance of zero does not imply identical decoded
            # pixels.  Keep it as a review candidate unless the exact pixel
            # hash already records the same image above.
            if (
                historical["decoded_pixel_sha256"] != pixel_hash
                and distance <= 4
            ):
                near.append((distance, row, historical))

    exact.sort(
        key=lambda item: (
            item["holdout"]["sample_id"],
            item["historical_source_id"],
        )
    )
    near.sort(
        key=lambda item: (
            item[0],
            item[1]["sample_id"],
            item[2]["source_id"],
        )
    )

    def threshold_pairs(threshold: int) -> list[dict[str, Any]]:
        return [
            {
                "phash_hamming_distance": distance,
                "holdout": _sample_summary(row),
                "historical_source_id": historical["source_id"],
            }
            for distance, row, historical in near
            if distance <= threshold
        ]

    return {
        "exact_decoded_pixel_matches": exact,
        "near_duplicate_interpretation": (
            "Candidates only. Distances 0-4 with different decoded pixels "
            "require image-level human review and never change the process "
            "exit code."
        ),
        "phash_candidates_le_2": threshold_pairs(2),
        "phash_candidates_le_4": threshold_pairs(4),
    }


def _cross_group_pixel_hashes(
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pixel_hash = row["decoded_pixel_sha256"]
        if HEX_64_RE.fullmatch(pixel_hash):
            by_hash[pixel_hash].append(row)
    violations = []
    for pixel_hash, matching_rows in by_hash.items():
        groups = sorted(
            {row["group_id"] for row in matching_rows if row["group_id"]}
        )
        if len(groups) <= 1:
            continue
        violations.append(
            {
                "decoded_pixel_sha256": pixel_hash,
                "physical_group_ids": groups,
                "samples": [
                    _sample_summary(row)
                    for row in sorted(
                        matching_rows, key=lambda item: item["sample_id"]
                    )
                ],
            }
        )
    return sorted(
        violations, key=lambda item: item["decoded_pixel_sha256"]
    )


def _partition_consistency_issues(
    combined: list[dict[str, str]],
    development: list[dict[str, str]],
    confirmatory: list[dict[str, str]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    combined_index = _index_unique(combined, "sample_id")
    development_index = _index_unique(development, "sample_id")
    confirmatory_index = _index_unique(confirmatory, "sample_id")
    partition_ids = set(development_index) | set(confirmatory_index)
    combined_ids = set(combined_index)
    if combined_ids != partition_ids:
        issues.append(
            _issue(
                "combined_partition_sample_set_mismatch",
                "combined sample IDs must equal the development/confirmatory union",
                missing_from_partitions=sorted(combined_ids - partition_ids),
                absent_from_combined=sorted(partition_ids - combined_ids),
            )
        )
    shared_groups = sorted(
        {row["group_id"] for row in development if row["group_id"]}
        & {row["group_id"] for row in confirmatory if row["group_id"]}
    )
    if shared_groups:
        issues.append(
            _issue(
                "development_confirmatory_group_overlap",
                "physical groups must be disjoint across field partitions",
                physical_group_ids=shared_groups,
            )
        )

    identity_fields = (
        "group_id",
        "normalized_image_path",
        "decoded_pixel_sha256",
        "perceptual_hash_phash64",
    )
    for partition_name, partition_index in (
        ("development", development_index),
        ("confirmatory", confirmatory_index),
    ):
        for sample_id in sorted(set(partition_index) & combined_ids):
            differences = {
                field: {
                    "combined": combined_index[sample_id][field],
                    partition_name: partition_index[sample_id][field],
                }
                for field in identity_fields
                if combined_index[sample_id][field]
                != partition_index[sample_id][field]
            }
            if differences:
                issues.append(
                    _issue(
                        "partition_identity_drift",
                        "partition identity fields differ from combined manifest",
                        partition=partition_name,
                        sample_id=sample_id,
                        differences=differences,
                    )
                )
    return issues


def audit_field_holdout_leakage(
    combined_manifest: Path,
    development_manifest: Path,
    confirmatory_manifest: Path,
    data_root: Path,
    xiangmu1_workbook: Path,
) -> dict[str, Any]:
    """Return a deterministic, label- and prediction-independent audit."""

    combined_manifest = combined_manifest.resolve()
    development_manifest = development_manifest.resolve()
    confirmatory_manifest = confirmatory_manifest.resolve()
    manifests = (
        ("combined", combined_manifest, None),
        ("development", development_manifest, "field_development"),
        ("confirmatory", confirmatory_manifest, "field_confirmatory"),
    )
    rows: dict[str, list[dict[str, str]]] = {}
    protocol_issues: list[dict[str, Any]] = []
    input_summary: dict[str, Any] = {}
    for label, path, expected_split in manifests:
        manifest_rows, issues = _read_manifest(
            path,
            label,
            expected_split=expected_split,
        )
        rows[label] = manifest_rows
        protocol_issues.extend(issues)
        input_summary[label] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "rows": len(manifest_rows),
            "physical_groups": len(
                {row["group_id"] for row in manifest_rows if row["group_id"]}
            ),
        }

    protocol_issues.extend(
        _partition_consistency_issues(
            rows["combined"],
            rows["development"],
            rows["confirmatory"],
        )
    )
    sample_intersection = _exact_intersection(
        rows["development"], rows["confirmatory"], "sample_id"
    )
    path_intersection = _exact_intersection(
        rows["development"],
        rows["confirmatory"],
        "normalized_image_path",
    )
    pixel_intersection = _exact_intersection(
        rows["development"],
        rows["confirmatory"],
        "decoded_pixel_sha256",
    )
    cross_group_hashes = _cross_group_pixel_hashes(rows["combined"])
    phash_cross_set = _cross_set_phash(
        rows["development"], rows["confirmatory"]
    )

    history, history_errors, history_summary = _scan_historical_sources(
        data_root,
        xiangmu1_workbook,
    )
    protocol_issues.extend(history_errors)
    history_matches = _holdout_history_matches(rows["combined"], history)

    fatal_findings: list[dict[str, Any]] = []
    if protocol_issues:
        fatal_findings.append(
            _issue(
                "protocol_inconsistency",
                "manifest or historical-source protocol checks failed",
                count=len(protocol_issues),
            )
        )
    for name, result in (
        ("sample_id", sample_intersection),
        ("normalized_image_path", path_intersection),
        ("decoded_pixel_sha256", pixel_intersection),
    ):
        if result["shared_values"]:
            fatal_findings.append(
                _issue(
                    "development_confirmatory_exact_intersection",
                    "development and confirmatory manifests share exact identity",
                    identity=name,
                    shared_values=result["shared_values"],
                )
            )
    if cross_group_hashes:
        fatal_findings.append(
            _issue(
                "decoded_pixel_hash_crosses_physical_groups",
                "an exact decoded image occurs in multiple physical groups",
                count=len(cross_group_hashes),
            )
        )
    exact_history = history_matches["exact_decoded_pixel_matches"]
    if exact_history:
        fatal_findings.append(
            _issue(
                "holdout_exactly_matches_historical_data",
                "holdout decoded pixels exactly match historical data",
                count=len(exact_history),
            )
        )

    return {
        "schema_version": 1,
        "protocol": AUDIT_PROTOCOL,
        "passed": not fatal_findings,
        "audit_scope": {
            "uses_model_predictions": False,
            "reads_prediction_files": False,
            "uses_ground_truth_readings_for_selection": False,
            "uses_only_identity_and_image_hash_metadata": True,
            "phash_candidates_are_fatal": False,
        },
        "inputs": input_summary,
        "protocol_consistency": {
            "passed": not protocol_issues,
            "issues": sorted(
                protocol_issues,
                key=lambda item: (
                    item["code"],
                    str(item.get("manifest", "")),
                    int(item.get("line", 0)),
                    str(item.get("sample_id", "")),
                    str(item.get("source_id", "")),
                ),
            ),
        },
        "development_confirmatory_exact_intersections": {
            "sample_id": sample_intersection,
            "normalized_image_path": path_intersection,
            "decoded_pixel_sha256": pixel_intersection,
        },
        "decoded_pixel_hash_cross_physical_groups": {
            "count": len(cross_group_hashes),
            "violations": cross_group_hashes,
        },
        "development_confirmatory_phash": phash_cross_set,
        "historical_scan": history_summary,
        "holdout_vs_historical": history_matches,
        "fatal_findings": fatal_findings,
    }


def _atomic_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--confirmatory-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--xiangmu1-workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_field_holdout_leakage(
        args.combined_manifest,
        args.development_manifest,
        args.confirmatory_manifest,
        args.data_root,
        args.xiangmu1_workbook,
    )
    _atomic_json(args.output, audit, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "passed": audit["passed"],
                "fatal_findings": audit["fatal_findings"],
                "phash_candidates_le_2": len(
                    audit["development_confirmatory_phash"][
                        "candidates_le_2"
                    ]
                ),
                "phash_candidates_le_4": len(
                    audit["development_confirmatory_phash"][
                        "candidates_le_4"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["passed"] else 1


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
