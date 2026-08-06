"""Build a sealed numeric-OCR corpus from GARC ``algorithm_fit`` only.

This replaces the original whole-SyncG OCR split, whose inner training groups
overlap the frozen GARC outer evaluation groups.  The builder authenticates all
four label-free GARC rosters, admits exactly the 551 ``algorithm_fit`` groups,
and creates a deterministic inner train/calibration/validation group split.

Only public SyncG/train annotations belonging to admitted algorithm-fit sample
IDs are opened.  No image pixels, outer annotations, numeric outer labels,
field/test/sealed/confirmatory data, or model checkpoint is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    PARTITIONS as GARC_PARTITIONS,
    canonical_bytes,
    canonical_sha256,
    load_frozen_protocol,
    load_partition_roster,
    require,
    sha256_file,
    strict_json,
    strict_jsonl,
    verify_bound_file,
)
from experiments.syncg_numeric_ocr import (
    DEFAULT_CHARACTERS,
    PROTOCOL as OCR_PROTOCOL,
    canonical_roi_bounds,
    grouped_three_way_split,
    normalize_bbox_to_roi,
    normalize_numeric_text,
)


ALIGNMENT_PROTOCOL: Final[str] = "syncg_public_numeric_ocr_garc_aligned_v1"
INNER_PARTITIONS: Final[tuple[str, ...]] = ("train", "calibration", "validation")
OUTER_PARTITIONS: Final[tuple[str, ...]] = (
    "calibration",
    "development_excluded",
    "independent_validation",
)
SAFE_OUTPUT_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
DEFAULT_GARC_PROTOCOL: Final[Path] = Path(
    r"C:\pointer_read\automatic_numeric_range_public_protocol_20260806_v1\protocol.json"
)
DEFAULT_OUTPUT: Final[Path] = SAFE_OUTPUT_ROOT / "syncg_numeric_ocr_garc_aligned_v1"
PUBLIC_IMAGE_ROOT: Final[Path] = (
    PROJECT_ROOT / "datasets/SyncG/syncG/images/train"
).resolve()
PUBLIC_ANNOTATION_ROOT: Final[Path] = (
    PROJECT_ROOT / "datasets/SyncG/syncG/annotations/train"
).resolve()
EXPECTED_ALGORITHM_FIT: Final[tuple[int, int]] = (12_176, 551)
EXPECTED_OUTER: Final[dict[str, tuple[int, int]]] = {
    "calibration": (2_224, 100),
    "development_excluded": (520, 24),
    "independent_validation": (1_080, 50),
}


def _require_under(path: Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(Path(root).resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes allowed public root: {resolved}") from error
    return resolved


def _relative_project(path: Path) -> str:
    return Path(path).resolve(strict=True).relative_to(PROJECT_ROOT.resolve(strict=True)).as_posix()


def _identity(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {str(row["sample_id"]) for row in rows},
        {str(row["group_id"]) for row in rows},
    )


def assert_outer_exclusion(
    algorithm_fit: Sequence[Mapping[str, Any]],
    outer: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Fail closed if any admitted sample or physical group enters an outer split."""

    fit_samples, fit_groups = _identity(algorithm_fit)
    require(bool(fit_samples and fit_groups), "algorithm_fit roster is empty")
    result: dict[str, Any] = {}
    seen_outer_samples: set[str] = set()
    seen_outer_groups: set[str] = set()
    for partition in OUTER_PARTITIONS:
        rows = list(outer.get(partition) or [])
        require(bool(rows), f"missing outer roster: {partition}")
        samples, groups = _identity(rows)
        sample_overlap = fit_samples & samples
        group_overlap = fit_groups & groups
        require(not sample_overlap, f"algorithm_fit/{partition} sample overlap")
        require(not group_overlap, f"algorithm_fit/{partition} group overlap")
        require(
            not (seen_outer_samples & samples),
            f"outer sample overlap involving {partition}",
        )
        require(
            not (seen_outer_groups & groups),
            f"outer group overlap involving {partition}",
        )
        seen_outer_samples.update(samples)
        seen_outer_groups.update(groups)
        result[partition] = {
            "samples": len(samples),
            "groups": len(groups),
            "sample_overlap": 0,
            "group_overlap": 0,
            "sample_ids_sha256": canonical_sha256(sorted(samples)),
            "group_ids_sha256": canonical_sha256(sorted(groups)),
        }
    return result


def assign_inner_partitions(
    algorithm_fit: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    assignment = grouped_three_way_split(
        algorithm_fit,
        seed=int(seed),
        calibration_fraction=float(calibration_fraction),
        validation_fraction=float(validation_fraction),
    )
    require(set(assignment) == {str(row["sample_id"]) for row in algorithm_fit}, "inner assignment incomplete")
    return assignment


def _load_inputs(
    garc_protocol: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    Path,
    Path,
    list[dict[str, Any]],
    dict[str, Any],
]:
    protocol_path, protocol = load_frozen_protocol(garc_protocol)
    rosters: dict[str, list[dict[str, Any]]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for partition in GARC_PARTITIONS:
        _, manifest_path, rows, audit = load_partition_roster(protocol_path, partition)
        rosters[partition] = rows
        bindings[partition] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "samples": audit["samples"],
            "groups": audit["groups"],
            "sample_ids_sha256": audit["sample_ids_sha256"],
            "group_ids_sha256": audit["group_ids_sha256"],
        }
    require(
        (len(rosters["algorithm_fit"]), len(_identity(rosters["algorithm_fit"])[1]))
        == EXPECTED_ALGORITHM_FIT,
        "GARC algorithm_fit inventory drift",
    )
    for partition, expected in EXPECTED_OUTER.items():
        require(
            (len(rosters[partition]), len(_identity(rosters[partition])[1])) == expected,
            f"GARC {partition} inventory drift",
        )
    assert_outer_exclusion(
        rosters["algorithm_fit"],
        {partition: rosters[partition] for partition in OUTER_PARTITIONS},
    )

    manifest_path = verify_bound_file(protocol, "source_bindings", "syncg_train_manifest")
    manifest_protocol_path = verify_bound_file(
        protocol, "source_bindings", "syncg_train_manifest_protocol"
    )
    manifest_protocol = strict_json(manifest_protocol_path)
    require(manifest_protocol.get("dataset") == "SyncG", "source manifest dataset drift")
    require(manifest_protocol.get("split") == "train", "source manifest split drift")
    require(manifest_protocol.get("strict_release") is True, "source manifest is not strict")
    require(
        manifest_protocol.get("release_identity_verified") is True,
        "source release identity is unverified",
    )
    manifest_rows = strict_jsonl(manifest_path)
    require(len(manifest_rows) == 16_000, "source SyncG/train inventory drift")
    require(
        len({str(row["sample_id"]) for row in manifest_rows}) == len(manifest_rows),
        "duplicate source sample id",
    )
    require(
        len({str(row["group_id"]) for row in manifest_rows}) == 725,
        "source physical-group inventory drift",
    )
    return (
        protocol_path,
        protocol,
        rosters,
        bindings,
        manifest_path,
        manifest_protocol_path,
        manifest_rows,
        manifest_protocol,
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(canonical_bytes(dict(value), pretty=True))
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("xb") as handle:
        for row in rows:
            handle.write(canonical_bytes(dict(row)))
        handle.flush()
        os.fsync(handle.fileno())


def _partition_stats(
    sample_rows: Sequence[Mapping[str, Any]], token_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups_by_partition: dict[str, set[str]] = {}
    for partition in INNER_PARTITIONS:
        samples = [row for row in sample_rows if row["partition"] == partition]
        tokens = [row for row in token_rows if row["partition"] == partition]
        groups = {str(row["group_id"]) for row in samples}
        groups_by_partition[partition] = groups
        result[partition] = {
            "samples": len(samples),
            "groups": len(groups),
            "tokens": len(tokens),
            "sample_ids_sha256": canonical_sha256(
                sorted(str(row["sample_id"]) for row in samples)
            ),
            "group_ids_sha256": canonical_sha256(sorted(groups)),
            "token_ids_sha256": canonical_sha256(
                sorted(str(row["token_id"]) for row in tokens)
            ),
        }
    require(
        not groups_by_partition["train"]
        & (groups_by_partition["calibration"] | groups_by_partition["validation"])
        and not groups_by_partition["calibration"] & groups_by_partition["validation"],
        "inner OCR split has physical-group leakage",
    )
    return result


def build_corpus(
    *,
    garc_protocol: Path,
    output_dir: Path,
    split_seed: int,
    calibration_fraction: float,
    validation_fraction: float,
) -> Path:
    output = Path(output_dir).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"aligned OCR output must stay below {SAFE_OUTPUT_ROOT}") from error
    require(output != SAFE_OUTPUT_ROOT, "refusing broad output root")
    require(not output.exists(), f"refusing to overwrite aligned OCR corpus: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    require(not staging.exists(), f"aligned OCR staging path exists: {staging}")

    (
        protocol_path,
        protocol,
        rosters,
        roster_bindings,
        manifest_path,
        manifest_protocol_path,
        manifest_rows,
        manifest_protocol,
    ) = _load_inputs(garc_protocol)
    algorithm_fit = rosters["algorithm_fit"]
    outer_audit = assert_outer_exclusion(
        algorithm_fit,
        {partition: rosters[partition] for partition in OUTER_PARTITIONS},
    )
    assignment = assign_inner_partitions(
        algorithm_fit,
        seed=split_seed,
        calibration_fraction=calibration_fraction,
        validation_fraction=validation_fraction,
    )
    source_by_id = {str(row["sample_id"]): row for row in manifest_rows}
    admitted_ids, admitted_groups = _identity(algorithm_fit)
    require(admitted_ids <= set(source_by_id), "algorithm_fit sample absent from source manifest")

    sample_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    annotation_identities: list[dict[str, Any]] = []
    character_frequency: Counter[str] = Counter()
    canonical_heights: list[float] = []
    canonical_widths: list[float] = []
    fit_by_id = {str(row["sample_id"]): row for row in algorithm_fit}
    for sample_id in sorted(admitted_ids):
        roster_row = fit_by_id[sample_id]
        manifest_row = source_by_id[sample_id]
        group_id = str(roster_row["group_id"])
        require(str(manifest_row["group_id"]) == group_id, f"{sample_id}: group drift")
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
        raw = annotation_path.read_bytes()
        annotation = json.loads(raw.decode("utf-8"))
        require(isinstance(annotation, Mapping), f"{sample_id}: annotation is not an object")
        require(str(annotation.get("file_name")) == sample_id, f"{sample_id}: annotation id drift")
        width, height = int(annotation["width"]), int(annotation["height"])
        require(width >= 2 and height >= 2, f"{sample_id}: invalid dimensions")
        dial_bbox = [float(value) for value in annotation["dial_bbox_annotations"][:4]]
        manifest_bbox = [float(value) for value in metadata["dial_bbox"][:4]]
        roster_bbox = [float(value) for value in roster_row["dial_bbox"][:4]]
        require(
            np.allclose(dial_bbox, manifest_bbox, atol=1e-6)
            and np.allclose(dial_bbox, roster_bbox, atol=1e-6),
            f"{sample_id}: dial bbox drift",
        )
        roi_bounds = canonical_roi_bounds((height, width, 3), dial_bbox)
        sample_tokens: list[dict[str, Any]] = []
        for source_index, entry in enumerate(annotation.get("text_bbox_annotations") or []):
            if not isinstance(entry, Mapping) or str(entry.get("type") or "").casefold() != "text":
                continue
            text = normalize_numeric_text(entry.get("value"))
            bbox = [float(value) for value in entry["bbox"][:4]]
            require(
                len(bbox) == 4
                and np.isfinite(bbox).all()
                and bbox[2] > bbox[0]
                and bbox[3] > bbox[1],
                f"{sample_id}: invalid text bbox",
            )
            normalized_bbox = normalize_bbox_to_roi(bbox, roi_bounds)
            token_id = f"{sample_id}:text:{source_index}"
            token = {
                "token_id": token_id,
                "sample_id": sample_id,
                "group_id": group_id,
                "partition": assignment[sample_id],
                "image_path": _relative_project(image_path),
                "annotation_path": _relative_project(annotation_path),
                "text": text,
                "numeric_value": float(text),
                "bbox_original": bbox,
                "bbox_roi_normalized": list(normalized_bbox),
            }
            token_rows.append(token)
            sample_tokens.append(
                {
                    "token_id": token_id,
                    "text": text,
                    "bbox_roi_normalized": list(normalized_bbox),
                }
            )
            character_frequency.update(text)
            canonical_widths.append((normalized_bbox[2] - normalized_bbox[0]) * 768.0)
            canonical_heights.append((normalized_bbox[3] - normalized_bbox[1]) * 768.0)
        require(bool(sample_tokens), f"{sample_id}: no public numeric Text annotations")
        sample_rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "partition": assignment[sample_id],
                "image_path": _relative_project(image_path),
                "annotation_path": _relative_project(annotation_path),
                "image_width": width,
                "image_height": height,
                "dial_bbox": dial_bbox,
                "tokens": sample_tokens,
            }
        )
        annotation_identities.append(
            {
                "sample_id": sample_id,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    require(len(sample_rows) == EXPECTED_ALGORITHM_FIT[0], "emitted sample inventory drift")
    require(
        {str(row["sample_id"]) for row in sample_rows} == admitted_ids,
        "emitted samples do not exactly cover algorithm_fit",
    )
    require(
        {str(row["group_id"]) for row in sample_rows} == admitted_groups,
        "emitted groups do not exactly cover algorithm_fit",
    )
    partition_stats = _partition_stats(sample_rows, token_rows)
    negative = sum(str(row["text"]).startswith("-") for row in token_rows)
    decimal = sum("." in str(row["text"]) for row in token_rows)

    staging.mkdir(parents=True, exist_ok=False)
    samples_path = staging / "samples.jsonl"
    tokens_path = staging / "tokens.jsonl"
    _write_jsonl(samples_path, sample_rows)
    _write_jsonl(tokens_path, token_rows)
    summary = {
        "schema_version": 2,
        "protocol": OCR_PROTOCOL,
        "alignment_protocol": ALIGNMENT_PROTOCOL,
        "status": "complete",
        "scope": {
            "dataset": "SyncG",
            "split": "train",
            "admission": "frozen GARC algorithm_fit partition only",
            "source": "public algorithm_fit annotations only; image pixels unopened while building",
            "caller_supplied_boxes_at_inference": False,
            "numeric_prediction_space": "variable-length signed real strings",
            "integer_endpoint_classification": False,
        },
        "parent_garc_protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "identity": protocol["protocol"],
            "status": protocol["status"],
        },
        "garc_partition_bindings": roster_bindings,
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_protocol": str(manifest_protocol_path),
            "manifest_protocol_sha256": sha256_file(manifest_protocol_path),
            "reference_huggingface_commit": manifest_protocol.get(
                "reference_huggingface_commit"
            ),
            "annotation_content_inventory_sha256": canonical_sha256(annotation_identities),
        },
        "split": {
            "scope": "inner split within GARC algorithm_fit groups only",
            "method": "two-stage SHA256 physical-group ordering",
            "seed": int(split_seed),
            "calibration_fraction_target": float(calibration_fraction),
            "validation_fraction_target": float(validation_fraction),
            "partitions": partition_stats,
        },
        "inventory": {
            "samples": len(sample_rows),
            "groups": len(admitted_groups),
            "tokens": len(token_rows),
            "negative_tokens": negative,
            "decimal_tokens_in_public_annotations": decimal,
            "character_frequency": dict(sorted(character_frequency.items())),
            "vocabulary": list(DEFAULT_CHARACTERS),
            "bbox_at_768": {
                "height_p05": float(np.percentile(canonical_heights, 5)),
                "height_median": float(np.median(canonical_heights)),
                "height_p95": float(np.percentile(canonical_heights, 95)),
                "width_p05": float(np.percentile(canonical_widths, 5)),
                "width_median": float(np.median(canonical_widths)),
                "width_p95": float(np.percentile(canonical_widths, 95)),
            },
        },
        "alignment_audit": {
            "algorithm_fit_exact_coverage": {
                "samples": len(admitted_ids),
                "groups": len(admitted_groups),
                "sample_ids_sha256": canonical_sha256(sorted(admitted_ids)),
                "group_ids_sha256": canonical_sha256(sorted(admitted_groups)),
                "manifest_sha256": roster_bindings["algorithm_fit"]["sha256"],
            },
            "outer_exclusion": outer_audit,
            "all_outer_group_overlap_zero": all(
                value["group_overlap"] == 0 for value in outer_audit.values()
            ),
            "all_outer_sample_overlap_zero": all(
                value["sample_overlap"] == 0 for value in outer_audit.values()
            ),
            "public_algorithm_fit_annotations_opened": len(annotation_identities),
            "public_images_opened": 0,
            "outer_annotations_opened": 0,
            "outer_images_opened": 0,
            "outer_numeric_values_opened": 0,
            "restricted_namespace_images_opened": 0,
        },
        "training_contract": {
            "detector": "whole canonical ROI word-occupancy detector",
            "recognizer": "variable-length CTC, not endpoint classes",
            "selection_partition": "inner calibration drawn only from algorithm_fit groups",
            "independent_component_validation": "inner validation drawn only from algorithm_fit groups",
            "outer_garc_partitions_permitted_for_training_or_selection": False,
            "decimal_support": (
                "character vocabulary includes '.', and train-only deterministic dot synthesis "
                "is permitted; inner calibration/validation labels remain untouched"
            ),
        },
        "artifacts": {
            "samples": samples_path.name,
            "samples_sha256": sha256_file(samples_path),
            "tokens": tokens_path.name,
            "tokens_sha256": sha256_file(tokens_path),
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }
    summary_path = staging / "summary.json"
    _write_json(summary_path, summary)
    seal = {
        "schema_version": 2,
        "protocol": OCR_PROTOCOL,
        "alignment_protocol": ALIGNMENT_PROTOCOL,
        "status": "sealed",
        "summary_sha256": sha256_file(summary_path),
        "samples_sha256": summary["artifacts"]["samples_sha256"],
        "tokens_sha256": summary["artifacts"]["tokens_sha256"],
        "split_sha256": canonical_sha256(summary["split"]),
        "parent_garc_protocol_sha256": sha256_file(protocol_path),
        "algorithm_fit_manifest_sha256": roster_bindings["algorithm_fit"]["sha256"],
        "algorithm_fit_group_ids_sha256": roster_bindings["algorithm_fit"][
            "group_ids_sha256"
        ],
        "outer_exclusion_sha256": canonical_sha256(outer_audit),
    }
    _write_json(staging / "seal.json", seal)
    os.replace(staging, output)
    verify_corpus(output)
    return output / "summary.json"


def verify_local_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    corpus_root = Path(root).resolve(strict=True)
    summary_path = corpus_root / "summary.json"
    samples_path = corpus_root / "samples.jsonl"
    tokens_path = corpus_root / "tokens.jsonl"
    seal_path = corpus_root / "seal.json"
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == OCR_PROTOCOL, "aligned OCR protocol drift")
    require(summary.get("alignment_protocol") == ALIGNMENT_PROTOCOL, "alignment identity drift")
    require(summary.get("status") == "complete", "aligned OCR corpus incomplete")
    require(seal.get("protocol") == OCR_PROTOCOL, "aligned OCR seal protocol drift")
    require(seal.get("alignment_protocol") == ALIGNMENT_PROTOCOL, "aligned OCR seal identity drift")
    require(seal.get("status") == "sealed", "aligned OCR corpus not sealed")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "summary seal drift")
    require(
        seal.get("samples_sha256")
        == summary.get("artifacts", {}).get("samples_sha256")
        == sha256_file(samples_path),
        "sample artifact hash drift",
    )
    require(
        seal.get("tokens_sha256")
        == summary.get("artifacts", {}).get("tokens_sha256")
        == sha256_file(tokens_path),
        "token artifact hash drift",
    )
    require(seal.get("split_sha256") == canonical_sha256(summary["split"]), "split seal drift")
    require(
        seal.get("parent_garc_protocol_sha256")
        == summary["parent_garc_protocol"]["sha256"],
        "parent protocol seal drift",
    )
    require(
        seal.get("algorithm_fit_manifest_sha256")
        == summary["garc_partition_bindings"]["algorithm_fit"]["sha256"],
        "algorithm_fit manifest seal drift",
    )
    require(
        seal.get("algorithm_fit_group_ids_sha256")
        == summary["garc_partition_bindings"]["algorithm_fit"]["group_ids_sha256"],
        "algorithm_fit group roster seal drift",
    )
    samples = strict_jsonl(samples_path)
    tokens = strict_jsonl(tokens_path)
    require(len(samples) == int(summary["inventory"]["samples"]), "sample count drift")
    require(len(tokens) == int(summary["inventory"]["tokens"]), "token count drift")
    sample_by_id: dict[str, Mapping[str, Any]] = {}
    for row in samples:
        sample_id = str(row.get("sample_id") or "")
        require(sample_id and sample_id not in sample_by_id, f"duplicate sample: {sample_id}")
        require(row.get("partition") in INNER_PARTITIONS, f"bad inner partition: {sample_id}")
        sample_by_id[sample_id] = row
    token_ids: set[str] = set()
    for row in tokens:
        token_id = str(row.get("token_id") or "")
        sample_id = str(row.get("sample_id") or "")
        require(token_id and token_id not in token_ids, f"duplicate token: {token_id}")
        require(sample_id in sample_by_id, f"orphan token: {token_id}")
        sample = sample_by_id[sample_id]
        require(row.get("group_id") == sample.get("group_id"), f"token group drift: {token_id}")
        require(row.get("partition") == sample.get("partition"), f"token split drift: {token_id}")
        token_ids.add(token_id)
    recomputed = _partition_stats(samples, tokens)
    require(recomputed == summary["split"]["partitions"], "inner partition statistics drift")
    require(
        summary["alignment_audit"]["all_outer_group_overlap_zero"] is True
        and summary["alignment_audit"]["all_outer_sample_overlap_zero"] is True,
        "aligned OCR summary lacks outer exclusion",
    )
    require(
        seal.get("outer_exclusion_sha256")
        == canonical_sha256(summary["alignment_audit"]["outer_exclusion"]),
        "outer exclusion seal drift",
    )
    return summary, seal


def verify_corpus(root: Path) -> dict[str, Any]:
    summary, seal = verify_local_artifacts(root)
    parent_path = Path(summary["parent_garc_protocol"]["path"])
    protocol_path, _ = load_frozen_protocol(parent_path)
    require(sha256_file(protocol_path) == summary["parent_garc_protocol"]["sha256"], "parent protocol drift")
    rosters: dict[str, list[dict[str, Any]]] = {}
    for partition in GARC_PARTITIONS:
        _, manifest, rows, audit = load_partition_roster(protocol_path, partition)
        binding = summary["garc_partition_bindings"][partition]
        require(str(manifest) == binding["path"], f"{partition} manifest path drift")
        require(sha256_file(manifest) == binding["sha256"], f"{partition} manifest hash drift")
        for key in ("samples", "groups", "sample_ids_sha256", "group_ids_sha256"):
            require(audit[key] == binding[key], f"{partition}.{key} drift")
        rosters[partition] = rows
    local_samples = strict_jsonl(Path(root).resolve(strict=True) / "samples.jsonl")
    local_ids, local_groups = _identity(local_samples)
    fit_ids, fit_groups = _identity(rosters["algorithm_fit"])
    require(local_ids == fit_ids, "aligned OCR samples do not exactly cover algorithm_fit")
    require(local_groups == fit_groups, "aligned OCR groups do not exactly cover algorithm_fit")
    outer_audit = assert_outer_exclusion(
        local_samples,
        {partition: rosters[partition] for partition in OUTER_PARTITIONS},
    )
    require(
        outer_audit == summary["alignment_audit"]["outer_exclusion"],
        "outer exclusion evidence drift",
    )
    return {
        "protocol": ALIGNMENT_PROTOCOL,
        "status": "verified",
        "corpus_root": str(Path(root).resolve(strict=True)),
        "summary_sha256": sha256_file(Path(root) / "summary.json"),
        "seal_sha256": sha256_file(Path(root) / "seal.json"),
        "samples": len(local_ids),
        "groups": len(local_groups),
        "algorithm_fit_exact_coverage": True,
        "outer_group_overlap_zero": True,
        "outer_sample_overlap_zero": True,
        "restricted_namespace_images_opened": 0,
        "seal": seal,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--garc-protocol", type=Path, default=DEFAULT_GARC_PROTOCOL)
    build.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--split-seed", type=int, default=20260807)
    build.add_argument("--calibration-fraction", type=float, default=0.10)
    build.add_argument("--validation-fraction", type=float, default=0.10)
    verify = commands.add_parser("verify")
    verify.add_argument("--corpus", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        result: Any = str(
            build_corpus(
                garc_protocol=args.garc_protocol,
                output_dir=args.output_dir,
                split_seed=args.split_seed,
                calibration_fraction=args.calibration_fraction,
                validation_fraction=args.validation_fraction,
            )
        )
    else:
        result = verify_corpus(args.corpus)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ALIGNMENT_PROTOCOL",
    "assert_outer_exclusion",
    "assign_inner_partitions",
    "build_corpus",
    "verify_corpus",
    "verify_local_artifacts",
]
