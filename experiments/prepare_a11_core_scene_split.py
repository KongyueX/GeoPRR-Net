"""Create-once physical Core train/audit manifests for A11 SCORT.

Only the existing 9,825-row physical Core manifest is accepted.  Records are
filtered in their original byte-for-byte text order by the explicit 72/17
scene roster in :mod:`experiments.a11_scort_protocol`.  No model output,
checkpoint, hash, other partition, formal data, or field photograph is read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TextIO

from experiments.a11_scort_protocol import (
    CORE_AUDIT_SAMPLES,
    CORE_AUDIT_SCENES,
    CORE_AUDIT_SCENE_COUNT,
    CORE_SAMPLES,
    CORE_SCENES,
    CORE_TRAIN_SAMPLES,
    CORE_TRAIN_SCENES,
    CORE_TRAIN_SCENE_COUNT,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
    summarize_fixed_core_split,
)
from experiments.resnet18_direct_progress import DirectSample, load_syncg_samples


PROTOCOL: Final[str] = "a11_physical_core_explicit_scene72_17_v1"
DEFAULT_SOURCE_CORE_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a8_physical_manifests_v1/"
    "syncg_a8_core_9825.jsonl"
)
DEFAULT_TRAIN_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a11_physical_manifests_v1/"
    "syncg_a11_core_train_7939.jsonl"
)
DEFAULT_AUDIT_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a11_physical_manifests_v1/"
    "syncg_a11_core_audit_1886.jsonl"
)


class A11ManifestError(ValueError):
    """An A11 physical manifest is malformed, out of scope, or pre-existing."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A11ManifestError(message)


def _row_scene(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), "physical Core metadata is missing")
    scene = metadata.get("scene_name")
    _require(isinstance(scene, str) and bool(scene), "physical Core scene_name is missing")
    return scene


def _row_identity(row: Mapping[str, Any]) -> str:
    sample_id = row.get("sample_id")
    _require(isinstance(sample_id, str) and bool(sample_id), "sample_id is missing")
    return sample_id


def _validate_row_target(row: Mapping[str, Any]) -> None:
    try:
        value = float(row["ground_truth"])
        start = float(row["scale_start"])
        end = float(row["scale_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise A11ManifestError("physical Core target fields are malformed") from exc
    _require(
        all(math.isfinite(number) for number in (value, start, end)) and end != start,
        "physical Core target is non-finite or has zero range",
    )
    progress = (value - start) / (end - start)
    _require(0.0 <= progress <= 1.0, "physical Core normalized target is outside [0,1]")


def _parse_core_text_lines(
    lines: Sequence[str],
    *,
    require_complete_core: bool,
) -> dict[str, Any]:
    """Parse and partition text while retaining each original JSONL line."""

    source_lines = tuple(lines)
    _require(bool(source_lines), "source physical Core manifest is empty")
    train_scenes = set(CORE_TRAIN_SCENES)
    audit_scenes = set(CORE_AUDIT_SCENES)
    rows: list[Mapping[str, Any]] = []
    train_lines: list[str] = []
    audit_lines: list[str] = []
    train_ids: list[str] = []
    audit_ids: list[str] = []
    all_ids: list[str] = []
    source_scenes: list[str] = []
    for line_number, raw_line in enumerate(source_lines, start=1):
        _require(bool(raw_line.strip()), f"source line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise A11ManifestError(f"source line {line_number} is invalid JSON") from exc
        _require(isinstance(row, Mapping), f"source line {line_number} is not an object")
        sample_id = _row_identity(row)
        scene = _row_scene(row)
        _validate_row_target(row)
        _require(
            scene in train_scenes or scene in audit_scenes,
            f"source line {line_number} scene is outside the fixed Core roster",
        )
        rows.append(row)
        all_ids.append(sample_id)
        source_scenes.append(scene)
        if scene in train_scenes:
            train_lines.append(raw_line)
            train_ids.append(sample_id)
        else:
            audit_lines.append(raw_line)
            audit_ids.append(sample_id)
    _require(len(all_ids) == len(set(all_ids)), "source sample IDs are duplicated")
    _require(
        train_ids == [
            sample_id
            for sample_id, scene in zip(all_ids, source_scenes, strict=True)
            if scene in train_scenes
        ],
        "train sample order differs from source filtering order",
    )
    _require(
        audit_ids == [
            sample_id
            for sample_id, scene in zip(all_ids, source_scenes, strict=True)
            if scene in audit_scenes
        ],
        "audit sample order differs from source filtering order",
    )
    summary = summarize_fixed_core_split(
        rows, require_complete_core=require_complete_core
    )
    if require_complete_core:
        _require(len(source_lines) == CORE_SAMPLES, "source Core count differs")
        _require(len(set(source_scenes)) == CORE_SCENES, "source Core scenes differ")
        _require(len(train_lines) == CORE_TRAIN_SAMPLES, "train row count differs")
        _require(len(audit_lines) == CORE_AUDIT_SAMPLES, "audit row count differs")
    return {
        "train_lines": tuple(train_lines),
        "audit_lines": tuple(audit_lines),
        "source_ids": tuple(all_ids),
        "train_ids": tuple(train_ids),
        "audit_ids": tuple(audit_ids),
        "summary": summary,
    }


def _write_lines(handle: TextIO, lines: Sequence[str]) -> None:
    for line in lines:
        handle.write(line)


def _write_partitions_create_once(
    *,
    train_output_path: Path,
    audit_output_path: Path,
    train_lines: Sequence[str],
    audit_lines: Sequence[str],
) -> None:
    train_output = Path(train_output_path).resolve()
    audit_output = Path(audit_output_path).resolve()
    _require(train_output != audit_output, "train and audit outputs must differ")
    _require(not train_output.exists(), f"train output already exists: {train_output}")
    _require(not audit_output.exists(), f"audit output already exists: {audit_output}")
    train_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with train_output.open("x", encoding="utf-8", newline="") as train_handle:
            created.append(train_output)
            with audit_output.open("x", encoding="utf-8", newline="") as audit_handle:
                created.append(audit_output)
                _write_lines(train_handle, train_lines)
                _write_lines(audit_handle, audit_lines)
    except BaseException:
        # Roll back only the exact resolved output files created by this call.
        for created_path in reversed(created):
            created_path.unlink(missing_ok=True)
        raise


def _read_text_lines(path: Path, *, label: str) -> tuple[str, ...]:
    source = Path(path).resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        return tuple(handle)


def prepare_a11_core_scene_split(
    *,
    source_core_manifest_path: Path,
    train_output_path: Path,
    audit_output_path: Path,
) -> dict[str, Any]:
    """Read the physical Core once and create the two fixed JSONL outputs."""

    source = Path(source_core_manifest_path).resolve()
    train_output = Path(train_output_path).resolve()
    audit_output = Path(audit_output_path).resolve()
    _require(source.is_file(), f"source physical Core does not exist: {source}")
    _require(source not in (train_output, audit_output), "source cannot be an output")
    partition = _parse_core_text_lines(
        _read_text_lines(source, label="source physical Core"),
        require_complete_core=True,
    )
    _write_partitions_create_once(
        train_output_path=train_output,
        audit_output_path=audit_output,
        train_lines=partition["train_lines"],
        audit_lines=partition["audit_lines"],
    )
    written_train = _read_text_lines(train_output, label="written train manifest")
    written_audit = _read_text_lines(audit_output, label="written audit manifest")
    _require(written_train == partition["train_lines"], "written train text/order differs")
    _require(written_audit == partition["audit_lines"], "written audit text/order differs")
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "source_core_manifest": str(source),
        "train_manifest": str(train_output),
        "audit_manifest": str(audit_output),
        "source_samples": CORE_SAMPLES,
        "train_samples": CORE_TRAIN_SAMPLES,
        "audit_samples": CORE_AUDIT_SAMPLES,
        "train_scenes": CORE_TRAIN_SCENE_COUNT,
        "audit_scenes": CORE_AUDIT_SCENE_COUNT,
        "scene_overlap_count": 0,
        "source_order_preserved_within_each_partition": True,
        "hash_used": False,
        "automatic_gate_used": False,
        "other_partition_or_model_output_access": False,
        "distribution": partition["summary"],
    }


def _validate_direct_samples(
    samples: Sequence[DirectSample],
    *,
    expected_samples: int,
    expected_scene_names: Sequence[str],
    label: str,
) -> tuple[DirectSample, ...]:
    values = tuple(samples)
    expected_stems = {Path(name).stem for name in expected_scene_names}
    ids = tuple(sample.sample_id for sample in values)
    scenes = {sample.scene_stem for sample in values}
    _require(len(values) == expected_samples, f"{label} sample count differs")
    _require(scenes == expected_stems, f"{label} scene roster differs")
    _require(len(ids) == len(set(ids)) and all(ids), f"{label} sample IDs differ")
    _require(
        all(
            math.isfinite(float(sample.normalized_target))
            and 0.0 <= float(sample.normalized_target) <= 1.0
            and 9 <= len(sample.protected_points_xy) <= 38
            and all(
                len(point) == 2
                and math.isfinite(float(point[0]))
                and math.isfinite(float(point[1]))
                for point in sample.protected_points_xy
            )
            for sample in values
        ),
        f"{label} physical target/geometry differs",
    )
    return values


def validate_a11_train_samples(samples: Sequence[DirectSample]) -> tuple[DirectSample, ...]:
    return _validate_direct_samples(
        samples,
        expected_samples=CORE_TRAIN_SAMPLES,
        expected_scene_names=CORE_TRAIN_SCENES,
        label="A11 physical Core-train manifest",
    )


def validate_a11_audit_samples(samples: Sequence[DirectSample]) -> tuple[DirectSample, ...]:
    return _validate_direct_samples(
        samples,
        expected_samples=CORE_AUDIT_SAMPLES,
        expected_scene_names=CORE_AUDIT_SCENES,
        label="A11 physical Core-audit manifest",
    )


def load_a11_train_manifest(path: Path) -> tuple[DirectSample, ...]:
    return validate_a11_train_samples(load_syncg_samples(Path(path).resolve()))


def load_a11_audit_manifest(path: Path) -> tuple[DirectSample, ...]:
    return validate_a11_audit_samples(load_syncg_samples(Path(path).resolve()))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-core-manifest", type=Path, default=DEFAULT_SOURCE_CORE_MANIFEST)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = prepare_a11_core_scene_split(
        source_core_manifest_path=args.source_core_manifest,
        train_output_path=args.train_output,
        audit_output_path=args.audit_output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A11ManifestError",
    "DEFAULT_AUDIT_MANIFEST",
    "DEFAULT_SOURCE_CORE_MANIFEST",
    "DEFAULT_TRAIN_MANIFEST",
    "PROTOCOL",
    "load_a11_audit_manifest",
    "load_a11_train_manifest",
    "prepare_a11_core_scene_split",
    "validate_a11_audit_samples",
    "validate_a11_train_samples",
]
