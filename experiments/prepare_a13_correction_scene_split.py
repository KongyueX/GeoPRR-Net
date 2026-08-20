"""Create the physical A13 correction-train/development JSONL manifests once.

The only input is the existing 7,939-row physical A11 Core-train manifest.
Rows are filtered in original text order by the predeclared 60/12 scene
roster.  No checkpoint, prediction, previous Core audit, Fold-B, formal data,
or field photograph is read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, TextIO

from experiments.a13_correction_dev_protocol import (
    CORRECTION_DEV_SAMPLES,
    CORRECTION_DEV_SCENES,
    CORRECTION_DEV_SCENE_COUNT,
    CORRECTION_TRAIN_SAMPLES,
    CORRECTION_TRAIN_SCENES,
    CORRECTION_TRAIN_SCENE_COUNT,
    DEVELOPMENT_INTERPRETATION,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
    SOURCE_SAMPLES,
    SOURCE_SCENES,
    validate_recorded_split,
)
if TYPE_CHECKING:
    from experiments.resnet18_direct_progress import DirectSample


PROTOCOL: Final[str] = "a13_physical_a11train_scene60_12_correction_dev_v1"
DEFAULT_SOURCE_A11_TRAIN_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a11_physical_manifests_v1/"
    "syncg_a11_core_train_7939.jsonl"
)
DEFAULT_CORRECTION_TRAIN_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a13_physical_manifests_v1/"
    "syncg_a13_correction_train_6616.jsonl"
)
DEFAULT_CORRECTION_DEV_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a13_physical_manifests_v1/"
    "syncg_a13_correction_dev_1323.jsonl"
)


class A13CorrectionManifestError(ValueError):
    """The source, physical partition, or create-once target is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A13CorrectionManifestError(message)


def _row_scene(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), "source A11-train metadata is missing")
    scene = metadata.get("scene_name")
    _require(isinstance(scene, str) and bool(scene), "source A11-train scene is missing")
    return scene


def _validate_row(row: Mapping[str, Any], *, line_number: int) -> str:
    sample_id = row.get("sample_id")
    _require(
        isinstance(sample_id, str) and bool(sample_id),
        f"source line {line_number} sample_id is missing",
    )
    meter = row.get("meter_id")
    _require(
        isinstance(meter, str) and bool(meter),
        f"source line {line_number} meter_id is missing",
    )
    try:
        value = float(row["ground_truth"])
        start = float(row["scale_start"])
        end = float(row["scale_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise A13CorrectionManifestError(
            f"source line {line_number} target fields are malformed"
        ) from exc
    _require(
        all(math.isfinite(number) for number in (value, start, end)) and end != start,
        f"source line {line_number} target is non-finite or has zero range",
    )
    target = (value - start) / (end - start)
    _require(
        0.0 <= target <= 1.0,
        f"source line {line_number} normalized target is outside [0,1]",
    )
    return sample_id


def _parse_source_text_lines(
    lines: Sequence[str],
    *,
    require_complete_source: bool = True,
) -> dict[str, Any]:
    """Parse source JSONL while retaining every original text line."""

    source_lines = tuple(lines)
    _require(bool(source_lines), "source A11-train manifest is empty")
    train_scene_set = set(CORRECTION_TRAIN_SCENES)
    dev_scene_set = set(CORRECTION_DEV_SCENES)
    rows: list[Mapping[str, Any]] = []
    sample_ids: list[str] = []
    train_lines: list[str] = []
    dev_lines: list[str] = []
    train_ids: list[str] = []
    dev_ids: list[str] = []
    source_scenes: list[str] = []
    for line_number, raw_line in enumerate(source_lines, start=1):
        _require(bool(raw_line.strip()), f"source line {line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise A13CorrectionManifestError(
                f"source line {line_number} is invalid JSON"
            ) from exc
        _require(
            isinstance(row, Mapping), f"source line {line_number} is not an object"
        )
        sample_id = _validate_row(row, line_number=line_number)
        scene = _row_scene(row)
        _require(
            scene in train_scene_set or scene in dev_scene_set,
            f"source line {line_number} scene is outside A11-train roster",
        )
        rows.append(row)
        sample_ids.append(sample_id)
        source_scenes.append(scene)
        if scene in train_scene_set:
            train_lines.append(raw_line)
            train_ids.append(sample_id)
        else:
            dev_lines.append(raw_line)
            dev_ids.append(sample_id)
    _require(len(sample_ids) == len(set(sample_ids)), "source sample IDs are duplicated")
    split = validate_recorded_split(rows) if require_complete_source else None
    if require_complete_source:
        _require(len(source_lines) == SOURCE_SAMPLES, "source row count differs")
        _require(len(set(source_scenes)) == SOURCE_SCENES, "source scene count differs")
        _require(
            len(train_lines) == CORRECTION_TRAIN_SAMPLES,
            "correction-train row count differs",
        )
        _require(
            len(dev_lines) == CORRECTION_DEV_SAMPLES,
            "correction-dev row count differs",
        )
    _require(
        train_ids
        == [
            sample_id
            for sample_id, scene in zip(sample_ids, source_scenes, strict=True)
            if scene in train_scene_set
        ],
        "correction-train order differs from source filtering order",
    )
    _require(
        dev_ids
        == [
            sample_id
            for sample_id, scene in zip(sample_ids, source_scenes, strict=True)
            if scene in dev_scene_set
        ],
        "correction-dev order differs from source filtering order",
    )
    return {
        "train_lines": tuple(train_lines),
        "dev_lines": tuple(dev_lines),
        "source_ids": tuple(sample_ids),
        "train_ids": tuple(train_ids),
        "dev_ids": tuple(dev_ids),
        "split": split,
    }


def _write_lines(handle: TextIO, lines: Sequence[str]) -> None:
    for line in lines:
        handle.write(line)


def _write_partitions_create_once(
    *,
    train_output_path: Path,
    dev_output_path: Path,
    train_lines: Sequence[str],
    dev_lines: Sequence[str],
) -> None:
    train_output = Path(train_output_path).resolve()
    dev_output = Path(dev_output_path).resolve()
    _require(train_output != dev_output, "train and dev output paths must differ")
    _require(not train_output.exists(), f"train output already exists: {train_output}")
    _require(not dev_output.exists(), f"dev output already exists: {dev_output}")
    train_output.parent.mkdir(parents=True, exist_ok=True)
    dev_output.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        with train_output.open("x", encoding="utf-8", newline="") as train_handle:
            created.append(train_output)
            with dev_output.open("x", encoding="utf-8", newline="") as dev_handle:
                created.append(dev_output)
                _write_lines(train_handle, train_lines)
                _write_lines(dev_handle, dev_lines)
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def _read_text_lines(path: Path, *, label: str) -> tuple[str, ...]:
    source = Path(path).resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        return tuple(handle)


def prepare_a13_correction_scene_split(
    *,
    source_a11_train_manifest_path: Path,
    correction_train_output_path: Path,
    correction_dev_output_path: Path,
) -> dict[str, Any]:
    """Create the two physical manifests after one metadata-only split audit."""

    source = Path(source_a11_train_manifest_path).resolve()
    train_output = Path(correction_train_output_path).resolve()
    dev_output = Path(correction_dev_output_path).resolve()
    _require(source.is_file(), f"source A11-train manifest does not exist: {source}")
    _require(source not in (train_output, dev_output), "source cannot be an output")
    partition = _parse_source_text_lines(
        _read_text_lines(source, label="source A11-train manifest"),
        require_complete_source=True,
    )
    _write_partitions_create_once(
        train_output_path=train_output,
        dev_output_path=dev_output,
        train_lines=partition["train_lines"],
        dev_lines=partition["dev_lines"],
    )
    written_train = _read_text_lines(train_output, label="written correction-train")
    written_dev = _read_text_lines(dev_output, label="written correction-dev")
    _require(
        written_train == partition["train_lines"],
        "written correction-train text/order differs",
    )
    _require(
        written_dev == partition["dev_lines"],
        "written correction-dev text/order differs",
    )
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "source_a11_train_manifest": str(source),
        "correction_train_manifest": str(train_output),
        "correction_dev_manifest": str(dev_output),
        "source_samples": SOURCE_SAMPLES,
        "source_scenes": SOURCE_SCENES,
        "correction_train_samples": CORRECTION_TRAIN_SAMPLES,
        "correction_train_scenes": CORRECTION_TRAIN_SCENE_COUNT,
        "correction_dev_samples": CORRECTION_DEV_SAMPLES,
        "correction_dev_scenes": CORRECTION_DEV_SCENE_COUNT,
        "scene_overlap_count": 0,
        "source_order_preserved_within_each_partition": True,
        "split_selection_trace": partition["split"]["selection_trace"],
        "distribution": {
            name: partition["split"][name] for name in ("source", "train", "dev")
        },
        "q0_has_seen_correction_dev_scenes": True,
        "system_heldout_interpretation": False,
        "development_interpretation": DEVELOPMENT_INTERPRETATION,
        "hash_used": False,
        "automatic_gate_used": False,
        "model_output_or_other_partition_access": False,
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
    _require(len(values) == expected_samples, f"{label} sample count differs")
    _require(
        {sample.scene_stem for sample in values} == expected_stems,
        f"{label} scene roster differs",
    )
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


def validate_a13_correction_train_samples(
    samples: Sequence[DirectSample],
) -> tuple[DirectSample, ...]:
    return _validate_direct_samples(
        samples,
        expected_samples=CORRECTION_TRAIN_SAMPLES,
        expected_scene_names=CORRECTION_TRAIN_SCENES,
        label="A13 physical correction-train manifest",
    )


def validate_a13_correction_dev_samples(
    samples: Sequence[DirectSample],
) -> tuple[DirectSample, ...]:
    return _validate_direct_samples(
        samples,
        expected_samples=CORRECTION_DEV_SAMPLES,
        expected_scene_names=CORRECTION_DEV_SCENES,
        label="A13 physical correction-dev manifest",
    )


def load_a13_correction_train_manifest(path: Path) -> tuple[DirectSample, ...]:
    from experiments.resnet18_direct_progress import load_syncg_samples

    return validate_a13_correction_train_samples(
        load_syncg_samples(Path(path).resolve())
    )


def load_a13_correction_dev_manifest(path: Path) -> tuple[DirectSample, ...]:
    from experiments.resnet18_direct_progress import load_syncg_samples

    return validate_a13_correction_dev_samples(load_syncg_samples(Path(path).resolve()))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-a11-train-manifest",
        type=Path,
        default=DEFAULT_SOURCE_A11_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--correction-train-output",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--correction-dev-output",
        type=Path,
        default=DEFAULT_CORRECTION_DEV_MANIFEST,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = prepare_a13_correction_scene_split(
        source_a11_train_manifest_path=args.source_a11_train_manifest,
        correction_train_output_path=args.correction_train_output,
        correction_dev_output_path=args.correction_dev_output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A13CorrectionManifestError",
    "DEFAULT_CORRECTION_DEV_MANIFEST",
    "DEFAULT_CORRECTION_TRAIN_MANIFEST",
    "DEFAULT_SOURCE_A11_TRAIN_MANIFEST",
    "PROTOCOL",
    "load_a13_correction_dev_manifest",
    "load_a13_correction_train_manifest",
    "prepare_a13_correction_scene_split",
    "validate_a13_correction_dev_samples",
    "validate_a13_correction_train_samples",
]
