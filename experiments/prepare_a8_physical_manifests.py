"""One-time physical manifest preparation for the A8 internal screen.

The command reads the existing 14,442-row fit-only SyncG manifest once and
copies each selected JSONL line unchanged into exactly two outputs: the fixed
89-scene Core training roster and fixed 14-scene Fold A evaluation roster.  It
does not create a Fold B manifest.  Training and evaluation modules never call
this preparation function automatically.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.a8_internal_crossfold_protocol import (
    A8_FOLD_A_SCENE_STEMS,
    A8_FOLD_B_SCENE_STEMS,
)
from experiments.resnet18_direct_progress import (
    DirectSample,
    _sample_identity,
    load_syncg_samples,
)
from experiments.sgca_syncg_internal_pilot import (
    FIT_SAMPLES,
    FIT_SCENES,
    INTERNAL_DEV_SCENE_STEMS,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    DEFAULT_SYNCG_SCENE_SPLIT,
)


PROTOCOL: Final[str] = "a8_physical_core_and_fold_a_manifests_v1"
DEFAULT_FIT_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/screen200_seed20262020/"
    "internal_diagnostics/syncg_formal_fit_14442.jsonl"
)
DEFAULT_CORE_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a8_physical_manifests_v1/"
    "syncg_a8_core_9825.jsonl"
)
DEFAULT_FOLD_A_MANIFEST: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/a8_physical_manifests_v1/"
    "syncg_a8_fold_a_1533.jsonl"
)
CORE_SAMPLES: Final[int] = 9_825
CORE_SCENES: Final[int] = 89
FOLD_A_SAMPLES: Final[int] = 1_533
FOLD_A_SCENES: Final[int] = 14


class A8ManifestPreparationError(ValueError):
    """The fit-only source or requested physical outputs are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A8ManifestPreparationError(message)


def read_fit_split_identities(split_path: Path) -> dict[str, Any]:
    """Interpret fit IDs/scenes and holdout scenes, not validation-sample IDs."""

    source = Path(split_path).resolve()
    _require(source.is_file(), f"SyncG split does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A8ManifestPreparationError(
            f"cannot read SyncG split identities: {source}"
        ) from exc
    _require(isinstance(value, Mapping), "SyncG split root is not an object")
    train_ids = value.get("train_sample_ids")
    train_scenes = value.get("train_scene_stems")
    validation_scenes = value.get("validation_scene_stems")
    _require(
        isinstance(train_ids, list)
        and len(train_ids) == FIT_SAMPLES
        and len(train_ids) == len(set(str(item) for item in train_ids))
        and all(isinstance(item, str) and item for item in train_ids),
        "fit sample identities are absent or duplicated",
    )
    _require(
        isinstance(train_scenes, list)
        and len(train_scenes) == FIT_SCENES
        and len(train_scenes) == len(set(str(item) for item in train_scenes))
        and all(isinstance(item, str) and item for item in train_scenes),
        "fit scene identities are absent or duplicated",
    )
    _require(
        isinstance(validation_scenes, list)
        and len(validation_scenes)
        == len(set(str(item) for item in validation_scenes))
        and all(isinstance(item, str) and item for item in validation_scenes),
        "validation scene identities are absent or duplicated",
    )
    _require(bool(value.get("scene_disjoint")), "SyncG split is not scene-disjoint")
    return {
        "protocol": str(value.get("protocol") or "unknown"),
        "train_sample_ids": tuple(str(item) for item in train_ids),
        "train_scene_stems": tuple(str(item) for item in train_scenes),
        "validation_scene_stems": tuple(str(item) for item in validation_scenes),
    }


def _read_and_partition_lines(
    fit_manifest_path: Path,
    split_identity: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    source = Path(fit_manifest_path).resolve()
    _require(source.is_file(), f"fit-only manifest does not exist: {source}")
    fit_scene_set = set(split_identity["train_scene_stems"])
    validation_scenes = set(split_identity["validation_scene_stems"])
    fold_a_scenes = set(A8_FOLD_A_SCENE_STEMS)
    fold_b_scenes = set(A8_FOLD_B_SCENE_STEMS)
    old_dev_scenes = set(INTERNAL_DEV_SCENE_STEMS)
    core_scenes = fit_scene_set - fold_a_scenes - fold_b_scenes - old_dev_scenes
    _require(
        len(core_scenes) == CORE_SCENES
        and len(fold_a_scenes) == FOLD_A_SCENES
        and fold_a_scenes <= fit_scene_set
        and fold_b_scenes <= fit_scene_set
        and old_dev_scenes <= fit_scene_set,
        "fixed A8 scene partition cannot be realized from fit scenes",
    )
    _require(
        fit_scene_set.isdisjoint(validation_scenes),
        "fit scenes overlap validation scene identities",
    )
    core_lines: list[str] = []
    fold_a_lines: list[str] = []
    sample_ids: list[str] = []
    observed_scenes: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise A8ManifestPreparationError(
                    f"fit-only manifest line {line_number} is invalid JSON"
                ) from exc
            _require(
                isinstance(row, Mapping),
                f"fit-only manifest line {line_number} is not an object",
            )
            identity = _sample_identity(row)
            sample_ids.append(identity.sample_id)
            observed_scenes.add(identity.scene_stem)
            if identity.scene_stem in core_scenes:
                core_lines.append(line)
            elif identity.scene_stem in fold_a_scenes:
                fold_a_lines.append(line)
    _require(
        tuple(sample_ids) == split_identity["train_sample_ids"],
        "fit-only manifest sample ID order differs from split fit IDs",
    )
    _require(
        observed_scenes == fit_scene_set,
        "fit-only manifest scenes differ from split fit scenes",
    )
    _require(
        len(core_lines) == CORE_SAMPLES
        and len(fold_a_lines) == FOLD_A_SAMPLES,
        "physical Core/Fold-A row counts differ",
    )
    return core_lines, fold_a_lines, {
        "core_scene_stems": tuple(sorted(core_scenes)),
        "fold_a_scene_stems": tuple(sorted(fold_a_scenes)),
        "source_rows": len(sample_ids),
    }


def prepare_a8_physical_manifests(
    *,
    fit_manifest_path: Path,
    split_path: Path,
    core_output_path: Path,
    fold_a_output_path: Path,
) -> dict[str, Any]:
    """Copy unchanged source lines into new Core-only and Fold-A-only files."""

    source = Path(fit_manifest_path).resolve()
    core_output = Path(core_output_path).resolve()
    fold_a_output = Path(fold_a_output_path).resolve()
    _require(not core_output.exists(), f"Core output already exists: {core_output}")
    _require(
        not fold_a_output.exists(),
        f"Fold-A output already exists: {fold_a_output}",
    )
    _require(
        len({source, core_output, fold_a_output}) == 3,
        "source/Core/Fold-A paths must differ",
    )
    split_identity = read_fit_split_identities(split_path)
    core_lines, fold_a_lines, partition = _read_and_partition_lines(
        source, split_identity
    )
    core_output.parent.mkdir(parents=True, exist_ok=True)
    fold_a_output.parent.mkdir(parents=True, exist_ok=True)
    with core_output.open("x", encoding="utf-8", newline="") as stream:
        stream.writelines(core_lines)
    with fold_a_output.open("x", encoding="utf-8", newline="") as stream:
        stream.writelines(fold_a_lines)
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "source_fit_manifest": str(source),
        "split_protocol": split_identity["protocol"],
        "core_manifest": str(core_output),
        "core_samples": len(core_lines),
        "core_scenes": len(partition["core_scene_stems"]),
        "fold_a_manifest": str(fold_a_output),
        "fold_a_samples": len(fold_a_lines),
        "fold_a_scenes": len(partition["fold_a_scene_stems"]),
        "fold_b_manifest_generated": False,
    }


def _load_physical_manifest(
    path: Path,
    *,
    expected_samples: int,
    expected_scenes: int,
    required_scenes: set[str] | None,
    excluded_scenes: set[str],
    label: str,
) -> tuple[DirectSample, ...]:
    samples = load_syncg_samples(path)
    scenes = {sample.scene_stem for sample in samples}
    _require(len(samples) == expected_samples, f"{label} sample count differs")
    _require(len(scenes) == expected_scenes, f"{label} scene count differs")
    _require(
        required_scenes is None or scenes == required_scenes,
        f"{label} scene roster differs",
    )
    _require(scenes.isdisjoint(excluded_scenes), f"{label} contains excluded scene")
    return tuple(samples)


def load_core_manifest(path: Path) -> tuple[DirectSample, ...]:
    return _load_physical_manifest(
        path,
        expected_samples=CORE_SAMPLES,
        expected_scenes=CORE_SCENES,
        required_scenes=None,
        excluded_scenes=(
            set(A8_FOLD_A_SCENE_STEMS)
            | set(A8_FOLD_B_SCENE_STEMS)
            | set(INTERNAL_DEV_SCENE_STEMS)
        ),
        label="A8 Core manifest",
    )


def load_fold_a_manifest(path: Path) -> tuple[DirectSample, ...]:
    return _load_physical_manifest(
        path,
        expected_samples=FOLD_A_SAMPLES,
        expected_scenes=FOLD_A_SCENES,
        required_scenes=set(A8_FOLD_A_SCENE_STEMS),
        excluded_scenes=(
            set(A8_FOLD_B_SCENE_STEMS) | set(INTERNAL_DEV_SCENE_STEMS)
        ),
        label="A8 Fold-A manifest",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SYNCG_SCENE_SPLIT)
    parser.add_argument("--core-output", type=Path, default=DEFAULT_CORE_MANIFEST)
    parser.add_argument(
        "--fold-a-output", type=Path, default=DEFAULT_FOLD_A_MANIFEST
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = prepare_a8_physical_manifests(
        fit_manifest_path=args.fit_manifest,
        split_path=args.split,
        core_output_path=args.core_output,
        fold_a_output_path=args.fold_a_output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A8ManifestPreparationError",
    "CORE_SAMPLES",
    "CORE_SCENES",
    "DEFAULT_CORE_MANIFEST",
    "DEFAULT_FIT_MANIFEST",
    "DEFAULT_FOLD_A_MANIFEST",
    "FOLD_A_SAMPLES",
    "FOLD_A_SCENES",
    "PROTOCOL",
    "build_argument_parser",
    "load_core_manifest",
    "load_fold_a_manifest",
    "prepare_a8_physical_manifests",
    "read_fit_split_identities",
]
