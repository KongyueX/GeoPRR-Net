"""Materialize the sole physical A10 Fold-B manifest after training.

This command reads the existing fit-only SyncG manifest and its scene-disjoint
split description.  It copies, byte-for-byte at the JSONL-line level, only the
rows whose scene is in the fixed 14-scene A8 Fold-B roster.  It has one output
and no Fold-A, formal-holdout, field-photo, model, or prediction input.

The command is intentionally separate from training and evaluation.  Importing
this module performs no filesystem access, and the output is created with
exclusive-create semantics.
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
from experiments.a10_pccot_data import (
    DEFAULT_FOLD_B_MANIFEST,
    validate_fold_b_samples,
)
from experiments.prepare_a8_physical_manifests import (
    DEFAULT_FIT_MANIFEST,
    read_fit_split_identities,
)
from experiments.resnet18_direct_progress import _direct_sample, _sample_identity
from experiments.sgca_syncg_internal_pilot import (
    FIT_SAMPLES,
    FIT_SCENES,
    INTERNAL_DEV_SCENE_STEMS,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    DEFAULT_SYNCG_SCENE_SPLIT,
)


PROTOCOL: Final[str] = "a10_physical_fold_b_manifest_after_training_v1"
FOLD_B_SAMPLES: Final[int] = 1_508
FOLD_B_SCENES: Final[int] = 14


class A10FoldBManifestPreparationError(ValueError):
    """The fit-only source, fixed scene partition, or sole output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A10FoldBManifestPreparationError(message)


def _selected_fold_b_lines(
    fit_manifest_path: Path,
    split_identity: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    source = Path(fit_manifest_path).resolve()
    _require(source.is_file(), f"fit-only manifest does not exist: {source}")
    fit_scenes = set(str(value) for value in split_identity["train_scene_stems"])
    validation_scenes = set(
        str(value) for value in split_identity["validation_scene_stems"]
    )
    fold_a_scenes = set(A8_FOLD_A_SCENE_STEMS)
    fold_b_scenes = set(A8_FOLD_B_SCENE_STEMS)
    old_dev_scenes = set(INTERNAL_DEV_SCENE_STEMS)
    core_scenes = fit_scenes - fold_a_scenes - fold_b_scenes - old_dev_scenes
    _require(
        len(fit_scenes) == FIT_SCENES
        and len(core_scenes) == 89
        and len(fold_a_scenes) == FOLD_B_SCENES
        and len(fold_b_scenes) == FOLD_B_SCENES
        and fold_a_scenes <= fit_scenes
        and fold_b_scenes <= fit_scenes
        and old_dev_scenes <= fit_scenes,
        "fixed A10 scene partition cannot be realized from fit scenes",
    )
    _require(
        fit_scenes.isdisjoint(validation_scenes),
        "fit scenes overlap formal validation scene identities",
    )
    _require(
        fold_a_scenes.isdisjoint(fold_b_scenes)
        and fold_a_scenes.isdisjoint(old_dev_scenes)
        and fold_b_scenes.isdisjoint(old_dev_scenes),
        "fixed A10 internal scene groups overlap",
    )

    selected_lines: list[str] = []
    selected_samples = []
    sample_ids: list[str] = []
    observed_scenes: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise A10FoldBManifestPreparationError(
                    f"fit-only manifest line {line_number} is invalid JSON"
                ) from exc
            _require(
                isinstance(row, Mapping),
                f"fit-only manifest line {line_number} is not an object",
            )
            identity = _sample_identity(row)
            sample_ids.append(identity.sample_id)
            observed_scenes.add(identity.scene_stem)
            if identity.scene_stem not in fold_b_scenes:
                continue
            image_path = row.get("image_path")
            _require(
                isinstance(image_path, str)
                and bool(image_path)
                and Path(image_path).is_absolute(),
                f"{identity.sample_id}: Fold-B image_path must remain absolute",
            )
            selected_samples.append(_direct_sample(row, manifest_root=source.parent))
            selected_lines.append(line)

    expected_ids = tuple(str(value) for value in split_identity["train_sample_ids"])
    _require(len(sample_ids) == FIT_SAMPLES, "fit-only source row count differs")
    _require(
        tuple(sample_ids) == expected_ids,
        "fit-only manifest sample ID order differs from split fit IDs",
    )
    _require(
        observed_scenes == fit_scenes,
        "fit-only manifest scenes differ from split fit scenes",
    )
    validated = validate_fold_b_samples(selected_samples)
    _require(
        len(selected_lines) == len(validated) == FOLD_B_SAMPLES,
        "physical Fold-B row count differs",
    )
    return selected_lines, {
        "source_rows": len(sample_ids),
        "source_scenes": len(observed_scenes),
        "fold_b_samples": len(validated),
        "fold_b_scene_stems": tuple(
            sorted({sample.scene_stem for sample in validated})
        ),
        "fold_b_sample_ids": tuple(sample.sample_id for sample in validated),
    }


def prepare_a10_fold_b_manifest(
    *,
    fit_manifest_path: Path,
    split_path: Path,
    fold_b_output_path: Path,
) -> dict[str, Any]:
    """Create exactly one 1,508-row/14-scene Fold-B JSONL output."""

    source = Path(fit_manifest_path).resolve()
    split = Path(split_path).resolve()
    output = Path(fold_b_output_path).resolve()
    _require(
        len({source, split, output}) == 3,
        "fit source, split, and Fold-B output paths must differ",
    )
    _require(not output.exists(), f"Fold-B output already exists: {output}")
    split_identity = read_fit_split_identities(split)
    selected_lines, metadata = _selected_fold_b_lines(source, split_identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.writelines(selected_lines)
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "source_fit_manifest": str(source),
        "source_split": str(split),
        "split_protocol": split_identity["protocol"],
        "fold_b_manifest": str(output),
        "fold_b_samples": metadata["fold_b_samples"],
        "fold_b_scenes": len(metadata["fold_b_scene_stems"]),
        "fold_b_scene_stems": list(metadata["fold_b_scene_stems"]),
        "fold_b_sample_ids": list(metadata["fold_b_sample_ids"]),
        "fold_a_manifest_generated": False,
        "formal_manifest_generated": False,
        "field_manifest_generated": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SYNCG_SCENE_SPLIT)
    parser.add_argument(
        "--fold-b-output", type=Path, default=DEFAULT_FOLD_B_MANIFEST
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = prepare_a10_fold_b_manifest(
        fit_manifest_path=args.fit_manifest,
        split_path=args.split,
        fold_b_output_path=args.fold_b_output,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A10FoldBManifestPreparationError",
    "FOLD_B_SAMPLES",
    "FOLD_B_SCENES",
    "PROTOCOL",
    "build_argument_parser",
    "prepare_a10_fold_b_manifest",
]
