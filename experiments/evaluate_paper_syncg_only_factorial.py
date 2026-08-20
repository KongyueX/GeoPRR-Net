"""Evaluate the three-seed, SyncG-only 2x2 factor experiment.

The training checkpoints are expected below ``--root``::

    resnet18_direct/seed_<seed>/terminal.pt
    cbam_only/seed_<seed>/terminal.pt
    geometry_aux_only/seed_<seed>/terminal.pt
    geometry_attention/seed_<seed>/terminal.pt

This module only performs inference and scoring.  It does not expose a training,
adaptation, calibration, or model-selection entry point.  New output directories,
dataset keys, and method identities use the canonical paper names; legacy field
paths remain read-only inputs for provenance compatibility.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from experiments import domain_balanced_db_gar18_factor_ablation as factor
from experiments import geoattn_resnet18_progress as geoattn
from experiments import resnet18_direct_progress as direct
from experiments import robustness_degradations
from experiments import score_db_gar18_factorial_3seed as factorial_score
from experiments import support_aware_roi_normalization_v2 as sarn_v2
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "paper_syncg_only_factorial_prediction_v1"
DEFAULT_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1/factorial")
SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
CELLS: Final[tuple[str, ...]] = ("00", "10", "01", "11")
VARIANTS: Final[tuple[str, ...]] = ("raw", "sarn_v2")
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
SARN_NOOP_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
)
RAW_VS_SARN_CELL: Final[str] = "11"
RAW_VS_SARN_PROTOCOL: Final[str] = (
    "paper_syncg_only_raw_vs_sarn_v2_paired_comparison_v1"
)
REPOSITORY_ROOT: Final[Path] = Path(__file__).resolve().parents[1]


class PaperFactorialEvaluationError(RuntimeError):
    """Invalid source checkpoint or evaluation configuration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperFactorialEvaluationError(message)


@dataclass(frozen=True, slots=True)
class Arm:
    cell: str
    checkpoint_directory: str
    paper_label: str
    component_arm: str | None = None


ARMS: Final[dict[str, Arm]] = {
    "00": Arm("00", "resnet18_direct", "Direct-ResNet18"),
    "10": Arm(
        "10",
        "cbam_only",
        "Attention-Only",
        factor.ARM_CBAM_ONLY,
    ),
    "01": Arm(
        "01",
        "geometry_aux_only",
        "Geometry-Auxiliary-Only",
        factor.ARM_GEOMETRY_AUX_ONLY,
    ),
    "11": Arm("11", "geometry_attention", "GeoAttn-ResNet18"),
}


@dataclass(frozen=True, slots=True)
class Dataset:
    slug: str
    paper_name: str
    labels: Path
    roster: Path
    manifest: Path
    expected_samples: int
    expected_groups: int
    group_unit: str
    group_source: str


DATASETS: Final[dict[str, Dataset]] = {
    "syncg_scene_holdout": Dataset(
        slug="syncg_scene_holdout",
        paper_name="SyncG Scene-Holdout",
        labels=REPOSITORY_ROOT / "artifacts/manifests/syncg_train.jsonl",
        roster=Path("C:/pointer_read/syncg_scene_disjoint_clean_v1/split.json"),
        manifest=Path(
            "C:/pointer_read/syncg_scene_disjoint_clean_v1/plain_inputs/input_manifest.jsonl"
        ),
        expected_samples=1558,
        expected_groups=14,
        group_unit="SyncG scene",
        group_source="metadata.scene_name stem",
    ),
    "field_gauge_roi_test_a": Dataset(
        slug="field_gauge_roi_test_a",
        paper_name="FieldGauge-ROI Test-A",
        labels=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_development_v1/labels.jsonl"
        ),
        roster=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_development_v1/sample_ids.json"
        ),
        manifest=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_development_v1/input_manifest.jsonl"
        ),
        expected_samples=434,
        expected_groups=11,
        group_unit="source-directory group",
        group_source="labels.group_id",
    ),
    "field_gauge_roi_test_b": Dataset(
        slug="field_gauge_roi_test_b",
        paper_name="FieldGauge-ROI Test-B",
        labels=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_confirmatory_v1/labels.jsonl"
        ),
        roster=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_confirmatory_v1/sample_ids.json"
        ),
        manifest=Path(
            "C:/pointer_read/cagh_v5_plain_real_xm2_confirmatory_v1/input_manifest.jsonl"
        ),
        expected_samples=814,
        expected_groups=20,
        group_unit="source-directory group",
        group_source="labels.group_id",
    ),
    "field_gauge_external_roi": Dataset(
        slug="field_gauge_external_roi",
        paper_name="FieldGauge-External-ROI",
        labels=Path("C:/pointer_read/cagh_resnet_xiangmu1_final_v1/labels.jsonl"),
        roster=Path("C:/pointer_read/cagh_resnet_xiangmu1_final_v1/roster_lock.json"),
        manifest=Path(
            "C:/pointer_read/cagh_resnet_xiangmu1_final_v1/input_manifest.jsonl"
        ),
        expected_samples=147,
        expected_groups=21,
        group_unit="provisional capture session",
        group_source="labels.group_id",
    ),
    "rf100": Dataset(
        slug="rf100",
        paper_name="RF100",
        labels=Path(
            "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/labels.jsonl"
        ),
        roster=Path(
            "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/sample_ids.json"
        ),
        manifest=Path(
            "C:/pointer_read/db_gar18_rf100_external_transfer_v1/prepared/input_manifest.jsonl"
        ),
        expected_samples=151,
        expected_groups=35,
        group_unit="RF100 source/evaluation group",
        group_source="labels.group_id",
    ),
}


def checkpoint_path(root: Path, *, cell: str, seed: int) -> Path:
    arm = ARMS[cell]
    return Path(root) / arm.checkpoint_directory / f"seed_{seed}" / "terminal.pt"


def method_id(*, cell: str, seed: int, variant: str) -> str:
    base = f"{ARMS[cell].paper_label}_seed_{seed}"
    return base if variant == "raw" else f"SARN-v2+{base}"


def prediction_artifacts(
    root: Path,
    *,
    cell: str,
    seed: int,
    dataset: Dataset,
    variant: str,
) -> dict[str, Path]:
    _require(variant in VARIANTS, f"unknown prediction variant: {variant}")
    base = (
        Path(root)
        / "evaluation"
        / variant
        / dataset.slug
        / ARMS[cell].checkpoint_directory
        / f"seed_{seed}"
    )
    artifacts = {"predictions": base / "predictions.jsonl"}
    if variant == "sarn_v2":
        artifacts.update(
            {
                "sidecar": base / "normalization.jsonl",
                "summary": base / "summary.json",
            }
        )
    return artifacts


def _load_source_predictor(
    checkpoint: Path,
    *,
    cell: str,
    seed: int,
    device_name: str,
) -> tuple[str, Callable[[Sequence[np.ndarray]], list[float]]]:
    """Load one source-only checkpoint without accepting adapted checkpoints."""

    _require(cell in ARMS, f"unknown factor cell: {cell}")
    _require(seed in SEEDS, f"unexpected training seed: {seed}")
    if cell == "00":
        source_method, predictor = direct.load_checkpoint_predictor(
            checkpoint, device_name=device_name
        )
        _require(
            source_method == f"resnet18_direct_seed_{seed}",
            "Direct-ResNet18 checkpoint seed mismatch",
        )
        return method_id(cell=cell, seed=seed, variant="raw"), predictor
    if cell == "11":
        source_method, predictor = geoattn.load_checkpoint_predictor(
            checkpoint, device_name=device_name
        )
        _require(
            source_method == f"{geoattn.METHOD_PREFIX}_seed_{seed}",
            "GeoAttn-ResNet18 checkpoint seed mismatch",
        )
        return method_id(cell=cell, seed=seed, variant="raw"), predictor

    component_arm = ARMS[cell].component_arm
    _require(component_arm is not None, "component arm is missing")
    model, metadata = factor._load_source_model(checkpoint, component_arm)
    _require(int(metadata.get("seed", -1)) == seed, "component checkpoint seed mismatch")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = model.to(device).eval()

    def predict(images_bgr: Sequence[np.ndarray]) -> list[float]:
        _require(bool(images_bgr), "prediction batch is empty")
        batch = torch.stack(
            [
                normalized_rgb_tensor(
                    direct_resize_whole_roi(image, size=factor.IMAGE_SIZE)
                )
                for image in images_bgr
            ]
        ).to(device)
        with torch.inference_mode():
            values = model(batch).detach().cpu().tolist()
        output = [float(value) for value in values]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in output),
            "component model returned invalid progress",
        )
        return output

    return method_id(cell=cell, seed=seed, variant="raw"), predict


def run_raw_prediction(
    *,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    cell: str,
    seed: int,
    device_name: str,
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    selected = tuple(conditions)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(CONDITIONS),
        "invalid evaluation conditions",
    )
    rows = load_manifest(manifest)
    method, predictor = _load_source_predictor(
        checkpoint,
        cell=cell,
        seed=seed,
        device_name=device_name,
    )
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for source in rows:
            _payload, clean = load_canonical_roi(source)
            images: list[np.ndarray] = []
            condition_pixels: list[str] = []
            for condition in selected:
                image, _metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                image = np.ascontiguousarray(image)
                images.append(image)
                condition_pixels.append(canonical_roi_pixel_sha256(image))
            try:
                values: list[float | None] = [
                    float(value) for value in predictor(images)
                ]
                _require(len(values) == len(images), "prediction batch length mismatch")
                _require(
                    all(
                        value is not None
                        and math.isfinite(value)
                        and 0.0 <= value <= 1.0
                        for value in values
                    ),
                    "prediction outside [0,1]",
                )
                failures: list[str | None] = [None] * len(images)
            except Exception as exc:
                values = [None] * len(images)
                failures = [f"model_exception:{type(exc).__name__}"] * len(images)
            for condition, pixel_identity, progress, failure in zip(
                selected,
                condition_pixels,
                values,
                failures,
                strict=True,
            ):
                passed = progress is not None and failure is None
                row = {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": source.sample_id,
                    "method": method,
                    "condition": condition,
                    "robustness_seed": ROBUSTNESS_SEED,
                    "status": "pass" if passed else "fail",
                    "normalized_progress": progress if passed else None,
                    "failure_code": None if passed else failure,
                    "roi_png_sha256": source.roi_png_sha256,
                    "roi_pixel_sha256": source.roi_pixel_sha256,
                    "condition_pixel_sha256": pixel_identity,
                }
                _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
                stream.write(direct._canonical_json_bytes(row).decode("utf-8") + "\n")
                count += 1
    return {
        "status": "complete",
        "variant": "raw",
        "method": method,
        "rows": count,
        "output": str(target),
    }


def run_sarn_prediction(
    *,
    checkpoint: Path,
    manifest: Path,
    output: Path,
    sidecar: Path,
    summary: Path,
    cell: str,
    seed: int,
    device_name: str,
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    def source_loader(
        checkpoint_path: Path,
        *,
        device_name: str,
    ) -> tuple[str, str, Callable[[Sequence[np.ndarray]], list[float]]]:
        _raw_method, predictor = _load_source_predictor(
            checkpoint_path,
            cell=cell,
            seed=seed,
            device_name=device_name,
        )
        return (
            ARMS[cell].paper_label,
            method_id(cell=cell, seed=seed, variant="sarn_v2"),
            predictor,
        )

    original_loader = sarn_v2.load_sarn_v2_predictor
    sarn_v2.load_sarn_v2_predictor = source_loader
    try:
        return sarn_v2.run_prediction(
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            output_path=output,
            sidecar_path=sidecar,
            summary_path=summary,
            device_name=device_name,
            conditions=conditions,
        )
    finally:
        sarn_v2.load_sarn_v2_predictor = original_loader


def run_prediction_job(
    *,
    root: Path,
    variant: str,
    dataset: Dataset,
    cell: str,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    checkpoint = checkpoint_path(root, cell=cell, seed=seed)
    artifacts = prediction_artifacts(
        root,
        cell=cell,
        seed=seed,
        dataset=dataset,
        variant=variant,
    )
    if variant == "raw":
        return run_raw_prediction(
            checkpoint=checkpoint,
            manifest=dataset.manifest,
            output=artifacts["predictions"],
            cell=cell,
            seed=seed,
            device_name=device_name,
        )
    return run_sarn_prediction(
        checkpoint=checkpoint,
        manifest=dataset.manifest,
        output=artifacts["predictions"],
        sidecar=artifacts["sidecar"],
        summary=artifacts["summary"],
        cell=cell,
        seed=seed,
        device_name=device_name,
    )


def build_score_spec(
    *,
    root: Path,
    variant: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    _require(variant in VARIANTS, f"unknown score variant: {variant}")
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    scorer_variant = "raw" if variant == "raw" else "sarn"
    checkpoints = {
        cell: {
            str(seed): str(checkpoint_path(root, cell=cell, seed=seed))
            for seed in SEEDS
        }
        for cell in CELLS
    }
    datasets: dict[str, Any] = {}
    for dataset in DATASETS.values():
        predictions: dict[str, dict[str, dict[str, str]]] = {}
        for cell in CELLS:
            predictions[cell] = {}
            for seed in SEEDS:
                artifacts = prediction_artifacts(
                    root,
                    cell=cell,
                    seed=seed,
                    dataset=dataset,
                    variant=variant,
                )
                binding = {
                    "path": str(artifacts["predictions"]),
                    "method": method_id(cell=cell, seed=seed, variant=variant),
                    "protocol": PROTOCOL if variant == "raw" else sarn_v2.PROTOCOL,
                }
                if variant == "sarn_v2":
                    binding["sidecar"] = str(artifacts["sidecar"])
                predictions[cell][str(seed)] = binding
        datasets[dataset.paper_name] = {
            "labels": str(dataset.labels),
            "roster": str(dataset.roster),
            "manifest": str(dataset.manifest),
            "expected_samples": dataset.expected_samples,
            "expected_groups": dataset.expected_groups,
            "group_unit": dataset.group_unit,
            "group_source": dataset.group_source,
            "conditions": list(CONDITIONS),
            "key_conditions": list(CONDITIONS),
            "predictions": predictions,
        }
    return {
        "schema_version": 1,
        "protocol": factorial_score.SPEC_PROTOCOL,
        "prediction_variant": scorer_variant,
        "robustness_seed": ROBUSTNESS_SEED,
        "seeds": list(SEEDS),
        "bootstrap": {"seed": 20260814, "replicates": bootstrap_replicates},
        "checkpoints": checkpoints,
        "datasets": datasets,
    }


def write_score_spec(
    *,
    root: Path,
    variant: str,
    bootstrap_replicates: int,
) -> Path:
    target = Path(root) / "evaluation" / "specs" / f"factorial_{variant}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            build_score_spec(
                root=root,
                variant=variant,
                bootstrap_replicates=bootstrap_replicates,
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target.resolve()


def score_variant(
    *,
    root: Path,
    variant: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    spec_path = write_score_spec(
        root=root,
        variant=variant,
        bootstrap_replicates=bootstrap_replicates,
    )
    result = factorial_score.score_from_spec_path(spec_path)
    output = Path(root) / "evaluation" / "results" / f"factorial_{variant}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "complete",
        "variant": variant,
        "spec": str(spec_path),
        "output": str(output.resolve()),
        "datasets": list(result["datasets"]),
    }


def _load_sarn_actions(
    path: Path,
    *,
    sample_ids: Sequence[str],
) -> dict[tuple[str, str], bool]:
    """Read the SARN action flag for the already validated all-six sidecar."""

    rows = factorial_score._read_jsonl(path, label=f"SARN sidecar ({path})")
    expected = {
        (sample_id, condition)
        for sample_id in sample_ids
        for condition in CONDITIONS
    }
    selected: dict[tuple[str, str], bool] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        condition = row.get("condition")
        key = (sample_id, condition)
        _require(key in expected, f"unexpected SARN action row: {key}")
        _require(key not in selected, f"duplicate SARN action row: {key}")
        applied = row.get("normalization_applied")
        _require(
            isinstance(applied, bool),
            f"SARN sidecar[{index}].normalization_applied is not boolean",
        )
        selected[key] = applied
    _require(set(selected) == expected, "SARN action sidecar Cartesian coverage drift")
    return selected


def _comparison_metrics(
    errors: Sequence[float],
    passed: Sequence[bool],
) -> dict[str, float]:
    _require(bool(errors) and len(errors) == len(passed), "metric vector drift")
    return {
        "nmae": statistics.fmean(float(error) for error in errors),
        "coverage": statistics.fmean(int(value) for value in passed),
        "acc_at_5pct": statistics.fmean(
            int(success and float(error) <= 0.05)
            for error, success in zip(errors, passed, strict=True)
        ),
    }


def _comparison_vectors(
    runs: Mapping[int, factorial_score.PredictionRun],
    *,
    targets: Sequence[Any],
    conditions: Sequence[str],
) -> dict[str, Any]:
    _require(tuple(runs) == SEEDS, "comparison run seed order drift")
    selected_conditions = tuple(conditions)
    _require(bool(selected_conditions), "comparison condition set is empty")
    per_seed_errors: list[list[float]] = [[] for _seed in SEEDS]
    per_seed_passed: list[list[bool]] = [[] for _seed in SEEDS]
    groups: list[str] = []
    sample_condition_keys: list[tuple[str, str]] = []
    for target in targets:
        for condition in selected_conditions:
            groups.append(str(target.group_id))
            sample_condition_keys.append((str(target.sample_id), condition))
            for index, seed in enumerate(SEEDS):
                prediction = runs[seed].rows[(target.sample_id, condition)]
                success = bool(prediction.passed)
                error = (
                    abs(
                        float(prediction.normalized_progress)
                        - float(target.normalized_target)
                    )
                    if success
                    else 1.0
                )
                per_seed_errors[index].append(error)
                per_seed_passed[index].append(success)

    averaged_errors = [
        statistics.fmean(seed_errors[index] for seed_errors in per_seed_errors)
        for index in range(len(groups))
    ]
    averaged_coverage = [
        statistics.fmean(int(seed_passed[index]) for seed_passed in per_seed_passed)
        for index in range(len(groups))
    ]
    averaged_accuracy = [
        statistics.fmean(
            int(per_seed_passed[seed_index][index]
                and per_seed_errors[seed_index][index] <= 0.05)
            for seed_index in range(len(SEEDS))
        )
        for index in range(len(groups))
    ]
    metrics = {
        "nmae": statistics.fmean(averaged_errors),
        "coverage": statistics.fmean(averaged_coverage),
        "acc_at_5pct": statistics.fmean(averaged_accuracy),
    }
    per_seed = [
        {
            "seed": seed,
            "method": runs[seed].method,
            "metrics": _comparison_metrics(
                per_seed_errors[index],
                per_seed_passed[index],
            ),
        }
        for index, seed in enumerate(SEEDS)
    ]
    return {
        "errors": averaged_errors,
        "groups": groups,
        "sample_condition_keys": sample_condition_keys,
        "metrics": metrics,
        "per_seed": per_seed,
    }


def _paired_raw_sarn_group_bootstrap(
    raw_errors: Sequence[float],
    sarn_errors: Sequence[float],
    groups: Sequence[str],
    *,
    group_unit: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap complete dataset groups after within-sample seed averaging."""

    _require(replicates >= 1, "bootstrap replicates must be positive")
    _require(
        len(raw_errors) == len(sarn_errors) == len(groups) and bool(groups),
        "paired bootstrap vectors are misaligned",
    )
    effects = [
        float(sarn) - float(raw)
        for raw, sarn in zip(raw_errors, sarn_errors, strict=True)
    ]
    group_sums: dict[str, float] = {}
    group_counts: dict[str, int] = {}
    for group, effect in zip(groups, effects, strict=True):
        group_sums[group] = group_sums.get(group, 0.0) + effect
        group_counts[group] = group_counts.get(group, 0) + 1
    group_ids = sorted(group_sums)
    _require(len(group_ids) >= 2, "paired bootstrap requires at least two groups")
    rng = random.Random(seed)
    draws: list[float] = []
    for _replicate in range(replicates):
        numerator = 0.0
        denominator = 0
        for _group in group_ids:
            chosen = rng.choice(group_ids)
            numerator += group_sums[chosen]
            denominator += group_counts[chosen]
        draws.append(numerator / denominator)
    point = statistics.fmean(effects)
    low = factorial_score._quantile(draws, 0.025)
    high = factorial_score._quantile(draws, 0.975)
    return {
        "metric": "full_denominator_nmae",
        "lower_is_better": True,
        "delta_definition": "sarn_v2_minus_raw",
        "failure_error": 1.0,
        "delta_nmae_sarn_v2_minus_raw": point,
        "paired_complete_group_bootstrap_ci95": {"low": low, "high": high},
        "evaluation_rows": len(effects),
        "group_clusters": len(group_ids),
        "group_unit": group_unit,
        "replicates": replicates,
        "seed": seed,
        "sarn_v2_better": point < 0.0,
        "superiority_ci95": high < 0.0,
    }


def compare_raw_sarn_dataset(
    *,
    root: Path,
    dataset: Dataset,
    bootstrap_replicates: int,
    dataset_index: int,
) -> dict[str, Any]:
    """Compare Raw and SARN-v2 GeoAttn predictions on one complete dataset."""

    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    sample_ids = factorial_score.load_validation_ids(dataset.roster)
    _require(
        len(sample_ids) == dataset.expected_samples,
        f"{dataset.paper_name}: sample count drift",
    )
    targets_tuple = factorial_score._load_targets_with_group_source(
        dataset.labels,
        sample_ids,
        group_source=dataset.group_source,
    )
    targets = {target.sample_id: target for target in targets_tuple}
    _require(
        len({target.group_id for target in targets_tuple}) == dataset.expected_groups,
        f"{dataset.paper_name}: group count drift",
    )
    manifest_by_id, _manifest_audit = factorial_score._load_plain_manifest(
        dataset.manifest,
        sample_ids=sample_ids,
    )

    raw_runs: dict[int, factorial_score.PredictionRun] = {}
    sarn_runs: dict[int, factorial_score.PredictionRun] = {}
    sarn_actions: dict[int, dict[tuple[str, str], bool]] = {}
    input_files: dict[str, Any] = {}
    for seed in SEEDS:
        raw_artifacts = prediction_artifacts(
            root,
            cell=RAW_VS_SARN_CELL,
            seed=seed,
            dataset=dataset,
            variant="raw",
        )
        sarn_artifacts = prediction_artifacts(
            root,
            cell=RAW_VS_SARN_CELL,
            seed=seed,
            dataset=dataset,
            variant="sarn_v2",
        )
        raw_runs[seed] = factorial_score._load_prediction_run(
            {
                "path": str(raw_artifacts["predictions"]),
                "method": method_id(cell=RAW_VS_SARN_CELL, seed=seed, variant="raw"),
                "protocol": PROTOCOL,
            },
            label=f"{dataset.paper_name}.raw.{seed}",
            base_dir=Path(root).resolve(),
            require_sidecar=False,
            robustness_seed=ROBUSTNESS_SEED,
            targets=targets,
            conditions=CONDITIONS,
            manifest_by_id=manifest_by_id,
        )
        sarn_runs[seed] = factorial_score._load_prediction_run(
            {
                "path": str(sarn_artifacts["predictions"]),
                "method": method_id(
                    cell=RAW_VS_SARN_CELL,
                    seed=seed,
                    variant="sarn_v2",
                ),
                "protocol": sarn_v2.PROTOCOL,
                "sidecar": str(sarn_artifacts["sidecar"]),
            },
            label=f"{dataset.paper_name}.sarn_v2.{seed}",
            base_dir=Path(root).resolve(),
            require_sidecar=True,
            robustness_seed=ROBUSTNESS_SEED,
            targets=targets,
            conditions=CONDITIONS,
            manifest_by_id=manifest_by_id,
        )
        sarn_actions[seed] = _load_sarn_actions(
            sarn_artifacts["sidecar"],
            sample_ids=sample_ids,
        )
        input_files[str(seed)] = {
            "raw_predictions": str(raw_artifacts["predictions"].resolve()),
            "sarn_v2_predictions": str(sarn_artifacts["predictions"].resolve()),
            "sarn_v2_sidecar": str(sarn_artifacts["sidecar"].resolve()),
        }

    paired_rows = 0
    noop_rows = 0
    projective_changed_rows = 0
    projective_applied_rows = 0
    for sample_id in sample_ids:
        for condition in CONDITIONS:
            raw_hashes: set[str] = set()
            sarn_pre_hashes: set[str] = set()
            sarn_post_hashes: set[str] = set()
            for seed in SEEDS:
                raw_value = raw_runs[seed].rows[(sample_id, condition)]
                sarn_value = sarn_runs[seed].rows[(sample_id, condition)]
                _require(
                    raw_value.condition_pixel_sha256
                    == sarn_value.condition_pixel_sha256,
                    f"{dataset.paper_name}/{sample_id}/{condition}/{seed}: "
                    "Raw and SARN-v2 input pixels differ",
                )
                _require(
                    sarn_runs[seed].sidecar_hashes is not None,
                    "SARN-v2 sidecar hashes are missing",
                )
                pre_hash, post_hash = sarn_runs[seed].sidecar_hashes[
                    (sample_id, condition)
                ]
                _require(
                    pre_hash == raw_value.condition_pixel_sha256,
                    f"{dataset.paper_name}/{sample_id}/{condition}/{seed}: "
                    "SARN-v2 pre-normalization pixels differ from Raw",
                )
                raw_hashes.add(raw_value.condition_pixel_sha256)
                sarn_pre_hashes.add(pre_hash)
                sarn_post_hashes.add(post_hash)
                applied = sarn_actions[seed][(sample_id, condition)]
                if condition in SARN_NOOP_CONDITIONS:
                    _require(
                        not applied and pre_hash == post_hash,
                        f"{dataset.paper_name}/{sample_id}/{condition}/{seed}: "
                        "SARN-v2 was not a strict no-op",
                    )
                    _require(
                        raw_value.passed == sarn_value.passed
                        and raw_value.normalized_progress
                        == sarn_value.normalized_progress
                        and raw_value.failure_code == sarn_value.failure_code,
                        f"{dataset.paper_name}/{sample_id}/{condition}/{seed}: "
                        "no-op Raw/SARN-v2 prediction differs",
                    )
                    noop_rows += 1
                else:
                    projective_changed_rows += int(pre_hash != post_hash)
                    projective_applied_rows += int(applied)
                paired_rows += 1
            _require(
                len(raw_hashes) == len(sarn_pre_hashes) == 1,
                f"{dataset.paper_name}/{sample_id}/{condition}: degraded input "
                "pixels differ across training seeds",
            )
            _require(
                len(sarn_post_hashes) == 1,
                f"{dataset.paper_name}/{sample_id}/{condition}: SARN-v2 output "
                "pixels differ across training seeds",
            )

    condition_sets = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": PROJECTIVE_CONDITIONS,
    }
    comparisons: dict[str, Any] = {}
    for condition_index, (label, selected_conditions) in enumerate(
        condition_sets.items()
    ):
        raw_vectors = _comparison_vectors(
            raw_runs,
            targets=targets_tuple,
            conditions=selected_conditions,
        )
        sarn_vectors = _comparison_vectors(
            sarn_runs,
            targets=targets_tuple,
            conditions=selected_conditions,
        )
        _require(
            raw_vectors["groups"] == sarn_vectors["groups"]
            and raw_vectors["sample_condition_keys"]
            == sarn_vectors["sample_condition_keys"],
            f"{dataset.paper_name}/{label}: paired vectors are misaligned",
        )
        effect = _paired_raw_sarn_group_bootstrap(
            raw_vectors["errors"],
            sarn_vectors["errors"],
            raw_vectors["groups"],
            group_unit=dataset.group_unit,
            replicates=bootstrap_replicates,
            seed=20260816 + dataset_index * len(condition_sets) + condition_index,
        )
        comparisons[label] = {
            "conditions": list(selected_conditions),
            "evaluation_rows": len(raw_vectors["errors"]),
            "groups": dataset.expected_groups,
            "raw": {
                "metrics": raw_vectors["metrics"],
                "per_seed": raw_vectors["per_seed"],
            },
            "sarn_v2": {
                "metrics": sarn_vectors["metrics"],
                "per_seed": sarn_vectors["per_seed"],
            },
            "paired_effect": effect,
        }

    return {
        "dataset": dataset.paper_name,
        "samples": dataset.expected_samples,
        "groups": dataset.expected_groups,
        "group_unit": dataset.group_unit,
        "group_source": dataset.group_source,
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "audit": {
            "same_sample_condition_seed_pre_normalization_pixels_raw_and_sarn_v2": True,
            "same_degraded_pixels_across_training_seeds": True,
            "same_sarn_v2_output_pixels_across_training_seeds": True,
            "clean_and_blur_sarn_v2_strict_noop": True,
            "clean_and_blur_raw_sarn_v2_predictions_identical": True,
            "paired_seed_sample_condition_rows": paired_rows,
            "clean_and_blur_noop_rows": noop_rows,
            "projective_rows": dataset.expected_samples
            * len(PROJECTIVE_CONDITIONS)
            * len(SEEDS),
            "projective_normalization_applied_rows": projective_applied_rows,
            "projective_changed_pixel_rows": projective_changed_rows,
        },
        "input_files": input_files,
        "comparisons": comparisons,
    }


def compare_variants(
    *,
    root: Path,
    bootstrap_replicates: int,
    output: Path | None = None,
) -> dict[str, Any]:
    """Write the formal five-dataset Raw-vs-SARN-v2 paired result."""

    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    datasets = {
        dataset.paper_name: compare_raw_sarn_dataset(
            root=root,
            dataset=dataset,
            bootstrap_replicates=bootstrap_replicates,
            dataset_index=index,
        )
        for index, dataset in enumerate(DATASETS.values())
    }
    payload = {
        "schema_version": 1,
        "protocol": RAW_VS_SARN_PROTOCOL,
        "status": "complete",
        "cell": RAW_VS_SARN_CELL,
        "raw_method": ARMS[RAW_VS_SARN_CELL].paper_label,
        "sarn_v2_method": f"SARN-v2+{ARMS[RAW_VS_SARN_CELL].paper_label}",
        "training_data": "SyncG scene-disjoint fit only",
        "field_data_role": "test only",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "projective_pooled_conditions": list(PROJECTIVE_CONDITIONS),
        "scoring_policy": {
            "failure_error": 1.0,
            "seed_handling": (
                "compute failure-penalized error independently for each training "
                "seed, then average the three errors within each sample-condition"
            ),
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "unit": "complete dataset-declared group; SyncG uses scene stem",
                "ci": "two-sided percentile 95%",
                "pairing": "SARN-v2 minus Raw on the identical sample-condition-seed",
            },
        },
        "datasets": datasets,
    }
    target = (
        Path(output)
        if output is not None
        else Path(root) / "evaluation" / "results" / "raw_vs_sarn_v2.json"
    ).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "complete",
        "output": str(target),
        "datasets": list(datasets),
        "bootstrap_replicates": bootstrap_replicates,
    }


def _selected_variants(value: str) -> tuple[str, ...]:
    return VARIANTS if value == "both" else (value,)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prediction.add_argument(
        "--variant", choices=("raw", "sarn_v2", "both"), default="both"
    )
    prediction.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    prediction.add_argument("--cell", action="append", choices=CELLS)
    prediction.add_argument("--seed", action="append", type=int, choices=SEEDS)
    prediction.add_argument("--device", default="cuda:0")

    score = commands.add_parser("score")
    score.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    score.add_argument(
        "--variant", choices=("raw", "sarn_v2", "both"), default="both"
    )
    score.add_argument("--bootstrap-replicates", type=int, default=20_000)

    comparison = commands.add_parser("compare-variants")
    comparison.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    comparison.add_argument("--bootstrap-replicates", type=int, default=20_000)
    comparison.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-variants":
        print(
            json.dumps(
                compare_variants(
                    root=args.root,
                    bootstrap_replicates=args.bootstrap_replicates,
                    output=args.output,
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.command == "score":
        for variant in _selected_variants(args.variant):
            print(
                json.dumps(
                    score_variant(
                        root=args.root,
                        variant=variant,
                        bootstrap_replicates=args.bootstrap_replicates,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0

    datasets = (
        tuple(DATASETS[name] for name in args.dataset)
        if args.dataset
        else tuple(DATASETS.values())
    )
    cells = tuple(args.cell) if args.cell else CELLS
    seeds = tuple(args.seed) if args.seed else SEEDS
    for variant in _selected_variants(args.variant):
        for cell in cells:
            for seed in seeds:
                for dataset in datasets:
                    print(
                        json.dumps(
                            {
                                "status": "starting",
                                "variant": variant,
                                "dataset": dataset.paper_name,
                                "method": method_id(
                                    cell=cell,
                                    seed=seed,
                                    variant=variant,
                                ),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    result = run_prediction_job(
                        root=args.root,
                        variant=variant,
                        dataset=dataset,
                        cell=cell,
                        seed=seed,
                        device_name=args.device,
                    )
                    print(
                        json.dumps(result, ensure_ascii=False, sort_keys=True),
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARMS",
    "CELLS",
    "DATASETS",
    "DEFAULT_ROOT",
    "PROJECTIVE_CONDITIONS",
    "PROTOCOL",
    "RAW_VS_SARN_CELL",
    "RAW_VS_SARN_PROTOCOL",
    "SARN_NOOP_CONDITIONS",
    "SEEDS",
    "VARIANTS",
    "build_score_spec",
    "checkpoint_path",
    "compare_raw_sarn_dataset",
    "compare_variants",
    "main",
    "method_id",
    "prediction_artifacts",
    "run_prediction_job",
    "run_raw_prediction",
    "run_sarn_prediction",
    "score_variant",
    "write_score_spec",
]
