"""Evaluate terminal A15.2 against existing external models on one exact cohort.

This runner regenerates the established six deterministic robustness conditions
for the historical 1,558-row SyncG scene holdout, runs the already-terminal
A15.2 model, and pairs every A15.2 prediction with the existing three-seed
Direct-ResNet18, MobileNetV3-Large, and EfficientNet-B0 prediction ledgers.

The pairing key is ``(sample_id, condition)``.  The established degraded-pixel
identity and existing SARN-v2 sidecar output identity must match all nine
external prediction ledgers before any metric is reported.  No training,
adaptation, checkpoint selection, or model mutation is performed here.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments import evaluate_paper_syncg_only_factorial as factorial
from experiments import evaluate_paper_syncg_only_lightweight_baselines as lightweight
from experiments import robustness_degradations
from experiments.a13_correction_dev_protocol import CORRECTION_TRAIN_SCENES
from experiments.a15_2_fteb_untouched_protocol import METHOD_A15_2
from experiments.evaluate_a15_2_fteb_fold_b import (
    EVALUATION_BATCH_SIZE,
    _configure_reproducibility,
    evaluate_same_batch_loader,
    load_fixed_final_models_before_fold_b,
)
from experiments.evaluate_a11_scort_core_audit import (
    _checkpoint_mapping as _validated_a11_checkpoint_mapping,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.prepare_a11_core_scene_split import (
    DEFAULT_TRAIN_MANIFEST as DEFAULT_A11_TRAIN_MANIFEST,
    load_a11_train_manifest,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_DEV_MANIFEST,
    load_a13_correction_dev_manifest,
)
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    ManifestRow,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest,
)
from experiments.summarize_support_normalized_cbam_pilot import load_scene_targets
from experiments.support_aware_roi_normalization_v2 import (
    PROTOCOL as SARN_V2_PROTOCOL,
    SIDECAR_KEYS,
    STAGE as SARN_V2_STAGE,
    normalize_support_aware_roi_v2,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
    _normalized_raw_to_sarn_homography,
)
from experiments.resnet18_direct_progress import IMAGE_SIZE
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PROTOCOL: Final[str] = "terminal_a15_2_syncg_scene_holdout_external_pairing_v1"
EXPECTED_SAMPLES: Final[int] = 1_558
EXPECTED_SCENES: Final[int] = 14
SEEDS: Final[tuple[int, ...]] = factorial.SEEDS
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
DEFAULT_PAPER_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1")
DEFAULT_MANIFEST: Final[Path] = factorial.DATASETS["syncg_scene_holdout"].manifest
DEFAULT_LABELS: Final[Path] = factorial.DATASETS["syncg_scene_holdout"].labels
DEFAULT_SPLIT: Final[Path] = factorial.DATASETS["syncg_scene_holdout"].roster
DEFAULT_A11: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/"
    "a11_scort_terminal5_seed20262020/a11_terminal.pt"
)
DEFAULT_A15_2: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/"
    "a15_2_fteb_terminal5_seed20262215/a15_2_terminal.pt"
)


class A152SceneHoldoutError(ValueError):
    """The shared cohort, pixels, checkpoints, or prediction rows differ."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152SceneHoldoutError(message)


def _jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSONL input does not exist: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise A152SceneHoldoutError(
                    f"invalid JSONL row: {source}:{line_number}"
                ) from exc
            _require(
                isinstance(row, Mapping),
                f"JSONL row is not an object: {source}:{line_number}",
            )
            yield row


def _pixel_digest(value: Any, *, label: str) -> str:
    text = str(value or "")
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{label} is not a lowercase pixel SHA-256",
    )
    return text


def _optional_fraction(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    _require(not isinstance(value, bool), f"{label} must not be boolean")
    number = float(value)
    _require(math.isfinite(number) and 0.0 <= number <= 1.0, f"{label} is invalid")
    return number


def _optional_bbox(value: Any, *, label: str) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    _require(
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value),
        f"{label} must be a four-integer list or null",
    )
    left, top, right, bottom = (int(item) for item in value)
    _require(
        0 <= left < right and 0 <= top < bottom,
        f"{label} is not an ordered nonempty box",
    )
    return left, top, right, bottom


class _SharedPixelDataset(Dataset[dict[str, Any]]):
    """Materialize exactly the historical benchmark pixels for A15.2."""

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        targets: Mapping[str, tuple[float, str]],
        *,
        condition: str,
    ) -> None:
        self.rows = tuple(rows)
        self.targets = targets
        self.condition = str(condition)
        _require(bool(self.rows), "shared-pixel dataset is empty")
        _require(self.condition in CONDITIONS, "unknown robustness condition")
        _require(
            {row.sample_id for row in self.rows} == set(targets),
            "manifest and target sample rosters differ",
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.rows[int(index)]
        _payload, clean = load_canonical_roi(source)
        degraded, _metadata = robustness_degradations.apply_degradation(
            clean,
            self.condition,
            sample_id=source.sample_id,
            seed=ROBUSTNESS_SEED,
        )
        degraded = np.ascontiguousarray(degraded)
        decision = normalize_support_aware_roi_v2(degraded)
        height, width = degraded.shape[:2]

        active = bool(decision.applied)
        raw_mask = decision.valid_support_mask
        if raw_mask is None:
            active = False
            support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            support = cv2.resize(
                np.asarray(raw_mask, dtype=np.float32),
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            support = np.clip(support, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(support).all() or float(support.sum()) <= 0.0:
                active = False
                support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        homography = (
            _normalized_raw_to_sarn_homography(
                decision,
                height=height,
                width=width,
            )
            if active
            else np.eye(3, dtype=np.float32)
        )
        target, scene = self.targets[source.sample_id]
        return {
            "sample_id": source.sample_id,
            "scene_stem": scene,
            "condition_name": self.condition,
            "condition_pixel_sha256": canonical_roi_pixel_sha256(degraded),
            "sarn_pixel_sha256": canonical_roi_pixel_sha256(decision.image),
            "original_view": normalized_rgb_tensor(
                direct_resize_whole_roi(degraded, size=IMAGE_SIZE)
            ),
            "sarn_view": normalized_rgb_tensor(
                direct_resize_whole_roi(decision.image, size=IMAGE_SIZE)
            ),
            "sarn_support_mask": torch.from_numpy(
                np.ascontiguousarray(support[None], dtype=np.float32)
            ),
            "sarn_active": torch.tensor(active, dtype=torch.bool),
            "raw_to_sarn_homography": torch.from_numpy(
                np.ascontiguousarray(homography, dtype=np.float32)
            ),
            "target": torch.tensor(target, dtype=torch.float32),
        }


class _PixelRecordingLoader:
    """Record pixel identities from the exact batches consumed by A15.2."""

    def __init__(self, loader: Any) -> None:
        self.loader = loader
        self.records: list[dict[str, str]] = []

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        for batch in self.loader:
            ids = batch.get("sample_id")
            conditions = batch.get("condition_name")
            pre = batch.get("condition_pixel_sha256")
            post = batch.get("sarn_pixel_sha256")
            _require(
                all(
                    isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                    for value in (ids, conditions, pre, post)
                ),
                "pixel identity batch fields are not sequences",
            )
            _require(
                len(ids) == len(conditions) == len(pre) == len(post),
                "pixel identity batch fields are misaligned",
            )
            self.records.extend(
                {
                    "sample_id": str(sample_id),
                    "condition": str(condition),
                    "condition_pixel_sha256": _pixel_digest(
                        pre_hash, label="A15.2 pre-SARN pixels"
                    ),
                    "sarn_pixel_sha256": _pixel_digest(
                        post_hash, label="A15.2 post-SARN pixels"
                    ),
                }
                for sample_id, condition, pre_hash, post_hash in zip(
                    ids, conditions, pre, post, strict=True
                )
            )
            yield batch


@dataclass(frozen=True, slots=True)
class _ExternalRun:
    method: str
    rows: Mapping[tuple[str, str], tuple[float | None, bool, str, str]]
    prediction_path: Path
    sidecar_path: Path


def _load_external_run(
    prediction_path: Path,
    sidecar_path: Path,
    *,
    expected_method: str,
    expected_keys: set[tuple[str, str]],
) -> _ExternalRun:
    predictions: dict[tuple[str, str], tuple[float | None, bool, str]] = {}
    protocols: set[str] = set()
    for index, row in enumerate(_jsonl(prediction_path)):
        _require(set(row) == set(OUTPUT_KEYS), f"external prediction schema drift: {index}")
        _require(row.get("schema_version") == 1, "external schema version drift")
        _require(row.get("robustness_seed") == ROBUSTNESS_SEED, "robustness seed drift")
        method = str(row.get("method") or "")
        _require(method == expected_method, "external method identity drift")
        key = (str(row.get("sample_id") or ""), str(row.get("condition") or ""))
        _require(key in expected_keys, f"unexpected external prediction key: {key}")
        _require(key not in predictions, f"duplicate external prediction key: {key}")
        protocols.add(str(row.get("protocol") or ""))
        status = str(row.get("status") or "")
        _require(status in {"pass", "fail"}, f"invalid external status: {key}")
        if status == "pass":
            value = float(row.get("normalized_progress"))
            _require(math.isfinite(value) and 0.0 <= value <= 1.0, "external value invalid")
            _require(row.get("failure_code") is None, "passing external row has failure")
        else:
            _require(row.get("normalized_progress") is None, "failed external row has value")
            value = None
        predictions[key] = (
            value,
            status == "pass",
            _pixel_digest(
                row.get("condition_pixel_sha256"), label="external pre-SARN pixels"
            ),
        )
    _require(set(predictions) == expected_keys, "external prediction Cartesian roster differs")
    _require(
        protocols == {SARN_V2_PROTOCOL},
        "external prediction protocol is not the historical SARN-v2 protocol",
    )

    sidecars: dict[tuple[str, str], tuple[str, str]] = {}
    for index, row in enumerate(_jsonl(sidecar_path)):
        _require(set(row) == set(SIDECAR_KEYS), f"external sidecar schema drift: {index}")
        _require(row.get("schema_version") == 1, "sidecar schema version drift")
        _require(row.get("protocol") == SARN_V2_PROTOCOL, "sidecar protocol drift")
        _require(row.get("stage") == SARN_V2_STAGE, "sidecar stage drift")
        _require(str(row.get("method") or "") == expected_method, "sidecar method drift")
        _require(row.get("robustness_seed") == ROBUSTNESS_SEED, "sidecar seed drift")
        key = (str(row.get("sample_id") or ""), str(row.get("condition") or ""))
        _require(key in expected_keys, f"unexpected sidecar key: {key}")
        _require(key not in sidecars, f"duplicate sidecar key: {key}")
        pre_hash = _pixel_digest(
            row.get("pre_normalization_pixel_sha256"),
            label="sidecar pre-SARN pixels",
        )
        post_hash = _pixel_digest(
            row.get("post_normalization_pixel_sha256"),
            label="sidecar post-SARN pixels",
        )
        applied = row.get("normalization_applied")
        _require(isinstance(applied, bool), f"sidecar action is not boolean: {key}")
        normalized_bbox = _optional_bbox(
            row.get("normalization_bbox_xyxy"), label="normalization bbox"
        )
        _optional_bbox(
            row.get("detected_support_bbox_xyxy"), label="detected support bbox"
        )
        _optional_fraction(
            row.get("detected_support_area_fraction"), label="detected support area"
        )
        crop_fraction = _optional_fraction(
            row.get("normalization_crop_area_fraction"), label="normalization crop area"
        )
        confidence = _optional_fraction(
            row.get("normalization_confidence"), label="normalization confidence"
        )
        _require(crop_fraction is not None and crop_fraction > 0.0, "crop area is missing")
        _require(confidence is not None, "normalization confidence is missing")
        components = row.get("significant_components_merged")
        _require(
            components is None
            or (
                isinstance(components, int)
                and not isinstance(components, bool)
                and components >= 1
            ),
            "significant component count is invalid",
        )
        epsilon = row.get("quad_epsilon_fraction")
        _require(
            epsilon is None
            or (
                not isinstance(epsilon, bool)
                and math.isfinite(float(epsilon))
                and float(epsilon) > 0.0
            ),
            "quad epsilon is invalid",
        )
        ratio = _optional_fraction(
            row.get("quad_area_hull_ratio"), label="quad area/hull ratio"
        )
        if ratio is not None:
            _require(ratio > 0.0, "quad area/hull ratio must be positive")
        fallback = row.get("normalization_fallback")
        if applied:
            _require(
                normalized_bbox is not None and fallback is None,
                f"applied SARN row lacks bbox or has fallback: {key}",
            )
        else:
            _require(
                normalized_bbox is None
                and isinstance(fallback, str)
                and bool(fallback),
                f"fallback SARN row has invalid bbox/reason: {key}",
            )
            _require(
                crop_fraction == 1.0 and post_hash == pre_hash,
                f"fallback SARN row is not a pixel no-op: {key}",
            )
        sidecars[key] = (pre_hash, post_hash)
    _require(set(sidecars) == expected_keys, "external sidecar Cartesian roster differs")
    combined: dict[tuple[str, str], tuple[float | None, bool, str, str]] = {}
    for key, (value, passed, pre_hash) in predictions.items():
        sidecar_pre, sidecar_post = sidecars[key]
        _require(pre_hash == sidecar_pre, f"external prediction/sidecar pixels differ: {key}")
        combined[key] = (value, passed, pre_hash, sidecar_post)
    return _ExternalRun(
        method=expected_method,
        rows=combined,
        prediction_path=Path(prediction_path).resolve(),
        sidecar_path=Path(sidecar_path).resolve(),
    )


def _external_specs(root: Path) -> dict[str, list[tuple[int, str, Path, Path]]]:
    dataset = factorial.DATASETS["syncg_scene_holdout"]
    factorial_root = Path(root) / "factorial"
    result: dict[str, list[tuple[int, str, Path, Path]]] = {
        "Direct-ResNet18": []
    }
    for seed in SEEDS:
        artifacts = factorial.prediction_artifacts(
            factorial_root,
            cell="00",
            seed=seed,
            dataset=dataset,
            variant="sarn_v2",
        )
        result["Direct-ResNet18"].append(
            (
                seed,
                factorial.method_id(cell="00", seed=seed, variant="sarn_v2"),
                artifacts["predictions"],
                artifacts["sidecar"],
            )
        )
    for architecture in lightweight.ARCHITECTURES:
        label = lightweight.paper_name(architecture)
        result[label] = []
        for seed in SEEDS:
            artifacts = lightweight.prediction_artifacts(
                root,
                architecture=architecture,
                seed=seed,
                dataset=dataset,
                variant="sarn_v2",
            )
            result[label].append(
                (
                    seed,
                    lightweight.method_id(architecture, seed, variant="sarn_v2"),
                    artifacts["predictions"],
                    artifacts["sidecar"],
                )
            )
    return result


def _metrics(errors: Sequence[float], passed: Sequence[bool]) -> dict[str, float]:
    _require(bool(errors) and len(errors) == len(passed), "metric vectors differ")
    values = np.asarray(errors, dtype=np.float64)
    success = np.asarray(passed, dtype=np.float64)
    return {
        "nmae": float(values.mean()),
        "coverage": float(success.mean()),
        "acc_at_1pct": float((values <= 0.01).mean()),
        "acc_at_2pct": float((values <= 0.02).mean()),
        "acc_at_5pct": float((values <= 0.05).mean()),
    }


def paired_scene_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(
        len(candidate_errors) == len(comparator_errors) == len(groups) and bool(groups),
        "paired bootstrap vectors differ",
    )
    _require(replicates >= 1, "bootstrap replicates must be positive")
    group_order = sorted(set(groups))
    _require(len(group_order) >= 2, "paired bootstrap needs at least two scenes")
    deltas = np.asarray(candidate_errors, dtype=np.float64) - np.asarray(
        comparator_errors, dtype=np.float64
    )
    group_sums = np.asarray(
        [sum(deltas[i] for i, group in enumerate(groups) if group == name) for name in group_order],
        dtype=np.float64,
    )
    group_sizes = np.asarray(
        [sum(group == name for group in groups) for name in group_order], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(group_order), size=len(group_order))
        draws[index] = float(group_sums[selected].sum() / group_sizes[selected].sum())
    return {
        "delta_nmae_a15_2_minus_external": float(deltas.mean()),
        "scene_bootstrap_ci95": {
            "low": float(np.quantile(draws, 0.025)),
            "high": float(np.quantile(draws, 0.975)),
        },
        "scenes": len(group_order),
        "replicates": replicates,
        "seed": seed,
        "uncertainty_scope": (
            "scene resampling conditional on one fixed A15.2 terminal prediction "
            "and each row's fixed three-seed external mean error"
        ),
        "covers_training_seed_uncertainty": False,
    }


def _training_overlap_evidence(
    *,
    holdout_ids: set[str],
    holdout_scenes: set[str],
    a11_train_ids: set[str],
    a11_train_scenes: set[str],
    a15_correction_train_ids: set[str],
    a15_correction_train_scenes: set[str],
    a13_correction_dev_ids: set[str],
    a13_correction_dev_scenes: set[str],
) -> dict[str, Any]:
    a11_id_overlap = holdout_ids & a11_train_ids
    a11_scene_overlap = holdout_scenes & a11_train_scenes
    a15_id_overlap = holdout_ids & a15_correction_train_ids
    a15_scene_overlap = holdout_scenes & a15_correction_train_scenes
    dev_id_overlap = holdout_ids & a13_correction_dev_ids
    dev_scene_overlap = holdout_scenes & a13_correction_dev_scenes
    _require(not a11_id_overlap, "holdout overlaps terminal A11 anchor training sample IDs")
    _require(not a11_scene_overlap, "holdout overlaps terminal A11 anchor training scenes")
    _require(not a15_id_overlap, "holdout overlaps A15.2 correction training sample IDs")
    _require(not a15_scene_overlap, "holdout overlaps A15.2 correction training scenes")
    _require(not dev_id_overlap, "holdout overlaps A13 correction-development sample IDs")
    _require(not dev_scene_overlap, "holdout overlaps A13 correction-development scenes")
    return {
        "a11_anchor_train_sample_overlap": 0,
        "a11_anchor_train_scene_overlap": 0,
        "a15_correction_train_sample_overlap": 0,
        "a15_correction_train_scene_overlap": 0,
        "a13_correction_dev_sample_overlap": 0,
        "a13_correction_dev_scene_overlap": 0,
        "a13_correction_dev_overlap_checked": True,
    }


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    condition_sets: dict[str, tuple[str, ...]] = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": PROJECTIVE_CONDITIONS,
        "all_conditions": CONDITIONS,
    }
    result: dict[str, Any] = {}
    for condition_index, (label, selected) in enumerate(condition_sets.items()):
        subset = [row for row in rows if str(row["condition"]) in selected]
        _require(bool(subset), f"empty summary condition: {label}")
        a15_errors = [float(row["a15_2"]["absolute_error"]) for row in subset]
        groups = [str(row["scene_stem"]) for row in subset]
        model_results: dict[str, Any] = {}
        for model_index, model in enumerate(
            ("Direct-ResNet18", "MobileNetV3-Large", "EfficientNet-B0")
        ):
            per_seed_metrics: list[dict[str, Any]] = []
            per_row_external_error: list[float] = []
            for seed_index, seed in enumerate(SEEDS):
                errors = [
                    float(row["external"][model]["absolute_errors"][seed_index])
                    for row in subset
                ]
                passed = [
                    bool(row["external"][model]["passed"][seed_index]) for row in subset
                ]
                per_seed_metrics.append({"seed": seed, **_metrics(errors, passed)})
            for row in subset:
                per_row_external_error.append(
                    statistics.fmean(row["external"][model]["absolute_errors"])
                )
            metric_names = tuple(_metrics([0.0], [True]))
            model_results[model] = {
                "aggregation": "score each training seed, then average metrics",
                "per_seed": per_seed_metrics,
                "mean_across_seeds": {
                    name: statistics.fmean(value[name] for value in per_seed_metrics)
                    for name in metric_names
                },
                "sample_sd_across_seeds": {
                    name: statistics.stdev(value[name] for value in per_seed_metrics)
                    for name in metric_names
                },
                "paired_a15_2_vs_external_seed_mean_error": paired_scene_bootstrap(
                    a15_errors,
                    per_row_external_error,
                    groups,
                    replicates=bootstrap_replicates,
                    seed=20260818 + model_index * len(condition_sets) + condition_index,
                ),
            }
        result[label] = {
            "conditions": list(selected),
            "rows": len(subset),
            "scenes": len(set(groups)),
            "a15_2": {
                **_metrics(a15_errors, [True] * len(a15_errors)),
                "fixed_terminal_count": 1,
                "training_seed_sd": None,
            },
            "external_models": model_results,
        }
    return result


def evaluate_same_syncg_scene_holdout(
    *,
    manifest_path: Path,
    labels_path: Path,
    split_path: Path,
    terminal_a11_checkpoint_path: Path,
    a11_train_manifest_path: Path,
    a13_correction_dev_manifest_path: Path,
    terminal_a15_2_checkpoint_path: Path,
    external_root: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
    batch_size: int = EVALUATION_BATCH_SIZE,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    _require(workers >= 0, "workers must be nonnegative")
    _require(1 <= batch_size <= EVALUATION_BATCH_SIZE, "batch size differs")
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(device)

    # Checkpoints are fixed and validated before opening any cohort artifact.
    anchor, correction, checkpoint_metadata = load_fixed_final_models_before_fold_b(
        terminal_a11_checkpoint_path=terminal_a11_checkpoint_path,
        terminal_a15_2_correction_checkpoint_path=terminal_a15_2_checkpoint_path,
        device=device,
    )
    validated_a11_checkpoint = _validated_a11_checkpoint_mapping(
        terminal_a11_checkpoint_path
    )
    recorded_a11_manifest = Path(
        str(validated_a11_checkpoint["training"]["train_manifest"])
    ).resolve()
    supplied_a11_manifest = Path(a11_train_manifest_path).resolve()
    _require(
        supplied_a11_manifest == recorded_a11_manifest,
        "A11 train manifest differs from the terminal checkpoint record",
    )
    a11_train_samples = tuple(load_a11_train_manifest(supplied_a11_manifest))
    a11_train_ids = {str(sample.sample_id) for sample in a11_train_samples}
    a11_train_scenes = {str(sample.scene_stem) for sample in a11_train_samples}
    correction_dev_manifest = Path(a13_correction_dev_manifest_path).resolve()
    correction_dev_samples = tuple(
        load_a13_correction_dev_manifest(correction_dev_manifest)
    )
    correction_dev_ids = {
        str(sample.sample_id) for sample in correction_dev_samples
    }
    correction_dev_scenes = {
        str(sample.scene_stem) for sample in correction_dev_samples
    }
    targets = load_scene_targets(labels_path, split_path)
    _require(len(targets) == EXPECTED_SAMPLES, "SyncG holdout sample count differs")
    _require(
        len({scene for _target, scene in targets.values()}) == EXPECTED_SCENES,
        "SyncG holdout scene count differs",
    )
    manifest_rows = load_manifest(manifest_path)
    by_id = {row.sample_id: row for row in manifest_rows}
    _require(len(by_id) == len(manifest_rows), "manifest sample IDs repeat")
    _require(set(by_id) == set(targets), "manifest and target rosters differ")
    ordered_rows = tuple(by_id[sample_id] for sample_id in targets)

    terminal_metadata = checkpoint_metadata.get("terminal_a15_2")
    _require(isinstance(terminal_metadata, dict), "A15.2 checkpoint metadata is missing")
    train_ids = terminal_metadata.pop("physical_correction_train_sample_ids", None)
    _require(isinstance(train_ids, list) and bool(train_ids), "correction train IDs missing")
    train_scenes = {Path(scene).stem for scene in CORRECTION_TRAIN_SCENES}
    holdout_scenes = {scene for _target, scene in targets.values()}
    overlap_evidence = _training_overlap_evidence(
        holdout_ids=set(targets),
        holdout_scenes=holdout_scenes,
        a11_train_ids=a11_train_ids,
        a11_train_scenes=a11_train_scenes,
        a15_correction_train_ids=set(train_ids),
        a15_correction_train_scenes=train_scenes,
        a13_correction_dev_ids=correction_dev_ids,
        a13_correction_dev_scenes=correction_dev_scenes,
    )

    a15_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        dataset = _SharedPixelDataset(ordered_rows, targets, condition=condition)
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            workers=workers,
            seed=20260818 + condition_index,
            cuda=device.type == "cuda",
        )
        recording_loader = _PixelRecordingLoader(loader)
        evaluation = evaluate_same_batch_loader(
            anchor,
            correction,
            recording_loader,
            device=device,
            expected_sample_ids=tuple(targets),
            expected_condition=condition,
        )
        rows = evaluation["per_sample"]
        _require(len(rows) == len(recording_loader.records), "A15.2 pixel rows differ")
        for row, pixels in zip(rows, recording_loader.records, strict=True):
            key = (str(row["sample_id"]), str(row["condition"]))
            _require(
                key == (pixels["sample_id"], pixels["condition"]),
                "A15.2 prediction/pixel row order differs",
            )
            _require(key not in a15_rows, f"duplicate A15.2 row: {key}")
            target, scene = targets[key[0]]
            prediction = float(row["mean"][METHOD_A15_2])
            a15_rows[key] = {
                "sample_id": key[0],
                "scene_stem": scene,
                "condition": key[1],
                "normalized_target": target,
                "condition_pixel_sha256": pixels["condition_pixel_sha256"],
                "sarn_pixel_sha256": pixels["sarn_pixel_sha256"],
                "a15_2": {
                    "prediction": prediction,
                    "absolute_error": abs(prediction - target),
                },
                "external": {},
            }

    expected_keys = {
        (sample_id, condition) for sample_id in targets for condition in CONDITIONS
    }
    _require(set(a15_rows) == expected_keys, "A15.2 Cartesian roster differs")
    external_bindings: dict[str, list[dict[str, Any]]] = {}
    for label, specs in _external_specs(external_root).items():
        runs: list[_ExternalRun] = []
        external_bindings[label] = []
        for seed, method, prediction_path, sidecar_path in specs:
            run = _load_external_run(
                prediction_path,
                sidecar_path,
                expected_method=method,
                expected_keys=expected_keys,
            )
            runs.append(run)
            external_bindings[label].append(
                {
                    "seed": seed,
                    "method": method,
                    "predictions": str(run.prediction_path),
                    "sarn_sidecar": str(run.sidecar_path),
                }
            )
        _require(len(runs) == len(SEEDS), f"external seed roster differs: {label}")
        for key in expected_keys:
            a15 = a15_rows[key]
            values: list[float | None] = []
            passed: list[bool] = []
            errors: list[float] = []
            target = float(a15["normalized_target"])
            for run in runs:
                value, success, pre_hash, post_hash = run.rows[key]
                _require(
                    pre_hash == a15["condition_pixel_sha256"],
                    f"A15.2/external degraded pixels differ: {label}/{key}",
                )
                _require(
                    post_hash == a15["sarn_pixel_sha256"],
                    f"A15.2/external SARN pixels differ: {label}/{key}",
                )
                values.append(value)
                passed.append(success)
                errors.append(abs(float(value) - target) if success else 1.0)
            a15["external"][label] = {
                "seeds": list(SEEDS),
                "predictions": values,
                "passed": passed,
                "absolute_errors": errors,
            }

    ordered_output_rows = [
        a15_rows[(sample_id, condition)]
        for sample_id in targets
        for condition in CONDITIONS
    ]
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "fixed_terminal_a15_2": True,
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "training_or_adaptation": False,
            "checkpoint_or_method_selection": False,
            "historical_syncg_holdout_already_used_by_prior_experiments": True,
            "new_blind_test_claim": False,
        },
        "data": {
            "manifest": str(Path(manifest_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "split": str(Path(split_path).resolve()),
            "samples": EXPECTED_SAMPLES,
            "scenes": EXPECTED_SCENES,
            "conditions": list(CONDITIONS),
            "rows": len(ordered_output_rows),
            "robustness_seed": ROBUSTNESS_SEED,
            "a11_anchor_train_manifest": str(supplied_a11_manifest),
            "a11_anchor_train_samples": len(a11_train_ids),
            "a11_anchor_train_scenes": len(a11_train_scenes),
            "a13_correction_dev_manifest": str(correction_dev_manifest),
            "a13_correction_dev_samples": len(correction_dev_ids),
            "a13_correction_dev_scenes": len(correction_dev_scenes),
            **overlap_evidence,
        },
        "pixel_pairing": {
            "key": ["sample_id", "condition"],
            "degraded_pre_sarn_pixels_match_all_models_and_seeds": True,
            "post_sarn_pixels_match_all_models_and_seeds": True,
            "a15_2_and_external_models_share_sarn_v2_materialization": True,
        },
        "uncertainty": {
            "a15_2": "one fixed terminal checkpoint; no training-seed SD estimated",
            "external_models": "three fixed training seeds; metrics report mean and sample SD",
            "paired_ci": (
                "14-scene block bootstrap conditional on the fixed A15.2 predictions "
                "and per-row external three-seed mean errors"
            ),
            "paired_ci_covers_training_seed_uncertainty": False,
        },
        "checkpoints": checkpoint_metadata,
        "external_prediction_bindings": external_bindings,
        "summary": _summarize_rows(
            ordered_output_rows,
            bootstrap_replicates=bootstrap_replicates,
        ),
        "per_sample_condition": ordered_output_rows,
    }
    payload = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise A152SceneHoldoutError(f"output already exists: {output}") from exc
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--terminal-a11-checkpoint", type=Path, default=DEFAULT_A11)
    parser.add_argument(
        "--a11-train-manifest", type=Path, default=DEFAULT_A11_TRAIN_MANIFEST
    )
    parser.add_argument(
        "--a13-correction-dev-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_DEV_MANIFEST,
    )
    parser.add_argument(
        "--terminal-a15-2-checkpoint", type=Path, default=DEFAULT_A15_2
    )
    parser.add_argument("--external-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_same_syncg_scene_holdout(
        manifest_path=args.manifest,
        labels_path=args.labels,
        split_path=args.split,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        a11_train_manifest_path=args.a11_train_manifest,
        a13_correction_dev_manifest_path=args.a13_correction_dev_manifest,
        terminal_a15_2_checkpoint_path=args.terminal_a15_2_checkpoint,
        external_root=args.external_root,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A152SceneHoldoutError",
    "PROTOCOL",
    "build_argument_parser",
    "evaluate_same_syncg_scene_holdout",
    "paired_scene_bootstrap",
]
