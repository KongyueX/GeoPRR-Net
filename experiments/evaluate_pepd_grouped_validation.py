"""Evaluate a converged PEPD seed on its SyncG-train grouped validation split.

The frozen conditions are clean, 25-degree perspective, 45-degree perspective,
and severe perspective plus blur.  Every model forward is decoded through
direct-only, circular-only, and fused views without retraining or checkpoint
selection.  The evaluator is cohort-gated and cannot accept an arbitrary
manifest, checkpoint, dataset, or output location.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    DECODER_VIEWS,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    GROUPED_VAL_BOOTSTRAP_ITERATIONS,
    GROUPED_VAL_BOOTSTRAP_SEED,
    GROUPED_VAL_CONDITIONS,
    GROUPED_VAL_DEGRADATION_SEED,
    PEPD_COHORT_PROTOCOL,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_DECODER_VIEW_PROTOCOL,
    PEPD_GROUPED_VAL_EVALUATION_PROTOCOL,
    PEPD_MECHANISM_COHORT_PROTOCOL,
    PEPD_MECHANISM_RUN_PROTOCOL,
    PEPD_MECHANISM_VERIFICATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PRIMARY_MECHANISM_ARMS,
    PROJECT_ROOT,
    formal_manifest_path,
    formal_output_dir,
    formal_parent_pin,
    mechanism_output_dir,
    sha256_file,
)
from experiments.pepd_convergence_extension_v2_protocol import (
    EXTENSION_SEED,
    PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
    PEPD_EXTENSION_PROTOCOL,
    PEPD_EXTENSION_VERIFICATION_PROTOCOL,
    authoritative_cohort_path,
    extension_output_dir,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import (
    _transform_point_homography,
    build_probabilistic_pivot_direction_model,
    circular_delta,
    decode_probabilistic_pivot_direction,
    soft_pivot_coordinates,
    transform_pivot_direction,
)
from experiments.robustness_degradations import (
    ROBUSTNESS_PROTOCOL,
    apply_degradation,
)
from experiments.vdn_baseline import (
    affine_for_dial,
    angular_error_degrees,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_source_file,
    transform_point,
)


COHORT_PATH = authoritative_cohort_path()
MECHANISM_COHORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_mechanism_phase2"
    / "cohort.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=PRIMARY_MECHANISM_ARMS,
        default="full",
    )
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--condition",
        choices=GROUPED_VAL_CONDITIONS,
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _mechanism_cohort_gate() -> dict[str, Any]:
    cohort = _load_json(MECHANISM_COHORT_PATH)
    if (
        cohort.get("protocol") != PEPD_MECHANISM_COHORT_PROTOCOL
        or cohort.get("status") != "complete"
        or cohort.get("all_runs_verified_and_converged") is not True
        or cohort.get("controlled_grouped_validation_authorized") is not True
        or cohort.get("public_test_field_evaluation_authorized") is not False
        or tuple(cohort.get("seeds") or ()) != FORMAL_SEEDS
        or tuple(cohort.get("arms") or ()) != PRIMARY_MECHANISM_ARMS
    ):
        raise ValueError("PEPD mechanism cohort gate failed")
    horizon_note = cohort.get("horizon_note")
    if not isinstance(horizon_note, str) or "bounded extension" not in horizon_note:
        raise ValueError("PEPD mechanism cohort horizon note is missing")
    source_identity = cohort.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ValueError("PEPD mechanism cohort source identity is missing")
    if source_identity.get("strict_json") != strict_json_source_sha256():
        raise ValueError("PEPD mechanism cohort strict-JSON identity drifted")
    if source_identity.get("protocol") != sha256_source_file(
        PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
    ):
        raise ValueError("PEPD mechanism cohort protocol source identity drifted")
    if source_identity.get("authoritative_cohort") != sha256_file(COHORT_PATH):
        raise ValueError("PEPD mechanism cohort authoritative binding drifted")
    return cohort


def _cohort_run(
    cohort: Mapping[str, Any],
    *,
    seed: int,
    arm: str | None = None,
) -> Mapping[str, Any]:
    runs = cohort.get("runs")
    if not isinstance(runs, list):
        raise ValueError("cohort run membership is missing")
    matches = [
        row
        for row in runs
        if isinstance(row, Mapping)
        and int(row.get("seed", -1)) == seed
        and (arm is None or row.get("arm") == arm)
    ]
    if len(matches) != 1:
        raise ValueError(f"cohort has no unique run binding for {arm}:{seed}")
    return matches[0]


def _authoritative_full_binding(
    cohort: Mapping[str, Any],
    mechanism_cohort: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Path, str, str, str]:
    if (
        cohort.get("protocol") != PEPD_AUTHORITATIVE_COHORT_PROTOCOL
        or cohort.get("status") != "converged"
        or cohort.get("all_runs_verified") is not True
        or cohort.get("all_runs_converged") is not True
        or cohort.get("grouped_validation_controlled_perspective_authorized")
        is not True
        or cohort.get("grouped_validation_controlled_robustness_authorized")
        is not True
        or cohort.get("public_test_field_evaluation_authorized") is not False
        or tuple(cohort.get("seeds") or ()) != FORMAL_SEEDS
        or cohort.get("mixed_authority")
        != {
            "20260720": "convergence_v1",
            "20260721": "bounded_extension_v2",
            "20260722": "convergence_v1",
        }
    ):
        raise ValueError("mixed authoritative PEPD v2 cohort gate failed")
    if (cohort.get("source_identity") or {}).get(
        "strict_json"
    ) != strict_json_source_sha256():
        raise ValueError("authoritative cohort strict-JSON identity drifted")

    if seed == EXTENSION_SEED:
        run_dir = extension_output_dir()
        summary_protocol = PEPD_EXTENSION_PROTOCOL
        verification_protocol = PEPD_EXTENSION_VERIFICATION_PROTOCOL
        source_phase = "bounded_extension_v2"
    else:
        run_dir = formal_output_dir(seed)
        summary_protocol = PEPD_CONTINUATION_PROTOCOL
        verification_protocol = PEPD_RUN_VERIFICATION_PROTOCOL
        source_phase = "convergence_v1"
    row = _cohort_run(cohort, seed=seed)
    mechanism_row = _cohort_run(mechanism_cohort, seed=seed, arm="full")
    if (
        row.get("verified") is not True
        or row.get("converged") is not True
        or row.get("source_phase") != source_phase
        or Path(str(row.get("source_run_dir"))).resolve() != run_dir
        or row.get("authoritative_run_protocol") != summary_protocol
        or row.get("verification_protocol") != verification_protocol
    ):
        raise ValueError(f"full seed {seed} authoritative run binding drifted")
    for name in (
        "best_checkpoint_sha256",
        "summary_sha256",
        "verification_sha256",
    ):
        if mechanism_row.get(name) != row.get(name):
            raise ValueError(
                f"full seed {seed} mechanism/authoritative {name} mismatch"
            )
    return run_dir, summary_protocol, verification_protocol, source_phase


def _decode_direction_views(
    outputs: Sequence[torch.Tensor],
    fused_prediction: Any,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Decode three fixed views from one shared set of model outputs."""

    if len(outputs) != 4:
        raise ValueError("PEPD decoder views require exactly four model outputs")
    direction_raw = outputs[1].float()
    angle_logits = outputs[2].float()
    raw_norm = torch.linalg.vector_norm(direction_raw, dim=1)
    direct = F.normalize(direction_raw, dim=1, eps=1e-8)

    probability = torch.softmax(angle_logits, dim=1)
    centers = torch.arange(
        angle_logits.shape[1],
        device=angle_logits.device,
        dtype=probability.dtype,
    ) * (2.0 * math.pi / float(angle_logits.shape[1]))
    circular_vector = torch.stack(
        (
            torch.sum(probability * torch.cos(centers), dim=1),
            torch.sum(probability * torch.sin(centers), dim=1),
        ),
        dim=1,
    )
    circular_norm = torch.linalg.vector_norm(circular_vector, dim=1)
    circular = F.normalize(circular_vector, dim=1, eps=1e-8)

    common_finite = (
        torch.isfinite(fused_prediction.pivot_peak)
        & torch.isfinite(fused_prediction.log_variance)
    )
    direct_valid = (
        common_finite
        & torch.isfinite(direct).all(dim=1)
        & (raw_norm > 1e-8)
    )
    circular_valid = (
        common_finite
        & torch.isfinite(circular).all(dim=1)
        & (circular_norm > 1e-8)
    )
    views = {
        "direct": (direct, direct_valid),
        "circular": (circular, circular_valid),
        "fused": (
            fused_prediction.direction,
            fused_prediction.valid,
        ),
    }
    if tuple(views) != DECODER_VIEWS:
        raise RuntimeError("decoder-view order drifted from the frozen protocol")
    return views


class GroupedValidationPerspectiveDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Any],
        *,
        image_size: int,
        expansion: float,
        condition: str,
        degradation_seed: int,
    ) -> None:
        if condition not in GROUPED_VAL_CONDITIONS:
            raise ValueError(f"unsupported grouped-validation condition: {condition}")
        self.samples = list(samples)
        if not self.samples:
            raise ValueError("grouped-validation dataset is empty")
        self.image_size = int(image_size)
        self.expansion = float(expansion)
        self.condition = condition
        self.degradation_seed = int(degradation_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = cv2.imread(
            sample.image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if image is None:
            raise ValueError(f"failed to read grouped-validation image: {sample.image_path}")
        affine = affine_for_dial(
            sample.dial_bbox,
            output_size=self.image_size,
            expansion=self.expansion,
        )
        crop = cv2.warpAffine(
            image,
            affine,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        pivot = transform_point(sample.pointer_tail, affine).astype(np.float32)
        tip = transform_point(sample.pointer_tip, affine).astype(np.float32)
        degraded, metadata = apply_degradation(
            crop,
            self.condition,
            sample_id=sample.sample_id,
            seed=self.degradation_seed,
        )
        perspective = metadata.get("perspective")
        homography = np.eye(3, dtype=np.float32)
        if isinstance(perspective, Mapping):
            homography = np.asarray(perspective["homography"], dtype=np.float32)
            pivot = _transform_point_homography(pivot, homography)
            tip = _transform_point_homography(tip, homography)
        target = tip - pivot
        norm = float(np.linalg.norm(target))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f"{sample.sample_id}: transformed target collapsed")
        target = (target / norm).astype(np.float32)
        return {
            "image": normalized_rgb_tensor(degraded),
            "reference_image": normalized_rgb_tensor(crop),
            "homography": torch.from_numpy(homography),
            "target_direction": torch.from_numpy(target),
            "target_pivot": torch.from_numpy(pivot),
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
        }


def _quantile_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        error = (
            float(row["angle_error_degrees"])
            if row.get("valid") is True
            else 180.0
        )
        by_group[str(row["group_id"])].append(error)
    if not by_group:
        raise ValueError("no valid groups for grouped bootstrap")
    group_means = np.asarray(
        [float(np.mean(values)) for _, values in sorted(by_group.items())],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = generator.integers(0, len(group_means), size=len(group_means))
        draws[index] = float(np.mean(group_means[selected]))
    return {
        "groups": int(len(group_means)),
        "macro_angle_mae_degrees": float(np.mean(group_means)),
        "group_bootstrap_iterations": int(iterations),
        "group_bootstrap_seed": int(seed),
        "macro_angle_mae_95ci": _quantile_interval(draws),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    errors = np.asarray(
        [
            (
                float(row["angle_error_degrees"])
                if row.get("valid") is True
                else 180.0
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    signed = np.asarray(
        [
            (
                float(row["signed_angle_error_degrees"])
                if row.get("valid") is True
                else 180.0
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    stds = np.asarray(
        [float(row["angle_std_degrees"]) for row in rows],
        dtype=np.float64,
    )
    pivot_valid = [
        row
        for row in rows
        if row.get("pivot_valid", row.get("valid")) is True
    ]
    pivots = np.asarray(
        [
            (
                float(row["pivot_error_fraction"])
                if row.get("pivot_valid", row.get("valid")) is True
                else math.sqrt(2.0)
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if not errors.size:
        raise ValueError("PEPD grouped-validation evaluation has no samples")
    variances = np.maximum(np.square(np.deg2rad(stds)), 1e-12)
    nll = 0.5 * (
        np.square(np.deg2rad(signed)) / variances + np.log(variances)
    )
    return {
        "samples": len(rows),
        "valid_directions": len(valid),
        "direction_coverage": len(valid) / len(rows),
        "invalid_direction_error_degrees": 180.0,
        "angle_mae_degrees": float(np.mean(errors)),
        "angle_median_degrees": float(np.median(errors)),
        "angle_acc_1deg": float(np.mean(errors <= 1.0)),
        "angle_acc_3deg": float(np.mean(errors <= 3.0)),
        "angle_acc_5deg": float(np.mean(errors <= 5.0)),
        "angular_calibration_nll": float(np.mean(nll)),
        "angle_within_1sigma": float(
            np.mean(
                np.asarray(
                    [
                        row.get("valid") is True
                        and float(row["angle_error_degrees"])
                        <= float(row["angle_std_degrees"])
                        for row in rows
                    ]
                )
            )
        ),
        "angle_within_2sigma": float(
            np.mean(
                np.asarray(
                    [
                        row.get("valid") is True
                        and float(row["angle_error_degrees"])
                        <= 2.0 * float(row["angle_std_degrees"])
                        for row in rows
                    ]
                )
            )
        ),
        "mean_angle_std_degrees": float(np.mean(stds)),
        "pivot_valid": len(pivot_valid),
        "pivot_coverage": len(pivot_valid) / len(rows),
        "invalid_pivot_error_fraction": math.sqrt(2.0),
        "pivot_mean_error_fraction": float(np.mean(pivots)),
        "pivot_median_error_fraction": float(np.median(pivots)),
        "successful_only_diagnostic": {
            "angle_mae_degrees": (
                None
                if not valid
                else float(
                    np.mean(
                        [
                            float(row["angle_error_degrees"])
                            for row in valid
                        ]
                    )
                )
            ),
            "pivot_mean_error_fraction": (
                None
                if not pivot_valid
                else float(
                    np.mean(
                        [
                            float(row["pivot_error_fraction"])
                            for row in pivot_valid
                        ]
                    )
                )
            ),
        },
        "primary_denominator_policy": (
            "all rows; invalid direction=180 degrees, invalid pivot=sqrt(2) "
            "image fraction, accuracy failures remain false"
        ),
    }


def _rows_for_decoder_view(
    rows: Sequence[Mapping[str, Any]],
    view: str,
) -> list[dict[str, Any]]:
    if view not in DECODER_VIEWS:
        raise ValueError(f"unsupported decoder view: {view}")
    selected: list[dict[str, Any]] = []
    for row in rows:
        views = row.get("decoder_views")
        if not isinstance(views, Mapping) or set(views) != set(DECODER_VIEWS):
            raise ValueError("grouped-validation row has incomplete decoder views")
        values = views.get(view)
        if not isinstance(values, Mapping):
            raise ValueError(f"grouped-validation row has no {view} view")
        projected = dict(row)
        projected.update(
            {
                "valid": values.get("valid"),
                "angle_error_degrees": values.get("angle_error_degrees"),
                "signed_angle_error_degrees": values.get(
                    "signed_angle_error_degrees"
                ),
            }
        )
        selected.append(projected)
    return selected


def _group_bootstrap_decoder_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    comparator: str,
    reference: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired group bootstrap of all-denominator decoder error differences."""

    if comparator not in DECODER_VIEWS or reference not in DECODER_VIEWS:
        raise ValueError("decoder contrast uses an unsupported view")
    if comparator == reference:
        raise ValueError("decoder contrast requires two different views")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        views = row.get("decoder_views")
        if not isinstance(views, Mapping):
            raise ValueError("decoder contrast row has no decoder views")
        left = views.get(comparator)
        right = views.get(reference)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError("decoder contrast row has incomplete views")
        left_error = (
            float(left["angle_error_degrees"])
            if left.get("valid") is True
            else 180.0
        )
        right_error = (
            float(right["angle_error_degrees"])
            if right.get("valid") is True
            else 180.0
        )
        grouped[str(row["group_id"])].append(left_error - right_error)
    if not grouped:
        raise ValueError("decoder contrast has no physical meter groups")
    group_deltas = np.asarray(
        [
            float(np.mean(values))
            for _, values in sorted(grouped.items())
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = generator.integers(
            0,
            len(group_deltas),
            size=len(group_deltas),
        )
        draws[index] = float(np.mean(group_deltas[selected]))
    return {
        "comparator": comparator,
        "reference": reference,
        "effect_definition": (
            f"{comparator} all-denominator angle MAE minus "
            f"{reference} all-denominator angle MAE"
        ),
        "positive_effect_favors": reference,
        "groups": len(group_deltas),
        "group_macro_effect_degrees": float(np.mean(group_deltas)),
        "group_macro_effect_degrees_95ci": _quantile_interval(draws),
        "group_bootstrap_iterations": int(iterations),
        "group_bootstrap_seed": int(seed),
        "resampling_unit": "physical_meter_group",
        "invalid_direction_error_degrees": 180.0,
    }


def _equivariance_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("paired equivariance residual has no samples")
    valid = [
        row for row in rows if row.get("equivariance_valid") is True
    ]
    direction = np.asarray(
        [
            float(row["equivariance_direction_residual_degrees"])
            for row in valid
        ],
        dtype=np.float64,
    )
    pivot = np.asarray(
        [
            float(row["equivariance_pivot_residual_fraction"])
            for row in valid
        ],
        dtype=np.float64,
    )
    return {
        "samples": len(rows),
        "valid_pairs": len(valid),
        "valid_pair_fraction": len(valid) / len(rows),
        "direction_residual_mean_degrees": (
            None if not valid else float(np.mean(direction))
        ),
        "direction_residual_median_degrees": (
            None if not valid else float(np.median(direction))
        ),
        "pivot_residual_mean_fraction": (
            None if not valid else float(np.mean(pivot))
        ),
        "pivot_residual_median_fraction": (
            None if not valid else float(np.median(pivot))
        ),
        "residual_definition": (
            "predict reference and deterministic homography pair; transform "
            "reference soft pivot/local ray analytically; compare paired "
            "prediction to transported prediction"
        ),
        "interpretation": (
            "diagnostic residual of projective-equivariance regularization; "
            "not proof of an exactly equivariant architecture"
        ),
    }


def _group_bootstrap_equivariance(
    rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    if not grouped:
        raise ValueError("paired equivariance bootstrap has no groups")
    group_ids = sorted(grouped)
    validity = np.asarray(
        [
            np.mean(
                [
                    row.get("equivariance_valid") is True
                    for row in grouped[group_id]
                ]
            )
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    direction = np.asarray(
        [
            np.mean(
                [
                    (
                        float(
                            row[
                                "equivariance_direction_residual_degrees"
                            ]
                        )
                        if row.get("equivariance_valid") is True
                        else 180.0
                    )
                    for row in grouped[group_id]
                ]
            )
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    pivot = np.asarray(
        [
            np.mean(
                [
                    (
                        float(
                            row["equivariance_pivot_residual_fraction"]
                        )
                        if row.get("equivariance_valid") is True
                        else math.sqrt(2.0)
                    )
                    for row in grouped[group_id]
                ]
            )
            for group_id in group_ids
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    validity_draws = np.empty(iterations, dtype=np.float64)
    direction_draws = np.empty(iterations, dtype=np.float64)
    pivot_draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = generator.integers(
            0,
            len(group_ids),
            size=len(group_ids),
        )
        validity_draws[index] = float(np.mean(validity[selected]))
        direction_draws[index] = float(np.mean(direction[selected]))
        pivot_draws[index] = float(np.mean(pivot[selected]))
    return {
        "groups": len(group_ids),
        "groups_with_valid_pairs": sum(
            any(
                row.get("equivariance_valid") is True
                for row in grouped[group_id]
            )
            for group_id in group_ids
        ),
        "group_bootstrap_iterations": int(iterations),
        "group_bootstrap_seed": int(seed),
        "resampling_unit": "physical_meter_group",
        "macro_valid_pair_fraction": float(np.mean(validity)),
        "macro_valid_pair_fraction_95ci": _quantile_interval(
            validity_draws
        ),
        "invalid_pair_penalty": {
            "direction_residual_degrees": 180.0,
            "pivot_residual_fraction": math.sqrt(2.0),
        },
        "macro_direction_residual_degrees_failure_penalized": float(
            np.mean(direction)
        ),
        "macro_direction_residual_degrees_failure_penalized_95ci": _quantile_interval(
            direction_draws
        ),
        "macro_pivot_residual_fraction_failure_penalized": float(
            np.mean(pivot)
        ),
        "macro_pivot_residual_fraction_failure_penalized_95ci": _quantile_interval(
            pivot_draws
        ),
    }


def _write_or_validate(path: Path, content: str) -> None:
    """Publish once: accept byte-identical reruns, reject every overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} exists with different content")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    arm = str(args.arm)
    seed = int(args.seed)
    condition = str(args.condition)
    if args.batch_size != 64:
        raise ValueError("formal grouped-validation batch size is fixed at 64")
    manifest = formal_manifest_path(args.manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    mechanism_cohort = _mechanism_cohort_gate()
    horizon_note = str(mechanism_cohort["horizon_note"])
    if arm == "full":
        cohort_path = COHORT_PATH
        cohort = _load_json(cohort_path)
        (
            run_dir,
            expected_summary_protocol,
            expected_verification_protocol,
            run_authority,
        ) = _authoritative_full_binding(
            cohort,
            mechanism_cohort,
            seed=seed,
        )
    else:
        cohort_path = MECHANISM_COHORT_PATH
        cohort = mechanism_cohort
        cohort_row = _cohort_run(cohort, seed=seed, arm=arm)
        if cohort_row.get("converged") is not True:
            raise ValueError(f"mechanism run {arm}:{seed} is not converged")
        expected_summary_protocol = PEPD_MECHANISM_RUN_PROTOCOL
        expected_verification_protocol = PEPD_MECHANISM_VERIFICATION_PROTOCOL
        run_dir = mechanism_output_dir(arm, seed)
        run_authority = "mechanism_phase2"
    if cohort.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("cohort scope unexpectedly authorizes forbidden evaluation")

    verification_path = run_dir / "verification.json"
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "best.pt"
    verification = _load_json(verification_path)
    summary = _load_json(summary_path)
    if (
        summary.get("protocol") != expected_summary_protocol
        or summary.get("status") != "complete"
        or int(summary.get("seed", -1)) != seed
    ):
        raise ValueError("PEPD run summary protocol/identity mismatch")
    if verification.get("protocol") != expected_verification_protocol:
        raise ValueError("PEPD run verification protocol mismatch")
    if verification.get("verified") is not True or verification.get("converged") is not True:
        raise ValueError("PEPD seed is not verified and converged")
    if verification.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("PEPD summary changed after verification")
    checkpoint_hash = sha256_file(checkpoint_path)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("PEPD checkpoint changed after verification")
    bound_row = (
        _cohort_run(cohort, seed=seed)
        if arm == "full"
        else _cohort_run(cohort, seed=seed, arm=arm)
    )
    if (
        bound_row.get("summary_sha256") != sha256_file(summary_path)
        or bound_row.get("best_checkpoint_sha256") != checkpoint_hash
        or bound_row.get("verification_sha256") != sha256_file(verification_path)
    ):
        raise ValueError("PEPD run artifacts drifted from their cohort binding")
    training_signature = summary["parent_training_signature"]

    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    _, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(training_signature["validation_fraction"]),
        seed=seed,
    )
    pin = formal_parent_pin(seed)
    if len(validation) != pin.validation_samples:
        raise ValueError("grouped-validation sample count drifted")
    if sample_ids_hash(validation) != pin.validation_ids_sha256:
        raise ValueError("grouped-validation sample identity drifted")
    dataset = GroupedValidationPerspectiveDataset(
        validation,
        image_size=int(training_signature["image_size"]),
        expansion=float(training_signature["expansion"]),
        condition=condition,
        degradation_seed=GROUPED_VAL_DEGRADATION_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal grouped-validation evaluation is CUDA-only")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "formal evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(training_signature["angle_bins"]),
        imagenet_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    rows: list[dict[str, Any]] = []
    stride = float(training_signature["image_size"]) / float(
        training_signature["heatmap_size"]
    )
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            reference_images = batch["reference_image"].to(
                device,
                non_blocking=True,
            )
            homography = batch["homography"].to(
                device,
                non_blocking=True,
            )
            targets = batch["target_direction"].to(device, non_blocking=True)
            target_pivots = batch["target_pivot"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=True):
                outputs = model(images)
                reference_outputs = model(reference_images)
            prediction = decode_probabilistic_pivot_direction(
                *(value.float() for value in outputs)
            )
            decoder_views = _decode_direction_views(outputs, prediction)
            reference_prediction = decode_probabilistic_pivot_direction(
                *(value.float() for value in reference_outputs)
            )
            reference_soft_pivot = (
                soft_pivot_coordinates(reference_outputs[0].float())
                * stride
            )
            paired_soft_pivot = (
                soft_pivot_coordinates(outputs[0].float()) * stride
            )
            expected_pivot, expected_direction, valid_h = (
                transform_pivot_direction(
                    reference_soft_pivot,
                    reference_prediction.direction,
                    homography,
                    ray_length=float(training_signature["image_size"])
                    * 0.25,
                )
            )
            equivariance_valid = (
                valid_h
                & reference_prediction.valid
                & prediction.valid
            )
            equivariance_direction = angular_error_degrees(
                prediction.direction,
                expected_direction,
            )
            equivariance_pivot = torch.linalg.vector_norm(
                paired_soft_pivot - expected_pivot,
                dim=1,
            ) / float(training_signature["image_size"])
            target_angle = torch.atan2(targets[:, 1], targets[:, 0])
            decoder_metrics: dict[
                str,
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ] = {}
            for view, (direction, valid) in decoder_views.items():
                predicted_angle = torch.atan2(
                    direction[:, 1],
                    direction[:, 0],
                )
                signed = circular_delta(predicted_angle, target_angle) * (
                    180.0 / math.pi
                )
                decoder_metrics[view] = (valid, torch.abs(signed), signed)
            fused_valid, fused_error, fused_signed = decoder_metrics["fused"]
            pivot_error = torch.linalg.vector_norm(
                prediction.pivot_xy * stride - target_pivots,
                dim=1,
            ) / float(training_signature["image_size"])
            pivot_valid = (
                torch.isfinite(prediction.pivot_xy).all(dim=1)
                & torch.isfinite(pivot_error)
            )
            for index, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "group_id": str(batch["group_id"][index]),
                        "valid": bool(fused_valid[index].item()),
                        "angle_error_degrees": float(
                            fused_error[index].item()
                        ),
                        "signed_angle_error_degrees": float(
                            fused_signed[index].item()
                        ),
                        "angle_std_degrees": float(
                            prediction.angle_std_degrees[index].item()
                        ),
                        "pivot_error_fraction": float(pivot_error[index].item()),
                        "pivot_valid": bool(pivot_valid[index].item()),
                        "equivariance_valid": bool(
                            equivariance_valid[index].item()
                        ),
                        "equivariance_direction_residual_degrees": (
                            float(equivariance_direction[index].item())
                            if bool(equivariance_valid[index].item())
                            else None
                        ),
                        "equivariance_pivot_residual_fraction": (
                            float(equivariance_pivot[index].item())
                            if bool(equivariance_valid[index].item())
                            else None
                        ),
                        "decoder_views": {
                            view: {
                                "valid": bool(
                                    values[0][index].item()
                                ),
                                "angle_error_degrees": float(
                                    values[1][index].item()
                                ),
                                "signed_angle_error_degrees": float(
                                    values[2][index].item()
                                ),
                            }
                            for view, values in decoder_metrics.items()
                        },
                    }
                )
    expected_ids = [sample.sample_id for sample in validation]
    actual_ids = [row["sample_id"] for row in rows]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("grouped-validation output identity/order mismatch")

    condition_index = GROUPED_VAL_CONDITIONS.index(condition)
    decoder_rows = {
        view: _rows_for_decoder_view(rows, view)
        for view in DECODER_VIEWS
    }
    decoder_view_metrics = {
        view: {
            "metrics": _summarize(decoder_rows[view]),
            "grouped_metrics": _group_bootstrap(
                decoder_rows[view],
                iterations=GROUPED_VAL_BOOTSTRAP_ITERATIONS,
                seed=GROUPED_VAL_BOOTSTRAP_SEED + condition_index,
            ),
        }
        for view in DECODER_VIEWS
    }
    decoder_contrasts = {
        f"{view}_minus_fused": _group_bootstrap_decoder_contrast(
            rows,
            comparator=view,
            reference="fused",
            iterations=GROUPED_VAL_BOOTSTRAP_ITERATIONS,
            seed=(
                GROUPED_VAL_BOOTSTRAP_SEED
                + 200
                + 10 * condition_index
                + DECODER_VIEWS.index(view)
            ),
        )
        for view in ("direct", "circular")
    }

    output_dir = run_dir / "grouped_validation"
    output_path = output_dir / f"{condition}.jsonl"
    payload = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    _write_or_validate(output_path, payload)
    summary_output = output_path.with_suffix(".summary.json")
    report = {
        "schema_version": 1,
        "protocol": PEPD_GROUPED_VAL_EVALUATION_PROTOCOL,
        "status": "complete",
        "scope": "SyncG official train grouped validation only",
        "arm": arm,
        "seed": seed,
        "condition": condition,
        "run_authority": run_authority,
        "horizon_note": horizon_note,
        "degradation_protocol": ROBUSTNESS_PROTOCOL,
        "degradation_seed": GROUPED_VAL_DEGRADATION_SEED,
        "dial_crop_before_degradation": True,
        "metrics": decoder_view_metrics["fused"]["metrics"],
        "grouped_metrics": decoder_view_metrics["fused"]["grouped_metrics"],
        "decoder_view_ablation": {
            "protocol": PEPD_DECODER_VIEW_PROTOCOL,
            "views": list(DECODER_VIEWS),
            "primary_view": "fused",
            "training_or_finetuning": False,
            "checkpoint_selection": False,
            "same_model_forward_and_logits": True,
            "shared_pivot_and_uncertainty_heads": True,
            "all_denominator_policy": (
                "invalid direction=180 degrees and accuracy=false"
            ),
            "per_view": decoder_view_metrics,
            "paired_group_contrasts_vs_fused": decoder_contrasts,
            "interpretation": (
                "fixed representation/decoder diagnostic on the retained "
                "checkpoint; never a post-hoc decoder selection budget"
            ),
        },
        "paired_equivariance_residual": {
            "metrics": _equivariance_summary(rows),
            "group_bootstrap": _group_bootstrap_equivariance(
                rows,
                iterations=GROUPED_VAL_BOOTSTRAP_ITERATIONS,
                seed=(
                    GROUPED_VAL_BOOTSTRAP_SEED
                    + 50
                    + condition_index
                ),
            ),
            "claim_boundary": (
                "projective-equivariance-regularized/trained; residual is "
                "empirical and does not establish an exact equivariant "
                "architecture"
            ),
        },
        "validation_samples": len(validation),
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "run_verification_sha256": sha256_file(verification_path),
        "cohort_sha256": sha256_file(cohort_path),
        "mechanism_cohort_sha256": sha256_file(MECHANISM_COHORT_PATH),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "evaluator": sha256_source_file(Path(__file__).resolve()),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "model": FORMAL_MODEL_SOURCE_SHA256,
            "degradation": sha256_source_file(
                PROJECT_ROOT / "experiments" / "robustness_degradations.py"
            ),
        },
        "public_test_field_evaluation": False,
    }
    report_payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_or_validate(summary_output, report_payload)
    print(report_payload, end="")
    print(summary_output)
    return summary_output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
