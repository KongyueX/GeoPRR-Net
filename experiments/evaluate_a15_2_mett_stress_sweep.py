"""Evaluate A15.2-METT on a paired 18-condition SyncG stress sweep.

Every condition is regenerated from the same ordered scene-holdout roster.  A
fresh SARN decision is made after applying each stressor, then the frozen raw
anchor, SARN endpoint, and METT posterior correction are evaluated together.
This is a stress-characterisation experiment: it performs no training,
adaptation, sample routing, or external-prediction fusion.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.a15_2_mett_stress_degradations import (
    PROTOCOL as STRESS_PROTOCOL,
    STRESS_SEED,
    STRESS_SPECS,
    StressSpec,
    apply_stress,
)
from experiments.evaluate_a15_2_mett_syncg import (
    _posterior_summary,
    posterior_batch_diagnostics,
)
from experiments.evaluate_a15_2_syncg_scene_holdout import (
    DEFAULT_LABELS,
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import IMAGE_SIZE
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.run_cagh_v5_plain_paper_batch import (
    ManifestRow,
    load_canonical_roi,
    load_manifest,
)
from experiments.summarize_support_normalized_cbam_pilot import load_scene_targets
from experiments.support_aware_roi_normalization_v2 import (
    normalize_support_aware_roi_v2,
)
from experiments.train_a15_2_mett import (
    METT_VARIANTS,
    load_mett_models,
    publication_model_identity,
    validate_mett_variant_metadata,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
    _normalized_raw_to_sarn_homography,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi
from experiments.a15_fteb import frozen_twin_endpoint_forward


PROTOCOL: Final[str] = "a15_2_mett_syncg_stress_sweep_v1"
METHODS: Final[tuple[str, ...]] = ("raw_anchor", "sarn_endpoint", "mett")
PERSPECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_15",
    "perspective_30",
    "perspective_45",
    "perspective_60",
)
EVALUATION_SEED: Final[int] = 20_260_818


class METTStressSweepError(ValueError):
    """The paired stress roster, model output, or summary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise METTStressSweepError(message)


class _StressDataset(Dataset[dict[str, Any]]):
    """Materialize one stress condition over an invariant ordered roster."""

    def __init__(
        self,
        rows: Sequence[ManifestRow],
        targets: Mapping[str, tuple[float, str]],
        *,
        spec: StressSpec,
    ) -> None:
        self.rows = tuple(rows)
        self.targets = targets
        self.spec = spec
        _require(bool(self.rows), "METT stress dataset is empty")
        _require(
            {row.sample_id for row in self.rows} == set(targets),
            "METT stress manifest and target rosters differ",
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.rows[int(index)]
        _payload, clean = load_canonical_roi(source)
        stressed, stress_metadata = apply_stress(
            clean,
            self.spec,
            sample_id=source.sample_id,
            seed=STRESS_SEED,
        )
        decision = normalize_support_aware_roi_v2(stressed)
        height, width = stressed.shape[:2]

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
            "condition_name": self.spec.name,
            "stress_sample_seed": int(stress_metadata["sample_seed"]),
            "original_view": normalized_rgb_tensor(
                direct_resize_whole_roi(stressed, size=IMAGE_SIZE)
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


def _metrics(errors: Sequence[float]) -> dict[str, float]:
    values = np.asarray(tuple(float(value) for value in errors), dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0, "stress metric rows are empty")
    _require(
        bool(np.isfinite(values).all()) and bool((values >= 0.0).all()),
        "stress metric errors are invalid",
    )
    return {
        "nmae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p95_absolute_error": float(np.quantile(values, 0.95)),
        "p99_absolute_error": float(np.quantile(values, 0.99)),
        "maximum_absolute_error": float(values.max()),
        "acc_at_1_percent": float(np.mean(values <= 0.01)),
        "acc_at_2_percent": float(np.mean(values <= 0.02)),
        "acc_at_5_percent": float(np.mean(values <= 0.05)),
    }


def _paired_delta_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, float]:
    deltas = np.asarray(
        [
            float(row[candidate]["absolute_error"])
            - float(row[comparator]["absolute_error"])
            for row in rows
        ],
        dtype=np.float64,
    )
    _require(deltas.size > 0, "paired stress comparison is empty")
    return {
        "mean_absolute_error_delta": float(deltas.mean()),
        "median_absolute_error_delta": float(np.median(deltas)),
        "p05_absolute_error_delta": float(np.quantile(deltas, 0.05)),
        "p95_absolute_error_delta": float(np.quantile(deltas, 0.95)),
        "improvement_fraction": float(np.mean(deltas < 0.0)),
        "negative_transfer_fraction": float(np.mean(deltas > 0.0)),
        "tie_fraction": float(np.mean(deltas == 0.0)),
    }


def _validate_cartesian_roster(
    rows: Sequence[Mapping[str, Any]],
    *,
    ordered_sample_ids: Sequence[str],
    condition_names: Sequence[str],
) -> None:
    expected = [
        (str(sample_id), str(condition))
        for condition in condition_names
        for sample_id in ordered_sample_ids
    ]
    observed = [
        (str(row["sample_id"]), str(row["condition"])) for row in rows
    ]
    _require(
        observed == expected,
        "METT stress rows are not the ordered sample-by-condition Cartesian roster",
    )


def _relative_to_clean(
    condition_metrics: Mapping[str, float],
    clean_metrics: Mapping[str, float],
) -> dict[str, float | None]:
    condition_nmae = float(condition_metrics["nmae"])
    clean_nmae = float(clean_metrics["nmae"])
    condition_rmse = float(condition_metrics["rmse"])
    clean_rmse = float(clean_metrics["rmse"])
    return {
        "nmae_absolute_change": condition_nmae - clean_nmae,
        "nmae_ratio": condition_nmae / clean_nmae if clean_nmae > 0.0 else None,
        "nmae_relative_degradation_fraction": (
            (condition_nmae - clean_nmae) / clean_nmae
            if clean_nmae > 0.0
            else None
        ),
        "rmse_absolute_change": condition_rmse - clean_rmse,
        "rmse_ratio": condition_rmse / clean_rmse if clean_rmse > 0.0 else None,
    }


def summarize_conditions(
    rows: Sequence[Mapping[str, Any]],
    *,
    specs: Mapping[str, StressSpec] = STRESS_SPECS,
) -> dict[str, Any]:
    """Summarize metrics and paired transfer for every stress condition."""

    _require("clean" in specs, "stress specs omit the clean condition")
    by_condition = {
        name: [row for row in rows if str(row["condition"]) == name]
        for name in specs
    }
    for name, subset in by_condition.items():
        _require(bool(subset), f"stress condition is empty: {name}")

    metrics_by_condition = {
        name: {
            method: _metrics(
                [float(row[method]["absolute_error"]) for row in subset]
            )
            for method in METHODS
        }
        for name, subset in by_condition.items()
    }
    clean_metrics = metrics_by_condition["clean"]
    result: dict[str, Any] = {}
    for name, spec in specs.items():
        subset = by_condition[name]
        scenes = sorted({str(row["scene_stem"]) for row in subset})
        result[name] = {
            "spec": asdict(spec),
            "rows": len(subset),
            "scenes": len(scenes),
            "metrics": metrics_by_condition[name],
            "relative_to_clean": {
                method: _relative_to_clean(
                    metrics_by_condition[name][method], clean_metrics[method]
                )
                for method in METHODS
            },
            "mett_vs_raw_anchor": _paired_delta_summary(
                subset, candidate="mett", comparator="raw_anchor"
            ),
            "mett_vs_sarn_endpoint": _paired_delta_summary(
                subset, candidate="mett", comparator="sarn_endpoint"
            ),
            "macro_scene_nmae": {
                method: statistics.fmean(
                    statistics.fmean(
                        float(row[method]["absolute_error"])
                        for row in subset
                        if str(row["scene_stem"]) == scene
                    )
                    for scene in scenes
                )
                for method in METHODS
            },
            "posterior": {
                method: _posterior_summary(subset, method=method)
                for method in METHODS
            },
            "relation_available_fraction": statistics.fmean(
                float(bool(row["relation_available"])) for row in subset
            ),
            "sarn_applied_fraction": statistics.fmean(
                float(bool(row["sarn_applied"])) for row in subset
            ),
        }
    return result


def summarize_family_curves(
    condition_summary: Mapping[str, Mapping[str, Any]],
    *,
    specs: Mapping[str, StressSpec] = STRESS_SPECS,
) -> dict[str, Any]:
    """Build paired clean-to-stress curves for each degradation family."""

    clean = condition_summary["clean"]
    families = sorted({spec.family for spec in specs.values()} - {"clean"})
    result: dict[str, Any] = {}
    for family in families:
        family_specs = sorted(
            (spec for spec in specs.values() if spec.family == family),
            key=lambda spec: (float(spec.severity), spec.name),
        )
        methods: dict[str, Any] = {}
        for method in METHODS:
            clean_nmae = float(clean["metrics"][method]["nmae"])
            points = [
                {
                    "condition": "clean",
                    "severity": 0.0,
                    "nmae": clean_nmae,
                    "nmae_change_from_clean": 0.0,
                }
            ]
            points.extend(
                {
                    "condition": spec.name,
                    "severity": float(spec.severity),
                    "nmae": float(
                        condition_summary[spec.name]["metrics"][method]["nmae"]
                    ),
                    "nmae_change_from_clean": float(
                        condition_summary[spec.name]["metrics"][method]["nmae"]
                    )
                    - clean_nmae,
                }
                for spec in family_specs
            )
            stressed = points[1:]
            methods[method] = {
                "points": points,
                "mean_stressed_nmae": statistics.fmean(
                    float(point["nmae"]) for point in stressed
                ),
                "worst_stressed_nmae": max(
                    float(point["nmae"]) for point in stressed
                ),
                "worst_condition": max(
                    stressed, key=lambda point: float(point["nmae"])
                )["condition"],
            }
        result[family] = {
            "conditions": [spec.name for spec in family_specs],
            "methods": methods,
        }
    return result


def summarize_perspective_curve(
    condition_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report the 0/15/30/45/60 degree curve, normalized AUC, and slope."""

    conditions = ("clean", *PERSPECTIVE_CONDITIONS)
    degrees = np.asarray((0.0, 15.0, 30.0, 45.0, 60.0), dtype=np.float64)
    result: dict[str, Any] = {
        "conditions": list(conditions),
        "degrees": degrees.tolist(),
        "methods": {},
    }
    for method in METHODS:
        nmae = np.asarray(
            [
                float(condition_summary[name]["metrics"][method]["nmae"])
                for name in conditions
            ],
            dtype=np.float64,
        )
        clean = float(nmae[0])
        slope, intercept = np.polyfit(degrees, nmae, deg=1)
        result["methods"][method] = {
            "points": [
                {"condition": name, "degrees": float(x), "nmae": float(y)}
                for name, x, y in zip(conditions, degrees, nmae, strict=True)
            ],
            "normalized_nmae_auc_0_to_60": float(
                np.trapz(nmae, degrees) / 60.0
            ),
            "normalized_excess_nmae_auc_0_to_60": float(
                np.trapz(nmae - clean, degrees) / 60.0
            ),
            "least_squares_slope_nmae_per_degree": float(slope),
            "least_squares_intercept": float(intercept),
            "endpoint_nmae_change_from_clean": float(nmae[-1] - clean),
        }
    return result


def evaluate_stress_sweep(
    *,
    manifest_path: Path,
    labels_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
    expected_variant: str = "full",
) -> dict[str, Any]:
    _require(workers >= 0 and batch_size >= 1, "stress evaluation sizes differ")
    _require(expected_variant in METT_VARIANTS, "unknown METT experiment variant")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"stress evaluation output exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "stress device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")

    torch.manual_seed(EVALUATION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(EVALUATION_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    targets = load_scene_targets(labels_path, split_path)
    ordered_sample_ids = tuple(targets)
    manifest_rows = load_manifest(manifest_path)
    by_id = {row.sample_id: row for row in manifest_rows}
    _require(
        len(by_id) == len(manifest_rows), "stress manifest sample IDs repeat"
    )
    _require(
        set(by_id) == set(ordered_sample_ids),
        "stress manifest and target rosters differ",
    )
    ordered_manifest = tuple(by_id[sample_id] for sample_id in ordered_sample_ids)

    anchor, correction, model_metadata = load_mett_models(
        checkpoint_path, device=device
    )
    validated_variant = validate_mett_variant_metadata(
        model_metadata, expected_variant=expected_variant
    )
    model_metadata = {
        **model_metadata,
        "validated_mett_variant": validated_variant,
    }
    if expected_variant == "full":
        model_metadata["validated_full_mett"] = validated_variant
    anchor.eval()
    correction.eval()
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, spec in enumerate(STRESS_SPECS.values()):
            dataset = _StressDataset(ordered_manifest, targets, spec=spec)
            loader = _loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=EVALUATION_SEED + condition_index,
                cuda=device.type == "cuda",
            )
            condition_ids: list[str] = []
            for raw_batch in loader:
                original = raw_batch["original_view"].to(device)
                sarn = raw_batch["sarn_view"].to(device)
                support = raw_batch["sarn_support_mask"].to(device)
                active = raw_batch["sarn_active"].to(device).bool()
                homography = raw_batch["raw_to_sarn_homography"].to(device)
                target = raw_batch["target"].float()
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                    prediction = forward_a15_correction(
                        correction,
                        {
                            "raw_posterior": endpoints["raw_posterior"],
                            "sarn_posterior": endpoints["sarn_posterior"],
                            "raw_mean": endpoints["raw_mean"],
                            "sarn_mean": endpoints["sarn_mean"],
                            "raw_features": endpoints["raw_features"],
                            "sarn_features": endpoints["sarn_features"],
                            "sarn_support_mask": support,
                            "sarn_active": active,
                            "raw_to_sarn_homography": homography,
                        },
                        endpoint_null=False,
                    )
                values = {
                    "raw_anchor": prediction["raw_anchor_mean"].float().cpu(),
                    "sarn_endpoint": prediction["sarn_endpoint_mean"].float().cpu(),
                    "mett": prediction["mean"].float().cpu(),
                }
                posterior_values = {
                    "raw_anchor": prediction["raw_anchor_posterior"],
                    "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                    "mett": prediction["progress_posterior"],
                }
                diagnostics = {
                    method: posterior_batch_diagnostics(posterior, target)
                    for method, posterior in posterior_values.items()
                }
                relation = prediction["relation_available"].bool().cpu()
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                scenes = tuple(str(value) for value in raw_batch["scene_stem"])
                seeds = raw_batch["stress_sample_seed"].long().cpu()
                active_cpu = active.cpu()
                condition_ids.extend(ids)
                for row_index, sample_id in enumerate(ids):
                    expected_target, expected_scene = targets[sample_id]
                    observed_target = float(target[row_index])
                    _require(
                        abs(expected_target - observed_target) <= 1.0e-6,
                        "stress regenerated target differs",
                    )
                    _require(
                        scenes[row_index] == expected_scene,
                        "stress regenerated scene differs",
                    )
                    row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "scene_stem": expected_scene,
                        "condition": spec.name,
                        "family": spec.family,
                        "severity": float(spec.severity),
                        "normalized_target": float(expected_target),
                        "stress_sample_seed": int(seeds[row_index]),
                        "sarn_applied": bool(active_cpu[row_index]),
                        "relation_available": bool(relation[row_index]),
                    }
                    for method, tensor in values.items():
                        scalar = float(tensor[row_index])
                        _require(
                            math.isfinite(scalar) and 0.0 <= scalar <= 1.0,
                            "stress prediction is invalid",
                        )
                        row[method] = {
                            "prediction": scalar,
                            "absolute_error": abs(scalar - expected_target),
                            "posterior": {
                                name: (
                                    bool(value[row_index])
                                    if value.dtype == torch.bool
                                    else float(value[row_index])
                                )
                                for name, value in diagnostics[method].items()
                            },
                        }
                    rows.append(row)
            _require(
                tuple(condition_ids) == ordered_sample_ids,
                f"stress condition roster order differs: {spec.name}",
            )

    condition_names = tuple(STRESS_SPECS)
    _validate_cartesian_roster(
        rows,
        ordered_sample_ids=ordered_sample_ids,
        condition_names=condition_names,
    )
    condition_summary = summarize_conditions(rows)
    family_summary = summarize_family_curves(condition_summary)
    perspective_summary = summarize_perspective_curve(condition_summary)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "development_stress_characterisation": True,
            "historical_syncg_scene_holdout": True,
            "training_or_adaptation_during_evaluation": False,
            "sample_router_or_external_prediction_fusion": False,
            "fresh_sarn_after_every_stress": True,
            "paired_clean_and_stress_roster": True,
            "experiment_variant": expected_variant,
            "inference_precision": (
                str(autocast_dtype).removeprefix("torch.")
                if autocast_enabled
                else "float32"
            ),
        },
        "publication_model": publication_model_identity(expected_variant),
        "model": model_metadata,
        "data": {
            "manifest": str(Path(manifest_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "split": str(Path(split_path).resolve()),
            "samples": len(ordered_sample_ids),
            "scenes": len({scene for _target, scene in targets.values()}),
            "conditions": list(condition_names),
            "condition_count": len(condition_names),
            "rows": len(rows),
            "stress_protocol": STRESS_PROTOCOL,
            "stress_seed": STRESS_SEED,
            "stress_specs": {
                name: asdict(spec) for name, spec in STRESS_SPECS.items()
            },
        },
        "summary": {
            "conditions": condition_summary,
            "families": family_summary,
            "perspective_continuous_curve": perspective_summary,
        },
        "per_sample_condition": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--experiment-variant", choices=tuple(METT_VARIANTS), default="full"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use CUDA BF16/FP16 instead of the default FP32 evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_stress_sweep(
        manifest_path=args.manifest,
        labels_path=args.labels,
        split_path=args.split,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
        expected_variant=args.experiment_variant,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "conditions": result["data"]["condition_count"],
                "rows": result["data"]["rows"],
                "perspective": result["summary"]["perspective_continuous_curve"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_SEED",
    "METTStressSweepError",
    "METHODS",
    "PERSPECTIVE_CONDITIONS",
    "PROTOCOL",
    "build_argument_parser",
    "evaluate_stress_sweep",
    "summarize_conditions",
    "summarize_family_curves",
    "summarize_perspective_curve",
]
