"""Train-only monotone residual calibration for the frozen Raw endpoint.

The probe collects clean/moderate-blur/severe-blur predictions from the fixed
inner training population, fits a shared seven-knot monotone piecewise-linear
map by exact L1 linear programming, and evaluates it once on the untouched
inner-dev scenes.  Projection conditions and the formal holdout are not read.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack
from torch.utils.data import DataLoader

from experiments.raw_angular_moment_refiner_probe import (
    DEFAULT_ENDPOINT_CONTROL_CHECKPOINT,
    load_endpoint_controls,
)
from experiments.raw_multiscale_progress_probe import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_REFINER_EPOCHS,
    DEFAULT_SEED,
    DEFAULT_WORKERS,
    load_inner_scene_population,
)
from experiments.raw_relative_angular_frame_probe import (
    DEFAULT_FOUNDATION_CHECKPOINT,
    RelativeAngularFrameProbeError,
    RelativeFrameConditionedDataset,
    build_relative_frame_probe_from_foundation,
)
from experiments.raw_context_endpoint_control_probe import load_inner_foundation
from experiments.resnet18_direct_progress import (
    DirectProgressError,
    DirectSample,
    _configure_reproducibility,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS
from experiments.syncg_lightweight_regression_baselines import DEFAULT_SCENE_SPLIT


PROTOCOL: Final[str] = "syncg_raw_monotone_residual_calibration_probe_v1"
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/raw_monotone_residual_calibration_probe/seed_20262020"
)
RAW_CONDITIONS: Final[tuple[str, ...]] = tuple(CONDITIONS[:3])
DEFAULT_KNOT_COUNT: Final[int] = 7
METHOD_IDENTITY: Final[str] = "linear_refresh"
METHOD_MEDIAN_BIAS: Final[str] = "train_median_bias"
METHOD_MONOTONE_SPLINE: Final[str] = "train_monotone_spline_7"
METHODS: Final[tuple[str, ...]] = (
    METHOD_IDENTITY,
    METHOD_MEDIAN_BIAS,
    METHOD_MONOTONE_SPLINE,
)


class MonotoneCalibrationError(ValueError):
    """The calibration population, optimization, or artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MonotoneCalibrationError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )


@dataclass(frozen=True, slots=True)
class MonotoneLinearSpline:
    knots_x: np.ndarray
    knots_y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.knots_x, dtype=np.float64)
        y = np.asarray(self.knots_y, dtype=np.float64)
        _require(
            x.ndim == y.ndim == 1
            and len(x) == len(y)
            and len(x) >= 2
            and bool(np.all(np.isfinite(x)))
            and bool(np.all(np.isfinite(y)))
            and bool(np.all(np.diff(x) > 0.0))
            and bool(np.all(np.diff(y) >= -1.0e-10))
            and float(y.min()) >= -1.0e-10
            and float(y.max()) <= 1.0 + 1.0e-10,
            "monotone spline knots are invalid",
        )
        object.__setattr__(self, "knots_x", x)
        object.__setattr__(self, "knots_y", y)

    def predict(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        _require(bool(np.all(np.isfinite(source))), "spline input is non-finite")
        return np.interp(source, self.knots_x, self.knots_y)

    def as_dict(self) -> dict[str, Any]:
        return {
            "knots_x": self.knots_x.tolist(),
            "knots_y": self.knots_y.tolist(),
            "monotone": bool(np.all(np.diff(self.knots_y) >= -1.0e-10)),
        }


def _linear_interpolation_design(
    values: np.ndarray,
    knots_x: np.ndarray,
) -> csr_matrix:
    source = np.asarray(values, dtype=np.float64)
    knots = np.asarray(knots_x, dtype=np.float64)
    _require(
        source.ndim == 1
        and knots.ndim == 1
        and len(knots) >= 2
        and bool(np.all(np.diff(knots) > 0.0)),
        "interpolation design inputs differ",
    )
    clipped = np.clip(source, knots[0], knots[-1])
    lower = np.searchsorted(knots, clipped, side="right") - 1
    lower = np.clip(lower, 0, len(knots) - 2)
    upper = lower + 1
    widths = knots[upper] - knots[lower]
    alpha = np.divide(
        clipped - knots[lower],
        widths,
        out=np.zeros_like(clipped),
        where=widths > 0.0,
    )
    row_indices = np.repeat(np.arange(len(source), dtype=np.int64), 2)
    column_indices = np.stack((lower, upper), axis=1).reshape(-1)
    data = np.stack((1.0 - alpha, alpha), axis=1).reshape(-1)
    return csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(source), len(knots)),
        dtype=np.float64,
    )


def fit_monotone_l1_spline(
    predictions: np.ndarray | Sequence[float],
    targets: np.ndarray | Sequence[float],
    *,
    knot_count: int = DEFAULT_KNOT_COUNT,
) -> tuple[MonotoneLinearSpline, dict[str, Any]]:
    """Solve the bounded monotone piecewise-linear exact-L1 problem globally."""

    source = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    _require(
        source.ndim == expected.ndim == 1
        and len(source) == len(expected)
        and len(source) >= knot_count
        and knot_count >= 3
        and bool(np.all(np.isfinite(source)))
        and bool(np.all(np.isfinite(expected)))
        and float(source.min()) >= 0.0
        and float(source.max()) <= 1.0
        and float(expected.min()) >= 0.0
        and float(expected.max()) <= 1.0,
        "calibration fit inputs differ",
    )
    quantiles = np.linspace(0.0, 1.0, knot_count)
    knots_x = np.quantile(source, quantiles)
    if np.any(np.diff(knots_x) <= 1.0e-8):
        knots_x = np.linspace(float(source.min()), float(source.max()), knot_count)
    _require(
        float(knots_x[-1] - knots_x[0]) > 1.0e-8,
        "calibration predictions collapsed",
    )
    design = _linear_interpolation_design(source, knots_x)
    count = len(source)
    identity = eye(count, format="csr", dtype=np.float64)
    monotone = np.zeros((knot_count - 1, knot_count), dtype=np.float64)
    for index in range(knot_count - 1):
        monotone[index, index] = 1.0
        monotone[index, index + 1] = -1.0
    constraints = vstack(
        (
            hstack((design, -identity), format="csr"),
            hstack((-design, -identity), format="csr"),
            hstack(
                (
                    csr_matrix(monotone),
                    csr_matrix((knot_count - 1, count), dtype=np.float64),
                ),
                format="csr",
            ),
        ),
        format="csr",
    )
    bounds = [(0.0, 1.0)] * knot_count + [(0.0, None)] * count
    objective = np.concatenate(
        (np.zeros(knot_count, dtype=np.float64), np.ones(count) / count)
    )
    upper_bounds = np.concatenate(
        (expected, -expected, np.zeros(knot_count - 1, dtype=np.float64))
    )
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=upper_bounds,
        bounds=bounds,
        method="highs",
    )
    _require(bool(result.success), f"monotone L1 solver failed: {result.message}")
    spline = MonotoneLinearSpline(knots_x, result.x[:knot_count])
    fitted = spline.predict(source)
    exact_l1 = float(np.mean(np.abs(fitted - expected)))
    _require(
        math.isfinite(exact_l1)
        and abs(exact_l1 - float(result.fun)) <= 1.0e-7,
        "solver objective and reconstructed L1 differ",
    )
    return spline, {
        "solver": "scipy.optimize.linprog_highs",
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective_exact_l1": exact_l1,
        "knot_count": int(knot_count),
        "knot_placement": "pooled_train_prediction_quantiles",
    }


def fit_median_bias(
    predictions: np.ndarray | Sequence[float],
    targets: np.ndarray | Sequence[float],
) -> float:
    source = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    _require(
        source.shape == expected.shape
        and source.ndim == 1
        and len(source) >= 1
        and bool(np.all(np.isfinite(source)))
        and bool(np.all(np.isfinite(expected))),
        "median-bias inputs differ",
    )
    return float(np.median(expected - source))


def apply_median_bias(values: np.ndarray, bias: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64) + float(bias), 0.0, 1.0)


def collect_endpoint_predictions(
    model: torch.nn.Module,
    samples: Sequence[DirectSample],
    *,
    population: str,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    dataset = RelativeFrameConditionedDataset(samples, condition=condition)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(seed),
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(
                device, non_blocking=device.type == "cuda"
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                representation = model.encoder(images)["representation"]
                prediction = torch.sigmoid(
                    model.linear_refresh(representation).squeeze(1)
                )
            predictions = prediction.float().cpu().tolist()
            targets = batch["progress"].float().tolist()
            for sample_id, scene_stem, target, value in zip(
                batch["sample_id"],
                batch["scene_stem"],
                targets,
                predictions,
                strict=True,
            ):
                _require(
                    math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0,
                    "endpoint prediction is invalid",
                )
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "scene_stem": str(scene_stem),
                        "condition": condition,
                        "target": float(target),
                        "linear_refresh": float(value),
                    }
                )
            if (batch_index + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "endpoint_replay_progress",
                            "population": population,
                            "condition": condition,
                            "batches": batch_index + 1,
                            "rows": len(rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    _require(len(rows) == len(samples), "endpoint replay row count differs")
    return tuple(rows)


def _bootstrap_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    candidate_error = np.abs(
        np.asarray(
            [float(row["predictions"][candidate]) for row in rows],
            dtype=np.float64,
        )
        - targets
    )
    reference_error = np.abs(
        np.asarray(
            [float(row["predictions"][METHOD_IDENTITY]) for row in rows],
            dtype=np.float64,
        )
        - targets
    )
    delta = candidate_error - reference_error
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_scene[str(row["scene_stem"])].append(index)
    scenes = tuple(sorted(by_scene))
    _require(len(scenes) >= 2, "calibration bootstrap needs multiple scenes")
    indices = tuple(
        np.asarray(by_scene[scene], dtype=np.int64) for scene in scenes
    )
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(scenes), size=(replicates, len(scenes)))
    bootstrap = np.empty(replicates, dtype=np.float64)
    for replicate, chosen in enumerate(selected):
        sampled_rows = np.concatenate(
            tuple(indices[int(scene_index)] for scene_index in chosen)
        )
        bootstrap[replicate] = float(delta[sampled_rows].mean())
    return {
        "candidate": candidate,
        "reference": METHOD_IDENTITY,
        "mean_nmae_delta": float(delta.mean()),
        "relative_error_reduction": float(-delta.mean() / reference_error.mean()),
        "scene_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "scene_bootstrap_probability_delta_below_zero": float(
            np.mean(bootstrap < 0.0)
        ),
        "paired_sample_wins": int(np.sum(candidate_error < reference_error)),
        "paired_sample_losses": int(np.sum(candidate_error > reference_error)),
        "paired_sample_ties": int(np.sum(candidate_error == reference_error)),
        "scenes": len(scenes),
        "replicates": int(replicates),
    }


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _require(bool(rows), "calibration evaluation is empty")
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    nmae = {
        method: float(
            np.mean(
                np.abs(
                    np.asarray(
                        [float(row["predictions"][method]) for row in rows],
                        dtype=np.float64,
                    )
                    - targets
                )
            )
        )
        for method in METHODS
    }
    return {
        "rows": len(rows),
        "scenes": len({str(row["scene_stem"]) for row in rows}),
        "nmae": nmae,
        "comparisons": {
            method: _bootstrap_comparison(
                rows,
                candidate=method,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + method_index,
            )
            for method_index, method in enumerate(
                (METHOD_MEDIAN_BIAS, METHOD_MONOTONE_SPLINE)
            )
        },
    }


def _residual_deciles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    predictions = np.asarray(
        [float(row["predictions"][METHOD_IDENTITY]) for row in rows],
        dtype=np.float64,
    )
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    edges = np.quantile(predictions, np.linspace(0.0, 1.0, 11))
    groups = np.searchsorted(edges[1:-1], predictions, side="right")
    result: list[dict[str, Any]] = []
    for index in range(10):
        selected = groups == index
        if not np.any(selected):
            continue
        result.append(
            {
                "decile": index + 1,
                "rows": int(np.sum(selected)),
                "prediction_mean": float(predictions[selected].mean()),
                "target_mean": float(targets[selected].mean()),
                "signed_residual_mean_target_minus_prediction": float(
                    (targets[selected] - predictions[selected]).mean()
                ),
            }
        )
    return result


def run_probe(
    *,
    foundation_checkpoint_path: Path,
    endpoint_control_checkpoint_path: Path,
    fit_manifest_path: Path,
    outer_split_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    device_name: str = "cuda:0",
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    knot_count: int = DEFAULT_KNOT_COUNT,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    _require(
        batch_size >= 1
        and workers >= 0
        and knot_count >= 3
        and bootstrap_replicates >= 1,
        "invalid calibration probe configuration",
    )
    root = Path(output_dir).resolve()
    paths = {
        "fit_replay": root / "inner_train_fit_predictions.pt",
        "calibrator": root / "calibrator.json",
        "predictions": root / "inner_dev_calibrated_predictions.jsonl",
        "results": root / "results.json",
    }
    _require(
        not any(path.exists() for path in paths.values()),
        "calibration output artifact already exists",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(seed, device)
    internal = load_inner_scene_population(fit_manifest_path, outer_split_path)
    foundation, foundation_metadata = load_inner_foundation(
        foundation_checkpoint_path,
        internal=internal,
        expected_seed=seed,
    )
    model = build_relative_frame_probe_from_foundation(foundation)
    del foundation
    endpoint_metadata = load_endpoint_controls(
        endpoint_control_checkpoint_path,
        model=model,
        expected_seed=seed,
        expected_epochs=DEFAULT_REFINER_EPOCHS,
        foundation_metadata=foundation_metadata,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.to(device).eval()
    started = time.perf_counter()
    fit_rows: list[dict[str, Any]] = []
    dev_source_rows: list[dict[str, Any]] = []
    for population_index, (population, samples) in enumerate(
        (("inner_train", internal.train), ("inner_dev", internal.dev))
    ):
        for condition_index, condition in enumerate(RAW_CONDITIONS):
            condition_started = time.perf_counter()
            collected = collect_endpoint_predictions(
                model,
                samples,
                population=population,
                condition=condition,
                device=device,
                batch_size=batch_size,
                workers=workers,
                seed=seed + population_index * 100 + condition_index,
            )
            if population == "inner_train":
                fit_rows.extend(collected)
            else:
                dev_source_rows.extend(collected)
            condition_nmae = float(
                np.mean(
                    [
                        abs(float(row["linear_refresh"]) - float(row["target"]))
                        for row in collected
                    ]
                )
            )
            print(
                json.dumps(
                    {
                        "phase": "endpoint_replay_complete",
                        "population": population,
                        "condition": condition,
                        "rows": len(collected),
                        "nmae": condition_nmae,
                        "elapsed_seconds": time.perf_counter() - condition_started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    expected_fit_rows = len(internal.train) * len(RAW_CONDITIONS)
    expected_dev_rows = len(internal.dev) * len(RAW_CONDITIONS)
    _require(len(fit_rows) == expected_fit_rows, "fit replay size differs")
    _require(len(dev_source_rows) == expected_dev_rows, "dev replay size differs")
    fit_predictions = np.asarray(
        [float(row["linear_refresh"]) for row in fit_rows], dtype=np.float64
    )
    fit_targets = np.asarray(
        [float(row["target"]) for row in fit_rows], dtype=np.float64
    )
    median_bias = fit_median_bias(fit_predictions, fit_targets)
    spline, solver = fit_monotone_l1_spline(
        fit_predictions,
        fit_targets,
        knot_count=knot_count,
    )
    fit_methods = {
        METHOD_IDENTITY: fit_predictions,
        METHOD_MEDIAN_BIAS: apply_median_bias(fit_predictions, median_bias),
        METHOD_MONOTONE_SPLINE: spline.predict(fit_predictions),
    }
    fit_nmae = {
        method: float(np.mean(np.abs(values - fit_targets)))
        for method, values in fit_methods.items()
    }
    dev_predictions = np.asarray(
        [float(row["linear_refresh"]) for row in dev_source_rows], dtype=np.float64
    )
    calibrated = {
        METHOD_IDENTITY: dev_predictions,
        METHOD_MEDIAN_BIAS: apply_median_bias(dev_predictions, median_bias),
        METHOD_MONOTONE_SPLINE: spline.predict(dev_predictions),
    }
    dev_rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(dev_source_rows):
        dev_rows.append(
            {
                "sample_id": source_row["sample_id"],
                "scene_stem": source_row["scene_stem"],
                "condition": source_row["condition"],
                "target": float(source_row["target"]),
                "predictions": {
                    method: float(values[index])
                    for method, values in calibrated.items()
                },
            }
        )
    raw_pooled = _summarize_rows(
        dev_rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 50_000,
    )
    by_condition = {
        condition: _summarize_rows(
            [row for row in dev_rows if row["condition"] == condition],
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=seed + 60_000 + condition_index,
        )
        for condition_index, condition in enumerate(RAW_CONDITIONS)
    }
    calibrator = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "fit_population": "inner_train_clean_blur_pooled",
        "fit_rows": len(fit_rows),
        "fit_scenes": len(internal.train_scenes),
        "source_method": METHOD_IDENTITY,
        "median_bias": median_bias,
        "monotone_spline": spline.as_dict(),
        "solver": solver,
        "fit_nmae": fit_nmae,
    }
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "predictions": torch.from_numpy(fit_predictions).float(),
            "targets": torch.from_numpy(fit_targets).float(),
            "sample_ids": tuple(str(row["sample_id"]) for row in fit_rows),
            "scene_stems": tuple(str(row["scene_stem"]) for row in fit_rows),
            "conditions": tuple(str(row["condition"]) for row in fit_rows),
        },
        paths["fit_replay"],
    )
    _write_json(paths["calibrator"], calibrator)
    _write_jsonl(paths["predictions"], dev_rows)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": int(seed),
        "scientific_question": (
            "Does a train-only bounded monotone exact-L1 mapping remove stable "
            "Raw endpoint bias on unseen inner-dev scenes?"
        ),
        "data": {
            "fit_manifest": str(Path(fit_manifest_path).resolve()),
            "outer_split": str(Path(outer_split_path).resolve()),
            "inner_train_samples": len(internal.train),
            "inner_train_scenes": len(internal.train_scenes),
            "inner_dev_samples": len(internal.dev),
            "inner_dev_scenes": len(internal.dev_scenes),
            "fit_conditions": RAW_CONDITIONS,
            "evaluation_conditions": RAW_CONDITIONS,
            "inner_dev_labels_used_for_fit": False,
            "formal_holdout_accessed": False,
        },
        "source": {
            "foundation_checkpoint": str(Path(foundation_checkpoint_path).resolve()),
            "endpoint_control": endpoint_metadata,
            "source_method": METHOD_IDENTITY,
        },
        "calibrator": calibrator,
        "inner_dev": {
            "by_condition": by_condition,
            "raw_pooled": raw_pooled,
            "identity_residual_deciles": _residual_deciles(dev_rows),
        },
        "projection_path_modified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }
    _write_json(paths["results"], result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--foundation-checkpoint", type=Path, default=DEFAULT_FOUNDATION_CHECKPOINT
    )
    parser.add_argument(
        "--endpoint-control-checkpoint",
        type=Path,
        default=DEFAULT_ENDPOINT_CONTROL_CHECKPOINT,
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--outer-split", type=Path, default=DEFAULT_SCENE_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--knot-count", type=int, default=DEFAULT_KNOT_COUNT)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_probe(
            foundation_checkpoint_path=args.foundation_checkpoint,
            endpoint_control_checkpoint_path=args.endpoint_control_checkpoint,
            fit_manifest_path=args.fit_manifest,
            outer_split_path=args.outer_split,
            output_dir=args.output_dir,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            knot_count=args.knot_count,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (
        MonotoneCalibrationError,
        RelativeAngularFrameProbeError,
        DirectProgressError,
    ) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "results": result["artifacts"]["results"],
                "fit_nmae": result["calibrator"]["fit_nmae"],
                "inner_dev_raw_pooled": result["inner_dev"]["raw_pooled"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "METHOD_IDENTITY",
    "METHOD_MEDIAN_BIAS",
    "METHOD_MONOTONE_SPLINE",
    "MonotoneLinearSpline",
    "apply_median_bias",
    "fit_median_bias",
    "fit_monotone_l1_spline",
    "run_probe",
]
