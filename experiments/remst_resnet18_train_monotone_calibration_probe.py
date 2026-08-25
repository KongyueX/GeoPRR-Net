"""Fit-only monotone Raw calibration for the refined ReMST-ResNet18 anchor.

The seven-knot exact-L1 map is fitted on all 131 source-fit scenes under the
three Raw conditions.  It is then applied post hoc to the already-materialized
main-table development predictions, so the screening evaluation uses exactly
the benchmark pixels and never touches the formal holdout.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments import robustness_degradations
from experiments.raw_layer4_endpoint_fullfit_probe import (
    DEFAULT_FIT_MANIFEST,
    load_raw_layer4_fullfit_remst,
)
from experiments.raw_monotone_residual_calibration_probe import (
    DEFAULT_KNOT_COUNT,
    METHOD_IDENTITY,
    METHOD_MEDIAN_BIAS,
    METHOD_MONOTONE_SPLINE,
    apply_median_bias,
    fit_median_bias,
    fit_monotone_l1_spline,
)
from experiments.resnet18_direct_progress import (
    IMAGE_SIZE,
    DirectSample,
    _configure_reproducibility,
    canonical_tight_roi_native,
    direct_resize_whole_roi,
    load_syncg_samples,
    normalized_rgb_tensor,
)
from experiments.run_cagh_v5_plain_paper_batch import CONDITIONS, ROBUSTNESS_SEED


PROTOCOL: Final[str] = "remst_resnet18_fit_monotone_raw_calibration_probe_v1"
DEFAULT_SOURCE_CHECKPOINT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_layer4_fullfit/seed_20262020/terminal.pt"
)
DEFAULT_DEVELOPMENT_RESULT: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_raw_layer4_fullfit/seed_20262020/"
    "six_condition_results.json"
)
DEFAULT_OUTPUT_DIR: Final[Path] = Path(
    "artifacts/runs/remst_resnet18_train_monotone_calibration/seed_20262020"
)
DEFAULT_SEED: Final[int] = 20_262_020
DEFAULT_BATCH_SIZE: Final[int] = 64
DEFAULT_WORKERS: Final[int] = 4
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 5_000
RAW_CONDITIONS: Final[tuple[str, ...]] = tuple(CONDITIONS[:3])


class ReMSTMonotoneCalibrationError(ValueError):
    """The fit population, checkpoint, or evaluation replay is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTMonotoneCalibrationError(message)


def _write_json(path: Path, value: Any) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


class FitRawConditionDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: Sequence[DirectSample], *, condition: str) -> None:
        self.samples = tuple(samples)
        self.condition = str(condition)
        _require(bool(self.samples), "fit condition dataset is empty")
        _require(self.condition in RAW_CONDITIONS, "fit condition is not Raw")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        image = cv2.imread(
            str(sample.image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        _require(image is not None, f"{sample.sample_id}: source decode failed")
        roi, _bounds = canonical_tight_roi_native(image, sample.dial_bbox)
        conditioned, _metadata = robustness_degradations.apply_degradation(
            roi,
            self.condition,
            sample_id=sample.sample_id,
            seed=ROBUSTNESS_SEED,
        )
        resized = direct_resize_whole_roi(conditioned, size=IMAGE_SIZE)
        return {
            "image": normalized_rgb_tensor(resized),
            "target": torch.tensor(sample.normalized_target, dtype=torch.float32),
            "sample_id": sample.sample_id,
            "scene_stem": sample.scene_stem,
        }


def _collect_fit_rows(
    anchor: torch.nn.Module,
    samples: Sequence[DirectSample],
    *,
    condition: str,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    loader = DataLoader(
        FitRawConditionDataset(samples, condition=condition),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    rows: list[dict[str, Any]] = []
    anchor.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            features = anchor.raw_encoder(images)
            predictions = anchor.raw_posterior_head.point_progress(
                features["representation"]
            )
            for sample_id, scene, target, prediction in zip(
                batch["sample_id"],
                batch["scene_stem"],
                batch["target"].tolist(),
                predictions.float().cpu().tolist(),
                strict=True,
            ):
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "scene_stem": str(scene),
                        "condition": condition,
                        "target": float(target),
                        "prediction": float(prediction),
                    }
                )
            if (batch_index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "fit_prediction_replay",
                            "condition": condition,
                            "batches": batch_index + 1,
                            "rows": len(rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    _require(len(rows) == len(samples), "fit prediction row count differs")
    return tuple(rows)


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    targets = np.asarray([float(row["target"]) for row in rows], dtype=np.float64)
    reference = np.asarray(
        [float(row["predictions"][METHOD_IDENTITY]) for row in rows],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(row["predictions"][candidate]) for row in rows], dtype=np.float64
    )
    reference_error = np.abs(reference - targets)
    candidate_error = np.abs(values - targets)
    delta = candidate_error - reference_error
    by_scene: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_scene[str(row["scene_stem"])].append(index)
    scene_names = tuple(sorted(by_scene))
    _require(len(scene_names) >= 2, "comparison needs at least two scenes")
    scene_indices = tuple(
        np.asarray(by_scene[name], dtype=np.int64) for name in scene_names
    )
    generator = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = generator.integers(0, len(scene_names), size=len(scene_names))
        indices = np.concatenate(tuple(scene_indices[int(index)] for index in sampled))
        bootstrap[replicate] = float(delta[indices].mean())
    reference_nmae = float(reference_error.mean())
    candidate_nmae = float(candidate_error.mean())
    return {
        "reference": METHOD_IDENTITY,
        "candidate": candidate,
        "rows": len(rows),
        "scenes": len(scene_names),
        "reference_nmae": reference_nmae,
        "candidate_nmae": candidate_nmae,
        "candidate_minus_reference": candidate_nmae - reference_nmae,
        "relative_error_reduction": (reference_nmae - candidate_nmae)
        / reference_nmae,
        "scene_cluster_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "scene_cluster_bootstrap_probability_candidate_better": float(
            np.mean(bootstrap < 0.0)
        ),
        "paired_row_wins": int(np.sum(delta < 0.0)),
        "paired_row_ties": int(np.sum(delta == 0.0)),
        "paired_row_losses": int(np.sum(delta > 0.0)),
    }


def _load_development_rows(
    path: Path,
    *,
    median_bias: float,
    spline: Any,
) -> tuple[dict[str, Any], ...]:
    source = Path(path).resolve()
    _require(source.is_file(), f"development replay is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    source_rows = payload.get("per_sample_condition")
    _require(
        payload.get("status") == "complete" and isinstance(source_rows, list),
        "development replay is incomplete",
    )
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        condition = str(source_row["condition"])
        if condition not in RAW_CONDITIONS:
            continue
        prediction = float(source_row["raw_anchor"]["prediction"])
        calibrated = float(spline.predict(np.asarray([prediction]))[0])
        biased = float(apply_median_bias(np.asarray([prediction]), median_bias)[0])
        rows.append(
            {
                "sample_id": str(source_row["sample_id"]),
                "scene_stem": str(source_row["scene_stem"]),
                "condition": condition,
                "target": float(source_row["normalized_target"]),
                "predictions": {
                    METHOD_IDENTITY: prediction,
                    METHOD_MEDIAN_BIAS: biased,
                    METHOD_MONOTONE_SPLINE: calibrated,
                },
            }
        )
    _require(len(rows) == 4_674, "development Raw replay row count differs")
    return tuple(rows)


def run_probe(
    *,
    source_checkpoint_path: Path,
    fit_manifest_path: Path,
    development_result_path: Path,
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
        "probe configuration is invalid",
    )
    root = Path(output_dir).resolve()
    calibrator_path = root / "calibrator.json"
    results_path = root / "results.json"
    _require(
        not calibrator_path.exists() and not results_path.exists(),
        "calibration probe output already exists",
    )
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(int(seed), device)
    anchor, _correction, source_metadata = load_raw_layer4_fullfit_remst(
        source_checkpoint_path, device=device
    )
    fit_samples = tuple(load_syncg_samples(Path(fit_manifest_path).resolve()))
    _require(
        len(fit_samples) == 14_442
        and len({sample.scene_stem for sample in fit_samples}) == 131,
        "fit population identity differs",
    )
    started = time.perf_counter()
    fit_rows: list[dict[str, Any]] = []
    for index, condition in enumerate(RAW_CONDITIONS):
        condition_rows = _collect_fit_rows(
            anchor,
            fit_samples,
            condition=condition,
            device=device,
            batch_size=batch_size,
            workers=workers,
            seed=int(seed) + index,
        )
        fit_rows.extend(condition_rows)
        print(
            json.dumps(
                {
                    "phase": "fit_condition_complete",
                    "condition": condition,
                    "nmae": statistics.fmean(
                        abs(float(row["prediction"]) - float(row["target"]))
                        for row in condition_rows
                    ),
                    "rows": len(condition_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    predictions = np.asarray(
        [float(row["prediction"]) for row in fit_rows], dtype=np.float64
    )
    targets = np.asarray([float(row["target"]) for row in fit_rows], dtype=np.float64)
    _require(len(predictions) == 43_326, "fit replay size differs")
    median_bias = fit_median_bias(predictions, targets)
    spline, solver = fit_monotone_l1_spline(
        predictions, targets, knot_count=int(knot_count)
    )
    fit_nmae = {
        METHOD_IDENTITY: float(np.mean(np.abs(predictions - targets))),
        METHOD_MEDIAN_BIAS: float(
            np.mean(np.abs(apply_median_bias(predictions, median_bias) - targets))
        ),
        METHOD_MONOTONE_SPLINE: float(
            np.mean(np.abs(spline.predict(predictions) - targets))
        ),
    }
    development_rows = _load_development_rows(
        development_result_path, median_bias=median_bias, spline=spline
    )
    scopes = {
        **{
            condition: tuple(
                row for row in development_rows if row["condition"] == condition
            )
            for condition in RAW_CONDITIONS
        },
        "raw_pooled": development_rows,
    }
    development = {
        scope: {
            METHOD_MEDIAN_BIAS: _comparison(
                rows,
                candidate=METHOD_MEDIAN_BIAS,
                replicates=int(bootstrap_replicates),
                seed=int(seed) + 50_000 + index * 1_009,
            ),
            METHOD_MONOTONE_SPLINE: _comparison(
                rows,
                candidate=METHOD_MONOTONE_SPLINE,
                replicates=int(bootstrap_replicates),
                seed=int(seed) + 60_000 + index * 1_009,
            ),
        }
        for index, (scope, rows) in enumerate(scopes.items())
    }
    calibrator = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_raw_checkpoint": str(Path(source_checkpoint_path).resolve()),
        "source_raw_metadata": source_metadata,
        "fit_population": "complete_original_source_fit_14442_clean_blur_pooled",
        "fit_rows": len(fit_rows),
        "fit_scenes": 131,
        "fit_conditions": list(RAW_CONDITIONS),
        "development_labels_used_for_fit": False,
        "formal_holdout_access": False,
        "median_bias": float(median_bias),
        "monotone_spline": spline.as_dict(),
        "solver": solver,
        "fit_nmae": fit_nmae,
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "seed": int(seed),
        "scientific_question": (
            "Does one shared fit-only monotone endpoint map materially improve "
            "all three Raw main-table conditions?"
        ),
        "scope": {
            "development_cohort": True,
            "formal_holdout_touched": False,
            "projection_path_evaluated": False,
            "single_backbone": True,
            "additional_image_encoders": 0,
            "additional_trainable_parameters": 0,
        },
        "calibrator": calibrator,
        "development": development,
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {
            "calibrator": str(calibrator_path),
            "results": str(results_path),
        },
    }
    _write_json(calibrator_path, calibrator)
    _write_json(results_path, result)
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT
    )
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument(
        "--development-result", type=Path, default=DEFAULT_DEVELOPMENT_RESULT
    )
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
    args = build_argument_parser().parse_args(argv)
    try:
        result = run_probe(
            source_checkpoint_path=args.source_checkpoint,
            fit_manifest_path=args.fit_manifest,
            development_result_path=args.development_result,
            output_dir=args.output_dir,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            knot_count=args.knot_count,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except ReMSTMonotoneCalibrationError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "fit_nmae": result["calibrator"]["fit_nmae"],
                "development": result["development"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FitRawConditionDataset", "PROTOCOL", "run_probe"]
