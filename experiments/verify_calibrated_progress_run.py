"""Verify progress-calibration and final-router provenance and predictions."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments.calibrated_progress_router import (
    CALIBRATED_PROGRESS_ROUTER_PROTOCOL,
    FEATURE_NAMES as ROUTER_FEATURE_NAMES,
)
from experiments.evaluate_calibrated_progress_router import (
    EVALUATION_PROTOCOL as ROUTER_EVALUATION_PROTOCOL,
)
from experiments.evaluate_progress_calibrator import (
    EVALUATION_PROTOCOL as CALIBRATOR_EVALUATION_PROTOCOL,
)
from experiments.progress_calibrator import (
    FEATURE_NAMES as CALIBRATOR_FEATURE_NAMES,
    PROGRESS_CALIBRATOR_PROTOCOL,
    apply_progress_correction,
    reading_from_progress,
)
from experiments.quality_router import (
    finite_float,
    normalized_error,
    read_jsonl,
    route_prediction,
)
from experiments.summarize_calibrated_progress import CONDITIONS
from experiments.train_calibrated_progress_router import (
    TRAINING_PROTOCOL as ROUTER_TRAINING_PROTOCOL,
)
from experiments.train_progress_calibrator import (
    TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrator-root",
        type=Path,
        default=Path("artifacts/runs/progress_calibrator_syncg"),
    )
    parser.add_argument(
        "--router-root",
        type=Path,
        default=Path("artifacts/runs/calibrated_progress_router_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/calibrated_progress_router_syncg/verification.json"
        ),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _close(first: Any, second: Any, tolerance: float = 1e-10) -> bool:
    first_value = finite_float(first)
    second_value = finite_float(second)
    if first_value is None or second_value is None:
        return first_value is None and second_value is None
    return math.isclose(first_value, second_value, rel_tol=0.0, abs_tol=tolerance)


def _check_sources(
    recorded: dict[str, Any], expected: dict[str, Path], errors: list[str], prefix: str
) -> None:
    for name, path in expected.items():
        if recorded.get(name) != sha256_file(path):
            errors.append(f"{prefix} {name} source hash mismatch")


def main() -> None:
    args = parse_args()
    calibrator_root = args.calibrator_root.resolve()
    router_root = args.router_root.resolve()
    output = args.output.resolve()
    oof_path = (
        PROJECT_DIR
        / "artifacts"
        / "runs"
        / "uncertainty_fusion_syncg"
        / "probabilistic_oof_clean.jsonl"
    )
    calibrator_model_path = calibrator_root / "model" / "progress_calibrator.joblib"
    calibrator_training_path = calibrator_root / "model" / "training_summary.json"
    calibrator_diagnostics_path = calibrator_root / "model" / "nested_oof_predictions.jsonl"
    router_model_path = router_root / "model" / "calibrated_progress_router.joblib"
    router_training_path = router_root / "model" / "training_summary.json"
    router_diagnostics_path = router_root / "model" / "nested_oof_routing.jsonl"
    for path in (
        oof_path,
        calibrator_model_path,
        calibrator_training_path,
        calibrator_diagnostics_path,
        router_model_path,
        router_training_path,
        router_diagnostics_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    calibrator = joblib.load(calibrator_model_path)
    router = joblib.load(router_model_path)
    calibrator_training = _load(calibrator_training_path)
    router_training = _load(router_training_path)
    errors: list[str] = []

    if calibrator_training.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL:
        errors.append("calibrator training protocol mismatch")
    if calibrator_training.get("status") != "complete":
        errors.append("calibrator training is incomplete")
    if calibrator_training.get("input_sha256") != sha256_file(oof_path):
        errors.append("calibrator training input hash mismatch")
    if calibrator_training.get("model_sha256") != sha256_file(calibrator_model_path):
        errors.append("calibrator model hash mismatch")
    if calibrator_training.get("diagnostics_sha256") != sha256_file(
        calibrator_diagnostics_path
    ):
        errors.append("calibrator diagnostics hash mismatch")
    if int(calibrator_training.get("group_leakage_count", -1)) != 0:
        errors.append("calibrator training reports group leakage")
    if int(calibrator_training.get("test_samples_used", -1)) != 0:
        errors.append("calibrator training used test samples")
    if calibrator.get("protocol") != PROGRESS_CALIBRATOR_PROTOCOL:
        errors.append("calibrator artifact protocol mismatch")
    if calibrator.get("training_protocol") != CALIBRATOR_TRAINING_PROTOCOL:
        errors.append("calibrator artifact training protocol mismatch")
    if calibrator.get("train_only_certified") is not True:
        errors.append("calibrator artifact lacks train-only certification")
    if tuple(calibrator.get("feature_names") or ()) != CALIBRATOR_FEATURE_NAMES:
        errors.append("calibrator artifact feature schema mismatch")
    if calibrator.get("test_sets_used") != []:
        errors.append("calibrator artifact lists test-set use")
    _check_sources(
        calibrator.get("source_sha256") or {},
        {
            "features_and_policy": PROJECT_DIR / "experiments" / "progress_calibrator.py",
            "trainer": PROJECT_DIR / "experiments" / "train_progress_calibrator.py",
        },
        errors,
        "calibrator",
    )

    if router_training.get("protocol") != ROUTER_TRAINING_PROTOCOL:
        errors.append("router training protocol mismatch")
    if router_training.get("status") != "complete":
        errors.append("router training is incomplete")
    if router_training.get("input_sha256") != sha256_file(oof_path):
        errors.append("router training input hash mismatch")
    if router_training.get("calibration_diagnostics_sha256") != sha256_file(
        calibrator_diagnostics_path
    ):
        errors.append("router calibration diagnostics hash mismatch")
    if router_training.get("model_sha256") != sha256_file(router_model_path):
        errors.append("router model hash mismatch")
    if router_training.get("diagnostics_sha256") != sha256_file(
        router_diagnostics_path
    ):
        errors.append("router diagnostics hash mismatch")
    if int(router_training.get("group_leakage_count", -1)) != 0:
        errors.append("router training reports group leakage")
    if int(router_training.get("test_samples_used", -1)) != 0:
        errors.append("router training used test samples")
    if router.get("protocol") != CALIBRATED_PROGRESS_ROUTER_PROTOCOL:
        errors.append("router artifact protocol mismatch")
    if router.get("training_protocol") != ROUTER_TRAINING_PROTOCOL:
        errors.append("router artifact training protocol mismatch")
    if router.get("train_only_certified") is not True:
        errors.append("router artifact lacks train-only certification")
    if tuple(router.get("feature_names") or ()) != ROUTER_FEATURE_NAMES:
        errors.append("router artifact feature schema mismatch")
    if router.get("test_sets_used") != []:
        errors.append("router artifact lists test-set use")
    _check_sources(
        router.get("source_sha256") or {},
        {
            "features": PROJECT_DIR / "experiments" / "calibrated_progress_router.py",
            "quality_features": PROJECT_DIR / "experiments" / "uncertainty_fusion.py",
            "policy": PROJECT_DIR / "experiments" / "quality_router.py",
            "trainer": PROJECT_DIR / "experiments" / "train_calibrated_progress_router.py",
            "shared_training_helpers": PROJECT_DIR / "experiments" / "train_quality_router.py",
        },
        errors,
        "router",
    )

    train_rows = read_jsonl(oof_path)
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    if len(train_ids) != len(train_rows):
        errors.append("OOF input contains duplicate IDs")
    calibrator_diagnostics = read_jsonl(calibrator_diagnostics_path)
    router_diagnostics = read_jsonl(router_diagnostics_path)
    if {str(row.get("sample_id")) for row in calibrator_diagnostics} != train_ids:
        errors.append("calibrator diagnostics IDs differ from OOF input")
    if {str(row.get("sample_id")) for row in router_diagnostics} != train_ids:
        errors.append("router diagnostics IDs differ from OOF input")

    checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        calibration_dir = calibrator_root / "evaluations" / condition
        calibration_summary_path = calibration_dir / "summary.json"
        calibration_predictions_path = calibration_dir / "predictions.jsonl"
        calibration_summary = _load(calibration_summary_path)
        calibration_predictions = read_jsonl(calibration_predictions_path)
        if calibration_summary.get("protocol") != CALIBRATOR_EVALUATION_PROTOCOL:
            errors.append(f"{condition}: calibrator evaluation protocol mismatch")
        if calibration_summary.get("status") != "complete":
            errors.append(f"{condition}: calibrator evaluation incomplete")
        if calibration_summary.get("calibrator_sha256") != sha256_file(
            calibrator_model_path
        ):
            errors.append(f"{condition}: calibrator evaluation model mismatch")
        if calibration_summary.get("predictions_sha256") != sha256_file(
            calibration_predictions_path
        ):
            errors.append(f"{condition}: calibrator prediction hash mismatch")
        if calibration_summary.get("source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "evaluate_progress_calibrator.py"
        ):
            errors.append(f"{condition}: calibrator evaluator source mismatch")
        calibration_ids = [str(row.get("sample_id")) for row in calibration_predictions]
        if len(calibration_ids) != len(set(calibration_ids)):
            errors.append(f"{condition}: duplicate calibrated prediction IDs")
        overlap = train_ids & set(calibration_ids)
        if overlap:
            errors.append(f"{condition}: train/test ID overlap")
        calibration_mismatches = 0
        calibration_errors: list[float] = []
        calibration_success: list[bool] = []
        for row in calibration_predictions:
            progress = apply_progress_correction(
                row.get("raw_progress"),
                row.get("predicted_progress_residual"),
                correction_clip=float(row.get("correction_clip")),
            )
            prediction = reading_from_progress(
                progress, row.get("scale_start"), row.get("scale_end")
            )
            if not _close(progress, row.get("progress")) or not _close(
                prediction, row.get("prediction")
            ):
                calibration_mismatches += 1
            recorded = finite_float(row.get("prediction"))
            calibration_errors.append(normalized_error(row, recorded))
            calibration_success.append(recorded is not None)
        calibration_metric = calibration_summary["metrics"]["calibrated_vector"]
        calibration_nmae = float(np.mean(calibration_errors))
        calibration_acc2 = float(np.mean(np.asarray(calibration_errors) <= 0.02))
        calibration_coverage = float(np.mean(calibration_success))
        if not _close(calibration_nmae, calibration_metric.get("nmae")):
            errors.append(f"{condition}: calibrated NMAE mismatch")
        if not _close(calibration_acc2, calibration_metric.get("acc_2pct")):
            errors.append(f"{condition}: calibrated Acc@2 mismatch")
        if not _close(calibration_coverage, calibration_metric.get("coverage")):
            errors.append(f"{condition}: calibrated coverage mismatch")
        if calibration_mismatches:
            errors.append(
                f"{condition}: {calibration_mismatches} calibration prediction mismatches"
            )

        router_dir = router_root / "evaluations" / condition
        router_summary_path = router_dir / "summary.json"
        router_predictions_path = router_dir / "predictions.jsonl"
        router_summary = _load(router_summary_path)
        router_predictions = read_jsonl(router_predictions_path)
        if router_summary.get("protocol") != ROUTER_EVALUATION_PROTOCOL:
            errors.append(f"{condition}: router evaluation protocol mismatch")
        if router_summary.get("status") != "complete":
            errors.append(f"{condition}: router evaluation incomplete")
        if router_summary.get("router_sha256") != sha256_file(router_model_path):
            errors.append(f"{condition}: router evaluation model mismatch")
        if router_summary.get("predictions_sha256") != sha256_file(
            router_predictions_path
        ):
            errors.append(f"{condition}: router prediction hash mismatch")
        if router_summary.get("source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "evaluate_calibrated_progress_router.py"
        ):
            errors.append(f"{condition}: router evaluator source mismatch")
        router_ids = [str(row.get("sample_id")) for row in router_predictions]
        if set(router_ids) != set(calibration_ids):
            errors.append(f"{condition}: router/calibrator prediction IDs differ")
        route_mismatches = 0
        router_errors: list[float] = []
        router_success: list[bool] = []
        for row in router_predictions:
            prediction, route = route_prediction(
                base_prediction=row.get("base_prediction"),
                vector_prediction=row.get("calibrated_prediction"),
                score=row.get("router_score"),
                threshold=float(row.get("router_threshold")),
            )
            route = {
                "vector_quality_switch": "calibrated_quality_switch",
                "vector_hard_fallback": "calibrated_hard_fallback",
            }.get(route, route)
            if route != row.get("route") or not _close(prediction, row.get("prediction")):
                route_mismatches += 1
            recorded = finite_float(row.get("prediction"))
            router_errors.append(normalized_error(row, recorded))
            router_success.append(recorded is not None)
        router_metric = router_summary["metrics"]["calibrated_progress_router"]
        router_nmae = float(np.mean(router_errors))
        router_acc2 = float(np.mean(np.asarray(router_errors) <= 0.02))
        router_coverage = float(np.mean(router_success))
        if not _close(router_nmae, router_metric.get("nmae")):
            errors.append(f"{condition}: router NMAE mismatch")
        if not _close(router_acc2, router_metric.get("acc_2pct")):
            errors.append(f"{condition}: router Acc@2 mismatch")
        if not _close(router_coverage, router_metric.get("coverage")):
            errors.append(f"{condition}: router coverage mismatch")
        if route_mismatches:
            errors.append(f"{condition}: {route_mismatches} router mismatches")
        checks[condition] = {
            "samples": len(router_predictions),
            "train_sample_id_overlap": len(overlap),
            "calibration_prediction_mismatches": calibration_mismatches,
            "route_mismatches": route_mismatches,
            "recomputed_calibrated_nmae": calibration_nmae,
            "recomputed_router_nmae": router_nmae,
            "recomputed_router_acc_2pct": router_acc2,
            "recomputed_router_coverage": router_coverage,
        }

    comparison_path = router_root / "calibrated_progress_comparison.json"
    comparison = _load(comparison_path)
    if comparison.get("protocol") != "calibrated_progress_paper_summary_v2":
        errors.append("paper summary protocol mismatch")
    for condition in CONDITIONS:
        calibration_hash = sha256_file(
            calibrator_root / "evaluations" / condition / "summary.json"
        )
        router_hash = sha256_file(
            router_root / "evaluations" / condition / "summary.json"
        )
        recorded = comparison["evaluations"][condition]
        if recorded.get("calibrator_summary_sha256") != calibration_hash:
            errors.append(f"{condition}: paper calibrator summary hash mismatch")
        if recorded.get("router_summary_sha256") != router_hash:
            errors.append(f"{condition}: paper router summary hash mismatch")

    result = {
        "schema_version": 1,
        "protocol": "calibrated_progress_verification_v2",
        "verified": not errors,
        "errors": errors,
        "checks": {
            "oof_samples": len(train_rows),
            "oof_groups": len({str(row.get("group_id")) for row in train_rows}),
            "calibrator_test_samples_used": calibrator_training.get("test_samples_used"),
            "router_test_samples_used": router_training.get("test_samples_used"),
            "calibrator_model_sha256": sha256_file(calibrator_model_path),
            "router_model_sha256": sha256_file(router_model_path),
            "evaluations": checks,
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
