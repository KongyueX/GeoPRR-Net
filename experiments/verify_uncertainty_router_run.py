"""Verify uncertainty-router provenance, routes, predictions, and metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments.ablate_uncertainty_router_features import ABLATION_PROTOCOL
from experiments.evaluate_uncertainty_router import EVALUATION_PROTOCOL
from experiments.quality_router import (
    finite_float,
    normalized_error,
    read_jsonl,
    route_prediction,
)
from experiments.summarize_uncertainty_router import CONDITIONS
from experiments.train_uncertainty_router import (
    TRAINING_PROTOCOL,
    UNCERTAINTY_ROUTER_PROTOCOL,
)
from experiments.uncertainty_fusion import FEATURE_NAMES
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/uncertainty_router_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/uncertainty_router_syncg/verification.json"),
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


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    output = args.output.resolve()
    model_path = root / "model" / "uncertainty_router.joblib"
    training_path = root / "model" / "training_summary.json"
    diagnostics_path = root / "model" / "nested_oof_routing.jsonl"
    ablation_path = root / "feature_ablation.json"
    oof_path = (
        PROJECT_DIR
        / "artifacts"
        / "runs"
        / "uncertainty_fusion_syncg"
        / "probabilistic_oof_clean.jsonl"
    )
    for path in (model_path, training_path, diagnostics_path, ablation_path, oof_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact = joblib.load(model_path)
    training = _load(training_path)
    ablation = _load(ablation_path)
    errors: list[str] = []

    if training.get("protocol") != TRAINING_PROTOCOL:
        errors.append("training protocol mismatch")
    if training.get("status") != "complete":
        errors.append("training status is not complete")
    if training.get("input_sha256") != sha256_file(oof_path):
        errors.append("training input hash mismatch")
    if training.get("model_sha256") != sha256_file(model_path):
        errors.append("training model hash mismatch")
    if training.get("diagnostics_sha256") != sha256_file(diagnostics_path):
        errors.append("training diagnostics hash mismatch")
    if int(training.get("group_leakage_count", -1)) != 0:
        errors.append("router training reports group leakage")
    if int(training.get("test_samples_used", -1)) != 0:
        errors.append("router training used test samples")

    if artifact.get("protocol") != UNCERTAINTY_ROUTER_PROTOCOL:
        errors.append("router artifact protocol mismatch")
    if artifact.get("training_protocol") != TRAINING_PROTOCOL:
        errors.append("router artifact training protocol mismatch")
    if artifact.get("train_only_certified") is not True:
        errors.append("router artifact lacks train-only certification")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        errors.append("router artifact feature schema mismatch")
    if artifact.get("training_oof_pairs_sha256") != sha256_file(oof_path):
        errors.append("router artifact training input mismatch")
    if artifact.get("test_sets_used") != []:
        errors.append("router artifact lists test-set use")
    expected_sources = {
        "features": PROJECT_DIR / "experiments" / "uncertainty_fusion.py",
        "policy": PROJECT_DIR / "experiments" / "quality_router.py",
        "trainer": PROJECT_DIR / "experiments" / "train_uncertainty_router.py",
        "shared_training_helpers": PROJECT_DIR
        / "experiments"
        / "train_quality_router.py",
    }
    recorded_sources = artifact.get("source_sha256") or {}
    for name, path in expected_sources.items():
        if recorded_sources.get(name) != sha256_file(path):
            errors.append(f"router {name} source hash mismatch")
    threshold = finite_float(artifact.get("threshold"))
    if threshold is None or not hasattr(artifact.get("estimator"), "predict"):
        errors.append("router threshold or estimator is invalid")

    if ablation.get("protocol") != ABLATION_PROTOCOL:
        errors.append("feature-ablation protocol mismatch")
    if ablation.get("input_sha256") != sha256_file(oof_path):
        errors.append("feature-ablation input mismatch")
    if int(ablation.get("group_leakage_count", -1)) != 0:
        errors.append("feature ablation reports group leakage")
    if int(ablation.get("test_samples_used", -1)) != 0:
        errors.append("feature ablation used test samples")
    if ablation.get("source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "ablate_uncertainty_router_features.py"
    ):
        errors.append("feature-ablation source hash mismatch")

    train_rows = read_jsonl(oof_path)
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    if len(train_ids) != len(train_rows):
        errors.append("OOF training rows contain duplicate IDs")
    diagnostic_rows = read_jsonl(diagnostics_path)
    if {str(row.get("sample_id")) for row in diagnostic_rows} != train_ids:
        errors.append("router diagnostics IDs differ from OOF input")

    evaluation_checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        directory = root / "evaluations" / condition
        summary_path = directory / "summary.json"
        predictions_path = directory / "predictions.jsonl"
        summary = _load(summary_path)
        predictions = read_jsonl(predictions_path)
        if summary.get("protocol") != EVALUATION_PROTOCOL:
            errors.append(f"{condition}: evaluation protocol mismatch")
        if summary.get("status") != "complete":
            errors.append(f"{condition}: evaluation is incomplete")
        if summary.get("router_sha256") != sha256_file(model_path):
            errors.append(f"{condition}: router model hash mismatch")
        if summary.get("predictions_sha256") != sha256_file(predictions_path):
            errors.append(f"{condition}: prediction file hash mismatch")
        if summary.get("source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "evaluate_uncertainty_router.py"
        ):
            errors.append(f"{condition}: evaluator source hash mismatch")
        if summary.get("feature_source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ):
            errors.append(f"{condition}: feature source hash mismatch")
        if summary.get("policy_source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "quality_router.py"
        ):
            errors.append(f"{condition}: policy source hash mismatch")
        identifiers = [str(row.get("sample_id")) for row in predictions]
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{condition}: duplicate prediction IDs")
        overlap = train_ids & set(identifiers)
        if overlap:
            errors.append(f"{condition}: {len(overlap)} train/test IDs overlap")

        route_mismatches = 0
        prediction_mismatches = 0
        recomputed_errors: list[float] = []
        recomputed_success: list[bool] = []
        for row in predictions:
            prediction, route = route_prediction(
                base_prediction=row.get("base_prediction"),
                vector_prediction=row.get("vector_prediction"),
                score=row.get("router_score"),
                threshold=float(row.get("router_threshold")),
            )
            if route == "vector_quality_switch":
                route = "vector_uncertainty_switch"
            if row.get("route") != route:
                route_mismatches += 1
            if not _close(row.get("prediction"), prediction):
                prediction_mismatches += 1
            recorded = finite_float(row.get("prediction"))
            recomputed_errors.append(normalized_error(row, recorded))
            recomputed_success.append(recorded is not None)
        metric = summary["metrics"]["uncertainty_router"]
        nmae = float(np.mean(recomputed_errors))
        acc2 = float(np.mean(np.asarray(recomputed_errors) <= 0.02))
        coverage = float(np.mean(recomputed_success))
        if not _close(nmae, metric.get("nmae")):
            errors.append(f"{condition}: NMAE mismatch")
        if not _close(acc2, metric.get("acc_2pct")):
            errors.append(f"{condition}: Acc@2% mismatch")
        if not _close(coverage, metric.get("coverage")):
            errors.append(f"{condition}: coverage mismatch")
        if route_mismatches or prediction_mismatches:
            errors.append(
                f"{condition}: route={route_mismatches}, "
                f"prediction={prediction_mismatches} mismatches"
            )
        evaluation_checks[condition] = {
            "samples": len(predictions),
            "train_sample_id_overlap": len(overlap),
            "route_mismatches": route_mismatches,
            "prediction_mismatches": prediction_mismatches,
            "recomputed_nmae": nmae,
            "recomputed_acc_2pct": acc2,
            "recomputed_coverage": coverage,
        }

    comparison_path = root / "uncertainty_router_comparison.json"
    comparison = _load(comparison_path)
    if comparison.get("protocol") != "uncertainty_router_paper_summary_v1":
        errors.append("paper summary protocol mismatch")
    for condition in CONDITIONS:
        expected_hash = sha256_file(root / "evaluations" / condition / "summary.json")
        if comparison["evaluations"][condition].get("summary_sha256") != expected_hash:
            errors.append(f"{condition}: paper summary input hash mismatch")

    result = {
        "schema_version": 1,
        "protocol": "uncertainty_router_verification_v1",
        "verified": not errors,
        "errors": errors,
        "checks": {
            "oof_samples": len(train_rows),
            "oof_groups": len({str(row.get("group_id")) for row in train_rows}),
            "training_test_samples_used": training.get("test_samples_used"),
            "model_sha256": sha256_file(model_path),
            "evaluations": evaluation_checks,
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
