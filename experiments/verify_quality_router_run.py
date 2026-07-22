"""Verify quality-router hashes, route decisions, metrics, and train/test separation."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.collect_quality_router_oof import QUALITY_ROUTER_OOF_PROTOCOL
from experiments.evaluate_quality_router import EVALUATION_PROTOCOL
from experiments.quality_router import (
    FEATURE_NAMES,
    QUALITY_ROUTER_PROTOCOL,
    finite_float,
    normalized_error,
    read_jsonl,
    route_prediction,
    sha256_file,
)
from experiments.summarize_quality_router import CONDITIONS
from experiments.train_quality_router import TRAINING_PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/verification.json"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _close(first: float, second: float, *, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    root = args.run_root.resolve()
    args.output = args.output.resolve()
    oof_path = root / "oof_clean.jsonl"
    oof_summary_path = root / "oof_clean.summary.json"
    model_path = root / "model" / "quality_router.joblib"
    training_path = root / "model" / "training_summary.json"
    oof_summary = _load(oof_summary_path)
    training = _load(training_path)
    artifact = joblib.load(model_path)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if oof_summary.get("protocol") != QUALITY_ROUTER_OOF_PROTOCOL:
        errors.append("OOF protocol mismatch")
    if oof_summary.get("output_sha256") != sha256_file(oof_path):
        errors.append("OOF output hash mismatch")
    if oof_summary.get("group_leakage_count") != 0:
        errors.append("OOF group leakage is non-zero")
    if oof_summary.get("test_samples_used") != 0:
        errors.append("OOF collection used test samples")
    if (oof_summary.get("signature") or {}).get("source_sha256") != sha256_file(
        PROJECT_ROOT / "experiments" / "collect_quality_router_oof.py"
    ):
        errors.append("OOF collector source hash mismatch")
    if training.get("protocol") != TRAINING_PROTOCOL:
        errors.append("training protocol mismatch")
    if training.get("input_sha256") != sha256_file(oof_path):
        errors.append("training input hash mismatch")
    if training.get("model_sha256") != sha256_file(model_path):
        errors.append("training model hash mismatch")
    if training.get("test_samples_used") != 0:
        errors.append("training summary reports test samples")
    if artifact.get("protocol") != QUALITY_ROUTER_PROTOCOL:
        errors.append("model protocol mismatch")
    if tuple(artifact.get("feature_names") or []) != FEATURE_NAMES:
        errors.append("model feature schema mismatch")
    if artifact.get("training_oof_pairs_sha256") != sha256_file(oof_path):
        errors.append("model training input hash mismatch")
    if artifact.get("test_sets_used") != []:
        errors.append("model artifact lists test-set use")
    artifact_sources = artifact.get("source_sha256") or {}
    if artifact_sources.get("features_and_policy") != sha256_file(
        PROJECT_ROOT / "experiments" / "quality_router.py"
    ):
        errors.append("model feature/policy source hash mismatch")
    if artifact_sources.get("trainer") != sha256_file(
        PROJECT_ROOT / "experiments" / "train_quality_router.py"
    ):
        errors.append("model trainer source hash mismatch")

    train_rows = read_jsonl(oof_path)
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    if len(train_ids) != len(train_rows):
        errors.append("duplicate OOF sample IDs")
    evaluation_checks: dict[str, Any] = {}
    for condition in CONDITIONS:
        directory = root / "evaluations" / condition
        summary_path = directory / "summary.json"
        predictions_path = directory / "predictions.jsonl"
        summary = _load(summary_path)
        predictions = read_jsonl(predictions_path)
        if summary.get("protocol") != EVALUATION_PROTOCOL:
            errors.append(f"{condition}: evaluation protocol mismatch")
        if summary.get("router_sha256") != sha256_file(model_path):
            errors.append(f"{condition}: router hash mismatch")
        if summary.get("predictions_sha256") != sha256_file(predictions_path):
            errors.append(f"{condition}: predictions hash mismatch")
        if summary.get("source_sha256") != sha256_file(
            PROJECT_ROOT / "experiments" / "evaluate_quality_router.py"
        ):
            errors.append(f"{condition}: evaluator source hash mismatch")
        if summary.get("router_feature_source_sha256") != sha256_file(
            PROJECT_ROOT / "experiments" / "quality_router.py"
        ):
            errors.append(f"{condition}: router feature source hash mismatch")
        test_ids = {str(row.get("sample_id")) for row in predictions}
        overlap = train_ids & test_ids
        if overlap:
            errors.append(f"{condition}: {len(overlap)} train/test sample IDs overlap")
        route_mismatches = 0
        prediction_mismatches = 0
        recomputed_errors = []
        recomputed_success = []
        for row in predictions:
            expected_prediction, expected_route = route_prediction(
                base_prediction=row.get("base_prediction"),
                vector_prediction=row.get("vector_prediction"),
                score=row.get("router_score"),
                threshold=float(row["router_threshold"]),
            )
            if row.get("route") != expected_route:
                route_mismatches += 1
            recorded = finite_float(row.get("prediction"))
            if recorded is None and expected_prediction is not None:
                prediction_mismatches += 1
            elif recorded is not None and expected_prediction is None:
                prediction_mismatches += 1
            elif recorded is not None and not _close(recorded, expected_prediction):
                prediction_mismatches += 1
            recomputed_errors.append(normalized_error(row, recorded))
            recomputed_success.append(recorded is not None)
        metrics = summary["metrics"]["quality_router"]
        recomputed_nmae = float(np.mean(recomputed_errors))
        recomputed_acc2 = float(np.mean(np.asarray(recomputed_errors) <= 0.02))
        recomputed_coverage = float(np.mean(recomputed_success))
        if not _close(recomputed_nmae, metrics["nmae"]):
            errors.append(f"{condition}: NMAE mismatch")
        if not _close(recomputed_acc2, metrics["acc_2pct"]):
            errors.append(f"{condition}: Acc@2% mismatch")
        if not _close(recomputed_coverage, metrics["coverage"]):
            errors.append(f"{condition}: coverage mismatch")
        if route_mismatches or prediction_mismatches:
            errors.append(
                f"{condition}: route={route_mismatches}, prediction={prediction_mismatches} mismatches"
            )
        evaluation_checks[condition] = {
            "samples": len(predictions),
            "train_sample_id_overlap": len(overlap),
            "route_mismatches": route_mismatches,
            "prediction_mismatches": prediction_mismatches,
            "recomputed_nmae": recomputed_nmae,
            "recomputed_acc_2pct": recomputed_acc2,
            "recomputed_coverage": recomputed_coverage,
        }

    checks.update(
        {
            "oof_samples": len(train_rows),
            "oof_groups": oof_summary.get("groups"),
            "oof_group_leakage_count": oof_summary.get("group_leakage_count"),
            "training_test_samples_used": training.get("test_samples_used"),
            "model_sha256": sha256_file(model_path),
            "evaluations": evaluation_checks,
        }
    )
    result = {
        "schema_version": 1,
        "protocol": "quality_router_verification_v1",
        "verified": not errors,
        "errors": errors,
        "checks": checks,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(args.output)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
