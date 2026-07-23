"""Verify uncertainty-fusion provenance, routes, predictions, and metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.collect_uncertainty_fusion_oof import (
    UNCERTAINTY_FUSION_OOF_PROTOCOL,
)
from experiments.evaluate_uncertainty_fusion import EVALUATION_PROTOCOL
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.summarize_uncertainty_fusion import CONDITIONS
from experiments.train_uncertainty_fusion import TRAINING_PROTOCOL
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    UNCERTAINTY_FUSION_PROTOCOL,
    soft_fusion_prediction,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/uncertainty_fusion_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/uncertainty_fusion_syncg/verification.json"),
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
    oof_path = root / "probabilistic_oof_clean.jsonl"
    oof_meta_path = oof_path.with_name(oof_path.name + ".meta.json")
    oof_summary_path = oof_path.with_name(oof_path.stem + ".summary.json")
    model_path = root / "model" / "uncertainty_fusion.pt"
    training_path = root / "model" / "training_summary.json"
    nested_path = root / "model" / "nested_oof_predictions.jsonl"
    feature_ablation_path = root / "feature_ablation.json"
    for path in (
        oof_path,
        oof_meta_path,
        oof_summary_path,
        model_path,
        training_path,
        nested_path,
        feature_ablation_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    oof_meta = _load(oof_meta_path)
    oof_summary = _load(oof_summary_path)
    training = _load(training_path)
    artifact = torch.load(model_path, map_location="cpu", weights_only=False)
    feature_ablation = _load(feature_ablation_path)
    errors: list[str] = []
    checks: dict[str, Any] = {}

    if oof_summary.get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        errors.append("OOF protocol mismatch")
    if oof_summary.get("status") != "complete":
        errors.append("OOF status is not complete")
    if oof_summary.get("output_sha256") != sha256_file(oof_path):
        errors.append("OOF output hash mismatch")
    if int(oof_summary.get("group_leakage_count", -1)) != 0:
        errors.append("OOF group leakage is non-zero")
    if int(oof_summary.get("test_samples_used", -1)) != 0:
        errors.append("OOF collection used test samples")
    signature = oof_meta.get("signature") or {}
    if signature != (oof_summary.get("signature") or {}):
        errors.append("OOF metadata/summary signature mismatch")
    if signature.get("source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "collect_uncertainty_fusion_oof.py"
    ):
        errors.append("OOF collector source hash mismatch")
    if signature.get("test_sets_used") != []:
        errors.append("OOF signature lists test-set use")

    if training.get("protocol") != TRAINING_PROTOCOL:
        errors.append("training protocol mismatch")
    if training.get("status") != "complete":
        errors.append("training status is not complete")
    if training.get("input_sha256") != sha256_file(oof_path):
        errors.append("training input hash mismatch")
    if training.get("model_sha256") != sha256_file(model_path):
        errors.append("training model hash mismatch")
    if training.get("oof_predictions_sha256") != sha256_file(nested_path):
        errors.append("nested OOF prediction hash mismatch")
    if int(training.get("group_leakage_count", -1)) != 0:
        errors.append("nested training group leakage is non-zero")
    if int(training.get("test_samples_used", -1)) != 0:
        errors.append("fusion training used test samples")

    if artifact.get("protocol") != UNCERTAINTY_FUSION_PROTOCOL:
        errors.append("fusion artifact protocol mismatch")
    if artifact.get("training_protocol") != TRAINING_PROTOCOL:
        errors.append("fusion artifact training protocol mismatch")
    if artifact.get("train_only_certified") is not True:
        errors.append("fusion artifact lacks train-only certification")
    if tuple(artifact.get("feature_names") or ()) != FEATURE_NAMES:
        errors.append("fusion artifact feature schema mismatch")
    if artifact.get("training_oof_pairs_sha256") != sha256_file(oof_path):
        errors.append("fusion artifact training input mismatch")
    if artifact.get("feature_source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
    ):
        errors.append("fusion feature/policy source hash mismatch")
    if artifact.get("trainer_source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "train_uncertainty_fusion.py"
    ):
        errors.append("fusion trainer source hash mismatch")
    state = artifact.get("model_state") or {}
    if not state or not all(torch.isfinite(tensor).all() for tensor in state.values()):
        errors.append("fusion model state is empty or non-finite")
    if feature_ablation.get("protocol") != (
        "grouped_oof_uncertainty_fusion_feature_ablation_v1"
    ):
        errors.append("feature ablation protocol mismatch")
    if feature_ablation.get("input_sha256") != sha256_file(oof_path):
        errors.append("feature ablation input mismatch")
    if int(feature_ablation.get("group_leakage_count", -1)) != 0:
        errors.append("feature ablation reports group leakage")
    if int(feature_ablation.get("test_samples_used", -1)) != 0:
        errors.append("feature ablation used test samples")
    if feature_ablation.get("source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "ablate_uncertainty_fusion_features.py"
    ):
        errors.append("feature ablation source hash mismatch")

    train_rows = read_jsonl(oof_path)
    train_ids = {str(row.get("sample_id")) for row in train_rows}
    if len(train_ids) != len(train_rows):
        errors.append("OOF contains duplicate sample IDs")
    nested_rows = read_jsonl(nested_path)
    if {str(row.get("sample_id")) for row in nested_rows} != train_ids:
        errors.append("nested OOF IDs differ from collector output")

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
        if summary.get("fusion_model_sha256") != sha256_file(model_path):
            errors.append(f"{condition}: fusion model hash mismatch")
        if summary.get("predictions_sha256") != sha256_file(predictions_path):
            errors.append(f"{condition}: prediction file hash mismatch")
        if summary.get("source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "evaluate_uncertainty_fusion.py"
        ):
            errors.append(f"{condition}: evaluator source hash mismatch")
        if summary.get("feature_source_sha256") != sha256_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ):
            errors.append(f"{condition}: feature source hash mismatch")
        identifiers = [str(row.get("sample_id")) for row in predictions]
        if len(identifiers) != len(set(identifiers)):
            errors.append(f"{condition}: duplicate prediction IDs")
        overlap = train_ids & set(identifiers)
        if overlap:
            errors.append(f"{condition}: {len(overlap)} train/test IDs overlap")

        route_mismatches = 0
        prediction_mismatches = 0
        weight_mismatches = 0
        recomputed_errors: list[float] = []
        recomputed_success: list[bool] = []
        for row in predictions:
            prediction, route, weight, effective = soft_fusion_prediction(
                base_prediction=row.get("base_prediction"),
                vector_prediction=row.get("vector_prediction"),
                scale_start=row.get("scale_start"),
                scale_end=row.get("scale_end"),
                mask_log_variance=row.get("mask_log_variance"),
                vector_log_variance=row.get("vector_log_variance"),
                temperature=float(artifact["temperature"]),
            )
            if row.get("route") != route:
                route_mismatches += 1
            if not _close(row.get("prediction"), prediction):
                prediction_mismatches += 1
            if not _close(row.get("mask_weight"), weight):
                weight_mismatches += 1
            if not _close(row.get("effective_log_variance"), effective):
                weight_mismatches += 1
            recorded = finite_float(row.get("prediction"))
            recomputed_errors.append(normalized_error(row, recorded))
            recomputed_success.append(recorded is not None)
        metrics = summary["metrics"]["uncertainty_fusion"]
        nmae = float(np.mean(recomputed_errors))
        acc2 = float(np.mean(np.asarray(recomputed_errors) <= 0.02))
        coverage = float(np.mean(recomputed_success))
        if not _close(nmae, metrics.get("nmae")):
            errors.append(f"{condition}: NMAE mismatch")
        if not _close(acc2, metrics.get("acc_2pct")):
            errors.append(f"{condition}: Acc@2% mismatch")
        if not _close(coverage, metrics.get("coverage")):
            errors.append(f"{condition}: coverage mismatch")
        if route_mismatches or prediction_mismatches or weight_mismatches:
            errors.append(
                f"{condition}: route={route_mismatches}, prediction="
                f"{prediction_mismatches}, weight={weight_mismatches} mismatches"
            )
        evaluation_checks[condition] = {
            "samples": len(predictions),
            "train_sample_id_overlap": len(overlap),
            "route_mismatches": route_mismatches,
            "prediction_mismatches": prediction_mismatches,
            "weight_mismatches": weight_mismatches,
            "recomputed_nmae": nmae,
            "recomputed_acc_2pct": acc2,
            "recomputed_coverage": coverage,
        }

    comparison_path = root / "uncertainty_fusion_comparison.json"
    comparison = _load(comparison_path)
    if comparison.get("protocol") != "uncertainty_fusion_paper_summary_v1":
        errors.append("paper summary protocol mismatch")
    for condition in CONDITIONS:
        expected_hash = sha256_file(root / "evaluations" / condition / "summary.json")
        if comparison["evaluations"][condition].get("summary_sha256") != expected_hash:
            errors.append(f"{condition}: paper summary input hash mismatch")

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
        "protocol": "uncertainty_fusion_verification_v1",
        "verified": not errors,
        "errors": errors,
        "checks": checks,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
