"""Fit a conservative mask/vector router from SyncG train-only OOF pairs."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.collect_quality_router_oof import QUALITY_ROUTER_OOF_PROTOCOL
from experiments.quality_router import (
    FEATURE_NAMES,
    QUALITY_ROUTER_PROTOCOL,
    extract_quality_features,
    feature_matrix,
    finite_float,
    normalized_error,
    read_jsonl,
    route_prediction,
    sha256_file,
)


TRAINING_PROTOCOL = "syncg_grouped_cross_fit_quality_router_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-pairs",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/oof_clean.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/model"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def _build_model(args: argparse.Namespace, *, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                ExtraTreesRegressor(
                    n_estimators=int(args.trees),
                    max_depth=int(args.max_depth),
                    min_samples_leaf=int(args.min_samples_leaf),
                    max_features=float(args.max_features),
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise ValueError("router scores are all non-finite")
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 401))
    # A finite value immediately above the maximum represents the exact
    # no-switch policy without emitting non-standard JSON ``Infinity``.
    no_switch = np.nextafter(float(np.max(finite)), math.inf)
    if not math.isfinite(no_switch):
        raise ValueError("router scores are too large for a finite threshold")
    return np.unique(np.concatenate((quantiles, np.asarray([0.0, no_switch]))))


def _choose_threshold(
    scores: np.ndarray,
    base_errors: np.ndarray,
    vector_errors: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best: tuple[float, float, float] | None = None
    # Ties go to the larger threshold: fewer learned switches are safer.
    for threshold in _candidate_thresholds(scores):
        use_vector = scores > threshold
        routed = np.where(use_vector, vector_errors, base_errors)
        objective = float(np.mean(routed))
        switch_rate = float(np.mean(use_vector))
        key = (objective, switch_rate, -float(threshold))
        if best is None or key < best:
            best = key
    assert best is not None
    threshold = -best[2]
    use_vector = scores > threshold
    return threshold, {
        "nmae": float(np.mean(np.where(use_vector, vector_errors, base_errors))),
        "switch_rate": float(np.mean(use_vector)),
    }


def _metrics(errors: np.ndarray, successful: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(errors.size),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "coverage": float(np.mean(successful)),
        "successful": int(np.sum(successful)),
    }


def _paired_group_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int = 5000,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in selected])
        deltas[iteration] = float(
            np.mean(candidate[sample_indices]) - np.mean(baseline[sample_indices])
        )
    return {
        "delta_nmae": float(np.mean(candidate) - np.mean(baseline)),
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "iterations": int(iterations),
        "groups": int(len(unique)),
    }


def _joint_payload(row: dict[str, Any]) -> tuple[float | None, float | None]:
    return (
        finite_float((row.get("base") or {}).get("prediction")),
        finite_float((row.get("vector") or {}).get("prediction")),
    )


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.folds < 3 or args.trees <= 0 or args.min_samples_leaf <= 0:
        raise ValueError("invalid router training parameters")
    if not 0.0 < args.max_features <= 1.0:
        raise ValueError("max-features must be in (0, 1]")
    summary_path = args.oof_pairs.with_name(args.oof_pairs.stem + ".summary.json")
    if not args.oof_pairs.is_file() or not summary_path.is_file():
        raise FileNotFoundError("OOF pairs or their summary are missing")
    pair_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if pair_summary.get("protocol") != QUALITY_ROUTER_OOF_PROTOCOL:
        raise ValueError("input is not a signed quality-router OOF collection")
    if pair_summary.get("status") != "complete":
        raise ValueError("OOF collection is incomplete")
    if pair_summary.get("group_leakage_count") != 0 or pair_summary.get("test_samples_used") != 0:
        raise ValueError("OOF collection failed its leakage audit")
    if pair_summary.get("output_sha256") != sha256_file(args.oof_pairs):
        raise ValueError("OOF pairs changed after collection")

    rows = read_jsonl(args.oof_pairs)
    if any(str(row.get("dataset")) != "SyncG" or str(row.get("split")) != "train" for row in rows):
        raise ValueError("router training accepts only SyncG/train rows")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("OOF training rows contain duplicate sample IDs")
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    base_prediction = np.asarray([_joint_payload(row)[0] for row in rows], dtype=object)
    vector_prediction = np.asarray([_joint_payload(row)[1] for row in rows], dtype=object)
    base_success = np.asarray([value is not None for value in base_prediction], dtype=bool)
    vector_success = np.asarray([value is not None for value in vector_prediction], dtype=bool)
    joint = base_success & vector_success
    if int(np.sum(joint)) < 1000:
        raise ValueError("too few joint-success OOF rows for router training")
    feature_rows = [
        extract_quality_features(raw_row=row, base_row=row, vector_row=row)
        for row in rows
    ]
    matrix_all = feature_matrix(feature_rows)
    matrix = matrix_all[joint]
    groups = groups_all[joint]
    base_error_all = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, base_prediction)],
        dtype=np.float64,
    )
    vector_error_all = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, vector_prediction)],
        dtype=np.float64,
    )
    base_error = base_error_all[joint]
    vector_error = vector_error_all[joint]
    gain = np.clip(base_error - vector_error, -1.0, 1.0)

    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_score = np.full(len(rows), np.nan, dtype=np.float64)
    outer_fold = np.full(len(rows), -1, dtype=np.int64)
    joint_indices = np.flatnonzero(joint)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, gain, groups), start=1
    ):
        model = _build_model(args, seed=args.seed + fold)
        model.fit(matrix[train_index], gain[train_index])
        destination = joint_indices[validation_index]
        oof_score[destination] = model.predict(matrix[validation_index])
        outer_fold[destination] = fold
    if not np.isfinite(oof_score[joint]).all() or np.any(outer_fold[joint] < 1):
        raise RuntimeError("grouped OOF router predictions are incomplete")

    final_threshold, final_threshold_metrics = _choose_threshold(
        oof_score[joint], base_error, vector_error
    )
    nested_thresholds: dict[int, float] = {}
    nested_route = np.zeros(len(rows), dtype=bool)
    for fold in range(1, args.folds + 1):
        validation = joint & (outer_fold == fold)
        threshold_train = joint & (outer_fold != fold)
        threshold, _ = _choose_threshold(
            oof_score[threshold_train],
            base_error_all[threshold_train],
            vector_error_all[threshold_train],
        )
        nested_thresholds[fold] = threshold
        nested_route[validation] = oof_score[validation] > threshold

    # Base failures retain the pre-registered hard fallback independently of the model.
    nested_route[~base_success & vector_success] = True
    nested_prediction: list[float | None] = []
    route_names: list[str] = []
    final_routes: list[str] = []
    for index in range(len(rows)):
        if not base_success[index]:
            prediction = finite_float(vector_prediction[index])
            route = "vector_hard_fallback" if prediction is not None else "failure"
        elif nested_route[index] and vector_success[index]:
            prediction = finite_float(vector_prediction[index])
            route = "vector_quality_switch"
        else:
            prediction = finite_float(base_prediction[index])
            route = "base"
        nested_prediction.append(prediction)
        route_names.append(route)
        _, final_route = route_prediction(
            base_prediction=base_prediction[index],
            vector_prediction=vector_prediction[index],
            score=oof_score[index],
            threshold=final_threshold,
        )
        final_routes.append(final_route)
    nested_error = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, nested_prediction)],
        dtype=np.float64,
    )
    nested_success = np.asarray([value is not None for value in nested_prediction], dtype=bool)
    hard_prediction = [
        finite_float(base) if finite_float(base) is not None else finite_float(vector)
        for base, vector in zip(base_prediction, vector_prediction)
    ]
    hard_error = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, hard_prediction)],
        dtype=np.float64,
    )
    hard_success = base_success | vector_success
    oracle_error = np.minimum(base_error_all, vector_error_all)

    final_model = _build_model(args, seed=args.seed)
    final_model.fit(matrix, gain)
    model_path = args.output_dir / "quality_router.joblib"
    diagnostics_path = args.output_dir / "oof_routing.jsonl"
    training_summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, training_summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "protocol": QUALITY_ROUTER_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "feature_names": list(FEATURE_NAMES),
        "threshold": float(final_threshold),
        "estimator": final_model,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
        "seed": int(args.seed),
        "failure_policy": "base failure -> vector; joint failure -> failure",
        "test_sets_used": [],
        "source_sha256": {
            "features_and_policy": sha256_file(
                PROJECT_ROOT / "experiments" / "quality_router.py"
            ),
            "trainer": sha256_file(Path(__file__).resolve()),
        },
    }
    _atomic_joblib(model_path, artifact)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            payload = {
                "sample_id": row.get("sample_id"),
                "group_id": row.get("group_id"),
                "held_out_seed": row.get("held_out_seed"),
                "router_outer_fold": int(outer_fold[index]) if joint[index] else None,
                "router_score_oof": finite_float(oof_score[index]),
                "base_error": float(base_error_all[index]),
                "vector_error": float(vector_error_all[index]),
                "nested_route": route_names[index],
                "final_threshold_route_diagnostic": final_routes[index],
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    transformed_names = final_model.named_steps["imputer"].get_feature_names_out(
        list(FEATURE_NAMES)
    )
    importances = final_model.named_steps["regressor"].feature_importances_
    ranked_importances = [
        {"feature": str(name), "importance": float(value)}
        for name, value in sorted(
            zip(transformed_names, importances), key=lambda item: item[1], reverse=True
        )
    ]
    base_metrics = _metrics(base_error_all, base_success)
    vector_metrics = _metrics(vector_error_all, vector_success)
    hard_metrics = _metrics(hard_error, hard_success)
    nested_metrics = _metrics(nested_error, nested_success)
    oracle_metrics = _metrics(oracle_error, hard_success)
    switched = np.asarray([route == "vector_quality_switch" for route in route_names])
    positive_transfer = switched & (vector_error_all < base_error_all)
    negative_transfer = switched & (vector_error_all > base_error_all)
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "joint_training_samples": int(np.sum(joint)),
        "groups": int(len(np.unique(groups_all))),
        "group_folds": int(args.folds),
        "test_samples_used": 0,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "features": list(FEATURE_NAMES),
        "feature_importance": ranked_importances,
        "regressor": {
            "type": "ExtraTreesRegressor",
            "trees": int(args.trees),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": float(args.max_features),
            "target": "clipped(base_nmae - vector_nmae)",
        },
        "threshold": {
            "final": float(final_threshold),
            "selection": "minimum train-only grouped-OOF NMAE; ties prefer fewer switches",
            "selection_diagnostic": final_threshold_metrics,
            "nested_leave_fold_out": {
                str(fold): float(value) for fold, value in nested_thresholds.items()
            },
        },
        "metrics": {
            "base_mask": base_metrics,
            "vector": vector_metrics,
            "hard_fallback": hard_metrics,
            "quality_router_nested_oof": nested_metrics,
            "oracle": oracle_metrics,
        },
        "paired_vs_hard_fallback": _paired_group_bootstrap(
            nested_error, hard_error, groups_all, seed=args.seed
        ),
        "routing": {
            "counts": dict(sorted(Counter(route_names).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(np.sum(positive_transfer)),
            "negative_transfers": int(np.sum(negative_transfer)),
            "ties": int(np.sum(switched & (vector_error_all == base_error_all))),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "source_sha256": artifact["source_sha256"],
    }
    _atomic_json(training_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(training_summary_path)


if __name__ == "__main__":
    main()
