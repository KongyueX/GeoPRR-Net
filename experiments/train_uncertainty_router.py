"""Train an uncertainty-aware selective router on probabilistic grouped OOF rows."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold

from experiments.quality_router import finite_float, normalized_error, read_jsonl, route_prediction
from experiments.train_quality_router import (
    _build_model,
    _choose_threshold,
    _metrics,
    _paired_group_bootstrap,
)
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    UNCERTAINTY_FUSION_OOF_PROTOCOL,
    extract_uncertainty_features,
    feature_matrix,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


UNCERTAINTY_ROUTER_PROTOCOL = "mask_vector_probabilistic_uncertainty_router_v1"
TRAINING_PROTOCOL = "syncg_grouped_cross_fit_probabilistic_uncertainty_router_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-pairs",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/probabilistic_oof_clean.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/runs/uncertainty_router_syncg/model"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


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


def _prediction(row: dict[str, Any], name: str) -> float | None:
    return finite_float((row.get(name) or {}).get("prediction"))


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.folds < 3 or args.trees <= 0 or args.min_samples_leaf <= 0:
        raise ValueError("invalid uncertainty-router training parameters")
    for path in (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(_metadata_path(args.oof_pairs).read_text(encoding="utf-8"))
    pair_summary = json.loads(_summary_path(args.oof_pairs).read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("input is not a signed probabilistic OOF collection")
    if (
        pair_summary.get("status") != "complete"
        or int(pair_summary.get("group_leakage_count", -1)) != 0
        or int(pair_summary.get("test_samples_used", -1)) != 0
        or pair_summary.get("output_sha256") != sha256_file(args.oof_pairs)
    ):
        raise ValueError("probabilistic OOF collection failed its train-only audit")

    rows = read_jsonl(args.oof_pairs)
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("uncertainty router accepts only SyncG/train rows")
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("uncertainty-router OOF rows contain duplicate IDs")
    base = [_prediction(row, "base") for row in rows]
    vector = [_prediction(row, "vector") for row in rows]
    base_success = np.asarray([value is not None for value in base], dtype=bool)
    vector_success = np.asarray([value is not None for value in vector], dtype=bool)
    joint = base_success & vector_success
    if int(np.sum(joint)) < 1000:
        raise ValueError("too few joint-success OOF rows")
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    groups = groups_all[joint]
    matrix_all = feature_matrix(
        [
            extract_uncertainty_features(raw_row=row, base_row=row, vector_row=row)
            for row in rows
        ]
    )
    matrix = matrix_all[joint]
    base_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, base)],
        dtype=np.float64,
    )
    vector_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, vector)],
        dtype=np.float64,
    )
    target = np.clip(base_error[joint] - vector_error[joint], -1.0, 1.0)
    joint_indices = np.flatnonzero(joint)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    score = np.full(len(rows), math.nan, dtype=np.float64)
    fold_id = np.full(len(rows), -1, dtype=np.int64)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("uncertainty-router group leakage")
        model = _build_model(args, seed=args.seed + fold)
        model.fit(matrix[train_index], target[train_index])
        destination = joint_indices[validation_index]
        score[destination] = model.predict(matrix[validation_index])
        fold_id[destination] = fold
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
            }
        )
    if not np.isfinite(score[joint]).all() or np.any(fold_id[joint] < 1):
        raise RuntimeError("uncertainty-router grouped OOF scores are incomplete")

    final_threshold, threshold_diagnostic = _choose_threshold(
        score[joint], base_error[joint], vector_error[joint]
    )
    nested_thresholds: dict[int, float] = {}
    use_vector = np.zeros(len(rows), dtype=bool)
    for fold in range(1, args.folds + 1):
        validation = joint & (fold_id == fold)
        threshold_train = joint & (fold_id != fold)
        threshold, _ = _choose_threshold(
            score[threshold_train],
            base_error[threshold_train],
            vector_error[threshold_train],
        )
        nested_thresholds[fold] = float(threshold)
        use_vector[validation] = score[validation] > threshold
    use_vector[~base_success & vector_success] = True

    nested_predictions: list[float | None] = []
    route_names: list[str] = []
    for index in range(len(rows)):
        if not base_success[index]:
            value = vector[index]
            route = "vector_hard_fallback" if value is not None else "failure"
        elif use_vector[index] and vector_success[index]:
            value = vector[index]
            route = "vector_uncertainty_switch"
        else:
            value = base[index]
            route = "base"
        nested_predictions.append(value)
        route_names.append(route)
    nested_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, nested_predictions)],
        dtype=np.float64,
    )
    hard_values = [first if first is not None else second for first, second in zip(base, vector)]
    hard_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, hard_values)],
        dtype=np.float64,
    )
    hard_success = base_success | vector_success
    oracle_error = np.minimum(base_error, vector_error)
    nested_success = np.asarray([value is not None for value in nested_predictions])

    final_model = _build_model(args, seed=args.seed)
    final_model.fit(matrix, target)
    model_path = args.output_dir / "uncertainty_router.joblib"
    diagnostics_path = args.output_dir / "nested_oof_routing.jsonl"
    summary_path = args.output_dir / "training_summary.json"
    for path in (model_path, diagnostics_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    source_hashes = {
        "features": sha256_file(PROJECT_DIR / "experiments" / "uncertainty_fusion.py"),
        "policy": sha256_file(PROJECT_DIR / "experiments" / "quality_router.py"),
        "trainer": sha256_file(Path(__file__).resolve()),
        "shared_training_helpers": sha256_file(
            PROJECT_DIR / "experiments" / "train_quality_router.py"
        ),
    }
    artifact = {
        "protocol": UNCERTAINTY_ROUTER_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "feature_names": list(FEATURE_NAMES),
        "threshold": float(final_threshold),
        "estimator": final_model,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
        "failure_policy": "base failure -> vector; joint failure -> failure",
        "seed": int(args.seed),
        "test_sets_used": [],
        "source_sha256": source_hashes,
    }
    _atomic_joblib(model_path, artifact)
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row.get("sample_id"),
                        "group_id": row.get("group_id"),
                        "held_out_seed": row.get("held_out_seed"),
                        "router_fold": int(fold_id[index]) if joint[index] else None,
                        "router_score_oof": finite_float(score[index]),
                        "nested_route": route_names[index],
                        "base_error": float(base_error[index]),
                        "vector_error": float(vector_error[index]),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    transformed_names = final_model.named_steps["imputer"].get_feature_names_out(
        list(FEATURE_NAMES)
    )
    importances = final_model.named_steps["regressor"].feature_importances_
    ranked = [
        {"feature": str(name), "importance": float(value)}
        for name, value in sorted(
            zip(transformed_names, importances), key=lambda item: item[1], reverse=True
        )
    ]
    switched = np.asarray([name == "vector_uncertainty_switch" for name in route_names])
    positive = switched & (vector_error < base_error)
    negative = switched & (vector_error > base_error)
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "groups": len(set(groups_all.tolist())),
        "joint_training_samples": int(np.sum(joint)),
        "folds": args.folds,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "fold_summaries": fold_summaries,
        "feature_names": list(FEATURE_NAMES),
        "feature_importance": ranked,
        "threshold": {
            "final": float(final_threshold),
            "selection_diagnostic": threshold_diagnostic,
            "nested_leave_fold_out": {
                str(key): value for key, value in nested_thresholds.items()
            },
        },
        "metrics": {
            "base_mask": _metrics(base_error, base_success),
            "probabilistic_vector": _metrics(vector_error, vector_success),
            "hard_fallback": _metrics(hard_error, hard_success),
            "uncertainty_router_nested_oof": _metrics(nested_error, nested_success),
            "oracle": _metrics(oracle_error, hard_success),
        },
        "paired_vs_hard_fallback": _paired_group_bootstrap(
            nested_error, hard_error, groups_all, seed=args.seed
        ),
        "routing": {
            "counts": dict(sorted(Counter(route_names).items())),
            "uncertainty_switches": int(np.sum(switched)),
            "positive_transfers": int(np.sum(positive)),
            "negative_transfers": int(np.sum(negative)),
        },
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "source_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
