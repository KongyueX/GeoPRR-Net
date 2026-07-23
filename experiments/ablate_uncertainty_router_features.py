"""Grouped-OOF feature ablation for the probabilistic uncertainty router."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.train_quality_router import _choose_threshold, _metrics
from experiments.train_uncertainty_router import _build_model
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    NATIVE_UNCERTAINTY_FEATURE_NAMES,
    UNCERTAINTY_FUSION_OOF_PROTOCOL,
    extract_uncertainty_features,
    feature_matrix,
)
from experiments.vdn_baseline import sha256_file


ABLATION_PROTOCOL = "grouped_oof_uncertainty_router_feature_ablation_v1"


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
        "--output",
        type=Path,
        default=Path("artifacts/runs/uncertainty_router_syncg/feature_ablation.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prediction(row: dict[str, Any], name: str) -> float | None:
    return finite_float((row.get(name) or {}).get("prediction"))


def _evaluate(
    *,
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    joint: np.ndarray,
    base: Sequence[float | None],
    vector: Sequence[float | None],
    base_error: np.ndarray,
    vector_error: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    joint_indices = np.flatnonzero(joint)
    joint_groups = groups[joint]
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scores = np.full(len(groups), math.nan, dtype=np.float64)
    fold_ids = np.full(len(groups), -1, dtype=np.int64)
    folds: list[dict[str, int]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, target, joint_groups), start=1
    ):
        train_groups = set(joint_groups[train_index].tolist())
        validation_groups = set(joint_groups[validation_index].tolist())
        overlap = len(train_groups & validation_groups)
        if overlap:
            raise RuntimeError("feature-ablation group leakage")
        model = _build_model(args, seed=args.seed + fold)
        model.fit(matrix[train_index], target[train_index])
        destination = joint_indices[validation_index]
        scores[destination] = model.predict(matrix[validation_index])
        fold_ids[destination] = fold
        folds.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "group_overlap": overlap,
            }
        )
    use_vector = np.zeros(len(groups), dtype=bool)
    nested_thresholds: dict[str, float] = {}
    for fold in range(1, args.folds + 1):
        validation = joint & (fold_ids == fold)
        threshold_train = joint & (fold_ids != fold)
        threshold, _ = _choose_threshold(
            scores[threshold_train],
            base_error[threshold_train],
            vector_error[threshold_train],
        )
        nested_thresholds[str(fold)] = float(threshold)
        use_vector[validation] = scores[validation] > threshold
    base_success = np.asarray([value is not None for value in base])
    vector_success = np.asarray([value is not None for value in vector])
    use_vector[~base_success & vector_success] = True
    predictions = [
        vector_value if switch and vector_value is not None else base_value
        for switch, base_value, vector_value in zip(use_vector, base, vector)
    ]
    errors = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, predictions)
        ],
        dtype=np.float64,
    )
    successful = np.asarray([value is not None for value in predictions])
    switched = joint & use_vector
    metrics = _metrics(errors, successful)
    metrics.update(
        {
            "switches": int(np.sum(switched)),
            "positive_transfers": int(
                np.sum(switched & (vector_error < base_error))
            ),
            "negative_transfers": int(
                np.sum(switched & (vector_error > base_error))
            ),
            "nested_thresholds": nested_thresholds,
        }
    )
    return metrics, folds


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    metadata_path = args.oof_pairs.with_name(args.oof_pairs.name + ".meta.json")
    summary_path = args.oof_pairs.with_name(args.oof_pairs.stem + ".summary.json")
    for path in (args.oof_pairs, metadata_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (metadata.get("signature") or {}).get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("input has the wrong OOF protocol")
    if (
        summary.get("status") != "complete"
        or summary.get("output_sha256") != sha256_file(args.oof_pairs)
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("input OOF collection failed its train-only audit")
    rows = read_jsonl(args.oof_pairs)
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("feature ablation accepts only SyncG/train rows")
    base = [_prediction(row, "base") for row in rows]
    vector = [_prediction(row, "vector") for row in rows]
    base_success = np.asarray([value is not None for value in base])
    vector_success = np.asarray([value is not None for value in vector])
    joint = base_success & vector_success
    groups = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    base_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, base)],
        dtype=np.float64,
    )
    vector_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, vector)],
        dtype=np.float64,
    )
    target = np.clip(base_error[joint] - vector_error[joint], -1.0, 1.0)
    full_matrix = feature_matrix(
        [
            extract_uncertainty_features(raw_row=row, base_row=row, vector_row=row)
            for row in rows
        ]
    )
    native_count = len(NATIVE_UNCERTAINTY_FEATURE_NAMES)
    feature_sets = {
        "full_quality_plus_native_uncertainty": np.arange(len(FEATURE_NAMES)),
        "quality_only": np.arange(len(FEATURE_NAMES) - native_count),
        "native_uncertainty_only": np.arange(
            len(FEATURE_NAMES) - native_count, len(FEATURE_NAMES)
        ),
    }
    results: dict[str, Any] = {}
    folds: dict[str, Any] = {}
    for name, columns in feature_sets.items():
        results[name], folds[name] = _evaluate(
            args=args,
            rows=rows,
            matrix=full_matrix[joint][:, columns],
            target=target,
            groups=groups,
            joint=joint,
            base=base,
            vector=vector,
            base_error=base_error,
            vector_error=vector_error,
        )
    payload = {
        "schema_version": 1,
        "protocol": ABLATION_PROTOCOL,
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "groups": len(set(groups.tolist())),
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_sets": {
            name: [FEATURE_NAMES[index] for index in columns]
            for name, columns in feature_sets.items()
        },
        "results": results,
        "folds": folds,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
