"""Train-only grouped-OOF feature ablations for the mask/vector router."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.quality_router import (
    FEATURE_NAMES,
    RAW_QUALITY_FEATURES,
    extract_quality_features,
    finite_float,
    normalized_error,
    read_jsonl,
    sha256_file,
)
from experiments.train_quality_router import (
    _build_model,
    _choose_threshold,
    _paired_group_bootstrap,
)


ABLATION_PROTOCOL = "syncg_quality_router_feature_ablation_v1"
DISAGREEMENT_FEATURES = FEATURE_NAMES[:10]
CALIBRATOR_FEATURES = (
    "gate_probability",
    "residual_abs_normalized",
    "residual_std_normalized",
    "correction_applied",
)
VECTOR_QUALITY_FEATURES = (
    "pivot_peak",
    "pivot_center_distance_fraction",
    "meter_confidence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-pairs",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/oof_clean.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/feature_ablation.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=600)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _prediction(row: dict[str, Any], side: str) -> float | None:
    return finite_float((row.get(side) or {}).get("prediction"))


def _configuration() -> dict[str, tuple[str, ...]]:
    disagreement = set(DISAGREEMENT_FEATURES)
    calibrator = set(CALIBRATOR_FEATURES)
    mask = set(RAW_QUALITY_FEATURES)
    compact = disagreement | set(VECTOR_QUALITY_FEATURES) | calibrator
    return {
        "full": FEATURE_NAMES,
        "compact_disagreement_uncertainty": tuple(
            name for name in FEATURE_NAMES if name in compact
        ),
        "without_cross_branch_disagreement": tuple(
            name for name in FEATURE_NAMES if name not in disagreement
        ),
        "without_calibrator_uncertainty": tuple(
            name for name in FEATURE_NAMES if name not in calibrator
        ),
        "without_mask_quality": tuple(
            name for name in FEATURE_NAMES if name not in mask
        ),
        "single_base_vector_disagreement": ("base_vector_progress_abs",),
    }


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output = args.output.resolve()
    markdown_path = args.output.with_suffix(".md")
    for path in (args.output, markdown_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    rows = read_jsonl(args.oof_pairs)
    if any(row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows):
        raise ValueError("feature ablations accept only SyncG/train OOF pairs")
    features = [
        extract_quality_features(raw_row=row, base_row=row, vector_row=row)
        for row in rows
    ]
    base_values = [_prediction(row, "base") for row in rows]
    vector_values = [_prediction(row, "vector") for row in rows]
    base_success = np.asarray([value is not None for value in base_values])
    vector_success = np.asarray([value is not None for value in vector_values])
    joint = base_success & vector_success
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    groups = groups_all[joint]
    base_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, base_values)]
    )
    vector_error = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, vector_values)]
    )
    target = np.clip(base_error[joint] - vector_error[joint], -1.0, 1.0)
    joint_indices = np.flatnonzero(joint)
    hard_error = np.where(base_success, base_error, vector_error)
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.zeros((len(groups), 1)), target, groups))
    results: dict[str, Any] = {}

    for configuration, names in _configuration().items():
        matrix_all = np.asarray(
            [[float(row[name]) for name in names] for row in features], dtype=np.float64
        )
        matrix = matrix_all[joint]
        score = np.full(len(rows), np.nan, dtype=np.float64)
        fold_id = np.full(len(rows), -1, dtype=np.int64)
        for fold, (train_index, validation_index) in enumerate(splits, start=1):
            model = _build_model(args, seed=args.seed + fold)
            model.fit(matrix[train_index], target[train_index])
            destination = joint_indices[validation_index]
            score[destination] = model.predict(matrix[validation_index])
            fold_id[destination] = fold
        routed_vector = np.zeros(len(rows), dtype=bool)
        nested_thresholds: dict[str, float] = {}
        for fold in range(1, args.folds + 1):
            validation = joint & (fold_id == fold)
            threshold_train = joint & (fold_id != fold)
            threshold, _ = _choose_threshold(
                score[threshold_train],
                base_error[threshold_train],
                vector_error[threshold_train],
            )
            nested_thresholds[str(fold)] = float(threshold)
            routed_vector[validation] = score[validation] > threshold
        routed_vector[~base_success & vector_success] = True
        routed_error = np.where(routed_vector, vector_error, base_error)
        routed_error[~base_success & ~vector_success] = 1.0
        learned_switch = routed_vector & base_success & vector_success
        positive = learned_switch & (vector_error < base_error)
        negative = learned_switch & (vector_error > base_error)
        comparison = _paired_group_bootstrap(
            routed_error,
            hard_error,
            groups_all,
            seed=args.seed,
            iterations=5000,
        )
        results[configuration] = {
            "features": list(names),
            "feature_count": len(names),
            "nmae": float(np.mean(routed_error)),
            "acc_2pct": float(np.mean(routed_error <= 0.02)),
            "quality_switches": int(np.sum(learned_switch)),
            "positive_transfers": int(np.sum(positive)),
            "negative_transfers": int(np.sum(negative)),
            "nested_thresholds": nested_thresholds,
            "paired_vs_hard_fallback": comparison,
        }

    payload = {
        "schema_version": 1,
        "protocol": ABLATION_PROTOCOL,
        "status": "complete",
        "split": "SyncG/train grouped OOF only",
        "test_samples_used": 0,
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "groups": int(len(np.unique(groups_all))),
        "hard_fallback": {
            "nmae": float(np.mean(hard_error)),
            "acc_2pct": float(np.mean(hard_error <= 0.02)),
        },
        "configurations": results,
    }
    _atomic_write(
        args.output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Quality-router feature ablation (SyncG/train grouped OOF)",
        "",
        "No SyncG test or RPM-10K sample is used for feature or threshold selection.",
        "",
        "| Variant | Features | NMAE | Acc@2% | Switches | Positive / negative | Δ vs hard (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        comparison = result["paired_vs_hard_fallback"]
        interval = comparison["group_bootstrap_95ci"]
        lines.append(
            f"| {name} | {result['feature_count']} | {result['nmae']:.4f} | "
            f"{result['acc_2pct']:.4f} | {result['quality_switches']} | "
            f"{result['positive_transfers']} / {result['negative_transfers']} | "
            f"{comparison['delta_nmae']:+.4f} [{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    _atomic_write(markdown_path, "\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
