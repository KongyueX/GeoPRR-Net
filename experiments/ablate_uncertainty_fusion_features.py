"""Grouped nested-OOF feature ablation for the uncertainty fusion model."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.train_uncertainty_fusion import _fit_model, _predict
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    NATIVE_UNCERTAINTY_FEATURE_NAMES,
    UNCERTAINTY_FUSION_OOF_PROTOCOL,
    extract_uncertainty_features,
    feature_matrix,
    soft_fusion_prediction,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


ABLATION_PROTOCOL = "grouped_oof_uncertainty_fusion_feature_ablation_v1"


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
        "--full-training-summary",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/model/training_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/uncertainty_fusion_syncg/feature_ablation.json"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fusion-loss-weight", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


def _prediction(row: dict[str, Any], name: str) -> float | None:
    return finite_float((row.get(name) or {}).get("prediction"))


def _progress(row: dict[str, Any], prediction: float | None) -> float | None:
    start = finite_float(row.get("scale_start"))
    end = finite_float(row.get("scale_end"))
    if prediction is None or start is None or end is None or abs(end - start) <= 1e-12:
        return None
    return (prediction - start) / (end - start)


def _metrics(rows: list[dict[str, Any]], values: list[float | None]) -> dict[str, Any]:
    errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, values)],
        dtype=np.float64,
    )
    return {
        "samples": len(rows),
        "coverage": float(np.mean([value is not None for value in values])),
        "nmae": float(np.mean(errors)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
    }


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
    args.oof_pairs = args.oof_pairs.resolve()
    args.full_training_summary = args.full_training_summary.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    for path in (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
        args.full_training_summary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(_metadata_path(args.oof_pairs).read_text(encoding="utf-8"))
    oof_summary = json.loads(_summary_path(args.oof_pairs).read_text(encoding="utf-8"))
    full_summary = json.loads(args.full_training_summary.read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("feature ablation requires signed probabilistic OOF rows")
    if (
        oof_summary.get("status") != "complete"
        or int(oof_summary.get("group_leakage_count", -1)) != 0
        or int(oof_summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("feature ablation input fails train-only audit")
    if full_summary.get("input_sha256") != sha256_file(args.oof_pairs):
        raise ValueError("full fusion summary was trained from another OOF file")

    rows = read_jsonl(args.oof_pairs)
    base = [_prediction(row, "base") for row in rows]
    vector = [_prediction(row, "vector") for row in rows]
    joint = np.asarray(
        [first is not None and second is not None for first, second in zip(base, vector)]
    )
    truth = np.asarray(
        [_progress(row, finite_float(row.get("ground_truth"))) for row in rows],
        dtype=np.float64,
    )
    base_progress = np.asarray(
        [_progress(row, value) for row, value in zip(rows, base)], dtype=np.float64
    )
    vector_progress = np.asarray(
        [_progress(row, value) for row, value in zip(rows, vector)], dtype=np.float64
    )
    matrix = feature_matrix(
        [
            extract_uncertainty_features(raw_row=row, base_row=row, vector_row=row)
            for row in rows
        ]
    )
    groups = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    joint_indices = np.flatnonzero(joint)
    joint_groups = groups[joint]
    splitter = GroupKFold(n_splits=args.folds)
    native_start = len(FEATURE_NAMES) - len(NATIVE_UNCERTAINTY_FEATURE_NAMES)
    variants = {
        "quality_features_only_no_native_uncertainty": np.arange(
            native_start, len(FEATURE_NAMES)
        ),
        "native_uncertainty_only": np.arange(0, native_start),
    }
    results: dict[str, Any] = {
        "full_all_features": full_summary["metrics"][
            "uncertainty_fusion_nested_oof"
        ]
    }
    for variant_index, (name, excluded) in enumerate(variants.items()):
        variant_matrix = matrix.copy()
        variant_matrix[:, excluded] = math.nan
        log_variance = np.full((len(rows), 2), math.nan, dtype=np.float64)
        fold_summaries: list[dict[str, Any]] = []
        for fold, (train_index, validation_index) in enumerate(
            splitter.split(variant_matrix[joint], groups=joint_groups), start=1
        ):
            train_groups = set(joint_groups[train_index].tolist())
            validation_groups = set(joint_groups[validation_index].tolist())
            if train_groups & validation_groups:
                raise RuntimeError("feature ablation group leakage")
            model, medians, scales, fit = _fit_model(
                train_matrix=variant_matrix[joint][train_index],
                train_base_error=np.clip(
                    base_progress[joint][train_index] - truth[joint][train_index],
                    -1.0,
                    1.0,
                ),
                train_vector_error=np.clip(
                    vector_progress[joint][train_index] - truth[joint][train_index],
                    -1.0,
                    1.0,
                ),
                train_base_progress=base_progress[joint][train_index],
                train_vector_progress=vector_progress[joint][train_index],
                train_truth_progress=truth[joint][train_index],
                validation_matrix=variant_matrix[joint][validation_index],
                validation_base_progress=base_progress[joint][validation_index],
                validation_vector_progress=vector_progress[joint][validation_index],
                validation_truth_progress=truth[joint][validation_index],
                args=args,
                seed=args.seed + variant_index * 10 + fold,
            )
            log_variance[joint_indices[validation_index]] = _predict(
                model,
                variant_matrix[joint][validation_index],
                medians=medians,
                scales=scales,
                device=torch.device(args.device),
            )
            fold_summaries.append(
                {
                    "fold": fold,
                    "train_groups": len(train_groups),
                    "validation_groups": len(validation_groups),
                    "group_overlap": 0,
                    "best_epoch": fit["best_epoch"],
                    "best_selection_mae": fit["best_selection_mae"],
                }
            )
        values: list[float | None] = []
        for index, row in enumerate(rows):
            variances = log_variance[index] if joint[index] else (math.nan, math.nan)
            prediction, _, _, _ = soft_fusion_prediction(
                base_prediction=base[index],
                vector_prediction=vector[index],
                scale_start=row.get("scale_start"),
                scale_end=row.get("scale_end"),
                mask_log_variance=variances[0],
                vector_log_variance=variances[1],
                temperature=args.temperature,
            )
            values.append(prediction)
        results[name] = {
            **_metrics(rows, values),
            "excluded_features": [FEATURE_NAMES[index] for index in excluded],
            "folds": fold_summaries,
        }
    payload = {
        "schema_version": 1,
        "protocol": ABLATION_PROTOCOL,
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "results": results,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "feature_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ),
        "trainer_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "train_uncertainty_fusion.py"
        ),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
