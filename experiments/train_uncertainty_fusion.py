"""Fit a grouped-OOF uncertainty calibrator for mask/vector soft fusion."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    UNCERTAINTY_FUSION_OOF_PROTOCOL,
    UNCERTAINTY_FUSION_PROTOCOL,
    DualVarianceMLP,
    extract_uncertainty_features,
    feature_matrix,
    fit_robust_preprocessor,
    inverse_variance_weights,
    soft_fusion_prediction,
    transform_feature_matrix,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file


TRAINING_PROTOCOL = "syncg_grouped_cross_fit_uncertainty_fusion_v1"


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
        default=Path("artifacts/runs/uncertainty_fusion_syncg/model"),
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


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".summary.json")


def _prediction(row: dict[str, Any], name: str) -> float | None:
    payload = row.get(name) or {}
    return finite_float(payload.get("prediction"))


def _progress(row: dict[str, Any], prediction: float | None) -> float | None:
    start = finite_float(row.get("scale_start"))
    end = finite_float(row.get("scale_end"))
    if prediction is None or start is None or end is None or abs(end - start) <= 1e-12:
        return None
    return (prediction - start) / (end - start)


def _truth_progress(row: dict[str, Any]) -> float:
    truth = finite_float(row.get("ground_truth"))
    value = _progress(row, truth)
    if value is None:
        raise ValueError(f"{row.get('sample_id')}: invalid reading range")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _variance_loss(
    model: DualVarianceMLP,
    features: torch.Tensor,
    base_error: torch.Tensor,
    vector_error: torch.Tensor,
    base_progress: torch.Tensor,
    vector_progress: torch.Tensor,
    truth_progress: torch.Tensor,
    *,
    fusion_loss_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_variance = model(features)
    nll = 0.5 * (
        base_error.square() * torch.exp(-log_variance[:, 0])
        + log_variance[:, 0]
        + vector_error.square() * torch.exp(-log_variance[:, 1])
        + log_variance[:, 1]
    )
    weights = inverse_variance_weights(log_variance, temperature=temperature)
    fused = weights[:, 0] * base_progress + weights[:, 1] * vector_progress
    fusion = F.smooth_l1_loss(fused, truth_progress, beta=0.02)
    total = torch.mean(nll) + float(fusion_loss_weight) * fusion
    return total, {
        "nll": torch.mean(nll).detach(),
        "fusion_loss": fusion.detach(),
        "fused_mae": torch.mean(torch.abs(fused - truth_progress)).detach(),
        "mean_mask_weight": torch.mean(weights[:, 0]).detach(),
    }


@torch.no_grad()
def _predict(
    model: DualVarianceMLP,
    matrix: np.ndarray,
    *,
    medians: np.ndarray,
    scales: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    transformed = transform_feature_matrix(matrix, medians=medians, scales=scales)
    output: list[np.ndarray] = []
    for start in range(0, len(transformed), 1024):
        tensor = torch.from_numpy(transformed[start : start + 1024]).to(device)
        output.append(model(tensor).cpu().numpy())
    return np.concatenate(output, axis=0) if output else np.empty((0, 2))


def _fit_model(
    *,
    train_matrix: np.ndarray,
    train_base_error: np.ndarray,
    train_vector_error: np.ndarray,
    train_base_progress: np.ndarray,
    train_vector_progress: np.ndarray,
    train_truth_progress: np.ndarray,
    validation_matrix: np.ndarray | None,
    validation_base_progress: np.ndarray | None,
    validation_vector_progress: np.ndarray | None,
    validation_truth_progress: np.ndarray | None,
    args: argparse.Namespace,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[DualVarianceMLP, np.ndarray, np.ndarray, dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(args.device)
    medians, scales = fit_robust_preprocessor(train_matrix)
    transformed = transform_feature_matrix(
        train_matrix,
        medians=medians,
        scales=scales,
    )
    dataset = TensorDataset(
        torch.from_numpy(transformed),
        torch.from_numpy(train_base_error.astype(np.float32)),
        torch.from_numpy(train_vector_error.astype(np.float32)),
        torch.from_numpy(train_base_progress.astype(np.float32)),
        torch.from_numpy(train_vector_progress.astype(np.float32)),
        torch.from_numpy(train_truth_progress.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = DualVarianceMLP(
        input_features=len(FEATURE_NAMES),
        hidden_features=args.hidden_features,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    maximum_epochs = int(fixed_epochs or args.epochs)
    best_metric = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "nll": 0.0, "fusion_loss": 0.0, "fused_mae": 0.0}
        seen = 0
        for batch in loader:
            values = [value.to(device) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            loss, components = _variance_loss(
                model,
                *values,
                fusion_loss_weight=args.fusion_loss_weight,
                temperature=args.temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = int(values[0].shape[0])
            seen += count
            totals["loss"] += float(loss.detach()) * count
            for name in ("nll", "fusion_loss", "fused_mae"):
                totals[name] += float(components[name]) * count
        train_metrics = {name: value / max(seen, 1) for name, value in totals.items()}
        if validation_matrix is not None:
            log_variance = _predict(
                model,
                validation_matrix,
                medians=medians,
                scales=scales,
                device=device,
            )
            logits = -log_variance / float(args.temperature)
            logits -= np.max(logits, axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= np.sum(weights, axis=1, keepdims=True)
            fused = (
                weights[:, 0] * validation_base_progress
                + weights[:, 1] * validation_vector_progress
            )
            metric = float(np.mean(np.abs(fused - validation_truth_progress)))
        else:
            metric = float(train_metrics["fused_mae"])
        history.append({"epoch": epoch, **train_metrics, "selection_mae": metric})
        if fixed_epochs is not None:
            best_epoch = epoch
            best_metric = metric
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            continue
        if metric < best_metric - 1e-7:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("uncertainty model training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, medians, scales, {
        "best_epoch": best_epoch,
        "best_selection_mae": best_metric,
        "epochs_ran": len(history),
        "history": history,
    }


def _metrics(
    rows: Sequence[dict[str, Any]], predictions: Sequence[float | None]
) -> dict[str, float | int]:
    errors = np.asarray(
        [normalized_error(row, prediction) for row, prediction in zip(rows, predictions)],
        dtype=np.float64,
    )
    success = np.asarray([prediction is not None for prediction in predictions])
    return {
        "samples": len(rows),
        "nmae": float(np.mean(errors)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "coverage": float(np.mean(success)),
        "failures": int(np.sum(~success)),
    }


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.folds < 2 or args.epochs <= 0 or args.patience <= 0:
        raise ValueError("folds/epochs/patience are invalid")
    if args.batch_size <= 0 or args.hidden_features < 4:
        raise ValueError("batch-size/hidden-features are invalid")
    if args.temperature <= 0.0 or args.fusion_loss_weight < 0.0:
        raise ValueError("temperature must be positive and fusion weight non-negative")
    for path in (
        args.oof_pairs,
        _metadata_path(args.oof_pairs),
        _summary_path(args.oof_pairs),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(_metadata_path(args.oof_pairs).read_text(encoding="utf-8"))
    oof_summary = json.loads(_summary_path(args.oof_pairs).read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("protocol") != UNCERTAINTY_FUSION_OOF_PROTOCOL:
        raise ValueError("input is not a signed probabilistic OOF collection")
    if oof_summary.get("status") != "complete":
        raise ValueError("probabilistic OOF collection is incomplete")
    if int(oof_summary.get("group_leakage_count", -1)) != 0:
        raise ValueError("probabilistic OOF collection reports group leakage")
    if int(oof_summary.get("test_samples_used", -1)) != 0:
        raise ValueError("test samples were used to construct fusion supervision")
    rows = read_jsonl(args.oof_pairs)
    if not rows or {row.get("dataset") for row in rows} != {"SyncG"}:
        raise ValueError("fusion training accepts only SyncG rows")
    if {row.get("split") for row in rows} != {"train"}:
        raise ValueError("fusion training accepts only SyncG/train rows")

    base_predictions = [_prediction(row, "base") for row in rows]
    vector_predictions = [_prediction(row, "vector") for row in rows]
    joint = np.asarray(
        [base is not None and vector is not None for base, vector in zip(base_predictions, vector_predictions)],
        dtype=bool,
    )
    if int(np.sum(joint)) < 500:
        raise ValueError("too few joint-success OOF rows for uncertainty training")
    feature_rows = [
        extract_uncertainty_features(raw_row=row, base_row=row, vector_row=row)
        for row in rows
    ]
    matrix_all = feature_matrix(feature_rows)
    truth_all = np.asarray([_truth_progress(row) for row in rows], dtype=np.float64)
    base_progress_all = np.asarray(
        [
            _progress(row, prediction) if prediction is not None else math.nan
            for row, prediction in zip(rows, base_predictions)
        ],
        dtype=np.float64,
    )
    vector_progress_all = np.asarray(
        [
            _progress(row, prediction) if prediction is not None else math.nan
            for row, prediction in zip(rows, vector_predictions)
        ],
        dtype=np.float64,
    )
    matrix = matrix_all[joint]
    base_progress = base_progress_all[joint]
    vector_progress = vector_progress_all[joint]
    truth = truth_all[joint]
    base_error = np.clip(base_progress - truth, -1.0, 1.0)
    vector_error = np.clip(vector_progress - truth, -1.0, 1.0)
    groups_all = np.asarray([str(row.get("group_id")) for row in rows], dtype=object)
    groups = groups_all[joint]
    if len(set(groups.tolist())) < args.folds:
        raise ValueError("too few groups for grouped cross fitting")

    splitter = GroupKFold(n_splits=args.folds)
    oof_log_variance = np.full((len(rows), 2), np.nan, dtype=np.float64)
    joint_indices = np.flatnonzero(joint)
    fold_summaries: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(matrix, groups=groups), start=1
    ):
        train_groups = set(groups[train_index].tolist())
        validation_groups = set(groups[validation_index].tolist())
        if train_groups & validation_groups:
            raise RuntimeError("group leakage in uncertainty cross fitting")
        model, medians, scales, fit_summary = _fit_model(
            train_matrix=matrix[train_index],
            train_base_error=base_error[train_index],
            train_vector_error=vector_error[train_index],
            train_base_progress=base_progress[train_index],
            train_vector_progress=vector_progress[train_index],
            train_truth_progress=truth[train_index],
            validation_matrix=matrix[validation_index],
            validation_base_progress=base_progress[validation_index],
            validation_vector_progress=vector_progress[validation_index],
            validation_truth_progress=truth[validation_index],
            args=args,
            seed=args.seed + fold,
        )
        predictions = _predict(
            model,
            matrix[validation_index],
            medians=medians,
            scales=scales,
            device=torch.device(args.device),
        )
        oof_log_variance[joint_indices[validation_index]] = predictions
        fold_summaries.append(
            {
                "fold": fold,
                "train_samples": len(train_index),
                "validation_samples": len(validation_index),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": 0,
                **fit_summary,
            }
        )
    if not np.isfinite(oof_log_variance[joint]).all():
        raise RuntimeError("grouped OOF uncertainty predictions are incomplete")

    fusion_predictions: list[float | None] = []
    routes: list[str] = []
    mask_weights: list[float | None] = []
    for index, row in enumerate(rows):
        if joint[index]:
            variances = oof_log_variance[index]
        else:
            variances = (math.nan, math.nan)
        prediction, route, mask_weight, _ = soft_fusion_prediction(
            base_prediction=base_predictions[index],
            vector_prediction=vector_predictions[index],
            scale_start=row.get("scale_start"),
            scale_end=row.get("scale_end"),
            mask_log_variance=variances[0],
            vector_log_variance=variances[1],
            temperature=args.temperature,
        )
        fusion_predictions.append(prediction)
        routes.append(route)
        mask_weights.append(mask_weight)
    hard_predictions = [
        base if base is not None else vector
        for base, vector in zip(base_predictions, vector_predictions)
    ]
    oracle_predictions: list[float | None] = []
    for row, base, vector in zip(rows, base_predictions, vector_predictions):
        if base is None:
            oracle_predictions.append(vector)
        elif vector is None:
            oracle_predictions.append(base)
        else:
            oracle_predictions.append(
                base
                if normalized_error(row, base) <= normalized_error(row, vector)
                else vector
            )

    selected_epochs = [int(item["best_epoch"]) for item in fold_summaries]
    final_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    final_model, medians, scales, final_fit = _fit_model(
        train_matrix=matrix,
        train_base_error=base_error,
        train_vector_error=vector_error,
        train_base_progress=base_progress,
        train_vector_progress=vector_progress,
        train_truth_progress=truth,
        validation_matrix=None,
        validation_base_progress=None,
        validation_vector_progress=None,
        validation_truth_progress=None,
        args=args,
        seed=args.seed,
        fixed_epochs=final_epochs,
    )
    model_path = args.output_dir / "uncertainty_fusion.pt"
    summary_path = args.output_dir / "training_summary.json"
    oof_output = args.output_dir / "nested_oof_predictions.jsonl"
    if any(path.exists() for path in (model_path, summary_path, oof_output)) and not args.overwrite:
        raise FileExistsError("uncertainty fusion outputs exist; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "protocol": UNCERTAINTY_FUSION_PROTOCOL,
        "training_protocol": TRAINING_PROTOCOL,
        "train_only_certified": True,
        "feature_names": FEATURE_NAMES,
        "hidden_features": int(args.hidden_features),
        "model_state": {
            name: tensor.detach().cpu() for name, tensor in final_model.state_dict().items()
        },
        "preprocessor_medians": medians,
        "preprocessor_scales": scales,
        "temperature": float(args.temperature),
        "final_epochs": final_epochs,
        "training_oof_pairs_sha256": sha256_file(args.oof_pairs),
        "feature_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
        ),
        "trainer_source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": int(args.seed),
    }
    _atomic_torch(model_path, artifact)
    with oof_output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row.get("sample_id"),
                        "group_id": row.get("group_id"),
                        "mask_log_variance_oof": finite_float(oof_log_variance[index, 0]),
                        "vector_log_variance_oof": finite_float(oof_log_variance[index, 1]),
                        "mask_weight_oof": mask_weights[index],
                        "prediction": fusion_predictions[index],
                        "route": routes[index],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    summary = {
        "schema_version": 1,
        "protocol": TRAINING_PROTOCOL,
        "status": "complete",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "groups": len(set(groups_all.tolist())),
        "joint_success_samples": int(np.sum(joint)),
        "joint_success_groups": len(set(groups.tolist())),
        "folds": args.folds,
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "feature_names": FEATURE_NAMES,
        "fold_summaries": fold_summaries,
        "selected_final_epochs": final_epochs,
        "final_fit": final_fit,
        "metrics": {
            "base": _metrics(rows, base_predictions),
            "vector": _metrics(rows, vector_predictions),
            "hard_fallback": _metrics(rows, hard_predictions),
            "uncertainty_fusion_nested_oof": _metrics(rows, fusion_predictions),
            "oracle": _metrics(rows, oracle_predictions),
        },
        "mean_mask_weight_joint": float(
            np.mean([value for value in mask_weights if value is not None])
        ),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "oof_predictions": str(oof_output),
        "oof_predictions_sha256": sha256_file(oof_output),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
        },
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(summary_path)


if __name__ == "__main__":
    main()
