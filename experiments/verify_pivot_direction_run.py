"""Verify a completed pivot-direction fallback training run before evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from experiments.pivot_direction_fallback import PIVOT_DIRECTION_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--expected-epochs", type=int, default=30)
    parser.add_argument("--expected-batch-size", type=int, default=48)
    parser.add_argument("--expected-seed", type=int, default=20260722)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_health(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not state:
        raise ValueError("checkpoint has an empty model state")
    non_finite: list[str] = []
    parameters = 0
    absolute_sum = 0.0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"model state item is not a tensor: {name}")
        parameters += int(tensor.numel())
        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                non_finite.append(name)
            absolute_sum += float(tensor.detach().float().abs().sum())
    if non_finite:
        raise ValueError(f"checkpoint has non-finite tensors: {non_finite[:5]}")
    if not math.isfinite(absolute_sum) or absolute_sum <= 0.0:
        raise ValueError("checkpoint weights are collapsed or non-finite")
    return {
        "state_tensors": len(state),
        "state_elements": parameters,
        "floating_absolute_sum": absolute_sum,
        "non_finite_tensors": 0,
    }


def verify_run(
    run_dir: Path,
    *,
    manifest: Path,
    expected_epochs: int,
    expected_batch_size: int,
    expected_seed: int,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    paths = {
        "summary": run_dir / "summary.json",
        "best": run_dir / "best.pt",
        "last": run_dir / "last.pt",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _read_json(paths["summary"])
    if summary.get("status") != "complete":
        raise ValueError("training summary is not complete")
    if summary.get("protocol") != PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("unexpected training protocol")
    signature = summary.get("signature") or {}
    expected_fields = {
        "protocol": PIVOT_DIRECTION_PROTOCOL,
        "epochs": expected_epochs,
        "batch_size": expected_batch_size,
        "seed": expected_seed,
        "diagnostic_limit": None,
        "image_size": 256,
        "heatmap_size": 64,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    for key, expected in expected_fields.items():
        if signature.get(key) != expected:
            raise ValueError(
                f"training signature {key}={signature.get(key)!r}; expected {expected!r}"
            )
    model_source = PROJECT_DIR / "experiments" / "pivot_direction_fallback.py"
    trainer_source = PROJECT_DIR / "experiments" / "train_pivot_direction_syncg.py"
    if signature.get("model_source_sha256") != sha256_file(model_source):
        raise ValueError("model source changed after training started")
    if signature.get("trainer_source_sha256") != sha256_file(trainer_source):
        raise ValueError("trainer source changed after training started")
    initialization = signature.get("imagenet_initialization")
    if signature.get("imagenet_pretrained") is not True or not initialization:
        raise ValueError("formal run must use the pinned torchvision initialization")
    initialization_path = Path(str(initialization))
    if (
        not initialization_path.is_file()
        or signature.get("imagenet_initialization_sha256")
        != sha256_file(initialization_path)
    ):
        raise ValueError("ImageNet initialization identity mismatch")

    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=expected_seed,
    )
    train_groups = {sample.group_id for sample in train}
    validation_groups = {sample.group_id for sample in validation}
    if train_groups & validation_groups:
        raise ValueError("training and validation groups overlap")
    identity_checks = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_sample_ids_sha256": sample_ids_hash(train),
        "validation_sample_ids_sha256": sample_ids_hash(validation),
    }
    for key, expected in identity_checks.items():
        if signature.get(key) != expected:
            raise ValueError(f"training identity mismatch: {key}")

    history = summary.get("history") or []
    if len(history) != expected_epochs:
        raise ValueError(f"history contains {len(history)} epochs")
    expected_batches = math.ceil(len(train) / expected_batch_size)
    base_lr = float(signature["learning_rate"])
    eta_min = base_lr * 0.01
    optimizer_steps = 0
    skipped_steps = 0
    best_key: tuple[float, float, int] | None = None
    for index, record in enumerate(history, start=1):
        if int(record.get("epoch", -1)) != index:
            raise ValueError("history epochs are not consecutive")
        expected_lr = eta_min + 0.5 * (base_lr - eta_min) * (
            1.0 + math.cos(math.pi * float(index - 1) / float(expected_epochs))
        )
        if not math.isclose(
            float(record.get("learning_rate", math.nan)),
            expected_lr,
            rel_tol=1e-7,
            abs_tol=1e-12,
        ):
            raise ValueError(f"epoch {index}: cosine learning rate mismatch")
        train_metrics = record.get("train") or {}
        validation_metrics = record.get("validation") or {}
        if int(train_metrics.get("samples", -1)) != len(train):
            raise ValueError(f"epoch {index}: training sample count mismatch")
        if int(validation_metrics.get("samples", -1)) != len(validation):
            raise ValueError(f"epoch {index}: validation sample count mismatch")
        optimizer_steps += int(train_metrics.get("optimizer_steps", -1))
        skipped_steps += int(train_metrics.get("skipped_optimizer_steps", -1))
        if (
            int(train_metrics.get("optimizer_steps", -1))
            + int(train_metrics.get("skipped_optimizer_steps", -1))
            != expected_batches
        ):
            raise ValueError(f"epoch {index}: optimizer accounting mismatch")
        finite_values = [
            train_metrics.get("loss"),
            train_metrics.get("pivot_loss"),
            train_metrics.get("direction_loss"),
            validation_metrics.get("loss"),
            validation_metrics.get("angle_mae_degrees"),
            validation_metrics.get("pivot_mean_error_fraction"),
        ]
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError(f"epoch {index}: non-finite metric")
        if float(validation_metrics.get("direction_coverage", 0.0)) != 1.0:
            raise ValueError(f"epoch {index}: invalid validation directions")
        key = (
            float(validation_metrics["angle_mae_degrees"]),
            float(validation_metrics["pivot_mean_error_fraction"]),
            index,
        )
        if best_key is None or key[:2] < best_key[:2]:
            best_key = key
    if optimizer_steps + skipped_steps != expected_batches * expected_epochs:
        raise ValueError("total optimizer accounting mismatch")
    assert best_key is not None
    best_epoch = int(summary.get("best_epoch", -1))
    if best_epoch != best_key[2] or not math.isclose(
        float(summary.get("best_validation_angle_mae_degrees", math.nan)),
        best_key[0],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("best validation epoch/metric is inconsistent")

    best = torch.load(paths["best"], map_location="cpu", weights_only=False)
    last = torch.load(paths["last"], map_location="cpu", weights_only=False)
    if best.get("signature") != signature or last.get("signature") != signature:
        raise ValueError("checkpoint signature mismatch")
    if best.get("protocol") != PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("best checkpoint protocol mismatch")
    if int(best.get("epoch", -1)) != best_epoch:
        raise ValueError("best checkpoint epoch mismatch")
    if int(last.get("epoch", -1)) != expected_epochs:
        raise ValueError("last checkpoint is not the final epoch")
    health = _state_health(best.get("model_state") or {})
    return {
        "protocol": "formal_pivot_direction_run_verification_v1",
        "verified": True,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": best_key[0],
        "best_validation_pivot_error_fraction": best_key[1],
        "epochs": expected_epochs,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "group_overlap": 0,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
        "best_checkpoint_sha256": sha256_file(paths["best"]),
        "last_checkpoint_sha256": sha256_file(paths["last"]),
        "summary_sha256": sha256_file(paths["summary"]),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "model_state_health": health,
    }


def main() -> None:
    args = parse_args()
    result = verify_run(
        args.run_dir,
        manifest=args.manifest.resolve(),
        expected_epochs=args.expected_epochs,
        expected_batch_size=args.expected_batch_size,
        expected_seed=args.expected_seed,
    )
    output = args.run_dir.resolve() / "verification.json"
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
