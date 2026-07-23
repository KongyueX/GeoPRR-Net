"""Verify a completed probabilistic-direction training ablation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_file,
)
from experiments.verify_probabilistic_pivot_direction_run import _state_health


VERIFICATION_PROTOCOL = "formal_probabilistic_direction_ablation_verification_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument("--ablation-name", required=True)
    parser.add_argument("--expected-equivariance-weight", type=float, required=True)
    parser.add_argument("--expected-perspective-probability", type=float, required=True)
    parser.add_argument("--expected-paired-supervision-weight", type=float, default=1.0)
    parser.add_argument("--expected-epochs", type=int, default=30)
    parser.add_argument("--expected-batch-size", type=int, default=24)
    parser.add_argument("--expected-seed", type=int, default=20260722)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = args.manifest.resolve()
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    summary = _load(summary_path)
    for path in (best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if summary.get("status") != "complete":
        raise ValueError("ablation training summary is not complete")
    if summary.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("ablation has the wrong training protocol")
    signature = summary.get("signature") or {}
    exact = {
        "protocol": PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
        "epochs": args.expected_epochs,
        "batch_size": args.expected_batch_size,
        "seed": args.expected_seed,
        "diagnostic_limit": None,
        "image_size": 256,
        "heatmap_size": 64,
        "angle_bins": 72,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    }
    for name, expected in exact.items():
        if signature.get(name) != expected:
            raise ValueError(f"signature {name} mismatch")
    floating = {
        "equivariance_weight": args.expected_equivariance_weight,
        "perspective_probability": args.expected_perspective_probability,
        "paired_supervision_weight": args.expected_paired_supervision_weight,
    }
    for name, expected in floating.items():
        if not math.isclose(
            float(signature.get(name, math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"signature {name} mismatch")
    if signature.get("model_source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
    ):
        raise ValueError("model source changed after ablation training")
    if signature.get("trainer_source_sha256") != sha256_file(
        PROJECT_DIR / "experiments" / "train_probabilistic_pivot_direction_syncg.py"
    ):
        raise ValueError("trainer source changed after ablation training")
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=args.expected_seed,
    )
    train_groups = {sample.group_id for sample in train}
    validation_groups = {sample.group_id for sample in validation}
    if train_groups & validation_groups:
        raise ValueError("ablation grouped split leaks meter groups")
    expected_identity = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_sample_ids_sha256": sample_ids_hash(train),
        "validation_sample_ids_sha256": sample_ids_hash(validation),
    }
    for name, expected in expected_identity.items():
        if signature.get(name) != expected:
            raise ValueError(f"training identity mismatch: {name}")
    history = summary.get("history") or []
    if len(history) != args.expected_epochs:
        raise ValueError("ablation history length mismatch")
    expected_batches = math.ceil(len(train) / args.expected_batch_size)
    for epoch, record in enumerate(history, start=1):
        if int(record.get("epoch", -1)) != epoch:
            raise ValueError("ablation epochs are not consecutive")
        train_metrics = record.get("train") or {}
        validation_metrics = record.get("validation") or {}
        if (
            int(train_metrics.get("optimizer_steps", -1))
            + int(train_metrics.get("skipped_optimizer_steps", -1))
            != expected_batches
        ):
            raise ValueError(f"epoch {epoch}: optimizer accounting mismatch")
        required = (
            train_metrics.get("loss"),
            train_metrics.get("equivariance_loss"),
            validation_metrics.get("loss"),
            validation_metrics.get("angle_mae_degrees"),
            validation_metrics.get("angular_calibration_nll"),
        )
        if not all(math.isfinite(float(value)) for value in required):
            raise ValueError(f"epoch {epoch}: non-finite metric")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if best.get("signature") != signature or last.get("signature") != signature:
        raise ValueError("ablation checkpoint signature mismatch")
    if int(best.get("epoch", -1)) != int(summary.get("best_epoch", -1)):
        raise ValueError("ablation best checkpoint mismatch")
    if int(last.get("epoch", -1)) != args.expected_epochs:
        raise ValueError("ablation last checkpoint is not final")
    result = {
        "schema_version": 1,
        "protocol": VERIFICATION_PROTOCOL,
        "verified": True,
        "ablation_name": args.ablation_name,
        "run_dir": str(run_dir),
        "best_epoch": int(summary["best_epoch"]),
        "best_validation_angle_mae_degrees": float(
            summary["best_validation_angle_mae_degrees"]
        ),
        "epochs": args.expected_epochs,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "group_overlap": 0,
        "expected_configuration": floating,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "summary_sha256": sha256_file(summary_path),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "model_state_health": _state_health(best.get("model_state") or {}),
    }
    output = run_dir / "verification.json"
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
