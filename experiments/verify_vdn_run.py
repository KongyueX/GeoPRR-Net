"""Verify a completed formal VDN retraining run before paper evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PINNED_COMMIT,
    VDN_PROTOCOL,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_file,
    verify_vdn_source,
)


OFFICIAL_RESNET18_SHA256 = (
    "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=Path("artifacts/vendor/VectorDetectionNetwork"),
    )
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument("--expected-batch-size", type=int, default=8)
    parser.add_argument("--expected-seed", type=int, default=20260720)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
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
        raise ValueError("VDN checkpoint has an empty model state")
    non_finite = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"model state {name} is not a tensor")
        if not torch.isfinite(tensor).all():
            non_finite.append(name)
    if non_finite:
        raise ValueError(f"VDN checkpoint contains non-finite tensors: {non_finite}")
    required = (
        "conv1.weight",
        "layer1.0.conv1.weight",
        "deconv_layers.0.weight",
        "final_layer_hm.weight",
        "final_layer_v.weight",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"VDN checkpoint misses architecture tensors: {missing}")
    statistics = {
        name: {
            "mean": float(state[name].float().mean()),
            "std": float(state[name].float().std()),
            "abs_max": float(state[name].float().abs().max()),
        }
        for name in required
    }
    if statistics["conv1.weight"]["std"] < 0.01:
        raise ValueError("VDN backbone conv1 collapsed; reject this training run")
    if statistics["layer1.0.conv1.weight"]["std"] < 0.005:
        raise ValueError("VDN backbone layer1 collapsed; reject this training run")
    return {
        "tensor_count": len(state),
        "non_finite_tensors": 0,
        "statistics": statistics,
    }


def verify_run(
    run_dir: Path,
    *,
    manifest: Path,
    vdn_source: Path,
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
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"VDN run is incomplete: {missing}")
    summary = _load_json(paths["summary"])
    if summary.get("status") != "complete":
        raise ValueError("VDN summary is not marked complete")
    signature = summary.get("signature") or {}
    expected_signature = {
        "protocol": VDN_PROTOCOL,
        "vdn_source_commit": VDN_PINNED_COMMIT,
        "epochs": expected_epochs,
        "batch_size": expected_batch_size,
        "seed": expected_seed,
        "diagnostic_limit": None,
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "image_size": 384,
        "validation_fraction": 0.10,
        "scale_factor": 0.02,
        "rotation_factor": 90.0,
        "imagenet_pretrained": True,
        "imagenet_initialization_sha256": OFFICIAL_RESNET18_SHA256,
    }
    mismatches = {
        key: {"actual": signature.get(key), "expected": expected}
        for key, expected in expected_signature.items()
        if signature.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"VDN formal signature mismatch: {mismatches}")
    if verify_vdn_source(vdn_source) != VDN_PINNED_COMMIT:
        raise ValueError("VDN source commit verification failed")
    source_expectations = {
        "adapter_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "trainer_source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "train_vdn_syncg.py"
        ),
        "vdn_model_source_sha256": sha256_file(
            vdn_source / "libs" / "models" / "vdn_model.py"
        ),
    }
    changed = {
        key: {"signed": signature.get(key), "current": value}
        for key, value in source_expectations.items()
        if signature.get(key) != value
    }
    if changed:
        raise ValueError(f"VDN signed training sources changed: {changed}")

    manifest_protocol = summary.get("manifest_protocol") or {}
    manifest_expectations = {
        "dataset": "SyncG",
        "split": "train",
        "protocol": "syncg_official_split_v1",
        "expected_rows": 16000,
        "emitted_rows": 16000,
        "release_identity_verified": True,
        "strict_release": True,
        "reference_huggingface_commit": (
            "14204c3f5b35d160fafa39ad195cd5a63e6e9c12"
        ),
        "sample_ids_sha256": (
            "6c1bcfd7a6a83c07e6a8d6c133f0fc48abed543ca02d02c6ee46804240216e75"
        ),
    }
    manifest_mismatches = {
        key: {"actual": manifest_protocol.get(key), "expected": expected}
        for key, expected in manifest_expectations.items()
        if manifest_protocol.get(key) != expected
    }
    if manifest_mismatches:
        raise ValueError(f"VDN training manifest mismatch: {manifest_mismatches}")
    if int(signature["train_samples"]) + int(signature["validation_samples"]) != 16000:
        raise ValueError("VDN train/validation split does not cover all SyncG train rows")
    manifest = manifest.resolve()
    if (
        not manifest.is_file()
        or sha256_file(manifest) != signature.get("manifest_sha256")
    ):
        raise ValueError("VDN training manifest file does not match the signed manifest")
    manifest_protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if (
        not manifest_protocol_path.is_file()
        or sha256_file(manifest_protocol_path)
        != signature.get("manifest_protocol_sha256")
    ):
        raise ValueError("VDN training manifest protocol does not match its signature")
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train_samples, validation_samples = grouped_train_val_split(
        samples,
        validation_fraction=float(signature["validation_fraction"]),
        seed=int(signature["seed"]),
    )
    split_expectations = {
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
    }
    split_mismatches = {
        key: {"actual": signature.get(key), "expected": expected}
        for key, expected in split_expectations.items()
        if signature.get(key) != expected
    }
    if split_mismatches:
        raise ValueError(f"VDN grouped split mismatch: {split_mismatches}")
    train_groups = {sample.group_id for sample in train_samples}
    validation_groups = {sample.group_id for sample in validation_samples}
    if train_groups & validation_groups:
        raise ValueError("VDN train/validation groups overlap")

    history = summary.get("history") or []
    if len(history) != expected_epochs:
        raise ValueError(f"VDN history has {len(history)} epochs, expected {expected_epochs}")
    if [int(item.get("epoch", -1)) for item in history] != list(
        range(1, expected_epochs + 1)
    ):
        raise ValueError("VDN history epoch sequence is incomplete")
    milestones = sorted(
        {
            max(1, int(round(expected_epochs * 0.70))),
            max(1, int(round(expected_epochs * 0.95))),
        }
    )
    for item in history:
        epoch = int(item["epoch"])
        passed_milestones = sum(epoch > milestone for milestone in milestones)
        expected_lr = 1e-3 * (0.1**passed_milestones)
        if not math.isclose(
            float(item["learning_rate"]),
            expected_lr,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"VDN epoch {epoch} has an unexpected learning rate")
        expected_vector_weight = (
            1.0
            if expected_epochs == 1
            else float(epoch - 1) / float(expected_epochs - 1)
        )
        if not math.isclose(
            float(item["train"]["vector_weight"]),
            expected_vector_weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"VDN epoch {epoch} has an unexpected vector-loss weight")
        if int(item["train"]["samples"]) != int(signature["train_samples"]):
            raise ValueError(f"VDN epoch {epoch} did not consume the full train split")
        if int(item["validation"]["samples"]) != int(signature["validation_samples"]):
            raise ValueError(f"VDN epoch {epoch} did not consume the full validation split")
        finite_metrics = (
            float(item["train"]["loss"]),
            float(item["train"]["heatmap_loss"]),
            float(item["train"]["vector_loss"]),
            float(item["validation"]["loss"]),
            float(item["validation"]["heatmap_loss"]),
            float(item["validation"]["vector_loss"]),
            float(item["validation"]["angle_mae_degrees"]),
        )
        if not np.isfinite(finite_metrics).all():
            raise ValueError(f"VDN epoch {epoch} contains non-finite metrics")
    expected_steps = math.ceil(
        int(signature["train_samples"]) / int(signature["batch_size"])
    )
    optimizer_steps = [int(item["train"]["optimizer_steps"]) for item in history]
    skipped_steps = [
        int(item["train"]["skipped_optimizer_steps"])
        for item in history
    ]
    if any(done + skipped != expected_steps for done, skipped in zip(optimizer_steps, skipped_steps)):
        raise ValueError("VDN epoch optimizer-step accounting is inconsistent")
    if sum(optimizer_steps) <= 0:
        raise ValueError("VDN run contains no successful optimizer steps")
    finite_angles = [
        (int(item["epoch"]), float(item["validation"]["angle_mae_degrees"]))
        for item in history
        if np.isfinite(float(item["validation"]["angle_mae_degrees"]))
    ]
    if not finite_angles:
        raise ValueError("VDN validation never produced a valid direction")
    best_epoch, best_angle = min(finite_angles, key=lambda item: item[1])
    if best_epoch != int(summary["best_epoch"]) or not math.isclose(
        best_angle,
        float(summary["best_validation_angle_mae_degrees"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("VDN best validation epoch/metric is inconsistent")

    best = torch.load(paths["best"], map_location="cpu", weights_only=False)
    last = torch.load(paths["last"], map_location="cpu", weights_only=False)
    if best.get("signature") != signature or last.get("signature") != signature:
        raise ValueError("VDN checkpoints do not match the summary signature")
    if int(best.get("epoch", -1)) != best_epoch:
        raise ValueError("VDN best checkpoint epoch is inconsistent")
    if int(last.get("epoch", -1)) != expected_epochs:
        raise ValueError("VDN last checkpoint is not the final epoch")
    health = _state_health(best.get("model_state") or {})
    return {
        "protocol": "formal_vdn_run_verification_v1",
        "verified": True,
        "run_dir": str(run_dir),
        "vdn_source_commit": VDN_PINNED_COMMIT,
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": best_angle,
        "epochs": expected_epochs,
        "train_samples": int(signature["train_samples"]),
        "validation_samples": int(signature["validation_samples"]),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "group_overlap": 0,
        "optimizer_steps": sum(optimizer_steps),
        "skipped_optimizer_steps": sum(skipped_steps),
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
        vdn_source=args.vdn_source.resolve(),
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
