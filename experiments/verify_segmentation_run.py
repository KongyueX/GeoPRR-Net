"""Verify that a completed segmentation run is safe to reuse in paper experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from experiments.datasets import (
    SYNCG_PINNED_COMMIT,
    SYNCG_SAMPLE_IDS_SHA256,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a checkpoint mapping")
    return value


def _state_dict(checkpoint: Mapping[str, Any], path: Path) -> Mapping[str, torch.Tensor]:
    value = checkpoint.get("state_dict", checkpoint)
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{path} contains no state_dict")
    if not all(isinstance(item, torch.Tensor) for item in value.values()):
        raise ValueError(f"{path} state_dict contains non-tensor values")
    return value


def _assert_same_state_dict(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    if set(expected) != set(actual):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(
            "released checkpoint parameter keys differ from initial weights: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for key in expected:
        if not torch.equal(expected[key].cpu(), actual[key].cpu()):
            raise ValueError(
                "released_calibrated.pt changed released model parameters at "
                f"{key!r}; it must package only a validation-selected threshold"
            )


def verify_segmentation_run(
    run_dir: Path,
    *,
    initial_weights: Path | None = None,
    require_formal_syncg: bool = False,
    expected_epochs: int | None = None,
    expected_batch_size: int | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    released_path = run_dir / "released_calibrated.pt"
    for path in (config_path, summary_path, best_path, released_path):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete segmentation run: missing {path}")

    config = _load_json(config_path)
    summary = _load_json(summary_path)
    signature = str(config.get("signature") or "")
    if not signature or summary.get("run_signature") != signature:
        raise ValueError("summary/run_config signature mismatch")
    if config.get("protocol") != "syncg_train_only_u2netp_finetune_v1":
        raise ValueError(f"unexpected training protocol: {config.get('protocol')!r}")

    expected_values = {
        "epochs": expected_epochs,
        "batch_size": expected_batch_size,
        "seed": expected_seed,
    }
    for key, expected in expected_values.items():
        if expected is not None and int(config.get(key, -1)) != int(expected):
            raise ValueError(
                f"completed run {key}={config.get(key)!r}, requested {expected!r}"
            )

    if require_formal_syncg:
        formal_checks = {
            "limit": None,
            "syncg_reference_commit": SYNCG_PINNED_COMMIT,
            "syncg_release_identity_verified": True,
            "syncg_sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
        }
        for key, expected in formal_checks.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"segmentation run is diagnostic/non-pinned: "
                    f"{key}={config.get(key)!r}, expected {expected!r}"
                )
        project_root = Path(__file__).resolve().parents[1]
        current_sources = {
            "training": Path(__file__).resolve().parent
            / "train_syncg_segmentation.py",
            "dataset_protocol": Path(__file__).resolve().parent / "datasets.py",
            "u2netp": project_root
            / "utils"
            / "angleDetect"
            / "pointerSeg"
            / "u2netp.py",
            "checkpoint_loader": project_root
            / "utils"
            / "angleDetect"
            / "pointerSeg"
            / "detectSeg.py",
        }
        recorded_sources = config.get("source_sha256")
        if not isinstance(recorded_sources, Mapping):
            raise ValueError("formal run_config has no source SHA-256 mapping")
        for name, path in current_sources.items():
            current_hash = _sha256(path)
            if recorded_sources.get(name) != current_hash:
                raise ValueError(
                    f"segmentation source changed since training: {name}; "
                    "do not silently reuse the stale formal run"
                )

    hashes = {
        "best.pt": _sha256(best_path),
        "released_calibrated.pt": _sha256(released_path),
    }
    if hashes["best.pt"] != summary.get("best_checkpoint_sha256"):
        raise ValueError("best.pt SHA-256 does not match summary.json")
    if (
        hashes["released_calibrated.pt"]
        != summary.get("released_calibrated_checkpoint_sha256")
    ):
        raise ValueError(
            "released_calibrated.pt SHA-256 does not match summary.json"
        )

    best_checkpoint = _torch_load(best_path)
    released_checkpoint = _torch_load(released_path)
    for name, checkpoint in (
        ("best.pt", best_checkpoint),
        ("released_calibrated.pt", released_checkpoint),
    ):
        training_protocol = checkpoint.get("training_protocol")
        if (
            not isinstance(training_protocol, Mapping)
            or training_protocol.get("signature") != signature
        ):
            raise ValueError(f"{name} training protocol signature mismatch")
        threshold = float(checkpoint.get("probability_threshold", math.nan))
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"{name} contains invalid probability threshold")

    configured_initial = config.get("initial_weights")
    if initial_weights is None:
        if not configured_initial:
            raise ValueError(
                "cannot verify released_calibrated.pt for a from-scratch run"
            )
        initial_weights = Path(str(configured_initial))
    initial_weights = initial_weights.resolve()
    if not initial_weights.is_file():
        raise FileNotFoundError(initial_weights)
    if (
        config.get("initial_weights_sha256") is not None
        and _sha256(initial_weights) != config.get("initial_weights_sha256")
    ):
        raise ValueError("initial segmentation weights SHA-256 mismatch")
    _assert_same_state_dict(
        _state_dict(_torch_load(initial_weights), initial_weights),
        _state_dict(released_checkpoint, released_path),
    )

    result = {
        "status": "verified",
        "run_dir": str(run_dir),
        "run_signature": signature,
        "formal_syncg": bool(require_formal_syncg),
        "best_epoch": int(summary["best_epoch"]),
        "best_checkpoint_sha256": hashes["best.pt"],
        "released_checkpoint_sha256": hashes["released_calibrated.pt"],
        "released_weights_unchanged": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--require-formal-syncg", action="store_true")
    parser.add_argument("--expected-epochs", type=int)
    parser.add_argument("--expected-batch-size", type=int)
    parser.add_argument("--expected-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_segmentation_run(
        args.run_dir,
        initial_weights=args.initial_weights,
        require_formal_syncg=args.require_formal_syncg,
        expected_epochs=args.expected_epochs,
        expected_batch_size=args.expected_batch_size,
        expected_seed=args.expected_seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
