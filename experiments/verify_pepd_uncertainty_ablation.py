"""Verify one frozen PEPD uncertainty-objective training run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    FORMAL_BATCH_SIZE,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_SEEDS,
    PARENT_EPOCH,
    PEPD_UNCERTAINTY_TRAINING_PROTOCOL,
    PEPD_UNCERTAINTY_VERIFICATION_PROTOCOL,
    PROJECT_ROOT,
    TERMINAL_EPOCH,
    UNCERTAINTY_MECHANISM_ARMS,
    audit_main_convergence_cohort,
    convergence_audit,
    formal_manifest_path,
    formal_parent_pin,
    sha256_file,
    uncertainty_output_dir,
    validate_combined_history,
)
from experiments.preflight_pepd_convergence import _state_health
from experiments.probabilistic_pivot_direction import (
    build_probabilistic_pivot_direction_model,
)
from experiments.train_pepd_uncertainty_ablation_syncg import (
    _signature,
    model_state_sha256,
)
from experiments.pepd_uncertainty_objectives import (
    make_global_log_variance,
)
from experiments.vdn_baseline import (
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    set_random_seed,
    sha256_source_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=UNCERTAINTY_MECHANISM_ARMS,
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _validate_schedule(history: list[dict[str, Any]]) -> None:
    base_lr = 3.0e-4
    for epoch, row in enumerate(history, start=1):
        if epoch <= PARENT_EPOCH:
            expected = CONTINUATION_LEARNING_RATE + 0.5 * (
                base_lr - CONTINUATION_LEARNING_RATE
            ) * (
                1.0
                + math.cos(
                    math.pi * float(epoch - 1) / float(PARENT_EPOCH)
                )
            )
        else:
            expected = CONTINUATION_LEARNING_RATE
        if not math.isclose(
            float(row.get("learning_rate", math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"epoch {epoch}: uncertainty LR schedule mismatch")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    signature: Mapping[str, Any],
    expected_epoch: int,
) -> dict[str, int]:
    if checkpoint.get("protocol") != PEPD_UNCERTAINTY_TRAINING_PROTOCOL:
        raise ValueError("uncertainty checkpoint protocol mismatch")
    if checkpoint.get("signature") != signature:
        raise ValueError("uncertainty checkpoint signature mismatch")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise ValueError("uncertainty checkpoint epoch mismatch")
    history = checkpoint.get("history")
    if not isinstance(history, list) or len(history) != expected_epoch:
        raise ValueError("uncertainty checkpoint history mismatch")
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, Mapping) or not optimizer.get("param_groups"):
        raise ValueError("uncertainty checkpoint optimizer state is missing")
    param_groups = optimizer["param_groups"]
    if len(param_groups) != 2:
        raise ValueError("uncertainty optimizer must have two parameter groups")
    lr = float(param_groups[0]["lr"])
    if any(
        not math.isclose(
            float(group["lr"]),
            lr,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for group in param_groups
    ):
        raise ValueError("uncertainty optimizer group LR mismatch")
    if not math.isclose(
        float(param_groups[0]["weight_decay"]),
        1.0e-4,
        rel_tol=0.0,
        abs_tol=1e-15,
    ) or not math.isclose(
        float(param_groups[1]["weight_decay"]),
        0.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("uncertainty optimizer weight-decay groups drifted")
    if expected_epoch >= PARENT_EPOCH and not math.isclose(
        lr,
        CONTINUATION_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("uncertainty checkpoint phase-2 LR mismatch")
    if expected_epoch < PARENT_EPOCH:
        scheduler = checkpoint.get("scheduler_state")
        if not isinstance(scheduler, Mapping):
            raise ValueError("uncertainty phase-1 checkpoint has no scheduler")
        if int(scheduler.get("last_epoch", -1)) != expected_epoch:
            raise ValueError("uncertainty scheduler epoch mismatch")
    else:
        if "scheduler_state" in checkpoint:
            raise ValueError("retired uncertainty scheduler is still active")
        retired = checkpoint.get("retired_scheduler") or {}
        if int(retired.get("retired_after_epoch", -1)) != PARENT_EPOCH:
            raise ValueError("uncertainty scheduler retirement is not recorded")
    scaler = checkpoint.get("scaler_state")
    if not isinstance(scaler, Mapping) or float(scaler.get("scale", 0.0)) <= 0:
        raise ValueError("uncertainty checkpoint GradScaler state is invalid")
    for name in (
        "python_rng_state",
        "numpy_rng_state",
        "torch_rng_state",
        "cuda_rng_state_all",
        "loader_generator_state",
    ):
        if name not in checkpoint:
            raise ValueError(f"uncertainty checkpoint is missing {name}")
    global_state = checkpoint.get("global_log_variance_state")
    if (
        not isinstance(global_state, torch.Tensor)
        or global_state.numel() != 1
        or not torch.isfinite(global_state).all()
    ):
        raise ValueError(
            "uncertainty checkpoint global log-variance state is invalid"
        )
    return _state_health(checkpoint.get("model_state") or {})


def _variance_head_changed(
    initial_state: Mapping[str, torch.Tensor],
    trained_state: Mapping[str, torch.Tensor],
) -> bool:
    names = (
        "log_variance_head.weight",
        "log_variance_head.bias",
    )
    for name in names:
        if name not in initial_state or name not in trained_state:
            raise ValueError(f"uncertainty state is missing {name}")
    return any(
        not torch.equal(
            initial_state[name].detach().cpu(),
            trained_state[name].detach().cpu(),
        )
        for name in names
    )


def _global_variance_changed(
    checkpoint: Mapping[str, Any],
    *,
    initial_value: float,
) -> bool:
    value = checkpoint.get("global_log_variance_state")
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("checkpoint has no global log-variance scalar")
    return not torch.equal(
        value.detach().cpu().reshape(()),
        torch.tensor(float(initial_value), dtype=torch.float32),
    )


def build_verification(
    arm: str,
    seed: int,
    manifest: Path,
) -> dict[str, Any]:
    manifest = formal_manifest_path(manifest)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train protocol hash drifted")
    main_gate = audit_main_convergence_cohort()
    samples, _ = load_syncg_manifest(manifest, expected_split="train")
    train, validation = grouped_train_val_split(
        samples,
        validation_fraction=0.10,
        seed=seed,
    )
    pin = formal_parent_pin(seed)
    if (
        len(train) != pin.train_samples
        or len(validation) != pin.validation_samples
        or sample_ids_hash(train) != pin.train_ids_sha256
        or sample_ids_hash(validation) != pin.validation_ids_sha256
    ):
        raise ValueError("uncertainty grouped split identity drifted")
    set_random_seed(seed)
    initialized_model = build_probabilistic_pivot_direction_model(
        angle_bins=72,
        imagenet_pretrained=True,
    )
    initialized_global = make_global_log_variance(device=torch.device("cpu"))
    initial_model_state = {
        name: value.detach().cpu().clone()
        for name, value in initialized_model.state_dict().items()
    }
    initial_model_hash = model_state_sha256(initial_model_state)
    del initialized_model
    expected_signature = _signature(
        arm=arm,
        seed=seed,
        train_samples=train,
        validation_samples=validation,
        manifest=manifest,
        initial_model_state_sha256=initial_model_hash,
        main_convergence_cohort_sha256=main_gate["sha256"],
    )
    run_dir = uncertainty_output_dir(arm, seed)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    summary = _load(summary_path)
    if summary.get("protocol") != PEPD_UNCERTAINTY_TRAINING_PROTOCOL:
        raise ValueError("uncertainty summary protocol mismatch")
    if summary.get("status") != "complete":
        raise ValueError("uncertainty summary is not complete")
    if summary.get("signature") != expected_signature:
        raise ValueError("uncertainty summary signature drifted")
    if summary.get("eligible_for_model_selection") is not False:
        raise ValueError("uncertainty arm must not be model-selection eligible")
    if summary.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("uncertainty summary scope drifted")
    for path in (best_path, last_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    best_hash = sha256_file(best_path)
    last_hash = sha256_file(last_path)
    if summary.get("best_checkpoint_sha256") != best_hash:
        raise ValueError("uncertainty best checkpoint hash mismatch")
    if summary.get("last_checkpoint_sha256") != last_hash:
        raise ValueError("uncertainty last checkpoint hash mismatch")
    history = summary.get("history")
    if not isinstance(history, list):
        raise ValueError("uncertainty summary history is missing")
    validate_combined_history(
        history,
        train_samples=pin.train_samples,
        validation_samples=pin.validation_samples,
        require_calibration_metric=arm != "no_angular_nll",
    )
    _validate_schedule(history)
    for row in history:
        train_metrics = row["train"]
        validation_metrics = row["validation"]
        if train_metrics.get("uncertainty_mode") != arm:
            raise ValueError("uncertainty train metric mode mismatch")
        expected_angular = arm != "no_angular_nll"
        if (
            bool(train_metrics.get("angular_nll_in_training_objective"))
            != expected_angular
        ):
            raise ValueError("uncertainty train objective flag mismatch")
        if (
            bool(validation_metrics.get("angular_nll_in_training_objective"))
            != expected_angular
        ):
            raise ValueError("uncertainty validation objective flag mismatch")
        if (
            bool(
                train_metrics.get(
                    "per_sample_variance_head_in_training_objective"
                )
            )
            != (arm == "learned_heteroscedastic")
        ):
            raise ValueError("per-sample variance objective flag mismatch")
        if (
            bool(
                train_metrics.get(
                    "global_log_variance_in_training_objective"
                )
            )
            != (arm == "global_homoscedastic")
        ):
            raise ValueError("global variance objective flag mismatch")
        if (
            bool(validation_metrics.get("sample_ranking_available"))
            != (arm == "learned_heteroscedastic")
        ):
            raise ValueError("uncertainty ranking semantics mismatch")
    candidates = [
        (
            float(row["validation"]["angle_mae_degrees"]),
            float(row["validation"]["pivot_mean_error_fraction"]),
        )
        for row in history
    ]
    best_index = min(range(len(candidates)), key=candidates.__getitem__)
    best_epoch = best_index + 1
    if int(summary.get("best_epoch", -1)) != best_epoch:
        raise ValueError("uncertainty best selection mismatch")
    expected_best = candidates[best_index]
    actual_best = (
        float(summary["best_validation_angle_mae_degrees"]),
        float(summary["best_validation_pivot_error_fraction"]),
    )
    if actual_best != expected_best:
        raise ValueError("uncertainty best metrics mismatch")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    best_health = _validate_checkpoint(
        best,
        signature=expected_signature,
        expected_epoch=best_epoch,
    )
    last_health = _validate_checkpoint(
        last,
        signature=expected_signature,
        expected_epoch=TERMINAL_EPOCH,
    )
    best_variance_head_changed = _variance_head_changed(
        initial_model_state,
        best.get("model_state") or {},
    )
    last_variance_head_changed = _variance_head_changed(
        initial_model_state,
        last.get("model_state") or {},
    )
    if arm == "learned_heteroscedastic" and not (
        best_variance_head_changed and last_variance_head_changed
    ):
        raise ValueError("learned uncertainty head did not change from initialization")
    if arm != "learned_heteroscedastic" and (
        best_variance_head_changed or last_variance_head_changed
    ):
        raise ValueError(
            "global/no-NLL per-sample variance head changed despite being "
            "outside the objective"
        )
    best_global_changed = _global_variance_changed(
        best,
        initial_value=float(initialized_global.detach()),
    )
    last_global_changed = _global_variance_changed(
        last,
        initial_value=float(initialized_global.detach()),
    )
    if arm == "global_homoscedastic" and not (
        best_global_changed and last_global_changed
    ):
        raise ValueError(
            "learned global homoscedastic scalar did not change from initialization"
        )
    if arm != "global_homoscedastic" and (
        best_global_changed or last_global_changed
    ):
        raise ValueError(
            "inactive global log-variance scalar changed from initialization"
        )
    if last.get("history") != history:
        raise ValueError("uncertainty last checkpoint history mismatch")
    audit = convergence_audit(history, best_epoch=best_epoch)
    if summary.get("convergence_audit") != audit:
        raise ValueError("uncertainty convergence audit mismatch")
    return {
        "schema_version": 1,
        "protocol": PEPD_UNCERTAINTY_VERIFICATION_PROTOCOL,
        "verified": True,
        "converged": bool(audit["converged"]),
        "scope": "SyncG official train grouped validation only",
        "role": "secondary mechanism ablation; not algorithm selection",
        "arm": arm,
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_sha256": sha256_file(summary_path),
        "best_checkpoint_sha256": best_hash,
        "last_checkpoint_sha256": last_hash,
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": expected_best[0],
        "convergence_audit": audit,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
        "best_model_state_health": best_health,
        "last_model_state_health": last_health,
        "best_variance_head_changed_from_initialization": (
            best_variance_head_changed
        ),
        "last_variance_head_changed_from_initialization": (
            last_variance_head_changed
        ),
        "variance_head_training_expected": arm == "learned_heteroscedastic",
        "best_global_variance_changed_from_initialization": (
            best_global_changed
        ),
        "last_global_variance_changed_from_initialization": (
            last_global_changed
        ),
        "global_variance_training_expected": arm == "global_homoscedastic",
        "eligible_for_model_selection": False,
        "public_test_field_evaluation_authorized": False,
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "objective": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_uncertainty_objectives.py"
            ),
            "trainer": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "train_pepd_uncertainty_ablation_syncg.py"
            ),
            "uncertainty_metrics": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "pepd_uncertainty_metrics.py"
            ),
            "verifier": sha256_source_file(Path(__file__).resolve()),
        },
    }


def _write_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"{path} exists with different content")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    verification = build_verification(args.arm, args.seed, args.manifest)
    output = uncertainty_output_dir(args.arm, args.seed) / "verification.json"
    _write_or_validate(output, verification)
    print(
        json.dumps(
            verification,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
