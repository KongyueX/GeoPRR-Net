"""Frozen, one-shot bounded extension protocol for the single PEPD v1 miss.

The extension is eligible only when the pinned v1 cohort is intact and seed
20260721 is the sole non-converged run, with ``late_window_gain`` as its sole
failed check.  It continues the exact epoch-60 state through epochs 61..80 and
never authorizes another extension or any public/test/field evaluation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.pepd_convergence_protocol import (
    CONVERGENCE_BOUNDARY_EPOCHS,
    CONVERGENCE_LATE_GAIN_LIMIT_DEGREES,
    CONVERGENCE_RANGE_LIMIT_DEGREES,
    CONVERGENCE_SLOPE_LIMIT_DEG_PER_EPOCH,
    CONVERGENCE_WINDOW,
    CONTINUATION_LEARNING_RATE,
    FORMAL_BATCH_SIZE,
    FORMAL_MANIFEST,
    FORMAL_MANIFEST_PROTOCOL_SHA256,
    FORMAL_MANIFEST_SHA256,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    FORMAL_WORKERS,
    PEPD_COHORT_PROTOCOL,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PEPD_TRAINING_PROTOCOL,
    PROJECT_ROOT,
    REQUIRED_DIRECTION_COVERAGE,
    convergence_audit,
    formal_manifest_path,
    sha256_file,
    validate_combined_history,
)
from experiments.strict_json import strict_json_load, strict_json_source_sha256


EXTENSION_SEED = 20260721
EXTENSION_PARENT_EPOCH = 60
EXTENSION_TERMINAL_EPOCH = 80
EXTENSION_EPOCHS = EXTENSION_TERMINAL_EPOCH - EXTENSION_PARENT_EPOCH

PEPD_EXTENSION_PROTOCOL = "pepd_syncg_grouped_val_bounded_extension_v2"
PEPD_EXTENSION_VERIFICATION_PROTOCOL = (
    "pepd_syncg_grouped_val_bounded_extension_verification_v2"
)
PEPD_AUTHORITATIVE_COHORT_PROTOCOL = (
    "pepd_syncg_grouped_val_mixed_authoritative_cohort_v2"
)
PEPD_AUTHORITATIVE_HANDOFF_PROTOCOL = (
    "pepd_mixed_authoritative_oof_handoff_v2"
)
PEPD_AUTHORITATIVE_COLLECTOR_CONTRACT_PROTOCOL = (
    "pepd_mixed_authoritative_uncertainty_fusion_oof_collector_contract_v2"
)
PEPD_AUTHORITATIVE_OOF_PROTOCOL = (
    "syncg_probabilistic_uncertainty_fusion_oof_v2"
)

V1_PROTOCOL_SOURCE_SHA256 = (
    "73e228f19250c33b1dca64dc13971fe23e5222872c6f0c4f939b1da60bdd152f"
)
V1_TRAINER_SOURCE_SHA256 = (
    "8aade10411f8fbad98a6350e792411daa9dded5470b3120c952dda8c298df844"
)
V1_VERIFIER_SOURCE_SHA256 = (
    "f92316eebb0d101f38e8c3f3db1ebf07d94c4fbe6af9d08080f543c719d0b926"
)
IMPORTED_EPOCH_TRAINER_SOURCE_SHA256 = (
    "4557e313e14a5994726485365349b486cfd3fe2630d41da8d2a127b99c4065fb"
)
STRICT_JSON_SOURCE_SHA256 = (
    "86e8370c8990da2d61cecdb281403e5fb2d163689109d277fd0f2011f49cc6a5"
)
V1_COHORT_SHA256 = (
    "572e93984bed26fdbb80bb13ab8a31cd047a0aac3665cc567a5203238da8cf93"
)
V1_FAILED_CHECK = "late_window_gain"
V1_FAILED_LATE_WINDOW_GAIN_DEGREES = 0.02232416646971558


@dataclass(frozen=True)
class V1RunPin:
    seed: int
    run_dir: str
    summary_sha256: str
    verification_sha256: str
    best_sha256: str
    last_sha256: str
    best_epoch: int
    best_angle_mae_degrees: float
    converged: bool


V1_RUN_PINS: dict[int, V1RunPin] = {
    20260720: V1RunPin(
        seed=20260720,
        run_dir="artifacts/runs/pepd_convergence_phase2/seed_20260720",
        summary_sha256=(
            "c435d4cfbb7bc9fdf9d08c81d367b19a895c66e6d60521017b73fe938c34fd42"
        ),
        verification_sha256=(
            "c0b03557ffed65e1e10b99351a2e2bd938218a08f82db82e340b2cbf1951ad13"
        ),
        best_sha256=(
            "6c3560f5c29f430db33721beb753ec4f5582580792fa098c4a6f1057746a01d9"
        ),
        last_sha256=(
            "b8087915aa3df9e12d723cca8f9316ab2b582e92fac820b40c16ddcc97eab070"
        ),
        best_epoch=60,
        best_angle_mae_degrees=0.8120797983064101,
        converged=True,
    ),
    20260721: V1RunPin(
        seed=20260721,
        run_dir="artifacts/runs/pepd_convergence_phase2/seed_20260721",
        summary_sha256=(
            "eb0f12a21de2b7522ee86b08007cd3907d9f4770eeaabdae5bb39bd9d9d86d57"
        ),
        verification_sha256=(
            "98a9e8ed302a086c0bf32662163423936b79fc3b17418dee2725dea34ac5f64c"
        ),
        best_sha256=(
            "410d250b312de49cf88cd10924497380ba3ca494703e6b6d603d9ccc28ecda8a"
        ),
        last_sha256=(
            "d1060c9c28a021a7f9f2332db6f6e1d98b6071b133bdead2f1a79077cfa617fc"
        ),
        best_epoch=59,
        best_angle_mae_degrees=0.7771483635032399,
        converged=False,
    ),
    20260722: V1RunPin(
        seed=20260722,
        run_dir="artifacts/runs/pepd_convergence_phase2/seed_20260722",
        summary_sha256=(
            "39a1449132ca3bc875129a3df51abb401454f1ab9aa5e70a9b38f92d865516c4"
        ),
        verification_sha256=(
            "ae790f1b1cc8b68bfe305b127ba7f0c18889d8e0644be03c24b4230a5eaa2bdf"
        ),
        best_sha256=(
            "d952db3746528c4ebc06919f39e49ff86017887fb468dc4358256609f7fbdd21"
        ),
        last_sha256=(
            "fcccd07d4d35fa14d6447dc25430b918baa09d9c6b25990921b57ba337003f43"
        ),
        best_epoch=58,
        best_angle_mae_degrees=0.5926580212756842,
        converged=True,
    ),
}


def v1_cohort_path() -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_convergence_phase2"
        / "cohort.json"
    ).resolve()


def v1_run_dir(seed: int) -> Path:
    try:
        pin = V1_RUN_PINS[int(seed)]
    except KeyError as exc:
        raise ValueError(f"seed must be one of {FORMAL_SEEDS}") from exc
    return (PROJECT_ROOT / pin.run_dir).resolve()


def extension_output_dir() -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_convergence_extension_v2"
        / f"seed_{EXTENSION_SEED}"
    ).resolve()


def authoritative_output_dir() -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_convergence_authoritative_v2"
    ).resolve()


def authoritative_cohort_path() -> Path:
    return (authoritative_output_dir() / "cohort.json").resolve()


def authoritative_handoff_path() -> Path:
    return (authoritative_output_dir() / "oof_handoff.json").resolve()


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 drifted")


def _failed_checks(audit: Mapping[str, Any]) -> list[str]:
    checks = audit.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("v1 convergence audit has no checks")
    expected = {
        "absolute_trailing_slope",
        "trailing_range",
        "late_window_gain",
        "boundary_not_materially_better",
        "best_not_materially_boundary_censored",
    }
    if set(checks) != expected or any(type(value) is not bool for value in checks.values()):
        raise ValueError("v1 convergence check schema drifted")
    return sorted(name for name, passed in checks.items() if not passed)


def audit_v1_artifacts() -> dict[str, Any]:
    """Audit the immutable v1 cohort and the only eligible extension parent."""

    cohort_path = v1_cohort_path()
    _require_hash(cohort_path, V1_COHORT_SHA256, label="v1 cohort")
    cohort = strict_json_load(cohort_path)
    if (
        cohort.get("protocol") != PEPD_COHORT_PROTOCOL
        or cohort.get("status") != "not_converged"
        or cohort.get("all_runs_verified") is not True
        or cohort.get("all_runs_converged") is not False
        or cohort.get("seeds") != list(FORMAL_SEEDS)
        or cohort.get("public_test_field_evaluation_authorized") is not False
    ):
        raise ValueError("v1 cohort is not the pinned verified/non-converged cohort")
    rows = cohort.get("runs")
    if not isinstance(rows, list) or [
        row.get("seed") if isinstance(row, Mapping) else None for row in rows
    ] != list(FORMAL_SEEDS):
        raise ValueError("v1 cohort run membership/order drifted")
    by_seed = {int(row["seed"]): row for row in rows}
    non_converged = [
        seed for seed in FORMAL_SEEDS if by_seed[seed].get("converged") is not True
    ]
    if non_converged != [EXTENSION_SEED]:
        raise ValueError("v2 is eligible only when seed 20260721 is the sole v1 miss")

    documents: dict[int, dict[str, Any]] = {}
    for seed in FORMAL_SEEDS:
        pin = V1_RUN_PINS[seed]
        run_dir = v1_run_dir(seed)
        summary_path = run_dir / "summary.json"
        verification_path = run_dir / "verification.json"
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        _require_hash(summary_path, pin.summary_sha256, label=f"v1 seed {seed} summary")
        _require_hash(
            verification_path,
            pin.verification_sha256,
            label=f"v1 seed {seed} verification",
        )
        _require_hash(best_path, pin.best_sha256, label=f"v1 seed {seed} best")
        _require_hash(last_path, pin.last_sha256, label=f"v1 seed {seed} last")
        summary = strict_json_load(summary_path)
        verification = strict_json_load(verification_path)
        if (
            summary.get("protocol") != PEPD_CONTINUATION_PROTOCOL
            or summary.get("status") != "complete"
            or summary.get("seed") != seed
            or summary.get("best_checkpoint_sha256") != pin.best_sha256
            or summary.get("last_checkpoint_sha256") != pin.last_sha256
            or summary.get("best_epoch") != pin.best_epoch
            or not math.isclose(
                float(summary.get("best_validation_angle_mae_degrees", math.nan)),
                pin.best_angle_mae_degrees,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"v1 seed {seed} summary binding drifted")
        if (
            verification.get("protocol") != PEPD_RUN_VERIFICATION_PROTOCOL
            or verification.get("verified") is not True
            or verification.get("converged") is not pin.converged
            or verification.get("seed") != seed
            or verification.get("summary_sha256") != pin.summary_sha256
            or verification.get("best_checkpoint_sha256") != pin.best_sha256
            or verification.get("last_checkpoint_sha256") != pin.last_sha256
            or verification.get("public_or_field_evaluation_authorized") is not False
        ):
            raise ValueError(f"v1 seed {seed} verification binding drifted")
        row = by_seed[seed]
        if (
            row.get("verified") is not True
            or row.get("converged") is not pin.converged
            or row.get("summary_sha256") != pin.summary_sha256
            or row.get("verification_sha256") != pin.verification_sha256
            or row.get("best_checkpoint_sha256") != pin.best_sha256
        ):
            raise ValueError(f"v1 seed {seed} cohort binding drifted")
        stored_audit = summary.get("convergence_audit")
        history = summary.get("history")
        if not isinstance(history, list):
            raise ValueError(f"v1 seed {seed} summary history is missing")
        recomputed = convergence_audit(history, best_epoch=pin.best_epoch)
        if stored_audit != recomputed or verification.get("convergence_audit") != recomputed:
            raise ValueError(f"v1 seed {seed} convergence audit drifted")
        documents[seed] = {
            "pin": pin,
            "summary": summary,
            "verification": verification,
            "run_dir": run_dir,
        }

    extension_audit = documents[EXTENSION_SEED]["summary"]["convergence_audit"]
    if _failed_checks(extension_audit) != [V1_FAILED_CHECK]:
        raise ValueError("v2 requires late_window_gain to be the sole failed v1 check")
    if not math.isclose(
        float(extension_audit.get("late_window_gain_degrees", math.nan)),
        V1_FAILED_LATE_WINDOW_GAIN_DEGREES,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("v1 late-window gain no longer matches the frozen decision")
    return {
        "cohort": cohort,
        "cohort_path": cohort_path,
        "runs": documents,
        "eligible_seed": EXTENSION_SEED,
        "sole_failed_check": V1_FAILED_CHECK,
    }


def build_extension_signature(
    *,
    extension_trainer_source_sha256: str,
    imported_v1_trainer_source_sha256: str,
    imported_epoch_trainer_source_sha256: str,
    model_source_sha256: str,
) -> dict[str, Any]:
    pin = V1_RUN_PINS[EXTENSION_SEED]
    return {
        "protocol": PEPD_EXTENSION_PROTOCOL,
        "scope": "SyncG official train grouped validation only",
        "role": "one-shot convergence audit extension; not algorithm selection",
        "seed": EXTENSION_SEED,
        "parent_epoch": EXTENSION_PARENT_EPOCH,
        "terminal_epoch": EXTENSION_TERMINAL_EPOCH,
        "extension_epochs": EXTENSION_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "learning_rate": CONTINUATION_LEARNING_RATE,
        "learning_rate_policy": "unchanged constant from v1 phase2",
        "optimizer": "AdamW state restored exactly from pinned epoch-60 last.pt",
        "grad_scaler": "state restored exactly from pinned epoch-60 last.pt",
        "rng_policy": (
            "Python, NumPy, torch CPU/CUDA, and DataLoader generator states "
            "restored exactly from pinned epoch-60 last.pt"
        ),
        "scheduler": "none; unchanged from v1 phase2",
        "early_stopping": False,
        "checkpoint_selection": (
            "lexicographic minimum grouped-validation "
            "(angle_mae_degrees, angular_calibration_nll, "
            "pivot_mean_error_fraction) over epochs 1..80"
        ),
        "terminal_audit": (
            "exact v1 10+10 windows and thresholds, evaluated only at epoch 80"
        ),
        "further_extension_authorized": False,
        "eligibility": {
            "required_v1_status": "verified with exactly one failed check",
            "required_failed_check": V1_FAILED_CHECK,
            "v1_late_window_gain_degrees": V1_FAILED_LATE_WINDOW_GAIN_DEGREES,
        },
        "parent_run_dir": pin.run_dir,
        "parent_summary_sha256": pin.summary_sha256,
        "parent_verification_sha256": pin.verification_sha256,
        "parent_best_sha256": pin.best_sha256,
        "parent_last_sha256": pin.last_sha256,
        "v1_cohort_sha256": V1_COHORT_SHA256,
        "manifest": FORMAL_MANIFEST.as_posix(),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "train_samples": 14390,
        "validation_samples": 1610,
        "train_sample_ids_sha256": (
            "46da07391eeb6da6eb95187ece78f98764fe62a25ab8ac0575790285a55970f3"
        ),
        "validation_sample_ids_sha256": (
            "5df65a5a2c33e2393dc07f5a405d5950d83e6b1f66d7e447c38031094e4883b0"
        ),
        "v1_protocol_source_sha256": V1_PROTOCOL_SOURCE_SHA256,
        "v1_trainer_source_sha256": imported_v1_trainer_source_sha256,
        "imported_epoch_trainer_source_sha256": (
            imported_epoch_trainer_source_sha256
        ),
        "extension_trainer_source_sha256": extension_trainer_source_sha256,
        "model_source_sha256": model_source_sha256,
        "strict_json_source_sha256": strict_json_source_sha256(),
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
    }


def validate_extended_history(
    history: Sequence[Mapping[str, Any]],
    *,
    train_samples: int = 14390,
    validation_samples: int = 1610,
) -> None:
    if len(history) != EXTENSION_TERMINAL_EPOCH:
        raise ValueError(
            f"extended history must contain exactly {EXTENSION_TERMINAL_EPOCH} epochs"
        )
    validate_combined_history(
        history[:EXTENSION_PARENT_EPOCH],
        train_samples=train_samples,
        validation_samples=validation_samples,
    )
    expected_batches = math.ceil(train_samples / FORMAL_BATCH_SIZE)
    for expected_epoch, record in enumerate(
        history[EXTENSION_PARENT_EPOCH:],
        start=EXTENSION_PARENT_EPOCH + 1,
    ):
        if int(record.get("epoch", -1)) != expected_epoch:
            raise ValueError("extension history epochs are not consecutive")
        train = record.get("train")
        validation = record.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise ValueError(f"epoch {expected_epoch}: metrics are missing")
        if int(train.get("samples", -1)) != train_samples:
            raise ValueError(f"epoch {expected_epoch}: train sample count mismatch")
        if int(validation.get("samples", -1)) != validation_samples:
            raise ValueError(
                f"epoch {expected_epoch}: validation sample count mismatch"
            )
        steps = int(train.get("optimizer_steps", -1))
        skipped = int(train.get("skipped_optimizer_steps", -1))
        if steps < 0 or skipped < 0 or steps + skipped != expected_batches:
            raise ValueError(
                f"epoch {expected_epoch}: optimizer-step accounting mismatch"
            )
        for section, names in (
            (train, ("loss",)),
            (
                validation,
                (
                    "loss",
                    "angle_mae_degrees",
                    "angular_calibration_nll",
                    "pivot_mean_error_fraction",
                    "direction_coverage",
                ),
            ),
        ):
            for name in names:
                if not math.isfinite(float(section.get(name, math.nan))):
                    raise ValueError(f"epoch {expected_epoch}: non-finite {name}")
        if not math.isclose(
            float(validation["direction_coverage"]),
            REQUIRED_DIRECTION_COVERAGE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"epoch {expected_epoch}: direction coverage is not exactly 1"
            )
        if not math.isclose(
            float(record.get("learning_rate", math.nan)),
            CONTINUATION_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"epoch {expected_epoch}: learning rate drifted")


def extension_convergence_audit(
    history: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int,
) -> dict[str, Any]:
    """Apply the unchanged v1 convergence test exactly once at epoch 80."""

    if len(history) != EXTENSION_TERMINAL_EPOCH:
        raise ValueError("extension audit requires the complete 80-epoch history")
    errors = np.asarray(
        [
            float((record.get("validation") or {}).get("angle_mae_degrees"))
            for record in history
        ],
        dtype=np.float64,
    )
    if (
        errors.shape != (EXTENSION_TERMINAL_EPOCH,)
        or not np.all(np.isfinite(errors))
    ):
        raise ValueError("extension audit received invalid validation MAE history")
    trailing = errors[-CONVERGENCE_WINDOW:]
    previous = errors[
        -(2 * CONVERGENCE_WINDOW) : -CONVERGENCE_WINDOW
    ]
    boundary = errors[-CONVERGENCE_BOUNDARY_EPOCHS:]
    preceding = errors[
        -CONVERGENCE_WINDOW : -CONVERGENCE_BOUNDARY_EPOCHS
    ]
    x = np.arange(CONVERGENCE_WINDOW, dtype=np.float64)
    slope = float(np.polyfit(x, trailing, deg=1)[0])
    spread = float(np.ptp(trailing))
    late_gain = float(np.mean(previous) - np.mean(trailing))
    boundary_gain = float(np.min(preceding) - np.min(boundary))
    checks = {
        "absolute_trailing_slope": (
            abs(slope) <= CONVERGENCE_SLOPE_LIMIT_DEG_PER_EPOCH
        ),
        "trailing_range": spread <= CONVERGENCE_RANGE_LIMIT_DEGREES,
        "late_window_gain": late_gain <= CONVERGENCE_LATE_GAIN_LIMIT_DEGREES,
        "boundary_not_materially_better": (
            boundary_gain <= CONVERGENCE_LATE_GAIN_LIMIT_DEGREES
        ),
        "best_not_materially_boundary_censored": not (
            int(best_epoch)
            > EXTENSION_TERMINAL_EPOCH - CONVERGENCE_BOUNDARY_EPOCHS
            and boundary_gain > CONVERGENCE_LATE_GAIN_LIMIT_DEGREES
        ),
    }
    return {
        "protocol": PEPD_EXTENSION_PROTOCOL,
        "window_epochs": CONVERGENCE_WINDOW,
        "trailing_epochs": [
            EXTENSION_TERMINAL_EPOCH - CONVERGENCE_WINDOW + 1,
            EXTENSION_TERMINAL_EPOCH,
        ],
        "trailing_slope_degrees_per_epoch": slope,
        "trailing_range_degrees": spread,
        "previous_window_mean_degrees": float(np.mean(previous)),
        "trailing_window_mean_degrees": float(np.mean(trailing)),
        "late_window_gain_degrees": late_gain,
        "boundary_gain_degrees": boundary_gain,
        "best_epoch": int(best_epoch),
        "thresholds": {
            "absolute_slope_max_degrees_per_epoch": (
                CONVERGENCE_SLOPE_LIMIT_DEG_PER_EPOCH
            ),
            "trailing_range_max_degrees": CONVERGENCE_RANGE_LIMIT_DEGREES,
            "late_gain_max_degrees": CONVERGENCE_LATE_GAIN_LIMIT_DEGREES,
            "boundary_epochs": CONVERGENCE_BOUNDARY_EPOCHS,
        },
        "checks": checks,
        "converged": all(checks.values()),
        "failure_action": (
            "stop permanently under v2; no further epoch extension and no "
            "public/test/field/sealed/confirmatory access"
        ),
    }


def validate_formal_manifest(path: Path) -> Path:
    manifest = formal_manifest_path(path)
    if sha256_file(manifest) != FORMAL_MANIFEST_SHA256:
        raise ValueError("SyncG train manifest hash drifted")
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if sha256_file(protocol_path) != FORMAL_MANIFEST_PROTOCOL_SHA256:
        raise ValueError("SyncG train manifest protocol hash drifted")
    return manifest
