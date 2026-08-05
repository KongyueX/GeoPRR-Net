"""Frozen train/grouped-validation-only protocol for PEPD convergence.

This module intentionally contains no dataset iteration or model execution.  It
is the single source of truth shared by the preflight, continuation trainer,
run verifier, grouped-validation evaluator, and cohort summarizer.

The protocol is a convergence audit of the already retained PEPD method.  It is
not an algorithm-search budget and it never authorizes SyncG test or any public,
RPM, Pointer, field, sealed, or confirmatory data access.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PEPD_TRAINING_PROTOCOL = (
    "syncg_probabilistic_perspective_equivariant_direction_v1"
)
PEPD_CONTINUATION_PROTOCOL = "pepd_syncg_grouped_val_continuation_v1"
PEPD_PREFLIGHT_PROTOCOL = "pepd_syncg_grouped_val_preflight_v1"
PEPD_RUN_VERIFICATION_PROTOCOL = "pepd_syncg_grouped_val_run_verification_v1"
PEPD_COHORT_PROTOCOL = "pepd_syncg_grouped_val_convergence_cohort_v1"
PEPD_GROUPED_VAL_EVALUATION_PROTOCOL = (
    "pepd_syncg_grouped_val_robustness_decoder_views_v2"
)
PEPD_DECODER_VIEW_PROTOCOL = "pepd_frozen_same_logits_decoder_views_v1"
PEPD_MECHANISM_PROTOCOL = "pepd_syncg_grouped_val_mechanism_ablation_plan_v1"
PEPD_MECHANISM_RUN_PROTOCOL = (
    "pepd_syncg_grouped_val_mechanism_continuation_v1"
)
PEPD_MECHANISM_VERIFICATION_PROTOCOL = (
    "pepd_syncg_grouped_val_mechanism_run_verification_v1"
)
PEPD_MECHANISM_COHORT_PROTOCOL = (
    "pepd_syncg_grouped_val_mechanism_cohort_v1"
)
PEPD_UNCERTAINTY_TRAINING_PROTOCOL = (
    "pepd_syncg_grouped_val_uncertainty_ablation_v1"
)
PEPD_UNCERTAINTY_VERIFICATION_PROTOCOL = (
    "pepd_syncg_grouped_val_uncertainty_ablation_verification_v1"
)
PEPD_UNCERTAINTY_COHORT_PROTOCOL = (
    "pepd_syncg_grouped_val_uncertainty_ablation_cohort_v1"
)

FORMAL_SEEDS = (20260720, 20260721, 20260722)
PARENT_EPOCH = 30
TERMINAL_EPOCH = 60
CONTINUATION_EPOCHS = TERMINAL_EPOCH - PARENT_EPOCH
CONTINUATION_LEARNING_RATE = 3.0e-6
FORMAL_BATCH_SIZE = 24
FORMAL_WORKERS = 0
FORMAL_MANIFEST = Path("artifacts/manifests/syncg_train.jsonl")
FORMAL_MANIFEST_PROTOCOL = Path(
    "artifacts/manifests/syncg_train.jsonl.protocol.json"
)
FORMAL_MANIFEST_SHA256 = (
    "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
)
FORMAL_MANIFEST_PROTOCOL_SHA256 = (
    "315e17ac8aba46d00f84a0060dba145d7e036fa096bd26423dbd34a170600c59"
)
FORMAL_MODEL_SOURCE_SHA256 = (
    "8fc58de6794d593a02e4cd7456d58001b1e999309a3a69d835881a81a922472e"
)
IMAGENET_RESNET18_INITIALIZATION_SHA256 = (
    "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
LEGACY_PARENT_TRAINER_SOURCE_SHA256 = (
    "a992be129084e3ab19a230d95f333b91e37faad10882b50e1c4b7682c63cf284"
)

# The phase boundary is deliberately explicit.  The legacy checkpoints did not
# preserve Python/NumPy/DataLoader RNG states, so the continuation cannot be
# represented honestly as a bitwise replay of an uninterrupted 60-epoch run.
# It is a deterministic, pre-declared second phase initialized from the exact
# epoch-30 model, AdamW moments, and GradScaler state.
CONTINUATION_SEED_OFFSET = 1_000_003
DETERMINISM_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",  # runner replaces this with the formal base seed
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

GROUPED_VAL_CONDITIONS = (
    "clean",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
UNCERTAINTY_GROUPED_VAL_CONDITIONS = (
    "clean",
    "perspective_moderate",
    "perspective_severe",
)
DECODER_VIEWS = ("direct", "circular", "fused")
GROUPED_VAL_DEGRADATION_SEED = 20260724
GROUPED_VAL_BOOTSTRAP_ITERATIONS = 5000
GROUPED_VAL_BOOTSTRAP_SEED = 20260725

CONVERGENCE_WINDOW = 10
CONVERGENCE_SLOPE_LIMIT_DEG_PER_EPOCH = 0.005
CONVERGENCE_RANGE_LIMIT_DEGREES = 0.10
CONVERGENCE_LATE_GAIN_LIMIT_DEGREES = 0.02
CONVERGENCE_BOUNDARY_EPOCHS = 3
REQUIRED_DIRECTION_COVERAGE = 1.0


@dataclass(frozen=True)
class ParentPin:
    seed: int
    run_dir: str
    summary_sha256: str
    best_sha256: str
    last_sha256: str
    train_samples: int
    validation_samples: int
    train_ids_sha256: str
    validation_ids_sha256: str
    best_epoch: int
    best_angle_mae_degrees: float


PARENT_PINS: dict[int, ParentPin] = {
    20260720: ParentPin(
        seed=20260720,
        run_dir=(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260720"
        ),
        summary_sha256=(
            "8b3140a5d19b2451ce6bc91370b682da6b2a3d4f16d5f5b03af812eac1e79f7f"
        ),
        best_sha256=(
            "e4cc60e11c79153bd3982af5e1799024ffa1db0396cec32052ebcf540f8ea0ae"
        ),
        last_sha256=(
            "393087fa5cc2c0e02d96079e4bc34654a9628acf829900538e2fa13566d03fc3"
        ),
        train_samples=14375,
        validation_samples=1625,
        train_ids_sha256=(
            "b38e477fdf8667bc012023731756aa4fa275fd6e931a87bc065fec9d8ef88979"
        ),
        validation_ids_sha256=(
            "7550cf807f6669723c8a58cf80c7e1046af4aaea8bacfebc781dcb70d899fca2"
        ),
        best_epoch=29,
        best_angle_mae_degrees=0.8518195296766666,
    ),
    20260721: ParentPin(
        seed=20260721,
        run_dir=(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260721"
        ),
        summary_sha256=(
            "0c7670ff2c2c654972bad6c4901651aa10b796279fcc6704d7ac9680228de9cc"
        ),
        best_sha256=(
            "44ddbc7ad2d3eb8f8226738faac70e8df93aa09caad8fb9e99c2bb62a41e10a0"
        ),
        last_sha256=(
            "dc47902ca8248a21eca12c887d0c297414bdfea7cebacb342d21e9d13c00684b"
        ),
        train_samples=14390,
        validation_samples=1610,
        train_ids_sha256=(
            "46da07391eeb6da6eb95187ece78f98764fe62a25ab8ac0575790285a55970f3"
        ),
        validation_ids_sha256=(
            "5df65a5a2c33e2393dc07f5a405d5950d83e6b1f66d7e447c38031094e4883b0"
        ),
        best_epoch=29,
        best_angle_mae_degrees=0.853290960601074,
    ),
    20260722: ParentPin(
        seed=20260722,
        run_dir=(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260722"
        ),
        summary_sha256=(
            "ca67459fb5f749e085727c499b53f7a7377fedaf686602fb6e08b658e226709e"
        ),
        best_sha256=(
            "b585a5f6092af4c74fc65f6b8e999744c3d3bae5a665ea379670b066d73fe11f"
        ),
        last_sha256=(
            "c0516a8c67af2f0a59d02d27a1f426398b87050d7d50565bbb0befa7b63da3b9"
        ),
        train_samples=14398,
        validation_samples=1602,
        train_ids_sha256=(
            "160d7852ef4a8d52486450b3c038484b50d6ac125533919925142453bb0c868b"
        ),
        validation_ids_sha256=(
            "1fa614f7dbf4507e43a0cb6444379a504660e4c7346db9c35712d5bf7628da3c"
        ),
        best_epoch=29,
        best_angle_mae_degrees=0.6309960483737318,
    ),
}


@dataclass(frozen=True)
class MechanismArm:
    name: str
    perspective_probability: float
    paired_supervision_weight: float
    equivariance_weight: float
    uncertainty_objective: str
    formal_priority: str
    interpretation: str


# These are evidence arms, never candidates for choosing a replacement model.
# ``paired_supervision_only`` is the precise name of the legacy
# ``no_equivariance_loss`` arm; "no-equiv" is therefore an alias, not a fourth
# geometry arm.
MECHANISM_ARMS: dict[str, MechanismArm] = {
    "full": MechanismArm(
        name="full",
        perspective_probability=0.80,
        paired_supervision_weight=1.0,
        equivariance_weight=0.50,
        uncertainty_objective="learned_heteroscedastic_angular_nll",
        formal_priority="primary",
        interpretation=(
            "paired labels plus explicit projective-equivariance consistency "
            "regularization; not an exactly equivariant architecture"
        ),
    ),
    "paired_supervision_only": MechanismArm(
        name="paired_supervision_only",
        perspective_probability=0.80,
        paired_supervision_weight=1.0,
        equivariance_weight=0.0,
        uncertainty_objective="learned_heteroscedastic_angular_nll",
        formal_priority="primary",
        interpretation=(
            "projectively transformed paired labels without consistency loss "
            "(legacy alias: no_equivariance_loss / no-equiv)"
        ),
    ),
    "no_projective_pair": MechanismArm(
        name="no_projective_pair",
        perspective_probability=0.0,
        paired_supervision_weight=1.0,
        equivariance_weight=0.0,
        uncertainty_objective="learned_heteroscedastic_angular_nll",
        formal_priority="primary",
        interpretation=(
            "same-geometry photometric partner; no projective pair and no "
            "equivariance term"
        ),
    ),
    "global_homoscedastic": MechanismArm(
        name="global_homoscedastic",
        perspective_probability=0.80,
        paired_supervision_weight=1.0,
        equivariance_weight=0.50,
        uncertainty_objective="learned_global_homoscedastic_angular_nll",
        formal_priority="secondary",
        interpretation=(
            "pre-declared learned global log-variance baseline; separate "
            "same-start training arm, never a validation-selected constant"
        ),
    ),
    "no_angular_nll": MechanismArm(
        name="no_angular_nll",
        perspective_probability=0.80,
        paired_supervision_weight=1.0,
        equivariance_weight=0.50,
        uncertainty_objective="bin_plus_cosine_without_angular_nll",
        formal_priority="secondary",
        interpretation=(
            "pre-declared uncertainty-objective ablation; separate training arm"
        ),
    ),
}

PRIMARY_MECHANISM_ARMS = (
    "full",
    "paired_supervision_only",
    "no_projective_pair",
)
UNCERTAINTY_MECHANISM_ARMS = (
    "learned_heteroscedastic",
    "global_homoscedastic",
    "no_angular_nll",
)
GLOBAL_LOG_VARIANCE_INITIAL_VALUE = 0.0
LOG_VARIANCE_CLAMP_MIN = -9.0
LOG_VARIANCE_CLAMP_MAX = 2.0

# Existing seed-20260722 mechanism parents were produced before this
# convergence protocol.  They are pinned here; missing seed-20260720/21
# parents must be generated at the exact pre-declared paths and settings before
# any mechanism continuation can start.
LEGACY_MECHANISM_PARENT_PINS: dict[str, dict[str, str]] = {
    "paired_supervision_only": {
        "run_dir": (
            "artifacts/runs/probabilistic_direction_ablations/"
            "no_equivariance_loss/seed_20260722"
        ),
        "summary_sha256": (
            "a591194138fbba878f4f57fcdfe2194d58d2ffdf0d53a6a9a7a20e62d688c2c3"
        ),
        "best_sha256": (
            "9209ff93865f521c5a7cb273913c1a165ddcc632de7c8947507431a3e5c398b0"
        ),
        "last_sha256": (
            "0cd90a47519e65598f645448a57002c8768434c08a13d4e05e3e8f4ab4e33f3b"
        ),
    },
    "no_projective_pair": {
        "run_dir": (
            "artifacts/runs/probabilistic_direction_ablations/"
            "no_projective_pair/seed_20260722"
        ),
        "summary_sha256": (
            "9a3bea73057daa1317afd760aee9e1cedd9def6bbe303012842fba10867fcf62"
        ),
        "best_sha256": (
            "81593b2738f99f7558e02788f73676b96613fa80d8de7c3c139349cefebe7ec1"
        ),
        "last_sha256": (
            "c6d0fb94376d35e0699d23052ef242883b1ef32a571e900a504bdb7305541ed9"
        ),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def formal_manifest_path(path: Path) -> Path:
    resolved = path.resolve()
    expected = (PROJECT_ROOT / FORMAL_MANIFEST).resolve()
    if resolved != expected:
        raise ValueError(
            "PEPD convergence is restricted to the pinned SyncG train manifest; "
            f"got {resolved}"
        )
    return resolved


def formal_parent_pin(seed: int) -> ParentPin:
    try:
        return PARENT_PINS[int(seed)]
    except KeyError as exc:
        raise ValueError(f"seed must be one of {FORMAL_SEEDS}") from exc


def formal_parent_dir(seed: int) -> Path:
    return (PROJECT_ROOT / formal_parent_pin(seed).run_dir).resolve()


def formal_output_dir(seed: int) -> Path:
    formal_parent_pin(seed)
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_convergence_phase2"
        / f"seed_{int(seed)}"
    ).resolve()


def main_convergence_cohort_path() -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_convergence_phase2"
        / "cohort.json"
    ).resolve()


def audit_main_convergence_cohort(
    path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless the exact three-seed main cohort has converged."""

    resolved = (
        main_convergence_cohort_path()
        if path is None
        else Path(path).resolve()
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"converged main PEPD cohort is required: {resolved}"
        )
    value = strict_json_load(resolved)
    if value.get("protocol") != PEPD_COHORT_PROTOCOL:
        raise ValueError("main PEPD cohort protocol mismatch")
    if value.get("status") != "converged":
        raise ValueError("main PEPD cohort status is not converged")
    if (
        value.get("all_runs_verified") is not True
        or value.get("all_runs_converged") is not True
    ):
        raise ValueError("main PEPD cohort is not verified and converged")
    if tuple(value.get("seeds") or ()) != FORMAL_SEEDS:
        raise ValueError("main PEPD cohort seed set/order drifted")
    rows = value.get("runs")
    if not isinstance(rows, list) or len(rows) != len(FORMAL_SEEDS):
        raise ValueError("main PEPD cohort run membership drifted")
    by_seed = {
        int(row.get("seed", -1)): row
        for row in rows
        if isinstance(row, Mapping)
    }
    if set(by_seed) != set(FORMAL_SEEDS):
        raise ValueError("main PEPD cohort run seed membership drifted")
    for seed in FORMAL_SEEDS:
        row = by_seed[seed]
        if row.get("verified") is not True or row.get("converged") is not True:
            raise ValueError(f"main PEPD seed {seed} is not verified/converged")
        for name in (
            "best_checkpoint_sha256",
            "summary_sha256",
            "verification_sha256",
        ):
            digest = row.get(name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"main PEPD seed {seed} has invalid {name}"
                )
    if (
        value.get("grouped_validation_controlled_perspective_authorized")
        is not True
    ):
        raise ValueError("main PEPD grouped-validation gate is not authorized")
    if (
        value.get("grouped_validation_controlled_robustness_authorized")
        is not True
    ):
        raise ValueError(
            "main PEPD four-condition robustness gate is not authorized"
        )
    if value.get("public_test_field_evaluation_authorized") is not False:
        raise ValueError("main PEPD cohort scope drifted")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "protocol": PEPD_COHORT_PROTOCOL,
        "seeds": list(FORMAL_SEEDS),
        "verified": True,
        "converged": True,
        "public_test_field_evaluation_authorized": False,
    }


def primary_mechanism_arm(name: str) -> MechanismArm:
    if name not in PRIMARY_MECHANISM_ARMS:
        raise ValueError(
            f"formal primary mechanism arm must be one of {PRIMARY_MECHANISM_ARMS}"
        )
    return MECHANISM_ARMS[name]


def mechanism_parent_dir(name: str, seed: int) -> Path:
    primary_mechanism_arm(name)
    formal_parent_pin(seed)
    if name == "full":
        return formal_parent_dir(seed)
    if int(seed) == 20260722:
        return (
            PROJECT_ROOT / LEGACY_MECHANISM_PARENT_PINS[name]["run_dir"]
        ).resolve()
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_mechanism_phase1"
        / name
        / f"seed_{int(seed)}"
    ).resolve()


def mechanism_output_dir(name: str, seed: int) -> Path:
    primary_mechanism_arm(name)
    formal_parent_pin(seed)
    if name == "full":
        return formal_output_dir(seed)
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_mechanism_phase2"
        / name
        / f"seed_{int(seed)}"
    ).resolve()


def uncertainty_mechanism_arm(name: str) -> str:
    if name not in UNCERTAINTY_MECHANISM_ARMS:
        raise ValueError(
            "uncertainty arm must be one of "
            f"{UNCERTAINTY_MECHANISM_ARMS}"
        )
    return name


def uncertainty_output_dir(name: str, seed: int) -> Path:
    uncertainty_mechanism_arm(name)
    formal_parent_pin(seed)
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "pepd_uncertainty_ablation_v1"
        / name
        / f"seed_{int(seed)}"
    ).resolve()


def continuation_seed(seed: int) -> int:
    formal_parent_pin(seed)
    return int(seed) + CONTINUATION_SEED_OFFSET


def expected_full_configuration() -> dict[str, Any]:
    return {
        "protocol": PEPD_TRAINING_PROTOCOL,
        "epochs": PARENT_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "diagnostic_limit": None,
        "image_size": 256,
        "heatmap_size": 64,
        "angle_bins": 72,
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-4,
        "pivot_loss_weight": 1.0,
        "bin_loss_weight": 0.20,
        "vector_loss_weight": 0.50,
        "paired_supervision_weight": 1.0,
        "equivariance_weight": 0.50,
        "equivariance_pivot_weight": 1.0,
        "soft_target_sigma_bins": 1.25,
        "validation_fraction": 0.10,
        "expansion": 1.25,
        "scale_factor": 0.10,
        "rotation_factor": 90.0,
        "translation_factor": 0.12,
        "heatmap_sigma": 1.5,
        "perspective_probability": 0.80,
        "max_perspective_degrees": 45.0,
        "max_blur_sigma": 3.0,
        "imagenet_pretrained": True,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": True,
        "grad_scaler_initial_scale": 512.0,
    }


def validate_parent_summary(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> None:
    pin = formal_parent_pin(seed)
    if summary.get("protocol") != PEPD_TRAINING_PROTOCOL:
        raise ValueError("parent summary has the wrong training protocol")
    if summary.get("status") != "complete":
        raise ValueError("parent summary is not complete")
    signature = summary.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("parent summary has no signature")
    for name, expected in expected_full_configuration().items():
        actual = signature.get(name)
        if isinstance(expected, float):
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"parent signature {name} mismatch")
        elif actual != expected:
            raise ValueError(f"parent signature {name} mismatch")
    exact = {
        "seed": pin.seed,
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "model_source_sha256": FORMAL_MODEL_SOURCE_SHA256,
        "trainer_source_sha256": LEGACY_PARENT_TRAINER_SOURCE_SHA256,
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
    }
    for name, expected in exact.items():
        if signature.get(name) != expected:
            raise ValueError(f"parent signature {name} mismatch")
    history = summary.get("history")
    if not isinstance(history, Sequence) or len(history) != PARENT_EPOCH:
        raise ValueError("parent history must contain exactly 30 epochs")
    validate_parent_history(
        history,
        train_samples=pin.train_samples,
        validation_samples=pin.validation_samples,
    )
    if int(summary.get("best_epoch", -1)) != pin.best_epoch:
        raise ValueError("parent best epoch does not match its frozen pin")
    if not math.isclose(
        float(summary.get("best_validation_angle_mae_degrees", math.nan)),
        pin.best_angle_mae_degrees,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("parent best validation MAE does not match its frozen pin")


def validate_parent_history(
    history: Sequence[Mapping[str, Any]],
    *,
    train_samples: int,
    validation_samples: int,
) -> None:
    if len(history) != PARENT_EPOCH:
        raise ValueError("parent history must contain exactly 30 epochs")
    expected_batches = math.ceil(train_samples / FORMAL_BATCH_SIZE)
    base_lr = 3.0e-4
    eta_min = CONTINUATION_LEARNING_RATE
    for epoch, record in enumerate(history, start=1):
        if int(record.get("epoch", -1)) != epoch:
            raise ValueError("parent history epochs are not consecutive")
        expected_lr = eta_min + 0.5 * (base_lr - eta_min) * (
            1.0 + math.cos(math.pi * float(epoch - 1) / float(PARENT_EPOCH))
        )
        if not math.isclose(
            float(record.get("learning_rate", math.nan)),
            expected_lr,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"parent epoch {epoch}: cosine learning rate mismatch")
        train = record.get("train") or {}
        validation = record.get("validation") or {}
        if int(train.get("samples", -1)) != train_samples:
            raise ValueError(f"parent epoch {epoch}: train sample count mismatch")
        if int(validation.get("samples", -1)) != validation_samples:
            raise ValueError(
                f"parent epoch {epoch}: validation sample count mismatch"
            )
        steps = int(train.get("optimizer_steps", -1))
        skipped = int(train.get("skipped_optimizer_steps", -1))
        if steps < 0 or skipped < 0 or steps + skipped != expected_batches:
            raise ValueError(
                f"parent epoch {epoch}: optimizer-step accounting mismatch"
            )
        required = (
            train.get("loss"),
            validation.get("loss"),
            validation.get("angle_mae_degrees"),
            validation.get("angular_calibration_nll"),
            validation.get("pivot_mean_error_fraction"),
            validation.get("direction_coverage"),
        )
        if not all(math.isfinite(float(value)) for value in required):
            raise ValueError(f"parent epoch {epoch}: non-finite metric")
        if not math.isclose(
            float(validation["direction_coverage"]),
            REQUIRED_DIRECTION_COVERAGE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"parent epoch {epoch}: coverage is not exactly 1")


def build_continuation_signature(
    *,
    seed: int,
    parent_summary_sha256: str,
    parent_best_sha256: str,
    parent_last_sha256: str,
    continuation_source_sha256: str,
    imported_trainer_source_sha256: str,
    model_source_sha256: str,
) -> dict[str, Any]:
    pin = formal_parent_pin(seed)
    return {
        "protocol": PEPD_CONTINUATION_PROTOCOL,
        "scope": "SyncG official train grouped validation only",
        "role": "convergence audit of retained PEPD; not algorithm selection",
        "seed": int(seed),
        "continuation_seed": continuation_seed(seed),
        "parent_epoch": PARENT_EPOCH,
        "terminal_epoch": TERMINAL_EPOCH,
        "continuation_epochs": CONTINUATION_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "learning_rate_policy": "constant_at_parent_cosine_eta_min",
        "continuation_learning_rate": CONTINUATION_LEARNING_RATE,
        "early_stopping": False,
        "checkpoint_selection": (
            "lexicographic minimum grouped-validation "
            "(angle_mae_degrees, angular_calibration_nll, "
            "pivot_mean_error_fraction) over epochs 1..60"
        ),
        "rng_boundary_policy": (
            "deterministic phase-boundary reset because legacy epoch-30 "
            "checkpoints did not store RNG/DataLoader states"
        ),
        "parent_run_dir": pin.run_dir,
        "parent_summary_sha256": parent_summary_sha256,
        "parent_best_sha256": parent_best_sha256,
        "parent_last_sha256": parent_last_sha256,
        "manifest": FORMAL_MANIFEST.as_posix(),
        "manifest_sha256": FORMAL_MANIFEST_SHA256,
        "manifest_protocol_sha256": FORMAL_MANIFEST_PROTOCOL_SHA256,
        "train_samples": pin.train_samples,
        "validation_samples": pin.validation_samples,
        "train_sample_ids_sha256": pin.train_ids_sha256,
        "validation_sample_ids_sha256": pin.validation_ids_sha256,
        "model_source_sha256": model_source_sha256,
        "strict_json_source_sha256": strict_json_source_sha256(),
        "imported_epoch_trainer_source_sha256": imported_trainer_source_sha256,
        "continuation_trainer_source_sha256": continuation_source_sha256,
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "optimizer": "AdamW state continued exactly from parent last.pt",
        "grad_scaler": "state continued exactly from parent last.pt",
        "scheduler": (
            "parent scheduler is audited at terminal eta_min and then retired; "
            "no phase-2 scheduler state is introduced"
        ),
    }


def _finite_metric(record: Mapping[str, Any], section: str, name: str) -> float:
    value = float((record.get(section) or {}).get(name, math.nan))
    if not math.isfinite(value):
        raise ValueError(f"epoch {record.get('epoch')}: non-finite {section}.{name}")
    return value


def validate_combined_history(
    history: Sequence[Mapping[str, Any]],
    *,
    train_samples: int,
    validation_samples: int,
    require_calibration_metric: bool = True,
) -> None:
    if len(history) != TERMINAL_EPOCH:
        raise ValueError(
            f"combined history must contain exactly {TERMINAL_EPOCH} epochs"
        )
    expected_batches = math.ceil(train_samples / FORMAL_BATCH_SIZE)
    for expected_epoch, record in enumerate(history, start=1):
        if int(record.get("epoch", -1)) != expected_epoch:
            raise ValueError("combined history epochs are not consecutive")
        train = record.get("train") or {}
        validation = record.get("validation") or {}
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
        validation_names = [
            "loss",
            "angle_mae_degrees",
            "pivot_mean_error_fraction",
            "direction_coverage",
        ]
        if require_calibration_metric:
            validation_names.append("angular_calibration_nll")
        for section, names in {
            "train": ("loss",),
            "validation": tuple(validation_names),
        }.items():
            for name in names:
                _finite_metric(record, section, name)
        coverage = float(validation["direction_coverage"])
        if not math.isclose(
            coverage,
            REQUIRED_DIRECTION_COVERAGE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"epoch {expected_epoch}: direction coverage is not exactly 1"
            )
        if expected_epoch > PARENT_EPOCH and not math.isclose(
            float(record.get("learning_rate", math.nan)),
            CONTINUATION_LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"epoch {expected_epoch}: continuation learning rate drifted"
            )


def convergence_audit(
    history: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int,
) -> dict[str, Any]:
    """Evaluate the fixed, non-adaptive convergence gate.

    The gate is diagnostic and is evaluated only after epoch 60.  It never
    shortens training and never authorizes an adaptive extension.
    """

    if len(history) < TERMINAL_EPOCH:
        raise ValueError("convergence audit requires the complete 60-epoch history")
    errors = np.asarray(
        [
            float((record.get("validation") or {}).get("angle_mae_degrees"))
            for record in history
        ],
        dtype=np.float64,
    )
    if errors.shape != (TERMINAL_EPOCH,) or not np.all(np.isfinite(errors)):
        raise ValueError("convergence audit received invalid validation MAE history")
    trailing = errors[-CONVERGENCE_WINDOW:]
    x = np.arange(CONVERGENCE_WINDOW, dtype=np.float64)
    slope = float(np.polyfit(x, trailing, deg=1)[0])
    spread = float(np.ptp(trailing))
    previous = errors[
        -(2 * CONVERGENCE_WINDOW) : -CONVERGENCE_WINDOW
    ]
    late_gain = float(np.mean(previous) - np.mean(trailing))
    boundary = errors[-CONVERGENCE_BOUNDARY_EPOCHS:]
    preceding = errors[
        -CONVERGENCE_WINDOW : -CONVERGENCE_BOUNDARY_EPOCHS
    ]
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
            int(best_epoch) > TERMINAL_EPOCH - CONVERGENCE_BOUNDARY_EPOCHS
            and boundary_gain > CONVERGENCE_LATE_GAIN_LIMIT_DEGREES
        ),
    }
    return {
        "protocol": PEPD_CONTINUATION_PROTOCOL,
        "window_epochs": CONVERGENCE_WINDOW,
        "trailing_epochs": [
            TERMINAL_EPOCH - CONVERGENCE_WINDOW + 1,
            TERMINAL_EPOCH,
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
            "stop; do not inspect public/test/field data and do not extend "
            "epochs without a newly versioned pre-frozen protocol"
        ),
    }


def sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
