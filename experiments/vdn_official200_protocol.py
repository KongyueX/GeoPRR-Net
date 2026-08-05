"""Frozen training-only protocol for the official 200-epoch VDN endpoint.

This protocol is intentionally independent from every test/public/field
evaluation path.  It defines one equal-budget, from-scratch run for each of
three seeds and a hard stopping boundary at epoch 200.  Tail behaviour is
reported as a diagnostic only and can never authorize an adaptive extension.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.compare_pct_determinism_runs import semantic_sha256
from experiments.freeze_vdn_phase2_syncg_train_inventory import (
    DEFAULT_OUTPUT as DEFAULT_CONTENT_INVENTORY,
    INVENTORY_PROTOCOL,
    INVENTORY_VERIFICATION_PROTOCOL,
    SYNCG_TRAIN_EXPECTED_ROWS,
)
from experiments.train_vdn_syncg import sample_order_sha256
from experiments.vdn_baseline import (
    PROJECT_DIR,
    VDN_PINNED_COMMIT,
    sample_ids_hash,
    sha256_file,
    sha256_source_file,
)
from experiments.vdn_phase2_protocol import (
    PHASE2_ADAM_PARAM_GROUP_POLICY,
    PHASE2_ADAM_STATE_POLICY,
    PHASE2_AMP_SCALER_POLICY,
    PHASE2_DETERMINISM_POLICY,
    PHASE2_MAX_SKIPPED_STEP_RATE,
    PHASE2_TRAIN_METRIC_KEYS,
    PHASE2_VALIDATION_METRIC_KEYS,
    VDN_TRAINABLE_PARAMETER_TENSORS,
    apply_phase2_determinism_policy,
    build_phase2_adam_optimizer,
    current_runtime_environment,
    expected_optimizer_steps,
    normalize_content_inventory_identity,
    strict_model_state_health,
    validate_adam_optimizer_state,
    validate_live_adam_optimizer,
    validate_scaler_state,
    validate_scaler_transition,
)


OFFICIAL200_PROTOCOL = "vdn_syncg_official_200_epoch_from_scratch_v1"
OFFICIAL200_PREFLIGHT_PROTOCOL = "vdn_official200_training_preflight_v1"
OFFICIAL200_CHECKPOINT_PROTOCOL = (
    "vdn_official200_atomic_authoritative_checkpoint_v1"
)
OFFICIAL200_VERIFICATION_PROTOCOL = "formal_vdn_official200_verification_v1"
OFFICIAL200_COHORT_PROTOCOL = (
    "formal_vdn_official200_three_seed_cohort_verification_v1"
)
OFFICIAL200_DETERMINISM_PROTOCOL = (
    "vdn_official200_full_epoch_determinism_probe_v1"
)
OFFICIAL200_SCHEMA_VERSION = 1
OFFICIAL200_EPOCHS = 200
OFFICIAL200_MILESTONES = (140, 190)
OFFICIAL200_LR_SEGMENTS = (
    {"start_epoch": 1, "end_epoch": 140, "learning_rate": 1e-3},
    {"start_epoch": 141, "end_epoch": 190, "learning_rate": 1e-4},
    {"start_epoch": 191, "end_epoch": 200, "learning_rate": 1e-5},
)
OFFICIAL200_FORMAL_SEEDS = (20260720, 20260721, 20260722)
OFFICIAL200_EPOCH_SEED_MULTIPLIER = 1009
OFFICIAL200_BATCH_SIZE = 8
OFFICIAL200_WORKERS = 4
OFFICIAL200_IMAGE_SIZE = 384
OFFICIAL200_VALIDATION_FRACTION = 0.10
OFFICIAL200_SCALE_FACTOR = 0.02
OFFICIAL200_ROTATION_FACTOR = 90.0
OFFICIAL200_INITIAL_LEARNING_RATE = 1e-3
OFFICIAL200_WEIGHT_DECAY = 0.0
OFFICIAL200_MAX_SKIPPED_STEP_RATE = PHASE2_MAX_SKIPPED_STEP_RATE
OFFICIAL200_SOURCE_HASH_PROTOCOL = "utf8_source_newlines_lf_v1"
OFFICIAL200_CONTENT_INVENTORY_PROTOCOL = INVENTORY_PROTOCOL
OFFICIAL200_CONTENT_INVENTORY_VERIFICATION_PROTOCOL = (
    INVENTORY_VERIFICATION_PROTOCOL
)
OFFICIAL200_CONTENT_INVENTORY_PATH = DEFAULT_CONTENT_INVENTORY
OFFICIAL200_CONTENT_INVENTORY_ROWS = SYNCG_TRAIN_EXPECTED_ROWS
OFFICIAL200_TAIL_WINDOW = 10
OFFICIAL200_BOUNDARY_WINDOW = 5
OFFICIAL200_TAIL_RELATIVE_CHANGE_LIMIT = 0.01
OFFICIAL200_STOPPING_POLICY = {
    "official_budget_epochs": OFFICIAL200_EPOCHS,
    "hard_stopping_boundary": True,
    "adaptive_extension_allowed": False,
    "phase4_allowed": False,
    "tail_diagnostic_is_authorization_gate": False,
    "non_plateau_action": (
        "report_manuscript_limitation_without_additional_training"
    ),
}
OFFICIAL200_SCOPE = (
    "pinned SyncG official train grouped validation only; no SyncG test, "
    "public, RPM-10K, Pointer-10K, field, sealed, or confirmatory input"
)
OFFICIAL200_FORBIDDEN_PATH_NAMESPACES = frozenset(
    {
        "test",
        "tests",
        "public",
        "field",
        "sealed",
        "confirmatory",
        "rpm-10k",
        "rpm_10k",
        "pointer-10k",
        "pointer_10k",
        "syncg_test",
    }
)
OFFICIAL200_DETERMINISM_POLICY = PHASE2_DETERMINISM_POLICY
OFFICIAL200_ADAM_PARAM_GROUP_POLICY = PHASE2_ADAM_PARAM_GROUP_POLICY
OFFICIAL200_ADAM_STATE_POLICY = PHASE2_ADAM_STATE_POLICY
OFFICIAL200_AMP_SCALER_POLICY = PHASE2_AMP_SCALER_POLICY
OFFICIAL200_TRAIN_METRIC_KEYS = PHASE2_TRAIN_METRIC_KEYS
OFFICIAL200_VALIDATION_METRIC_KEYS = PHASE2_VALIDATION_METRIC_KEYS
OFFICIAL200_HISTORY_ROW_KEYS = frozenset(
    {
        "epoch",
        "epoch_seed",
        "learning_rate",
        "train",
        "validation",
        "best",
        "training_elapsed_seconds",
        "preflight_journal",
        "determinism_authorization",
        "determinism_policy",
    }
)
OFFICIAL200_DETERMINISM_REPORT_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "status",
        "official200_training_start_authorized",
        "supporting_test_evaluation_authorized",
        "field_confirmatory_evaluation_authorized",
        "fixed_probe",
        "preflight",
        "scientific_identity",
        "components",
        "worker_execution",
        "source_hash_protocol",
        "probe_source_sha256",
        "training_source_sha256",
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
        "canonical_probe_payload_sha256",
    }
)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_train_only_path(path: Path, *, label: str) -> Path:
    """Reject any formal path carrying an evaluation-data namespace."""

    resolved = Path(path).resolve()
    for part in resolved.parts:
        lowered = part.casefold()
        tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", lowered)
            if token
        }
        if (
            lowered in OFFICIAL200_FORBIDDEN_PATH_NAMESPACES
            or lowered.replace("_", "-")
            in OFFICIAL200_FORBIDDEN_PATH_NAMESPACES
            or tokens.intersection(
                {"test", "tests", "public", "field", "sealed", "confirmatory"}
            )
            or "rpm-10k" in lowered.replace("_", "-")
            or "pointer-10k" in lowered.replace("_", "-")
            or "syncg-test" in lowered.replace("_", "-")
        ):
            raise ValueError(
                f"{label} enters a forbidden evaluation namespace: "
                f"{resolved}"
            )
    return resolved


def assert_syncg_train_manifest_path(path: Path) -> Path:
    resolved = assert_train_only_path(path, label="official-200 manifest")
    if resolved.name.casefold() != "syncg_train.jsonl":
        raise ValueError(
            "official-200 manifest must be the pinned syncg_train.jsonl"
        )
    return resolved


def validate_determinism_report_payload(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    preflight_binding: Mapping[str, Any],
    content_inventory_identity: Mapping[str, Any],
    vdn_source: Path,
) -> dict[str, Any]:
    """Validate a parsed full-epoch probe without importing its GPU module."""

    report_path = assert_train_only_path(
        report_path,
        label="official-200 determinism report",
    )
    if set(report) != OFFICIAL200_DETERMINISM_REPORT_KEYS:
        raise ValueError("official-200 determinism report schema drifted")
    canonical = report.get("canonical_probe_payload_sha256")
    payload = dict(report)
    payload.pop("canonical_probe_payload_sha256", None)
    if canonical != canonical_json_sha256(payload):
        raise ValueError(
            "official-200 determinism report canonical digest drifted"
        )
    expected_probe_source = sha256_source_file(
        PROJECT_DIR
        / "experiments"
        / "probe_vdn_official200_determinism.py"
    )
    expected_training_sources = official200_source_hashes(vdn_source)
    if (
        report.get("protocol") != OFFICIAL200_DETERMINISM_PROTOCOL
        or int(report.get("schema_version", -1))
        != OFFICIAL200_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("official200_training_start_authorized") is not True
        or report.get("supporting_test_evaluation_authorized") is not False
        or report.get("field_confirmatory_evaluation_authorized") is not False
        or report.get("preflight") != dict(preflight_binding)
        or report.get("probe_source_sha256") != expected_probe_source
        or report.get("training_source_sha256")
        != expected_training_sources
    ):
        raise ValueError("official-200 determinism authorization is stale")
    fixed = report.get("fixed_probe")
    if (
        not isinstance(fixed, Mapping)
        or fixed.get("seed") != OFFICIAL200_FORMAL_SEEDS[0]
        or fixed.get("epoch") != 1
        or fixed.get("epoch_seed")
        != official200_epoch_seed(OFFICIAL200_FORMAL_SEEDS[0], 1)
        or fixed.get("complete_train_and_validation_epoch") is not True
        or fixed.get("separate_python_processes") is not True
        or fixed.get(
            "model_optimizer_scaler_dataloader_rebuilt_per_worker"
        )
        is not True
    ):
        raise ValueError("official-200 fixed determinism probe drifted")
    components = report.get("components")
    if not isinstance(components, Mapping) or not components:
        raise ValueError("official-200 determinism components are absent")
    for name, component in components.items():
        if (
            not isinstance(component, Mapping)
            or set(component) != {"exact", "semantic_sha256"}
            or component.get("exact") is not True
            or not isinstance(component.get("semantic_sha256"), str)
            or len(component["semantic_sha256"]) != 64
        ):
            raise ValueError(
                f"official-200 determinism component {name} is not exact"
            )
    for field in (
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
    ):
        if report.get(field) is not False:
            raise ValueError(
                f"official-200 determinism provenance drifted: {field}"
            )
    identity = report.get("scientific_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("preflight") != dict(preflight_binding)
        or identity.get("content_inventory")
        != dict(content_inventory_identity)
        or identity.get("training_source_sha256")
        != expected_training_sources
    ):
        raise ValueError(
            "official-200 determinism scientific identity drifted"
        )
    execution = report.get("worker_execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("process_tokens_distinct") is not True
        or execution.get("a_pid") == execution.get("b_pid")
        or execution.get("a_semantic_payload_sha256")
        != execution.get("b_semantic_payload_sha256")
    ):
        raise ValueError(
            "official-200 determinism worker execution drifted"
        )
    return {
        "protocol": OFFICIAL200_DETERMINISM_PROTOCOL,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "canonical_payload_sha256": canonical,
        "semantic_payload_sha256": execution[
            "a_semantic_payload_sha256"
        ],
    }


def validate_authorized_runtime_environment(
    report: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> dict[str, str | bool]:
    identity = report.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("official-200 probe scientific identity is absent")
    authorized = dict(identity.get("runtime_environment") or {})
    current = dict(runtime_environment)
    authorized.pop("pythonhashseed", None)
    current.pop("pythonhashseed", None)
    authorized.pop("determinism_authorization", None)
    current.pop("determinism_authorization", None)
    if authorized != current:
        raise RuntimeError(
            "official-200 runtime differs from deterministic probe hardware"
        )
    return {
        "exact_authorized_runtime_except_seed_specific_pythonhashseed": True,
        "probe_runtime_semantic_sha256": semantic_sha256(authorized),
        "current_runtime_semantic_sha256": semantic_sha256(current),
    }


def require_formal_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in OFFICIAL200_FORMAL_SEEDS:
        raise ValueError(
            f"official-200 seed must be one of "
            f"{list(OFFICIAL200_FORMAL_SEEDS)}, got {seed}"
        )
    return seed


def official200_learning_rate(epoch: int) -> float:
    epoch = int(epoch)
    for segment in OFFICIAL200_LR_SEGMENTS:
        if int(segment["start_epoch"]) <= epoch <= int(
            segment["end_epoch"]
        ):
            return float(segment["learning_rate"])
    raise ValueError(
        f"official-200 epoch must be in [1, {OFFICIAL200_EPOCHS}], "
        f"got {epoch}"
    )


def official200_vector_weight(epoch: int) -> float:
    epoch = int(epoch)
    if not 1 <= epoch <= OFFICIAL200_EPOCHS:
        raise ValueError(f"invalid official-200 epoch {epoch}")
    return float(epoch - 1) / float(OFFICIAL200_EPOCHS - 1)


def official200_epoch_seed(seed: int, epoch: int) -> int:
    seed = require_formal_seed(seed)
    epoch = int(epoch)
    if not 1 <= epoch <= OFFICIAL200_EPOCHS:
        raise ValueError(f"invalid official-200 epoch {epoch}")
    return seed + (epoch - 1) * OFFICIAL200_EPOCH_SEED_MULTIPLIER


def official200_initial_scaler_state() -> dict[str, Any]:
    return {
        "scale": float(OFFICIAL200_AMP_SCALER_POLICY["initial_scale"]),
        "growth_factor": float(
            OFFICIAL200_AMP_SCALER_POLICY["growth_factor"]
        ),
        "backoff_factor": float(
            OFFICIAL200_AMP_SCALER_POLICY["backoff_factor"]
        ),
        "growth_interval": int(
            OFFICIAL200_AMP_SCALER_POLICY["growth_interval"]
        ),
        "_growth_tracker": 0,
    }


def official200_source_hashes(vdn_source: Path) -> dict[str, str]:
    vdn_source = Path(vdn_source).resolve()
    return {
        "official200_protocol": sha256_source_file(Path(__file__).resolve()),
        "official200_preflight": sha256_source_file(
            PROJECT_DIR / "experiments" / "preflight_vdn_official200.py"
        ),
        "official200_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_official200.py"
        ),
        "official200_determinism_probe": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "probe_vdn_official200_determinism.py"
        ),
        "official200_supervisor": sha256_source_file(
            PROJECT_DIR / "experiments" / "run_vdn_official200.ps1"
        ),
        "official200_protocol_document": sha256_source_file(
            PROJECT_DIR / "docs" / "VDN_OFFICIAL200_PROTOCOL_CN.md"
        ),
        "base_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_syncg.py"
        ),
        "hardened_state_protocol": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_phase2_protocol.py"
        ),
        "content_inventory_tool": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "freeze_vdn_phase2_syncg_train_inventory.py"
        ),
        "adapter": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "vdn_model": sha256_source_file(
            vdn_source / "libs" / "models" / "vdn_model.py"
        ),
    }


def model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("VDN model state is empty")
    return semantic_sha256(dict(state))


def expected_sample_order_sha256(
    train_samples: Sequence[Any],
    *,
    epoch_seed: int,
    batch_size: int = OFFICIAL200_BATCH_SIZE,
) -> str:
    sample_ids = [str(sample.sample_id) for sample in train_samples]
    if not sample_ids:
        raise ValueError("official-200 train split is empty")
    loader = DataLoader(
        sample_ids,
        batch_size=int(batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(epoch_seed)),
        num_workers=0,
        persistent_workers=False,
    )
    observed: list[str] = []
    for batch in loader:
        observed.extend(str(sample_id) for sample_id in batch)
    return sample_order_sha256(observed)


def full_run_skip_budget(train_samples: int) -> int:
    attempts = expected_optimizer_steps(
        int(train_samples),
        OFFICIAL200_BATCH_SIZE,
    ) * OFFICIAL200_EPOCHS
    return math.floor(attempts * OFFICIAL200_MAX_SKIPPED_STEP_RATE)


def build_official200_signature(
    *,
    seed: int,
    manifest: Path,
    manifest_protocol: Path,
    vdn_source: Path,
    train_samples: Sequence[Any],
    validation_samples: Sequence[Any],
    initialization_checkpoint: Path,
    initialization_state_sha256: str,
    content_inventory_identity: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
    determinism_authorization: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    seed = require_formal_seed(seed)
    manifest = Path(manifest).resolve()
    manifest_protocol = Path(manifest_protocol).resolve()
    vdn_source = Path(vdn_source).resolve()
    initialization_checkpoint = Path(initialization_checkpoint).resolve()
    if not manifest.is_file() or not manifest_protocol.is_file():
        raise FileNotFoundError("official-200 manifest identity is incomplete")
    if not initialization_checkpoint.is_file():
        raise FileNotFoundError(initialization_checkpoint)
    if len(train_samples) + len(validation_samples) != (
        OFFICIAL200_CONTENT_INVENTORY_ROWS
    ):
        raise ValueError("official-200 grouped split does not cover 16000 rows")
    if (
        not isinstance(initialization_state_sha256, str)
        or len(initialization_state_sha256) != 64
    ):
        raise ValueError("official-200 initial model digest is invalid")
    source_sha256 = dict(source_sha256)
    if source_sha256 != official200_source_hashes(vdn_source):
        raise ValueError("official-200 training source identity is stale")
    content_inventory_identity = dict(content_inventory_identity)
    if (
        content_inventory_identity.get("inventory_protocol")
        != OFFICIAL200_CONTENT_INVENTORY_PROTOCOL
        or content_inventory_identity.get("verification_protocol")
        != OFFICIAL200_CONTENT_INVENTORY_VERIFICATION_PROTOCOL
        or content_inventory_identity.get("fresh_content_rehashed") is not True
        or int(content_inventory_identity.get("rows", -1))
        != OFFICIAL200_CONTENT_INVENTORY_ROWS
    ):
        raise ValueError("official-200 content inventory identity is invalid")
    preflight_binding = dict(preflight_binding)
    if set(preflight_binding) != {
        "protocol",
        "report_path",
        "report_sha256",
        "canonical_payload_sha256",
    }:
        raise ValueError("official-200 preflight binding schema is invalid")
    if preflight_binding["protocol"] != OFFICIAL200_PREFLIGHT_PROTOCOL:
        raise ValueError("official-200 preflight protocol is invalid")
    determinism_authorization = dict(determinism_authorization)
    if set(determinism_authorization) != {
        "protocol",
        "report_path",
        "report_sha256",
        "canonical_payload_sha256",
        "semantic_payload_sha256",
    }:
        raise ValueError(
            "official-200 determinism authorization schema is invalid"
        )
    if (
        determinism_authorization["protocol"]
        != OFFICIAL200_DETERMINISM_PROTOCOL
    ):
        raise ValueError(
            "official-200 determinism authorization protocol is invalid"
        )
    runtime_environment = dict(runtime_environment)
    if (
        runtime_environment.get("amp") is not True
        or runtime_environment.get("determinism")
        != OFFICIAL200_DETERMINISM_POLICY
    ):
        raise ValueError("official-200 runtime environment is not authorized")

    attempts_per_epoch = expected_optimizer_steps(
        len(train_samples),
        OFFICIAL200_BATCH_SIZE,
    )
    return {
        "protocol": OFFICIAL200_PROTOCOL,
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "scope": OFFICIAL200_SCOPE,
        "seed": seed,
        "epochs": OFFICIAL200_EPOCHS,
        "milestones": list(OFFICIAL200_MILESTONES),
        "learning_rate_segments": [
            dict(segment) for segment in OFFICIAL200_LR_SEGMENTS
        ],
        "vector_weight_schedule": (
            "linear_epoch_minus_1_over_199_from_0_to_1"
        ),
        "batch_size": OFFICIAL200_BATCH_SIZE,
        "workers": OFFICIAL200_WORKERS,
        "image_size": OFFICIAL200_IMAGE_SIZE,
        "heatmap_size": OFFICIAL200_IMAGE_SIZE // 4,
        "validation_fraction": OFFICIAL200_VALIDATION_FRACTION,
        "scale_factor": OFFICIAL200_SCALE_FACTOR,
        "rotation_factor": OFFICIAL200_ROTATION_FACTOR,
        "optimizer": "Adam",
        "adam_param_group_policy": OFFICIAL200_ADAM_PARAM_GROUP_POLICY,
        "adam_state_policy": OFFICIAL200_ADAM_STATE_POLICY,
        "weight_decay": OFFICIAL200_WEIGHT_DECAY,
        "mixed_precision": True,
        "amp_scaler_policy": OFFICIAL200_AMP_SCALER_POLICY,
        "max_skipped_optimizer_step_rate": (
            OFFICIAL200_MAX_SKIPPED_STEP_RATE
        ),
        "optimizer_attempts_per_epoch": attempts_per_epoch,
        "full_run_optimizer_attempts": (
            attempts_per_epoch * OFFICIAL200_EPOCHS
        ),
        "full_run_max_skipped_optimizer_steps": full_run_skip_budget(
            len(train_samples)
        ),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(manifest_protocol),
        "content_inventory": content_inventory_identity,
        "vdn_source": str(vdn_source),
        "vdn_source_commit": VDN_PINNED_COMMIT,
        "imagenet_pretrained": True,
        "imagenet_initialization_checkpoint": str(
            initialization_checkpoint
        ),
        "imagenet_initialization_sha256": sha256_file(
            initialization_checkpoint
        ),
        "initial_model_state_sha256": initialization_state_sha256,
        "determinism": OFFICIAL200_DETERMINISM_POLICY,
        "epoch_seed_protocol": (
            "formal_seed_plus_epoch_minus_1_times_1009_v1"
        ),
        "initial_scaler_state": official200_initial_scaler_state(),
        "preflight": preflight_binding,
        "determinism_authorization": determinism_authorization,
        "runtime_environment": runtime_environment,
        "source_hash_protocol": OFFICIAL200_SOURCE_HASH_PROTOCOL,
        "source_sha256": source_sha256,
        "stopping_policy": OFFICIAL200_STOPPING_POLICY,
    }


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}"
        )


def validate_official200_history(
    train_samples: Sequence[Any],
    history: Sequence[Mapping[str, Any]],
    *,
    through_epoch: int,
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    through_epoch = int(through_epoch)
    if not 1 <= through_epoch <= OFFICIAL200_EPOCHS:
        raise ValueError("official-200 history endpoint is invalid")
    if len(history) != through_epoch:
        raise ValueError("official-200 history length is invalid")
    if int(signature.get("seed", -1)) not in OFFICIAL200_FORMAL_SEEDS:
        raise ValueError("official-200 history signature seed is invalid")
    expected_attempts = expected_optimizer_steps(
        int(signature["train_samples"]),
        int(signature["batch_size"]),
    )
    full_attempts = expected_attempts * OFFICIAL200_EPOCHS
    full_skip_budget = int(
        signature["full_run_max_skipped_optimizer_steps"]
    )
    _require_equal(
        full_skip_budget,
        math.floor(
            full_attempts * OFFICIAL200_MAX_SKIPPED_STEP_RATE
        ),
        label="official-200 full-run skip budget",
    )
    previous_scaler_state = dict(signature["initial_scaler_state"])
    running_best = math.inf
    cumulative_successful = 0
    cumulative_skipped = 0
    elapsed_values: list[float] = []
    for expected_epoch, row_value in enumerate(history, start=1):
        row = dict(row_value)
        _require_equal(
            set(row),
            OFFICIAL200_HISTORY_ROW_KEYS,
            label=f"official-200 epoch {expected_epoch} row schema",
        )
        _require_equal(
            int(row.get("epoch", -1)),
            expected_epoch,
            label=f"official-200 epoch {expected_epoch} identity",
        )
        epoch_seed = official200_epoch_seed(
            int(signature["seed"]),
            expected_epoch,
        )
        _require_equal(
            int(row.get("epoch_seed", -1)),
            epoch_seed,
            label=f"official-200 epoch {expected_epoch} seed",
        )
        if not math.isclose(
            float(row.get("learning_rate", math.nan)),
            official200_learning_rate(expected_epoch),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} learning rate drifted"
            )
        _require_equal(
            row.get("preflight_journal"),
            signature["preflight"],
            label=f"official-200 epoch {expected_epoch} preflight journal",
        )
        _require_equal(
            row.get("determinism_policy"),
            signature["determinism"],
            label=f"official-200 epoch {expected_epoch} determinism",
        )
        _require_equal(
            row.get("determinism_authorization"),
            signature["determinism_authorization"],
            label=(
                f"official-200 epoch {expected_epoch} "
                "determinism authorization"
            ),
        )

        train = row.get("train")
        validation = row.get("validation")
        if not isinstance(train, Mapping) or set(train) != (
            OFFICIAL200_TRAIN_METRIC_KEYS
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} train schema drifted"
            )
        if not isinstance(validation, Mapping) or set(validation) != (
            OFFICIAL200_VALIDATION_METRIC_KEYS
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} validation schema drifted"
            )
        for field in OFFICIAL200_TRAIN_METRIC_KEYS - {
            "sample_order_sha256",
            "scaler_start_state",
            "scaler_skipped_batch_indices",
            "scaler_end_state",
        }:
            value = train[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"official-200 epoch {expected_epoch} train.{field} "
                    "is not finite numeric"
                )
        for field in OFFICIAL200_VALIDATION_METRIC_KEYS:
            value = validation[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"official-200 epoch {expected_epoch} validation.{field} "
                    "is not finite numeric"
                )
        for field in (
            "samples",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            if type(train[field]) is not int:
                raise ValueError(
                    f"official-200 epoch {expected_epoch} train.{field} "
                    "is not integer"
                )
        for field in ("samples", "valid_directions"):
            if type(validation[field]) is not int:
                raise ValueError(
                    f"official-200 epoch {expected_epoch} "
                    f"validation.{field} is not integer"
                )

        successful = int(train["optimizer_steps"])
        skipped = int(train["skipped_optimizer_steps"])
        if (
            successful <= 0
            or skipped < 0
            or successful + skipped != expected_attempts
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} optimizer accounting "
                "does not close"
            )
        _require_equal(
            train["scaler_start_state"],
            previous_scaler_state,
            label=f"official-200 epoch {expected_epoch} scaler continuity",
        )
        transition = validate_scaler_transition(
            train["scaler_start_state"],
            train["scaler_end_state"],
            train["scaler_skipped_batch_indices"],
            attempted_steps=expected_attempts,
            label=f"official-200 epoch {expected_epoch}",
        )
        _require_equal(
            transition["successful_steps"],
            successful,
            label=f"official-200 epoch {expected_epoch} successful steps",
        )
        _require_equal(
            transition["skipped_steps"],
            skipped,
            label=f"official-200 epoch {expected_epoch} skipped steps",
        )
        previous_scaler_state = dict(train["scaler_end_state"])
        cumulative_successful += successful
        cumulative_skipped += skipped
        if cumulative_skipped > full_skip_budget:
            raise ValueError(
                "official-200 cumulative AMP skips exceed the frozen "
                "full-run budget"
            )

        _require_equal(
            int(train["samples"]),
            int(signature["train_samples"]),
            label=f"official-200 epoch {expected_epoch} train samples",
        )
        _require_equal(
            int(validation["samples"]),
            int(signature["validation_samples"]),
            label=f"official-200 epoch {expected_epoch} validation samples",
        )
        if not math.isclose(
            float(train["vector_weight"]),
            official200_vector_weight(expected_epoch),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} vector weight drifted"
            )
        expected_order = expected_sample_order_sha256(
            train_samples,
            epoch_seed=epoch_seed,
            batch_size=int(signature["batch_size"]),
        )
        _require_equal(
            train["sample_order_sha256"],
            expected_order,
            label=f"official-200 epoch {expected_epoch} sample order",
        )

        valid_directions = int(validation["valid_directions"])
        validation_samples = int(validation["samples"])
        if not 0 < valid_directions <= validation_samples:
            raise ValueError(
                f"official-200 epoch {expected_epoch} direction count invalid"
            )
        if not math.isclose(
            float(validation["direction_coverage"]),
            valid_directions / validation_samples,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} coverage drifted"
            )
        for field in ("angle_acc_1deg", "angle_acc_3deg", "angle_acc_5deg"):
            if not 0.0 <= float(validation[field]) <= 1.0:
                raise ValueError(
                    f"official-200 epoch {expected_epoch} {field} invalid"
                )
        if not (
            float(validation["angle_acc_1deg"])
            <= float(validation["angle_acc_3deg"])
            <= float(validation["angle_acc_5deg"])
        ):
            raise ValueError(
                f"official-200 epoch {expected_epoch} accuracies inconsistent"
            )

        current_angle = float(validation["angle_mae_degrees"])
        improved = current_angle < running_best
        _require_equal(
            row["best"],
            improved,
            label=f"official-200 epoch {expected_epoch} best flag",
        )
        if improved:
            running_best = current_angle
        elapsed = float(row["training_elapsed_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                f"official-200 epoch {expected_epoch} elapsed time invalid"
            )
        elapsed_values.append(elapsed)

    if any(
        current < previous
        for previous, current in zip(elapsed_values, elapsed_values[1:])
    ):
        raise ValueError("official-200 elapsed time is not monotonic")
    return {
        "epochs": through_epoch,
        "attempted_optimizer_steps_per_epoch": expected_attempts,
        "cumulative_attempted_optimizer_steps": (
            expected_attempts * through_epoch
        ),
        "cumulative_optimizer_steps": cumulative_successful,
        "cumulative_skipped_optimizer_steps": cumulative_skipped,
        "full_run_attempted_optimizer_steps": full_attempts,
        "full_run_max_skipped_optimizer_steps": full_skip_budget,
        "terminal_skipped_optimizer_step_rate": (
            cumulative_skipped / (expected_attempts * through_epoch)
        ),
        "terminal_scaler_state": previous_scaler_state,
        "sample_orders_verified": through_epoch,
    }


def tail_diagnostics(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report the frozen 200-epoch tail without authorizing more training."""

    if len(history) != OFFICIAL200_EPOCHS:
        raise ValueError("tail diagnostic requires exactly 200 epochs")
    actual_epochs = [int(row.get("epoch", -1)) for row in history]
    if actual_epochs != list(range(1, OFFICIAL200_EPOCHS + 1)):
        raise ValueError("tail diagnostic epoch sequence is invalid")
    angles = np.asarray(
        [
            float(row["validation"]["angle_mae_degrees"])
            for row in history
        ],
        dtype=np.float64,
    )
    losses = np.asarray(
        [float(row["validation"]["loss"]) for row in history],
        dtype=np.float64,
    )
    if not np.isfinite(angles).all() or not np.isfinite(losses).all():
        raise ValueError("tail diagnostic contains non-finite values")

    previous_slice = slice(
        OFFICIAL200_EPOCHS - 2 * OFFICIAL200_TAIL_WINDOW,
        OFFICIAL200_EPOCHS - OFFICIAL200_TAIL_WINDOW,
    )
    final_slice = slice(
        OFFICIAL200_EPOCHS - OFFICIAL200_TAIL_WINDOW,
        OFFICIAL200_EPOCHS,
    )
    previous_angle = float(np.mean(angles[previous_slice]))
    final_angle = float(np.mean(angles[final_slice]))
    previous_loss = float(np.mean(losses[previous_slice]))
    final_loss = float(np.mean(losses[final_slice]))
    epsilon = np.finfo(np.float64).eps
    angle_relative_change = (final_angle - previous_angle) / max(
        abs(previous_angle),
        epsilon,
    )
    loss_relative_change = (final_loss - previous_loss) / max(
        abs(previous_loss),
        epsilon,
    )
    best_index = int(np.argmin(angles))
    best_epoch = best_index + 1
    boundary_start = (
        OFFICIAL200_EPOCHS - OFFICIAL200_BOUNDARY_WINDOW + 1
    )
    checks = {
        "best_not_in_final_boundary_window": best_epoch < boundary_start,
        "angle_absolute_relative_change_below_1pct": (
            abs(angle_relative_change)
            < OFFICIAL200_TAIL_RELATIVE_CHANGE_LIMIT
        ),
        "loss_absolute_relative_change_below_1pct": (
            abs(loss_relative_change)
            < OFFICIAL200_TAIL_RELATIVE_CHANGE_LIMIT
        ),
    }
    plateau_observed = all(checks.values())
    return {
        "protocol": "vdn_official200_tail_diagnostic_v1",
        "diagnostic_only": True,
        "authorization_gate": False,
        "plateau_observed": plateau_observed,
        "checks": checks,
        "best_epoch": best_epoch,
        "best_validation_angle_mae_degrees": float(angles[best_index]),
        "boundary_window": [boundary_start, OFFICIAL200_EPOCHS],
        "rolling_windows": {
            "previous": [
                OFFICIAL200_EPOCHS - 2 * OFFICIAL200_TAIL_WINDOW + 1,
                OFFICIAL200_EPOCHS - OFFICIAL200_TAIL_WINDOW,
            ],
            "final": [
                OFFICIAL200_EPOCHS - OFFICIAL200_TAIL_WINDOW + 1,
                OFFICIAL200_EPOCHS,
            ],
        },
        "angle_mae": {
            "previous_mean": previous_angle,
            "final_mean": final_angle,
            "relative_change": angle_relative_change,
            "absolute_relative_change": abs(angle_relative_change),
        },
        "validation_loss": {
            "previous_mean": previous_loss,
            "final_mean": final_loss,
            "relative_change": loss_relative_change,
            "absolute_relative_change": abs(loss_relative_change),
        },
        "relative_change_limit": OFFICIAL200_TAIL_RELATIVE_CHANGE_LIMIT,
        "manuscript_limitation_required": not plateau_observed,
        "hard_stopping_boundary_reached": True,
        "additional_training_authorized": False,
        "phase4_authorized": False,
    }


def validate_authoritative_checkpoint(
    state: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    train_samples: Sequence[Any],
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("official-200 checkpoint is not a mapping")
    expected_keys = {
        "schema_version",
        "checkpoint_protocol",
        "signature",
        "status",
        "epoch",
        "best_epoch",
        "best_angle",
        "current_model_state",
        "best_model_state",
        "optimizer_state",
        "scaler_state",
        "history",
        "training_elapsed_seconds",
        "environment",
    }
    _require_equal(
        set(state),
        expected_keys,
        label="official-200 checkpoint schema",
    )
    _require_equal(
        int(state.get("schema_version", -1)),
        OFFICIAL200_SCHEMA_VERSION,
        label="official-200 checkpoint schema version",
    )
    _require_equal(
        state.get("checkpoint_protocol"),
        OFFICIAL200_CHECKPOINT_PROTOCOL,
        label="official-200 checkpoint protocol",
    )
    _require_equal(
        state.get("signature"),
        signature,
        label="official-200 checkpoint signature",
    )
    epoch = int(state.get("epoch", -1))
    if not 1 <= epoch <= OFFICIAL200_EPOCHS:
        raise ValueError("official-200 checkpoint epoch is invalid")
    expected_status = (
        "complete" if epoch == OFFICIAL200_EPOCHS else "running"
    )
    _require_equal(
        state.get("status"),
        expected_status,
        label="official-200 checkpoint status",
    )
    history = state.get("history")
    if not isinstance(history, Sequence):
        raise ValueError("official-200 checkpoint history is invalid")
    history_health = validate_official200_history(
        train_samples,
        history,
        through_epoch=epoch,
        signature=signature,
    )
    angles = [
        float(row["validation"]["angle_mae_degrees"]) for row in history
    ]
    expected_best_index = int(np.argmin(np.asarray(angles)))
    expected_best_epoch = expected_best_index + 1
    expected_best_angle = angles[expected_best_index]
    _require_equal(
        int(state.get("best_epoch", -1)),
        expected_best_epoch,
        label="official-200 checkpoint best epoch",
    )
    if not math.isclose(
        float(state.get("best_angle", math.nan)),
        expected_best_angle,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("official-200 checkpoint best angle drifted")
    elapsed = float(state.get("training_elapsed_seconds", math.nan))
    if (
        not math.isfinite(elapsed)
        or elapsed < 0.0
        or elapsed != float(history[-1]["training_elapsed_seconds"])
    ):
        raise ValueError("official-200 checkpoint elapsed time drifted")
    _require_equal(
        state.get("environment"),
        signature["runtime_environment"],
        label="official-200 checkpoint environment",
    )
    current_health = strict_model_state_health(
        model,
        state.get("current_model_state") or {},
        label="official-200 current model state",
    )
    best_health = strict_model_state_health(
        model,
        state.get("best_model_state") or {},
        label="official-200 best model state",
    )
    optimizer_health = validate_adam_optimizer_state(
        model,
        state.get("optimizer_state") or {},
        expected_step=int(history_health["cumulative_optimizer_steps"]),
        expected_learning_rate=official200_learning_rate(epoch),
        label="official-200 authoritative",
    )
    scaler_health = validate_scaler_state(
        state.get("scaler_state") or {},
        label="official-200 authoritative",
    )
    _require_equal(
        state.get("scaler_state"),
        history_health["terminal_scaler_state"],
        label="official-200 checkpoint scaler/history",
    )
    return {
        "epoch": epoch,
        "status": expected_status,
        "best_epoch": expected_best_epoch,
        "best_angle": expected_best_angle,
        "history": history_health,
        "current_model": current_health,
        "best_model": best_health,
        "optimizer": optimizer_health,
        "scaler": scaler_health,
    }


__all__ = [
    "OFFICIAL200_PROTOCOL",
    "OFFICIAL200_PREFLIGHT_PROTOCOL",
    "OFFICIAL200_CHECKPOINT_PROTOCOL",
    "OFFICIAL200_VERIFICATION_PROTOCOL",
    "OFFICIAL200_COHORT_PROTOCOL",
    "OFFICIAL200_DETERMINISM_PROTOCOL",
    "OFFICIAL200_SCHEMA_VERSION",
    "OFFICIAL200_EPOCHS",
    "OFFICIAL200_MILESTONES",
    "OFFICIAL200_LR_SEGMENTS",
    "OFFICIAL200_FORMAL_SEEDS",
    "OFFICIAL200_BATCH_SIZE",
    "OFFICIAL200_WORKERS",
    "OFFICIAL200_IMAGE_SIZE",
    "OFFICIAL200_VALIDATION_FRACTION",
    "OFFICIAL200_SCALE_FACTOR",
    "OFFICIAL200_ROTATION_FACTOR",
    "OFFICIAL200_MAX_SKIPPED_STEP_RATE",
    "OFFICIAL200_STOPPING_POLICY",
    "OFFICIAL200_SCOPE",
    "OFFICIAL200_DETERMINISM_POLICY",
    "OFFICIAL200_TRAIN_METRIC_KEYS",
    "OFFICIAL200_VALIDATION_METRIC_KEYS",
    "OFFICIAL200_HISTORY_ROW_KEYS",
    "VDN_TRAINABLE_PARAMETER_TENSORS",
    "apply_phase2_determinism_policy",
    "build_phase2_adam_optimizer",
    "current_runtime_environment",
    "normalize_content_inventory_identity",
    "validate_live_adam_optimizer",
    "validate_scaler_state",
    "validate_scaler_transition",
    "canonical_json_sha256",
    "assert_train_only_path",
    "assert_syncg_train_manifest_path",
    "validate_determinism_report_payload",
    "validate_authorized_runtime_environment",
    "require_formal_seed",
    "official200_learning_rate",
    "official200_vector_weight",
    "official200_epoch_seed",
    "official200_initial_scaler_state",
    "official200_source_hashes",
    "model_state_sha256",
    "expected_sample_order_sha256",
    "full_run_skip_budget",
    "build_official200_signature",
    "validate_official200_history",
    "tail_diagnostics",
    "validate_authoritative_checkpoint",
]
