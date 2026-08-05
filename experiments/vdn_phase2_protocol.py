"""Frozen protocol helpers for the formal VDN convergence continuation.

This module deliberately contains no test-set or field-data interfaces.  It
validates an audited 100-epoch SyncG-train parent, defines the deterministic
phase-2 schedule, and supplies strict state checks shared by the trainer and
its independent verifier.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.freeze_vdn_phase2_syncg_train_inventory import (
    DEFAULT_OUTPUT as DEFAULT_PHASE2_CONTENT_INVENTORY,
    INVENTORY_PROTOCOL,
    INVENTORY_VERIFICATION_PROTOCOL,
    SYNCG_TRAIN_EXPECTED_ROWS,
)
from experiments.train_vdn_syncg import (
    TRAIN_SAMPLE_ORDER_SHA256_PROTOCOL,
    sample_order_sha256,
)
from experiments.vdn_baseline import (
    VDN_PROTOCOL,
    VDN_PINNED_COMMIT,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    sha256_file,
    sha256_source_file,
)


PHASE2_PROTOCOL = "vdn_syncg_deterministic_phase2_continuation_v2"
PHASE2_VERIFICATION_PROTOCOL = "formal_vdn_phase2_verification_v2"
PHASE2_CONVERGENCE_PROTOCOL = "vdn_phase2_validation_convergence_gate_v3"
PHASE2_CHECKPOINT_PROTOCOL = (
    "atomic_authoritative_last_with_embedded_current_and_best_v2"
)
PHASE2_SCHEMA_VERSION = 2
PARENT_EPOCHS = 100
PHASE2_START_EPOCH = 101
PHASE2_END_EPOCH = 150
PHASE2_VECTOR_WEIGHT = 1.0
PHASE2_SOURCE_HASH_PROTOCOL = "utf8_source_newlines_lf_v1"
PHASE2_CONTENT_INVENTORY_PROTOCOL = INVENTORY_PROTOCOL
PHASE2_CONTENT_INVENTORY_VERIFICATION_PROTOCOL = (
    INVENTORY_VERIFICATION_PROTOCOL
)
PHASE2_CONTENT_INVENTORY_PATH = DEFAULT_PHASE2_CONTENT_INVENTORY
PHASE2_CONTENT_INVENTORY_ROWS = SYNCG_TRAIN_EXPECTED_ROWS
PHASE2_DETERMINISM_JOURNAL_PROTOCOL = (
    "vdn_phase2_determinism_authorization_journal_binding_v1"
)
VDN_TRAINABLE_PARAMETER_TENSORS = 73
FORMAL_WORKERS = 4
PHASE2_HISTORY_ROW_KEYS = frozenset(
    {
        "epoch",
        "epoch_seed",
        "learning_rate",
        "train",
        "validation",
        "best",
        "phase_elapsed_seconds",
        "determinism_authorization",
        "determinism_policy",
    }
)
PHASE2_TRAIN_METRIC_KEYS = frozenset(
    {
        "loss",
        "heatmap_loss",
        "vector_loss",
        "samples",
        "vector_weight",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "sample_order_sha256",
        "scaler_start_state",
        "scaler_skipped_batch_indices",
        "scaler_end_state",
    }
)
PHASE2_VALIDATION_METRIC_KEYS = frozenset(
    {
        "loss",
        "heatmap_loss",
        "vector_loss",
        "samples",
        "valid_directions",
        "direction_coverage",
        "angle_mae_degrees",
        "angle_median_degrees",
        "angle_acc_1deg",
        "angle_acc_3deg",
        "angle_acc_5deg",
        "mean_heatmap_peak",
    }
)

FORMAL_PHASE2_SEEDS = {
    20260720: 21260723,
    20260721: 21260724,
    20260722: 21260725,
}

PHASE2_LR_SEGMENTS = (
    {"start_epoch": 101, "end_epoch": 120, "learning_rate": 1e-5},
    {"start_epoch": 121, "end_epoch": 150, "learning_rate": 1e-6},
)

EPOCH_SEED_MULTIPLIER = 1009
CONVERGENCE_FINAL_WINDOW = 10
CONVERGENCE_BOUNDARY_WINDOW = 5
CONVERGENCE_RELATIVE_CHANGE_LIMIT = 0.01
CONVERGENCE_PARENT_RETENTION_RELATIVE_LIMIT = 0.01

ADAM_PARAM_GROUP_KEYS = frozenset(
    {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "initial_lr",
        "params",
    }
)
PHASE2_ADAM_PARAM_GROUP_POLICY = {
    "schema_keys": sorted(ADAM_PARAM_GROUP_KEYS),
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
    "maximize": False,
    "foreach": None,
    "capturable": False,
    "differentiable": False,
    "fused": None,
    "decoupled_weight_decay": False,
    "initial_lr": 0.001,
    "params_layout": (
        "single_group_model_parameters_in_declared_order_zero_based_ids_v1"
    ),
}
PHASE2_ADAM_STATE_POLICY = {
    "state_schema": ["step", "exp_avg", "exp_avg_sq"],
    "step_scalar_dtype": "torch.float32",
    "step_scalar_device": "cpu",
    "moment_dtype": "equal_to_parameter_dtype",
    "moment_device": "equal_to_parameter_device",
    "exp_avg_sq_nonnegative": True,
}
PHASE2_AMP_SCALER_POLICY = {
    "initial_scale": 512.0,
    "growth_factor": 2.0,
    "backoff_factor": 0.5,
    "growth_interval": 2000,
    "successful_step_accounting": (
        "attempts_success_skips_and_batch_index_transition_replay_v2"
    ),
}
PHASE2_MAX_SKIPPED_STEP_RATE = 0.0005

PHASE2_DETERMINISM_POLICY = {
    "torch_deterministic_algorithms": True,
    "torch_deterministic_warn_only": False,
    "cublas_workspace_config": ":4096:8",
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
    "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
    "float32_matmul_precision": "highest",
}


@dataclass(frozen=True)
class ParentLineage:
    run_dir: Path
    summary_path: Path
    last_path: Path
    best_path: Path
    verification_path: Path
    summary: dict[str, Any]
    verification: dict[str, Any]
    last_checkpoint: dict[str, Any]
    best_checkpoint: dict[str, Any]
    train_samples: list[Any]
    validation_samples: list[Any]

    @property
    def seed(self) -> int:
        return int(self.summary["signature"]["seed"])

    def artifact_hashes(self) -> dict[str, str]:
        return {
            "summary_sha256": sha256_file(self.summary_path),
            "last_checkpoint_sha256": sha256_file(self.last_path),
            "best_checkpoint_sha256": sha256_file(self.best_path),
            "verification_sha256": sha256_file(self.verification_path),
        }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def formal_phase2_seed(parent_seed: int) -> int:
    try:
        return FORMAL_PHASE2_SEEDS[int(parent_seed)]
    except KeyError as exc:
        raise ValueError(
            f"parent seed {parent_seed} is not a frozen formal VDN seed"
        ) from exc


def phase2_learning_rate(epoch: int) -> float:
    epoch = int(epoch)
    for segment in PHASE2_LR_SEGMENTS:
        if int(segment["start_epoch"]) <= epoch <= int(segment["end_epoch"]):
            return float(segment["learning_rate"])
    raise ValueError(
        f"phase-2 epoch must be in [{PHASE2_START_EPOCH}, "
        f"{PHASE2_END_EPOCH}], got {epoch}"
    )


def phase2_epoch_seed(parent_seed: int, epoch: int) -> int:
    phase_seed = formal_phase2_seed(parent_seed)
    epoch = int(epoch)
    if not PHASE2_START_EPOCH <= epoch <= PHASE2_END_EPOCH:
        raise ValueError(f"invalid phase-2 epoch {epoch}")
    return phase_seed + (epoch - PHASE2_START_EPOCH) * EPOCH_SEED_MULTIPLIER


def expected_optimizer_steps(train_samples: int, batch_size: int) -> int:
    train_samples = int(train_samples)
    batch_size = int(batch_size)
    if train_samples <= 0 or batch_size <= 0:
        raise ValueError("train samples and batch size must be positive")
    return math.ceil(train_samples / batch_size)


def normalize_content_inventory_identity(
    verification: dict[str, Any],
    *,
    inventory_tool_source: Path,
    manifest: Path,
) -> dict[str, Any]:
    """Validate and canonicalize one fresh train-content verification result."""

    if not isinstance(verification, dict):
        raise ValueError("VDN phase-2 content inventory verification is invalid")
    expected_keys = {
        "protocol",
        "verified",
        "content_rehashed",
        "report_path",
        "inventory_report_sha256",
        "canonical_inventory_sha256",
        "canonical_report_payload_sha256",
        "manifest_sha256",
        "manifest_protocol_sha256",
        "vdn_protocol_document_sha256",
        "vdn_protocol_source_sha256",
        "rows",
    }
    _require_equal(
        set(verification),
        expected_keys,
        label="phase content inventory verification schema",
    )
    _require_equal(
        verification.get("protocol"),
        PHASE2_CONTENT_INVENTORY_VERIFICATION_PROTOCOL,
        label="phase content inventory verification protocol",
    )
    _require_equal(
        verification.get("verified"),
        True,
        label="phase content inventory verification status",
    )
    _require_equal(
        verification.get("content_rehashed"),
        True,
        label="phase content inventory fresh-rehash status",
    )
    _require_equal(
        int(verification.get("rows", -1)),
        PHASE2_CONTENT_INVENTORY_ROWS,
        label="phase content inventory rows",
    )
    manifest = Path(manifest)
    manifest_protocol = manifest.with_name(manifest.name + ".protocol.json")
    _require_equal(
        verification.get("manifest_sha256"),
        sha256_file(manifest),
        label="phase content inventory manifest SHA-256",
    )
    _require_equal(
        verification.get("manifest_protocol_sha256"),
        sha256_file(manifest_protocol),
        label="phase content inventory manifest protocol SHA-256",
    )

    digest_keys = (
        "inventory_report_sha256",
        "canonical_inventory_sha256",
        "canonical_report_payload_sha256",
        "manifest_sha256",
        "manifest_protocol_sha256",
        "vdn_protocol_document_sha256",
        "vdn_protocol_source_sha256",
    )
    for key in digest_keys:
        digest = verification.get(key)
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}",
            digest,
        ) is None:
            raise ValueError(
                f"phase content inventory {key} is not a lowercase SHA-256"
            )
    report_path = verification.get("report_path")
    if (
        not isinstance(report_path, str)
        or not report_path
        or Path(report_path).is_absolute()
        or ".." in Path(report_path).parts
    ):
        raise ValueError("phase content inventory report path is invalid")
    return {
        "inventory_protocol": PHASE2_CONTENT_INVENTORY_PROTOCOL,
        "verification_protocol": (
            PHASE2_CONTENT_INVENTORY_VERIFICATION_PROTOCOL
        ),
        "report_path": report_path,
        "report_sha256": verification["inventory_report_sha256"],
        "canonical_inventory_sha256": verification[
            "canonical_inventory_sha256"
        ],
        "canonical_report_payload_sha256": verification[
            "canonical_report_payload_sha256"
        ],
        "inventory_tool_source_sha256": sha256_source_file(
            Path(inventory_tool_source)
        ),
        "manifest_sha256": verification["manifest_sha256"],
        "manifest_protocol_sha256": verification[
            "manifest_protocol_sha256"
        ],
        "vdn_protocol_document_sha256": verification[
            "vdn_protocol_document_sha256"
        ],
        "vdn_protocol_source_sha256": verification[
            "vdn_protocol_source_sha256"
        ],
        "rows": PHASE2_CONTENT_INVENTORY_ROWS,
        "fresh_content_rehashed": True,
    }


def apply_phase2_determinism_policy() -> dict[str, Any]:
    """Apply and return the exact deterministic runtime policy."""

    expected_workspace = str(PHASE2_DETERMINISM_POLICY["cublas_workspace_config"])
    current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current_workspace not in (None, expected_workspace):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen VDN phase-2 policy"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for attribute in (
        "allow_fp16_reduced_precision_reduction",
        "allow_bf16_reduced_precision_reduction",
    ):
        if not hasattr(torch.backends.cuda.matmul, attribute):
            raise RuntimeError(
                f"torch.backends.cuda.matmul.{attribute} is unavailable"
            )
        setattr(torch.backends.cuda.matmul, attribute, False)
    torch.set_float32_matmul_precision("highest")
    snapshot = phase2_determinism_snapshot()
    _require_equal(
        snapshot,
        PHASE2_DETERMINISM_POLICY,
        label="VDN phase-2 deterministic runtime policy",
    )
    return snapshot


def phase2_determinism_snapshot() -> dict[str, Any]:
    return {
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def current_runtime_environment(device: torch.device) -> dict[str, Any]:
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("formal VDN phase 2 requires a CUDA runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_index": int(index),
        "gpu": torch.cuda.get_device_name(index),
        "gpu_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "amp": True,
        "determinism": phase2_determinism_snapshot(),
    }


def expected_phase2_sample_order_sha256(
    train_samples: Sequence[Any],
    *,
    batch_size: int,
    epoch_seed: int,
) -> str:
    """Reproduce the DataLoader sampler order without opening an image."""

    sample_ids = [str(sample.sample_id) for sample in train_samples]
    if not sample_ids:
        raise ValueError("VDN phase-2 train split is empty")
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


def load_parent_lineage(
    run_dir: Path,
    *,
    manifest: Path,
    load_checkpoints: bool = True,
) -> ParentLineage:
    """Validate immutable parent identities without rerunning its old verifier."""

    run_dir = Path(run_dir).resolve()
    manifest = Path(manifest).resolve()
    paths = {
        "summary": run_dir / "summary.json",
        "last": run_dir / "last.pt",
        "best": run_dir / "best.pt",
        "verification": run_dir / "verification.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete VDN parent run: {missing}")

    summary = _read_json(paths["summary"])
    verification = _read_json(paths["verification"])
    if summary.get("status") != "complete":
        raise ValueError("VDN parent summary is not complete")
    if verification.get("verified") is not True:
        raise ValueError("VDN parent verification is not successful")
    signature = summary.get("signature") or {}
    _require_equal(
        signature.get("protocol"),
        VDN_PROTOCOL,
        label="parent training protocol",
    )
    _require_equal(
        signature.get("vdn_source_commit"),
        VDN_PINNED_COMMIT,
        label="parent VDN commit",
    )
    _require_equal(
        int(signature.get("epochs", -1)),
        PARENT_EPOCHS,
        label="parent epoch budget",
    )
    formal_phase2_seed(int(signature.get("seed", -1)))
    _require_equal(
        float(signature.get("weight_decay", math.nan)),
        0.0,
        label="parent weight decay",
    )
    _require_equal(
        int(signature.get("batch_size", -1)),
        8,
        label="parent batch size",
    )
    _require_equal(
        signature.get("mixed_precision"),
        True,
        label="parent mixed-precision policy",
    )
    _require_equal(
        verification.get("summary_sha256"),
        sha256_file(paths["summary"]),
        label="parent summary SHA-256",
    )
    _require_equal(
        verification.get("last_checkpoint_sha256"),
        sha256_file(paths["last"]),
        label="parent last checkpoint SHA-256",
    )
    _require_equal(
        verification.get("best_checkpoint_sha256"),
        sha256_file(paths["best"]),
        label="parent best checkpoint SHA-256",
    )
    _require_equal(
        int(verification.get("epochs", -1)),
        PARENT_EPOCHS,
        label="verified parent epochs",
    )
    _require_equal(
        int(verification.get("best_epoch", -1)),
        int(summary.get("best_epoch", -2)),
        label="verified parent best epoch",
    )

    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    _require_equal(
        sha256_file(manifest),
        signature.get("manifest_sha256"),
        label="SyncG train manifest SHA-256",
    )
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    _require_equal(
        sha256_file(protocol_path),
        signature.get("manifest_protocol_sha256"),
        label="SyncG train manifest protocol SHA-256",
    )
    samples, manifest_protocol = load_syncg_manifest(
        manifest,
        expected_split="train",
    )
    _require_equal(
        manifest_protocol.get("dataset"),
        "SyncG",
        label="parent manifest dataset",
    )
    _require_equal(
        manifest_protocol.get("split"),
        "train",
        label="parent manifest split",
    )
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
    for key, expected in split_expectations.items():
        _require_equal(signature.get(key), expected, label=f"parent {key}")
    if {sample.group_id for sample in train_samples} & {
        sample.group_id for sample in validation_samples
    }:
        raise ValueError("parent train/validation groups overlap")

    if load_checkpoints:
        last_checkpoint = torch.load(
            paths["last"],
            map_location="cpu",
            weights_only=False,
        )
        best_checkpoint = torch.load(
            paths["best"],
            map_location="cpu",
            weights_only=False,
        )
        if last_checkpoint.get("signature") != signature:
            raise ValueError("parent last checkpoint signature mismatch")
        if best_checkpoint.get("signature") != signature:
            raise ValueError("parent best checkpoint signature mismatch")
        _require_equal(
            int(last_checkpoint.get("epoch", -1)),
            PARENT_EPOCHS,
            label="parent last checkpoint epoch",
        )
        _require_equal(
            int(best_checkpoint.get("epoch", -1)),
            int(summary["best_epoch"]),
            label="parent best checkpoint epoch",
        )
        required_last = {
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "history",
            "best_epoch",
            "best_angle",
        }
        missing_last = sorted(required_last - set(last_checkpoint))
        if missing_last:
            raise ValueError(
                f"parent last checkpoint lacks continuation state: {missing_last}"
            )
        if "model_state" not in best_checkpoint:
            raise ValueError("parent best checkpoint lacks model state")
    else:
        last_checkpoint = {}
        best_checkpoint = {}

    return ParentLineage(
        run_dir=run_dir,
        summary_path=paths["summary"],
        last_path=paths["last"],
        best_path=paths["best"],
        verification_path=paths["verification"],
        summary=summary,
        verification=verification,
        last_checkpoint=last_checkpoint,
        best_checkpoint=best_checkpoint,
        train_samples=train_samples,
        validation_samples=validation_samples,
    )


def strict_model_state_health(
    model: torch.nn.Module,
    state: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{label} is empty")
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"{label} tensor {name} is not tensor-valued")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} tensor {name} is non-finite")
    model.load_state_dict(state, strict=True)
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise ValueError(f"{label} state schema is not exact")
    return {
        "tensor_count": len(state),
        "parameter_tensor_count": len(list(model.parameters())),
        "non_finite_tensors": 0,
    }


def _optimizer_scalar_step(value: Any, *, label: str) -> int:
    if (
        not torch.is_tensor(value)
        or value.ndim != 0
        or value.numel() != 1
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{label} Adam step tensor policy changed")
    number = float(value.item())
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{label} Adam step is invalid")
    return int(number)


def build_phase2_adam_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
) -> torch.optim.Adam:
    """Construct Adam with every pinned PyTorch 2.11 param-group option."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        decoupled_weight_decay=False,
    )
    optimizer.param_groups[0]["initial_lr"] = float(
        PHASE2_ADAM_PARAM_GROUP_POLICY["initial_lr"]
    )
    return optimizer


def _validate_adam_param_group_options(
    group: dict[str, Any],
    *,
    expected_learning_rate: float,
    label: str,
) -> None:
    if not isinstance(group, dict) or set(group) != ADAM_PARAM_GROUP_KEYS:
        raise ValueError(f"{label} Adam parameter-group schema changed")
    learning_rate = group["lr"]
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) != float(expected_learning_rate)
    ):
        raise ValueError(f"{label} Adam learning rate is invalid")
    if type(group["betas"]) is not tuple or group["betas"] != (0.9, 0.999):
        raise ValueError(f"{label} Adam betas changed")
    if (
        isinstance(group["eps"], bool)
        or not isinstance(group["eps"], (int, float))
        or float(group["eps"]) != 1e-8
    ):
        raise ValueError(f"{label} Adam epsilon policy changed")
    if (
        isinstance(group["weight_decay"], bool)
        or not isinstance(group["weight_decay"], (int, float))
        or float(group["weight_decay"]) != 0.0
    ):
        raise ValueError(f"{label} Adam weight decay is not zero")
    if (
        isinstance(group["initial_lr"], bool)
        or not isinstance(group["initial_lr"], (int, float))
        or float(group["initial_lr"])
        != float(PHASE2_ADAM_PARAM_GROUP_POLICY["initial_lr"])
    ):
        raise ValueError(f"{label} Adam initial_lr policy changed")
    expected_exact = {
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    for field, expected in expected_exact.items():
        actual = group[field]
        if expected is None:
            valid = actual is None
        else:
            valid = type(actual) is bool and actual is expected
        if not valid:
            raise ValueError(
                f"{label} Adam parameter-group field {field} changed"
            )


def _validate_adam_parameter_states(
    parameters: Sequence[torch.nn.Parameter],
    states: Sequence[dict[str, Any]],
    *,
    expected_step: int,
    label: str,
) -> None:
    observed_steps: set[int] = set()
    for index, (parameter, state) in enumerate(zip(parameters, states)):
        if not isinstance(state, dict):
            raise ValueError(f"{label} Adam parameter {index} lacks state")
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"{label} Adam parameter {index} state schema changed")
        step = _optimizer_scalar_step(
            state["step"],
            label=f"{label} parameter {index}",
        )
        observed_steps.add(step)
        if not torch.isfinite(parameter).all():
            raise ValueError(f"{label} Adam parameter {index} is non-finite")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state[moment_name]
            if not torch.is_tensor(moment):
                raise ValueError(
                    f"{label} Adam parameter {index} {moment_name} is not a tensor"
                )
            if tuple(moment.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{label} Adam parameter {index} {moment_name} shape mismatch"
                )
            if moment.dtype != parameter.dtype:
                raise ValueError(
                    f"{label} Adam parameter {index} {moment_name} dtype mismatch"
                )
            if moment.device != parameter.device:
                raise ValueError(
                    f"{label} Adam parameter {index} {moment_name} device mismatch"
                )
            if not torch.isfinite(moment).all():
                raise ValueError(
                    f"{label} Adam parameter {index} {moment_name} is non-finite"
                )
            if moment_name == "exp_avg_sq" and torch.any(moment < 0):
                raise ValueError(
                    f"{label} Adam parameter {index} exp_avg_sq is negative"
                )
    if observed_steps != {int(expected_step)}:
        raise ValueError(
            f"{label} Adam steps are {sorted(observed_steps)}, "
            f"expected {expected_step}"
        )


def validate_adam_optimizer_state(
    model: torch.nn.Module,
    optimizer_state: dict[str, Any],
    *,
    expected_step: int,
    expected_learning_rate: float,
    expected_parameter_count: int = VDN_TRAINABLE_PARAMETER_TENSORS,
    label: str,
) -> dict[str, Any]:
    if (
        not isinstance(optimizer_state, dict)
        or set(optimizer_state) != {"state", "param_groups"}
    ):
        raise ValueError(f"{label} optimizer state schema changed")
    groups = optimizer_state["param_groups"]
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError(f"{label} must have exactly one Adam parameter group")
    group = groups[0]
    _validate_adam_param_group_options(
        group,
        expected_learning_rate=expected_learning_rate,
        label=label,
    )
    parameters = list(model.parameters())
    if len(parameters) != int(expected_parameter_count):
        raise ValueError(
            f"{label} Adam parameter count is {len(parameters)}, "
            f"expected {expected_parameter_count}"
        )
    serialized_parameters = group["params"]
    expected_serialized_parameters = list(range(int(expected_parameter_count)))
    if (
        type(serialized_parameters) is not list
        or any(type(value) is not int for value in serialized_parameters)
        or serialized_parameters != expected_serialized_parameters
    ):
        raise ValueError(f"{label} Adam parameter layout/order changed")
    state = optimizer_state["state"]
    if not isinstance(state, dict):
        raise ValueError(f"{label} Adam state table is invalid")
    if (
        any(type(value) is not int for value in state)
        or set(state) != set(expected_serialized_parameters)
    ):
        raise ValueError(f"{label} Adam state parameter layout changed")
    if len(state) != int(expected_parameter_count):
        raise ValueError(
            f"{label} Adam state count is {len(state)}, "
            f"expected {expected_parameter_count}"
        )
    _validate_adam_parameter_states(
        parameters,
        [state[index] for index in expected_serialized_parameters],
        expected_step=expected_step,
        label=label,
    )
    return {
        "parameter_count": len(parameters),
        "state_count": len(state),
        "step": int(expected_step),
        "learning_rate": float(group["lr"]),
        "param_group_policy": dict(PHASE2_ADAM_PARAM_GROUP_POLICY),
        "state_policy": dict(PHASE2_ADAM_STATE_POLICY),
        "parameter_layout_exact": True,
        "moments_finite": True,
    }


def validate_live_adam_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected_step: int,
    expected_learning_rate: float,
    expected_parameter_count: int = VDN_TRAINABLE_PARAMETER_TENSORS,
    label: str,
) -> dict[str, Any]:
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError(f"{label} optimizer is not the pinned Adam class")
    if len(optimizer.param_groups) != 1:
        raise ValueError(f"{label} must have exactly one Adam parameter group")
    group = optimizer.param_groups[0]
    _validate_adam_param_group_options(
        group,
        expected_learning_rate=expected_learning_rate,
        label=label,
    )
    expected_parameters = list(model.parameters())
    live_parameters = group["params"]
    if (
        type(live_parameters) is not list
        or len(expected_parameters) != int(expected_parameter_count)
        or len(live_parameters) != int(expected_parameter_count)
        or any(
            actual is not expected
            for actual, expected in zip(live_parameters, expected_parameters)
        )
    ):
        raise ValueError(f"{label} Adam live parameter layout/order changed")
    if len(optimizer.state) != int(expected_parameter_count) or any(
        parameter not in optimizer.state for parameter in expected_parameters
    ):
        raise ValueError(
            f"{label} Adam state does not cover the exact model parameters"
        )
    _validate_adam_parameter_states(
        expected_parameters,
        [optimizer.state[parameter] for parameter in expected_parameters],
        expected_step=expected_step,
        label=label,
    )
    return {
        "parameter_count": len(expected_parameters),
        "state_count": len(optimizer.state),
        "step": int(expected_step),
        "learning_rate": float(group["lr"]),
        "param_group_policy": dict(PHASE2_ADAM_PARAM_GROUP_POLICY),
        "state_policy": dict(PHASE2_ADAM_STATE_POLICY),
        "parameter_layout_exact": True,
        "moments_finite": True,
    }


def validate_scaler_state(
    scaler_state: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    required = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    if not isinstance(scaler_state, dict) or set(scaler_state) != required:
        raise ValueError(f"{label} AMP scaler state schema is invalid")
    numeric: dict[str, float] = {}
    for key in ("scale", "growth_factor", "backoff_factor"):
        value = scaler_state[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{label} AMP scaler contains invalid numeric values")
        numeric[key] = float(value)
    if numeric["growth_factor"] != float(
        PHASE2_AMP_SCALER_POLICY["growth_factor"]
    ):
        raise ValueError(f"{label} AMP scaler growth_factor policy changed")
    if numeric["backoff_factor"] != float(
        PHASE2_AMP_SCALER_POLICY["backoff_factor"]
    ):
        raise ValueError(f"{label} AMP scaler backoff_factor policy changed")
    if type(scaler_state["growth_interval"]) is not int:
        raise ValueError(f"{label} AMP scaler growth_interval is invalid")
    if type(scaler_state["_growth_tracker"]) is not int:
        raise ValueError(f"{label} AMP scaler growth tracker is invalid")
    growth_interval = scaler_state["growth_interval"]
    growth_tracker = scaler_state["_growth_tracker"]
    if growth_interval != int(
        PHASE2_AMP_SCALER_POLICY["growth_interval"]
    ):
        raise ValueError(f"{label} AMP scaler growth_interval policy changed")
    if not 0 <= growth_tracker < growth_interval:
        raise ValueError(f"{label} AMP scaler growth tracker is out of range")
    if math.frexp(numeric["scale"])[0] != 0.5:
        raise ValueError(f"{label} AMP scaler scale is not a positive power of two")
    return {
        **numeric,
        "growth_interval": growth_interval,
        "growth_tracker": growth_tracker,
        "policy": dict(PHASE2_AMP_SCALER_POLICY),
    }


def validate_scaler_transition(
    start_state: dict[str, Any],
    end_state: dict[str, Any],
    skipped_batch_indices: list[int],
    *,
    attempted_steps: int,
    label: str,
) -> dict[str, Any]:
    if type(attempted_steps) is not int or attempted_steps <= 0:
        raise ValueError(f"{label} AMP scaler attempted-step count is invalid")
    if type(skipped_batch_indices) is not list or any(
        type(index) is not int for index in skipped_batch_indices
    ):
        raise ValueError(f"{label} AMP scaler skipped indices are invalid")
    if skipped_batch_indices != sorted(set(skipped_batch_indices)):
        raise ValueError(
            f"{label} AMP scaler skipped indices are not strictly increasing"
        )
    if any(
        index < 0 or index >= attempted_steps
        for index in skipped_batch_indices
    ):
        raise ValueError(f"{label} AMP scaler skipped index is out of range")

    start = validate_scaler_state(start_state, label=f"{label} start")
    end = validate_scaler_state(end_state, label=f"{label} end")
    scale = float(start["scale"])
    growth_tracker = int(start["growth_tracker"])
    skipped = set(skipped_batch_indices)
    growth_factor = float(PHASE2_AMP_SCALER_POLICY["growth_factor"])
    backoff_factor = float(PHASE2_AMP_SCALER_POLICY["backoff_factor"])
    growth_interval = int(PHASE2_AMP_SCALER_POLICY["growth_interval"])
    for batch_index in range(attempted_steps):
        if batch_index in skipped:
            scale *= backoff_factor
            growth_tracker = 0
        else:
            growth_tracker += 1
            if growth_tracker == growth_interval:
                scale *= growth_factor
                growth_tracker = 0
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{label} AMP scaler replay became non-finite")
    if scale != float(end["scale"]) or growth_tracker != int(
        end["growth_tracker"]
    ):
        raise ValueError(f"{label} AMP scaler transition replay mismatch")
    return {
        "attempted_steps": attempted_steps,
        "successful_steps": attempted_steps - len(skipped_batch_indices),
        "skipped_steps": len(skipped_batch_indices),
        "skipped_batch_indices": list(skipped_batch_indices),
        "start": start,
        "end": end,
        "replay_exact": True,
    }


def nested_state_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(nested_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(nested_state_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def validate_parent_training_state(
    lineage: ParentLineage,
    model: torch.nn.Module,
) -> dict[str, Any]:
    last = lineage.last_checkpoint
    best = lineage.best_checkpoint
    summary = lineage.summary
    history = summary.get("history") or []
    _require_equal(
        [int(row.get("epoch", -1)) for row in history],
        list(range(1, PARENT_EPOCHS + 1)),
        label="parent history epochs",
    )
    _require_equal(last.get("history"), history, label="parent last history")
    _require_equal(
        int(last.get("best_epoch", -1)),
        int(summary["best_epoch"]),
        label="parent last best epoch",
    )
    if not math.isclose(
        float(last.get("best_angle", math.nan)),
        float(summary["best_validation_angle_mae_degrees"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("parent last best angle mismatch")
    if not math.isclose(
        float(best.get("best_angle", math.nan)),
        float(summary["best_validation_angle_mae_degrees"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("parent best checkpoint angle mismatch")

    current_model_health = strict_model_state_health(
        model,
        last["model_state"],
        label="parent last model state",
    )
    best_model_health = strict_model_state_health(
        model,
        best["model_state"],
        label="parent best model state",
    )
    expected_batches = expected_optimizer_steps(
        int(summary["signature"]["train_samples"]),
        int(summary["signature"]["batch_size"]),
    )
    attempted_steps = 0
    successful_steps = 0
    skipped_steps = 0
    for row in history:
        done = row["train"]["optimizer_steps"]
        skipped = row["train"]["skipped_optimizer_steps"]
        if (
            type(done) is not int
            or type(skipped) is not int
            or done < 0
            or skipped < 0
        ):
            raise ValueError("parent optimizer-step accounting is invalid")
        if done + skipped != expected_batches:
            raise ValueError("parent optimizer-step accounting is invalid")
        attempted_steps += expected_batches
        successful_steps += done
        skipped_steps += skipped
    skipped_step_rate = skipped_steps / attempted_steps
    if skipped_step_rate > PHASE2_MAX_SKIPPED_STEP_RATE:
        raise ValueError("parent AMP skipped-step rate exceeds the frozen cap")
    optimizer_health = validate_adam_optimizer_state(
        model,
        last["optimizer_state"],
        expected_step=successful_steps,
        expected_learning_rate=1e-5,
        label="parent last",
    )
    scaler_health = validate_scaler_state(
        last["scaler_state"],
        label="parent last",
    )
    scheduler = last["scheduler_state"]
    if not isinstance(scheduler, dict):
        raise ValueError("parent scheduler state is invalid")
    _require_equal(
        int(scheduler.get("last_epoch", -1)),
        PARENT_EPOCHS,
        label="parent scheduler epoch",
    )
    last_lr = scheduler.get("_last_lr")
    if (
        not isinstance(last_lr, list)
        or len(last_lr) != 1
        or not math.isclose(
            float(last_lr[0]),
            1e-5,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("parent scheduler terminal learning rate is invalid")
    return {
        "attempted_optimizer_steps": attempted_steps,
        "optimizer_step": successful_steps,
        "skipped_optimizer_steps": skipped_steps,
        "skipped_optimizer_step_rate": skipped_step_rate,
        "max_skipped_optimizer_step_rate": PHASE2_MAX_SKIPPED_STEP_RATE,
        "current_model": current_model_health,
        "best_model": best_model_health,
        "optimizer": optimizer_health,
        "scaler": scaler_health,
        "scheduler_last_epoch": PARENT_EPOCHS,
        "scheduler_last_lr": 1e-5,
    }


def build_phase2_signature(
    lineage: ParentLineage,
    *,
    manifest: Path,
    vdn_source_commit: str,
    source_sha256: dict[str, str],
    runtime_environment: dict[str, Any],
    parent_optimizer_step: int,
    content_inventory_identity: dict[str, Any],
    determinism_authorization: dict[str, Any],
) -> dict[str, Any]:
    parent_signature = lineage.summary["signature"]
    protocol_path = Path(manifest).with_name(Path(manifest).name + ".protocol.json")
    phase_attempts_per_epoch = expected_optimizer_steps(
        int(parent_signature["train_samples"]),
        int(parent_signature["batch_size"]),
    )
    full_phase_attempted_steps = phase_attempts_per_epoch * (
        PHASE2_END_EPOCH - PHASE2_START_EPOCH + 1
    )
    full_phase_max_skipped_steps = math.floor(
        full_phase_attempted_steps * PHASE2_MAX_SKIPPED_STEP_RATE
    )
    journal_authorization = {
        "protocol": PHASE2_DETERMINISM_JOURNAL_PROTOCOL,
        "report_protocol": determinism_authorization["report_protocol"],
        "report_schema_version": determinism_authorization[
            "report_schema_version"
        ],
        "report_sha256": determinism_authorization["report_sha256"],
        "canonical_report_sha256": determinism_authorization[
            "canonical_report_sha256"
        ],
    }
    return {
        "protocol": PHASE2_PROTOCOL,
        "schema_version": PHASE2_SCHEMA_VERSION,
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
        "parent": {
            "run_dir": str(lineage.run_dir),
            "protocol": parent_signature["protocol"],
            "seed": lineage.seed,
            "epochs": int(parent_signature["epochs"]),
            "best_epoch": int(lineage.summary["best_epoch"]),
            "best_validation_angle_mae_degrees": float(
                lineage.summary["best_validation_angle_mae_degrees"]
            ),
            "optimizer_step": int(parent_optimizer_step),
            **lineage.artifact_hashes(),
        },
        "phase_seed": formal_phase2_seed(lineage.seed),
        "epoch_seed_protocol": (
            "phase_seed_plus_(epoch_minus_101)_times_1009_v1"
        ),
        "phase_start_epoch": PHASE2_START_EPOCH,
        "phase_end_epoch": PHASE2_END_EPOCH,
        "epochs": PHASE2_END_EPOCH,
        "learning_rate_segments": [dict(value) for value in PHASE2_LR_SEGMENTS],
        "vector_weight": PHASE2_VECTOR_WEIGHT,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "optimizer_param_group_policy": dict(
            PHASE2_ADAM_PARAM_GROUP_POLICY
        ),
        "optimizer_state_policy": dict(PHASE2_ADAM_STATE_POLICY),
        "optimizer_state_origin": "strict_audited_parent_last_checkpoint",
        "amp_scaler_policy": dict(PHASE2_AMP_SCALER_POLICY),
        "scaler_state_origin": "strict_audited_parent_last_checkpoint",
        "scheduler_policy": "explicit_epoch_learning_rate_no_scheduler_v1",
        "optimizer_step_invariant": (
            "attempts_equal_batches_adam_steps_equal_cumulative_success_v2"
        ),
        "phase_attempts_per_epoch": phase_attempts_per_epoch,
        "full_phase_attempted_steps": full_phase_attempted_steps,
        "max_skipped_step_rate": PHASE2_MAX_SKIPPED_STEP_RATE,
        "full_phase_max_skipped_steps": full_phase_max_skipped_steps,
        "scaler_trace_protocol": (
            "epoch_start_end_and_zero_based_skipped_batch_replay_v1"
        ),
        "trainable_parameter_tensors": VDN_TRAINABLE_PARAMETER_TENSORS,
        "batch_size": int(parent_signature["batch_size"]),
        "workers": FORMAL_WORKERS,
        "persistent_workers": False,
        "image_size": int(parent_signature["image_size"]),
        "heatmap_size": int(parent_signature["heatmap_size"]),
        "validation_fraction": float(parent_signature["validation_fraction"]),
        "scale_factor": float(parent_signature["scale_factor"]),
        "rotation_factor": float(parent_signature["rotation_factor"]),
        "mixed_precision": True,
        "vdn_source_commit": str(vdn_source_commit),
        "manifest_sha256": sha256_file(Path(manifest)),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "content_inventory": dict(content_inventory_identity),
        "determinism_authorization": dict(determinism_authorization),
        "determinism_authorization_journal": journal_authorization,
        "train_sample_ids_sha256": parent_signature[
            "train_sample_ids_sha256"
        ],
        "validation_sample_ids_sha256": parent_signature[
            "validation_sample_ids_sha256"
        ],
        "train_samples": int(parent_signature["train_samples"]),
        "validation_samples": int(parent_signature["validation_samples"]),
        "sample_order_sha256_protocol": TRAIN_SAMPLE_ORDER_SHA256_PROTOCOL,
        "determinism": dict(PHASE2_DETERMINISM_POLICY),
        "runtime_environment": dict(runtime_environment),
        "source_sha256": dict(source_sha256),
    }


def validate_phase_history(
    lineage: ParentLineage,
    history: Sequence[dict[str, Any]],
    *,
    through_epoch: int,
    expected_signature: dict[str, Any],
) -> dict[str, Any]:
    through_epoch = int(through_epoch)
    expected_epochs = (
        []
        if through_epoch == PARENT_EPOCHS
        else list(range(PHASE2_START_EPOCH, through_epoch + 1))
    )
    _require_equal(
        [int(row.get("epoch", -1)) for row in history],
        expected_epochs,
        label="phase history epochs",
    )
    signature = lineage.summary["signature"]
    batch_size = int(signature["batch_size"])
    expected_attempts = expected_optimizer_steps(
        int(signature["train_samples"]),
        batch_size,
    )
    full_phase_attempts = expected_attempts * (
        PHASE2_END_EPOCH - PHASE2_START_EPOCH + 1
    )
    full_phase_skip_budget = math.floor(
        full_phase_attempts * PHASE2_MAX_SKIPPED_STEP_RATE
    )
    cumulative_successful_steps = 0
    cumulative_skipped_steps = 0
    cumulative_attempted_steps = 0
    previous_scaler_state = lineage.last_checkpoint["scaler_state"]
    running_best_angle = float(
        lineage.summary["best_validation_angle_mae_degrees"]
    )
    for row in history:
        _require_equal(
            set(row),
            PHASE2_HISTORY_ROW_KEYS,
            label="phase history row schema",
        )
        epoch = int(row["epoch"])
        _require_equal(
            row.get("determinism_authorization"),
            expected_signature["determinism_authorization_journal"],
            label=f"phase epoch {epoch} determinism authorization",
        )
        _require_equal(
            row.get("determinism_policy"),
            expected_signature["determinism"],
            label=f"phase epoch {epoch} determinism policy",
        )
        epoch_seed = phase2_epoch_seed(lineage.seed, epoch)
        _require_equal(
            int(row.get("epoch_seed", -1)),
            epoch_seed,
            label=f"phase epoch {epoch} seed",
        )
        if not math.isclose(
            float(row.get("learning_rate", math.nan)),
            phase2_learning_rate(epoch),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"phase epoch {epoch} learning rate mismatch")
        train = row.get("train") or {}
        validation = row.get("validation") or {}
        _require_equal(
            set(train),
            PHASE2_TRAIN_METRIC_KEYS,
            label=f"phase epoch {epoch} train metric schema",
        )
        _require_equal(
            set(validation),
            PHASE2_VALIDATION_METRIC_KEYS,
            label=f"phase epoch {epoch} validation metric schema",
        )
        for field in PHASE2_TRAIN_METRIC_KEYS - {
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
                    f"phase epoch {epoch} train.{field} is not finite numeric"
                )
        for field in PHASE2_VALIDATION_METRIC_KEYS:
            value = validation[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"phase epoch {epoch} validation.{field} "
                    "is not finite numeric"
                )
        for field in (
            "samples",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            if type(train[field]) is not int:
                raise ValueError(
                    f"phase epoch {epoch} train.{field} is not an integer"
                )
        for field in ("samples", "valid_directions"):
            if type(validation[field]) is not int:
                raise ValueError(
                    f"phase epoch {epoch} validation.{field} is not an integer"
                )
        successful_steps = train["optimizer_steps"]
        skipped_steps = train["skipped_optimizer_steps"]
        if (
            successful_steps < 0
            or skipped_steps < 0
            or successful_steps + skipped_steps != expected_attempts
        ):
            raise ValueError(
                f"phase epoch {epoch} optimizer attempt accounting is invalid"
            )
        _require_equal(
            train["scaler_start_state"],
            previous_scaler_state,
            label=f"phase epoch {epoch} scaler continuity",
        )
        scaler_transition = validate_scaler_transition(
            train["scaler_start_state"],
            train["scaler_end_state"],
            train["scaler_skipped_batch_indices"],
            attempted_steps=expected_attempts,
            label=f"phase epoch {epoch}",
        )
        _require_equal(
            scaler_transition["successful_steps"],
            successful_steps,
            label=f"phase epoch {epoch} successful optimizer steps",
        )
        _require_equal(
            scaler_transition["skipped_steps"],
            skipped_steps,
            label=f"phase epoch {epoch} skipped optimizer steps",
        )
        previous_scaler_state = train["scaler_end_state"]
        cumulative_attempted_steps += expected_attempts
        cumulative_successful_steps += successful_steps
        cumulative_skipped_steps += skipped_steps
        if cumulative_skipped_steps > full_phase_skip_budget:
            raise ValueError(
                "phase cumulative skipped optimizer steps exceed the "
                "full-run frozen budget"
            )
        _require_equal(
            int(train.get("samples", -1)),
            int(signature["train_samples"]),
            label=f"phase epoch {epoch} train samples",
        )
        _require_equal(
            int(validation.get("samples", -1)),
            int(signature["validation_samples"]),
            label=f"phase epoch {epoch} validation samples",
        )
        if not math.isclose(
            float(train.get("vector_weight", math.nan)),
            PHASE2_VECTOR_WEIGHT,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"phase epoch {epoch} vector weight mismatch")
        expected_order = expected_phase2_sample_order_sha256(
            lineage.train_samples,
            batch_size=batch_size,
            epoch_seed=epoch_seed,
        )
        _require_equal(
            train.get("sample_order_sha256"),
            expected_order,
            label=f"phase epoch {epoch} sample order",
        )
        valid_directions = int(validation["valid_directions"])
        validation_samples = int(validation["samples"])
        if not 0 < valid_directions <= validation_samples:
            raise ValueError(
                f"phase epoch {epoch} valid-direction count is invalid"
            )
        expected_coverage = valid_directions / validation_samples
        if not math.isclose(
            float(validation["direction_coverage"]),
            expected_coverage,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"phase epoch {epoch} direction coverage is inconsistent"
            )
        for field in ("angle_acc_1deg", "angle_acc_3deg", "angle_acc_5deg"):
            if not 0.0 <= float(validation[field]) <= 1.0:
                raise ValueError(
                    f"phase epoch {epoch} validation.{field} is invalid"
                )
        if not (
            float(validation["angle_acc_1deg"])
            <= float(validation["angle_acc_3deg"])
            <= float(validation["angle_acc_5deg"])
        ):
            raise ValueError(
                f"phase epoch {epoch} angular accuracies are inconsistent"
            )
        current_angle = float(validation["angle_mae_degrees"])
        improved = current_angle < running_best_angle
        _require_equal(
            row.get("best"),
            improved,
            label=f"phase epoch {epoch} best flag",
        )
        if improved:
            running_best_angle = current_angle
        elapsed = float(row.get("phase_elapsed_seconds", math.nan))
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError(
                f"phase epoch {epoch} contains invalid elapsed time"
            )
    elapsed_values = [
        float(row["phase_elapsed_seconds"]) for row in history
    ]
    if any(
        current < previous
        for previous, current in zip(elapsed_values, elapsed_values[1:])
    ):
        raise ValueError("phase elapsed time is not monotonic")
    return {
        "epochs": len(expected_epochs),
        "attempted_optimizer_steps_per_epoch": expected_attempts,
        "cumulative_attempted_optimizer_steps": cumulative_attempted_steps,
        "cumulative_optimizer_steps": cumulative_successful_steps,
        "cumulative_skipped_optimizer_steps": cumulative_skipped_steps,
        "full_phase_attempted_optimizer_steps": full_phase_attempts,
        "full_phase_max_skipped_optimizer_steps": full_phase_skip_budget,
        "terminal_skipped_optimizer_step_rate": (
            cumulative_skipped_steps / cumulative_attempted_steps
            if cumulative_attempted_steps
            else 0.0
        ),
        "terminal_scaler_state": previous_scaler_state,
        "sample_orders_verified": len(expected_epochs),
    }


def _combined_best(
    parent_history: Sequence[dict[str, Any]],
    phase_history: Sequence[dict[str, Any]],
) -> tuple[int, float]:
    candidates = [
        (int(row["epoch"]), float(row["validation"]["angle_mae_degrees"]))
        for row in [*parent_history, *phase_history]
    ]
    if not candidates:
        raise ValueError("VDN history has no validation angle")
    return min(candidates, key=lambda value: (value[1], value[0]))


def validate_authoritative_checkpoint(
    state: dict[str, Any],
    *,
    lineage: ParentLineage,
    model: torch.nn.Module,
    expected_signature: dict[str, Any],
    parent_health: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("VDN phase-2 authoritative checkpoint is not a dictionary")
    expected_keys = {
        "schema_version",
        "checkpoint_protocol",
        "signature",
        "status",
        "epoch",
        "best_epoch",
        "best_angle",
        "best_origin",
        "current_model_state",
        "best_model_state",
        "optimizer_state",
        "scaler_state",
        "history",
        "phase_elapsed_seconds",
        "environment",
    }
    _require_equal(
        set(state),
        expected_keys,
        label="phase authoritative checkpoint schema",
    )
    _require_equal(
        int(state.get("schema_version", -1)),
        PHASE2_SCHEMA_VERSION,
        label="phase authoritative schema",
    )
    _require_equal(
        state.get("checkpoint_protocol"),
        PHASE2_CHECKPOINT_PROTOCOL,
        label="phase authoritative checkpoint protocol",
    )
    _require_equal(
        state.get("signature"),
        expected_signature,
        label="phase authoritative signature",
    )
    epoch = int(state.get("epoch", -1))
    if not PARENT_EPOCHS <= epoch <= PHASE2_END_EPOCH:
        raise ValueError("phase authoritative epoch is outside [100, 150]")
    expected_status = (
        "bootstrap"
        if epoch == PARENT_EPOCHS
        else ("complete" if epoch == PHASE2_END_EPOCH else "running")
    )
    _require_equal(
        state.get("status"),
        expected_status,
        label="phase authoritative status",
    )
    history = state.get("history") or []
    history_health = validate_phase_history(
        lineage,
        history,
        through_epoch=epoch,
        expected_signature=expected_signature,
    )
    elapsed = float(state.get("phase_elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("phase authoritative elapsed time is invalid")
    expected_elapsed = (
        0.0
        if epoch == PARENT_EPOCHS
        else float(history[-1]["phase_elapsed_seconds"])
    )
    if elapsed != expected_elapsed:
        raise ValueError("phase authoritative elapsed time mismatch")
    expected_best_epoch, expected_best_angle = _combined_best(
        lineage.summary["history"],
        history,
    )
    _require_equal(
        int(state.get("best_epoch", -1)),
        expected_best_epoch,
        label="phase authoritative best epoch",
    )
    if not math.isclose(
        float(state.get("best_angle", math.nan)),
        expected_best_angle,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("phase authoritative best angle mismatch")
    expected_origin = (
        "parent_best_checkpoint"
        if expected_best_epoch <= PARENT_EPOCHS
        else "phase2_validation"
    )
    _require_equal(
        state.get("best_origin"),
        expected_origin,
        label="phase authoritative best origin",
    )
    current_health = strict_model_state_health(
        model,
        state.get("current_model_state") or {},
        label="phase current model state",
    )
    best_health = strict_model_state_health(
        model,
        state.get("best_model_state") or {},
        label="phase best model state",
    )
    if expected_best_epoch <= PARENT_EPOCHS and not nested_state_equal(
        state["best_model_state"],
        lineage.best_checkpoint["model_state"],
    ):
        raise ValueError("phase embedded parent-best model state changed")

    expected_step = int(parent_health["optimizer_step"]) + int(
        history_health["cumulative_optimizer_steps"]
    )
    expected_lr = (
        1e-5 if epoch == PARENT_EPOCHS else phase2_learning_rate(epoch)
    )
    optimizer_health = validate_adam_optimizer_state(
        model,
        state.get("optimizer_state") or {},
        expected_step=expected_step,
        expected_learning_rate=expected_lr,
        label="phase authoritative",
    )
    scaler_health = validate_scaler_state(
        state.get("scaler_state") or {},
        label="phase authoritative",
    )
    _require_equal(
        state.get("scaler_state"),
        history_health["terminal_scaler_state"],
        label="phase authoritative scaler/history terminal state",
    )
    _require_equal(
        state.get("environment"),
        expected_signature["runtime_environment"],
        label="phase authoritative environment",
    )
    if epoch == PARENT_EPOCHS:
        if not nested_state_equal(
            state["current_model_state"],
            lineage.last_checkpoint["model_state"],
        ):
            raise ValueError("phase bootstrap current model differs from parent last")
        if not nested_state_equal(
            state["optimizer_state"],
            lineage.last_checkpoint["optimizer_state"],
        ):
            raise ValueError("phase bootstrap optimizer differs from parent last")
        if not nested_state_equal(
            state["scaler_state"],
            lineage.last_checkpoint["scaler_state"],
        ):
            raise ValueError("phase bootstrap scaler differs from parent last")
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


def convergence_diagnostics(
    parent_history: Sequence[dict[str, Any]],
    phase_history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen validation-only gate to a complete 150-epoch history."""

    combined = [*parent_history, *phase_history]
    expected_epochs = list(range(1, PHASE2_END_EPOCH + 1))
    actual_epochs = [int(row.get("epoch", -1)) for row in combined]
    if actual_epochs != expected_epochs:
        raise ValueError("combined VDN history must cover epochs 1..150 exactly")
    if len(phase_history) < 2 * CONVERGENCE_FINAL_WINDOW:
        raise ValueError("phase history is too short for the convergence gate")

    angles = np.asarray(
        [
            float(row["validation"]["angle_mae_degrees"])
            for row in combined
        ],
        dtype=np.float64,
    )
    losses = np.asarray(
        [float(row["validation"]["loss"]) for row in combined],
        dtype=np.float64,
    )
    if not np.isfinite(angles).all() or not np.isfinite(losses).all():
        raise ValueError("convergence history contains non-finite validation values")

    best_index = int(np.argmin(angles))
    best_epoch = best_index + 1
    previous_slice = slice(
        PHASE2_END_EPOCH - 2 * CONVERGENCE_FINAL_WINDOW,
        PHASE2_END_EPOCH - CONVERGENCE_FINAL_WINDOW,
    )
    parent_terminal_slice = slice(
        PARENT_EPOCHS - CONVERGENCE_FINAL_WINDOW,
        PARENT_EPOCHS,
    )
    final_slice = slice(
        PHASE2_END_EPOCH - CONVERGENCE_FINAL_WINDOW,
        PHASE2_END_EPOCH,
    )
    parent_terminal_angle = float(np.mean(angles[parent_terminal_slice]))
    parent_terminal_loss = float(np.mean(losses[parent_terminal_slice]))
    previous_angle = float(np.mean(angles[previous_slice]))
    final_angle = float(np.mean(angles[final_slice]))
    previous_loss = float(np.mean(losses[previous_slice]))
    final_loss = float(np.mean(losses[final_slice]))
    angle_relative_change = (final_angle - previous_angle) / max(
        abs(previous_angle),
        np.finfo(np.float64).eps,
    )
    loss_relative_change = (final_loss - previous_loss) / max(
        abs(previous_loss),
        np.finfo(np.float64).eps,
    )
    angle_parent_retention_relative_change = (
        final_angle - parent_terminal_angle
    ) / max(
        abs(parent_terminal_angle),
        np.finfo(np.float64).eps,
    )
    loss_parent_retention_relative_change = (
        final_loss - parent_terminal_loss
    ) / max(
        abs(parent_terminal_loss),
        np.finfo(np.float64).eps,
    )
    boundary_start = PHASE2_END_EPOCH - CONVERGENCE_BOUNDARY_WINDOW + 1
    retention_tolerance = (
        CONVERGENCE_PARENT_RETENTION_RELATIVE_LIMIT + 1e-12
    )
    checks = {
        "best_not_in_final_boundary_window": best_epoch < boundary_start,
        "angle_absolute_relative_change_below_limit": (
            abs(angle_relative_change) < CONVERGENCE_RELATIVE_CHANGE_LIMIT
        ),
        "loss_absolute_relative_change_below_limit": (
            abs(loss_relative_change) < CONVERGENCE_RELATIVE_CHANGE_LIMIT
        ),
        "angle_parent_terminal_retention_within_limit": (
            angle_parent_retention_relative_change <= retention_tolerance
        ),
        "loss_parent_terminal_retention_within_limit": (
            loss_parent_retention_relative_change <= retention_tolerance
        ),
    }
    return {
        "protocol": PHASE2_CONVERGENCE_PROTOCOL,
        "passed": all(checks.values()),
        "checks": checks,
        "best_epoch": best_epoch,
        "phase2_improved_over_parent": best_epoch > PARENT_EPOCHS,
        "best_validation_angle_mae_degrees": float(angles[best_index]),
        "boundary_window": [
            boundary_start,
            PHASE2_END_EPOCH,
        ],
        "rolling_windows": {
            "parent_terminal": [
                PARENT_EPOCHS - CONVERGENCE_FINAL_WINDOW + 1,
                PARENT_EPOCHS,
            ],
            "previous": [
                PHASE2_END_EPOCH - 2 * CONVERGENCE_FINAL_WINDOW + 1,
                PHASE2_END_EPOCH - CONVERGENCE_FINAL_WINDOW,
            ],
            "final": [
                PHASE2_END_EPOCH - CONVERGENCE_FINAL_WINDOW + 1,
                PHASE2_END_EPOCH,
            ],
        },
        "angle_mae": {
            "parent_terminal_mean": parent_terminal_angle,
            "previous_mean": previous_angle,
            "final_mean": final_angle,
            "relative_change": angle_relative_change,
            "absolute_relative_change": abs(angle_relative_change),
            "final_vs_parent_terminal_relative_change": (
                angle_parent_retention_relative_change
            ),
            "final_vs_parent_terminal_regression": max(
                angle_parent_retention_relative_change,
                0.0,
            ),
        },
        "validation_loss": {
            "parent_terminal_mean": parent_terminal_loss,
            "previous_mean": previous_loss,
            "final_mean": final_loss,
            "relative_change": loss_relative_change,
            "absolute_relative_change": abs(loss_relative_change),
            "final_vs_parent_terminal_relative_change": (
                loss_parent_retention_relative_change
            ),
            "final_vs_parent_terminal_regression": max(
                loss_parent_retention_relative_change,
                0.0,
            ),
        },
        "absolute_relative_change_limit": CONVERGENCE_RELATIVE_CHANGE_LIMIT,
        "parent_retention_relative_limit": (
            CONVERGENCE_PARENT_RETENTION_RELATIVE_LIMIT
        ),
    }
