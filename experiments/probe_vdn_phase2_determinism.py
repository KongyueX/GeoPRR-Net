"""Fail-closed two-run determinism gate for formal VDN Phase-2 startup.

The probe is deliberately restricted to the pinned SyncG official-train
manifest and its parent-seed-20260720 grouped validation split.  Replicates A
and B run in separate Python processes.  Each process independently reloads
the audited parent, rebuilds the model, Adam optimizer, AMP scaler and both
DataLoaders, then executes the complete epoch 101 and validation pass.

The final immutable JSON compares actual sample-order hashes, metrics, and
semantic SHA-256 summaries of initial/final model, optimizer, scaler and RNG
state.  Passing this probe authorizes only formal Phase-2 training startup.  It
never authorizes or opens SyncG test, public, RPM, Pointer, field, sealed or
confirmatory inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments import freeze_vdn_phase2_syncg_train_inventory as content_inventory
from experiments.compare_pct_determinism_runs import (
    first_semantic_mismatch,
    semantic_sha256,
)
from experiments.train_projective_circular_transport_syncg import (
    PINNED_SYNCG_TRAIN_INPUT_INVENTORY_SHA256,
    _training_input_inventory_sha256,
    _validate_pinned_training_input_inventory,
    _validate_syncg_train_scope,
)
from experiments.train_vdn_syncg import (
    _train_epoch,
    _validate,
    sample_order_sha256,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SyncGVDNDataset,
    build_vdn_model,
    load_syncg_manifest,
    seed_worker,
    set_random_seed,
    sha256_file,
    sha256_source_file,
    verify_vdn_source,
)
from experiments.vdn_phase2_protocol import (
    FORMAL_WORKERS,
    PHASE2_END_EPOCH,
    PHASE2_MAX_SKIPPED_STEP_RATE,
    PHASE2_SOURCE_HASH_PROTOCOL,
    PHASE2_START_EPOCH,
    PHASE2_VECTOR_WEIGHT,
    build_phase2_adam_optimizer,
    formal_phase2_seed,
    load_parent_lineage,
    phase2_epoch_seed,
    phase2_learning_rate,
    validate_live_adam_optimizer,
    validate_parent_training_state,
    validate_scaler_transition,
)


VDN_PHASE2_DETERMINISM_PROBE_PROTOCOL = (
    "vdn_phase2_full_epoch_determinism_probe_v2"
)
VDN_PHASE2_DETERMINISM_WORKER_PROTOCOL = (
    "vdn_phase2_full_epoch_determinism_worker_v2"
)
VDN_PHASE2_DETERMINISM_SCHEMA_VERSION = 2
VDN_PHASE2_AUTHORIZATION_BINDING_PROTOCOL = (
    "vdn_phase2_determinism_authorization_binding_v1"
)
FIXED_PARENT_SEED = 20260720
FIXED_EPOCH = 101
FIXED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FIXED_PARENT_RUN = (
    PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg" / "seed_20260720"
)
FIXED_MANIFEST = PROJECT_DIR / "artifacts" / "manifests" / "syncg_train.jsonl"
FIXED_VDN_SOURCE = PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"
FIXED_CONTENT_INVENTORY = (
    PROJECT_DIR / content_inventory.DEFAULT_OUTPUT
)
DEFAULT_DETERMINISM_REPORT = Path(
    "artifacts/protocols/vdn_phase2_determinism_probe_v2.json"
)
FIXED_DETERMINISM_REPORT = PROJECT_DIR / DEFAULT_DETERMINISM_REPORT
FIXED_VDN_PROTOCOL_DOCUMENT = (
    PROJECT_DIR / content_inventory.DEFAULT_VDN_PROTOCOL_DOCUMENT
)
FIXED_VDN_PROTOCOL_SOURCE = (
    PROJECT_DIR / content_inventory.DEFAULT_VDN_PROTOCOL_SOURCE
)
DEFAULT_INVENTORY_WORKERS = min(8, max(1, os.cpu_count() or 1))
FORBIDDEN_EVALUATION_SCOPES = (
    "SyncG test",
    "public",
    "RPM-10K",
    "Pointer-10K",
    "field",
    "sealed",
    "confirmatory",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_WORKER_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "replicate",
        "execution",
        "identity",
        "sample_order",
        "metrics",
        "state_summaries",
        "state_health",
    }
)
_PASSED_REPORT_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "status",
        "first_mismatch",
        "vdn_phase2_training_start_authorized",
        "test_evaluation_authorized",
        "authorization_scope",
        "fixed_probe",
        "scientific_identity",
        "components",
        "sample_order",
        "metrics",
        "state_summaries",
        "state_health",
        "worker_execution",
        "source_hash_protocol",
        "probe_source_sha256",
    }
)
_METRIC_KEYS = {
    "train": frozenset(
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
    ),
    "validation": frozenset(
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
    ),
}
_INVENTORY_VERIFICATION_KEYS = frozenset(
    {
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
)
_CONTENT_INVENTORY_IDENTITY_KEYS = _INVENTORY_VERIFICATION_KEYS | {
    "artifact_path",
    "inventory_tool_source_sha256",
    "fresh_rehash_workers",
}
_SCIENTIFIC_IDENTITY_KEYS = frozenset(
    {
        "scope",
        "forbidden_evaluation_scopes",
        "parent_seed",
        "phase_seed",
        "epoch",
        "epoch_seed",
        "batch_size",
        "workers",
        "learning_rate",
        "optimizer",
        "weight_decay",
        "mixed_precision",
        "vector_weight",
        "max_skipped_step_rate",
        "full_phase_max_skipped_steps",
        "scaler_trace_protocol",
        "parent_run",
        "parent_artifacts",
        "parent_signature_semantic_sha256",
        "manifest",
        "manifest_sha256",
        "manifest_protocol_sha256",
        "content_inventory",
        "legacy_content_inventory_crosscheck",
        "train_samples",
        "validation_samples",
        "train_groups",
        "validation_groups",
        "train_sample_ids_sha256",
        "validation_sample_ids_sha256",
        "vdn_source",
        "vdn_source_commit",
        "source_hash_protocol",
        "source_sha256",
        "environment",
    }
)
_RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "platform",
        "torch",
        "numpy",
        "opencv",
        "cuda_runtime",
        "cudnn",
        "gpu_name",
        "gpu_capability",
        "gpu_total_memory_bytes",
        "device",
        "cuda_visible_devices",
        "amp",
        "policy",
    }
)
_POLICY_EXPECTED = {
    "cublas_workspace_config": FIXED_CUBLAS_WORKSPACE_CONFIG,
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "pythonhashseed": str(formal_phase2_seed(FIXED_PARENT_SEED)),
    "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
    "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
}


class VDNPhase2DeterminismMismatch(ValueError):
    """First fail-closed mismatch between the two complete workers."""


class VDNPhase2WorkerFailure(RuntimeError):
    """A worker failed before producing an authenticated snapshot."""


class _RecordingLoader:
    """Record the IDs actually yielded while preserving DataLoader semantics."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.sample_ids: list[str] = []
        self._iterated = False

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self._iterated:
            raise RuntimeError("a determinism-probe loader may be iterated once")
        self._iterated = True
        for batch in self.loader:
            if not isinstance(batch, Sequence) or len(batch) != 5:
                raise ValueError("VDN batch schema differs from the frozen loader")
            identifiers = batch[4]
            if not isinstance(identifiers, Sequence) or isinstance(
                identifiers,
                (str, bytes),
            ):
                raise ValueError("VDN batch lacks its sample-ID sequence")
            normalized = [str(value) for value in identifiers]
            if any(not value for value in normalized):
                raise ValueError("VDN batch contains an empty sample ID")
            self.sample_ids.extend(normalized)
            yield batch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parent-run", type=Path, default=FIXED_PARENT_RUN)
    parser.add_argument("--manifest", type=Path, default=FIXED_MANIFEST)
    parser.add_argument("--vdn-source", type=Path, default=FIXED_VDN_SOURCE)
    parser.add_argument(
        "--content-inventory",
        type=Path,
        default=FIXED_CONTENT_INVENTORY,
    )
    parser.add_argument(
        "--inventory-workers",
        type=int,
        default=DEFAULT_INVENTORY_WORKERS,
    )
    parser.add_argument("--device", default="cuda")
    worker = parser.add_argument_group("internal worker arguments")
    worker.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    worker.add_argument(
        "--replicate",
        choices=("a", "b"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a worker JSON file: {resolved}")
    value = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"worker JSON root is not an object: {resolved}")
    return value


def _atomic_write_json_no_clobber(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    """Publish JSON atomically without ever replacing an existing path."""

    output = path.resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite probe artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite probe artifact: {output}"
            ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_ids_sha256(values: Iterable[str]) -> str:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("sample-order IDs must be non-empty strings")
        normalized.append(value)
    return sample_order_sha256(normalized)


def _current_source_hashes(vdn_source: Path) -> dict[str, str]:
    return {
        "probe": sha256_source_file(Path(__file__).resolve()),
        "phase2_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_phase2.py"
        ),
        "phase2_protocol": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_phase2_protocol.py"
        ),
        "phase2_supervisor": sha256_source_file(
            PROJECT_DIR / "experiments" / "run_vdn_phase2.ps1"
        ),
        "base_trainer": sha256_source_file(
            PROJECT_DIR / "experiments" / "train_vdn_syncg.py"
        ),
        "vdn_adapter": sha256_source_file(
            PROJECT_DIR / "experiments" / "vdn_baseline.py"
        ),
        "vdn_model": sha256_source_file(
            vdn_source.resolve() / "libs" / "models" / "vdn_model.py"
        ),
        "semantic_hasher": sha256_source_file(
            PROJECT_DIR / "experiments" / "compare_pct_determinism_runs.py"
        ),
        "inventory_scope_guard": sha256_source_file(
            PROJECT_DIR
            / "experiments"
            / "train_projective_circular_transport_syncg.py"
        ),
        "content_inventory_tool": sha256_source_file(
            Path(content_inventory.__file__).resolve()
        ),
    }


def _verify_content_inventory_fresh(
    content_inventory_path: Path,
    *,
    manifest: Path,
    workers: int,
) -> dict[str, Any]:
    """Freshly rehash the formal inventory and bind its immutable identities."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
    ):
        raise ValueError("inventory workers must be a positive integer")
    report_path = content_inventory_path.resolve(strict=True)
    if not report_path.is_file():
        raise ValueError(
            f"content inventory is not a regular file: {report_path}"
        )
    try:
        expected_report_path = report_path.relative_to(
            PROJECT_DIR.resolve(strict=True)
        ).as_posix()
    except ValueError as exc:
        raise ValueError(
            "formal content-inventory artifact must be inside the project root"
        ) from exc
    manifest = manifest.resolve(strict=True)
    raw_sha_before = content_inventory.sha256_file_raw(report_path)
    verification = content_inventory.verify_inventory_artifact(
        report_path,
        project_root=PROJECT_DIR,
        manifest=manifest,
        manifest_protocol=manifest.with_name(
            manifest.name + ".protocol.json"
        ),
        vdn_protocol_document=FIXED_VDN_PROTOCOL_DOCUMENT,
        vdn_protocol_source=FIXED_VDN_PROTOCOL_SOURCE,
        inventory_tool_source=Path(content_inventory.__file__).resolve(),
        expected_rows=content_inventory.SYNCG_TRAIN_EXPECTED_ROWS,
        formal_identity=True,
        workers=int(workers),
    )
    if not isinstance(verification, Mapping) or set(verification) != (
        _INVENTORY_VERIFICATION_KEYS
    ):
        raise ValueError("formal content-inventory verification schema drifted")
    raw_sha_after = content_inventory.sha256_file_raw(report_path)
    if (
        raw_sha_after != raw_sha_before
        or verification.get("inventory_report_sha256") != raw_sha_after
    ):
        raise ValueError("formal content-inventory artifact changed during rehash")
    if verification.get("report_path") != expected_report_path:
        raise ValueError(
            "formal content-inventory report path does not identify the "
            "verified artifact"
        )
    if (
        verification.get("protocol")
        != content_inventory.INVENTORY_VERIFICATION_PROTOCOL
        or verification.get("verified") is not True
        or verification.get("content_rehashed") is not True
        or isinstance(verification.get("rows"), bool)
        or not isinstance(verification.get("rows"), int)
        or verification.get("rows")
        != content_inventory.SYNCG_TRAIN_EXPECTED_ROWS
    ):
        raise ValueError("formal content-inventory rehash did not verify")
    for field in (
        "inventory_report_sha256",
        "canonical_inventory_sha256",
        "canonical_report_payload_sha256",
        "manifest_sha256",
        "manifest_protocol_sha256",
        "vdn_protocol_document_sha256",
        "vdn_protocol_source_sha256",
    ):
        _require_sha256(
            verification.get(field),
            f"content_inventory.{field}",
        )
    expected_tool_source = content_inventory.sha256_source_file(
        Path(content_inventory.__file__).resolve()
    )
    return {
        **dict(verification),
        "artifact_path": str(report_path),
        "inventory_tool_source_sha256": expected_tool_source,
        "fresh_rehash_workers": int(workers),
    }


def validate_determinism_policy(policy: Mapping[str, Any]) -> None:
    for field, expected in _POLICY_EXPECTED.items():
        if policy.get(field) != expected:
            raise ValueError(
                f"strict deterministic policy mismatch for {field}: "
                f"{policy.get(field)!r} != {expected!r}"
            )


def _configure_strict_determinism(device_text: str) -> tuple[torch.device, dict[str, Any]]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before deterministic policy")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        FIXED_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before worker startup"
        )
    if os.environ.get("PYTHONHASHSEED") != str(
        formal_phase2_seed(FIXED_PARENT_SEED)
    ):
        raise RuntimeError("PYTHONHASHSEED differs from the frozen phase seed")

    requested_device = torch.device(device_text)
    if requested_device.type != "cuda":
        raise ValueError("formal VDN Phase-2 determinism probe requires CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    optional_policy: dict[str, bool] = {}
    for attribute, label in (
        (
            "allow_fp16_reduced_precision_reduction",
            "cuda_matmul_allow_fp16_reduced_precision_reduction",
        ),
        (
            "allow_bf16_reduced_precision_reduction",
            "cuda_matmul_allow_bf16_reduced_precision_reduction",
        ),
    ):
        if not hasattr(torch.backends.cuda.matmul, attribute):
            raise RuntimeError(
                f"torch.backends.cuda.matmul.{attribute} is unavailable"
            )
        setattr(torch.backends.cuda.matmul, attribute, False)
        optional_policy[label] = bool(
            getattr(torch.backends.cuda.matmul, attribute)
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the formal VDN probe")
    device = torch.device(
        "cuda",
        0 if requested_device.index is None else requested_device.index,
    )
    torch.cuda.set_device(device)
    policy: dict[str, Any] = {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        **optional_policy,
    }
    validate_determinism_policy(policy)
    return device, policy


def _runtime_environment(
    device: torch.device,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_capability": [properties.major, properties.minor],
        "gpu_total_memory_bytes": int(properties.total_memory),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp": True,
        "policy": dict(policy),
    }


def _to_cpu_semantic(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            key: _to_cpu_semantic(child) for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_to_cpu_semantic(child) for child in value)
    if isinstance(value, list):
        return [_to_cpu_semantic(child) for child in value]
    return value


def _state_health(value: Any) -> dict[str, int]:
    tensors = 0
    elements = 0
    floating_elements = 0
    nonfinite_floating_elements = 0

    def visit(child: Any) -> None:
        nonlocal tensors
        nonlocal elements
        nonlocal floating_elements
        nonlocal nonfinite_floating_elements
        if torch.is_tensor(child):
            tensors += 1
            elements += child.numel()
            if child.is_floating_point() or child.is_complex():
                floating_elements += child.numel()
                nonfinite_floating_elements += int(
                    (~torch.isfinite(child.detach())).sum().item()
                )
            return
        if isinstance(child, Mapping):
            for nested in child.values():
                visit(nested)
            return
        if isinstance(child, Sequence) and not isinstance(
            child,
            (str, bytes),
        ):
            for nested in child:
                visit(nested)

    visit(value)
    return {
        "tensors": tensors,
        "elements": elements,
        "floating_elements": floating_elements,
        "nonfinite_floating_elements": nonfinite_floating_elements,
    }


def _require_healthy_state(value: Any, field: str) -> dict[str, int]:
    health = _state_health(value)
    if health["tensors"] <= 0:
        raise ValueError(f"{field} contains no tensor state")
    if health["nonfinite_floating_elements"] != 0:
        raise ValueError(f"{field} contains non-finite tensor state")
    return health


def _rng_state(
    *,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "train_loader_generator": train_generator.get_state(),
        "validation_loader_generator": validation_generator.get_state(),
    }


def _state_summary(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
    validation_generator: torch.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    states = {
        "model": _to_cpu_semantic(model.state_dict()),
        "optimizer": _to_cpu_semantic(optimizer.state_dict()),
        "scaler": _to_cpu_semantic(scaler.state_dict()),
        "rng": _to_cpu_semantic(
            _rng_state(
                train_generator=train_generator,
                validation_generator=validation_generator,
            )
        ),
    }
    model_health = _require_healthy_state(states["model"], "model state")
    optimizer_health = _require_healthy_state(
        states["optimizer"],
        "optimizer state",
    )
    summaries = {
        name: {
            "semantic_sha256": semantic_sha256(state),
        }
        for name, state in states.items()
    }
    summaries["combined_semantic_sha256"] = _canonical_json_sha256(summaries)
    health = {
        "model": model_health,
        "optimizer": optimizer_health,
    }
    return summaries, health


def _validate_finite_metrics(
    metrics: Mapping[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    if kind not in _METRIC_KEYS or set(metrics) != _METRIC_KEYS[kind]:
        raise ValueError(f"{kind} metric schema differs from the frozen VDN API")
    normalized = dict(metrics)
    for field, value in normalized.items():
        if kind == "train" and field == "sample_order_sha256":
            _require_sha256(value, f"{kind}.{field}")
            continue
        if kind == "train" and field in {
            "scaler_start_state",
            "scaler_skipped_batch_indices",
            "scaler_end_state",
        }:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{kind}.{field} is not numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{kind}.{field} is non-finite")
    return normalized


def _order_report(
    actual_ids: Sequence[str],
    expected_ids: Sequence[str],
    *,
    require_exact_order: bool,
) -> dict[str, Any]:
    if (
        len(actual_ids) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
        or set(actual_ids) != set(expected_ids)
    ):
        raise ValueError("actual loader sample population differs from its split")
    if require_exact_order and list(actual_ids) != list(expected_ids):
        raise ValueError("validation loader order differs from manifest order")
    return {
        "samples": len(actual_ids),
        "unique_samples": len(set(actual_ids)),
        "actual_order_sha256": _ordered_ids_sha256(actual_ids),
        "expected_population_sha256": _ordered_ids_sha256(
            sorted(expected_ids)
        ),
        "exact_manifest_order_required": require_exact_order,
    }


def _worker_identity(
    *,
    lineage: Any,
    manifest: Path,
    vdn_source: Path,
    content_inventory_verification: Mapping[str, Any],
    legacy_inventory_sha256: str,
    device: torch.device,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    protocol_path = manifest.with_name(manifest.name + ".protocol.json")
    train_groups = {sample.group_id for sample in lineage.train_samples}
    validation_groups = {
        sample.group_id for sample in lineage.validation_samples
    }
    if train_groups & validation_groups:
        raise ValueError("VDN probe train/validation groups overlap")
    attempts_per_epoch = math.ceil(
        len(lineage.train_samples)
        / int(lineage.summary["signature"]["batch_size"])
    )
    full_phase_max_skipped_steps = math.floor(
        attempts_per_epoch
        * (PHASE2_END_EPOCH - PHASE2_START_EPOCH + 1)
        * PHASE2_MAX_SKIPPED_STEP_RATE
    )
    return {
        "scope": (
            "pinned SyncG official train grouped validation only; no "
            "test/public/RPM/Pointer/field/sealed/confirmatory input"
        ),
        "forbidden_evaluation_scopes": list(FORBIDDEN_EVALUATION_SCOPES),
        "parent_seed": FIXED_PARENT_SEED,
        "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
        "epoch": FIXED_EPOCH,
        "epoch_seed": phase2_epoch_seed(FIXED_PARENT_SEED, FIXED_EPOCH),
        "batch_size": int(lineage.summary["signature"]["batch_size"]),
        "workers": FORMAL_WORKERS,
        "learning_rate": phase2_learning_rate(FIXED_EPOCH),
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "mixed_precision": True,
        "vector_weight": PHASE2_VECTOR_WEIGHT,
        "max_skipped_step_rate": PHASE2_MAX_SKIPPED_STEP_RATE,
        "full_phase_max_skipped_steps": full_phase_max_skipped_steps,
        "scaler_trace_protocol": (
            "epoch_start_end_and_zero_based_skipped_batch_replay_v1"
        ),
        "parent_run": str(lineage.run_dir),
        "parent_artifacts": lineage.artifact_hashes(),
        "parent_signature_semantic_sha256": semantic_sha256(
            lineage.summary["signature"]
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(protocol_path),
        "content_inventory": dict(content_inventory_verification),
        "legacy_content_inventory_crosscheck": {
            "protocol": "pct_style_sample_image_annotation_sha256_v1",
            "sha256": legacy_inventory_sha256,
            "role": (
                "cross_evidence_only_not_a_substitute_for_the_formal_"
                "content_inventory_artifact"
            ),
        },
        "train_samples": len(lineage.train_samples),
        "validation_samples": len(lineage.validation_samples),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_sample_ids_sha256": _ordered_ids_sha256(
            sample.sample_id for sample in lineage.train_samples
        ),
        "validation_sample_ids_sha256": _ordered_ids_sha256(
            sample.sample_id for sample in lineage.validation_samples
        ),
        "vdn_source": str(vdn_source),
        "vdn_source_commit": verify_vdn_source(vdn_source),
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "source_sha256": _current_source_hashes(vdn_source),
        "environment": _runtime_environment(device, policy),
    }


def run_worker(
    *,
    replicate: str,
    parent_run: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory_path: Path,
    inventory_workers: int,
    device_text: str,
) -> dict[str, Any]:
    if replicate not in {"a", "b"}:
        raise ValueError("worker replicate must be 'a' or 'b'")
    device, policy = _configure_strict_determinism(device_text)
    parent_run = parent_run.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    vdn_source = vdn_source.resolve(strict=True)
    content_inventory_path = content_inventory_path.resolve(strict=True)

    print(
        f"vdn_probe_worker={replicate} "
        "stage=fresh_formal_content_inventory_rehash",
        flush=True,
    )
    formal_inventory_verification = _verify_content_inventory_fresh(
        content_inventory_path,
        manifest=manifest,
        workers=inventory_workers,
    )
    print(
        f"vdn_probe_worker={replicate} stage=legacy_inventory_crosscheck",
        flush=True,
    )
    all_samples, manifest_protocol = load_syncg_manifest(
        manifest,
        expected_split="train",
    )
    if (
        manifest_protocol.get("dataset") != "SyncG"
        or manifest_protocol.get("split") != "train"
    ):
        raise ValueError("VDN probe manifest is not SyncG official train")
    _validate_syncg_train_scope(manifest, all_samples)
    legacy_inventory_sha256 = _validate_pinned_training_input_inventory(
        _training_input_inventory_sha256(all_samples)
    )
    lineage = load_parent_lineage(
        parent_run,
        manifest=manifest,
        load_checkpoints=True,
    )
    if lineage.seed != FIXED_PARENT_SEED:
        raise ValueError("VDN determinism probe requires parent seed 20260720")

    print(f"vdn_probe_worker={replicate} stage=rebuild_state", flush=True)
    phase_seed = formal_phase2_seed(FIXED_PARENT_SEED)
    epoch_seed = phase2_epoch_seed(FIXED_PARENT_SEED, FIXED_EPOCH)
    parent_signature = lineage.summary["signature"]
    batch_size = int(parent_signature["batch_size"])
    set_random_seed(phase_seed)

    train_dataset = SyncGVDNDataset(
        lineage.train_samples,
        image_size=int(parent_signature["image_size"]),
        training=True,
        scale_factor=float(parent_signature["scale_factor"]),
        rotation_factor=float(parent_signature["rotation_factor"]),
    )
    validation_dataset = SyncGVDNDataset(
        lineage.validation_samples,
        image_size=int(parent_signature["image_size"]),
        training=False,
        scale_factor=float(parent_signature["scale_factor"]),
        rotation_factor=float(parent_signature["rotation_factor"]),
    )
    validation_generator = torch.Generator().manual_seed(phase_seed)
    validation_loader = _RecordingLoader(
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=FORMAL_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            generator=validation_generator,
        )
    )

    model = build_vdn_model(
        vdn_source,
        image_size=int(parent_signature["image_size"]),
        imagenet_pretrained=False,
    )
    parent_health = validate_parent_training_state(lineage, model)
    model.load_state_dict(lineage.last_checkpoint["model_state"])
    model.to(device)
    optimizer = build_phase2_adam_optimizer(
        model,
        learning_rate=phase2_learning_rate(FIXED_EPOCH),
    )
    optimizer.load_state_dict(lineage.last_checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = phase2_learning_rate(FIXED_EPOCH)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=True,
        init_scale=512.0,
    )
    scaler.load_state_dict(lineage.last_checkpoint["scaler_state"])

    set_random_seed(epoch_seed)
    train_generator = torch.Generator().manual_seed(epoch_seed)
    train_loader = _RecordingLoader(
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=FORMAL_WORKERS,
            pin_memory=True,
            worker_init_fn=seed_worker,
            persistent_workers=False,
        )
    )
    initial_summaries, initial_health = _state_summary(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_generator=train_generator,
        validation_generator=validation_generator,
    )

    print(f"vdn_probe_worker={replicate} stage=train_epoch_101", flush=True)
    train_metrics = _validate_finite_metrics(
        _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=True,
            vector_weight=PHASE2_VECTOR_WEIGHT,
            epoch=FIXED_EPOCH,
            record_scaler_trace=True,
        ),
        kind="train",
    )
    expected_train_order_sha256 = sample_order_sha256(
        train_loader.sample_ids
    )
    if train_metrics["sample_order_sha256"] != expected_train_order_sha256:
        raise ValueError(
            "VDN probe train metric sample-order SHA-256 differs from the "
            "actual yielded batch order"
        )
    print(f"vdn_probe_worker={replicate} stage=grouped_validation", flush=True)
    validation_metrics = _validate_finite_metrics(
        _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=True,
        ),
        kind="validation",
    )
    torch.cuda.synchronize(device)
    expected_steps = math.ceil(len(lineage.train_samples) / batch_size)
    optimizer_steps = train_metrics["optimizer_steps"]
    skipped_optimizer_steps = train_metrics["skipped_optimizer_steps"]
    if (
        int(train_metrics["samples"]) != len(lineage.train_samples)
        or int(validation_metrics["samples"]) != len(
            lineage.validation_samples
        )
        or isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 0
        or isinstance(skipped_optimizer_steps, bool)
        or not isinstance(skipped_optimizer_steps, int)
        or skipped_optimizer_steps < 0
        or optimizer_steps + skipped_optimizer_steps != expected_steps
    ):
        raise ValueError("VDN probe epoch accounting is incomplete")
    if train_metrics["scaler_start_state"] != lineage.last_checkpoint[
        "scaler_state"
    ]:
        raise ValueError("VDN probe scaler does not start from parent terminal state")
    scaler_transition = validate_scaler_transition(
        train_metrics["scaler_start_state"],
        train_metrics["scaler_end_state"],
        train_metrics["scaler_skipped_batch_indices"],
        attempted_steps=expected_steps,
        label="VDN probe epoch 101",
    )
    if (
        scaler_transition["successful_steps"] != optimizer_steps
        or scaler_transition["skipped_steps"] != skipped_optimizer_steps
        or train_metrics["scaler_end_state"] != scaler.state_dict()
    ):
        raise ValueError("VDN probe scaler trace/accounting is inconsistent")
    validate_live_adam_optimizer(
        model,
        optimizer,
        expected_step=int(parent_health["optimizer_step"]) + optimizer_steps,
        expected_learning_rate=phase2_learning_rate(FIXED_EPOCH),
        label="VDN probe final",
    )

    final_summaries, final_health = _state_summary(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_generator=train_generator,
        validation_generator=validation_generator,
    )
    print(f"vdn_probe_worker={replicate} stage=finalize_snapshot", flush=True)
    identity = _worker_identity(
        lineage=lineage,
        manifest=manifest,
        vdn_source=vdn_source,
        content_inventory_verification=formal_inventory_verification,
        legacy_inventory_sha256=legacy_inventory_sha256,
        device=device,
        policy=policy,
    )
    return {
        "protocol": VDN_PHASE2_DETERMINISM_WORKER_PROTOCOL,
        "schema_version": VDN_PHASE2_DETERMINISM_SCHEMA_VERSION,
        "replicate": replicate,
        "execution": {
            "pid": os.getpid(),
            "process_token": uuid.uuid4().hex,
            "fresh_python_process": True,
        },
        "identity": identity,
        "sample_order": {
            "train": _order_report(
                train_loader.sample_ids,
                [sample.sample_id for sample in lineage.train_samples],
                require_exact_order=False,
            ),
            "validation": _order_report(
                validation_loader.sample_ids,
                [sample.sample_id for sample in lineage.validation_samples],
                require_exact_order=True,
            ),
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
        },
        "state_summaries": {
            "initial": initial_summaries,
            "final": final_summaries,
        },
        "state_health": {
            "initial": initial_health,
            "final": final_health,
        },
    }


def _compare_component(
    label: str,
    left: Any,
    right: Any,
) -> dict[str, Any]:
    mismatch = first_semantic_mismatch(left, right, path=label)
    if mismatch:
        raise VDNPhase2DeterminismMismatch(mismatch)
    digest = semantic_sha256(left)
    if digest != semantic_sha256(right):
        raise RuntimeError(f"{label}: equal values produced unequal digests")
    return {
        "exact": True,
        "semantic_sha256": digest,
    }


def _validate_worker_snapshot(
    snapshot: Mapping[str, Any],
    *,
    replicate: str,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> None:
    if set(snapshot) != _WORKER_KEYS:
        raise ValueError(f"worker {replicate} snapshot schema is invalid")
    if (
        snapshot.get("protocol") != VDN_PHASE2_DETERMINISM_WORKER_PROTOCOL
        or snapshot.get("schema_version")
        != VDN_PHASE2_DETERMINISM_SCHEMA_VERSION
        or snapshot.get("replicate") != replicate
    ):
        raise ValueError(f"worker {replicate} identity is invalid")
    execution = snapshot.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("fresh_python_process") is not True
        or not isinstance(execution.get("pid"), int)
        or execution["pid"] <= 0
        or not isinstance(execution.get("process_token"), str)
        or len(execution["process_token"]) != 32
    ):
        raise ValueError(f"worker {replicate} execution identity is invalid")

    identity = snapshot.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"worker {replicate} scientific identity is absent")
    if "content_inventory" not in identity:
        raise ValueError(
            f"worker {replicate} formal content-inventory identity is invalid"
        )
    if set(identity) != _SCIENTIFIC_IDENTITY_KEYS:
        raise ValueError(
            f"worker {replicate} scientific identity schema is invalid"
        )
    fixed = {
        "parent_seed": FIXED_PARENT_SEED,
        "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
        "epoch": FIXED_EPOCH,
        "epoch_seed": phase2_epoch_seed(FIXED_PARENT_SEED, FIXED_EPOCH),
        "batch_size": 8,
        "workers": FORMAL_WORKERS,
        "learning_rate": phase2_learning_rate(FIXED_EPOCH),
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "mixed_precision": True,
        "vector_weight": PHASE2_VECTOR_WEIGHT,
        "max_skipped_step_rate": PHASE2_MAX_SKIPPED_STEP_RATE,
        "scaler_trace_protocol": (
            "epoch_start_end_and_zero_based_skipped_batch_replay_v1"
        ),
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
    }
    for field, expected in fixed.items():
        if identity.get(field) != expected:
            raise ValueError(
                f"worker {replicate} identity mismatch for {field}"
            )
    if identity.get("forbidden_evaluation_scopes") != list(
        FORBIDDEN_EVALUATION_SCOPES
    ):
        raise ValueError(f"worker {replicate} forbidden-scope inventory drifted")
    scope = identity.get("scope")
    if (
        not isinstance(scope, str)
        or "SyncG official train grouped validation only" not in scope
        or "no test/public/RPM/Pointer/field/sealed/confirmatory" not in scope
    ):
        raise ValueError(f"worker {replicate} scope is not train-only")
    inventory = identity.get("content_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != (
        _CONTENT_INVENTORY_IDENTITY_KEYS
    ):
        raise ValueError(
            f"worker {replicate} formal content-inventory identity is invalid"
        )
    if (
        inventory.get("protocol")
        != content_inventory.INVENTORY_VERIFICATION_PROTOCOL
        or inventory.get("verified") is not True
        or inventory.get("content_rehashed") is not True
        or isinstance(inventory.get("rows"), bool)
        or not isinstance(inventory.get("rows"), int)
        or inventory.get("rows")
        != content_inventory.SYNCG_TRAIN_EXPECTED_ROWS
        or isinstance(inventory.get("fresh_rehash_workers"), bool)
        or not isinstance(inventory.get("fresh_rehash_workers"), int)
        or inventory.get("fresh_rehash_workers") <= 0
    ):
        raise ValueError(
            f"worker {replicate} formal content inventory was not freshly verified"
        )
    report_project_path = inventory.get("report_path")
    artifact_path = inventory.get("artifact_path")
    if (
        not isinstance(report_project_path, str)
        or not report_project_path
        or Path(report_project_path).is_absolute()
        or not isinstance(artifact_path, str)
        or not Path(artifact_path).is_absolute()
        or Path(artifact_path).resolve(strict=False)
        != (PROJECT_DIR / report_project_path).resolve(strict=False)
    ):
        raise ValueError(
            f"worker {replicate} content-inventory artifact path is inconsistent"
        )
    for field in (
        "inventory_report_sha256",
        "canonical_inventory_sha256",
        "canonical_report_payload_sha256",
        "manifest_sha256",
        "manifest_protocol_sha256",
        "vdn_protocol_document_sha256",
        "vdn_protocol_source_sha256",
        "inventory_tool_source_sha256",
    ):
        _require_sha256(
            inventory.get(field),
            f"worker {replicate}.content_inventory.{field}",
        )
    legacy_inventory = identity.get("legacy_content_inventory_crosscheck")
    expected_legacy = {
        "protocol": "pct_style_sample_image_annotation_sha256_v1",
        "sha256": PINNED_SYNCG_TRAIN_INPUT_INVENTORY_SHA256,
        "role": (
            "cross_evidence_only_not_a_substitute_for_the_formal_"
            "content_inventory_artifact"
        ),
    }
    if legacy_inventory != expected_legacy:
        raise ValueError(
            f"worker {replicate} legacy inventory cross-evidence is invalid"
        )
    for artifact, digest in (
        identity.get("parent_artifacts") or {}
    ).items():
        _require_sha256(digest, f"worker {replicate}.parent.{artifact}")
    source_hashes = identity.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError(f"worker {replicate} source inventory is absent")
    for source, digest in source_hashes.items():
        _require_sha256(digest, f"worker {replicate}.source.{source}")
    if (
        expected_source_hashes is not None
        and dict(source_hashes) != dict(expected_source_hashes)
    ):
        raise ValueError(f"worker {replicate} source inventory is stale")
    if (
        inventory["manifest_sha256"] != identity.get("manifest_sha256")
        or inventory["manifest_protocol_sha256"]
        != identity.get("manifest_protocol_sha256")
        or inventory["inventory_tool_source_sha256"]
        != source_hashes.get("content_inventory_tool")
        or inventory["vdn_protocol_source_sha256"]
        != source_hashes.get("phase2_protocol")
    ):
        raise ValueError(
            f"worker {replicate} content-inventory binding is inconsistent"
        )
    environment = identity.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _RUNTIME_ENVIRONMENT_KEYS
    ):
        raise ValueError(f"worker {replicate} environment is absent")
    policy = environment.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError(f"worker {replicate} deterministic policy is absent")
    validate_determinism_policy(policy)

    metrics = snapshot.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"worker {replicate} metrics are absent")
    for kind in ("train", "validation"):
        values = metrics.get(kind)
        if not isinstance(values, Mapping):
            raise ValueError(f"worker {replicate} {kind} metrics are absent")
        _validate_finite_metrics(values, kind=kind)

    orders = snapshot.get("sample_order")
    if not isinstance(orders, Mapping) or set(orders) != {
        "train",
        "validation",
    }:
        raise ValueError(f"worker {replicate} sample-order report is invalid")
    for kind in ("train", "validation"):
        report = orders[kind]
        if (
            not isinstance(report, Mapping)
            or int(report.get("samples", -1)) <= 0
            or report.get("samples") != report.get("unique_samples")
        ):
            raise ValueError(
                f"worker {replicate} {kind} sample order is incomplete"
            )
        _require_sha256(
            report.get("actual_order_sha256"),
            f"worker {replicate}.{kind}.actual_order_sha256",
        )
        _require_sha256(
            report.get("expected_population_sha256"),
            f"worker {replicate}.{kind}.expected_population_sha256",
        )
    if (
        metrics["train"]["sample_order_sha256"]
        != orders["train"]["actual_order_sha256"]
    ):
        raise ValueError(
            f"worker {replicate} train metric sample-order SHA-256 "
            "does not match the actual loader order"
        )

    summaries = snapshot.get("state_summaries")
    health = snapshot.get("state_health")
    if not isinstance(summaries, Mapping) or set(summaries) != {
        "initial",
        "final",
    }:
        raise ValueError(f"worker {replicate} state summaries are invalid")
    if not isinstance(health, Mapping) or set(health) != {
        "initial",
        "final",
    }:
        raise ValueError(f"worker {replicate} state health is invalid")
    for stage in ("initial", "final"):
        stage_summary = summaries[stage]
        if not isinstance(stage_summary, Mapping) or set(stage_summary) != {
            "model",
            "optimizer",
            "scaler",
            "rng",
            "combined_semantic_sha256",
        }:
            raise ValueError(
                f"worker {replicate} {stage} state summary is invalid"
            )
        for component in ("model", "optimizer", "scaler", "rng"):
            record = stage_summary[component]
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"worker {replicate} {stage}.{component} is invalid"
                )
            _require_sha256(
                record.get("semantic_sha256"),
                f"worker {replicate}.{stage}.{component}",
            )
        _require_sha256(
            stage_summary.get("combined_semantic_sha256"),
            f"worker {replicate}.{stage}.combined",
        )
        component_summaries = {
            component: dict(stage_summary[component])
            for component in ("model", "optimizer", "scaler", "rng")
        }
        if stage_summary["combined_semantic_sha256"] != (
            _canonical_json_sha256(component_summaries)
        ):
            raise ValueError(
                f"worker {replicate} {stage} combined state hash is invalid"
            )
        stage_health = health[stage]
        if not isinstance(stage_health, Mapping):
            raise ValueError(
                f"worker {replicate} {stage} state health is invalid"
            )
        for component in ("model", "optimizer"):
            record = stage_health.get(component)
            if (
                not isinstance(record, Mapping)
                or int(record.get("tensors", 0)) <= 0
                or int(record.get("elements", 0)) <= 0
                or int(record.get("nonfinite_floating_elements", -1)) != 0
            ):
                raise ValueError(
                    f"worker {replicate} {stage}.{component} is unhealthy"
                )

    train_samples = int(identity.get("train_samples", -1))
    validation_samples = int(identity.get("validation_samples", -1))
    if (
        train_samples <= 0
        or validation_samples <= 0
        or train_samples + validation_samples != int(inventory["rows"])
        or int(orders["train"]["samples"]) != train_samples
        or int(orders["validation"]["samples"]) != validation_samples
        or int(metrics["train"]["samples"]) != train_samples
        or int(metrics["validation"]["samples"]) != validation_samples
    ):
        raise ValueError(f"worker {replicate} sample accounting is inconsistent")
    batch_size = int(identity["batch_size"])
    expected_steps = math.ceil(train_samples / batch_size)
    optimizer_steps = metrics["train"]["optimizer_steps"]
    skipped_optimizer_steps = metrics["train"]["skipped_optimizer_steps"]
    if (
        isinstance(optimizer_steps, bool)
        or not isinstance(optimizer_steps, int)
        or optimizer_steps < 0
        or isinstance(skipped_optimizer_steps, bool)
        or not isinstance(skipped_optimizer_steps, int)
        or skipped_optimizer_steps < 0
        or optimizer_steps + skipped_optimizer_steps != expected_steps
    ):
        raise ValueError(
            f"worker {replicate} optimizer-step accounting is inconsistent"
        )
    full_phase_max_skipped_steps = math.floor(
        expected_steps
        * (PHASE2_END_EPOCH - PHASE2_START_EPOCH + 1)
        * PHASE2_MAX_SKIPPED_STEP_RATE
    )
    if (
        identity.get("full_phase_max_skipped_steps")
        != full_phase_max_skipped_steps
        or skipped_optimizer_steps > full_phase_max_skipped_steps
    ):
        raise ValueError(
            f"worker {replicate} skipped-step budget is inconsistent"
        )
    scaler_transition = validate_scaler_transition(
        metrics["train"]["scaler_start_state"],
        metrics["train"]["scaler_end_state"],
        metrics["train"]["scaler_skipped_batch_indices"],
        attempted_steps=expected_steps,
        label=f"worker {replicate} probe scaler",
    )
    if (
        scaler_transition["successful_steps"] != optimizer_steps
        or scaler_transition["skipped_steps"] != skipped_optimizer_steps
    ):
        raise ValueError(
            f"worker {replicate} scaler trace/accounting is inconsistent"
        )
    if semantic_sha256(
        metrics["train"]["scaler_start_state"]
    ) != summaries["initial"]["scaler"]["semantic_sha256"]:
        raise ValueError(
            f"worker {replicate} scaler start is not bound to initial state"
        )
    if semantic_sha256(
        metrics["train"]["scaler_end_state"]
    ) != summaries["final"]["scaler"]["semantic_sha256"]:
        raise ValueError(
            f"worker {replicate} scaler end is not bound to final state"
        )


def compare_worker_snapshots(
    worker_a: Mapping[str, Any],
    worker_b: Mapping[str, Any],
    *,
    expected_source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    _validate_worker_snapshot(
        worker_a,
        replicate="a",
        expected_source_hashes=expected_source_hashes,
    )
    _validate_worker_snapshot(
        worker_b,
        replicate="b",
        expected_source_hashes=expected_source_hashes,
    )
    if worker_a["execution"]["process_token"] == worker_b["execution"][
        "process_token"
    ]:
        raise VDNPhase2DeterminismMismatch(
            "workers.execution: A/B process tokens are identical"
        )

    components = {
        name: _compare_component(
            name,
            worker_a[name],
            worker_b[name],
        )
        for name in (
            "identity",
            "sample_order",
            "metrics",
            "state_summaries",
            "state_health",
        )
    }
    identity = dict(worker_a["identity"])
    return {
        "protocol": VDN_PHASE2_DETERMINISM_PROBE_PROTOCOL,
        "schema_version": VDN_PHASE2_DETERMINISM_SCHEMA_VERSION,
        "status": "passed",
        "first_mismatch": None,
        "vdn_phase2_training_start_authorized": True,
        "test_evaluation_authorized": False,
        "authorization_scope": (
            "formal VDN Phase-2 training startup only; convergence and all "
            "three seed verifications remain required before test evaluation"
        ),
        "fixed_probe": {
            "parent_seed": FIXED_PARENT_SEED,
            "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
            "epoch": FIXED_EPOCH,
            "epoch_seed": phase2_epoch_seed(
                FIXED_PARENT_SEED,
                FIXED_EPOCH,
            ),
            "complete_epoch": True,
            "separate_python_processes": True,
            "model_optimizer_scaler_dataloader_rebuilt_per_worker": True,
        },
        "scientific_identity": identity,
        "components": components,
        "sample_order": dict(worker_a["sample_order"]),
        "metrics": dict(worker_a["metrics"]),
        "state_summaries": dict(worker_a["state_summaries"]),
        "state_health": dict(worker_a["state_health"]),
        "worker_execution": {
            "a_pid": worker_a["execution"]["pid"],
            "b_pid": worker_b["execution"]["pid"],
            "process_tokens_distinct": True,
        },
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "probe_source_sha256": _require_sha256(
            identity["source_sha256"]["probe"],
            "identity.source_sha256.probe",
        ),
    }


def validate_determinism_authorization_report(
    report_path: Path,
    *,
    manifest: Path,
    vdn_source: Path,
    content_inventory_path: Path,
) -> dict[str, Any]:
    """Freshly validate the persisted v2 probe before any CUDA initialization."""

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before the VDN determinism authorization gate"
        )
    report_path = Path(report_path).resolve(strict=True)
    expected_report_path = FIXED_DETERMINISM_REPORT.resolve(strict=False)
    if report_path != expected_report_path:
        raise ValueError(
            f"determinism report path drifted: {report_path} != "
            f"{expected_report_path}"
        )
    if not report_path.is_file():
        raise ValueError(
            f"determinism report is not a regular file: {report_path}"
        )
    manifest = Path(manifest).resolve(strict=True)
    vdn_source = Path(vdn_source).resolve(strict=True)
    content_inventory_path = Path(content_inventory_path).resolve(strict=True)
    raw_sha_before = sha256_file(report_path)
    report = _read_strict_json(report_path)
    if set(report) != _PASSED_REPORT_KEYS:
        raise ValueError("determinism authorization report schema drifted")
    if (
        report.get("protocol") != VDN_PHASE2_DETERMINISM_PROBE_PROTOCOL
        or report.get("schema_version")
        != VDN_PHASE2_DETERMINISM_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("first_mismatch") is not None
        or report.get("vdn_phase2_training_start_authorized") is not True
        or report.get("test_evaluation_authorized") is not False
    ):
        raise ValueError("determinism report is not a v2 passed authorization")
    expected_scope = (
        "formal VDN Phase-2 training startup only; convergence and all "
        "three seed verifications remain required before test evaluation"
    )
    if report.get("authorization_scope") != expected_scope:
        raise ValueError("determinism authorization scope drifted")

    expected_fixed = {
        "parent_seed": FIXED_PARENT_SEED,
        "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
        "epoch": FIXED_EPOCH,
        "epoch_seed": phase2_epoch_seed(FIXED_PARENT_SEED, FIXED_EPOCH),
        "complete_epoch": True,
        "separate_python_processes": True,
        "model_optimizer_scaler_dataloader_rebuilt_per_worker": True,
    }
    if report.get("fixed_probe") != expected_fixed:
        raise ValueError("determinism report fixed epoch/seed identity drifted")

    identity = report.get("scientific_identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _SCIENTIFIC_IDENTITY_KEYS
    ):
        raise ValueError("determinism report scientific identity is invalid")
    reported_inventory = identity.get("content_inventory")
    if (
        not isinstance(reported_inventory, Mapping)
        or set(reported_inventory) != _CONTENT_INVENTORY_IDENTITY_KEYS
    ):
        raise ValueError("determinism report content inventory is invalid")
    inventory_workers = reported_inventory.get("fresh_rehash_workers")
    if (
        isinstance(inventory_workers, bool)
        or not isinstance(inventory_workers, int)
        or inventory_workers <= 0
    ):
        raise ValueError("determinism report inventory worker count is invalid")
    fresh_inventory = _verify_content_inventory_fresh(
        content_inventory_path,
        manifest=manifest,
        workers=inventory_workers,
    )
    if dict(reported_inventory) != fresh_inventory:
        raise ValueError(
            "determinism report content inventory differs from the fresh rehash"
        )

    current_sources = _current_source_hashes(vdn_source)
    if identity.get("source_sha256") != current_sources:
        raise ValueError("determinism report source inventory is stale")
    if report.get("probe_source_sha256") != current_sources["probe"]:
        raise ValueError("determinism report probe source SHA-256 is stale")
    if report.get("source_hash_protocol") != PHASE2_SOURCE_HASH_PROTOCOL:
        raise ValueError("determinism report source-hash protocol drifted")

    lineage = load_parent_lineage(
        FIXED_PARENT_RUN,
        manifest=manifest,
        load_checkpoints=False,
    )
    parent_signature = lineage.summary["signature"]
    expected_identity_fields = {
        "scope": (
            "pinned SyncG official train grouped validation only; no "
            "test/public/RPM/Pointer/field/sealed/confirmatory input"
        ),
        "forbidden_evaluation_scopes": list(FORBIDDEN_EVALUATION_SCOPES),
        "parent_seed": FIXED_PARENT_SEED,
        "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
        "epoch": FIXED_EPOCH,
        "epoch_seed": phase2_epoch_seed(FIXED_PARENT_SEED, FIXED_EPOCH),
        "batch_size": int(parent_signature["batch_size"]),
        "workers": FORMAL_WORKERS,
        "learning_rate": phase2_learning_rate(FIXED_EPOCH),
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "mixed_precision": True,
        "vector_weight": PHASE2_VECTOR_WEIGHT,
        "max_skipped_step_rate": PHASE2_MAX_SKIPPED_STEP_RATE,
        "full_phase_max_skipped_steps": math.floor(
            math.ceil(
                len(lineage.train_samples)
                / int(parent_signature["batch_size"])
            )
            * (PHASE2_END_EPOCH - PHASE2_START_EPOCH + 1)
            * PHASE2_MAX_SKIPPED_STEP_RATE
        ),
        "scaler_trace_protocol": (
            "epoch_start_end_and_zero_based_skipped_batch_replay_v1"
        ),
        "parent_run": str(lineage.run_dir),
        "parent_artifacts": lineage.artifact_hashes(),
        "parent_signature_semantic_sha256": semantic_sha256(
            parent_signature
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(
            manifest.with_name(manifest.name + ".protocol.json")
        ),
        "content_inventory": fresh_inventory,
        "legacy_content_inventory_crosscheck": {
            "protocol": "pct_style_sample_image_annotation_sha256_v1",
            "sha256": PINNED_SYNCG_TRAIN_INPUT_INVENTORY_SHA256,
            "role": (
                "cross_evidence_only_not_a_substitute_for_the_formal_"
                "content_inventory_artifact"
            ),
        },
        "train_samples": len(lineage.train_samples),
        "validation_samples": len(lineage.validation_samples),
        "train_groups": len(
            {sample.group_id for sample in lineage.train_samples}
        ),
        "validation_groups": len(
            {sample.group_id for sample in lineage.validation_samples}
        ),
        "train_sample_ids_sha256": _ordered_ids_sha256(
            sample.sample_id for sample in lineage.train_samples
        ),
        "validation_sample_ids_sha256": _ordered_ids_sha256(
            sample.sample_id for sample in lineage.validation_samples
        ),
        "vdn_source": str(vdn_source),
        "vdn_source_commit": verify_vdn_source(vdn_source),
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "source_sha256": current_sources,
    }
    for field, expected in expected_identity_fields.items():
        if identity.get(field) != expected:
            raise ValueError(
                f"determinism report scientific identity drifted for {field}"
            )

    environment = identity.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _RUNTIME_ENVIRONMENT_KEYS
    ):
        raise ValueError("determinism report runtime environment is invalid")
    policy = environment.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("determinism report policy is absent")
    validate_determinism_policy(policy)
    stable_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "amp": True,
    }
    for field, expected in stable_environment.items():
        if environment.get(field) != expected:
            raise ValueError(
                f"determinism report environment is stale for {field}"
            )
    if (
        environment.get("device") != "cuda:0"
        or not isinstance(environment.get("gpu_name"), str)
        or not environment["gpu_name"]
        or not isinstance(environment.get("gpu_capability"), list)
        or len(environment["gpu_capability"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in environment["gpu_capability"]
        )
        or isinstance(environment.get("gpu_total_memory_bytes"), bool)
        or not isinstance(environment.get("gpu_total_memory_bytes"), int)
        or environment["gpu_total_memory_bytes"] <= 0
    ):
        raise ValueError("determinism report GPU environment is invalid")

    worker_execution = report.get("worker_execution")
    if (
        not isinstance(worker_execution, Mapping)
        or set(worker_execution)
        != {"a_pid", "b_pid", "process_tokens_distinct"}
        or any(
            isinstance(worker_execution.get(field), bool)
            or not isinstance(worker_execution.get(field), int)
            or worker_execution[field] <= 0
            for field in ("a_pid", "b_pid")
        )
        or worker_execution.get("process_tokens_distinct") is not True
    ):
        raise ValueError("determinism report worker execution is invalid")
    synthetic_worker = {
        "protocol": VDN_PHASE2_DETERMINISM_WORKER_PROTOCOL,
        "schema_version": VDN_PHASE2_DETERMINISM_SCHEMA_VERSION,
        "replicate": "a",
        "execution": {
            "pid": worker_execution["a_pid"],
            "process_token": "a" * 32,
            "fresh_python_process": True,
        },
        "identity": dict(identity),
        "sample_order": report.get("sample_order"),
        "metrics": report.get("metrics"),
        "state_summaries": report.get("state_summaries"),
        "state_health": report.get("state_health"),
    }
    _validate_worker_snapshot(
        synthetic_worker,
        replicate="a",
        expected_source_hashes=current_sources,
    )

    components = report.get("components")
    component_values = {
        "identity": identity,
        "sample_order": report["sample_order"],
        "metrics": report["metrics"],
        "state_summaries": report["state_summaries"],
        "state_health": report["state_health"],
    }
    if not isinstance(components, Mapping) or set(components) != set(
        component_values
    ):
        raise ValueError("determinism report component ledger is invalid")
    for name, value in component_values.items():
        record = components[name]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"exact", "semantic_sha256"}
            or record.get("exact") is not True
            or record.get("semantic_sha256") != semantic_sha256(value)
        ):
            raise ValueError(
                f"determinism report component {name} is not exact"
            )

    raw_sha_after = sha256_file(report_path)
    if raw_sha_after != raw_sha_before:
        raise RuntimeError("determinism report changed during fresh validation")
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "determinism authorization validation initialized CUDA"
        )
    try:
        report_project_path = report_path.relative_to(
            PROJECT_DIR.resolve(strict=True)
        ).as_posix()
    except ValueError as exc:
        raise ValueError(
            "determinism report must remain inside the project root"
        ) from exc
    return {
        "protocol": VDN_PHASE2_AUTHORIZATION_BINDING_PROTOCOL,
        "report_protocol": VDN_PHASE2_DETERMINISM_PROBE_PROTOCOL,
        "report_schema_version": VDN_PHASE2_DETERMINISM_SCHEMA_VERSION,
        "report_path": report_project_path,
        "report_sha256": raw_sha_after,
        "canonical_report_sha256": _canonical_json_sha256(report),
        "status": "passed",
        "vdn_phase2_training_start_authorized": True,
        "test_evaluation_authorized": False,
        "fixed_probe": dict(report["fixed_probe"]),
        "scientific_identity": dict(identity),
        "components": dict(components),
        "source_hash_protocol": PHASE2_SOURCE_HASH_PROTOCOL,
        "probe_source_sha256": current_sources["probe"],
        "fresh_validation": {
            "report_reparsed_strictly": True,
            "content_inventory_rehashed": True,
            "current_sources_rehashed": True,
            "fixed_parent_lineage_revalidated": True,
            "cuda_uninitialized": True,
        },
    }


def validate_training_process_pre_cuda_policy(
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Require process-start policy for training, without initializing CUDA."""

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before the VDN training process policy gate"
        )
    identity = authorization.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("VDN determinism authorization identity is absent")
    environment = identity.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("VDN determinism authorization environment is absent")
    policy = environment.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("VDN determinism authorization policy is absent")
    validate_determinism_policy(policy)
    expected_hash_seed = str(policy["pythonhashseed"])
    if os.environ.get("PYTHONHASHSEED") != expected_hash_seed:
        raise RuntimeError(
            "PYTHONHASHSEED must be set before interpreter startup to "
            f"{expected_hash_seed}"
        )
    expected_cublas = str(policy["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_cublas:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be set before Python startup to "
            f"{expected_cublas}"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "VDN training process policy validation initialized CUDA"
        )
    return {
        "pythonhashseed": expected_hash_seed,
        "cublas_workspace_config": expected_cublas,
        "cuda_uninitialized": True,
    }


def validate_authorized_runtime_environment(
    authorization: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the trainer's initialized CUDA runtime to the authorized probe."""

    if authorization.get("protocol") != (
        VDN_PHASE2_AUTHORIZATION_BINDING_PROTOCOL
    ):
        raise ValueError("VDN determinism authorization binding is invalid")
    identity = authorization.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("VDN determinism authorization identity is absent")
    authorized = identity.get("environment")
    if not isinstance(authorized, Mapping):
        raise ValueError("VDN authorized runtime environment is absent")
    exact_pairs = {
        "python": "python",
        "torch": "torch",
        "numpy": "numpy",
        "cuda_runtime": "cuda_runtime",
        "cudnn": "cudnn_version",
        "gpu_name": "gpu",
        "gpu_capability": "gpu_capability",
        "gpu_total_memory_bytes": "gpu_total_memory_bytes",
        "amp": "amp",
    }
    for authorized_key, runtime_key in exact_pairs.items():
        if authorized.get(authorized_key) != runtime_environment.get(
            runtime_key
        ):
            raise ValueError(
                f"VDN runtime differs from determinism probe for "
                f"{authorized_key}"
            )
    if (
        authorized.get("device") != "cuda:0"
        or int(runtime_environment.get("device_index", -1)) != 0
    ):
        raise ValueError("VDN runtime CUDA device differs from the fixed probe")
    probe_policy = authorized.get("policy")
    runtime_policy = runtime_environment.get("determinism")
    if not isinstance(probe_policy, Mapping) or not isinstance(
        runtime_policy,
        Mapping,
    ):
        raise ValueError("VDN runtime deterministic policy is absent")
    policy_pairs = {
        "cublas_workspace_config": "cublas_workspace_config",
        "deterministic_algorithms": "torch_deterministic_algorithms",
        "deterministic_warn_only": "torch_deterministic_warn_only",
        "cudnn_deterministic": "cudnn_deterministic",
        "cudnn_benchmark": "cudnn_benchmark",
        "cuda_matmul_allow_tf32": "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32": "cudnn_allow_tf32",
        "cuda_matmul_allow_fp16_reduced_precision_reduction": (
            "cuda_matmul_allow_fp16_reduced_precision_reduction"
        ),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": (
            "cuda_matmul_allow_bf16_reduced_precision_reduction"
        ),
        "float32_matmul_precision": "float32_matmul_precision",
    }
    for probe_key, runtime_key in policy_pairs.items():
        if probe_policy.get(probe_key) != runtime_policy.get(runtime_key):
            raise ValueError(
                f"VDN runtime deterministic policy drifted for {probe_key}"
            )
    return {
        "probe_environment_semantic_sha256": semantic_sha256(authorized),
        "runtime_environment_semantic_sha256": semantic_sha256(
            runtime_environment
        ),
        "exact_authorized_runtime": True,
    }


def _worker_command(
    *,
    replicate: str,
    worker_output: Path,
    parent_run: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory_path: Path,
    inventory_workers: int,
    device_text: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.probe_vdn_phase2_determinism",
        "--worker-output",
        str(worker_output),
        "--replicate",
        replicate,
        "--parent-run",
        str(parent_run),
        "--manifest",
        str(manifest),
        "--vdn-source",
        str(vdn_source),
        "--content-inventory",
        str(content_inventory_path),
        "--inventory-workers",
        str(inventory_workers),
        "--device",
        device_text,
    ]


def _run_subprocess_worker(
    *,
    replicate: str,
    worker_output: Path,
    parent_run: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory_path: Path,
    inventory_workers: int,
    device_text: str,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        _worker_command(
            replicate=replicate,
            worker_output=worker_output,
            parent_run=parent_run,
            manifest=manifest,
            vdn_source=vdn_source,
            content_inventory_path=content_inventory_path,
            inventory_workers=inventory_workers,
            device_text=device_text,
        ),
        cwd=PROJECT_DIR,
        env=dict(environment),
        check=False,
    )
    if completed.returncode != 0:
        raise VDNPhase2WorkerFailure(
            f"worker {replicate} exited {completed.returncode}; "
            "its inherited stderr contains the diagnostic"
        )
    if not worker_output.is_file():
        raise VDNPhase2WorkerFailure(
            f"worker {replicate} produced no snapshot"
        )
    return _read_strict_json(worker_output)


def run_probe(
    *,
    parent_run: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory_path: Path,
    inventory_workers: int,
    device_text: str,
) -> dict[str, Any]:
    parent_run = parent_run.resolve(strict=True)
    manifest = manifest.resolve(strict=True)
    vdn_source = vdn_source.resolve(strict=True)
    content_inventory_path = content_inventory_path.resolve(strict=True)
    if (
        isinstance(inventory_workers, bool)
        or not isinstance(inventory_workers, int)
        or inventory_workers <= 0
    ):
        raise ValueError("inventory workers must be a positive integer")
    environment = dict(os.environ)
    existing_cublas = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, FIXED_CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "refusing conflicting CUBLAS_WORKSPACE_CONFIG="
            f"{existing_cublas!r}"
        )
    environment["CUBLAS_WORKSPACE_CONFIG"] = (
        FIXED_CUBLAS_WORKSPACE_CONFIG
    )
    environment["PYTHONHASHSEED"] = str(
        formal_phase2_seed(FIXED_PARENT_SEED)
    )
    environment["PYTHONUNBUFFERED"] = "1"

    with tempfile.TemporaryDirectory(
        prefix="vdn_phase2_determinism_",
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        snapshots = {}
        for replicate in ("a", "b"):
            snapshots[replicate] = _run_subprocess_worker(
                replicate=replicate,
                worker_output=temporary / f"worker_{replicate}.json",
                parent_run=parent_run,
                manifest=manifest,
                vdn_source=vdn_source,
                content_inventory_path=content_inventory_path,
                inventory_workers=inventory_workers,
                device_text=device_text,
                environment=environment,
            )
        current_sources = _current_source_hashes(vdn_source)
        return compare_worker_snapshots(
            snapshots["a"],
            snapshots["b"],
            expected_source_hashes=current_sources,
        )


def _failure_report(exc: BaseException) -> dict[str, Any]:
    return {
        "protocol": VDN_PHASE2_DETERMINISM_PROBE_PROTOCOL,
        "schema_version": VDN_PHASE2_DETERMINISM_SCHEMA_VERSION,
        "status": "failed",
        "first_mismatch": str(exc),
        "error_type": type(exc).__name__,
        "vdn_phase2_training_start_authorized": False,
        "test_evaluation_authorized": False,
        "authorization_scope": (
            "none; determinism probe failed before VDN Phase-2 startup"
        ),
        "fixed_probe": {
            "parent_seed": FIXED_PARENT_SEED,
            "phase_seed": formal_phase2_seed(FIXED_PARENT_SEED),
            "epoch": FIXED_EPOCH,
            "epoch_seed": phase2_epoch_seed(
                FIXED_PARENT_SEED,
                FIXED_EPOCH,
            ),
        },
        "forbidden_evaluation_scopes": list(FORBIDDEN_EVALUATION_SCOPES),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.worker_output is not None:
        if args.output is not None or args.replicate is None:
            raise ValueError(
                "worker mode requires --replicate and rejects --output"
            )
        snapshot = run_worker(
            replicate=args.replicate,
            parent_run=args.parent_run,
            manifest=args.manifest,
            vdn_source=args.vdn_source,
            content_inventory_path=args.content_inventory,
            inventory_workers=args.inventory_workers,
            device_text=args.device,
        )
        _atomic_write_json_no_clobber(args.worker_output, snapshot)
        print(args.worker_output.resolve(), flush=True)
        return snapshot

    if args.replicate is not None:
        raise ValueError("--replicate is private to worker mode")
    if args.output is None:
        raise ValueError("coordinator mode requires --output")
    output = args.output.resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite determinism report: {output}"
        )
    try:
        report = run_probe(
            parent_run=args.parent_run,
            manifest=args.manifest,
            vdn_source=args.vdn_source,
            content_inventory_path=args.content_inventory,
            inventory_workers=args.inventory_workers,
            device_text=args.device,
        )
    except Exception as exc:
        failure = _failure_report(exc)
        _atomic_write_json_no_clobber(output, failure)
        raise SystemExit(
            f"VDN Phase-2 determinism probe failed: {exc}"
        ) from exc
    _atomic_write_json_no_clobber(output, report)
    print(output, flush=True)
    return report


if __name__ == "__main__":
    main()
