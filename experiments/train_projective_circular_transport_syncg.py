"""Continue a signed SyncG v2 run with projective circular transport.

The formal comparison is deliberately narrow.  Both the continued-control
(``posterior_transport_weight == 0``) and PCT-full treatment start from the
same verified v2 ``best.pt`` checkpoint, use the same group split and training
budget, and differ only in the posterior-transport weight.  By default the
signed v2 supervised and mean-ray equivariance objectives are retained so the
transport contribution is isolated.  The v3 expert-supervision objective is
available only as an explicit experimental mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

# Required by deterministic CUDA matrix reductions when cuBLAS is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.datasets import (
    SYNCG_PINNED_COMMIT,
    SYNCG_SAMPLE_IDS_SHA256,
    SYNCG_TRAIN_ROWS,
    syncg_sample_ids_sha256,
)
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
    circular_delta,
    decode_probabilistic_pivot_direction,
    equivariance_loss,
    probabilistic_direction_loss,
)
from experiments.projective_circular_transport import (
    bidirectional_projective_posterior_consistency,
    fuse_circular_experts,
    periodic_linear_discrete_nll,
    projective_circular_supervised_loss_v3,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    angular_error_degrees,
    grouped_train_val_split,
    load_syncg_manifest,
    sample_ids_hash,
    seed_worker,
    set_random_seed,
    sha256_file,
    sha256_source_file,
)


PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL = (
    "syncg_projective_circular_transport_continuation_v2"
)
PINNED_SYNCG_TRAIN_INPUT_INVENTORY_SHA256 = (
    "d03011ee1e9afd80ed2698fd975ceb665c5247c982141ef0c5d007b16d7952d1"
)
SIGNED_V2_VERIFICATION_PROTOCOLS = frozenset(
    {
        "formal_probabilistic_direction_run_verification_v1",
        "formal_probabilistic_direction_run_verification_v2",
    }
)
LEGACY_V1_VERIFIER_SOURCE_SHA256 = (
    "b5fadf7008bb2a8a3b202886b53d2514c0dc41db0e455719f990a590bee785a6"
)
DEFAULT_INITIAL_RUN = (
    PROJECT_DIR
    / "artifacts"
    / "runs"
    / "probabilistic_pivot_direction_syncg"
    / "seed_20260722"
)
_FORBIDDEN_SCOPE_COMPONENT = re.compile(
    r"^(?:test(?:[-_]?(?:data|set|images?).*)?"
    r"|rpm(?:[-_]?(?:10k|data|dataset).*)?"
    r"|pointer(?:[-_]?(?:10k|data|dataset).*)?"
    r"|field(?:[-_]?(?:data|test|holdout|blind).*)?"
    r"|confirmatory(?:[-_].*)?)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=DEFAULT_INITIAL_RUN / "best.pt",
    )
    parser.add_argument(
        "--initial-verification",
        type=Path,
        default=DEFAULT_INITIAL_RUN / "verification.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/runs/projective_circular_transport_syncg/"
            "pct_full_seed_20260722"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--posterior-transport-weight", type=float, default=0.50)
    parser.add_argument(
        "--supervision-mode",
        choices=("legacy", "expert_v3"),
        default="legacy",
    )
    parser.add_argument(
        "--posterior-source",
        choices=("auto", "legacy_angle_bin_only", "gpoe_fused"),
        default="auto",
        help=(
            "auto selects angle-bin-only transport for legacy supervision and "
            "GPoE only for expert_v3; incompatible explicit combinations fail"
        ),
    )
    parser.add_argument("--posterior-bins", type=int, default=360)
    parser.add_argument("--min-direct-precision", type=float, default=0.05)
    parser.add_argument("--direct-precision-cap", type=float, default=400.0)
    parser.add_argument("--direct-precision-temperature", type=float, default=1.0)
    parser.add_argument("--bin-expert-power", type=float, default=0.5)
    parser.add_argument("--direct-expert-power", type=float, default=0.5)
    parser.add_argument("--expert-direct-nll-weight", type=float, default=1.0)
    parser.add_argument("--expert-bin-ce-weight", type=float, default=0.25)
    parser.add_argument("--expert-fused-posterior-weight", type=float, default=0.50)
    parser.add_argument("--expert-fused-mean-weight", type=float, default=0.25)
    parser.add_argument("--expert-overconfidence-weight", type=float, default=0.10)
    parser.add_argument("--expert-precision-barrier-weight", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scope_components(path: Path) -> list[str]:
    resolved = path.resolve()
    try:
        resolved = resolved.relative_to(PROJECT_DIR.resolve())
    except ValueError:
        pass
    return [part.lower() for part in resolved.parts]


def _reject_forbidden_scope_path(path: Path, *, role: str) -> None:
    bad = [
        part
        for part in _scope_components(path)
        if _FORBIDDEN_SCOPE_COMPONENT.fullmatch(part)
    ]
    if bad:
        raise ValueError(f"{role} is outside the SyncG train scope: {path}")


def _validate_syncg_train_scope(manifest: Path, samples: Iterable[Any]) -> None:
    """Fail closed on public-test, RPM, Pointer-10K, field, and confirmatory data."""

    manifest = manifest.resolve()
    if manifest.name.lower() != "syncg_train.jsonl":
        raise ValueError("PCT training requires a manifest named syncg_train.jsonl")
    _reject_forbidden_scope_path(manifest, role="manifest")
    sample_list = list(samples)
    protocol_path = _protocol_path(manifest)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = _read_json(protocol_path)
    required_protocol = {
        "protocol": "syncg_official_split_v1",
        "dataset": "SyncG",
        "split": "train",
        "strict_release": True,
        "release_identity_verified": True,
        "expected_rows": SYNCG_TRAIN_ROWS,
        "emitted_rows": SYNCG_TRAIN_ROWS,
        "reference_huggingface_commit": SYNCG_PINNED_COMMIT,
        "expected_sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
        "sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"SyncG train protocol {key}={protocol.get(key)!r}; "
                f"expected pinned value {expected!r}"
            )
    if len(sample_list) != SYNCG_TRAIN_ROWS:
        raise ValueError(
            f"SyncG train has {len(sample_list)} rows; expected {SYNCG_TRAIN_ROWS}"
        )
    sample_id_hash = syncg_sample_ids_sha256(
        str(sample.sample_id) for sample in sample_list
    )
    if sample_id_hash != SYNCG_SAMPLE_IDS_SHA256["train"]:
        raise ValueError("SyncG train sample identities differ from the pinned release")

    syncg_root_value = protocol.get("syncg_root")
    if not isinstance(syncg_root_value, str) or not syncg_root_value:
        raise ValueError("SyncG protocol lacks its resolved dataset root")
    syncg_root = Path(syncg_root_value).resolve()
    if not syncg_root.is_dir():
        raise FileNotFoundError(syncg_root)
    for sample in sample_list:
        if str(sample.dataset).strip().lower() != "syncg":
            raise ValueError(f"{sample.sample_id}: training dataset is not SyncG")
        if str(sample.split).strip().lower() != "train":
            raise ValueError(f"{sample.sample_id}: training split is not train")
        image_path = Path(str(sample.image_path)).resolve()
        _reject_forbidden_scope_path(image_path, role=f"{sample.sample_id} image")
        try:
            relative_image = image_path.relative_to(syncg_root)
        except ValueError as exc:
            raise ValueError(
                f"{sample.sample_id}: image escapes the pinned SyncG root"
            ) from exc
        if "train" not in [part.casefold() for part in relative_image.parts]:
            raise ValueError(
                f"{sample.sample_id}: image is not inside a SyncG train tree"
            )
        annotation = (sample.metadata or {}).get("annotation_path")
        if annotation:
            annotation_path = Path(str(annotation)).resolve()
            _reject_forbidden_scope_path(
                annotation_path,
                role=f"{sample.sample_id} annotation",
            )
            try:
                relative_annotation = annotation_path.relative_to(syncg_root)
            except ValueError as exc:
                raise ValueError(
                    f"{sample.sample_id}: annotation escapes the pinned SyncG root"
                ) from exc
            if "train" not in [
                part.casefold() for part in relative_annotation.parts
            ]:
                raise ValueError(
                    f"{sample.sample_id}: annotation is not inside a SyncG train tree"
                )


def _training_input_inventory_sha256(samples: Iterable[Any]) -> str:
    """Freeze row, image and annotation bytes for the formal training input."""

    inventory: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: str(item.sample_id)):
        image = Path(str(sample.image_path)).resolve()
        annotation_value = (sample.metadata or {}).get("annotation_path")
        annotation = (
            Path(str(annotation_value)).resolve() if annotation_value else None
        )
        if not image.is_file():
            raise FileNotFoundError(image)
        if annotation is None or not annotation.is_file():
            raise FileNotFoundError(
                annotation or f"{sample.sample_id}: annotation_path"
            )
        inventory.append(
            {
                "sample_id": str(sample.sample_id),
                "image_sha256": sha256_file(image),
                "annotation_sha256": sha256_file(annotation),
            }
        )
    return _canonical_json_sha256(inventory)


def _validate_pinned_training_input_inventory(
    inventory_sha256: str,
) -> str:
    """Fail closed unless the complete SyncG train content is the audited release."""

    if (
        not isinstance(inventory_sha256, str)
        or inventory_sha256
        != PINNED_SYNCG_TRAIN_INPUT_INVENTORY_SHA256
    ):
        raise ValueError(
            "SyncG train content inventory differs from the pinned release"
        )
    return inventory_sha256


def _model_state_health(state: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "encoder": False,
        "pivot_head": False,
        "vector_head": False,
        "angle_head": False,
        "log_variance_head": False,
    }
    floating_elements = 0
    floating_absolute_sum = 0.0
    tensors = 0
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            raise ValueError(f"initial model state item is not a tensor: {name}")
        tensors += 1
        for prefix in required:
            required[prefix] |= str(name).startswith(prefix + ".")
        if tensor.is_floating_point():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"initial model state is non-finite: {name}")
            floating_elements += int(tensor.numel())
            floating_absolute_sum += float(tensor.detach().float().abs().sum())
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise ValueError(f"initial checkpoint misses model heads: {missing}")
    if tensors == 0 or floating_elements == 0 or floating_absolute_sum <= 0.0:
        raise ValueError("initial checkpoint model state is empty or collapsed")
    return {
        "state_tensors": tensors,
        "floating_elements": floating_elements,
        "floating_absolute_sum": floating_absolute_sum,
        "required_heads_present": True,
    }


def _initialize_variance_head_for_supervision(
    model: torch.nn.Module,
    *,
    supervision_mode: str,
) -> None:
    """Give both expert-v3 arms the same semantically clean variance start."""

    if supervision_mode == "legacy":
        return
    if supervision_mode != "expert_v3":
        raise ValueError(f"unsupported supervision mode: {supervision_mode}")
    head = getattr(model, "log_variance_head", None)
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("model log_variance_head is not the expected Linear layer")
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)


def _validate_signed_v2_initialization(
    *,
    checkpoint_path: Path,
    verification_path: Path,
    manifest: Path,
    train_samples: list[Any],
    validation_samples: list[Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the exact v2 start point and its original group split."""

    checkpoint_path = checkpoint_path.resolve()
    verification_path = verification_path.resolve()
    summary_path = checkpoint_path.parent / "summary.json"
    for path in (checkpoint_path, verification_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if checkpoint_path.name != "best.pt":
        raise ValueError("formal continuation must initialize from signed best.pt")
    verification = _read_json(verification_path)
    summary = _read_json(summary_path)
    protocol = str(verification.get("protocol") or "")
    if protocol not in SIGNED_V2_VERIFICATION_PROTOCOLS:
        raise ValueError("unsupported v2 verification protocol")
    if verification.get("verified") is not True:
        raise ValueError("initial v2 run is not verified")
    verifier_source = (
        PROJECT_DIR / "experiments" / "verify_probabilistic_pivot_direction_run.py"
    )
    if protocol.endswith("_v1"):
        expected_verifier_hash = LEGACY_V1_VERIFIER_SOURCE_SHA256
        if verification.get("source_hash_protocol") is not None:
            raise ValueError("legacy v1 verification has unexpected hash metadata")
    else:
        if verification.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL:
            raise ValueError("v2 verification source hash protocol mismatch")
        expected_verifier_hash = sha256_source_file(verifier_source)
    if verification.get("verifier_source_sha256") != expected_verifier_hash:
        raise ValueError("initial verification source identity mismatch")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    summary_sha256 = sha256_file(summary_path)
    if verification.get("best_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("initial best checkpoint hash is not signed")
    if verification.get("summary_sha256") != summary_sha256:
        raise ValueError("initial summary hash is not signed")
    if summary.get("status") != "complete":
        raise ValueError("initial training summary is incomplete")
    if summary.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("initial summary is not the signed v2 direction protocol")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    signature = checkpoint.get("signature") or {}
    if (
        checkpoint.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
        or summary.get("signature") != signature
    ):
        raise ValueError("initial checkpoint/summary signature mismatch")
    if int(checkpoint.get("epoch", -1)) != int(verification.get("best_epoch", -2)):
        raise ValueError("initial checkpoint is not the signed best epoch")
    if int(summary.get("best_epoch", -1)) != int(checkpoint.get("epoch", -2)):
        raise ValueError("initial best epoch metadata mismatch")
    if int(signature.get("seed", -1)) != int(seed):
        raise ValueError("continuation seed must match the signed initialization seed")
    if signature.get("diagnostic_limit") is not None:
        raise ValueError("diagnostic v2 checkpoints cannot initialize a formal run")
    manifest_protocol = _protocol_path(manifest)
    expected_identity = {
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(manifest_protocol),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
    }
    for key, expected in expected_identity.items():
        if signature.get(key) != expected:
            raise ValueError(f"initial v2 data/split identity mismatch: {key}")
    if signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL:
        raise ValueError("initial v2 training protocol mismatch")
    model_source = PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
    if signature.get("model_source_sha256") != sha256_source_file(model_source):
        raise ValueError("signed v2 model source changed")
    health = _model_state_health(checkpoint.get("model_state") or {})
    identity = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "verification_protocol": protocol,
        "verification_source_sha256": expected_verifier_hash,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": summary_sha256,
        "best_epoch": int(checkpoint["epoch"]),
        "training_signature": signature,
        "model_state_health": health,
    }
    return checkpoint, identity


def _fusion_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "min_direct_precision": float(args.min_direct_precision),
        "direct_precision_cap": float(args.direct_precision_cap),
        "direct_precision_temperature": float(args.direct_precision_temperature),
        "bin_expert_power": float(args.bin_expert_power),
        "direct_expert_power": float(args.direct_expert_power),
        "posterior_bins": int(args.posterior_bins),
    }


def _resolve_posterior_source(args: argparse.Namespace) -> str:
    expected = (
        "legacy_angle_bin_only"
        if args.supervision_mode == "legacy"
        else "gpoe_fused"
    )
    requested = str(args.posterior_source)
    if requested == "auto":
        return expected
    if requested != expected:
        raise ValueError(
            f"{args.supervision_mode} supervision is semantically incompatible "
            f"with posterior source {requested}; expected {expected}"
        )
    return requested


def _posterior_prediction(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    args: argparse.Namespace,
):
    source = _resolve_posterior_source(args)
    if source == "legacy_angle_bin_only":
        # The signed v2 log-variance describes its already-fused mean ray, not
        # an independent direct expert.  A zero direct power prevents that
        # quantity from entering the transported distribution.
        return fuse_circular_experts(
            *outputs,
            min_direct_precision=float(args.min_direct_precision),
            direct_precision_cap=float(args.direct_precision_cap),
            direct_precision_temperature=1.0,
            bin_expert_power=1.0,
            direct_expert_power=0.0,
            posterior_bins=int(args.posterior_bins),
        )
    return fuse_circular_experts(*outputs, **_fusion_kwargs(args))


def _legacy_supervised_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    heatmap: torch.Tensor,
    direction: torch.Tensor,
    initial_signature: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return probabilistic_direction_loss(
        *outputs,
        heatmap,
        direction,
        pivot_weight=float(initial_signature["pivot_loss_weight"]),
        bin_weight=float(initial_signature["bin_loss_weight"]),
        vector_weight=float(initial_signature["vector_loss_weight"]),
        soft_target_sigma_bins=float(initial_signature["soft_target_sigma_bins"]),
    )


def _supervised_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    heatmap: torch.Tensor,
    direction: torch.Tensor,
    *,
    args: argparse.Namespace,
    initial_signature: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if args.supervision_mode == "legacy":
        return _legacy_supervised_loss(
            outputs,
            heatmap,
            direction,
            initial_signature,
        )
    if args.supervision_mode != "expert_v3":
        raise ValueError(f"unsupported supervision mode: {args.supervision_mode}")
    return projective_circular_supervised_loss_v3(
        *outputs,
        heatmap,
        direction,
        pivot_weight=float(initial_signature["pivot_loss_weight"]),
        direct_nll_weight=float(args.expert_direct_nll_weight),
        bin_ce_weight=float(args.expert_bin_ce_weight),
        fused_posterior_weight=float(args.expert_fused_posterior_weight),
        fused_mean_weight=float(args.expert_fused_mean_weight),
        overconfidence_weight=float(args.expert_overconfidence_weight),
        precision_barrier_weight=float(args.expert_precision_barrier_weight),
        soft_target_sigma_bins=float(initial_signature["soft_target_sigma_bins"]),
        **_fusion_kwargs(args),
    )


def _masked_transport(
    first_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    second_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    homography: torch.Tensor,
    *,
    image_size: int,
    heatmap_size: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    first = _posterior_prediction(first_outputs, args=args)
    second = _posterior_prediction(second_outputs, args=args)
    stride = float(image_size) / float(heatmap_size)
    consistency = bidirectional_projective_posterior_consistency(
        first.probabilities,
        second.probabilities,
        first.pivot_xy * stride,
        second.pivot_xy * stride,
        homography.float(),
        detach_pivot=True,
    )
    valid = consistency.valid & first.valid & second.valid
    rows = 0.5 * (consistency.forward_js + consistency.inverse_js)
    if bool(valid.any()):
        loss = torch.mean(rows[valid])
        forward = torch.mean(consistency.forward_js[valid])
        inverse = torch.mean(consistency.inverse_js[valid])
    else:
        loss = first.probabilities.sum() * 0.0
        forward = loss.detach()
        inverse = loss.detach()
    return loss, {
        "posterior_transport_forward_js": forward.detach(),
        "posterior_transport_inverse_js": inverse.detach(),
        "posterior_transport_valid_fraction": valid.float().mean().detach(),
        "first_fused_valid_fraction": first.valid.float().mean().detach(),
        "second_fused_valid_fraction": second.valid.float().mean().detach(),
        "mean_first_fused_resultant": first.resultant_length.mean().detach(),
        "mean_second_fused_resultant": second.resultant_length.mean().detach(),
        "mean_first_normalized_entropy": first.normalized_entropy.mean().detach(),
        "mean_second_normalized_entropy": second.normalized_entropy.mean().detach(),
    }


def _compose_objective(
    first_supervised: torch.Tensor,
    second_supervised: torch.Tensor,
    equivariance: torch.Tensor,
    posterior_transport: torch.Tensor,
    *,
    paired_supervision_weight: float,
    equivariance_weight: float,
    posterior_transport_weight: float,
) -> torch.Tensor:
    return (
        first_supervised
        + float(paired_supervision_weight) * second_supervised
        + float(equivariance_weight) * equivariance
        + float(posterior_transport_weight) * posterior_transport
    )


def _average_components(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(first) != set(second):
        raise ValueError("paired supervised component schemas differ")
    return {
        f"supervised_{name}": 0.5 * (first[name] + second[name])
        for name in sorted(first)
    }


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
    initial_signature: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    model.train()
    totals: dict[str, float] = {}
    samples = 0
    optimizer_steps = 0
    skipped_steps = 0
    image_size = int(initial_signature["image_size"])
    heatmap_size = int(initial_signature["heatmap_size"])
    progress = tqdm(
        loader,
        desc=f"PCT train {epoch}",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        paired_images = batch["paired_image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True)
        paired_heatmap = batch["paired_heatmap"].to(device, non_blocking=True)
        paired_direction = batch["paired_direction"].to(device, non_blocking=True)
        homography = batch["homography"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            joined_outputs = model(torch.cat((images, paired_images), dim=0))
        count = int(images.shape[0])
        first = tuple(value[:count] for value in joined_outputs)
        second = tuple(value[count:] for value in joined_outputs)
        first_loss, first_components = _supervised_loss(
            first,
            heatmap,
            direction,
            args=args,
            initial_signature=initial_signature,
        )
        second_loss, second_components = _supervised_loss(
            second,
            paired_heatmap,
            paired_direction,
            args=args,
            initial_signature=initial_signature,
        )
        mean_ray, mean_ray_components = equivariance_loss(
            first,
            second,
            homography,
            image_size=image_size,
            heatmap_size=heatmap_size,
            pivot_weight=float(initial_signature["equivariance_pivot_weight"]),
        )
        posterior, posterior_components = _masked_transport(
            first,
            second,
            homography,
            image_size=image_size,
            heatmap_size=heatmap_size,
            args=args,
        )
        loss = _compose_objective(
            first_loss,
            second_loss,
            mean_ray,
            posterior,
            paired_supervision_weight=float(
                initial_signature["paired_supervision_weight"]
            ),
            equivariance_weight=float(initial_signature["equivariance_weight"]),
            posterior_transport_weight=float(args.posterior_transport_weight),
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"epoch {epoch}: non-finite training objective")
        scaler.scale(loss).backward()
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < scale_before:
            skipped_steps += 1
        else:
            optimizer_steps += 1

        values: dict[str, torch.Tensor] = {
            "loss": loss.detach(),
            "first_supervised_loss": first_loss.detach(),
            "paired_supervised_loss": second_loss.detach(),
            "mean_ray_equivariance_loss": mean_ray.detach(),
            "posterior_transport_loss": posterior.detach(),
            "mean_perspective_degrees": batch["perspective_degrees"].mean(),
        }
        values.update(_average_components(first_components, second_components))
        values.update(
            {
                f"mean_ray_{name}": value
                for name, value in mean_ray_components.items()
            }
        )
        values.update(posterior_components)
        for name, value in values.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"epoch {epoch}: non-finite training component {name}"
                )
            totals[name] = totals.get(name, 0.0) + scalar * count
        samples += count
        progress.set_postfix(loss=f"{totals['loss'] / samples:.5f}")
    return {
        name: value / max(samples, 1)
        for name, value in sorted(totals.items())
    } | {
        "samples": samples,
        "batches": len(loader),
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }


def _interpolated_circular_nll(
    log_probabilities: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible private name for the shared paper score."""

    return periodic_linear_discrete_nll(log_probabilities, target_direction)


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
    args: argparse.Namespace,
    initial_signature: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, float] = {}
    angle_errors: list[float] = []
    selection_angle_errors: list[float] = []
    circular_nlls: list[float] = []
    selection_circular_nlls: list[float] = []
    pivot_errors: list[float] = []
    pivot_peaks: list[float] = []
    resultants: list[float] = []
    normalized_entropies: list[float] = []
    angle_stds: list[float] = []
    valid_count = 0
    prediction_invalid_count = 0
    nonfinite_output_count = 0
    samples = 0
    image_size = int(initial_signature["image_size"])
    heatmap_size = int(initial_signature["heatmap_size"])
    stride = float(image_size) / float(heatmap_size)
    for batch in tqdm(
        loader,
        desc="PCT validation",
        leave=False,
        dynamic_ncols=True,
    ):
        images = batch["image"].to(device, non_blocking=True)
        heatmap = batch["heatmap"].to(device, non_blocking=True)
        direction = batch["direction"].to(device, non_blocking=True).float()
        target_pivot = batch["pivot"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            outputs = model(images)
        supervised, components = _supervised_loss(
            outputs,
            heatmap,
            direction,
            args=args,
            initial_signature=initial_signature,
        )
        float_outputs = tuple(value.float() for value in outputs)
        posterior = _posterior_prediction(float_outputs, args=args)
        if _resolve_posterior_source(args) == "legacy_angle_bin_only":
            legacy = decode_probabilistic_pivot_direction(*float_outputs)
            predicted_direction = legacy.direction
            predicted_pivot = legacy.pivot_xy
            prediction_valid = legacy.valid
            predicted_angle = torch.atan2(
                predicted_direction[:, 1],
                predicted_direction[:, 0],
            )
            target_angle = torch.atan2(direction[:, 1], direction[:, 0])
            delta = circular_delta(predicted_angle, target_angle)
            nll = 0.5 * (
                delta.square() * torch.exp(-legacy.log_variance)
                + legacy.log_variance
            )
            selection_decoder = "legacy_v2_fused_mean"
        else:
            predicted_direction = posterior.mean_direction
            predicted_pivot = posterior.pivot_xy
            prediction_valid = posterior.valid
            nll = _interpolated_circular_nll(
                posterior.log_probabilities,
                direction,
            )
            selection_decoder = "gpoe_fused"
        angle = angular_error_degrees(predicted_direction, direction)
        pivot = (
            torch.linalg.vector_norm(predicted_pivot - target_pivot, dim=1)
            * stride
            / float(image_size)
        )
        valid = (
            prediction_valid
            & torch.isfinite(angle)
            & torch.isfinite(nll)
            & torch.isfinite(pivot)
        )
        finite_outputs = (
            torch.isfinite(angle)
            & torch.isfinite(nll)
            & torch.isfinite(pivot)
        )
        selection_angle = torch.where(
            valid,
            angle,
            torch.full_like(angle, 180.0),
        )
        selection_nll = torch.where(
            valid,
            nll,
            torch.full_like(nll, -math.log(1e-12)),
        )
        finite_pivot = torch.where(
            torch.isfinite(pivot),
            pivot,
            torch.full_like(pivot, math.sqrt(2.0)),
        )
        angle_errors.extend(angle[valid].cpu().tolist())
        selection_angle_errors.extend(selection_angle.cpu().tolist())
        circular_nlls.extend(nll[valid].cpu().tolist())
        selection_circular_nlls.extend(selection_nll.cpu().tolist())
        pivot_errors.extend(finite_pivot.cpu().tolist())
        pivot_peaks.extend(posterior.pivot_peak.cpu().tolist())
        resultants.extend(posterior.resultant_length.cpu().tolist())
        normalized_entropies.extend(
            posterior.normalized_entropy.cpu().tolist()
        )
        if selection_decoder == "legacy_v2_fused_mean":
            angle_stds.extend(legacy.angle_std_degrees.cpu().tolist())
        else:
            angle_stds.extend(posterior.angle_std_degrees.cpu().tolist())
        valid_count += int(valid.sum())
        prediction_invalid_count += int((~prediction_valid).sum())
        nonfinite_output_count += int((~finite_outputs).sum())
        count = int(images.shape[0])
        samples += count
        values = {"supervised_loss": supervised.detach()} | {
            f"supervised_{name}": value for name, value in components.items()
        }
        for name, value in values.items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"non-finite validation component {name}"
                )
            totals[name] = totals.get(name, 0.0) + scalar * count
    if samples == 0 or valid_count == 0:
        raise FloatingPointError("validation produced no valid direction predictions")
    errors = np.asarray(angle_errors, dtype=np.float64)
    selection_errors = np.asarray(selection_angle_errors, dtype=np.float64)
    nlls = np.asarray(circular_nlls, dtype=np.float64)
    selection_nlls = np.asarray(selection_circular_nlls, dtype=np.float64)
    pivots = np.asarray(pivot_errors, dtype=np.float64)
    coverage = valid_count / max(samples, 1)
    return {
        name: value / max(samples, 1)
        for name, value in sorted(totals.items())
    } | {
        "samples": samples,
        "valid_directions": valid_count,
        "invalid_directions": samples - valid_count,
        "prediction_invalid_count": prediction_invalid_count,
        "nonfinite_output_count": nonfinite_output_count,
        "direction_coverage": coverage,
        "angle_mae_degrees": float(np.mean(errors)) if errors.size else math.inf,
        "angle_median_degrees": (
            float(np.median(errors)) if errors.size else math.inf
        ),
        "angle_acc_1deg": float(np.mean(errors <= 1.0)) if errors.size else 0.0,
        "angle_acc_3deg": float(np.mean(errors <= 3.0)) if errors.size else 0.0,
        "angle_acc_5deg": float(np.mean(errors <= 5.0)) if errors.size else 0.0,
        "selection_angle_mae_degrees": (
            float(np.mean(selection_errors)) if selection_errors.size else math.inf
        ),
        "selection_nll": (
            float(np.mean(selection_nlls)) if selection_nlls.size else math.inf
        ),
        "conditional_selection_nll": (
            float(np.mean(nlls)) if nlls.size else math.inf
        ),
        "selection_decoder": selection_decoder,
        "pivot_mean_error_fraction": (
            float(np.mean(pivots)) if pivots.size else math.inf
        ),
        "pivot_median_error_fraction": (
            float(np.median(pivots)) if pivots.size else math.inf
        ),
        "mean_pivot_peak": float(np.mean(pivot_peaks)),
        "mean_fused_resultant": float(np.mean(resultants)),
        "mean_normalized_entropy": float(np.mean(normalized_entropies)),
        "mean_angle_std_degrees": float(np.mean(angle_stds)),
    }


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], generator: torch.Generator) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    if set(state) != required:
        raise ValueError("resume checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader_generator"])


def _selection_key(
    metrics: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    return (
        float(metrics["invalid_directions"]),
        float(metrics["selection_angle_mae_degrees"]),
        float(metrics["selection_nll"]),
        float(metrics["pivot_mean_error_fraction"]),
    )


def _load_committed_resume_journal(
    output_dir: Path,
    *,
    signature: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Load the latest immutable epoch journal without trusting sidecars."""

    journals = tuple(sorted(output_dir.glob("epoch_*.pt")))
    if not journals:
        raise FileNotFoundError(
            "resume requires at least one committed epoch_NNN.pt journal"
        )
    expected_names = [
        f"epoch_{index:03d}.pt" for index in range(1, len(journals) + 1)
    ]
    if [path.name for path in journals] != expected_names:
        raise ValueError("PCT epoch journals are not contiguous from epoch 1")
    checkpoint = torch.load(
        journals[-1],
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("signature") != signature:
        raise ValueError("PCT resume signature mismatch")
    if int(checkpoint.get("epoch", -1)) != len(journals):
        raise ValueError("latest PCT journal epoch does not match its filename")
    history = list(checkpoint.get("history") or [])
    if len(history) != int(checkpoint["epoch"]):
        raise ValueError("PCT resume journal history length mismatch")
    recomputed_best = min(
        (
            (_selection_key(record["validation"]), int(record["epoch"]))
            for record in history
        ),
        key=lambda item: item[0],
    )
    recorded_best = (
        tuple(float(value) for value in checkpoint.get("best_key") or ()),
        int(checkpoint.get("best_epoch", -1)),
    )
    if recomputed_best != recorded_best:
        raise ValueError("PCT resume journal best state is inconsistent")
    return checkpoint, journals


def _environment(
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "amp_enabled": amp_enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")
    finite_positive = (
        ("learning_rate", args.learning_rate),
        ("weight_decay", args.weight_decay),
        ("direct_precision_cap", args.direct_precision_cap),
        ("direct_precision_temperature", args.direct_precision_temperature),
        ("min_direct_precision", args.min_direct_precision),
    )
    for name, value in finite_positive:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    nonnegative = (
        args.posterior_transport_weight,
        args.bin_expert_power,
        args.direct_expert_power,
        args.expert_direct_nll_weight,
        args.expert_bin_ce_weight,
        args.expert_fused_posterior_weight,
        args.expert_fused_mean_weight,
        args.expert_overconfidence_weight,
        args.expert_precision_barrier_weight,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in nonnegative):
        raise ValueError("loss weights and expert powers must be finite/non-negative")
    if float(args.bin_expert_power) + float(args.direct_expert_power) <= 0.0:
        raise ValueError("the two expert powers cannot both be zero")
    if (
        args.supervision_mode == "expert_v3"
        and float(args.expert_precision_barrier_weight) <= 0.0
    ):
        raise ValueError("formal expert_v3 requires a positive two-sided barrier")
    if not 0.0 < float(args.min_direct_precision) < float(
        args.direct_precision_cap
    ):
        raise ValueError("direct precision bounds must satisfy 0 < min < cap")
    if args.posterior_bins < 8:
        raise ValueError("posterior-bins must be at least 8")
    _resolve_posterior_source(args)


def _training_signature(
    *,
    args: argparse.Namespace,
    manifest: Path,
    train_samples: list[Any],
    validation_samples: list[Any],
    initial_identity: Mapping[str, Any],
    amp_enabled: bool,
    training_input_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    initial = initial_identity["training_signature"]
    treatment = (
        "continued_control"
        if float(args.posterior_transport_weight) == 0.0
        else "pct_full"
    )
    posterior_source = _resolve_posterior_source(args)
    signature: dict[str, Any] = {
        "protocol": PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL,
        "treatment": treatment,
        "posterior_transport_weight": float(args.posterior_transport_weight),
        "supervision_mode": args.supervision_mode,
        "posterior_source": posterior_source,
        "selection_decoder": (
            "legacy_v2_fused_mean"
            if posterior_source == "legacy_angle_bin_only"
            else "gpoe_fused"
        ),
        "initial_variance_semantics": "legacy_fused_mean_angular_variance",
        "training_variance_semantics": (
            "legacy_fused_mean_angular_variance"
            if args.supervision_mode == "legacy"
            else "direct_angular_error_radians_squared_v1"
        ),
        "direct_variance_head_initialization": (
            "signed_v2_continuation"
            if args.supervision_mode == "legacy"
            else "zero_weight_unit_variance_bias_v1"
        ),
        "manifest_sha256": sha256_file(manifest),
        "manifest_protocol_sha256": sha256_file(_protocol_path(manifest)),
        "training_input_inventory_sha256": (
            training_input_inventory_sha256
            or _training_input_inventory_sha256(
                [*train_samples, *validation_samples]
            )
        ),
        "train_sample_ids_sha256": sample_ids_hash(train_samples),
        "validation_sample_ids_sha256": sample_ids_hash(validation_samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_groups": len({sample.group_id for sample in train_samples}),
        "validation_groups": len(
            {sample.group_id for sample in validation_samples}
        ),
        "initial_checkpoint_sha256": initial_identity["checkpoint_sha256"],
        "initial_verification_sha256": initial_identity["verification_sha256"],
        "initial_summary_sha256": initial_identity["summary_sha256"],
        "initial_best_epoch": initial_identity["best_epoch"],
        "initial_training_protocol": initial["protocol"],
        "initial_seed": int(initial["seed"]),
        "initial_train_sample_ids_sha256": initial["train_sample_ids_sha256"],
        "initial_validation_sample_ids_sha256": initial[
            "validation_sample_ids_sha256"
        ],
        "image_size": int(initial["image_size"]),
        "heatmap_size": int(initial["heatmap_size"]),
        "angle_bins": int(initial["angle_bins"]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": bool(amp_enabled),
        "grad_scaler_initial_scale": 512.0,
        "persistent_workers": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "posterior_transport_implementation": "dense_one_hot_splat_v1",
        "paired_supervision_weight": float(initial["paired_supervision_weight"]),
        "mean_ray_equivariance_weight": float(initial["equivariance_weight"]),
        "mean_ray_equivariance_pivot_weight": float(
            initial["equivariance_pivot_weight"]
        ),
        "legacy_pivot_loss_weight": float(initial["pivot_loss_weight"]),
        "legacy_bin_loss_weight": float(initial["bin_loss_weight"]),
        "legacy_vector_loss_weight": float(initial["vector_loss_weight"]),
        "soft_target_sigma_bins": float(initial["soft_target_sigma_bins"]),
        "posterior_bins": int(args.posterior_bins),
        "min_direct_precision": float(args.min_direct_precision),
        "direct_precision_cap": float(args.direct_precision_cap),
        "direct_precision_temperature": float(args.direct_precision_temperature),
        "bin_expert_power": float(args.bin_expert_power),
        "direct_expert_power": float(args.direct_expert_power),
        "expert_direct_nll_weight": float(args.expert_direct_nll_weight),
        "expert_bin_ce_weight": float(args.expert_bin_ce_weight),
        "expert_fused_posterior_weight": float(
            args.expert_fused_posterior_weight
        ),
        "expert_fused_mean_weight": float(args.expert_fused_mean_weight),
        "expert_overconfidence_weight": float(args.expert_overconfidence_weight),
        "expert_precision_barrier_weight": float(
            args.expert_precision_barrier_weight
        ),
        "dataset": {
            key: initial[key]
            for key in (
                "expansion",
                "scale_factor",
                "rotation_factor",
                "translation_factor",
                "heatmap_sigma",
                "perspective_probability",
                "max_perspective_degrees",
                "max_blur_sigma",
            )
        },
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
        "source_sha256": {
            "legacy_v2_model_dataset": sha256_source_file(
                PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
            ),
            "projective_circular_transport": sha256_source_file(
                PROJECT_DIR / "experiments" / "projective_circular_transport.py"
            ),
            "trainer": sha256_source_file(Path(__file__).resolve()),
            "training_data_utils": sha256_source_file(
                PROJECT_DIR / "experiments" / "vdn_baseline.py"
            ),
        },
        "scope_policy": (
            "verified SyncG official train manifest only; no public test, RPM, "
            "Pointer-10K, field, or confirmatory data"
        ),
        "selection_policy": (
            "train-group validation lexicographic: invalid direction count, "
            "all-denominator angle MAE (invalid=180deg), periodically "
            "interpolated mass NLL (invalid=-log(1e-12)), pivot error"
        ),
        "posterior_score_definition": (
            "negative_log_periodically_interpolated_discrete_target_mass_"
            "floor_1e-12_v1"
        ),
    }
    comparison_budget = {
        key: value
        for key, value in signature.items()
        if key not in {"treatment", "posterior_transport_weight"}
    }
    signature["comparison_budget"] = comparison_budget
    signature["comparison_budget_sha256"] = _canonical_json_sha256(
        comparison_budget
    )
    return signature


def main() -> None:
    args = parse_args()
    _validate_args(args)
    args.manifest = args.manifest.resolve()
    args.initial_checkpoint = args.initial_checkpoint.resolve()
    args.initial_verification = args.initial_verification.resolve()
    args.output_dir = args.output_dir.resolve()
    set_random_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = device.type == "cuda" and not args.no_amp

    samples, manifest_protocol = load_syncg_manifest(
        args.manifest,
        expected_split="train",
    )
    _validate_syncg_train_scope(args.manifest, samples)
    train_samples, validation_samples = grouped_train_val_split(
        samples,
        validation_fraction=0.10,
        seed=args.seed,
    )
    if {sample.group_id for sample in train_samples} & {
        sample.group_id for sample in validation_samples
    }:
        raise RuntimeError("training and validation groups overlap")
    initial_checkpoint, initial_identity = _validate_signed_v2_initialization(
        checkpoint_path=args.initial_checkpoint,
        verification_path=args.initial_verification,
        manifest=args.manifest,
        train_samples=train_samples,
        validation_samples=validation_samples,
        seed=args.seed,
    )
    initial_signature = initial_identity["training_signature"]
    if float(initial_signature["validation_fraction"]) != 0.10:
        raise ValueError("signed initialization did not use the formal 10% split")
    input_inventory_sha256 = _validate_pinned_training_input_inventory(
        _training_input_inventory_sha256(samples)
    )
    signature = _training_signature(
        args=args,
        manifest=args.manifest,
        train_samples=train_samples,
        validation_samples=validation_samples,
        initial_identity=initial_identity,
        amp_enabled=amp_enabled,
        training_input_inventory_sha256=input_inventory_sha256,
    )
    protected_outputs = tuple(
        args.output_dir / name for name in ("best.pt", "last.pt", "summary.json")
    )
    epoch_journals = tuple(sorted(args.output_dir.glob("epoch_*.pt")))
    if not args.resume and (
        any(path.exists() for path in protected_outputs) or epoch_journals
    ):
        raise FileExistsError(
            "refusing to overwrite an existing PCT run; use --resume only "
            "when its exact signature is unchanged"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_options = {
        "image_size": int(initial_signature["image_size"]),
        "heatmap_size": int(initial_signature["heatmap_size"]),
        "expansion": float(initial_signature["expansion"]),
        "scale_factor": float(initial_signature["scale_factor"]),
        "rotation_factor": float(initial_signature["rotation_factor"]),
        "translation_factor": float(initial_signature["translation_factor"]),
        "heatmap_sigma": float(initial_signature["heatmap_sigma"]),
        "perspective_probability": float(
            initial_signature["perspective_probability"]
        ),
        "max_perspective_degrees": float(
            initial_signature["max_perspective_degrees"]
        ),
        "max_blur_sigma": float(initial_signature["max_blur_sigma"]),
    }
    train_dataset = SyncGProbabilisticDirectionDataset(
        train_samples,
        training=True,
        **dataset_options,
    )
    validation_dataset = SyncGProbabilisticDirectionDataset(
        validation_samples,
        training=False,
        **dataset_options,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        # Recreating workers each epoch makes the saved generator state a
        # sufficient epoch-boundary resume identity.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(initial_signature["angle_bins"]),
        imagenet_pretrained=False,
    )
    model.load_state_dict(initial_checkpoint["model_state"], strict=True)
    _initialize_variance_head_for_supervision(
        model,
        supervision_mode=args.supervision_mode,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.learning_rate * 0.01,
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled,
        init_scale=512.0,
    )

    best_path = args.output_dir / "best.pt"
    last_path = args.output_dir / "last.pt"
    summary_path = args.output_dir / "summary.json"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_key = (math.inf, math.inf, math.inf, math.inf)
    best_epoch = 0
    elapsed_before = 0.0
    environment = _environment(device=device, amp_enabled=amp_enabled)
    if args.resume:
        checkpoint, journals = _load_committed_resume_journal(
            args.output_dir,
            signature=signature,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = list(checkpoint.get("history") or [])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_key = tuple(float(value) for value in checkpoint["best_key"])
        best_epoch = int(checkpoint["best_epoch"])
        _restore_rng(checkpoint.get("rng_state") or {}, generator)
        if summary_path.is_file():
            elapsed_before = float(
                _read_json(summary_path).get("elapsed_seconds", 0.0)
            )
        _atomic_copy(journals[-1], last_path)
        _atomic_copy(
            args.output_dir / f"epoch_{best_epoch:03d}.pt",
            best_path,
        )
        recovered_summary = {
            "protocol": PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL,
            "status": "running" if start_epoch <= args.epochs else "complete",
            "signature": signature,
            "manifest_protocol": manifest_protocol,
            "initialization": initial_identity,
            "best_epoch": best_epoch,
            "best_validation_invalid_directions": best_key[0],
            "best_validation_direction_coverage": (
                1.0 - best_key[0] / max(len(validation_samples), 1)
            ),
            "best_validation_angle_mae_degrees": best_key[1],
            "best_validation_selection_nll": best_key[2],
            "best_validation_pivot_error_fraction": best_key[3],
            "history": history,
            "environment": environment,
            "elapsed_seconds": elapsed_before,
        }
        _json_write(summary_path, recovered_summary)

    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            args=args,
            initial_signature=initial_signature,
            epoch=epoch,
        )
        validation_metrics = _validate(
            model,
            validation_loader,
            device=device,
            amp_enabled=amp_enabled,
            args=args,
            initial_signature=initial_signature,
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        candidate = _selection_key(validation_metrics)
        improved = candidate < best_key
        if improved:
            best_key = candidate
            best_epoch = epoch
        checkpoint = {
            "protocol": PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "rng_state": _capture_rng(generator),
            "history": history,
            "signature": signature,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
        }
        epoch_path = args.output_dir / f"epoch_{epoch:03d}.pt"
        if epoch_path.exists():
            raise FileExistsError(f"refusing to replace committed journal {epoch_path}")
        _torch_save(epoch_path, checkpoint)
        _atomic_copy(epoch_path, last_path)
        if improved:
            _atomic_copy(epoch_path, best_path)
        elapsed = elapsed_before + time.time() - started
        summary = {
            "protocol": PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL,
            "status": "running" if epoch < args.epochs else "complete",
            "signature": signature,
            "manifest_protocol": manifest_protocol,
            "initialization": initial_identity,
            "best_epoch": best_epoch,
            "best_validation_invalid_directions": best_key[0],
            "best_validation_direction_coverage": (
                1.0 - best_key[0] / max(len(validation_samples), 1)
            ),
            "best_validation_angle_mae_degrees": best_key[1],
            "best_validation_selection_nll": best_key[2],
            "best_validation_pivot_error_fraction": best_key[3],
            "history": history,
            "environment": environment,
            "elapsed_seconds": elapsed,
        }
        _json_write(summary_path, summary)
        print(
            f"epoch={epoch}/{args.epochs} treatment={signature['treatment']} "
            f"lr={learning_rate:.3e} train={train_metrics['loss']:.5f} "
            f"transport={train_metrics['posterior_transport_loss']:.5f} "
            f"valid={train_metrics['posterior_transport_valid_fraction']:.3f} "
            f"val_invalid={int(candidate[0])} "
            f"val_angle={candidate[1]:.4f}deg val_nll={candidate[2]:.5f} "
            f"val_pivot={candidate[3]:.5f} best={best_key[1]:.4f}deg@{best_epoch}",
            flush=True,
        )

    if not best_path.is_file() or not last_path.is_file():
        raise RuntimeError("PCT training did not produce best/last checkpoints")
    final_summary = _read_json(summary_path)
    final_summary.update(
        {
            "status": "complete",
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(last_path.resolve()),
            "last_checkpoint_sha256": sha256_file(last_path),
            "epoch_journals": [
                {
                    "epoch": index,
                    "path": str(
                        (
                            args.output_dir / f"epoch_{index:03d}.pt"
                        ).resolve()
                    ),
                    "sha256": sha256_file(
                        args.output_dir / f"epoch_{index:03d}.pt"
                    ),
                }
                for index in range(1, len(history) + 1)
            ],
            "optimizer_steps": sum(
                int(item["train"]["optimizer_steps"]) for item in history
            ),
            "skipped_optimizer_steps": sum(
                int(item["train"]["skipped_optimizer_steps"]) for item in history
            ),
        }
    )
    _json_write(summary_path, final_summary)
    print(summary_path)


if __name__ == "__main__":
    main()
