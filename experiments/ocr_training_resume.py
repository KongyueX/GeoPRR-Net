"""Strict, atomic epoch-resume primitives for public OCR training.

The journal is deliberately trainer-agnostic.  Each caller supplies a
JSON-canonical signature binding the seed, frozen corpus, full configuration
and source hashes.  Loading a journal mutates the supplied model/optimizer only
after all non-state invariants have been checked.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    _require(isinstance(value, Mapping), "resume RNG state is not a mapping")
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        _require(key in value, f"resume RNG state is missing {key}")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    cuda_states = value["torch_cuda"]
    _require(isinstance(cuda_states, Sequence), "resume CUDA RNG state is invalid")
    if cuda_states:
        _require(torch.cuda.is_available(), "resume journal requires unavailable CUDA RNG")
        _require(
            len(cuda_states) == torch.cuda.device_count(),
            "resume CUDA device inventory drift",
        )
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace an in-progress journal/checkpoint.

    A failed write never damages the previous complete epoch journal.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_dict_clone(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_epoch_journal(
    path: Path,
    *,
    protocol: str,
    component: str,
    signature: Mapping[str, Any],
    completed_epoch: int,
    total_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Sequence[Mapping[str, Any]],
    best_state: Mapping[str, torch.Tensor],
    best_metric: float,
    elapsed_seconds: float,
) -> None:
    _require(completed_epoch >= 1, "cannot journal before the first complete epoch")
    _require(completed_epoch <= total_epochs, "journal epoch exceeds configured total")
    _require(len(history) == completed_epoch, "journal history/epoch mismatch")
    _require(math.isfinite(best_metric), "journal best metric is non-finite")
    _require(elapsed_seconds >= 0.0, "journal elapsed time is negative")
    value = {
        "protocol": protocol,
        "status": "in_progress",
        "component": component,
        "signature": dict(signature),
        "signature_sha256": canonical_sha256(signature),
        "completed_epoch": int(completed_epoch),
        "total_epochs": int(total_epochs),
        "model_state_dict": _state_dict_clone(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": [dict(row) for row in history],
        "best_state_dict": _state_dict_clone(best_state),
        "best_metric": float(best_metric),
        "elapsed_seconds": float(elapsed_seconds),
        "rng_state": capture_rng_state(),
    }
    atomic_torch(path, value)


def _validate_tensor_state(
    candidate: Mapping[str, Any], reference: Mapping[str, torch.Tensor], *, label: str
) -> None:
    _require(isinstance(candidate, Mapping), f"{label} is not a mapping")
    _require(set(candidate) == set(reference), f"{label} parameter inventory drift")
    for key, expected in reference.items():
        observed = candidate[key]
        _require(torch.is_tensor(observed), f"{label} tensor missing: {key}")
        _require(tuple(observed.shape) == tuple(expected.shape), f"{label} shape drift: {key}")
        _require(observed.dtype == expected.dtype, f"{label} dtype drift: {key}")


def load_epoch_journal(
    path: Path,
    *,
    expected_protocol: str,
    expected_component: str,
    expected_signature: Mapping[str, Any],
    expected_total_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    restore_rng: bool = True,
) -> dict[str, Any]:
    value = torch.load(
        Path(path).resolve(strict=True), map_location="cpu", weights_only=False
    )
    _require(isinstance(value, Mapping), "resume journal is not a mapping")
    _require(value.get("protocol") == expected_protocol, "resume journal protocol drift")
    _require(value.get("status") == "in_progress", "resume journal is not in progress")
    _require(value.get("component") == expected_component, "resume journal component drift")
    signature = value.get("signature")
    _require(isinstance(signature, Mapping), "resume journal signature missing")
    expected_hash = canonical_sha256(expected_signature)
    _require(
        value.get("signature_sha256") == canonical_sha256(signature),
        "resume journal embedded signature hash drift",
    )
    _require(
        canonical_sha256(signature) == expected_hash,
        "resume journal signature does not match seed/corpus/config/code",
    )
    completed_epoch = int(value.get("completed_epoch", 0))
    _require(1 <= completed_epoch <= expected_total_epochs, "resume epoch is out of range")
    _require(
        int(value.get("total_epochs", -1)) == expected_total_epochs,
        "resume total epoch drift",
    )
    history = value.get("history")
    _require(isinstance(history, list), "resume history is not a list")
    _require(len(history) == completed_epoch, "resume history/epoch mismatch")
    _require(
        all(int(row.get("epoch", -1)) == index for index, row in enumerate(history, 1)),
        "resume history epoch sequence drift",
    )
    model_state = value.get("model_state_dict")
    best_state = value.get("best_state_dict")
    reference = model.state_dict()
    _validate_tensor_state(model_state, reference, label="resume model state")
    _validate_tensor_state(best_state, reference, label="resume best state")
    optimizer_state = value.get("optimizer_state_dict")
    _require(isinstance(optimizer_state, Mapping), "resume optimizer state missing")
    best_metric = float(value.get("best_metric", math.nan))
    elapsed_seconds = float(value.get("elapsed_seconds", math.nan))
    _require(math.isfinite(best_metric), "resume best metric is non-finite")
    _require(
        math.isfinite(elapsed_seconds) and elapsed_seconds >= 0.0,
        "resume elapsed time is invalid",
    )
    rng_state = value.get("rng_state")
    _require(isinstance(rng_state, Mapping), "resume RNG state missing")

    # Mutate runtime state only after every structural/signature check above.
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(dict(optimizer_state))
    if restore_rng:
        restore_rng_state(rng_state)
    return {
        "completed_epoch": completed_epoch,
        "history": [dict(row) for row in history],
        "best_state": _state_dict_clone(best_state),
        "best_metric": best_metric,
        "elapsed_seconds": elapsed_seconds,
        "signature_sha256": expected_hash,
    }
