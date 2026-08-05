"""Fail-closed semantic comparison of two complete PCT training runs.

PyTorch checkpoint container bytes are not a reliable determinism test because
ZIP metadata and storage identifiers may differ across saves.  This utility
instead compares every persisted semantic value, including model, optimizer,
scheduler, scaler and RNG tensors, and reports stable content digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.train_projective_circular_transport_syncg import (
    PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL,
)


COMPARISON_PROTOCOL = "pct_complete_run_semantic_determinism_v1"
_JOURNAL_PATTERN = re.compile(r"^epoch_(\d{3})\.pt$")
_VOLATILE_SUMMARY_KEYS = frozenset(
    {
        "elapsed_seconds",
        "best_checkpoint",
        "best_checkpoint_sha256",
        "last_checkpoint",
        "last_checkpoint_sha256",
        "epoch_journals",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _digest_part(hasher: Any, tag: str, payload: bytes = b"") -> None:
    tag_bytes = tag.encode("utf-8")
    hasher.update(struct.pack(">Q", len(tag_bytes)))
    hasher.update(tag_bytes)
    hasher.update(struct.pack(">Q", len(payload)))
    hasher.update(payload)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.layout != torch.strided:
        raise TypeError(f"unsupported tensor layout: {value.layout}")
    flattened = value.detach().cpu().contiguous().reshape(-1)
    return flattened.view(torch.uint8).numpy().tobytes()


def _update_semantic_digest(hasher: Any, value: Any) -> None:
    if value is None:
        _digest_part(hasher, "none")
    elif isinstance(value, bool):
        _digest_part(hasher, "bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _digest_part(hasher, "int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _digest_part(hasher, "float64", struct.pack(">d", value))
    elif isinstance(value, str):
        _digest_part(hasher, "str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _digest_part(hasher, "bytes", value)
    elif torch.is_tensor(value):
        metadata = json.dumps(
            {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "layout": str(value.layout),
                "requires_grad": bool(value.requires_grad),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _digest_part(hasher, "tensor_metadata", metadata)
        _digest_part(hasher, "tensor_bytes", _tensor_bytes(value))
    elif isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        metadata = json.dumps(
            {
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _digest_part(hasher, "ndarray_metadata", metadata)
        _digest_part(hasher, "ndarray_bytes", contiguous.tobytes())
    elif isinstance(value, np.generic):
        _digest_part(hasher, f"numpy_scalar:{value.dtype.str}", value.tobytes())
    elif isinstance(value, Mapping):
        _digest_part(hasher, f"mapping:{type(value).__name__}")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        _digest_part(hasher, "mapping_length", str(len(keys)).encode("ascii"))
        for key in keys:
            _update_semantic_digest(hasher, key)
            _update_semantic_digest(hasher, value[key])
    elif isinstance(value, Sequence):
        _digest_part(hasher, f"sequence:{type(value).__name__}")
        _digest_part(hasher, "sequence_length", str(len(value)).encode("ascii"))
        for child in value:
            _update_semantic_digest(hasher, child)
    else:
        raise TypeError(f"unsupported persisted value type: {type(value)!r}")


def semantic_sha256(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_semantic_digest(hasher, value)
    return hasher.hexdigest()


def _float_equal(left: float, right: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return struct.pack(">d", left) == struct.pack(">d", right)


def first_semantic_mismatch(
    left: Any,
    right: Any,
    *,
    path: str = "$",
) -> str | None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return f"{path}: tensor/non-tensor type mismatch"
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or left.layout != right.layout
            or bool(left.requires_grad) != bool(right.requires_grad)
        ):
            return f"{path}: tensor metadata mismatch"
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            return f"{path}: tensor values differ"
        return None
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return f"{path}: ndarray/non-ndarray type mismatch"
        if left.dtype != right.dtype or left.shape != right.shape:
            return f"{path}: ndarray metadata mismatch"
        if not np.array_equal(left, right, equal_nan=True):
            return f"{path}: ndarray values differ"
        return None
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        if not (isinstance(left, np.generic) and isinstance(right, np.generic)):
            return f"{path}: NumPy scalar type mismatch"
        if left.dtype != right.dtype or left.tobytes() != right.tobytes():
            return f"{path}: NumPy scalar differs"
        return None
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return f"{path}: mapping/non-mapping type mismatch"
        if set(left) != set(right):
            return f"{path}: mapping keys differ"
        for key in sorted(left, key=lambda item: (type(item).__name__, repr(item))):
            mismatch = first_semantic_mismatch(
                left[key],
                right[key],
                path=f"{path}[{key!r}]",
            )
            if mismatch:
                return mismatch
        return None
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
        ):
            return f"{path}: sequence/non-sequence type mismatch"
        if type(left) is not type(right):
            return f"{path}: sequence types differ"
        if len(left) != len(right):
            return f"{path}: sequence lengths differ"
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            mismatch = first_semantic_mismatch(
                left_child,
                right_child,
                path=f"{path}[{index}]",
            )
            if mismatch:
                return mismatch
        return None
    if type(left) is not type(right):
        return f"{path}: scalar types differ ({type(left)!r} != {type(right)!r})"
    if isinstance(left, float):
        return None if _float_equal(left, right) else f"{path}: floats differ"
    return None if left == right else f"{path}: values differ"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a checkpoint mapping")
    if checkpoint.get("protocol") != PROJECTIVE_CIRCULAR_TRANSPORT_PROTOCOL:
        raise ValueError(f"{path} is not a signed PCT checkpoint")
    return checkpoint


def _journal_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"{run_dir}: no committed epoch journals")
    expected = [f"epoch_{index:03d}.pt" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"{run_dir}: epoch journals are not contiguous")
    if any(_JOURNAL_PATTERN.fullmatch(path.name) is None for path in paths):
        raise ValueError(f"{run_dir}: malformed epoch journal name")
    return paths


def _normalized_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{path} is not a JSON object")
    normalized = {
        key: value
        for key, value in summary.items()
        if key not in _VOLATILE_SUMMARY_KEYS
    }
    journals = summary.get("epoch_journals")
    if not isinstance(journals, list):
        raise ValueError(f"{path}: epoch_journals is missing")
    normalized["epoch_journal_epochs"] = [
        int(record["epoch"]) for record in journals
    ]
    return normalized


def _compare_value(label: str, left: Any, right: Any) -> dict[str, Any]:
    mismatch = first_semantic_mismatch(left, right, path=label)
    if mismatch:
        raise ValueError(mismatch)
    left_hash = semantic_sha256(left)
    right_hash = semantic_sha256(right)
    if left_hash != right_hash:
        raise RuntimeError(f"{label}: equal values produced unequal digests")
    return {
        "exact": True,
        "semantic_sha256": left_hash,
    }


def compare_runs(run_a: Path, run_b: Path) -> dict[str, Any]:
    run_a = run_a.resolve()
    run_b = run_b.resolve()
    if run_a == run_b:
        raise ValueError("determinism comparison requires two distinct run directories")
    for run_dir in (run_a, run_b):
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)

    summary_a = _normalized_summary(run_a)
    summary_b = _normalized_summary(run_b)
    last_a = _load_checkpoint(run_a / "last.pt")
    last_b = _load_checkpoint(run_b / "last.pt")
    best_a = _load_checkpoint(run_a / "best.pt")
    best_b = _load_checkpoint(run_b / "best.pt")
    journals_a = _journal_paths(run_a)
    journals_b = _journal_paths(run_b)
    if [path.name for path in journals_a] != [path.name for path in journals_b]:
        raise ValueError("the two runs have different committed epoch journals")

    journal_reports = []
    for left_path, right_path in zip(journals_a, journals_b):
        journal_reports.append(
            {
                "epoch": int(_JOURNAL_PATTERN.fullmatch(left_path.name).group(1)),
                **_compare_value(
                    f"journals.{left_path.stem}",
                    _load_checkpoint(left_path),
                    _load_checkpoint(right_path),
                ),
            }
        )

    components = {
        name: _compare_value(
            f"last.{name}",
            last_a.get(name),
            last_b.get(name),
        )
        for name in (
            "signature",
            "history",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_state",
            "best_key",
            "best_epoch",
        )
    }
    return {
        "protocol": COMPARISON_PROTOCOL,
        "status": "passed",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "epochs": len(journals_a),
        "summary_core": _compare_value(
            "summary_core",
            summary_a,
            summary_b,
        ),
        "last_checkpoint": _compare_value(
            "last_checkpoint",
            last_a,
            last_b,
        ),
        "best_checkpoint": _compare_value(
            "best_checkpoint",
            best_a,
            best_b,
        ),
        "components": components,
        "epoch_journals": journal_reports,
        "container_sha256_intentionally_not_used": True,
    }


def main() -> None:
    args = parse_args()
    report = compare_runs(args.run_a, args.run_b)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite determinism report: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
