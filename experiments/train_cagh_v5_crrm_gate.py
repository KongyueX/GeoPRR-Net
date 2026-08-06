"""Public-only GroupKFold training scaffold for the deployable CRRM-V5 gate.

The learner consumes two physically separate JSONL rosters:

* a label-free deployable roster containing exactly 23 features, three frozen
  expert predictions/availability flags, and a production anchor; and
* an independent label roster containing only sample id, entity group id, and
  normalized target progress.

Formal runs require a third frozen input-protocol JSON that binds both files by
SHA-256 before any optimization.  GroupKFold produces one prediction per row
from a gate and robust normalizer that never saw that row's entity group.  A
separate terminal refit on all public rows is saved only for deployment.  No
field, test, confirmatory, sealed, xiangmu1, or xiangmu2 path is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from experiments.cagh_v5_crrm_gate import (
    CRRMV5SoftReliabilityGate,
    EXPERT_COUNT,
    EXPERT_NAMES,
    FEATURE_DIM,
    PROTOCOL as GATE_PROTOCOL,
    RobustMedianIQRNormalizer,
    crrm_v5_loss,
)
from experiments.fadr_multiseed_protocol import sha256_file
from experiments.vdn_baseline import sha256_source_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cagh_crrm_v5_public_groupkfold_training_v1"
PROTOCOL_PATH = PROJECT_ROOT / "experiments/cagh_v5_crrm_gate_public_protocol.json"
GATE_SOURCE = PROJECT_ROOT / "experiments/cagh_v5_crrm_gate.py"
OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_v5_crrm_gate\runs")
CACHE_ROOT = Path(r"C:\pointer_read\cagh_v5_crrm_gate\cache")
INPUT_PROTOCOL_NAME = "cagh_crrm_v5_public_input_bundle_v1"
FORBIDDEN_PATH_TOKENS = frozenset(
    {"field", "test", "confirmatory", "confirmation", "sealed", "xiangmu1", "xiangmu2"}
)
FEATURE_NAMES = (
    "mask_geometry_confidence",
    "mask_line_residual",
    "mask_center_distance",
    "mask_axis_length",
    "mask_foreground_fraction",
    "pepd_pivot_peak",
    "pepd_angle_std_degrees",
    "pepd_angle_entropy",
    "pepd_bin_resultant_length",
    "pepd_pointer_length",
    "pepd_direction_quality",
    "scalemark_start_peak",
    "scalemark_end_peak",
    "scalemark_start_entropy",
    "scalemark_end_entropy",
    "scalemark_endpoint_separation",
    "scalemark_tick_scale_disagreement",
    "scalemark_endpoint_coordinate_disagreement",
    "scalemark_endpoint_js_divergence",
    "scalemark_radius_reliability",
    "scalemark_arc_length_reliability",
    "scalemark_gate_confidence",
    "scalemark_uncertainty_score",
)
PREDICTION_KEYS = frozenset(
    {
        "sample_id",
        "features",
        "expert_progress",
        "expert_availability",
        "production_anchor_progress",
        "production_anchor_availability",
    }
)
LABEL_KEYS = frozenset({"sample_id", "group_id", "target_progress"})


@dataclass(frozen=True)
class GateBundle:
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    features: torch.Tensor
    expert_progress: torch.Tensor
    expert_availability: torch.Tensor
    anchor_progress: torch.Tensor
    anchor_availability: torch.Tensor
    target_progress: torch.Tensor
    identities: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.sample_ids)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_path(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    for part in resolved.parts:
        tokens = part.casefold().replace("-", "_").replace(".", "_").split("_")
        if any(token in FORBIDDEN_PATH_TOKENS for token in tokens):
            raise ValueError(f"{label} enters forbidden namespace: {part!r}")
    return resolved


def _json_loads(line: str, *, label: str) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-JSON constant {value!r}; use null")
    return json.loads(line, parse_constant=invalid_constant)


def _read_jsonl(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    resolved = _guard_path(path, label=label)
    _require(resolved.is_file(), f"missing {label}: {resolved}")
    rows: list[Mapping[str, Any]] = []
    with resolved.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = _json_loads(line, label=f"{label}:{line_number}")
            _require(isinstance(value, Mapping), f"{label}:{line_number} is not an object")
            rows.append(value)
    _require(bool(rows), f"{label} is empty")
    return rows


def _identifier(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty text")
    result = value.strip()
    _require(len(result) <= 256, f"{label} is unreasonably long")
    return result


def _optional_number(value: Any, *, label: str) -> float:
    if value is None:
        return float("nan")
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} must be numeric or null")
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite or null")
    return result


def _progress(value: Any, *, label: str, nullable: bool) -> float:
    result = _optional_number(value, label=label)
    if math.isnan(result):
        _require(nullable, f"{label} cannot be null")
        return result
    _require(0.0 <= result <= 1.0, f"{label} must be normalized to [0,1]")
    return result


def _feature(value: Any, *, label: str) -> float:
    result = _optional_number(value, label=label)
    return result


def _parse_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(set(row) == PREDICTION_KEYS,
                 f"prediction row {index} keys drift: {sorted(set(row) ^ PREDICTION_KEYS)}")
        sample_id = _identifier(row["sample_id"], label=f"prediction[{index}].sample_id")
        _require(sample_id not in parsed, f"duplicate prediction sample_id: {sample_id}")
        features = row["features"]
        _require(isinstance(features, list) and len(features) == FEATURE_DIM,
                 f"{sample_id}: features must contain {FEATURE_DIM} values")
        feature_values = [
            _feature(value, label=f"{sample_id}.features[{position}]")
            for position, value in enumerate(features)
        ]
        progress = row["expert_progress"]
        available = row["expert_availability"]
        _require(isinstance(progress, Mapping) and set(progress) == set(EXPERT_NAMES),
                 f"{sample_id}: expert_progress roster drift")
        _require(isinstance(available, Mapping) and set(available) == set(EXPERT_NAMES),
                 f"{sample_id}: expert_availability roster drift")
        progress_values: list[float] = []
        availability_values: list[bool] = []
        for expert in EXPERT_NAMES:
            flag = available[expert]
            _require(isinstance(flag, bool), f"{sample_id}.{expert}: availability must be bool")
            value = _progress(progress[expert], label=f"{sample_id}.{expert}", nullable=True)
            _require(not flag or math.isfinite(value),
                     f"{sample_id}.{expert}: available expert lacks a finite prediction")
            progress_values.append(value)
            availability_values.append(flag)
        anchor_available = row["production_anchor_availability"]
        _require(isinstance(anchor_available, bool),
                 f"{sample_id}: production_anchor_availability must be bool")
        anchor = _progress(
            row["production_anchor_progress"],
            label=f"{sample_id}.production_anchor_progress",
            nullable=True,
        )
        _require(not anchor_available or math.isfinite(anchor),
                 f"{sample_id}: available production anchor is non-finite")
        parsed[sample_id] = {
            "features": feature_values,
            "expert_progress": progress_values,
            "expert_availability": availability_values,
            "anchor_progress": anchor,
            "anchor_availability": anchor_available,
        }
    return parsed


def _parse_label_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(set(row) == LABEL_KEYS,
                 f"label row {index} keys drift: {sorted(set(row) ^ LABEL_KEYS)}")
        sample_id = _identifier(row["sample_id"], label=f"label[{index}].sample_id")
        _require(sample_id not in parsed, f"duplicate label sample_id: {sample_id}")
        parsed[sample_id] = {
            "group_id": _identifier(row["group_id"], label=f"{sample_id}.group_id"),
            "target_progress": _progress(
                row["target_progress"], label=f"{sample_id}.target_progress", nullable=False
            ),
        }
    return parsed


def _load_algorithm_protocol() -> Mapping[str, Any]:
    _require(PROTOCOL_PATH.is_file(), f"missing CRRM training protocol: {PROTOCOL_PATH}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _require(protocol.get("protocol") == PROTOCOL, "CRRM training protocol identity drift")
    _require(protocol.get("status") == "frozen_public_only", "CRRM training protocol is not frozen")
    _require(protocol["gate"]["protocol"] == GATE_PROTOCOL, "CRRM gate protocol drift")
    _require(tuple(protocol["feature_names"]) == FEATURE_NAMES, "23-feature schema drift")
    _require(tuple(protocol["expert_names"]) == EXPERT_NAMES, "expert roster drift")
    _require(sha256_file(GATE_SOURCE) == protocol["gate"]["source_sha256"],
             "CRRM gate source hash drift")
    return protocol


def _resolve_declared_path(value: Any, *, protocol_path: Path, label: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"{label}.path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = protocol_path.parent / path
    return _guard_path(path, label=label)


def _validate_input_protocol(
    path: Path,
    prediction_path: Path,
    label_path: Path,
    *,
    prediction_rows: int,
    label_rows: int,
    sample_ids_sha256: str,
) -> Mapping[str, Any]:
    resolved = _guard_path(path, label="CRRM input protocol")
    _require(resolved.is_file(), f"missing CRRM input protocol: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    _require(value.get("protocol") == INPUT_PROTOCOL_NAME, "CRRM input protocol identity drift")
    _require(value.get("status") == "frozen_before_gate_training",
             "CRRM input protocol is not frozen")
    scope = value.get("scope") or {}
    _require(scope.get("dataset") == "SyncG" and scope.get("split") == "train",
             "formal CRRM inputs are not restricted to SyncG/train")
    _require(scope.get("restricted_data_authorized") is False,
             "formal CRRM input protocol authorizes restricted data")
    _require(tuple(value.get("feature_names") or ()) == FEATURE_NAMES,
             "formal input feature order drift")
    _require(tuple(value.get("expert_names") or ()) == EXPERT_NAMES,
             "formal input expert roster drift")
    sources = value.get("sources") or {}
    for name, supplied, rows in (
        ("predictions", prediction_path, prediction_rows),
        ("labels", label_path, label_rows),
    ):
        record = sources.get(name) or {}
        declared = _resolve_declared_path(record.get("path"), protocol_path=resolved,
                                          label=f"input_protocol.{name}")
        _require(declared == supplied.resolve(), f"formal {name} path differs from freeze")
        _require(record.get("sha256") == sha256_file(supplied),
                 f"formal {name} SHA-256 drift")
        _require(int(record.get("rows", -1)) == rows, f"formal {name} row count drift")
        _require(record.get("sample_ids_sha256") == sample_ids_sha256,
                 f"formal {name} sample roster drift")
    return value


def load_bundle(
    prediction_path: Path,
    label_path: Path,
    *,
    input_protocol_path: Path | None,
    formal: bool,
) -> GateBundle:
    predictions_resolved = _guard_path(prediction_path, label="label-free predictions")
    labels_resolved = _guard_path(label_path, label="independent labels")
    _require(predictions_resolved != labels_resolved,
             "predictions and labels must be physically separate files")
    prediction_rows = _read_jsonl(predictions_resolved, label="label-free predictions")
    label_rows = _read_jsonl(labels_resolved, label="independent labels")
    predictions = _parse_prediction_rows(prediction_rows)
    labels = _parse_label_rows(label_rows)
    prediction_ids = set(predictions)
    label_ids = set(labels)
    _require(prediction_ids == label_ids,
             f"prediction/label overlap is not exact: predictions_only={len(prediction_ids-label_ids)}, "
             f"labels_only={len(label_ids-prediction_ids)}")
    ordered_ids = tuple(sorted(prediction_ids))
    sample_hash = canonical_sha256(ordered_ids)
    input_protocol = None
    if input_protocol_path is not None:
        input_protocol = _validate_input_protocol(
            input_protocol_path,
            predictions_resolved,
            labels_resolved,
            prediction_rows=len(prediction_rows),
            label_rows=len(label_rows),
            sample_ids_sha256=sample_hash,
        )
    _require(not formal or input_protocol is not None,
             "formal CRRM training requires --input-protocol")
    group_ids = tuple(labels[sample_id]["group_id"] for sample_id in ordered_ids)
    _require(len(set(group_ids)) >= 2, "CRRM training requires at least two entity groups")
    bundle = GateBundle(
        sample_ids=ordered_ids,
        group_ids=group_ids,
        features=torch.tensor(
            [predictions[sample_id]["features"] for sample_id in ordered_ids],
            dtype=torch.float32,
        ),
        expert_progress=torch.tensor(
            [predictions[sample_id]["expert_progress"] for sample_id in ordered_ids],
            dtype=torch.float32,
        ),
        expert_availability=torch.tensor(
            [predictions[sample_id]["expert_availability"] for sample_id in ordered_ids],
            dtype=torch.bool,
        ),
        anchor_progress=torch.tensor(
            [predictions[sample_id]["anchor_progress"] for sample_id in ordered_ids],
            dtype=torch.float32,
        ),
        anchor_availability=torch.tensor(
            [predictions[sample_id]["anchor_availability"] for sample_id in ordered_ids],
            dtype=torch.bool,
        ),
        target_progress=torch.tensor(
            [labels[sample_id]["target_progress"] for sample_id in ordered_ids],
            dtype=torch.float32,
        ),
        identities={
            "predictions": {
                "path": str(predictions_resolved),
                "sha256": sha256_file(predictions_resolved),
                "rows": len(prediction_rows),
            },
            "labels": {
                "path": str(labels_resolved),
                "sha256": sha256_file(labels_resolved),
                "rows": len(label_rows),
            },
            "input_protocol": None if input_protocol_path is None else {
                "path": str(Path(input_protocol_path).resolve()),
                "sha256": sha256_file(Path(input_protocol_path).resolve()),
            },
            "sample_ids_sha256": sample_hash,
            "group_ids_sha256": canonical_sha256(sorted(set(group_ids))),
        },
    )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: GateBundle) -> None:
    count = len(bundle)
    _require(bundle.features.shape == (count, FEATURE_DIM), "feature tensor shape drift")
    _require(bundle.expert_progress.shape == (count, EXPERT_COUNT),
             "expert progress tensor shape drift")
    _require(bundle.expert_availability.shape == (count, EXPERT_COUNT),
             "expert availability tensor shape drift")
    _require(bundle.anchor_progress.shape == (count,), "anchor tensor shape drift")
    _require(bundle.anchor_availability.shape == (count,), "anchor availability shape drift")
    _require(bundle.target_progress.shape == (count,), "target tensor shape drift")
    _require(bool(torch.isfinite(bundle.target_progress).all()), "targets are non-finite")
    _require(bool(((bundle.target_progress >= 0) & (bundle.target_progress <= 1)).all()),
             "targets escaped normalized range")
    finite_expert = torch.isfinite(bundle.expert_progress)
    _require(bool((~bundle.expert_availability | finite_expert).all()),
             "available expert contains a non-finite prediction")
    for expert_index, expert in enumerate(EXPERT_NAMES):
        _require(bool(bundle.expert_availability[:, expert_index].any()),
                 f"expert {expert!r} is never available")
    _require(bool(bundle.anchor_availability.any()), "production anchor is never available")


def synthetic_bundle(*, samples: int, groups: int, seed: int) -> GateBundle:
    _require(samples >= groups * 2 and groups >= 3, "synthetic smoke inventory is too small")
    rng = np.random.default_rng(seed)
    group_ids = tuple(f"synthetic_group_{index % groups:03d}" for index in range(samples))
    sample_ids = tuple(f"synthetic_{index:05d}" for index in range(samples))
    features = rng.normal(0.0, 1.0, size=(samples, FEATURE_DIM)).astype(np.float32)
    target = 1.0 / (1.0 + np.exp(-(0.7 * features[:, 0] - .4 * features[:, 4] +
                                  .25 * features[:, 9])))
    scales = np.stack(
        (
            .025 + .08 / (1.0 + np.exp(-features[:, 1])),
            .020 + .07 / (1.0 + np.exp(-features[:, 6])),
            .018 + .10 / (1.0 + np.exp(-features[:, 18])),
        ),
        axis=1,
    )
    prediction = np.clip(target[:, None] + rng.normal(size=(samples, EXPERT_COUNT)) * scales,
                         0.0, 1.0).astype(np.float32)
    availability = rng.random((samples, EXPERT_COUNT)) > np.asarray([.08, .12, .18])[None]
    # Explicitly exercise production-anchor and unavailable-fusion fallbacks.
    availability[::17] = False
    for expert_index in range(EXPERT_COUNT):
        availability[expert_index, expert_index] = True
    prediction[~availability] = np.nan
    anchor = np.clip(target + rng.normal(0.0, .075, size=samples), 0.0, 1.0).astype(np.float32)
    anchor_available = rng.random(samples) > .05
    missing = rng.random(features.shape) < .02
    features[missing] = np.nan
    return GateBundle(
        sample_ids=sample_ids,
        group_ids=group_ids,
        features=torch.from_numpy(features),
        expert_progress=torch.from_numpy(prediction),
        expert_availability=torch.from_numpy(availability),
        anchor_progress=torch.from_numpy(anchor),
        anchor_availability=torch.from_numpy(anchor_available),
        target_progress=torch.from_numpy(target.astype(np.float32)),
        identities={
            "synthetic": True,
            "generator": "crrm_correlated_heteroscedastic_smoke_v1",
            "seed": seed,
            "sample_ids_sha256": canonical_sha256(sample_ids),
            "group_ids_sha256": canonical_sha256(sorted(set(group_ids))),
        },
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _subset(tensor: torch.Tensor, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return tensor[torch.from_numpy(np.asarray(indices, dtype=np.int64))].to(device)


def fit_gate(
    bundle: GateBundle,
    train_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[CRRMV5SoftReliabilityGate, list[dict[str, Any]]]:
    _require(len(train_indices) > 0, "CRRM fold training partition is empty")
    seed_everything(seed)
    train_features_cpu = bundle.features[torch.from_numpy(train_indices)].float()
    normalizer = RobustMedianIQRNormalizer.from_training_features(
        train_features_cpu, minimum_iqr=args.minimum_iqr, clip=args.normalizer_clip
    )
    gate = CRRMV5SoftReliabilityGate(
        normalizer=normalizer,
        temperature=args.temperature,
        minimum_log_variance=args.minimum_log_variance,
        maximum_log_variance=args.maximum_log_variance,
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        gate.train()
        generator = torch.Generator().manual_seed(seed + epoch * 1009)
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        totals = {name: 0.0 for name in ("total", "expert_nll", "fused_huber", "anchor_regret")}
        samples_seen = 0
        for offset in range(0, len(order), args.batch_size):
            local = train_indices[order[offset:offset + args.batch_size]]
            features = _subset(bundle.features, local, device)
            expert_progress = _subset(bundle.expert_progress, local, device)
            expert_availability = _subset(bundle.expert_availability, local, device)
            anchor_progress = _subset(bundle.anchor_progress, local, device)
            anchor_availability = _subset(bundle.anchor_availability, local, device)
            target = _subset(bundle.target_progress, local, device)
            output = gate(
                features,
                expert_progress,
                expert_availability,
                production_anchor_progress=anchor_progress,
                production_anchor_availability=anchor_availability,
                mask_progress=expert_progress[:, 0],
                mask_availability=expert_availability[:, 0],
            )
            loss = crrm_v5_loss(
                output,
                target,
                expert_nll_weight=args.expert_nll_weight,
                fused_huber_weight=args.fused_huber_weight,
                anchor_regret_weight=args.anchor_regret_weight,
                huber_beta=args.huber_beta,
                anchor_regret_margin=args.anchor_regret_margin,
            )
            _require(bool(torch.isfinite(loss.total)), "CRRM loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(gate.network.parameters(), args.gradient_clip)
            optimizer.step()
            count = len(local)
            samples_seen += count
            for name in totals:
                totals[name] += float(getattr(loss, name).detach()) * count
        _require(samples_seen == len(train_indices), "CRRM epoch inventory drift")
        row = {"epoch": epoch, "samples": samples_seen} | {
            name: value / samples_seen for name, value in totals.items()
        }
        history.append(row)
        print(
            f"CRRM seed={seed} epoch={epoch}/{args.epochs} total={row['total']:.6f} "
            f"nll={row['expert_nll']:.6f} regret={row['anchor_regret']:.6f}",
            flush=True,
        )
    return gate, history


@torch.inference_mode()
def predict_indices(
    gate: CRRMV5SoftReliabilityGate,
    bundle: GateBundle,
    indices: np.ndarray,
    *,
    device: torch.device,
    fold: int,
) -> list[dict[str, Any]]:
    gate.eval(); rows: list[dict[str, Any]] = []
    for offset in range(0, len(indices), 512):
        local = indices[offset:offset + 512]
        features = _subset(bundle.features, local, device)
        expert_progress = _subset(bundle.expert_progress, local, device)
        expert_availability = _subset(bundle.expert_availability, local, device)
        anchor_progress = _subset(bundle.anchor_progress, local, device)
        anchor_availability = _subset(bundle.anchor_availability, local, device)
        output = gate(
            features,
            expert_progress,
            expert_availability,
            production_anchor_progress=anchor_progress,
            production_anchor_availability=anchor_availability,
            mask_progress=expert_progress[:, 0],
            mask_availability=expert_availability[:, 0],
        )
        for position, global_index in enumerate(local):
            available = bool(output.fused_available[position])
            target = float(bundle.target_progress[global_index])
            fused = float(output.fused_progress[position]) if available else None
            anchor_ok = bool(output.anchor_available[position])
            anchor = float(output.anchor_progress[position]) if anchor_ok else None
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "gate_protocol": GATE_PROTOCOL,
                    "sample_id": bundle.sample_ids[global_index],
                    "group_id": bundle.group_ids[global_index],
                    "fold": fold,
                    "target_progress": target,
                    "fused_progress": fused,
                    "fused_available": available,
                    "absolute_error": abs(fused - target) if fused is not None else 1.0,
                    "anchor_progress": anchor,
                    "anchor_available": anchor_ok,
                    "anchor_absolute_error": abs(anchor - target) if anchor is not None else 1.0,
                    "positive_anchor_regret": (
                        max(0.0, abs(fused - target) - abs(anchor - target))
                        if fused is not None and anchor is not None else 0.0
                    ),
                    "expert_progress": {
                        name: (float(output.expert_progress[position, expert_index])
                               if bool(output.expert_valid[position, expert_index]) else None)
                        for expert_index, name in enumerate(EXPERT_NAMES)
                    },
                    "expert_available": {
                        name: bool(output.expert_valid[position, expert_index])
                        for expert_index, name in enumerate(EXPERT_NAMES)
                    },
                    "weights": {
                        name: float(output.weights[position, expert_index])
                        for expert_index, name in enumerate(EXPERT_NAMES)
                    },
                    "log_variances": {
                        name: float(output.log_variances[position, expert_index])
                        for expert_index, name in enumerate(EXPERT_NAMES)
                    },
                    "fallback": {
                        "used_soft_fusion": bool(output.used_soft_fusion[position]),
                        "used_production_anchor": bool(output.used_production_anchor[position]),
                        "used_mask_fallback": bool(output.used_mask_fallback[position]),
                    },
                }
            )
    return rows


def _full_nmae(rows: Sequence[Mapping[str, Any]], value: str, availability: str) -> float:
    errors = []
    for row in rows:
        prediction = row[value]
        errors.append(
            abs(float(prediction) - float(row["target_progress"]))
            if bool(row[availability]) and prediction is not None else 1.0
        )
    return float(np.mean(errors))


def summarize_oof(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "OOF predictions are empty")
    regret = np.asarray([float(row["positive_anchor_regret"]) for row in rows])
    metrics: dict[str, Any] = {
        "samples": len(rows),
        "groups": len({str(row["group_id"]) for row in rows}),
        "coverage": sum(bool(row["fused_available"]) for row in rows) / len(rows),
        "full_denominator_nmae": _full_nmae(rows, "fused_progress", "fused_available"),
        "anchor_coverage": sum(bool(row["anchor_available"]) for row in rows) / len(rows),
        "anchor_full_denominator_nmae": _full_nmae(rows, "anchor_progress", "anchor_available"),
        "mean_positive_anchor_regret": float(regret.mean()),
        "positive_anchor_regret_fraction": float(np.mean(regret > 0.0)),
    }
    metrics["delta_nmae_vs_anchor"] = (
        metrics["full_denominator_nmae"] - metrics["anchor_full_denominator_nmae"]
    )
    expert_metrics = {}
    for expert in EXPERT_NAMES:
        errors = []
        available = 0
        for row in rows:
            value = row["expert_progress"][expert]
            valid = bool(row["expert_available"][expert]) and value is not None
            available += int(valid)
            errors.append(abs(float(value) - float(row["target_progress"])) if valid else 1.0)
        expert_metrics[expert] = {
            "coverage": available / len(rows),
            "full_denominator_nmae": float(np.mean(errors)),
        }
    metrics["experts"] = expert_metrics
    return metrics


def cross_fit(
    bundle: GateBundle,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    unique_groups = len(set(bundle.group_ids))
    _require(2 <= args.folds <= unique_groups, "fold count exceeds entity groups")
    splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    placeholder = np.zeros((len(bundle), 1), dtype=np.float32)
    groups = np.asarray(bundle.group_ids, dtype=object)
    oof_by_index: dict[int, dict[str, Any]] = {}
    fold_records: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(placeholder, groups=groups)
    ):
        train_groups = set(groups[train_indices].tolist())
        validation_groups = set(groups[validation_indices].tolist())
        _require(not train_groups.intersection(validation_groups),
                 f"fold {fold} entity-group overlap")
        _require(not set(train_indices).intersection(set(validation_indices)),
                 f"fold {fold} sample overlap")
        gate, history = fit_gate(
            bundle,
            train_indices,
            args=args,
            device=device,
            seed=args.seed + fold * 10_000,
        )
        rows = predict_indices(
            gate, bundle, validation_indices, device=device, fold=fold
        )
        for global_index, row in zip(validation_indices, rows, strict=True):
            _require(int(global_index) not in oof_by_index, "duplicate OOF prediction")
            oof_by_index[int(global_index)] = row
        fold_records.append(
            {
                "fold": fold,
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "train_group_ids_sha256": canonical_sha256(sorted(train_groups)),
                "validation_group_ids_sha256": canonical_sha256(sorted(validation_groups)),
                "group_overlap": 0,
            }
        )
        histories.append({"fold": fold, "epochs": history})
        del gate
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _require(set(oof_by_index) == set(range(len(bundle))), "OOF inventory is incomplete")
    return [oof_by_index[index] for index in range(len(bundle))], fold_records, histories


def _save_final_checkpoint(
    path: Path,
    gate: CRRMV5SoftReliabilityGate,
    history: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "gate_protocol": GATE_PROTOCOL,
        "status": "complete",
        "role": "terminal all-public refit for deployment; not used for OOF metrics",
        "feature_names": FEATURE_NAMES,
        "expert_names": EXPERT_NAMES,
        "history": list(history),
        "validation": validation,
        "gate_state": {name: tensor.detach().cpu() for name, tensor in gate.state_dict().items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--input-protocol", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261406)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--minimum-log-variance", type=float, default=-12.0)
    parser.add_argument("--maximum-log-variance", type=float, default=6.0)
    parser.add_argument("--minimum-iqr", type=float, default=1e-6)
    parser.add_argument("--normalizer-clip", type=float, default=12.0)
    parser.add_argument("--expert-nll-weight", type=float, default=1.0)
    parser.add_argument("--fused-huber-weight", type=float, default=2.0)
    parser.add_argument("--anchor-regret-weight", type=float, default=.5)
    parser.add_argument("--huber-beta", type=float, default=.02)
    parser.add_argument("--anchor-regret-margin", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--synthetic-samples", type=int, default=72)
    parser.add_argument("--synthetic-groups", type=int, default=12)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-synthetic", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def _validate_hyperparameters(args: argparse.Namespace) -> None:
    _require(args.folds >= 2 and args.epochs > 0 and args.batch_size > 0,
             "folds/epochs/batch-size are invalid")
    positives = (
        args.learning_rate, args.temperature, args.minimum_iqr,
        args.normalizer_clip, args.huber_beta, args.gradient_clip,
    )
    _require(all(math.isfinite(value) and value > 0 for value in positives),
             "positive CRRM hyperparameter is invalid")
    _require(math.isfinite(args.weight_decay) and args.weight_decay >= 0,
             "weight decay is invalid")
    _require(args.minimum_log_variance < args.maximum_log_variance,
             "log-variance bounds are reversed")


def validation_record(
    algorithm: Mapping[str, Any],
    bundle: GateBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "gate_protocol": GATE_PROTOCOL,
        "status": "validated",
        "mode": "synthetic_smoke" if args.smoke_synthetic else (
            "formal" if args.run_formal else "validate_only"
        ),
        "scope": {
            "samples": len(bundle),
            "groups": len(set(bundle.group_ids)),
            "field_samples_read": 0,
            "test_samples_read": 0,
            "confirmatory_samples_read": 0,
            "xiangmu_samples_read": 0,
        },
        "schema": {
            "feature_names": FEATURE_NAMES,
            "feature_dim": FEATURE_DIM,
            "expert_names": EXPERT_NAMES,
            "label_free_prediction_keys": sorted(PREDICTION_KEYS),
            "independent_label_keys": sorted(LABEL_KEYS),
        },
        "cross_fit": {
            "implementation": "sklearn.model_selection.GroupKFold",
            "folds": args.folds,
            "shuffle": True,
            "random_state": args.seed,
            "normalizer_fit": "fold-training rows only",
            "checkpoint_selection": "none; fixed terminal epoch",
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "expert_nll_weight": args.expert_nll_weight,
            "fused_huber_weight": args.fused_huber_weight,
            "anchor_regret_weight": args.anchor_regret_weight,
            "anchor_regret_margin": args.anchor_regret_margin,
            "huber_beta": args.huber_beta,
        },
        "inputs": bundle.identities,
        "input_sha256": {
            "algorithm_protocol": sha256_file(PROTOCOL_PATH),
            "gate_source": sha256_file(GATE_SOURCE),
            "trainer_source": sha256_source_file(Path(__file__).resolve()),
        },
        "algorithm_snapshot": algorithm,
    }


def run(args: argparse.Namespace) -> Path:
    _validate_hyperparameters(args)
    algorithm = _load_algorithm_protocol()
    if args.smoke_synthetic:
        args.folds = 3
        args.epochs = 2
        args.batch_size = 16
        args.device = "cpu"
        bundle = synthetic_bundle(
            samples=args.synthetic_samples, groups=args.synthetic_groups, seed=args.seed
        )
        _validate_bundle(bundle)
    else:
        _require(args.predictions is not None and args.labels is not None,
                 "--predictions and --labels are required")
        bundle = load_bundle(
            args.predictions,
            args.labels,
            input_protocol_path=args.input_protocol,
            formal=args.run_formal,
        )
    output = Path(args.output_dir) if args.output_dir else (
        OUTPUT_ROOT / ("synthetic_smoke" if args.smoke_synthetic else f"seed_{args.seed}")
    )
    output = _guard_path(output, label="CRRM output")
    cache = _guard_path(Path(args.cache_root), label="CRRM cache")
    _require(output != Path(output.anchor) and cache != Path(cache.anchor),
             "broad CRRM output/cache root rejected")
    if args.run_formal:
        frozen = algorithm["training"]
        _require(args.seed in frozen["seeds"], "formal CRRM seed outside frozen list")
        for key in (
            "folds", "epochs", "batch_size", "learning_rate", "weight_decay",
            "expert_nll_weight", "fused_huber_weight", "anchor_regret_weight",
            "anchor_regret_margin", "huber_beta",
        ):
            _require(math.isclose(float(getattr(args, key)), float(frozen[key])),
                     f"formal CRRM hyperparameter drift: {key}")
        _require(not (output / "summary.json").exists(), "formal CRRM run already exists")
    record = validation_record(algorithm, bundle, args)
    validation_path = output / "validation.json"
    atomic_json(validation_path, record)
    cache_signature = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "sample_ids_sha256": bundle.identities["sample_ids_sha256"],
        "group_ids_sha256": bundle.identities["group_ids_sha256"],
        "samples": len(bundle),
        "groups": len(set(bundle.group_ids)),
        "labels_cached": False,
        "features_cached": False,
    }
    atomic_json(cache / f"roster_{len(bundle)}_{len(set(bundle.group_ids))}.json", cache_signature)
    if args.validate_only:
        print(validation_path, flush=True)
        return validation_path

    device = torch.device(args.device)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
        _require("NVIDIA" in torch.cuda.get_device_name(device).upper(),
                 "CRRM CUDA device is not NVIDIA")
    started = time.time()
    oof_rows, folds, fold_histories = cross_fit(bundle, args=args, device=device)
    oof_metrics = summarize_oof(oof_rows)
    oof_path = output / "oof_predictions.jsonl"
    folds_path = output / "folds.json"
    atomic_jsonl(oof_path, oof_rows)
    atomic_json(folds_path, {"schema_version": 1, "protocol": PROTOCOL, "folds": folds})

    all_indices = np.arange(len(bundle), dtype=np.int64)
    final_gate, final_history = fit_gate(
        bundle,
        all_indices,
        args=args,
        device=device,
        seed=args.seed + 900_000,
    )
    checkpoint_path = output / "final_refit.pt"
    checkpoint_hash = _save_final_checkpoint(
        checkpoint_path, final_gate, final_history, record
    )
    summary = {
        **record,
        "status": "complete",
        "oof_metrics": oof_metrics,
        "folds": folds,
        "fold_histories": fold_histories,
        "final_refit_history": final_history,
        "artifacts": {
            "oof_predictions": str(oof_path),
            "oof_predictions_sha256": sha256_file(oof_path),
            "folds": str(folds_path),
            "folds_sha256": sha256_file(folds_path),
            "final_refit": str(checkpoint_path),
            "final_refit_sha256": checkpoint_hash,
        },
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    print(summary_path, flush=True)
    return summary_path


if __name__ == "__main__":
    run(parse_args())
