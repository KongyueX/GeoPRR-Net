"""Train-only Frozen-PEPD shared spatial mask--geometry posterior probe.

Validation and fit-cache modes access algorithm_fit only.  The formal mode is
explicit and opens algorithm_selection once, after the fixed fifth epoch.
No public/test/field/sealed/confirmatory source is accepted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from experiments.fadr_multiseed_protocol import assert_train_only_path
from experiments.geopepd_progress import GeoPEPDProgressFusionNet
from experiments.probabilistic_pivot_direction import (
    decode_probabilistic_pivot_direction,
)
from experiments.train_geopepd_progress_probe import (
    ProgressProbeRecord,
    bind_mask_geometry_progress,
    progress_soft_targets,
)
from experiments.train_geopepd_progress_v2_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_FIT_FEATURES,
    DEFAULT_FIT_MANIFEST,
    DEFAULT_RAW,
    DEFAULT_SELECTION_FEATURES,
    EXPECTED_TRANSPORT_SCHEMA,
    PROTOCOL as V2_PROTOCOL,
    ProgressProbeRecordV2,
    _progress_inputs_v2,
    bind_transport_runtime_features,
    transport_schema_sha256,
)
from experiments.train_geopepd_train_only_probe import (
    EXPECTED_FIT,
    EXPECTED_SELECTION,
    _EncoderDataset,
    _by_id,
    _read_jsonl,
    _records,
    _write_json,
    _write_jsonl,
    grouped_bootstrap_delta,
    sha256_file,
)
from experiments.uhpf_shared_mask_geometry import (
    SharedMaskGeometryPosteriorHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "uhpf_shared_mask_geometry_train_only_probe_v1"
SEED = 20260805
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEFAULT_STAGE_A_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/runs/geopepd_progress_train_only_probe_v2/stage_a_last.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/runs/uhpf_shared_mask_geometry_train_only_probe_v1"
)
VISUAL_BASELINE_NMAE = 0.3636912228413731
FORBIDDEN_INPUT_NAMES = frozenset(
    {
        "calibrated_progress",
        "calibrated_prediction",
        "fadr_progress",
        "fadr_prediction",
        "transformer_vector_progress_abs",
        "transformer_vector_angle_abs_fraction",
    }
)
_SAMPLE_ID_PATTERN = re.compile(r'"sample_id"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-checkpoint", type=Path, default=DEFAULT_STAGE_A_CHECKPOINT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--fit-manifest", type=Path, default=DEFAULT_FIT_MANIFEST)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--raw-clean", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--fit-features", type=Path, default=DEFAULT_FIT_FEATURES)
    parser.add_argument("--selection-features", type=Path, default=DEFAULT_SELECTION_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-spatial-cache", action="store_true")
    mode.add_argument("--run-formal", action="store_true")
    return parser.parse_args()


def _read_jsonl_subset(
    path: Path, allowed_ids: set[str], *, label: str
) -> list[dict[str, Any]]:
    """Decode only allowed algorithm-scope rows from a shared JSONL cache."""

    path = Path(path).resolve(strict=True)
    assert_train_only_path(path, label=label)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            match = _SAMPLE_ID_PATTERN.search(line)
            if match is None:
                raise ValueError(f"{label}:{number} has no sample_id token")
            sample_id = match.group(1)
            if sample_id not in allowed_ids:
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or sample_id in seen:
                raise ValueError(f"{label}:{number} invalid/duplicate selected row")
            seen.add(sample_id)
            rows.append(value)
    if seen != allowed_ids:
        raise ValueError(f"{label} selected inventory differs from manifest")
    return rows


def _mask_paths(annotation_paths: Sequence[str]) -> list[Path]:
    """Resolve masks from one directory inventory, not one directory scan per row."""

    annotations = [Path(value).resolve(strict=True) for value in annotation_paths]
    roots: set[Path] = set()
    for annotation in annotations:
        if (
            annotation.parent.name.casefold() != "train"
            or annotation.parent.parent.name.casefold() != "annotations"
        ):
            raise ValueError(f"only train annotations are accepted: {annotation}")
        roots.add(annotation.parents[2])
    indices: dict[Path, dict[str, list[Path]]] = {}
    for root in roots:
        directory = (root / "masks/train").resolve(strict=True)
        if directory.name.casefold() != "train" or directory.parent.name.casefold() != "masks":
            raise ValueError("resolved mask directory escaped train split")
        index: dict[str, list[Path]] = {}
        for path in directory.iterdir():
            if path.is_file():
                index.setdefault(path.stem, []).append(path.resolve())
        indices[root] = index
    masks: list[Path] = []
    for annotation in annotations:
        candidates = indices[annotation.parents[2]].get(annotation.stem, [])
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"expected one train pointer mask for {annotation.stem}"
            )
        masks.append(candidates[0])
    return masks


def load_scope_records(
    args: argparse.Namespace, *, scope: str
) -> tuple[list[ProgressProbeRecordV2], list[Path], dict[str, str]]:
    if scope == "algorithm_fit":
        manifest_path = Path(args.fit_manifest)
        feature_path = Path(args.fit_features)
        expected = EXPECTED_FIT
    elif scope == "algorithm_selection":
        manifest_path = Path(args.evaluation_manifest)
        feature_path = Path(args.selection_features)
        expected = EXPECTED_SELECTION
    else:
        raise ValueError(f"unsupported scope: {scope}")
    paths = {
        "checkpoint": Path(args.checkpoint).resolve(strict=True),
        f"{scope}_manifest": manifest_path.resolve(strict=True),
        "raw_clean": Path(args.raw_clean).resolve(strict=True),
        f"{scope}_features": feature_path.resolve(strict=True),
    }
    for label, path in paths.items():
        assert_train_only_path(path, label=label)
    if paths["checkpoint"] != DEFAULT_CHECKPOINT.resolve():
        raise ValueError("probe accepts only the frozen PEPD checkpoint identity")
    manifest = _read_jsonl(paths[f"{scope}_manifest"], label=f"{scope}_manifest")
    if any(str(row.get("split") or "") != "train" for row in manifest):
        raise ValueError(f"{scope} contains a non-train source row")
    features = _read_jsonl(paths[f"{scope}_features"], label=f"{scope}_features")
    ids = {str(row["sample_id"]) for row in manifest}
    raw = _read_jsonl_subset(paths["raw_clean"], ids, label=f"{scope}_raw_subset")
    base = _records(
        manifest,
        features,
        _by_id(raw),
        expected_scope=scope,
        expected_inventory=expected,
    )
    progress = bind_mask_geometry_progress(base, features, expected_scope=scope)
    records = bind_transport_runtime_features(progress, features, expected_scope=scope)
    manifest_by_id = _by_id(manifest)
    masks = _mask_paths(
        [
            str(manifest_by_id[record.sample_id]["metadata"]["annotation_path"])
            for record in records
        ]
    )
    identities = {label: sha256_file(path) for label, path in paths.items()}
    identities[f"{scope}_sample_ids_sha256"] = hashlib.sha256(
        "\n".join(record.sample_id for record in records).encode("utf-8")
    ).hexdigest()
    return records, masks, identities


def load_visual_model(
    checkpoint: Path, device: torch.device
) -> tuple[GeoPEPDProgressFusionNet, dict[str, Any]]:
    checkpoint = Path(checkpoint).resolve(strict=True)
    assert_train_only_path(checkpoint, label="v2_stage_a_checkpoint")
    if checkpoint != DEFAULT_STAGE_A_CHECKPOINT.resolve():
        raise ValueError("probe accepts only the frozen v2 Stage-A checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("protocol") != V2_PROTOCOL
        or payload.get("seed") != SEED
        or payload.get("transport_schema_sha256") != transport_schema_sha256()
        or tuple(payload.get("transport_schema") or ()) != EXPECTED_TRANSPORT_SCHEMA
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError("v2 Stage-A checkpoint protocol/schema drifted")
    model = GeoPEPDProgressFusionNet(imagenet_pretrained=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device), {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "strict_load": True,
    }


@torch.inference_mode()
def spatial_encoder_cache(
    visual: GeoPEPDProgressFusionNet,
    records: Sequence[ProgressProbeRecordV2],
    path: Path,
    *,
    scope: str,
    checkpoint_sha256: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    path = Path(path)
    assert_train_only_path(path.resolve(), label="spatial_cache")
    sample_ids = [record.sample_id for record in records]
    signature = {
        "protocol": PROTOCOL,
        "scope": scope,
        "checkpoint_sha256": checkpoint_sha256,
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sample_ids).encode("utf-8")
        ).hexdigest(),
        "shape": [len(records), 512, 8, 8],
        "dtype": "float16",
    }
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, Mapping)
            and payload.get("signature") == signature
            and payload.get("sample_ids") == sample_ids
            and isinstance(payload.get("encoder_spatial"), torch.Tensor)
            and payload["encoder_spatial"].shape == (len(records), 512, 8, 8)
            and bool(torch.isfinite(payload["encoder_spatial"]).all())
        ):
            return payload["encoder_spatial"].half()
        raise ValueError(f"stale/incompatible spatial cache: {path}")
    loader = DataLoader(
        _EncoderDataset(records),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=device.type == "cuda",
    )
    output = torch.empty((len(records), 512, 8, 8), dtype=torch.float16)
    visual.eval()
    for images, indices in loader:
        spatial = visual.encoder(images.to(device, non_blocking=True)).float().cpu()
        if spatial.shape[1:] != (512, 8, 8):
            raise ValueError(f"unexpected PEPD spatial shape: {tuple(spatial.shape)}")
        output[indices.long()] = spatial.half()
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("spatial encoder cache contains non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"signature": signature, "sample_ids": sample_ids, "encoder_spatial": output},
        temporary,
    )
    os.replace(temporary, path)
    return output


def load_aligned_masks(
    records: Sequence[ProgressProbeRecordV2], mask_paths: Sequence[Path]
) -> torch.Tensor:
    masks = torch.empty((len(records), 1, 64, 64), dtype=torch.float32)
    for index, (record, path) in enumerate(zip(records, mask_paths, strict=True)):
        source = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError(f"failed to read pointer mask: {path}")
        crop = cv2.warpAffine(
            source,
            record.affine,
            (256, 256),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        reduced = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        masks[index, 0] = torch.from_numpy((reduced > 31).astype(np.float32))
    return masks


@torch.inference_mode()
def frozen_components(
    visual: GeoPEPDProgressFusionNet,
    spatial: torch.Tensor,
    tensors: Sequence[torch.Tensor],
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    (
        geometry,
        mgc,
        start,
        angle_range,
        affine,
        geometry_available,
        reference_available,
        _,
        _,
        transport,
    ) = tensors
    pooled = F.adaptive_avg_pool2d(spatial.float(), 1).flatten(1)
    outputs = visual.forward_from_encoder_pooled(
        pooled,
        geometry,
        mgc,
        start,
        angle_range,
        affine,
        geometry_available,
        reference_available,
        transport_runtime_features=transport,
    )
    modules = tuple(visual.pivot_head.children())
    pivot_features = spatial.float()
    for module in modules[:-1]:
        pivot_features = module(pivot_features)
    pivot_logits = modules[-1](pivot_features)
    direction = decode_probabilistic_pivot_direction(
        pivot_logits,
        outputs.direction_raw,
        outputs.visual_angle_logits,
        outputs.visual_log_variance_raw,
    ).direction
    return outputs, pivot_features, pivot_logits, direction


def _weighted_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return value.new_zeros(())
    selected = weight.float()[mask]
    return (value.float()[mask] * selected).sum() / selected.sum().clamp_min(1e-8)


def probe_loss(
    outputs: Any,
    target_progress: torch.Tensor,
    target_mask: torch.Tensor,
    group_weight: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target_progress.float().clamp(0.0, 1.0)
    valid = outputs.valid.bool() & torch.isfinite(target_progress)
    geometry_valid = outputs.geometry_available.bool() & valid
    soft = progress_soft_targets(
        target,
        progress_bins=outputs.final_log_probability.shape[1],
        sigma_bins=1.25,
    )
    bce_row = F.binary_cross_entropy_with_logits(
        outputs.mask_logits.float(), target_mask.float(), reduction="none"
    ).mean((1, 2, 3))
    probability = torch.sigmoid(outputs.mask_logits.float())
    intersection = (probability * target_mask.float()).sum((1, 2, 3))
    dice_row = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum((1, 2, 3)) + target_mask.float().sum((1, 2, 3)) + 1.0
    )
    all_rows = torch.ones_like(valid)
    mask_bce = _weighted_mean(bce_row, all_rows, group_weight)
    mask_dice = _weighted_mean(dice_row, all_rows, group_weight)
    final_ce = _weighted_mean(
        -(soft * outputs.final_log_probability.float()).sum(1), valid, group_weight
    )
    final_expected = _weighted_mean(
        F.smooth_l1_loss(
            outputs.final_expected_progress.float(), target, reduction="none", beta=0.02
        ),
        valid,
        group_weight,
    )
    geometry_ce = _weighted_mean(
        -(soft * outputs.geometry_log_probability.float()).sum(1),
        geometry_valid,
        group_weight,
    )
    geometry_expected = _weighted_mean(
        F.smooth_l1_loss(
            outputs.geometry_expected_progress.float(), target, reduction="none", beta=0.02
        ),
        geometry_valid,
        group_weight,
    )
    loss = (
        mask_bce
        + mask_dice
        + final_ce
        + 2.0 * final_expected
        + 0.25 * geometry_ce
        + 0.5 * geometry_expected
    )
    parts = {
        "mask_bce": mask_bce,
        "mask_dice_loss": mask_dice,
        "final_soft_ce": final_ce,
        "final_expected_smooth_l1": final_expected,
        "geometry_soft_ce": geometry_ce,
        "geometry_expected_smooth_l1": geometry_expected,
    }
    if not bool(torch.isfinite(loss)) or any(
        not bool(torch.isfinite(value)) for value in parts.values()
    ):
        raise FloatingPointError("shared mask-geometry probe loss is non-finite")
    return loss, {name: value.detach() for name, value in parts.items()}


def train_head(
    visual: GeoPEPDProgressFusionNet,
    head: SharedMaskGeometryPosteriorHead,
    spatial: torch.Tensor,
    records: Sequence[ProgressProbeRecordV2],
    masks: torch.Tensor,
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    tensors = _progress_inputs_v2(records)
    dataset = TensorDataset(spatial, masks, *tensors)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, EPOCHS + 1):
        visual.eval()
        head.train()
        totals: Counter[str] = Counter()
        for batch in loader:
            spatial_b, mask_b, *input_b = [value.to(device) for value in batch]
            visual_outputs, pivot_features, pivot_logits, direction = frozen_components(
                visual, spatial_b, input_b
            )
            # frozen_components runs under inference_mode.  Clone at the
            # trainable boundary so Conv2d may save this input for weight
            # gradients without rebuilding the frozen spatial cache.
            pivot_features = pivot_features.clone()
            start_b, range_b, affine_b = input_b[2], input_b[3], input_b[4]
            outputs = head(
                pivot_features,
                pivot_logits.detach(),
                direction.detach(),
                visual_outputs.visual_progress_log_probability.detach(),
                start_b,
                range_b,
                affine_b,
                visual_outputs.valid.detach(),
            )
            loss, parts = probe_loss(outputs, input_b[7], mask_b, input_b[8])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in head.named_parameters():
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise FloatingPointError(f"missing/non-finite probe gradient: {name}")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            size = int(spatial_b.shape[0])
            totals["samples"] += size
            totals["loss"] += float(loss.detach()) * size
            for name, value in parts.items():
                totals[name] += float(value) * size
        row = {
            "epoch": float(epoch),
            "loss": totals["loss"] / totals["samples"],
            "samples": float(totals["samples"]),
        }
        for name in parts:
            row[name] = totals[name] / totals["samples"]
        history.append(row)
        print(
            f"UHPF-mask probe epoch={epoch}/{EPOCHS} loss={row['loss']:.6f}",
            flush=True,
        )
    return history


@torch.inference_mode()
def evaluate_once(
    visual: GeoPEPDProgressFusionNet,
    head: SharedMaskGeometryPosteriorHead,
    spatial: torch.Tensor,
    records: Sequence[ProgressProbeRecordV2],
    masks: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensors = _progress_inputs_v2(records)
    rows: list[dict[str, Any]] = []
    dice_values: list[float] = []
    mass_errors: list[float] = []
    fallback_errors: list[float] = []
    visual.eval()
    head.eval()
    for offset in range(0, len(records), BATCH_SIZE):
        stop = min(len(records), offset + BATCH_SIZE)
        input_b = [value[offset:stop].to(device) for value in tensors]
        visual_outputs, pivot_features, pivot_logits, direction = frozen_components(
            visual, spatial[offset:stop].to(device), input_b
        )
        output = head(
            pivot_features,
            pivot_logits,
            direction,
            visual_outputs.visual_progress_log_probability,
            input_b[2],
            input_b[3],
            input_b[4],
            visual_outputs.valid,
        )
        oracle = head(
            pivot_features,
            pivot_logits,
            direction,
            visual_outputs.visual_progress_log_probability,
            input_b[2],
            input_b[3],
            input_b[4],
            visual_outputs.valid,
            mask_probability_override=masks[offset:stop].to(device),
        )
        pred_binary = output.mask_probability >= 0.5
        target_binary = masks[offset:stop].to(device) >= 0.5
        intersection = (pred_binary & target_binary).sum((1, 2, 3)).float()
        dice = (2.0 * intersection + 1.0) / (
            pred_binary.sum((1, 2, 3)).float()
            + target_binary.sum((1, 2, 3)).float()
            + 1.0
        )
        dice_values.extend(dice.cpu().tolist())
        mass_errors.extend(output.posterior_mass_error.cpu().tolist())
        unavailable = ~output.geometry_available
        if bool(unavailable.any()):
            fallback_errors.extend(
                torch.max(
                    torch.abs(
                        output.final_log_probability[unavailable]
                        - output.visual_log_probability[unavailable]
                    ),
                    dim=1,
                ).values.cpu().tolist()
            )
        for local, record in enumerate(records[offset:stop]):
            valid = bool(visual_outputs.valid[local])
            progress = {
                "visual": float(output.visual_expected_progress[local]) if valid else None,
                "shared_mask_geometry": float(output.final_expected_progress[local]) if valid else None,
                "geometry_only": float(output.geometry_expected_progress[local]) if bool(output.geometry_available[local]) else None,
                "gt_mask_oracle_fusion": float(oracle.final_expected_progress[local]) if valid else None,
            }
            errors = {
                name: abs(float(value) - record.target_progress)
                if value is not None
                else 1.0
                for name, value in progress.items()
            }
            rows.append(
                {
                    "schema_version": 1,
                    "protocol": PROTOCOL,
                    "sample_id": record.sample_id,
                    "group_id": record.group_id,
                    "progress": progress,
                    "errors": errors,
                    "mask_dice": float(dice[local]),
                    "geometry_weight": float(output.geometry_weight[local]),
                    "geometry_available": bool(output.geometry_available[local]),
                }
            )
    methods: dict[str, Any] = {}
    for method in ("visual", "shared_mask_geometry", "geometry_only", "gt_mask_oracle_fusion"):
        covered = sum(row["progress"][method] is not None for row in rows)
        covered_errors = [
            row["errors"][method] for row in rows if row["progress"][method] is not None
        ]
        methods[method] = {
            "full_denominator_nmae": float(np.mean([row["errors"][method] for row in rows])),
            "coverage": covered / len(rows),
            "covered_samples": covered,
            "covered_p95_absolute_error": float(np.quantile(covered_errors, 0.95)),
        }
    bootstrap = grouped_bootstrap_delta(
        rows,
        "shared_mask_geometry",
        "visual",
        repetitions=5000,
    )
    rules = {
        "visual_baseline_reproduced_abs_2e_6": abs(
            methods["visual"]["full_denominator_nmae"] - VISUAL_BASELINE_NMAE
        )
        <= 2e-6,
        "predicted_mask_mean_dice_at_least_0_60": float(np.mean(dice_values)) >= 0.60,
        "gt_mask_oracle_improves_at_least_0_005": (
            methods["gt_mask_oracle_fusion"]["full_denominator_nmae"]
            <= methods["visual"]["full_denominator_nmae"] - 0.005
        ),
        "predicted_fusion_nmae_at_most_0_3616912228413731": (
            methods["shared_mask_geometry"]["full_denominator_nmae"]
            <= 0.3616912228413731
        ),
        "predicted_vs_visual_bootstrap_ci95_upper_below_zero": bootstrap["ci95_high"] < 0.0,
        "coverage_at_least_0_695": methods["shared_mask_geometry"]["coverage"] >= 0.695,
        "covered_p95_not_worse_than_visual_plus_0_01": (
            methods["shared_mask_geometry"]["covered_p95_absolute_error"]
            <= methods["visual"]["covered_p95_absolute_error"] + 0.01
        ),
        "posterior_mass_error_at_most_1e_6": max(mass_errors, default=0.0) <= 1e-6,
        "unavailable_fallback_error_at_most_1e_7": max(fallback_errors, default=0.0) <= 1e-7,
    }
    return {
        "methods": methods,
        "mask": {
            "mean_dice": float(np.mean(dice_values)),
            "median_dice": float(np.median(dice_values)),
        },
        "grouped_bootstrap": {
            "shared_mask_geometry_vs_visual": bootstrap
        },
        "numerics": {
            "posterior_mass_error_max": max(mass_errors, default=0.0),
            "unavailable_fallback_error_max": max(fallback_errors, default=0.0),
            "nonfinite": 0,
        },
        "decision": {"label": "GO" if all(rules.values()) else "NO_GO", "rules": rules},
    }, rows


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    assert_train_only_path(output_dir, label="uhpf_probe_output")
    fit, fit_masks, fit_identities = load_scope_records(args, scope="algorithm_fit")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    visual, stage_identity = load_visual_model(Path(args.stage_a_checkpoint), device)
    validation = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "validated",
        "mode": "algorithm_fit_only",
        "fit": {
            "samples": len(fit),
            "groups": len({record.group_id for record in fit}),
            "pointer_masks": len(fit_masks),
            "reference_available": sum(record.reference_start is not None and record.reference_range is not None for record in fit),
        },
        "spatial_cache_contract": {
            "shape": [len(fit), 512, 8, 8],
            "dtype": "float16",
            "estimated_bytes": len(fit) * 512 * 8 * 8 * 2,
        },
        "head": {
            "trainable_parameters": SharedMaskGeometryPosteriorHead.trainable_parameter_count(),
            "forbidden_inputs": sorted(FORBIDDEN_INPUT_NAMES),
            "forbidden_inputs_used": False,
        },
        "stage_a_checkpoint": stage_identity,
        "input_sha256": fit_identities,
        "restricted_data_use": {
            name: 0
            for name in (
                "public_samples",
                "test_samples",
                "field_samples",
                "sealed_samples",
                "confirmatory_samples",
                "confirmation_a_samples",
                "confirmation_b_samples",
                "algorithm_selection_samples",
            )
        },
    }
    _write_json(output_dir / "validation.json", validation)
    if args.validate_only:
        return output_dir / "validation.json"
    cache_path = output_dir / "cache/algorithm_fit_encoder_spatial_fp16.pt"
    fit_spatial = spatial_encoder_cache(
        visual,
        fit,
        cache_path,
        scope="algorithm_fit",
        checkpoint_sha256=stage_identity["sha256"],
        device=device,
        batch_size=int(args.cache_batch_size),
        workers=int(args.workers),
    )
    if args.prepare_spatial_cache:
        validation["status"] = "fit_spatial_cache_prepared"
        validation["spatial_cache"] = {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
        }
        _write_json(output_dir / "validation.json", validation)
        return output_dir / "validation.json"
    if (output_dir / "summary.json").exists():
        raise RuntimeError("formal UHPF mask-geometry probe already exists")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    fit_mask_tensor = load_aligned_masks(fit, fit_masks)
    head = SharedMaskGeometryPosteriorHead(progress_bins=visual.progress_bins).to(device)
    started = time.time()
    history = train_head(
        visual, head, fit_spatial, fit, fit_mask_tensor, device=device
    )
    # The only algorithm_selection read/evaluation occurs here, after fixed E5.
    selection, selection_masks, selection_identities = load_scope_records(
        args, scope="algorithm_selection"
    )
    selection_cache_path = output_dir / "cache/algorithm_selection_encoder_spatial_fp16.pt"
    selection_spatial = spatial_encoder_cache(
        visual,
        selection,
        selection_cache_path,
        scope="algorithm_selection",
        checkpoint_sha256=stage_identity["sha256"],
        device=device,
        batch_size=int(args.cache_batch_size),
        workers=int(args.workers),
    )
    selection_mask_tensor = load_aligned_masks(selection, selection_masks)
    metrics, predictions = evaluate_once(
        visual,
        head,
        selection_spatial,
        selection,
        selection_mask_tensor,
        device=device,
    )
    checkpoint_path = output_dir / "probe_last.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "seed": SEED,
            "head_state": head.state_dict(),
            "stage_a_checkpoint": stage_identity,
            "fit_input_sha256": fit_identities,
        },
        temporary,
    )
    os.replace(temporary, checkpoint_path)
    predictions_path = output_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, predictions)
    summary = {
        **validation,
        "status": "complete",
        "formal_result": True,
        "configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "checkpoint_selection": "none; fixed E5",
            "selection_evaluations": 1,
        },
        "history": history,
        "selection": {
            "samples": len(selection),
            "groups": len({record.group_id for record in selection}),
        },
        "metrics": metrics,
        "selection_input_sha256": selection_identities,
        "artifacts": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
            "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path), "rows": len(predictions)},
        },
        "elapsed_seconds": time.time() - started,
    }
    summary["restricted_data_use"]["algorithm_selection_samples"] = len(selection)
    _write_json(output_dir / "summary.json", summary)
    return output_dir / "summary.json"


if __name__ == "__main__":
    print(run(parse_args()))
