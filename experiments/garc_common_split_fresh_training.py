"""Fresh common-split PEPD + matched enhanced-V5 training.

The formal path is intentionally fail closed.  It accepts only a sealed
promotion authorization and a separately frozen execution specification.  The
existing common-split protocol defines the scientific partitions and gates,
but it does not currently pin the optimizer schedules or the progress-fusion
candidate grid; therefore no formal run is allowed until an execution spec
containing those choices is frozen before training.

Only SyncG/train ``algorithm_fit`` and ``calibration`` annotation files are
opened.  The all-in-one 16,000-row value-bearing manifest is never parsed by
this trainer, so the 1,080 independent-validation and 520 excluded labels stay
unopened until the later one-shot validation stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments import garc_common_split_progress as common
from experiments.automatic_numeric_range_public_protocol import (
    atomic_new_json,
    canonical_sha256,
    guard_public_path,
    load_partition_roster,
    require,
    sha256_file,
    strict_json,
)
from experiments.cagh_scalemark_reference_head_v5 import (
    build_head,
    multiscale_dense_tick_loss,
)
from experiments.probabilistic_pivot_direction import (
    IMAGENET_WEIGHTS,
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.vdn_baseline import (
    VDNSample,
    _pointer_points,
    seed_worker,
    set_random_seed,
)
import experiments.train_probabilistic_pivot_direction_syncg as pepd_train
import experiments.train_cagh_scalemark_reference_probe_v5 as v5_base
import experiments.train_cagh_scalemark_reference_probe_v5_enhanced as v5_enhanced


PREFLIGHT_PROTOCOL: Final[str] = "garc_common_split_fresh_training_preflight_v1"
EXECUTION_SPEC_PROTOCOL: Final[str] = "garc_common_split_fresh_execution_spec_v1"
SEED_SUMMARY_PROTOCOL: Final[str] = "garc_common_split_fresh_seed_summary_v1"
SEED_BUNDLE_PROTOCOL: Final[str] = "garc_common_split_fresh_seed_bundle_v1"
JOURNAL_PROTOCOL: Final[str] = "garc_common_split_fresh_epoch_journal_v1"
SELECTION_PROTOCOL: Final[str] = "garc_common_split_fresh_selection_v1"
ENSEMBLE_PROTOCOL: Final[str] = "garc_common_split_fresh_ensemble_v1"
ALLOWED_OUTPUT_PARENT: Final[Path] = Path(r"C:\pointer_read")
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    r"C:\pointer_read\garc_common_split_fresh_training_v1"
)
PUBLIC_SYNCG_ROOT: Final[Path] = _PROJECT_ROOT / "datasets/SyncG/syncG"
PUBLIC_IMAGE_ROOT: Final[Path] = PUBLIC_SYNCG_ROOT / "images/train"
PUBLIC_ANNOTATION_ROOT: Final[Path] = PUBLIC_SYNCG_ROOT / "annotations/train"
AUTHORIZATION_FORBIDDEN_HASHES: Final[frozenset[str]] = frozenset(
    {
        "6c3560f5c29f430db33721beb753ec4f5582580792fa098c4a6f1057746a01d9",
        "b0d44b19a98e4e8ce4f21dda811c199c04342dee359a33da27a6eb1ed72ee6fa",
    }
)
FORBIDDEN_LEGACY_HASHES: Final[frozenset[str]] = frozenset(
    {
        "6c3560f5c29f430db33721beb753ec4f5582580792fa098c4a6f1057746a01d9",
        "410d250b312de49cf88cd10924497380ba3ca494703e6b6d603d9ccc28ecda8a",
        "d952db3746528c4ebc06919f39e49ff86017887fb468dc4358256609f7fbdd21",
        "b0d44b19a98e4e8ce4f21dda811c199c04342dee359a33da27a6eb1ed72ee6fa",
        "83d522a23666740512c8992c70d6c470264314251622872281c950bc004dbafc",
    }
)
SOURCE_FILES: Final[dict[str, Path]] = {
    "pepd_model": _PROJECT_ROOT / "experiments/probabilistic_pivot_direction.py",
    "pepd_training_primitives": _PROJECT_ROOT
    / "experiments/train_probabilistic_pivot_direction_syncg.py",
    "v5_head": _PROJECT_ROOT / "experiments/cagh_scalemark_reference_head_v5.py",
    "v5_data_and_geometry": _PROJECT_ROOT
    / "experiments/train_cagh_scalemark_reference_probe_v5.py",
    "v5_enhanced_training_primitives": _PROJECT_ROOT
    / "experiments/train_cagh_scalemark_reference_probe_v5_enhanced.py",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text_source(path: Path) -> str:
    # The project already uses exact source-file hashes for research artifacts.
    return sha256_file(path.resolve(strict=True))


def _binding(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_output(path: Path, *, label: str, must_exist: bool = False) -> Path:
    resolved = guard_public_path(
        Path(path), label=label, must_exist=must_exist, expect_file=False
    )
    parent = ALLOWED_OUTPUT_PARENT.resolve(strict=True)
    require(resolved.is_relative_to(parent), f"{label} is outside {parent}")
    require(resolved != parent, f"{label} cannot be the broad output parent")
    return resolved


def _verify_binding(value: Any, *, label: str) -> Path:
    require(isinstance(value, Mapping), f"{label} binding absent")
    path = guard_public_path(Path(str(value.get("path") or "")), label=label)
    digest = str(value.get("sha256") or "")
    require(len(digest) == 64 and digest == digest.casefold(), f"{label} bad SHA-256")
    require(sha256_file(path) == digest, f"{label} hash drift")
    return path


def authenticate_promotion(
    *,
    protocol_path: Path,
    authorization_path: Path,
    promotion_seal_path: Path,
) -> dict[str, Any]:
    """Authenticate the promotion bundle without touching data or CUDA."""

    protocol_file, protocol = common.load_protocol(protocol_path)
    authorization_file = guard_public_path(
        authorization_path, label="common-split training authorization"
    )
    seal_file = guard_public_path(promotion_seal_path, label="promotion seal")
    require(
        authorization_file.parent == seal_file.parent,
        "authorization and promotion seal are not from one bundle",
    )
    seal = strict_json(seal_file)
    require(
        seal.get("protocol") == "garc_common_split_promotion_decision_v1"
        and seal.get("status") == "sealed",
        "promotion seal identity drift",
    )
    artifacts = seal.get("artifacts")
    require(isinstance(artifacts, Mapping), "promotion seal artifact roster absent")
    require(
        artifacts.get("training_authorization.json") == sha256_file(authorization_file),
        "promotion seal does not authenticate the authorization",
    )
    require(
        seal.get("bundle_sha256") == canonical_sha256(dict(artifacts)),
        "promotion seal bundle digest drift",
    )
    decision_file = authorization_file.parent / "decision.json"
    gate_file = authorization_file.parent / "pilot_gate.json"
    require(
        artifacts.get("decision.json") == sha256_file(decision_file),
        "promotion seal decision hash drift",
    )
    require(
        artifacts.get("pilot_gate.json") == sha256_file(gate_file),
        "promotion seal gate hash drift",
    )
    decision = strict_json(decision_file)
    require(
        decision.get("status") == "authorized"
        and decision.get("training_allowed") is True,
        "promotion decision does not authorize training",
    )
    declared_auth = decision.get("training_authorization")
    require(isinstance(declared_auth, Mapping), "decision authorization binding absent")
    require(
        Path(str(declared_auth.get("path") or "")).resolve(strict=True)
        == authorization_file
        and declared_auth.get("sha256") == sha256_file(authorization_file),
        "decision/authorization binding drift",
    )
    authorization = strict_json(authorization_file)
    require(
        authorization.get("protocol")
        == "garc_common_split_training_authorization_v1"
        and authorization.get("status") == "authorized_for_common_split_training"
        and authorization.get("training_allowed") is True,
        "training authorization identity drift",
    )
    frozen = authorization.get("frozen_protocol")
    require(isinstance(frozen, Mapping), "authorization protocol binding absent")
    require(
        Path(str(frozen.get("path") or "")).resolve(strict=True) == protocol_file
        and frozen.get("sha256") == sha256_file(protocol_file),
        "authorization/common protocol drift",
    )
    require(
        authorization.get("pilot_gate_report", {}).get("sha256")
        == sha256_file(gate_file),
        "authorization/pilot gate drift",
    )
    require(
        tuple(int(seed) for seed in authorization.get("seeds", []))
        == common.EXPECTED_SEEDS,
        "authorization seed roster drift",
    )
    forbidden = {
        str(row.get("sha256"))
        for row in authorization.get("forbidden_weight_reuse", [])
        if isinstance(row, Mapping)
    }
    require(
        AUTHORIZATION_FORBIDDEN_HASHES.issubset(forbidden),
        "authorization omits a forbidden legacy checkpoint",
    )
    gate = strict_json(gate_file)
    require(
        gate.get("protocol") == common.PREFLIGHT_PROTOCOL
        and gate.get("status") == "training_gate_passed"
        and gate.get("training_allowed") is True,
        "pilot gate did not pass",
    )
    checks = gate.get("pilot_gate", {}).get("checks")
    require(
        isinstance(checks, Mapping)
        and len(checks) == 8
        and all(value is True for value in checks.values()),
        "pilot gate checks are not all true",
    )
    return {
        "protocol_file": protocol_file,
        "protocol": protocol,
        "authorization_file": authorization_file,
        "authorization": authorization,
        "seal_file": seal_file,
        "decision_file": decision_file,
        "gate_file": gate_file,
    }


def execution_spec_gaps(protocol: Mapping[str, Any]) -> list[str]:
    """Describe why the current common protocol alone cannot launch training."""

    method = protocol.get("method")
    optimization = protocol.get("optimization")
    require(isinstance(method, Mapping), "common protocol method absent")
    require(isinstance(optimization, Mapping), "common protocol optimization absent")
    gaps: list[str] = []
    if not isinstance(method.get("pepd_training"), Mapping):
        gaps.append("method.pepd_training exact architecture/optimizer/schedule is absent")
    if not isinstance(method.get("v5_enhanced_training"), Mapping):
        gaps.append("method.v5_enhanced_training exact optimizer/schedule is absent")
    if not isinstance(optimization.get("fusion_candidate_grid"), Mapping):
        gaps.append("optimization.fusion_candidate_grid values are absent")
    if not isinstance(optimization.get("pepd_checkpoint_selection"), Mapping):
        gaps.append(
            "optimization.pepd_checkpoint_selection before matched-head training is ambiguous"
        )
    return gaps


def load_execution_spec(
    path: Path,
    *,
    protocol_file: Path,
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    spec_file = guard_public_path(path, label="fresh-training execution spec")
    spec = strict_json(spec_file)
    require(spec.get("schema_version") == 1, "execution spec schema drift")
    require(spec.get("protocol") == EXECUTION_SPEC_PROTOCOL, "execution spec identity drift")
    require(
        spec.get("status") == "frozen_before_common_split_training",
        "execution spec is not frozen before training",
    )
    parent = spec.get("parent_protocol")
    require(isinstance(parent, Mapping), "execution spec parent binding absent")
    require(
        Path(str(parent.get("path") or "")).resolve(strict=True) == protocol_file
        and parent.get("sha256") == sha256_file(protocol_file),
        "execution spec/common protocol drift",
    )
    require(
        tuple(int(seed) for seed in spec.get("seeds", [])) == common.EXPECTED_SEEDS,
        "execution spec seed roster drift",
    )
    declared_output = Path(str(spec.get("output_root") or ""))
    require(
        _guard_output(declared_output, label="execution output root")
        == DEFAULT_OUTPUT_ROOT.resolve(),
        "execution output root drift",
    )
    sources = spec.get("source_bindings")
    require(isinstance(sources, Mapping), "execution source bindings absent")
    for name, source in SOURCE_FILES.items():
        bound = sources.get(name)
        path_value = _verify_binding(bound, label=f"execution source {name}")
        require(path_value == source.resolve(strict=True), f"execution source path drift: {name}")
    initialization = _verify_binding(
        spec.get("pepd", {}).get("imagenet_initialization"),
        label="generic ImageNet initialization",
    )
    expected_initialization = (
        Path(torch.hub.get_dir()) / "checkpoints" / Path(IMAGENET_WEIGHTS.url).name
    ).resolve(strict=True)
    require(initialization == expected_initialization, "ImageNet initialization path drift")
    require(
        sha256_file(initialization) not in FORBIDDEN_LEGACY_HASHES,
        "generic initialization aliases a forbidden legacy checkpoint",
    )
    pepd = spec.get("pepd")
    v5 = spec.get("v5_enhanced")
    fusion = spec.get("progress_fusion")
    require(isinstance(pepd, Mapping), "PEPD execution configuration absent")
    require(isinstance(v5, Mapping), "enhanced-V5 execution configuration absent")
    require(isinstance(fusion, Mapping), "progress-fusion execution configuration absent")
    required_pepd = {
        "phase1_epochs",
        "phase2_epochs",
        "batch_size",
        "workers",
        "image_size",
        "heatmap_size",
        "angle_bins",
        "learning_rate",
        "phase1_eta_min",
        "phase2_learning_rate",
        "weight_decay",
        "pivot_loss_weight",
        "bin_loss_weight",
        "vector_loss_weight",
        "paired_supervision_weight",
        "equivariance_weight",
        "equivariance_pivot_weight",
        "soft_target_sigma_bins",
        "expansion",
        "scale_factor",
        "rotation_factor",
        "translation_factor",
        "heatmap_sigma",
        "perspective_probability",
        "max_perspective_degrees",
        "max_blur_sigma",
        "intermediate_selection_metric",
    }
    require(required_pepd.issubset(pepd), "PEPD execution fields are incomplete")
    require(
        pepd.get("intermediate_selection_metric")
        == "calibration_angle_mae_then_nll_then_pivot_error",
        "PEPD intermediate selection metric drift",
    )
    require(
        int(pepd["phase1_epochs"]) > 0
        and int(pepd["phase2_epochs"]) > 0
        and int(pepd["batch_size"]) > 0
        and 0 <= int(pepd["workers"]) <= 2,
        "PEPD schedule/loader values are invalid",
    )
    require(
        int(pepd["image_size"]) % 32 == 0
        and int(pepd["heatmap_size"]) == int(pepd["image_size"]) // 4
        and int(pepd["angle_bins"]) >= 8,
        "PEPD spatial configuration is invalid",
    )
    require(
        float(pepd["learning_rate"]) > 0.0
        and float(pepd["phase1_eta_min"]) > 0.0
        and float(pepd["phase2_learning_rate"]) > 0.0
        and math.isclose(
            float(pepd["phase1_eta_min"]),
            float(pepd["phase2_learning_rate"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "PEPD phase-boundary learning rates are invalid",
    )
    required_v5 = {
        "stage_a_epochs",
        "stage_b_epochs",
        "batch_size",
        "workers",
        "learning_rate",
        "weight_decay",
        "augmentation",
        "checkpoint_selection_metric",
    }
    require(required_v5.issubset(v5), "enhanced-V5 execution fields are incomplete")
    require(
        v5.get("checkpoint_selection_metric")
        == "minimum_calibration_progress_mae_failure_penalty_1_then_coverage",
        "enhanced-V5 selection metric drift",
    )
    require(
        int(v5["stage_a_epochs"]) > 0
        and int(v5["stage_b_epochs"]) > 0
        and int(v5["batch_size"]) > 0
        and 0 <= int(v5["workers"]) <= 2
        and float(v5["learning_rate"]) > 0.0
        and float(v5["weight_decay"]) >= 0.0,
        "enhanced-V5 schedule/optimizer values are invalid",
    )
    require(
        isinstance(v5.get("augmentation"), Mapping),
        "enhanced-V5 augmentation is not frozen",
    )
    require(
        set(v5["augmentation"])
        == set(v5_base.PhotoAugmentation.__dataclass_fields__),
        "enhanced-V5 augmentation field roster drift",
    )
    v5_base.PhotoAugmentation(**dict(v5["augmentation"])).validate()
    require(
        fusion.get("fit_partition") == "algorithm_fit"
        and fusion.get("selection_partition") == "calibration",
        "progress-fusion partition policy drift",
    )
    require(
        fusion.get("family") == "ridge_residual_plus_confidence_abstention",
        "progress-fusion family drift",
    )
    for key in ("ridge_lambdas", "residual_scales", "confidence_thresholds"):
        values = fusion.get(key)
        require(isinstance(values, list) and values, f"progress-fusion {key} grid absent")
        require(all(math.isfinite(float(value)) for value in values), f"bad {key} grid")
    require(
        all(float(value) >= 0.0 for value in fusion["ridge_lambdas"]),
        "negative progress-fusion ridge lambda",
    )
    require(
        all(0.0 <= float(value) <= 1.0 for value in fusion["confidence_thresholds"]),
        "progress-fusion confidence threshold outside [0,1]",
    )
    require(
        fusion.get("mask_geometry_policy")
        in {
            "omitted_no_eligible_algorithm_fit_only_binding",
            "eligible_algorithm_fit_only_binding",
        },
        "mask-geometry provenance policy absent",
    )
    if fusion.get("mask_geometry_policy") == "eligible_algorithm_fit_only_binding":
        _verify_binding(
            fusion.get("mask_geometry_binding"), label="algorithm-fit-only mask geometry"
        )
    else:
        require(
            fusion.get("mask_geometry_binding") is None,
            "omitted mask geometry unexpectedly binds weights",
        )
    for partition, expected in protocol["partitions"].items():
        declared = spec.get("partitions", {}).get(partition)
        require(isinstance(declared, Mapping), f"execution {partition} binding absent")
        for key in ("samples", "groups", "sha256", "sample_ids_sha256", "group_ids_sha256"):
            require(declared.get(key) == expected.get(key), f"execution {partition}.{key} drift")
    return spec_file, spec


@dataclass(frozen=True)
class MaterializedPartition:
    name: str
    samples: tuple[VDNSample, ...]
    v5_records: tuple[v5_base.PublicRecord, ...]
    manifest_path: Path
    evidence: dict[str, Any]


def _annotation_to_sample(row: Mapping[str, Any]) -> tuple[VDNSample, dict[str, str]]:
    sample_id = str(row.get("sample_id") or "")
    require(sample_id and Path(sample_id).name == sample_id, "unsafe public sample id")
    annotation_path = (PUBLIC_ANNOTATION_ROOT / f"{sample_id}.json").resolve(strict=True)
    image_path = (PUBLIC_IMAGE_ROOT / str(row.get("image_relpath") or "")).resolve(strict=True)
    require(annotation_path.is_relative_to(PUBLIC_ANNOTATION_ROOT.resolve(strict=True)), "annotation escaped SyncG/train")
    require(image_path.is_relative_to(PUBLIC_IMAGE_ROOT.resolve(strict=True)), "image escaped SyncG/train")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    require(isinstance(annotation, Mapping), f"{sample_id}: annotation is not an object")
    require(str(annotation.get("file_name") or sample_id) in {sample_id, image_path.name}, f"{sample_id}: file identity drift")
    scene = str(annotation.get("scene_name") or "unknown_scene")
    gauge = str(annotation.get("gauge_type") or "unknown_type")
    group_id = f"{gauge}::{Path(scene).stem}"
    require(group_id == str(row.get("group_id")), f"{sample_id}: group identity drift")
    dial_bbox = tuple(float(value) for value in annotation.get("dial_bbox_annotations", [])[:4])
    require(len(dial_bbox) == 4 and tuple(float(v) for v in row.get("dial_bbox", [])) == dial_bbox, f"{sample_id}: dial bbox drift")
    keypoints = annotation.get("keypoints_annotations")
    metadata = {
        "annotation_path": str(annotation_path),
        "scene_name": scene,
        "gauge_type": gauge,
        "pointer_angle": annotation.get("pointer_rotate_degree"),
        "long_interval_degree": annotation.get("long_interval_degree"),
        "long_interval_value": float(annotation["long_interval_value"]),
        "long_num": int(annotation["long_num"]),
        "dial_bbox": list(dial_bbox),
        "keypoints": keypoints,
        "homography": annotation.get("homo_matrix"),
    }
    # Reuse the canonical pointer parser without invoking the full-manifest
    # loader that would decode all 16,000 value-bearing rows.
    tip, tail = _pointer_points(metadata, sample_id)
    start = float(annotation["start_value"])
    end = start + (int(annotation["long_num"]) - 1) * float(
        annotation["long_interval_value"]
    )
    sample = VDNSample(
        sample_id=sample_id,
        group_id=group_id,
        dataset="SyncG",
        split="train",
        image_path=str(image_path),
        dial_bbox=dial_bbox,
        pointer_tip=tip,
        pointer_tail=tail,
        ground_truth=float(annotation["ground_truth"]),
        scale_start=start,
        scale_end=end,
        metadata=metadata,
    )
    return sample, {
        "sample_id": sample_id,
        "annotation_sha256": sha256_file(annotation_path),
        "image_sha256": sha256_file(image_path),
    }


def materialize_partition(
    *, protocol_path: Path, protocol: Mapping[str, Any], partition: str
) -> MaterializedPartition:
    require(partition in {"algorithm_fit", "calibration"}, "trainer may materialize only fit/calibration")
    _, manifest_path, rows, audit = load_partition_roster(protocol_path, partition)
    expected = protocol["partitions"][partition]
    require((audit["samples"], audit["groups"]) == (expected["samples"], expected["groups"]), f"{partition} inventory drift")
    samples: list[VDNSample] = []
    identities: list[dict[str, str]] = []
    for row in rows:
        sample, identity = _annotation_to_sample(row)
        samples.append(sample)
        identities.append(identity)
    counts = Counter(sample.group_id for sample in samples)
    records = tuple(
        v5_base.PublicRecord(
            sample=sample,
            group_weight=len(samples) / (len(counts) * counts[sample.group_id]),
            partition=partition,
        )
        for sample in samples
    )
    evidence = {
        "partition": partition,
        "samples": len(samples),
        "groups": len(counts),
        "roster": _binding(manifest_path),
        "sample_ids_sha256": canonical_sha256(sorted(sample.sample_id for sample in samples)),
        "group_ids_sha256": canonical_sha256(sorted(counts)),
        "selected_annotation_content_sha256": canonical_sha256(
            [[row["sample_id"], row["annotation_sha256"]] for row in identities]
        ),
        "selected_image_content_sha256": canonical_sha256(
            [[row["sample_id"], row["image_sha256"]] for row in identities]
        ),
        "annotations_opened": len(samples),
        "images_hashed": len(samples),
        "independent_validation_annotations_opened": 0,
        "development_excluded_annotations_opened": 0,
    }
    require(evidence["sample_ids_sha256"] == expected["sample_ids_sha256"], f"{partition} sample hash drift")
    require(evidence["group_ids_sha256"] == expected["group_ids_sha256"], f"{partition} group hash drift")
    return MaterializedPartition(
        name=partition,
        samples=tuple(samples),
        v5_records=records,
        manifest_path=manifest_path,
        evidence=evidence,
    )


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "loader_generator": generator.get_state().cpu(),
    }


def _restore_rng(value: Mapping[str, Any], generator: torch.Generator) -> None:
    require(isinstance(value, Mapping), "resume RNG state absent")
    for key in ("python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"):
        require(key in value, f"resume RNG state missing {key}")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    cuda = value["torch_cuda"]
    require(isinstance(cuda, Sequence), "resume CUDA RNG state invalid")
    if cuda:
        require(torch.cuda.is_available(), "resume requires unavailable CUDA RNG")
        require(len(cuda) == torch.cuda.device_count(), "resume CUDA topology drift")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])
    generator.set_state(value["loader_generator"].cpu())


def _configure_determinism(seed: int, device: torch.device) -> None:
    require(
        os.environ.get("PYTHONHASHSEED") == str(seed),
        f"formal seed {seed} requires PYTHONHASHSEED={seed} before Python starts",
    )
    require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "formal training requires CUBLAS_WORKSPACE_CONFIG=:4096:8",
    )
    require(device.type == "cuda" and torch.cuda.is_available(), "formal training is CUDA-only")
    require("NVIDIA" in torch.cuda.get_device_name(device).upper(), "formal CUDA device is not NVIDIA")
    set_random_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _pepd_args(spec: Mapping[str, Any]) -> argparse.Namespace:
    value = dict(spec["pepd"])
    value["epochs"] = int(value["phase1_epochs"]) + int(value["phase2_epochs"])
    return argparse.Namespace(**value)


def _v5_args(spec: Mapping[str, Any], *, seed: int) -> argparse.Namespace:
    value = dict(spec["v5_enhanced"])
    augmentation = dict(value.pop("augmentation"))
    value.update(augmentation)
    value.update(
        {
            "seed": int(seed),
            "augmentation_profile": "photo",
            "device": "cuda:0",
            "run_formal": True,
            "smoke": False,
            "validate_only": False,
        }
    )
    return argparse.Namespace(**value)


def _data_signature(
    fit: MaterializedPartition, calibration: MaterializedPartition
) -> dict[str, Any]:
    return {
        "algorithm_fit": fit.evidence,
        "calibration": calibration.evidence,
        "independent_validation": {
            "images_read": 0,
            "annotations_read": 0,
        },
        "development_excluded": {
            "images_read": 0,
            "annotations_read": 0,
        },
    }


def _training_signature(
    *,
    seed: int,
    protocol_file: Path,
    spec_file: Path,
    authorization_file: Path,
    promotion_seal_file: Path,
    data_signature: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": SEED_SUMMARY_PROTOCOL,
        "seed": int(seed),
        "common_protocol": _binding(protocol_file),
        "execution_spec": _binding(spec_file),
        "promotion_authorization": _binding(authorization_file),
        "promotion_seal": _binding(promotion_seal_file),
        "data": dict(data_signature),
        "source_sha256": {
            **{name: _sha256_text_source(path) for name, path in SOURCE_FILES.items()},
            "formal_trainer": _sha256_text_source(Path(__file__)),
        },
    }


def _pepd_candidate(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["angle_mae_degrees"]),
        float(metrics["angular_calibration_nll"]),
        float(metrics["pivot_mean_error_fraction"]),
    )


def _save_pepd_journal(
    path: Path,
    *,
    signature: Mapping[str, Any],
    epoch: int,
    total_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    history: Sequence[Mapping[str, Any]],
    best_state: Mapping[str, torch.Tensor],
    best_candidate: Sequence[float],
    best_epoch: int,
    generator: torch.Generator,
    elapsed_seconds: float,
) -> None:
    require(len(history) == epoch, "PEPD journal history/epoch mismatch")
    _atomic_torch(
        path,
        {
            "schema_version": 1,
            "protocol": JOURNAL_PROTOCOL,
            "component": "fresh_common_split_pepd",
            "status": "in_progress",
            "signature": dict(signature),
            "signature_sha256": canonical_sha256(signature),
            "completed_epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "model_state": {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": [dict(row) for row in history],
            "best_state": {
                name: tensor.detach().cpu() for name, tensor in best_state.items()
            },
            "best_candidate": [float(value) for value in best_candidate],
            "best_epoch": int(best_epoch),
            "rng": _capture_rng(generator),
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def _load_pepd_journal(
    path: Path,
    *,
    signature: Mapping[str, Any],
    total_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
) -> dict[str, Any]:
    value = torch.load(path.resolve(strict=True), map_location="cpu", weights_only=False)
    require(isinstance(value, Mapping), "PEPD resume journal is not a mapping")
    require(value.get("protocol") == JOURNAL_PROTOCOL, "PEPD journal protocol drift")
    require(value.get("component") == "fresh_common_split_pepd", "PEPD journal component drift")
    require(value.get("status") == "in_progress", "PEPD journal status drift")
    require(value.get("signature_sha256") == canonical_sha256(signature), "PEPD journal signature drift")
    require(canonical_sha256(value.get("signature")) == canonical_sha256(signature), "PEPD embedded signature drift")
    epoch = int(value.get("completed_epoch", 0))
    require(1 <= epoch <= total_epochs, "PEPD resume epoch drift")
    require(int(value.get("total_epochs", -1)) == total_epochs, "PEPD total epoch drift")
    history = value.get("history")
    require(isinstance(history, list) and len(history) == epoch, "PEPD resume history drift")
    # Mutate only after every cheap identity check has succeeded.
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    scheduler.load_state_dict(value["scheduler_state"])
    scaler.load_state_dict(value["scaler_state"])
    _restore_rng(value["rng"], generator)
    return {
        "epoch": epoch,
        "history": list(history),
        "best_state": dict(value["best_state"]),
        "best_candidate": tuple(float(item) for item in value["best_candidate"]),
        "best_epoch": int(value["best_epoch"]),
        "elapsed_seconds": float(value.get("elapsed_seconds", 0.0)),
    }


def train_fresh_pepd(
    *,
    fit: MaterializedPartition,
    calibration: MaterializedPartition,
    spec: Mapping[str, Any],
    signature: Mapping[str, Any],
    seed: int,
    device: torch.device,
    output: Path,
    resume: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = _pepd_args(spec)
    phase1_epochs = int(config.phase1_epochs)
    phase2_epochs = int(config.phase2_epochs)
    total_epochs = phase1_epochs + phase2_epochs
    require(total_epochs > 0 and phase1_epochs > 0, "invalid PEPD epoch schedule")
    require(int(config.batch_size) > 0 and 0 <= int(config.workers) <= 2, "invalid PEPD loader configuration")
    common_dataset = {
        "image_size": int(config.image_size),
        "heatmap_size": int(config.heatmap_size),
        "expansion": float(config.expansion),
        "scale_factor": float(config.scale_factor),
        "rotation_factor": float(config.rotation_factor),
        "translation_factor": float(config.translation_factor),
        "heatmap_sigma": float(config.heatmap_sigma),
        "perspective_probability": float(config.perspective_probability),
        "max_perspective_degrees": float(config.max_perspective_degrees),
        "max_blur_sigma": float(config.max_blur_sigma),
    }
    fit_dataset = SyncGProbabilisticDirectionDataset(
        fit.samples, training=True, **common_dataset
    )
    calibration_dataset = SyncGProbabilisticDirectionDataset(
        calibration.samples, training=False, **common_dataset
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "num_workers": int(config.workers),
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": False,
    }
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=int(config.batch_size),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=int(config.batch_size),
        shuffle=False,
        **loader_options,
    )
    model = build_probabilistic_pivot_direction_model(
        angle_bins=int(config.angle_bins), imagenet_pretrained=True
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=phase1_epochs,
        eta_min=float(config.phase1_eta_min),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=True, init_scale=512.0)
    journal = output / "pepd_last.pt"
    history: list[dict[str, Any]] = []
    best_candidate = (math.inf, math.inf, math.inf)
    best_state: dict[str, torch.Tensor] = {}
    best_epoch = 0
    start_epoch = 1
    elapsed_before = 0.0
    if resume and journal.is_file():
        recovered = _load_pepd_journal(
            journal,
            signature=signature,
            total_epochs=total_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generator=generator,
        )
        start_epoch = recovered["epoch"] + 1
        history = recovered["history"]
        best_candidate = recovered["best_candidate"]
        best_state = recovered["best_state"]
        best_epoch = recovered["best_epoch"]
        elapsed_before = recovered["elapsed_seconds"]
    elif journal.exists():
        raise FileExistsError(f"PEPD journal exists; pass --resume: {journal}")
    started = time.monotonic()
    for epoch in range(start_epoch, total_epochs + 1):
        if epoch == phase1_epochs + 1:
            for group in optimizer.param_groups:
                group["lr"] = float(config.phase2_learning_rate)
        lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = pepd_train._train_epoch(
            model,
            fit_loader,
            optimizer,
            scaler,
            device=device,
            amp_enabled=True,
            args=config,
            epoch=epoch,
        )
        calibration_metrics = pepd_train._validate(
            model,
            calibration_loader,
            device=device,
            amp_enabled=True,
            args=config,
        )
        if epoch <= phase1_epochs:
            scheduler.step()
        candidate = _pepd_candidate(calibration_metrics)
        if candidate < best_candidate:
            best_candidate = candidate
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        history.append(
            {
                "epoch": epoch,
                "phase": "phase1_cosine" if epoch <= phase1_epochs else "phase2_constant",
                "learning_rate": lr,
                "train": train_metrics,
                "calibration": calibration_metrics,
            }
        )
        _save_pepd_journal(
            journal,
            signature=signature,
            epoch=epoch,
            total_epochs=total_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_state=best_state,
            best_candidate=best_candidate,
            best_epoch=best_epoch,
            generator=generator,
            elapsed_seconds=elapsed_before + time.monotonic() - started,
        )
        print(
            f"common-split PEPD seed={seed} epoch={epoch}/{total_epochs} "
            f"val_angle={candidate[0]:.6f}deg best={best_candidate[0]:.6f}@{best_epoch}",
            flush=True,
        )
    require(best_state and best_epoch > 0, "PEPD did not produce a selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    checkpoint_path = output / "pepd_selected.pt"
    _atomic_torch(
        checkpoint_path,
        {
            "schema_version": 1,
            "protocol": SEED_BUNDLE_PROTOCOL,
            "component": "pepd_backbone",
            "status": "complete",
            "seed": seed,
            "signature_sha256": canonical_sha256(signature),
            "selected_epoch": best_epoch,
            "selection_metric": "calibration_angle_mae_then_nll_then_pivot_error",
            "selection_value": list(best_candidate),
            "model_state": best_state,
        },
    )
    result = {
        "selected_checkpoint": _binding(checkpoint_path),
        "selected_epoch": best_epoch,
        "selection_value": list(best_candidate),
        "history": history,
        "gradient_updates": sum(int(row["train"]["optimizer_steps"]) for row in history),
        "calibration_selection_queries": len(calibration.samples) * len(history),
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
    }
    return model, result


def _train_v5_epoch(
    *,
    backbone: torch.nn.Module,
    head: torch.nn.Module,
    dataset: v5_base.CanonicalTightROIDataset,
    optimizer: torch.optim.Optimizer,
    stage: str,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    offset = 0 if stage == "tick" else 1_000
    dataset.set_epoch(offset + epoch)
    loader = v5_enhanced._loader(
        dataset,
        args=args,
        shuffle=True,
        seed=int(args.seed) + offset + epoch,
        device=device,
    )
    backbone.eval()
    head.train()
    total = 0.0
    samples = 0
    augmented = 0
    optimizer_steps = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.no_grad():
            features = backbone.forward_multiscale_features(images)
            pooled = backbone.direction_features(features.c5)
            decoded = decode_probabilistic_pivot_direction(
                backbone.pivot_head(features.c5),
                backbone.vector_head(pooled),
                backbone.angle_head(pooled),
                backbone.log_variance_head(pooled),
            )
        output = head(features.c2.detach(), features.c5.detach())
        weight = batch["group_weight"].to(device, non_blocking=True)
        if stage == "tick":
            loss, _ = multiscale_dense_tick_loss(
                output,
                batch["tick_heatmap"].to(device, non_blocking=True),
                group_weight=weight,
            )
        else:
            pivot = decoded.pivot_xy / 63.0
            inverse = v5_enhanced._projective_local_inverse(
                batch["final_to_isotropic"].to(device, non_blocking=True), pivot
            )
            loss = v5_enhanced.enhanced_geometry_loss(
                output,
                batch["endpoints"].to(device, non_blocking=True),
                batch["gt_start"].to(device, non_blocking=True),
                batch["gt_range"].to(device, non_blocking=True),
                pivot,
                inverse,
                weight,
            )
        require(bool(torch.isfinite(loss)), f"enhanced-V5 {stage} loss non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in head.parameters() if parameter.requires_grad),
            5.0,
        )
        optimizer.step()
        count = int(images.shape[0])
        samples += count
        total += float(loss.detach()) * count
        augmented += int((batch["augmentation_code"] != 0).sum())
        optimizer_steps += 1
    require(samples == len(dataset), "enhanced-V5 epoch inventory drift")
    return {
        "epoch": int(epoch),
        "stage": stage,
        "loss": total / samples,
        "samples": samples,
        "augmented_fraction": augmented / samples,
        "optimizer_steps": optimizer_steps,
    }


def _save_head_journal(
    path: Path,
    *,
    signature: Mapping[str, Any],
    stage: str,
    stage_epoch: int,
    stage_total: int,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: Mapping[str, Sequence[Mapping[str, Any]]],
    best_state: Mapping[str, torch.Tensor],
    best_candidate: Sequence[float],
    best_stage_epoch: int,
    elapsed_seconds: float,
) -> None:
    _atomic_torch(
        path,
        {
            "schema_version": 1,
            "protocol": JOURNAL_PROTOCOL,
            "component": "matched_enhanced_v5_head",
            "status": "in_progress",
            "signature": dict(signature),
            "signature_sha256": canonical_sha256(signature),
            "stage": stage,
            "stage_epoch": int(stage_epoch),
            "stage_total": int(stage_total),
            "head_state": {
                name: tensor.detach().cpu() for name, tensor in head.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "history": {
                key: [dict(row) for row in rows] for key, rows in history.items()
            },
            "best_state": {
                name: tensor.detach().cpu() for name, tensor in best_state.items()
            },
            "best_candidate": [float(value) for value in best_candidate],
            "best_stage_epoch": int(best_stage_epoch),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state().cpu(),
                "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
            },
            "elapsed_seconds": float(elapsed_seconds),
        },
    )


def _restore_simple_rng(value: Mapping[str, Any]) -> None:
    require(isinstance(value, Mapping), "head resume RNG absent")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    cuda = value["torch_cuda"]
    if cuda:
        require(len(cuda) == torch.cuda.device_count(), "head resume CUDA topology drift")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def _head_candidate(metrics: Mapping[str, Any]) -> tuple[float, float]:
    return (float(metrics["full_denominator_nmae"]), -float(metrics["coverage"]))


def train_matched_v5(
    *,
    backbone: torch.nn.Module,
    pepd_result: Mapping[str, Any],
    fit: MaterializedPartition,
    calibration: MaterializedPartition,
    spec: Mapping[str, Any],
    signature: Mapping[str, Any],
    seed: int,
    device: torch.device,
    output: Path,
    resume: bool,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    args = _v5_args(spec, seed=seed)
    require(int(args.stage_a_epochs) > 0 and int(args.stage_b_epochs) > 0, "invalid enhanced-V5 schedule")
    require(int(args.batch_size) > 0 and 0 <= int(args.workers) <= 2, "invalid enhanced-V5 loader")
    augmentation = v5_base.augmentation_from_args(args)
    fit_dataset = v5_base.CanonicalTightROIDataset(
        fit.v5_records, training=True, seed=seed, augmentation=augmentation
    )
    calibration_dataset = v5_base.CanonicalTightROIDataset(
        calibration.v5_records,
        training=False,
        seed=seed,
        augmentation=v5_base.PhotoAugmentation.disabled(),
    )
    # This reset is the attested deterministic random initialization boundary.
    v5_base.seed_everything(seed)
    head = build_head().to(device)
    journal = output / "v5_enhanced_last.pt"
    history: dict[str, list[dict[str, Any]]] = {
        "stage_a_tick": [],
        "stage_b_geometry": [],
    }
    best_state: dict[str, torch.Tensor] = {}
    best_candidate = (math.inf, math.inf)
    best_stage_epoch = 0
    elapsed_before = 0.0
    resume_value: Mapping[str, Any] | None = None
    if resume and journal.is_file():
        value = torch.load(journal.resolve(strict=True), map_location="cpu", weights_only=False)
        require(isinstance(value, Mapping), "enhanced-V5 resume journal invalid")
        require(value.get("protocol") == JOURNAL_PROTOCOL, "enhanced-V5 journal protocol drift")
        require(value.get("component") == "matched_enhanced_v5_head", "enhanced-V5 journal component drift")
        require(value.get("signature_sha256") == canonical_sha256(signature), "enhanced-V5 journal signature drift")
        require(canonical_sha256(value.get("signature")) == canonical_sha256(signature), "enhanced-V5 embedded signature drift")
        head.load_state_dict(value["head_state"], strict=True)
        history = {key: list(rows) for key, rows in value["history"].items()}
        best_state = dict(value["best_state"])
        best_candidate = tuple(float(item) for item in value["best_candidate"])
        best_stage_epoch = int(value["best_stage_epoch"])
        elapsed_before = float(value.get("elapsed_seconds", 0.0))
        _restore_simple_rng(value["rng"])
        resume_value = value
    elif journal.exists():
        raise FileExistsError(f"enhanced-V5 journal exists; pass --resume: {journal}")
    started = time.monotonic()
    stages = (
        ("tick", int(args.stage_a_epochs), "stage_a_tick"),
        ("geometry", int(args.stage_b_epochs), "stage_b_geometry"),
    )
    for stage, total, key in stages:
        completed = len(history[key])
        require(completed <= total, f"enhanced-V5 {stage} history exceeds schedule")
        names = v5_base.set_stage(head, stage)
        require(bool(names), f"enhanced-V5 {stage} has no trainable parameters")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in head.parameters() if parameter.requires_grad),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        if (
            resume_value is not None
            and resume_value.get("stage") == stage
            and int(resume_value.get("stage_epoch", -1)) == completed
            and completed < total
        ):
            optimizer.load_state_dict(resume_value["optimizer_state"])
        for epoch in range(completed + 1, total + 1):
            row = _train_v5_epoch(
                backbone=backbone,
                head=head,
                dataset=fit_dataset,
                optimizer=optimizer,
                stage=stage,
                epoch=epoch,
                args=args,
                device=device,
            )
            if stage == "geometry":
                metrics, _ = v5_enhanced.evaluate_enhanced(
                    backbone,
                    head,
                    calibration_dataset,
                    args=args,
                    device=device,
                )
                row["calibration"] = metrics
                candidate = _head_candidate(metrics)
                if candidate < best_candidate:
                    best_candidate = candidate
                    best_stage_epoch = epoch
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in head.state_dict().items()
                    }
            history[key].append(row)
            _save_head_journal(
                journal,
                signature=signature,
                stage=stage,
                stage_epoch=epoch,
                stage_total=total,
                head=head,
                optimizer=optimizer,
                history=history,
                best_state=best_state,
                best_candidate=best_candidate,
                best_stage_epoch=best_stage_epoch,
                elapsed_seconds=elapsed_before + time.monotonic() - started,
            )
            print(
                f"common-split enhanced-V5 seed={seed} stage={stage} "
                f"epoch={epoch}/{total} loss={row['loss']:.6f}",
                flush=True,
            )
        resume_value = None
    require(best_state and best_stage_epoch > 0, "enhanced-V5 produced no selected geometry checkpoint")
    head.load_state_dict(best_state, strict=True)
    head.eval()
    fit_eval_dataset = v5_base.CanonicalTightROIDataset(
        fit.v5_records,
        training=False,
        seed=seed,
        augmentation=v5_base.PhotoAugmentation.disabled(),
    )
    fit_metrics, fit_rows = v5_enhanced.evaluate_enhanced(
        backbone, head, fit_eval_dataset, args=args, device=device
    )
    calibration_metrics, calibration_rows = v5_enhanced.evaluate_enhanced(
        backbone, head, calibration_dataset, args=args, device=device
    )
    checkpoint_path = output / "v5_enhanced_selected.pt"
    _atomic_torch(
        checkpoint_path,
        {
            "schema_version": 1,
            "protocol": SEED_BUNDLE_PROTOCOL,
            "component": "matched_enhanced_v5_head",
            "status": "complete",
            "seed": seed,
            "signature_sha256": canonical_sha256(signature),
            "matched_pepd_checkpoint": dict(pepd_result["selected_checkpoint"]),
            "selected_stage_epoch": best_stage_epoch,
            "selection_metric": "minimum_calibration_progress_mae_failure_penalty_1_then_coverage",
            "selection_value": list(best_candidate),
            "head_state": best_state,
        },
    )
    result = {
        "selected_checkpoint": _binding(checkpoint_path),
        "matched_pepd_checkpoint": dict(pepd_result["selected_checkpoint"]),
        "selected_stage_epoch": best_stage_epoch,
        "selection_value": list(best_candidate),
        "history": history,
        "fit_metrics": fit_metrics,
        "calibration_metrics": calibration_metrics,
        "gradient_updates": sum(
            int(row["optimizer_steps"])
            for rows in history.values()
            for row in rows
        ),
        "calibration_selection_queries": len(calibration.samples)
        * len(history["stage_b_geometry"]),
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
    }
    return head, result, fit_rows, calibration_rows


FUSION_FEATURES: Final[tuple[str, ...]] = (
    "predicted_progress",
    "gate_confidence",
    "uncertainty_score",
    "endpoint_peak_mean",
    "endpoint_entropy_mean",
    "endpoint_separation",
    "endpoint_coordinate_disagreement",
    "endpoint_js_divergence",
    "endpoint_consistency_reliability",
    "arc_length_reliability",
    "radius_reliability",
    "endpoint_tick_support",
)


def _fusion_features(row: Mapping[str, Any]) -> np.ndarray | None:
    if row.get("status") != "ok" or row.get("predicted_progress") is None:
        return None
    endpoint = row["endpoint"]
    reliability = row["reliability"]
    arc = row["arc"]
    peak = np.asarray(endpoint["peak"], dtype=np.float64)
    entropy = np.asarray(endpoint["entropy"], dtype=np.float64)
    values = np.asarray(
        [
            row["predicted_progress"],
            reliability["gate_confidence"],
            reliability["uncertainty_score"],
            float(peak.mean()),
            float(entropy.mean()),
            endpoint["separation"],
            endpoint["coordinate_disagreement"],
            endpoint["js_divergence"],
            endpoint["consistency_reliability"],
            arc["arc_length_reliability"],
            reliability["radius_reliability"],
            reliability["endpoint_tick_support"],
        ],
        dtype=np.float64,
    )
    return values if np.isfinite(values).all() else None


def fit_and_select_progress_fusion(
    *,
    fit_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
    output: Path,
) -> dict[str, Any]:
    fit_features: list[np.ndarray] = []
    fit_residual: list[float] = []
    for row in fit_rows:
        features = _fusion_features(row)
        if features is None:
            continue
        target = float(row["target_progress"])
        fit_features.append(features)
        fit_residual.append(target - float(row["predicted_progress"]))
    require(fit_features, "no valid algorithm-fit rows for progress fusion")
    matrix = np.stack(fit_features)
    target = np.asarray(fit_residual, dtype=np.float64)
    mean = matrix.mean(0)
    std = matrix.std(0)
    std = np.where(std > 1e-8, std, 1.0)
    standardized = (matrix - mean) / std
    design = np.concatenate((np.ones((len(matrix), 1)), standardized), axis=1)
    models: dict[float, np.ndarray] = {}
    for ridge in sorted({float(value) for value in config["ridge_lambdas"]}):
        require(ridge >= 0.0, "negative fusion ridge lambda")
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        models[ridge] = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    candidates: list[dict[str, Any]] = []
    for ridge, beta in models.items():
        for scale in sorted({float(value) for value in config["residual_scales"]}):
            for threshold in sorted(
                {float(value) for value in config["confidence_thresholds"]}
            ):
                require(0.0 <= threshold <= 1.0, "fusion confidence threshold outside [0,1]")
                errors: list[float] = []
                covered = 0
                for row in calibration_rows:
                    features = _fusion_features(row)
                    if features is None or float(row["reliability"]["gate_confidence"]) < threshold:
                        errors.append(1.0)
                        continue
                    x = np.concatenate(([1.0], (features - mean) / std))
                    prediction = float(
                        np.clip(
                            float(row["predicted_progress"])
                            + scale * float(x @ beta),
                            0.0,
                            1.0,
                        )
                    )
                    errors.append(abs(prediction - float(row["target_progress"])))
                    covered += 1
                candidates.append(
                    {
                        "ridge_lambda": ridge,
                        "residual_scale": scale,
                        "confidence_threshold": threshold,
                        "nmae_failure_penalty_1": float(np.mean(errors)),
                        "coverage": covered / len(calibration_rows),
                        "covered": covered,
                    }
                )
    require(candidates, "progress-fusion candidate grid is empty")
    selected = min(
        candidates,
        key=lambda row: (
            row["nmae_failure_penalty_1"],
            -row["coverage"],
            row["ridge_lambda"],
            abs(row["residual_scale"]),
            row["confidence_threshold"],
        ),
    )
    beta = models[float(selected["ridge_lambda"])]
    state = {
        "schema_version": 1,
        "protocol": SELECTION_PROTOCOL,
        "status": "fit_on_algorithm_fit_selected_on_calibration",
        "seed": seed,
        "family": "ridge_residual_plus_confidence_abstention",
        "feature_names": list(FUSION_FEATURES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "ridge_coefficients": beta.tolist(),
        "selected": selected,
        "candidate_grid": candidates,
        "fit": {
            "samples": len(fit_rows),
            "valid_samples": len(matrix),
            "target": "target_progress - raw_pepd_plus_v5_progress",
        },
        "calibration": {
            "samples": len(calibration_rows),
            "gradient_updates": 0,
            "selection_queries": len(candidates) * len(calibration_rows),
        },
        "mask_geometry_policy": config["mask_geometry_policy"],
        "mask_geometry_binding": config.get("mask_geometry_binding"),
    }
    state_path = output / "progress_fusion_state.json"
    _atomic_json(state_path, state)
    return {
        "state": _binding(state_path),
        "selected": selected,
        "fit_samples": len(fit_rows),
        "fit_valid_samples": len(matrix),
        "calibration_selection_queries": len(candidates) * len(calibration_rows),
    }


def _acquire_lock(path: Path, *, seed: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "seed": seed,
            "created_at": _utc_now(),
        },
        sort_keys=True,
    ).encode("utf-8")
    os.write(descriptor, payload)
    os.fsync(descriptor)
    return descriptor


def _release_lock(path: Path, descriptor: int) -> None:
    os.close(descriptor)
    path.unlink(missing_ok=True)


def _seed_summary_path(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}" / "summary.json"


def _strict_seed_summary(
    path: Path,
    *,
    expected_seed: int,
    expected_signature_sha256: str | None = None,
) -> dict[str, Any]:
    summary = strict_json(path)
    require(summary.get("protocol") == SEED_SUMMARY_PROTOCOL, "seed summary protocol drift")
    require(summary.get("status") == "complete", "seed summary incomplete")
    require(int(summary.get("seed", -1)) == expected_seed, "seed summary seed drift")
    if expected_signature_sha256 is not None:
        require(summary.get("signature_sha256") == expected_signature_sha256, "seed summary signature drift")
    artifacts = summary.get("artifacts")
    require(isinstance(artifacts, Mapping), "seed summary artifacts absent")
    for name in (
        "bundle_checkpoint",
        "pepd_checkpoint",
        "v5_enhanced_checkpoint",
        "progress_fusion_state",
    ):
        path_value = _verify_binding(artifacts.get(name), label=f"seed {expected_seed} {name}")
        require(sha256_file(path_value) not in FORBIDDEN_LEGACY_HASHES, f"seed {expected_seed} reuses legacy {name}")
    bundle_path = Path(str(artifacts["bundle_checkpoint"]["path"]))
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    require(isinstance(bundle, Mapping), "seed bundle checkpoint invalid")
    require(bundle.get("protocol") == SEED_BUNDLE_PROTOCOL, "seed bundle protocol drift")
    require(bundle.get("status") == "complete", "seed bundle incomplete")
    require(int(bundle.get("seed", -1)) == expected_seed, "seed bundle seed drift")
    require(bundle.get("signature_sha256") == summary.get("signature_sha256"), "seed bundle signature drift")
    require(isinstance(bundle.get("pepd_model_state"), Mapping) and bundle["pepd_model_state"], "seed bundle PEPD state absent")
    require(isinstance(bundle.get("v5_head_state"), Mapping) and bundle["v5_head_state"], "seed bundle V5 state absent")
    require(isinstance(bundle.get("progress_fusion_state"), Mapping), "seed bundle fusion state absent")
    require(
        bundle.get("component_sha256", {}).get("pepd_checkpoint")
        == artifacts["pepd_checkpoint"]["sha256"],
        "seed bundle PEPD mutual binding drift",
    )
    require(
        bundle.get("component_sha256", {}).get("v5_enhanced_checkpoint")
        == artifacts["v5_enhanced_checkpoint"]["sha256"],
        "seed bundle V5 mutual binding drift",
    )
    require(
        bundle.get("component_sha256", {}).get("progress_fusion_state")
        == artifacts["progress_fusion_state"]["sha256"],
        "seed bundle fusion mutual binding drift",
    )
    return summary


def train_seed(
    *,
    protocol_path: Path,
    execution_spec_path: Path,
    authorization_path: Path,
    promotion_seal_path: Path,
    output_root: Path,
    seed: int,
    device_name: str,
    resume: bool,
) -> Path:
    require(seed in common.EXPECTED_SEEDS, "seed outside frozen common-split roster")
    promotion = authenticate_promotion(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        promotion_seal_path=promotion_seal_path,
    )
    protocol_file = promotion["protocol_file"]
    protocol = promotion["protocol"]
    spec_file, spec = load_execution_spec(
        execution_spec_path, protocol_file=protocol_file, protocol=protocol
    )
    root = _guard_output(output_root, label="fresh common-split output root")
    require(root == DEFAULT_OUTPUT_ROOT.resolve(), "formal output root drift")
    seed_root = root / "seeds" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    summary_path = seed_root / "summary.json"
    if summary_path.is_file():
        _strict_seed_summary(summary_path, expected_seed=seed)
        return summary_path
    lock_path = seed_root / "run.lock"
    descriptor = _acquire_lock(lock_path, seed=seed)
    try:
        device = torch.device(device_name)
        _configure_determinism(seed, device)
        fit = materialize_partition(
            protocol_path=protocol_file,
            protocol=protocol,
            partition="algorithm_fit",
        )
        calibration = materialize_partition(
            protocol_path=protocol_file,
            protocol=protocol,
            partition="calibration",
        )
        data_signature = _data_signature(fit, calibration)
        inventory_path = seed_root / "data_inventory.json"
        if inventory_path.is_file():
            require(strict_json(inventory_path) == data_signature, "seed data inventory drift")
        else:
            _atomic_json(inventory_path, data_signature)
        signature = _training_signature(
            seed=seed,
            protocol_file=protocol_file,
            spec_file=spec_file,
            authorization_file=promotion["authorization_file"],
            promotion_seal_file=promotion["seal_file"],
            data_signature=data_signature,
        )
        signature_sha = canonical_sha256(signature)
        started_at = _utc_now()
        started = time.monotonic()
        backbone, pepd_result = train_fresh_pepd(
            fit=fit,
            calibration=calibration,
            spec=spec,
            signature=signature,
            seed=seed,
            device=device,
            output=seed_root,
            resume=resume,
        )
        head, v5_result, fit_rows, calibration_rows = train_matched_v5(
            backbone=backbone,
            pepd_result=pepd_result,
            fit=fit,
            calibration=calibration,
            spec=spec,
            signature=signature,
            seed=seed,
            device=device,
            output=seed_root,
            resume=resume,
        )
        del head
        fusion_result = fit_and_select_progress_fusion(
            fit_rows=fit_rows,
            calibration_rows=calibration_rows,
            config=spec["progress_fusion"],
            seed=seed,
            output=seed_root,
        )
        fusion_state = strict_json(Path(fusion_result["state"]["path"]))
        pepd_payload = torch.load(
            Path(pepd_result["selected_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        v5_payload = torch.load(
            Path(v5_result["selected_checkpoint"]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        bundle_path = seed_root / "seed_bundle.pt"
        _atomic_torch(
            bundle_path,
            {
                "schema_version": 1,
                "protocol": SEED_BUNDLE_PROTOCOL,
                "status": "complete",
                "seed": seed,
                "signature": signature,
                "signature_sha256": signature_sha,
                "component_sha256": {
                    "pepd_checkpoint": pepd_result["selected_checkpoint"]["sha256"],
                    "v5_enhanced_checkpoint": v5_result["selected_checkpoint"]["sha256"],
                    "progress_fusion_state": fusion_result["state"]["sha256"],
                },
                "pepd_model_state": pepd_payload["model_state"],
                "v5_head_state": v5_payload["head_state"],
                "progress_fusion_state": fusion_state,
                "matched_feature_distribution_attested": True,
                "legacy_checkpoint_weights_loaded": 0,
            },
        )
        require(sha256_file(bundle_path) not in FORBIDDEN_LEGACY_HASHES, "seed bundle aliases legacy weights")
        completed_at = _utc_now()
        summary = {
            "schema_version": 1,
            "protocol": SEED_SUMMARY_PROTOCOL,
            "status": "complete",
            "seed": seed,
            "started_at": started_at,
            "completed_at": completed_at,
            "signature": signature,
            "signature_sha256": signature_sha,
            "component_roles": {
                "pepd_backbone": "trained_on_algorithm_fit",
                "v5_enhanced_head": "trained_on_algorithm_fit_with_this_exact_pepd_backbone",
                "progress_fusion": "fit_on_algorithm_fit_selected_on_calibration",
            },
            "matched_feature_distribution_attested": True,
            "pepd": pepd_result,
            "v5_enhanced": v5_result,
            "progress_fusion": fusion_result,
            "artifacts": {
                "bundle_checkpoint": _binding(bundle_path),
                "pepd_checkpoint": dict(pepd_result["selected_checkpoint"]),
                "v5_enhanced_checkpoint": dict(v5_result["selected_checkpoint"]),
                "progress_fusion_state": dict(fusion_result["state"]),
                "data_inventory": _binding(inventory_path),
            },
            "inventory": {
                "algorithm_fit": {
                    **{
                        key: protocol["partitions"]["algorithm_fit"][key]
                        for key in ("samples", "groups", "sha256", "sample_ids_sha256", "group_ids_sha256")
                    },
                    "images_read": len(fit.samples),
                    "annotations_read": len(fit.samples),
                    "gradient_updates": pepd_result["gradient_updates"]
                    + v5_result["gradient_updates"],
                    "selection_queries": 0,
                },
                "calibration": {
                    **{
                        key: protocol["partitions"]["calibration"][key]
                        for key in ("samples", "groups", "sha256", "sample_ids_sha256", "group_ids_sha256")
                    },
                    "images_read": len(calibration.samples),
                    "annotations_read": len(calibration.samples),
                    "gradient_updates": 0,
                    "selection_queries": pepd_result["calibration_selection_queries"]
                    + v5_result["calibration_selection_queries"]
                    + fusion_result["calibration_selection_queries"],
                },
                "independent_validation": {
                    "images_read": 0,
                    "annotations_read": 0,
                    "gradient_updates": 0,
                    "selection_queries": 0,
                },
                "development_excluded": {
                    "images_read": 0,
                    "annotations_read": 0,
                    "gradient_updates": 0,
                    "selection_queries": 0,
                },
            },
            "initialization": [
                {
                    "component": "pepd_backbone",
                    "provenance": "external_generic_immutable",
                    "artifact": dict(spec["pepd"]["imagenet_initialization"]),
                },
                {
                    "component": "v5_enhanced_head",
                    "provenance": "deterministic_random_init",
                    "artifact": None,
                    "random_seed": seed,
                },
                {
                    "component": "progress_fusion",
                    "provenance": "deterministic_random_init",
                    "artifact": None,
                    "random_seed": seed,
                },
            ],
            "audit": {
                "legacy_pepd_checkpoint_weights_loaded": 0,
                "legacy_v5_checkpoint_weights_loaded": 0,
                "independent_validation_images_read": 0,
                "independent_validation_annotations_read": 0,
                "development_excluded_images_read": 0,
                "development_excluded_annotations_read": 0,
                "field_images_read": 0,
                "test_split_images_read": 0,
                "sealed_images_read": 0,
                "confirmatory_images_read": 0,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        _atomic_json(summary_path, summary)
        _strict_seed_summary(
            summary_path,
            expected_seed=seed,
            expected_signature_sha256=signature_sha,
        )
        return summary_path
    finally:
        _release_lock(lock_path, descriptor)


def finalize_training_run(
    *,
    protocol_path: Path,
    execution_spec_path: Path,
    authorization_path: Path,
    promotion_seal_path: Path,
    output_root: Path,
) -> Path:
    promotion = authenticate_promotion(
        protocol_path=protocol_path,
        authorization_path=authorization_path,
        promotion_seal_path=promotion_seal_path,
    )
    protocol_file = promotion["protocol_file"]
    protocol = promotion["protocol"]
    spec_file, _ = load_execution_spec(
        execution_spec_path, protocol_file=protocol_file, protocol=protocol
    )
    root = _guard_output(output_root, label="fresh common-split output root")
    require(root == DEFAULT_OUTPUT_ROOT.resolve(), "formal output root drift")
    run_path = root / "run_manifest.json"
    require(not run_path.exists(), "refusing to overwrite frozen training run manifest")
    summaries: list[tuple[Path, dict[str, Any]]] = []
    for seed in common.EXPECTED_SEEDS:
        path = _seed_summary_path(root, seed)
        summaries.append((path, _strict_seed_summary(path, expected_seed=seed)))
    selection = {
        "schema_version": 1,
        "protocol": SELECTION_PROTOCOL,
        "status": "all_seed_bundles_selected_on_calibration_only",
        "common_protocol": _binding(protocol_file),
        "execution_spec": _binding(spec_file),
        "seeds": [
            {
                "seed": int(summary["seed"]),
                "summary": _binding(path),
                "bundle_checkpoint": dict(summary["artifacts"]["bundle_checkpoint"]),
                "pepd_selected_epoch": summary["pepd"]["selected_epoch"],
                "v5_selected_stage_epoch": summary["v5_enhanced"]["selected_stage_epoch"],
                "fusion_selected": summary["progress_fusion"]["selected"],
            }
            for path, summary in summaries
        ],
        "selection_partition": "calibration",
        "independent_validation_queries": 0,
        "frozen_at": _utc_now(),
    }
    selection_path = root / "selection_artifact.json"
    atomic_new_json(selection_path, selection)
    ensemble = {
        "schema_version": 1,
        "protocol": ENSEMBLE_PROTOCOL,
        "status": "frozen_equal_weight_three_seed_ensemble",
        "aggregation": "equal_weight_mean_of_available_seed_progress_posteriors",
        "failure_policy": "failure_only_when_all_three_seed_bundles_abstain_or_fail",
        "members": [
            {
                "seed": int(summary["seed"]),
                "weight": 1.0 / 3.0,
                "checkpoint": dict(summary["artifacts"]["bundle_checkpoint"]),
            }
            for _, summary in summaries
        ],
        "selection_partition": "calibration",
        "independent_validation_queries": 0,
        "frozen_at": _utc_now(),
    }
    ensemble_path = root / "ensemble_binding.json"
    atomic_new_json(ensemble_path, ensemble)
    seed_runs = []
    for summary_path, summary in summaries:
        fit_inventory = dict(summary["inventory"]["algorithm_fit"])
        cal_inventory = dict(summary["inventory"]["calibration"])
        seed_runs.append(
            {
                "seed": int(summary["seed"]),
                "status": "complete",
                "checkpoint": dict(summary["artifacts"]["bundle_checkpoint"]),
                "training_summary": _binding(summary_path),
                "component_roles": dict(summary["component_roles"]),
                "matched_feature_distribution_attested": True,
                "inventory": {
                    "algorithm_fit": fit_inventory,
                    "calibration": cal_inventory,
                    "independent_validation": dict(summary["inventory"]["independent_validation"]),
                    "development_excluded": dict(summary["inventory"]["development_excluded"]),
                },
                "initialization": list(summary["initialization"]),
                "completed_at": summary["completed_at"],
            }
        )
    frozen_at = _utc_now()
    run = {
        "schema_version": 1,
        "protocol": common.TRAINING_RUN_PROTOCOL,
        "status": "training_and_calibration_frozen",
        "parent_protocol": _binding(protocol_file),
        "execution_spec": _binding(spec_file),
        "promotion_seal": _binding(promotion["seal_file"]),
        "pilot_gate_report": _binding(promotion["gate_file"]),
        "promotion_authorization": _binding(promotion["authorization_file"]),
        "partitions": {
            name: {
                key: binding[key]
                for key in ("samples", "groups", "sha256", "sample_ids_sha256", "group_ids_sha256")
            }
            for name, binding in protocol["partitions"].items()
        },
        "seed_runs": seed_runs,
        "selection_artifact": _binding(selection_path),
        "ensemble_binding": _binding(ensemble_path),
        "chronology": {
            "selection_frozen_at": frozen_at,
            "independent_validation_first_opened_at": None,
        },
        "audit": {
            "gradient_partition": "algorithm_fit",
            "selection_partition": "calibration",
            "independent_validation_reveal_count": 0,
            "independent_validation_images_read": 0,
            "independent_validation_annotations_read": 0,
            "independent_validation_selection_queries": 0,
            "development_excluded_images_read": 0,
            "field_images_read": 0,
            "test_split_images_read": 0,
            "sealed_images_read": 0,
            "confirmatory_images_read": 0,
            "matched_v5_head_trained_with_same_common_split_pepd": True,
            "legacy_pepd_checkpoint_weights_loaded": 0,
            "legacy_v5_checkpoint_weights_loaded": 0,
        },
    }
    atomic_new_json(run_path, run)
    # Compatibility is verified against the authoritative downstream gate;
    # the stricter seed/bundle checks above run first.
    common.validate_training_run(protocol_file, protocol, run_path)
    return run_path


def preflight(
    *,
    protocol_path: Path,
    output_path: Path,
    execution_spec_path: Path | None = None,
    authorization_path: Path | None = None,
    promotion_seal_path: Path | None = None,
) -> Path:
    protocol_file, protocol = common.load_protocol(protocol_path)
    gaps = execution_spec_gaps(protocol)
    execution_spec = None
    promotion = None
    if execution_spec_path is not None:
        spec_file, _ = load_execution_spec(
            execution_spec_path, protocol_file=protocol_file, protocol=protocol
        )
        execution_spec = _binding(spec_file)
    if authorization_path is not None or promotion_seal_path is not None:
        require(
            authorization_path is not None and promotion_seal_path is not None,
            "authorization and promotion seal must be supplied together",
        )
        authenticated = authenticate_promotion(
            protocol_path=protocol_file,
            authorization_path=authorization_path,
            promotion_seal_path=promotion_seal_path,
        )
        promotion = {
            "authorization": _binding(authenticated["authorization_file"]),
            "seal": _binding(authenticated["seal_file"]),
        }
    ready = execution_spec is not None and promotion is not None
    result = {
        "schema_version": 1,
        "protocol": PREFLIGHT_PROTOCOL,
        "status": (
            "ready_no_data_no_gpu_no_training"
            if ready
            else "blocked_before_training_execution_contract_or_promotion_missing"
        ),
        "training_allowed": ready,
        "common_protocol": _binding(protocol_file),
        "common_protocol_execution_gaps": gaps,
        "resolution": (
            "a separately frozen execution spec may resolve the common-protocol gaps"
        ),
        "execution_spec": execution_spec,
        "promotion": promotion,
        "audit": {
            "gpu_initialized": False,
            "training_started": False,
            "inference_started": False,
            "images_opened": 0,
            "annotations_opened": 0,
            "independent_validation_images_opened": 0,
            "development_excluded_images_opened": 0,
            "field_test_sealed_confirmatory_images_opened": 0,
            "process_wait_started": False,
            "feishu_message_sent": False,
        },
    }
    output = guard_public_path(output_path, label="fresh-training preflight output", must_exist=False)
    _atomic_json(output, result)
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--protocol", type=Path, default=common.DEFAULT_PROTOCOL)
    pre.add_argument("--execution-spec", type=Path)
    pre.add_argument("--authorization", type=Path)
    pre.add_argument("--promotion-seal", type=Path)
    pre.add_argument("--output", type=Path, required=True)
    seed = commands.add_parser("train-seed")
    seed.add_argument("--protocol", type=Path, default=common.DEFAULT_PROTOCOL)
    seed.add_argument("--execution-spec", type=Path, required=True)
    seed.add_argument("--authorization", type=Path, required=True)
    seed.add_argument("--promotion-seal", type=Path, required=True)
    seed.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    seed.add_argument("--seed", type=int, choices=common.EXPECTED_SEEDS, required=True)
    seed.add_argument("--device", default="cuda:0")
    seed.add_argument("--resume", action="store_true")
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--protocol", type=Path, default=common.DEFAULT_PROTOCOL)
    finalize.add_argument("--execution-spec", type=Path, required=True)
    finalize.add_argument("--authorization", type=Path, required=True)
    finalize.add_argument("--promotion-seal", type=Path, required=True)
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "preflight":
        output = preflight(
            protocol_path=args.protocol,
            execution_spec_path=args.execution_spec,
            authorization_path=args.authorization,
            promotion_seal_path=args.promotion_seal,
            output_path=args.output,
        )
    elif args.command == "train-seed":
        output = train_seed(
            protocol_path=args.protocol,
            execution_spec_path=args.execution_spec,
            authorization_path=args.authorization,
            promotion_seal_path=args.promotion_seal,
            output_root=args.output_root,
            seed=args.seed,
            device_name=args.device,
            resume=args.resume,
        )
    else:
        output = finalize_training_run(
            protocol_path=args.protocol,
            execution_spec_path=args.execution_spec,
            authorization_path=args.authorization,
            promotion_seal_path=args.promotion_seal,
            output_root=args.output_root,
        )
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "EXECUTION_SPEC_PROTOCOL",
    "PREFLIGHT_PROTOCOL",
    "authenticate_promotion",
    "execution_spec_gaps",
    "finalize_training_run",
    "fit_and_select_progress_fusion",
    "load_execution_spec",
    "materialize_partition",
    "preflight",
    "train_seed",
]
