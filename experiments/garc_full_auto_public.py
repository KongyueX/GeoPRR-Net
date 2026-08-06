"""Frozen, label-free public runner for GARC plus a full-auto progress model.

The inference command consumes only the three label-free partitions frozen by
``automatic_numeric_range_public_group_holdout_v1``.  SyncG dial boxes are used
only to prepare the common canonical ROI outside the method boundary.  The
method itself receives one unchanged ROI and never accepts a numeric range,
manual geometry, pointer annotation, or reference packet.

There are two deliberately separate phases:

* ``freeze-plan`` binds the public split manifests, model files, provider
  factory, implementation sources, recognizer family, and consensus ablation;
* ``infer`` authenticates that plan, produces label-free predictions, and
  seals them before any evaluator may open public annotations.

No command in this module accepts a label path.  ``development_excluded`` and
all field/test/sealed namespaces are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

_PROJECT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_PROJECT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_PROJECT_BOOTSTRAP))

import experiments.train_cagh_scalemark_reference_probe_v5 as public_v5
from experiments.automatic_numeric_range import (
    AutomaticNumericRangePipeline,
    NumericRangePrediction,
    image_sha256,
)
from experiments.automatic_numeric_range_public_protocol import (
    atomic_new_json,
    atomic_new_jsonl,
    canonical_bytes,
    canonical_sha256,
    guard_public_path,
    load_frozen_protocol,
    load_partition_roster,
    require,
    resolve_public_image,
    sha256_file,
    strict_json,
    strict_jsonl,
    verify_bound_file,
)
from experiments.garc_numeric_range_bridge import (
    GARCAutomaticNumericRangeProvider,
)
from experiments.garc_posterior_consensus import (
    GARCConsensusConfig,
    GARCPosteriorConsensusDecoder,
    TOP1_PROTOCOL,
    PROTOCOL as TOPK_PROTOCOL,
)
from experiments.screen_automatic_numeric_range_public import (
    FrozenPublicScaleMarkGeometryProvider,
)
from experiments.syncg_numeric_ocr import GaugeNumericOCRBackend
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    EXECUTION_SYNTHETIC,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    FullAutoPrediction,
    ProtocolViolation,
    UnifiedFullAutoAdapter,
)
from experiments.v5_unified_two_stage_retest import (
    REFERENCE_MODE_AUTO,
    REFERENCE_MODE_NATIVE,
    assert_label_free,
)


PLAN_PROTOCOL: Final[str] = "garc_full_auto_public_plan_v1"
PREDICTION_PROTOCOL: Final[str] = "garc_full_auto_public_predictions_v1"
RANGE_ADAPTER_PROTOCOL: Final[str] = "garc_full_auto_range_adapter_v1"
JOINT_OOF_ROW_PROTOCOL: Final[str] = "garc_progress_joint_oof_route_v1"
JOINT_OOF_SUMMARY_PROTOCOL: Final[str] = "garc_progress_joint_oof_cohort_v1"
DEFAULT_JOINT_OOF_ROOT: Final[Path] = Path(
    r"C:\pointer_read\garc_full_auto_public_v1\manifests"
)
EXPECTED_RANGE_INDEPENDENT: Final[tuple[int, int]] = (1_080, 50)
EXPECTED_JOINT_OOF: Final[tuple[int, int]] = (412, 19)
EXPECTED_JOINT_OOF_BY_SEED: Final[dict[int, int]] = {
    20260720: 168,
    20260721: 116,
    20260722: 128,
}
EXPECTED_JOINT_OOF_GROUPS_BY_SEED: Final[dict[int, int]] = {
    20260720: 8,
    20260721: 5,
    20260722: 6,
}
EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY: Final[dict[str, int]] = {
    "geometry_unseen_samples": 168,
    "geometry_unseen_groups": 8,
    "geometry_fit_overlap_samples": 912,
    "geometry_fit_overlap_groups": 42,
}
SUPPORTED_PARTITIONS: Final[tuple[str, ...]] = (
    "algorithm_fit",
    "calibration",
    "independent_validation",
)
RECOGNIZER_KINDS: Final[frozenset[str]] = frozenset({"tiny", "strong"})
CONSENSUS_MODES: Final[frozenset[str]] = frozenset({"top1", "topk"})
GEOMETRY_MODES: Final[frozenset[str]] = frozenset(
    {"v5", "v5_pepd_fusion", "v5_pepd_base_fusion"}
)
GEOMETRY_PROVIDERS: Final[frozenset[str]] = frozenset(
    {"legacy_fixed_v5", "enhanced_v5_oof_fold"}
)
BASE_POINTER_SEGMENTATION: Final[Path] = (
    _PROJECT_BOOTSTRAP / "utils/angleDetect/pointerSeg/resultSeg/best.pt"
).resolve()
BASE_REFERENCE_DETECTOR: Final[Path] = (
    _PROJECT_BOOTSTRAP / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt"
).resolve()
EXECUTION_SCOPE_FULL: Final[str] = "full_partition"
EXECUTION_SCOPE_JOINT_SHARD: Final[str] = "joint_oof_seed_shard"
EXECUTION_SCOPES: Final[frozenset[str]] = frozenset(
    {EXECUTION_SCOPE_FULL, EXECUTION_SCOPE_JOINT_SHARD}
)
RUNNER_SOURCE: Final[Path] = Path(__file__).resolve()
EVALUATOR_SOURCE: Final[Path] = RUNNER_SOURCE.with_name(
    "evaluate_garc_full_auto_public.py"
)
FACTORY_FUNCTION_DEFAULT: Final[str] = "build_progress_provider"

_PLAN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "status",
        "execution_mode",
        "claim_scope",
        "parent_protocol",
        "partitions",
        "method",
        "progress_component",
        "progress_factory",
        "joint_oof",
        "reference",
        "garc",
        "code_bindings",
        "audit",
    }
)
_PREDICTION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "partition",
        "sample_id",
        "group_id",
        "canonical_roi_sha256",
        "full_status",
        "range_status",
        "prediction_progress",
        "predicted_scale_start",
        "predicted_scale_end",
        "predicted_reading",
        "range_confidence",
        "failure_reason",
        "range_failure_reason",
        "progress_checkpoint_sha256",
        "geometry_head_checkpoint_sha256",
        "geometry_backbone_checkpoint_sha256",
        "geometry_oof_seed",
        "progress_group_unseen",
        "geometry_group_unseen",
        "joint_oof_eligible",
        "method_record",
        "range_prediction",
        "sample_seconds",
    }
)


def _sha256(value: Any, label: str) -> str:
    digest = str(value or "").strip().casefold()
    require(
        len(digest) == 64 and set(digest).issubset(set("0123456789abcdef")),
        f"{label} is not a lowercase SHA-256 digest",
    )
    return digest


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested_checkpoint_hashes(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = "".join(
                character for character in str(key).casefold() if character.isalnum()
            )
            if "checkpoint" in normalized and normalized.endswith("sha256"):
                try:
                    result.add(_sha256(nested, f"progress identity {key}"))
                except ValueError:
                    pass
            result.update(_nested_checkpoint_hashes(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            result.update(_nested_checkpoint_hashes(nested))
    return result


def progress_checkpoint_hashes(binding: FrozenComponentBinding) -> set[str]:
    """Return only checkpoint-like progress hashes, excluding generic sources."""

    values = {
        _sha256(value, f"progress artifact {key}")
        for key, value in binding.artifact_sha256.items()
        if "checkpoint" in "".join(
            character for character in str(key).casefold() if character.isalnum()
        )
        or "model" in str(key).casefold()
    }
    values.update(_nested_checkpoint_hashes(binding.provider_identity))
    return values


def _bound_file(path: Path, *, label: str) -> dict[str, Any]:
    resolved = guard_public_path(Path(path), label=label)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _source_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "runner": RUNNER_SOURCE,
        "evaluator": EVALUATOR_SOURCE,
        "garc_bridge": RUNNER_SOURCE.with_name("garc_numeric_range_bridge.py"),
        "garc_consensus": RUNNER_SOURCE.with_name("garc_posterior_consensus.py"),
        "garc_geometry_fusion": RUNNER_SOURCE.with_name("garc_geometry_fusion.py"),
        "garc_base_geometry": RUNNER_SOURCE.with_name(
            "garc_base_mask_geometry_provider.py"
        ),
        "tiny_ocr": RUNNER_SOURCE.with_name("syncg_numeric_ocr.py"),
        "strong_ocr": RUNNER_SOURCE.with_name("syncg_strong_numeric_ocr.py"),
        "numeric_range_core": RUNNER_SOURCE.with_name("automatic_numeric_range.py"),
        "geometry_provider": RUNNER_SOURCE.with_name(
            "screen_automatic_numeric_range_public.py"
        ),
        "enhanced_v5_oof_geometry_provider": RUNNER_SOURCE.with_name(
            "garc_enhanced_v5_oof_geometry_provider.py"
        ),
        "enhanced_v5_head": RUNNER_SOURCE.with_name(
            "cagh_scalemark_reference_head_v5.py"
        ),
        "canonical_roi": RUNNER_SOURCE.with_name(
            "train_cagh_scalemark_reference_probe_v5.py"
        ),
        "full_auto_adapter": RUNNER_SOURCE.with_name(
            "v5_unified_full_auto_adapter.py"
        ),
        "progress_wrappers": RUNNER_SOURCE.with_name(
            "v5_unified_full_auto_progress_providers.py"
        ),
        "legacy_adapters": RUNNER_SOURCE.with_name(
            "v5_unified_legacy_adapters.py"
        ),
        "shared_public_protocol": RUNNER_SOURCE.with_name(
            "automatic_numeric_range_public_protocol.py"
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = guard_public_path(path, label=f"code binding {name}")
        result[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return result


def freeze_joint_oof_cohort(
    *,
    protocol_path: Path,
    output_root: Path = DEFAULT_JOINT_OOF_ROOT,
    input_size: int = 768,
) -> Path:
    """Rebuild the exact range/progress jointly-unseen public cohort.

    The progress assignment is obtained from the already frozen authoritative
    PEPD grouped holdouts.  It is intersected with the 1,080-row range
    independent-validation roster.  Only public images are opened, solely to
    bind the deterministic canonical-ROI pixel hash; no annotation value is
    read by this function.
    """

    require(int(input_size) in (512, 768), "input_size must be 512 or 768")
    protocol_file, _ = load_frozen_protocol(protocol_path)
    _, range_manifest, range_rows, range_audit = load_partition_roster(
        protocol_file, "independent_validation"
    )
    require(
        (range_audit["samples"], range_audit["groups"])
        == EXPECTED_RANGE_INDEPENDENT,
        "range independent-validation inventory drift",
    )

    # Importing here keeps the ordinary label-free inference module free from
    # the authoritative training-discovery dependency.
    import experiments.run_cagh_v5_enhanced_oof as progress_oof

    oof_protocol = progress_oof.load_protocol()
    discovery = progress_oof.discover(oof_protocol)
    fold_by_seed = {fold.pepd_seed: fold for fold in discovery.folds}
    range_by_id = {str(row["sample_id"]): row for row in range_rows}
    joint_ids = sorted(set(range_by_id) & set(discovery.sample_assignment))
    cohort_rows = [dict(range_by_id[sample_id]) for sample_id in joint_ids]
    require(
        (len(cohort_rows), len({str(row["group_id"]) for row in cohort_rows}))
        == EXPECTED_JOINT_OOF,
        "joint range/progress OOF inventory drift",
    )

    seed_counts: Counter[int] = Counter()
    mapping_rows: list[dict[str, Any]] = []
    for roster_row in cohort_rows:
        sample_id = str(roster_row["sample_id"])
        group_id = str(roster_row["group_id"])
        seed = int(discovery.sample_assignment[sample_id])
        require(
            discovery.group_assignment[group_id] == seed,
            f"sample/group OOF assignment drift: {sample_id}",
        )
        fold = fold_by_seed[seed]
        require(group_id in fold.validation_groups, f"OOF group not held out: {group_id}")
        require(group_id not in fold.train_groups, f"OOF group leaks training: {group_id}")
        image_path = resolve_public_image(str(roster_row["image_relpath"]))
        image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        require(image is not None, f"cannot read public image: {image_path}")
        roi, _, _, _ = public_v5.canonical_tight_roi(
            image, roster_row["dial_bbox"], output_size=int(input_size)
        )
        seed_counts[seed] += 1
        mapping_rows.append(
            {
                "schema_version": 1,
                "protocol": JOINT_OOF_ROW_PROTOCOL,
                "partition": "independent_validation",
                "sample_id": sample_id,
                "group_id": group_id,
                "pepd_seed": seed,
                "pepd_checkpoint_sha256": fold.checkpoint_sha256,
                "canonical_roi_sha256": image_sha256(roi),
                "group_unseen_by_progress_checkpoint": True,
            }
        )
    require(
        dict(seed_counts) == EXPECTED_JOINT_OOF_BY_SEED,
        "joint OOF seed allocation drift",
    )
    assert_label_free(cohort_rows, location="joint_oof_cohort")
    assert_label_free(mapping_rows, location="joint_oof_mapping")

    output = guard_public_path(
        output_root,
        label="joint OOF manifest output",
        must_exist=False,
        expect_file=False,
    )
    require(not output.exists(), f"refusing to overwrite joint OOF output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    cohort_path = output / "joint_oof_independent_validation.label_free.jsonl"
    mapping_path = output / "joint_oof_progress_mapping.label_free.jsonl"
    atomic_new_jsonl(cohort_path, cohort_rows)
    atomic_new_jsonl(mapping_path, mapping_rows)
    summary = {
        "schema_version": 1,
        "protocol": JOINT_OOF_SUMMARY_PROTOCOL,
        "status": "frozen_label_free_joint_oof_cohort",
        "input_size": int(input_size),
        "range_independent_validation": {
            "samples": EXPECTED_RANGE_INDEPENDENT[0],
            "groups": EXPECTED_RANGE_INDEPENDENT[1],
            "manifest_path": str(range_manifest),
            "manifest_sha256": sha256_file(range_manifest),
        },
        "joint_oof": {
            "samples": EXPECTED_JOINT_OOF[0],
            "groups": EXPECTED_JOINT_OOF[1],
            "samples_by_pepd_seed": {
                str(key): value for key, value in EXPECTED_JOINT_OOF_BY_SEED.items()
            },
            "sample_ids_sha256": canonical_sha256(joint_ids),
            "group_ids_sha256": canonical_sha256(
                sorted({str(row["group_id"]) for row in cohort_rows})
            ),
        },
        "artifacts": {
            "cohort": {
                "path": str(cohort_path),
                "sha256": sha256_file(cohort_path),
            },
            "mapping": {
                "path": str(mapping_path),
                "sha256": sha256_file(mapping_path),
            },
        },
        "provenance": {
            "range_protocol": {
                "path": str(protocol_file),
                "sha256": sha256_file(protocol_file),
            },
            "progress_oof_protocol": {
                "path": str(progress_oof.PROTOCOL_PATH.resolve(strict=True)),
                "sha256": sha256_file(progress_oof.PROTOCOL_PATH),
            },
            "progress_oof_runner": {
                "path": str(Path(progress_oof.__file__).resolve(strict=True)),
                "sha256": sha256_file(Path(progress_oof.__file__)),
            },
            "sample_assignment_sha256": discovery.audit[
                "sample_assignment_sha256"
            ],
            "assignment_rule": discovery.audit["assignment_rule"],
        },
        "audit": {
            "range_progress_group_overlap_in_joint_rows": 0,
            "all_joint_rows_progress_group_unseen": True,
            "range_only_rows": EXPECTED_RANGE_INDEPENDENT[0] - EXPECTED_JOINT_OOF[0],
            "end_to_end_claim_limited_to_joint_oof": True,
            "numeric_range_annotations_opened": 0,
            "pointer_annotations_opened": 0,
            "restricted_namespace_images_opened": 0,
        },
    }
    assert_label_free(summary, location="joint_oof_summary")
    summary_path = output / "summary.json"
    atomic_new_json(summary_path, summary)
    return summary_path


def load_joint_oof_cohort(
    summary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Authenticate the deterministic 412/19 cohort and checkpoint mapping."""

    path = guard_public_path(summary_path, label="joint OOF cohort summary")
    summary = strict_json(path)
    require(summary.get("protocol") == JOINT_OOF_SUMMARY_PROTOCOL, "OOF identity drift")
    require(
        summary.get("status") == "frozen_label_free_joint_oof_cohort",
        "joint OOF cohort is not frozen",
    )
    require(
        (
            summary.get("range_independent_validation", {}).get("samples"),
            summary.get("range_independent_validation", {}).get("groups"),
        )
        == EXPECTED_RANGE_INDEPENDENT,
        "bound range inventory drift",
    )
    require(
        (
            summary.get("joint_oof", {}).get("samples"),
            summary.get("joint_oof", {}).get("groups"),
        )
        == EXPECTED_JOINT_OOF,
        "bound joint OOF inventory drift",
    )
    require(
        summary.get("joint_oof", {}).get("samples_by_pepd_seed")
        == {str(key): value for key, value in EXPECTED_JOINT_OOF_BY_SEED.items()},
        "bound joint OOF seed counts drift",
    )
    cohort_path = _verify_binding(
        summary["artifacts"]["cohort"], label="joint OOF cohort"
    )
    mapping_path = _verify_binding(
        summary["artifacts"]["mapping"], label="joint OOF mapping"
    )
    cohort = strict_jsonl(cohort_path)
    mapping = strict_jsonl(mapping_path)
    require(len(cohort) == len(mapping) == EXPECTED_JOINT_OOF[0], "OOF row drift")
    cohort_by_id = {str(row["sample_id"]): row for row in cohort}
    mapping_by_id: dict[str, dict[str, Any]] = {}
    seed_counts: Counter[int] = Counter()
    expected_keys = {
        "schema_version",
        "protocol",
        "partition",
        "sample_id",
        "group_id",
        "pepd_seed",
        "pepd_checkpoint_sha256",
        "canonical_roi_sha256",
        "group_unseen_by_progress_checkpoint",
    }
    for row in mapping:
        require(set(row) == expected_keys, "joint OOF mapping schema drift")
        require(row["protocol"] == JOINT_OOF_ROW_PROTOCOL, "OOF row protocol drift")
        require(row["partition"] == "independent_validation", "OOF partition drift")
        sample_id = str(row["sample_id"])
        require(sample_id in cohort_by_id, f"OOF mapping outside cohort: {sample_id}")
        require(sample_id not in mapping_by_id, f"duplicate OOF mapping: {sample_id}")
        require(
            str(row["group_id"]) == str(cohort_by_id[sample_id]["group_id"]),
            f"OOF group drift: {sample_id}",
        )
        require(row["group_unseen_by_progress_checkpoint"] is True, "OOF proof absent")
        _sha256(row["pepd_checkpoint_sha256"], "OOF PEPD checkpoint")
        _sha256(row["canonical_roi_sha256"], "OOF canonical ROI")
        seed_counts[int(row["pepd_seed"])] += 1
        mapping_by_id[sample_id] = row
    require(dict(seed_counts) == EXPECTED_JOINT_OOF_BY_SEED, "OOF seed count drift")
    require(set(mapping_by_id) == set(cohort_by_id), "OOF mapping coverage drift")
    require(
        len({str(row["group_id"]) for row in mapping}) == EXPECTED_JOINT_OOF[1],
        "OOF group count drift",
    )
    assert_label_free(summary, location="joint_oof_summary")
    assert_label_free(mapping, location="joint_oof_mapping")
    return summary, cohort, mapping


def _load_consensus_config(path: Path | None) -> GARCConsensusConfig:
    if path is None:
        return GARCConsensusConfig().validate()
    source = guard_public_path(path, label="GARC consensus configuration")
    value = strict_json(source)
    allowed = set(asdict(GARCConsensusConfig()))
    require(set(value) == allowed, "GARC consensus configuration schema drift")
    return GARCConsensusConfig(**value).validate()


def freeze_plan(
    *,
    protocol_path: Path,
    output_path: Path,
    progress_binding_path: Path,
    progress_factory_path: Path,
    progress_factory_function: str,
    detector_checkpoint: Path,
    recognizer_checkpoint: Path,
    geometry_checkpoint: Path,
    geometry_provider: str = "legacy_fixed_v5",
    geometry_backbone_checkpoint: Path | None = None,
    geometry_oof_summary_path: Path | None = None,
    geometry_oof_seed: int | None = None,
    recognizer_kind: str,
    consensus_mode: str,
    geometry_mode: str,
    method_name: str,
    reference_mode: str,
    reference_detector_sha256: str | None,
    execution_mode: str,
    device: str,
    input_size: int,
    detector_threshold: float,
    posterior_top_k: int,
    consensus_config_path: Path | None = None,
    joint_oof_summary_path: Path | None = None,
) -> Path:
    """Freeze one immutable method variant without loading any public image."""

    require(recognizer_kind in RECOGNIZER_KINDS, "unsupported recognizer kind")
    require(consensus_mode in CONSENSUS_MODES, "unsupported consensus mode")
    require(geometry_mode in GEOMETRY_MODES, "unsupported geometry mode")
    require(geometry_provider in GEOMETRY_PROVIDERS, "unsupported geometry provider")
    require(execution_mode in (EXECUTION_FORMAL, EXECUTION_SYNTHETIC), "bad mode")
    require(int(input_size) in (512, 768), "input_size must be 512 or 768")
    require(
        math.isfinite(float(detector_threshold))
        and 0.0 < float(detector_threshold) < 1.0,
        "detector threshold must be in (0,1)",
    )
    config = _load_consensus_config(consensus_config_path)
    require(
        1 <= int(posterior_top_k) <= config.maximum_candidates_per_token,
        "posterior_top_k exceeds the frozen consensus candidate limit",
    )
    require(bool(str(progress_factory_function).strip()), "factory function is empty")

    protocol_file, protocol = load_frozen_protocol(protocol_path)
    for source_key in ("syncg_train_manifest", "syncg_train_manifest_protocol"):
        verify_bound_file(protocol, "source_bindings", source_key)
    partitions: dict[str, Any] = {}
    for partition in SUPPORTED_PARTITIONS:
        _, manifest_path, _, audit = load_partition_roster(protocol_file, partition)
        partitions[partition] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "samples": audit["samples"],
            "groups": audit["groups"],
            "sample_ids_sha256": audit["sample_ids_sha256"],
            "group_ids_sha256": audit["group_ids_sha256"],
        }

    joint_summary: dict[str, Any] | None = None
    joint_mapping: list[dict[str, Any]] = []
    if joint_oof_summary_path is not None:
        joint_summary, _, joint_mapping = load_joint_oof_cohort(
            joint_oof_summary_path
        )
        require(
            int(joint_summary["input_size"]) == int(input_size),
            "joint OOF ROI size differs from GARC input size",
        )

    progress_binding_file = guard_public_path(
        progress_binding_path, label="progress component binding"
    )
    progress_binding = FrozenComponentBinding.from_record(
        strict_json(progress_binding_file)
    )
    require(progress_binding.name == "progress", "progress binding name drift")
    if execution_mode == EXECUTION_FORMAL:
        require(progress_binding.frozen, "formal progress binding is not frozen")
        require(
            progress_binding.verified_complete,
            "formal progress binding is not verified complete",
        )
        require(not progress_binding.synthetic, "formal progress binding is synthetic")

    factory = _bound_file(progress_factory_path, label="progress factory")
    detector = _bound_file(detector_checkpoint, label="SyncG numeric detector")
    recognizer = _bound_file(recognizer_checkpoint, label="SyncG numeric recognizer")
    geometry = _bound_file(geometry_checkpoint, label="ScaleMark geometry checkpoint")
    geometry_fold: dict[str, Any]
    geometry_artifacts: dict[str, dict[str, Any]] = {"geometry": geometry}
    if geometry_provider == "enhanced_v5_oof_fold":
        require(
            geometry_backbone_checkpoint is not None
            and geometry_oof_summary_path is not None
            and geometry_oof_seed is not None,
            "enhanced-V5 OOF geometry requires backbone, summary, and seed",
        )
        require(
            joint_summary is not None and bool(joint_mapping),
            "enhanced-V5 OOF geometry requires the frozen joint cohort",
        )
        from experiments.garc_enhanced_v5_oof_geometry_provider import (
            authenticate_oof_fold,
        )

        authenticated = authenticate_oof_fold(
            oof_summary_path=geometry_oof_summary_path,
            pepd_seed=int(geometry_oof_seed),
            head_checkpoint_path=geometry_checkpoint,
            backbone_checkpoint_path=geometry_backbone_checkpoint,
        )
        require(
            authenticated["head_checkpoint"]["sha256"] == geometry["sha256"],
            "authenticated enhanced-V5 head binding drift",
        )
        geometry_backbone = _bound_file(
            geometry_backbone_checkpoint, label="fold-routed geometry PEPD backbone"
        )
        require(
            authenticated["backbone_checkpoint"]["sha256"]
            == geometry_backbone["sha256"],
            "authenticated geometry backbone binding drift",
        )
        seed = int(geometry_oof_seed)
        routed = [
            row for row in joint_mapping if int(row["pepd_seed"]) == seed
        ]
        unseen_groups = {str(row["group_id"]) for row in routed}
        require(
            len(routed) == EXPECTED_JOINT_OOF_BY_SEED[seed]
            and len(unseen_groups) == EXPECTED_JOINT_OOF_GROUPS_BY_SEED[seed],
            "geometry fold/joint cohort inventory drift",
        )
        route_hashes = {str(row["pepd_checkpoint_sha256"]) for row in routed}
        require(
            route_hashes == {geometry_backbone["sha256"]},
            "geometry backbone does not match the joint OOF route",
        )
        require(
            authenticated["sample_assignment_sha256"]
            == joint_summary["provenance"]["sample_assignment_sha256"],
            "enhanced-V5 and joint-cohort sample assignments differ",
        )
        independent_samples = int(partitions["independent_validation"]["samples"])
        independent_groups = int(partitions["independent_validation"]["groups"])
        fit_overlap_samples = independent_samples - len(routed)
        fit_overlap_groups = independent_groups - len(unseen_groups)
        if seed == 20260720:
            require(
                {
                    "geometry_unseen_samples": len(routed),
                    "geometry_unseen_groups": len(unseen_groups),
                    "geometry_fit_overlap_samples": fit_overlap_samples,
                    "geometry_fit_overlap_groups": fit_overlap_groups,
                }
                == EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY,
                "primary fixed-fold 168/8 versus 912/42 audit drift",
            )
        geometry_artifacts["geometry_backbone"] = geometry_backbone
        geometry_fold = {
            "provider": geometry_provider,
            "role": "joint_oof_fold_routed_pepd_and_enhanced_v5_head",
            "pepd_seed": seed,
            "oof_summary": authenticated["oof_summary"],
            "fold_summary": authenticated["fold_summary"],
            "head_training_seed": authenticated["head_training_seed"],
            "joint_assignment": {
                "sample_assignment_sha256": authenticated[
                    "sample_assignment_sha256"
                ],
                "group_assignment_sha256": authenticated[
                    "group_assignment_sha256"
                ],
            },
            "independent_validation_sensitivity": {
                "samples": independent_samples,
                "groups": independent_groups,
                "geometry_unseen_samples": len(routed),
                "geometry_unseen_groups": len(unseen_groups),
                "geometry_fit_overlap_samples": fit_overlap_samples,
                "geometry_fit_overlap_groups": fit_overlap_groups,
                "full_1080_all_component_unseen": False,
            },
            "joint_shard_all_component_unseen": True,
        }
    else:
        require(
            geometry_backbone_checkpoint is None
            and geometry_oof_summary_path is None
            and geometry_oof_seed is None,
            "legacy fixed V5 geometry cannot claim an OOF fold binding",
        )
        require(
            not (
                execution_mode == EXECUTION_FORMAL
                and joint_oof_summary_path is not None
            ),
            "legacy fixed V5 geometry is ineligible for formal joint OOF",
        )
        geometry_fold = {
            "provider": geometry_provider,
            "role": "fixed_backbone_component_or_sensitivity_only",
            "pepd_seed": None,
            "oof_summary": None,
            "fold_summary": None,
            "head_training_seed": None,
            "joint_assignment": None,
            "independent_validation_sensitivity": None,
            "joint_shard_all_component_unseen": False,
        }

    if reference_mode == REFERENCE_MODE_AUTO:
        detector_sha = _sha256(
            reference_detector_sha256, "automatic reference detector"
        )
        require(method_name.endswith("+auto-ref"), "auto method must end +auto-ref")
    else:
        require(reference_mode == REFERENCE_MODE_NATIVE, "bad reference mode")
        require(
            reference_detector_sha256 is None,
            "native progress cannot bind an automatic reference detector",
        )
        require(
            not method_name.endswith("+auto-ref"),
            "native method must not claim +auto-ref",
        )
        detector_sha = None

    descriptor = {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "status": "frozen_before_public_inference",
        "execution_mode": execution_mode,
        "claim_scope": "public SyncG/train entity-group-independent evaluation only",
        "parent_protocol": {
            "path": str(protocol_file),
            "sha256": sha256_file(protocol_file),
            "identity": protocol["protocol"],
        },
        "partitions": partitions,
        "method": {"name": str(method_name)},
        "progress_component": {
            "binding": progress_binding.as_record(),
            "binding_file": {
                "path": str(progress_binding_file),
                "sha256": sha256_file(progress_binding_file),
            },
        },
        "progress_factory": {
            **factory,
            "function": str(progress_factory_function),
        },
        "joint_oof": (
            {
                "status": "not_bound_component_or_range_only",
                "summary": None,
                "expected_joint_samples": EXPECTED_JOINT_OOF[0],
                "expected_joint_groups": EXPECTED_JOINT_OOF[1],
            }
            if joint_oof_summary_path is None
            else {
                "status": "bound_for_overlap_audit",
                "summary": _bound_file(
                    joint_oof_summary_path, label="joint OOF cohort summary"
                ),
                "expected_joint_samples": EXPECTED_JOINT_OOF[0],
                "expected_joint_groups": EXPECTED_JOINT_OOF[1],
            }
        ),
        "reference": {
            "mode": reference_mode,
            "detector_sha256": detector_sha,
        },
        "garc": {
            "recognizer_kind": recognizer_kind,
            "consensus_mode": consensus_mode,
            "geometry_mode": geometry_mode,
            "geometry_provider": geometry_provider,
            "geometry_fold": geometry_fold,
            "effective_decoder_protocol": (
                TOP1_PROTOCOL if consensus_mode == "top1" else TOPK_PROTOCOL
            ),
            "device": str(device),
            "input_size": int(input_size),
            "detector_threshold": float(detector_threshold),
            "posterior_top_k": int(posterior_top_k),
            "consensus_config": asdict(config),
            "artifacts": {
                "detector": detector,
                "recognizer": recognizer,
                **geometry_artifacts,
                **(
                    {
                        "base_pointer_segmentation": _bound_file(
                            BASE_POINTER_SEGMENTATION,
                            label="Base pointer segmentation",
                        ),
                        "base_reference_detector": _bound_file(
                            BASE_REFERENCE_DETECTOR,
                            label="Base automatic reference detector",
                        ),
                    }
                    if geometry_mode == "v5_pepd_base_fusion"
                    else {}
                ),
            },
        },
        "code_bindings": _source_bindings(),
        "audit": {
            "public_partitions_bound": list(SUPPORTED_PARTITIONS),
            "development_excluded_permitted": False,
            "caller_numeric_range_permitted": False,
            "caller_geometry_permitted": False,
            "caller_reference_packet_permitted": False,
            "public_images_opened_while_freezing": 0,
            "restricted_namespace_images_opened": 0,
        },
    }
    require(set(descriptor) == _PLAN_KEYS, "plan schema construction drift")
    assert_label_free(descriptor, location="garc_full_auto_plan")
    output = guard_public_path(
        output_path, label="GARC full-auto plan output", must_exist=False
    )
    atomic_new_json(output, descriptor)
    return output


def _verify_binding(binding: Mapping[str, Any], *, label: str) -> Path:
    require(
        {"path", "sha256"}.issubset(binding), f"{label} binding schema drift"
    )
    path = guard_public_path(Path(str(binding.get("path") or "")), label=label)
    require(sha256_file(path) == binding.get("sha256"), f"{label} hash drift")
    return path


def load_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    plan_path = guard_public_path(path, label="GARC full-auto plan")
    plan = strict_json(plan_path)
    require(set(plan) == _PLAN_KEYS, "GARC plan schema drift")
    require(plan.get("schema_version") == 1, "GARC plan schema version drift")
    require(plan.get("protocol") == PLAN_PROTOCOL, "GARC plan identity drift")
    require(
        plan.get("status") == "frozen_before_public_inference",
        "GARC plan is not frozen",
    )
    assert_label_free(plan, location="garc_full_auto_plan")
    require(
        set(plan.get("partitions", {})) == set(SUPPORTED_PARTITIONS),
        "GARC plan partition drift",
    )
    parent = plan.get("parent_protocol") or {}
    parent_path, _ = load_frozen_protocol(Path(str(parent.get("path") or "")))
    require(sha256_file(parent_path) == parent.get("sha256"), "parent protocol drift")
    for partition in SUPPORTED_PARTITIONS:
        _, manifest, _, audit = load_partition_roster(parent_path, partition)
        binding = plan["partitions"][partition]
        require(str(manifest) == str(binding.get("path")), f"{partition} path drift")
        require(sha256_file(manifest) == binding.get("sha256"), f"{partition} hash drift")
        for key in ("samples", "groups", "sample_ids_sha256", "group_ids_sha256"):
            require(audit[key] == binding.get(key), f"{partition}.{key} drift")
    for name, binding in plan.get("code_bindings", {}).items():
        _verify_binding(binding, label=f"code binding {name}")
    _verify_binding(plan["progress_factory"], label="progress factory")
    progress_file = _verify_binding(
        plan["progress_component"]["binding_file"],
        label="progress component binding",
    )
    frozen_progress = FrozenComponentBinding.from_record(
        plan["progress_component"]["binding"]
    )
    require(
        strict_json(progress_file) == frozen_progress.as_record(),
        "progress binding file content drift",
    )
    garc = plan.get("garc") or {}
    require(garc.get("recognizer_kind") in RECOGNIZER_KINDS, "recognizer drift")
    require(garc.get("consensus_mode") in CONSENSUS_MODES, "consensus drift")
    require(garc.get("geometry_mode") in GEOMETRY_MODES, "geometry mode drift")
    require(
        garc.get("geometry_provider") in GEOMETRY_PROVIDERS,
        "geometry provider drift",
    )
    config = GARCConsensusConfig(**garc.get("consensus_config", {})).validate()
    require(
        1 <= int(garc.get("posterior_top_k", 0))
        <= config.maximum_candidates_per_token,
        "posterior top-K drift",
    )
    for name, binding in garc.get("artifacts", {}).items():
        _verify_binding(binding, label=f"GARC artifact {name}")
    geometry_fold = garc.get("geometry_fold") or {}
    require(
        geometry_fold.get("provider") == garc.get("geometry_provider"),
        "geometry fold/provider drift",
    )
    if garc["geometry_provider"] == "enhanced_v5_oof_fold":
        from experiments.garc_enhanced_v5_oof_geometry_provider import (
            authenticate_oof_fold,
        )

        seed = int(geometry_fold.get("pepd_seed", -1))
        authenticated = authenticate_oof_fold(
            oof_summary_path=Path(geometry_fold["oof_summary"]["path"]),
            pepd_seed=seed,
            head_checkpoint_path=Path(garc["artifacts"]["geometry"]["path"]),
            backbone_checkpoint_path=Path(
                garc["artifacts"]["geometry_backbone"]["path"]
            ),
            expected_oof_summary_sha256=geometry_fold["oof_summary"]["sha256"],
            expected_fold_summary_sha256=geometry_fold["fold_summary"]["sha256"],
            expected_head_checkpoint_sha256=garc["artifacts"]["geometry"]["sha256"],
            expected_backbone_checkpoint_sha256=garc["artifacts"][
                "geometry_backbone"
            ]["sha256"],
        )
        require(
            authenticated["fold_summary"]["sha256"]
            == geometry_fold["fold_summary"]["sha256"],
            "geometry fold summary drift",
        )
        require(
            geometry_fold.get("joint_shard_all_component_unseen") is True,
            "enhanced geometry lacks joint unseen attestation",
        )
    else:
        require(
            geometry_fold.get("joint_shard_all_component_unseen") is False,
            "legacy geometry falsely claims joint unseen evidence",
        )
    reference = plan.get("reference") or {}
    require(
        reference.get("mode") in (REFERENCE_MODE_AUTO, REFERENCE_MODE_NATIVE),
        "reference mode drift",
    )
    if reference["mode"] == REFERENCE_MODE_AUTO:
        _sha256(reference.get("detector_sha256"), "automatic reference detector")
    else:
        require(reference.get("detector_sha256") is None, "native detector binding drift")
    joint = plan.get("joint_oof") or {}
    require(
        joint.get("status")
        in ("not_bound_component_or_range_only", "bound_for_overlap_audit"),
        "joint OOF binding status drift",
    )
    require(
        (
            joint.get("expected_joint_samples"),
            joint.get("expected_joint_groups"),
        )
        == EXPECTED_JOINT_OOF,
        "joint OOF expected inventory drift",
    )
    if joint["status"] == "bound_for_overlap_audit":
        summary_binding = joint.get("summary")
        require(isinstance(summary_binding, Mapping), "joint OOF summary binding absent")
        joint_path = _verify_binding(summary_binding, label="joint OOF cohort summary")
        joint_summary, _, _ = load_joint_oof_cohort(joint_path)
        require(
            int(joint_summary["input_size"]) == int(garc["input_size"]),
            "joint OOF ROI size drift",
        )
    else:
        require(joint.get("summary") is None, "unbound joint OOF has an artifact")
    return plan_path, plan


class _ConsensusModeDecoder:
    """Freeze literal recognizer-top1 versus posterior-topK decoding."""

    def __init__(self, decoder: GARCPosteriorConsensusDecoder, mode: str):
        require(mode in CONSENSUS_MODES, "unsupported decoder mode")
        self.decoder = decoder
        self.mode = mode

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "protocol": TOP1_PROTOCOL if self.mode == "top1" else TOPK_PROTOCOL,
            "ablation_mode": self.mode,
            "prediction_space": "real_numeric_scale_start_end",
            "input": "automatic arc progress and recognizer posterior",
            "supervised_inputs_allowed": False,
            "config": asdict(self.decoder.config),
        }

    def predict(self, tokens: Sequence[Any]):
        if self.mode == "top1":
            return self.decoder.top1_control(tokens)
        return self.decoder.predict(tokens)


class _RecordingProgressProvider:
    """Record the image-derived PEPD packet for optional geometry fusion."""

    def __init__(self, provider: Any) -> None:
        self.provider = provider
        self.last_image_sha256: str | None = None
        self.last_output: Mapping[str, Any] | None = None

    @property
    def identity(self) -> Mapping[str, Any]:
        # Preserve the exact frozen identity used by the component binding.
        return self.provider.identity

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        value = self.provider.predict(
            canonical_meter_roi_bgr,
            input_is_canonical_meter_roi=input_is_canonical_meter_roi,
        )
        require(isinstance(value, Mapping), "progress provider returned non-mapping")
        self.last_image_sha256 = image_sha256(canonical_meter_roi_bgr)
        self.last_output = value
        return value


class _RecordedPEPDStructuralProvider:
    """Reuse the already-computed PEPD packet without a second forward pass."""

    def __init__(self, progress: _RecordingProgressProvider) -> None:
        self.progress = progress
        self._identity = {
            "protocol": "garc_recorded_pepd_structural_reuse_v1",
            "provider": dict(progress.identity),
            "same_image_cache_required": True,
            "duplicate_progress_forward": False,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def predict(
        self,
        canonical_meter_roi_bgr: np.ndarray,
        *,
        input_is_canonical_meter_roi: bool,
    ) -> Mapping[str, Any]:
        require(input_is_canonical_meter_roi is True, "canonical ROI flag required")
        digest = image_sha256(canonical_meter_roi_bgr)
        require(
            self.progress.last_image_sha256 == digest
            and self.progress.last_output is not None,
            "PEPD structural cache is absent or belongs to another image",
        )
        return self.progress.last_output


class GARCFullAutoRangePipeline(AutomaticNumericRangePipeline):
    """Compatibility wrapper with the stricter full-auto range attestation."""

    def __init__(
        self,
        provider: GARCAutomaticNumericRangeProvider,
        *,
        recognizer_kind: str,
        consensus_mode: str,
    ) -> None:
        # AutomaticNumericRangePipeline is intentionally not initialized: GARC
        # owns its posterior-aware estimator, while subclassing preserves the
        # existing full-auto runtime type boundary.
        self.provider = provider
        self.recognizer_kind = recognizer_kind
        self.consensus_mode = consensus_mode
        self.last_prediction: NumericRangePrediction | None = None
        self._identity = {
            "protocol": RANGE_ADAPTER_PROTOCOL,
            "primary_input": "one whole canonical meter ROI BGR uint8",
            "caller_geometry_allowed": False,
            "caller_numeric_range_allowed": False,
            "caller_reference_packet_allowed": False,
            "recognizer_kind": recognizer_kind,
            "consensus_mode": consensus_mode,
            "geometry_provider": dict(provider.geometry_provider.identity),
            "posterior_ocr_backend": dict(provider.posterior_ocr_backend.identity),
            "tick_proximity_provider": (
                None
                if provider.tick_proximity_provider is None
                else dict(provider.tick_proximity_provider.identity)
            ),
            "ocr_input_size": provider.input_size,
            "consensus": dict(provider.decoder.identity),
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    def reset_trace(self) -> None:
        self.last_prediction = None

    def predict(self, canonical_meter_roi_bgr: np.ndarray) -> NumericRangePrediction:
        prediction = self.provider.predict(canonical_meter_roi_bgr)
        telemetry = dict(prediction.telemetry)
        primary = dict(telemetry.get("primary_adapter") or {})
        digest = image_sha256(canonical_meter_roi_bgr)
        require(
            primary.get("input_image_sha256") == digest,
            "GARC primary image digest drift",
        )
        require(
            primary.get("geometry_and_ocr_same_source_image_sha256") == digest,
            "GARC geometry/OCR source digest drift",
        )
        primary.update(
            {
                "accepts_manual_geometry": False,
                "accepts_reference_packet": False,
                "accepts_physical_scale_values": False,
                "geometry_and_ocr_same_image_sha256": digest,
            }
        )
        telemetry["primary_adapter"] = primary
        normalized = NumericRangePrediction(
            protocol=prediction.protocol,
            status=prediction.status,
            prediction_space=prediction.prediction_space,
            pred_start=prediction.pred_start,
            pred_end=prediction.pred_end,
            confidence=prediction.confidence,
            failure_reason=prediction.failure_reason,
            telemetry=telemetry,
        )
        self.last_prediction = normalized
        return normalized


def _load_progress_factory(plan: Mapping[str, Any]):
    binding = plan["progress_factory"]
    source = _verify_binding(binding, label="progress factory")
    function_name = str(binding.get("function") or "")
    require(bool(function_name), "progress factory function is empty")
    module_name = f"garc_progress_factory_{binding['sha256'][:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    require(spec is not None and spec.loader is not None, "cannot load progress factory")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, function_name, None)
    require(callable(factory), f"progress factory lacks callable {function_name}")
    return factory


def build_runtime(
    plan: Mapping[str, Any],
) -> tuple[UnifiedFullAutoAdapter, GARCFullAutoRangePipeline]:
    """Load bound providers and compose them before any public image opens."""

    progress_binding = FrozenComponentBinding.from_record(
        plan["progress_component"]["binding"]
    )
    raw_progress_provider = _load_progress_factory(plan)(plan)
    progress_binding.verify_provider(raw_progress_provider)
    progress_provider = _RecordingProgressProvider(raw_progress_provider)

    garc = plan["garc"]
    artifacts = garc["artifacts"]
    detector_path = _verify_binding(artifacts["detector"], label="GARC detector")
    recognizer_path = _verify_binding(artifacts["recognizer"], label="GARC recognizer")
    geometry_path = _verify_binding(artifacts["geometry"], label="GARC geometry")
    common = {
        "device": str(garc["device"]),
        "threshold": float(garc["detector_threshold"]),
        "posterior_top_k": int(garc["posterior_top_k"]),
    }
    if garc["recognizer_kind"] == "tiny":
        backend = GaugeNumericOCRBackend(
            detector_path, recognizer_path, **common
        )
    else:
        from experiments.syncg_strong_numeric_ocr import (
            StrongGaugeNumericOCRBackend,
        )

        backend = StrongGaugeNumericOCRBackend(
            detector_path, recognizer_path, **common
        )
    if garc["geometry_provider"] == "enhanced_v5_oof_fold":
        from experiments.garc_enhanced_v5_oof_geometry_provider import (
            FrozenEnhancedV5OOFGeometryProvider,
        )

        geometry_fold = garc["geometry_fold"]
        base_geometry = FrozenEnhancedV5OOFGeometryProvider(
            oof_summary_path=Path(geometry_fold["oof_summary"]["path"]),
            pepd_seed=int(geometry_fold["pepd_seed"]),
            head_checkpoint_path=geometry_path,
            backbone_checkpoint_path=_verify_binding(
                artifacts["geometry_backbone"],
                label="GARC fold-routed geometry backbone",
            ),
            expected_oof_summary_sha256=geometry_fold["oof_summary"]["sha256"],
            expected_fold_summary_sha256=geometry_fold["fold_summary"]["sha256"],
            expected_head_checkpoint_sha256=artifacts["geometry"]["sha256"],
            expected_backbone_checkpoint_sha256=artifacts["geometry_backbone"][
                "sha256"
            ],
        )
    else:
        base_geometry = FrozenPublicScaleMarkGeometryProvider(geometry_path)
    if garc["geometry_mode"] in (
        "v5_pepd_fusion",
        "v5_pepd_base_fusion",
    ):
        identity_text = json.dumps(
            progress_provider.identity,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).casefold()
        require("pepd" in identity_text, "PEPD geometry fusion requires a PEPD provider")
        from experiments.garc_geometry_fusion import (
            GARCCovarianceGeometryFusionProvider,
        )

        pepd_geometry = _RecordedPEPDStructuralProvider(progress_provider)
        if garc["geometry_mode"] == "v5_pepd_base_fusion":
            from experiments.garc_base_mask_geometry_provider import (
                ExistingBaseMaskGeometryProvider,
            )

            base_mask_geometry = ExistingBaseMaskGeometryProvider.from_base_only_checkpoints(
                pointer_segmentation=_verify_binding(
                    artifacts["base_pointer_segmentation"],
                    label="Base pointer segmentation",
                ),
                reference_detector=_verify_binding(
                    artifacts["base_reference_detector"],
                    label="Base automatic reference detector",
                ),
                device="cpu",
                task_config=None,
            )
            geometry = GARCCovarianceGeometryFusionProvider(
                base_geometry,
                pepd_geometry,
                mask_geometry_provider=base_mask_geometry,
            )
        else:
            geometry = GARCCovarianceGeometryFusionProvider(
                base_geometry,
                pepd_geometry,
            )
    else:
        geometry = base_geometry
    provider = GARCAutomaticNumericRangeProvider(
        geometry,
        backend,
        input_size=int(garc["input_size"]),
        consensus_config=GARCConsensusConfig(**garc["consensus_config"]),
    )
    provider.decoder = _ConsensusModeDecoder(provider.decoder, garc["consensus_mode"])
    range_pipeline = GARCFullAutoRangePipeline(
        provider,
        recognizer_kind=garc["recognizer_kind"],
        consensus_mode=garc["consensus_mode"],
    )
    source_hashes = {
        key: value["sha256"] for key, value in plan["code_bindings"].items()
    }
    range_binding = FrozenComponentBinding.from_provider(
        name="automatic_numeric_range",
        provider=range_pipeline,
        artifact_sha256={
            name: binding["sha256"] for name, binding in artifacts.items()
        },
        source_sha256=source_hashes,
        provider_protocol=RANGE_ADAPTER_PROTOCOL,
        synthetic=plan["execution_mode"] == EXECUTION_SYNTHETIC,
        verified_complete=True,
    )
    reference = plan["reference"]
    bundle = FrozenFullAutoBundle.create(
        method_name=plan["method"]["name"],
        progress_binding=progress_binding,
        range_binding=range_binding,
        factory_source_sha256=plan["progress_factory"]["sha256"],
        reference_mode=reference["mode"],
        reference_detector_sha256=reference["detector_sha256"],
        execution_mode=plan["execution_mode"],
    )
    adapter = UnifiedFullAutoAdapter(
        progress_provider=progress_provider,
        automatic_numeric_range_pipeline=range_pipeline,
        bundle=bundle,
    )
    return adapter, range_pipeline


def _smoke_subset(
    rows: Sequence[Mapping[str, Any]], *, groups: int, samples_per_group: int
) -> list[dict[str, Any]]:
    require(groups >= 1 and samples_per_group >= 1, "invalid smoke limits")
    by_group: dict[str, list[dict[str, Any]]] = {}
    for value in rows:
        row = dict(value)
        by_group.setdefault(str(row["group_id"]), []).append(row)
    selected_groups = sorted(
        by_group,
        key=lambda value: hashlib.sha256(f"garc-smoke:{value}".encode()).hexdigest(),
    )[:groups]
    selected: list[dict[str, Any]] = []
    for group in selected_groups:
        selected.extend(
            sorted(by_group[group], key=lambda row: str(row["sample_id"]))[
                :samples_per_group
            ]
        )
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def _failed_prediction(bundle: FrozenFullAutoBundle, reason: str) -> FullAutoPrediction:
    reference = None
    if bundle.reference_mode == REFERENCE_MODE_AUTO:
        reference = {
            "status": False,
            "start_angle": None,
            "range_angle": None,
            "reference_branch": "garc_public_runner:not_invoked",
            "failure_code": "canonical_roi_unavailable",
        }
    return FullAutoPrediction(
        status=False,
        prediction_progress=None,
        predicted_scale_start=None,
        predicted_scale_end=None,
        range_confidence=None,
        failure_code=reason,
        automatic_reference=reference,
        telemetry={
            "protocol": PREDICTION_PROTOCOL,
            "input_failure": reason,
            "providers_invoked": False,
        },
    )


def _prediction_row(
    *,
    partition: str,
    roster_row: Mapping[str, Any],
    roi: np.ndarray,
    full: FullAutoPrediction,
    range_prediction: NumericRangePrediction | None,
    bundle: FrozenFullAutoBundle,
    sample_seconds: float,
    progress_checkpoint_sha256: str | None,
    geometry_head_checkpoint_sha256: str,
    geometry_backbone_checkpoint_sha256: str | None,
    geometry_oof_seed: int | None,
    progress_group_unseen: bool,
    geometry_group_unseen: bool,
    joint_oof_eligible: bool,
) -> dict[str, Any]:
    digest = image_sha256(roi)
    range_status = bool(range_prediction is not None and range_prediction.status)
    start = _finite(range_prediction.pred_start) if range_prediction is not None else None
    end = _finite(range_prediction.pred_end) if range_prediction is not None else None
    confidence = (
        _finite(range_prediction.confidence) if range_prediction is not None else 0.0
    )
    if not range_status or start is None or end is None or end == start:
        range_status = False
        start = None
        end = None
    if confidence is None or not 0.0 <= confidence <= 1.0:
        range_status = False
        confidence = 0.0
    predicted_reading = None
    if full.status:
        require(
            full.prediction_progress is not None
            and full.predicted_scale_start is not None
            and full.predicted_scale_end is not None,
            "successful full prediction lacks numeric factors",
        )
        predicted_reading = float(full.predicted_scale_start) + float(
            full.prediction_progress
        ) * (
            float(full.predicted_scale_end) - float(full.predicted_scale_start)
        )
    method_record = full.as_method_record(
        bundle=bundle, canonical_roi_file_sha256=digest
    )
    row = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "partition": partition,
        "sample_id": str(roster_row["sample_id"]),
        "group_id": str(roster_row["group_id"]),
        "canonical_roi_sha256": digest,
        "full_status": bool(full.status),
        "range_status": range_status,
        "prediction_progress": full.prediction_progress if full.status else None,
        "predicted_scale_start": start,
        "predicted_scale_end": end,
        "predicted_reading": predicted_reading,
        "range_confidence": float(confidence),
        "failure_reason": full.failure_code,
        "range_failure_reason": (
            None
            if range_status
            else (
                range_prediction.failure_reason
                if range_prediction is not None
                else "range_provider_did_not_return"
            )
        ),
        "progress_checkpoint_sha256": progress_checkpoint_sha256,
        "geometry_head_checkpoint_sha256": geometry_head_checkpoint_sha256,
        "geometry_backbone_checkpoint_sha256": geometry_backbone_checkpoint_sha256,
        "geometry_oof_seed": geometry_oof_seed,
        "progress_group_unseen": bool(progress_group_unseen),
        "geometry_group_unseen": bool(geometry_group_unseen),
        "joint_oof_eligible": bool(joint_oof_eligible),
        "method_record": method_record,
        "range_prediction": (
            None if range_prediction is None else range_prediction.as_dict()
        ),
        "sample_seconds": float(sample_seconds),
    }
    validate_prediction_row(row, partition=partition, roster_row=roster_row)
    return row


def validate_prediction_row(
    row: Mapping[str, Any], *, partition: str, roster_row: Mapping[str, Any]
) -> None:
    require(set(row) == _PREDICTION_KEYS, "GARC prediction row schema drift")
    assert_label_free(row, location=f"garc_prediction.{row.get('sample_id')}")
    require(row.get("protocol") == PREDICTION_PROTOCOL, "prediction protocol drift")
    require(row.get("partition") == partition, "prediction partition drift")
    require(row.get("sample_id") == roster_row.get("sample_id"), "sample drift")
    require(row.get("group_id") == roster_row.get("group_id"), "group drift")
    _sha256(row.get("canonical_roi_sha256"), "canonical ROI")
    require(isinstance(row.get("full_status"), bool), "full status is not boolean")
    require(isinstance(row.get("range_status"), bool), "range status is not boolean")
    require(
        isinstance(row.get("joint_oof_eligible"), bool),
        "joint OOF eligibility is not boolean",
    )
    if row.get("progress_checkpoint_sha256") is not None:
        _sha256(row["progress_checkpoint_sha256"], "progress checkpoint")
    _sha256(row.get("geometry_head_checkpoint_sha256"), "geometry head checkpoint")
    if row.get("geometry_backbone_checkpoint_sha256") is not None:
        _sha256(row["geometry_backbone_checkpoint_sha256"], "geometry backbone checkpoint")
    require(
        isinstance(row.get("progress_group_unseen"), bool)
        and isinstance(row.get("geometry_group_unseen"), bool),
        "component unseen flags are not boolean",
    )
    if row.get("geometry_oof_seed") is not None:
        require(
            int(row["geometry_oof_seed"]) in EXPECTED_JOINT_OOF_BY_SEED,
            "geometry OOF seed drift",
        )
    if row["joint_oof_eligible"]:
        require(
            row.get("progress_checkpoint_sha256") is not None,
            "joint OOF row lacks progress checkpoint",
        )
        require(
            row["progress_group_unseen"] is True
            and row["geometry_group_unseen"] is True
            and row.get("geometry_backbone_checkpoint_sha256") is not None
            and row.get("geometry_oof_seed") is not None,
            "joint OOF row lacks all-component unseen proof",
        )
    confidence = _finite(row.get("range_confidence"))
    require(confidence is not None and 0.0 <= confidence <= 1.0, "bad confidence")
    seconds = _finite(row.get("sample_seconds"))
    require(seconds is not None and seconds >= 0.0, "bad sample duration")
    for key in (
        "prediction_progress",
        "predicted_scale_start",
        "predicted_scale_end",
        "predicted_reading",
    ):
        value = row.get(key)
        require(value is None or _finite(value) is not None, f"non-finite {key}")
    if row["range_status"]:
        require(
            row["predicted_scale_start"] is not None
            and row["predicted_scale_end"] is not None,
            "successful range has null endpoint",
        )
    if row["full_status"]:
        require(row["range_status"], "full success without range success")
        require(row["prediction_progress"] is not None, "full success lacks progress")
        require(row["predicted_reading"] is not None, "full success lacks reading")
    method = row.get("method_record")
    require(isinstance(method, Mapping), "method record absent")
    require(method.get("status") is row["full_status"], "method status drift")


def run_partition(
    *,
    plan_path: Path,
    partition: str,
    output_root: Path,
    mode: str,
    smoke_groups: int = 2,
    smoke_samples_per_group: int = 1,
    torch_cpu_threads: int = 4,
    joint_oof_seed: int | None = None,
) -> Path:
    require(partition in SUPPORTED_PARTITIONS, "partition is not inferable")
    require(mode in ("formal", "smoke"), "invalid run mode")
    plan_file, plan = load_plan(plan_path)
    require(
        mode != "formal" or plan["execution_mode"] == EXECUTION_FORMAL,
        "synthetic plan cannot produce formal predictions",
    )
    protocol_path = Path(plan["parent_protocol"]["path"])
    _, manifest_path, roster, roster_audit = load_partition_roster(
        protocol_path, partition
    )
    joint_mapping_by_id: dict[str, dict[str, Any]] = {}
    if (
        partition == "independent_validation"
        and plan["joint_oof"]["status"] == "bound_for_overlap_audit"
    ):
        _, _, joint_mapping = load_joint_oof_cohort(
            Path(plan["joint_oof"]["summary"]["path"])
        )
        joint_mapping_by_id = {
            str(row["sample_id"]): row for row in joint_mapping
        }
    if joint_oof_seed is not None:
        joint_oof_seed = int(joint_oof_seed)
        require(mode == "formal", "joint OOF shard must be a formal run")
        require(
            partition == "independent_validation",
            "joint OOF shard is restricted to independent_validation",
        )
        require(
            joint_oof_seed in EXPECTED_JOINT_OOF_BY_SEED,
            "joint OOF shard seed is not frozen",
        )
        require(bool(joint_mapping_by_id), "joint OOF shard mapping is absent")
        selected = [
            row
            for row in roster
            if (
                str(row["sample_id"]) in joint_mapping_by_id
                and int(joint_mapping_by_id[str(row["sample_id"])]["pepd_seed"])
                == joint_oof_seed
            )
        ]
        require(
            len(selected) == EXPECTED_JOINT_OOF_BY_SEED[joint_oof_seed],
            "joint OOF shard inventory drift",
        )
        execution_scope = EXECUTION_SCOPE_JOINT_SHARD
    else:
        selected = (
            roster
            if mode == "formal"
            else _smoke_subset(
                roster,
                groups=smoke_groups,
                samples_per_group=smoke_samples_per_group,
            )
        )
        execution_scope = EXECUTION_SCOPE_FULL
    output = guard_public_path(
        output_root,
        label="GARC full-auto prediction output",
        must_exist=False,
        expect_file=False,
    )
    require(not output.exists(), f"refusing to overwrite prediction output: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    require(not staging.exists(), f"staging output already exists: {staging}")

    torch.set_num_threads(max(1, min(8, int(torch_cpu_threads))))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    adapter, range_pipeline = build_runtime(plan)
    bound_progress_hashes = progress_checkpoint_hashes(
        adapter.bundle.progress_binding
    )
    bound_geometry_head_hash = str(plan["garc"]["artifacts"]["geometry"]["sha256"])
    geometry_fold = plan["garc"]["geometry_fold"]
    bound_geometry_seed = geometry_fold.get("pepd_seed")
    bound_geometry_backbone_hash = (
        str(plan["garc"]["artifacts"]["geometry_backbone"]["sha256"])
        if "geometry_backbone" in plan["garc"]["artifacts"]
        else None
    )
    if execution_scope == EXECUTION_SCOPE_JOINT_SHARD:
        expected_checkpoint_hashes = {
            str(joint_mapping_by_id[str(row["sample_id"])]["pepd_checkpoint_sha256"])
            for row in selected
        }
        require(
            len(expected_checkpoint_hashes) == 1,
            "joint OOF shard maps to multiple progress checkpoints",
        )
        require(
            next(iter(expected_checkpoint_hashes)) in bound_progress_hashes,
            "joint OOF shard plan binds the wrong progress checkpoint",
        )
        require(
            bound_geometry_seed == joint_oof_seed
            and bound_geometry_backbone_hash == next(iter(expected_checkpoint_hashes)),
            "joint OOF shard plan binds the wrong geometry fold/backbone",
        )

    staging.mkdir(parents=True, exist_ok=False)
    partial_path = staging / "predictions.partial.jsonl"
    prediction_path = staging / "predictions.label_free.jsonl"
    observed_samples: set[str] = set()
    observed_groups: set[str] = set()
    range_success = 0
    full_success = 0
    durations: list[float] = []
    failures: Counter[str] = Counter()
    joint_matched_samples: set[str] = set()
    joint_matched_groups: set[str] = set()
    joint_matched_by_seed: Counter[int] = Counter()
    progress_matched_samples: set[str] = set()
    progress_matched_groups: set[str] = set()
    progress_matched_by_seed: Counter[int] = Counter()
    geometry_matched_samples: set[str] = set()
    geometry_matched_groups: set[str] = set()
    geometry_matched_by_seed: Counter[int] = Counter()
    started = time.perf_counter()
    with partial_path.open("xb") as handle:
        for index, roster_row in enumerate(selected, 1):
            image_path = resolve_public_image(str(roster_row["image_relpath"]))
            image = cv2.imread(
                str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            require(image is not None, f"cannot read public image: {image_path}")
            sample_started = time.perf_counter()
            try:
                roi, _, _, _ = public_v5.canonical_tight_roi(
                    image,
                    roster_row["dial_bbox"],
                    output_size=int(plan["garc"]["input_size"]),
                )
            except (RuntimeError, ValueError, TypeError) as error:
                roi = np.zeros(
                    (
                        int(plan["garc"]["input_size"]),
                        int(plan["garc"]["input_size"]),
                        3,
                    ),
                    dtype=np.uint8,
                )
                full = _failed_prediction(
                    adapter.bundle, f"roi_exception:{type(error).__name__}"
                )
                range_prediction = None
            else:
                range_pipeline.reset_trace()
                try:
                    full = adapter.predict(roi, input_is_canonical_meter_roi=True)
                    range_prediction = range_pipeline.last_prediction
                except ProtocolViolation:
                    raise
                except (RuntimeError, ValueError, TypeError) as error:
                    full = _failed_prediction(
                        adapter.bundle, f"pipeline_exception:{type(error).__name__}"
                    )
                    range_prediction = range_pipeline.last_prediction
            elapsed = time.perf_counter() - sample_started
            sample_id = str(roster_row["sample_id"])
            route = joint_mapping_by_id.get(sample_id)
            progress_checkpoint: str | None = None
            progress_group_unseen = False
            geometry_group_unseen = False
            jointly_unseen = False
            if route is not None:
                require(
                    image_sha256(roi) == route["canonical_roi_sha256"],
                    f"joint OOF ROI hash drift: {sample_id}",
                )
                expected_checkpoint = str(route["pepd_checkpoint_sha256"])
                if expected_checkpoint in bound_progress_hashes:
                    progress_checkpoint = expected_checkpoint
                    progress_group_unseen = True
                    progress_matched_samples.add(sample_id)
                    progress_matched_groups.add(str(roster_row["group_id"]))
                    progress_matched_by_seed[int(route["pepd_seed"])] += 1
                if (
                    int(route["pepd_seed"]) == bound_geometry_seed
                    and expected_checkpoint == bound_geometry_backbone_hash
                    and geometry_fold.get("joint_shard_all_component_unseen") is True
                ):
                    geometry_group_unseen = True
                    geometry_matched_samples.add(sample_id)
                    geometry_matched_groups.add(str(roster_row["group_id"]))
                    geometry_matched_by_seed[int(route["pepd_seed"])] += 1
                if progress_group_unseen and geometry_group_unseen:
                    jointly_unseen = True
                    joint_matched_samples.add(sample_id)
                    joint_matched_groups.add(str(roster_row["group_id"]))
                    joint_matched_by_seed[int(route["pepd_seed"])] += 1
            if progress_checkpoint is None and len(bound_progress_hashes) == 1:
                progress_checkpoint = next(iter(bound_progress_hashes))
            row = _prediction_row(
                partition=partition,
                roster_row=roster_row,
                roi=roi,
                full=full,
                range_prediction=range_prediction,
                bundle=adapter.bundle,
                sample_seconds=elapsed,
                progress_checkpoint_sha256=progress_checkpoint,
                geometry_head_checkpoint_sha256=bound_geometry_head_hash,
                geometry_backbone_checkpoint_sha256=bound_geometry_backbone_hash,
                geometry_oof_seed=bound_geometry_seed,
                progress_group_unseen=progress_group_unseen,
                geometry_group_unseen=geometry_group_unseen,
                joint_oof_eligible=jointly_unseen,
            )
            sample_id = str(row["sample_id"])
            require(sample_id not in observed_samples, f"duplicate prediction: {sample_id}")
            observed_samples.add(sample_id)
            observed_groups.add(str(row["group_id"]))
            range_success += int(row["range_status"])
            full_success += int(row["full_status"])
            if not row["full_status"]:
                failures[str(row["failure_reason"] or "unspecified")] += 1
            durations.append(elapsed)
            handle.write(canonical_bytes(row))
            print(
                f"[{index}/{len(selected)}] {sample_id} "
                f"range={row['range_status']} full={row['full_status']} "
                f"confidence={row['range_confidence']:.4f} seconds={elapsed:.3f}",
                flush=True,
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_path, prediction_path)

    expected_samples = {str(row["sample_id"]) for row in selected}
    require(observed_samples == expected_samples, "prediction roster incomplete")
    if mode == "formal" and execution_scope == EXECUTION_SCOPE_FULL:
        require(
            observed_samples == roster_audit["sample_ids"],
            "formal prediction roster incomplete",
        )
        require(
            observed_groups == roster_audit["group_ids"],
            "formal prediction group roster incomplete",
        )
    bundle_path = staging / "full_auto_bundle.json"
    adapter.bundle.write(bundle_path)
    total_seconds = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "status": "predictions_sealed",
        "mode": mode,
        "execution_scope": execution_scope,
        "joint_oof_seed": joint_oof_seed,
        "claim_eligible": mode == "formal" and plan["execution_mode"] == EXECUTION_FORMAL,
        "partition": partition,
        "samples": len(observed_samples),
        "groups": len(observed_groups),
        "sample_ids_sha256": canonical_sha256(sorted(observed_samples)),
        "group_ids_sha256": canonical_sha256(sorted(observed_groups)),
        "parent_plan": {
            "path": str(plan_file),
            "sha256": sha256_file(plan_file),
            "identity": PLAN_PROTOCOL,
        },
        "parent_protocol": dict(plan["parent_protocol"]),
        "partition_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "variant": {
            "method_name": plan["method"]["name"],
            "recognizer_kind": plan["garc"]["recognizer_kind"],
            "consensus_mode": plan["garc"]["consensus_mode"],
            "geometry_mode": plan["garc"]["geometry_mode"],
            "geometry_provider": plan["garc"]["geometry_provider"],
            "geometry_oof_seed": bound_geometry_seed,
            "posterior_top_k": plan["garc"]["posterior_top_k"],
            "bundle_sha256": adapter.bundle.bundle_sha256,
            "range_binding_sha256": adapter.bundle.range_binding_sha256,
        },
        "counts": {
            "range_successful": range_success,
            "full_successful": full_success,
            "full_failure_breakdown": dict(sorted(failures.items())),
        },
        "overlap_audit": {
            "range_independent_validation_samples": (
                EXPECTED_RANGE_INDEPENDENT[0]
                if partition == "independent_validation"
                else None
            ),
            "range_independent_validation_groups": (
                EXPECTED_RANGE_INDEPENDENT[1]
                if partition == "independent_validation"
                else None
            ),
            "joint_oof_mapping_bound": bool(joint_mapping_by_id),
            "joint_oof_mapping_samples": len(joint_mapping_by_id),
            "joint_oof_mapping_groups": len(
                {str(row["group_id"]) for row in joint_mapping_by_id.values()}
            ),
            "progress_checkpoint_matched_samples": len(progress_matched_samples),
            "progress_checkpoint_matched_groups": len(progress_matched_groups),
            "geometry_fold_matched_samples": len(geometry_matched_samples),
            "geometry_fold_matched_groups": len(geometry_matched_groups),
            "all_component_jointly_unseen_samples": len(joint_matched_samples),
            "all_component_jointly_unseen_groups": len(joint_matched_groups),
            "progress_checkpoint_matched_by_seed": {
                str(seed): progress_matched_by_seed[seed]
                for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
            },
            "geometry_fold_matched_by_seed": {
                str(seed): geometry_matched_by_seed[seed]
                for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
            },
            "all_component_jointly_unseen_by_seed": {
                str(seed): joint_matched_by_seed[seed]
                for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
            },
            "primary_fixed_fold_sensitivity": (
                plan["garc"]["geometry_fold"].get(
                    "independent_validation_sensitivity"
                )
                if partition == "independent_validation"
                else None
            ),
            "full_joint_cohort_complete_in_this_bundle": (
                len(joint_matched_samples) == EXPECTED_JOINT_OOF[0]
                and len(joint_matched_groups) == EXPECTED_JOINT_OOF[1]
            ),
            "unmatched_rows_must_be_range_or_component_only": True,
        },
        "evidence_eligibility": {
            "range_component_claim": False,
            "ocr_and_fixed_fold_geometry_sensitivity": (
                mode == "formal"
                and plan["execution_mode"] == EXECUTION_FORMAL
                and partition == "independent_validation"
                and len(observed_samples) == EXPECTED_RANGE_INDEPENDENT[0]
                and plan["garc"]["geometry_provider"]
                == "enhanced_v5_oof_fold"
            ),
            "joint_oof_end_to_end_rows_in_this_bundle": len(joint_matched_samples),
            "joint_oof_end_to_end_complete": (
                mode == "formal"
                and len(joint_matched_samples) == EXPECTED_JOINT_OOF[0]
                and len(joint_matched_groups) == EXPECTED_JOINT_OOF[1]
            ),
            "full_1080_end_to_end_claim_allowed": False,
        },
        "timing": {
            "total_seconds": total_seconds,
            "mean_sample_seconds": float(np.mean(durations)),
            "median_sample_seconds": float(np.median(durations)),
        },
        "audit": {
            "numeric_range_annotations_opened": 0,
            "pointer_annotations_opened": 0,
            "restricted_namespace_images_opened": 0,
            "public_gt_dial_bbox_used_only_for_common_roi_preparation": True,
            "same_canonical_roi_passed_to_progress_and_garc": True,
            "caller_numeric_range_supplied": 0,
            "caller_geometry_supplied": 0,
            "caller_reference_packet_supplied": 0,
        },
        "artifacts": {
            "predictions": {
                "path": "predictions.label_free.jsonl",
                "sha256": sha256_file(prediction_path),
            },
            "bundle": {
                "path": "full_auto_bundle.json",
                "sha256": sha256_file(bundle_path),
            },
        },
    }
    summary_path = staging / "summary.json"
    atomic_new_json(summary_path, summary)
    atomic_new_json(
        staging / "seal.json",
        {
            "schema_version": 1,
            "protocol": PREDICTION_PROTOCOL,
            "partition": partition,
            "mode": mode,
            "execution_scope": execution_scope,
            "joint_oof_seed": joint_oof_seed,
            "plan_sha256": sha256_file(plan_file),
            "predictions_sha256": sha256_file(prediction_path),
            "bundle_sha256": sha256_file(bundle_path),
            "summary_sha256": sha256_file(summary_path),
        },
    )
    os.replace(staging, output)
    return output / "summary.json"


def load_prediction_bundle(
    *,
    plan_path: Path,
    prediction_root: Path,
    expected_partition: str,
    allow_smoke: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    FrozenFullAutoBundle,
]:
    """Authenticate a complete label-free bundle before any annotation opens."""

    require(expected_partition in SUPPORTED_PARTITIONS, "bad expected partition")
    plan_file, plan = load_plan(plan_path)
    protocol_path = Path(plan["parent_protocol"]["path"])
    _, manifest_path, roster, roster_audit = load_partition_roster(
        protocol_path, expected_partition
    )
    root = guard_public_path(
        prediction_root, label="GARC prediction bundle", expect_file=False
    )
    require(root.is_dir(), "GARC prediction bundle is not a directory")
    summary_path = guard_public_path(root / "summary.json", label="GARC summary")
    seal_path = guard_public_path(root / "seal.json", label="GARC seal")
    summary = strict_json(summary_path)
    seal = strict_json(seal_path)
    require(summary.get("protocol") == PREDICTION_PROTOCOL, "summary identity drift")
    require(summary.get("status") == "predictions_sealed", "bundle is not sealed")
    require(summary.get("partition") == expected_partition, "partition drift")
    require(summary.get("mode") in ("formal", "smoke"), "mode drift")
    execution_scope = summary.get("execution_scope", EXECUTION_SCOPE_FULL)
    require(execution_scope in EXECUTION_SCOPES, "execution scope drift")
    joint_oof_seed = summary.get("joint_oof_seed")
    if execution_scope == EXECUTION_SCOPE_JOINT_SHARD:
        require(summary.get("mode") == "formal", "joint OOF shard is not formal")
        require(
            expected_partition == "independent_validation",
            "joint OOF shard partition drift",
        )
        require(
            joint_oof_seed in EXPECTED_JOINT_OOF_BY_SEED,
            "joint OOF shard seed drift",
        )
    else:
        require(joint_oof_seed is None, "full partition declares a joint OOF seed")
    require(allow_smoke or summary.get("mode") == "formal", "smoke bundle ineligible")
    require(
        summary.get("parent_plan", {}).get("sha256") == sha256_file(plan_file),
        "prediction plan drift",
    )
    require(
        summary.get("parent_protocol", {}).get("sha256")
        == sha256_file(protocol_path),
        "prediction protocol drift",
    )
    require(
        summary.get("partition_manifest", {}).get("sha256")
        == sha256_file(manifest_path),
        "prediction manifest drift",
    )
    require(seal.get("protocol") == PREDICTION_PROTOCOL, "seal identity drift")
    require(
        seal.get("execution_scope", EXECUTION_SCOPE_FULL) == execution_scope,
        "seal execution scope drift",
    )
    require(seal.get("joint_oof_seed") == joint_oof_seed, "seal joint seed drift")
    require(seal.get("summary_sha256") == sha256_file(summary_path), "summary seal drift")
    require(seal.get("plan_sha256") == sha256_file(plan_file), "plan seal drift")
    prediction_path = guard_public_path(
        root / str(summary["artifacts"]["predictions"]["path"]),
        label="GARC label-free predictions",
    )
    bundle_path = guard_public_path(
        root / str(summary["artifacts"]["bundle"]["path"]),
        label="GARC full-auto bundle",
    )
    require(
        sha256_file(prediction_path)
        == summary["artifacts"]["predictions"]["sha256"]
        == seal.get("predictions_sha256"),
        "prediction artifact drift",
    )
    require(
        sha256_file(bundle_path)
        == summary["artifacts"]["bundle"]["sha256"]
        == seal.get("bundle_sha256"),
        "bundle artifact drift",
    )
    bundle = FrozenFullAutoBundle.load(bundle_path)
    require(
        bundle.bundle_sha256 == summary["variant"]["bundle_sha256"],
        "runtime bundle identity drift",
    )
    rows = strict_jsonl(prediction_path)
    roster_by_id = {str(row["sample_id"]): row for row in roster}
    joint_mapping_by_id: dict[str, dict[str, Any]] = {}
    if (
        expected_partition == "independent_validation"
        and plan["joint_oof"]["status"] == "bound_for_overlap_audit"
    ):
        _, _, joint_mapping = load_joint_oof_cohort(
            Path(plan["joint_oof"]["summary"]["path"])
        )
        joint_mapping_by_id = {
            str(value["sample_id"]): value for value in joint_mapping
        }
    observed: set[str] = set()
    groups: set[str] = set()
    joint_observed: set[str] = set()
    joint_groups: set[str] = set()
    joint_seed_counts: Counter[int] = Counter()
    progress_observed: set[str] = set()
    progress_groups: set[str] = set()
    progress_seed_counts: Counter[int] = Counter()
    geometry_observed: set[str] = set()
    geometry_groups: set[str] = set()
    geometry_seed_counts: Counter[int] = Counter()
    geometry_artifacts = plan["garc"]["artifacts"]
    geometry_fold = plan["garc"]["geometry_fold"]
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        require(sample_id in roster_by_id, f"prediction outside roster: {sample_id}")
        require(sample_id not in observed, f"duplicate prediction: {sample_id}")
        validate_prediction_row(
            row,
            partition=expected_partition,
            roster_row=roster_by_id[sample_id],
        )
        require(
            row["method_record"].get("checkpoint_sha256") == bundle.bundle_sha256,
            f"method bundle drift: {sample_id}",
        )
        require(
            row["geometry_head_checkpoint_sha256"]
            == geometry_artifacts["geometry"]["sha256"],
            f"geometry head drift: {sample_id}",
        )
        expected_geometry_backbone = (
            geometry_artifacts["geometry_backbone"]["sha256"]
            if "geometry_backbone" in geometry_artifacts
            else None
        )
        require(
            row["geometry_backbone_checkpoint_sha256"]
            == expected_geometry_backbone,
            f"geometry backbone drift: {sample_id}",
        )
        require(
            row["geometry_oof_seed"] == geometry_fold.get("pepd_seed"),
            f"geometry fold seed drift: {sample_id}",
        )
        route = joint_mapping_by_id.get(sample_id)
        if row["progress_group_unseen"]:
            require(route is not None, f"progress-unseen row outside mapping: {sample_id}")
            require(
                row["progress_checkpoint_sha256"]
                == route["pepd_checkpoint_sha256"],
                f"progress-unseen checkpoint drift: {sample_id}",
            )
            progress_observed.add(sample_id)
            progress_groups.add(str(row["group_id"]))
            progress_seed_counts[int(route["pepd_seed"])] += 1
        if row["geometry_group_unseen"]:
            require(route is not None, f"geometry-unseen row outside mapping: {sample_id}")
            require(
                row["geometry_oof_seed"] == int(route["pepd_seed"])
                and row["geometry_backbone_checkpoint_sha256"]
                == route["pepd_checkpoint_sha256"],
                f"geometry-unseen fold drift: {sample_id}",
            )
            geometry_observed.add(sample_id)
            geometry_groups.add(str(row["group_id"]))
            geometry_seed_counts[int(route["pepd_seed"])] += 1
        if row["joint_oof_eligible"]:
            require(route is not None, f"joint OOF row outside mapping: {sample_id}")
            require(
                row["progress_checkpoint_sha256"]
                == route["pepd_checkpoint_sha256"],
                f"joint OOF progress checkpoint drift: {sample_id}",
            )
            require(
                row["canonical_roi_sha256"] == route["canonical_roi_sha256"],
                f"joint OOF ROI drift: {sample_id}",
            )
            joint_observed.add(sample_id)
            joint_groups.add(str(row["group_id"]))
            joint_seed_counts[int(route["pepd_seed"])] += 1
        observed.add(sample_id)
        groups.add(str(row["group_id"]))
    require(len(observed) == summary.get("samples"), "sample count drift")
    require(len(groups) == summary.get("groups"), "group count drift")
    require(
        canonical_sha256(sorted(observed)) == summary.get("sample_ids_sha256"),
        "sample roster hash drift",
    )
    require(
        canonical_sha256(sorted(groups)) == summary.get("group_ids_sha256"),
        "group roster hash drift",
    )
    if summary["mode"] == "formal" and execution_scope == EXECUTION_SCOPE_FULL:
        require(observed == roster_audit["sample_ids"], "formal sample roster incomplete")
        require(groups == roster_audit["group_ids"], "formal group roster incomplete")
    elif execution_scope == EXECUTION_SCOPE_JOINT_SHARD:
        expected_shard = {
            sample_id
            for sample_id, route in joint_mapping_by_id.items()
            if int(route["pepd_seed"]) == int(joint_oof_seed)
        }
        require(
            len(expected_shard) == EXPECTED_JOINT_OOF_BY_SEED[int(joint_oof_seed)],
            "authenticated joint OOF shard inventory drift",
        )
        require(observed == expected_shard, "joint OOF shard roster incomplete")
    overlap = summary.get("overlap_audit") or {}
    require(
        overlap.get("progress_checkpoint_matched_samples") == len(progress_observed),
        "progress OOF matched sample count drift",
    )
    require(
        overlap.get("progress_checkpoint_matched_groups") == len(progress_groups),
        "progress OOF matched group count drift",
    )
    require(
        overlap.get("progress_checkpoint_matched_by_seed")
        == {
            str(seed): progress_seed_counts[seed]
            for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
        },
        "progress OOF seed audit drift",
    )
    require(
        overlap.get("geometry_fold_matched_samples") == len(geometry_observed)
        and overlap.get("geometry_fold_matched_groups") == len(geometry_groups),
        "geometry OOF matched inventory drift",
    )
    require(
        overlap.get("geometry_fold_matched_by_seed")
        == {
            str(seed): geometry_seed_counts[seed]
            for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
        },
        "geometry OOF seed audit drift",
    )
    require(
        overlap.get("all_component_jointly_unseen_samples") == len(joint_observed)
        and overlap.get("all_component_jointly_unseen_groups") == len(joint_groups),
        "all-component joint OOF inventory drift",
    )
    require(
        overlap.get("all_component_jointly_unseen_by_seed")
        == {
            str(seed): joint_seed_counts[seed]
            for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
        },
        "all-component joint OOF seed audit drift",
    )
    audit = summary.get("audit") or {}
    for key in (
        "numeric_range_annotations_opened",
        "pointer_annotations_opened",
        "restricted_namespace_images_opened",
        "caller_numeric_range_supplied",
        "caller_geometry_supplied",
        "caller_reference_packet_supplied",
    ):
        require(audit.get(key) == 0, f"prediction audit violation: {key}")
    return summary, rows, roster, bundle


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-plan")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--progress-binding", type=Path, required=True)
    freeze.add_argument("--progress-factory", type=Path, required=True)
    freeze.add_argument(
        "--progress-factory-function", default=FACTORY_FUNCTION_DEFAULT
    )
    freeze.add_argument("--detector-checkpoint", type=Path, required=True)
    freeze.add_argument("--recognizer-checkpoint", type=Path, required=True)
    freeze.add_argument("--geometry-checkpoint", type=Path, required=True)
    freeze.add_argument(
        "--geometry-provider",
        choices=sorted(GEOMETRY_PROVIDERS),
        default="legacy_fixed_v5",
    )
    freeze.add_argument("--geometry-backbone-checkpoint", type=Path)
    freeze.add_argument("--geometry-oof-summary", type=Path)
    freeze.add_argument(
        "--geometry-oof-seed",
        type=int,
        choices=sorted(EXPECTED_JOINT_OOF_BY_SEED),
    )
    freeze.add_argument("--recognizer", choices=sorted(RECOGNIZER_KINDS), required=True)
    freeze.add_argument("--consensus", choices=sorted(CONSENSUS_MODES), required=True)
    freeze.add_argument(
        "--geometry-mode", choices=sorted(GEOMETRY_MODES), default="v5"
    )
    freeze.add_argument("--method-name", required=True)
    freeze.add_argument("--reference-mode", choices=("auto", "native"), required=True)
    freeze.add_argument("--reference-detector-sha256")
    freeze.add_argument("--device", default="cuda:0")
    freeze.add_argument("--input-size", type=int, choices=(512, 768), default=768)
    freeze.add_argument("--detector-threshold", type=float, default=0.40)
    freeze.add_argument("--posterior-top-k", type=int, default=5)
    freeze.add_argument("--consensus-config", type=Path)
    freeze.add_argument(
        "--joint-oof-summary",
        type=Path,
        help="Optional frozen 412/19 progress/range overlap summary.",
    )
    plan_mode = freeze.add_mutually_exclusive_group(required=True)
    plan_mode.add_argument("--formal-plan", action="store_true")
    plan_mode.add_argument("--synthetic-plan", action="store_true")

    infer = commands.add_parser("infer")
    infer.add_argument("--plan", type=Path, required=True)
    infer.add_argument("--partition", choices=SUPPORTED_PARTITIONS, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--smoke-groups", type=int, default=2)
    infer.add_argument("--smoke-samples-per-group", type=int, default=1)
    infer.add_argument("--torch-cpu-threads", type=int, default=4)
    infer.add_argument(
        "--joint-oof-seed",
        type=int,
        choices=sorted(EXPECTED_JOINT_OOF_BY_SEED),
        help=(
            "Formal independent-validation shard containing only rows assigned "
            "to this frozen PEPD OOF seed."
        ),
    )
    run_mode = infer.add_mutually_exclusive_group(required=True)
    run_mode.add_argument("--formal", action="store_true")
    run_mode.add_argument("--smoke", action="store_true")

    cohort = commands.add_parser("freeze-joint-oof-cohort")
    cohort.add_argument("--protocol", type=Path, required=True)
    cohort.add_argument("--output-root", type=Path, default=DEFAULT_JOINT_OOF_ROOT)
    cohort.add_argument("--input-size", type=int, choices=(512, 768), default=768)

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--plan", type=Path, required=True)
    preflight.add_argument("--partition", choices=SUPPORTED_PARTITIONS, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--groups", type=int, default=2)
    preflight.add_argument("--samples-per-group", type=int, default=1)
    preflight.add_argument("--torch-cpu-threads", type=int, default=2)

    inspect = commands.add_parser("inspect-plan")
    inspect.add_argument("--plan", type=Path, required=True)
    validate = commands.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "freeze-plan":
        result = freeze_plan(
            protocol_path=args.protocol,
            output_path=args.output,
            progress_binding_path=args.progress_binding,
            progress_factory_path=args.progress_factory,
            progress_factory_function=args.progress_factory_function,
            detector_checkpoint=args.detector_checkpoint,
            recognizer_checkpoint=args.recognizer_checkpoint,
            geometry_checkpoint=args.geometry_checkpoint,
            geometry_provider=args.geometry_provider,
            geometry_backbone_checkpoint=args.geometry_backbone_checkpoint,
            geometry_oof_summary_path=args.geometry_oof_summary,
            geometry_oof_seed=args.geometry_oof_seed,
            recognizer_kind=args.recognizer,
            consensus_mode=args.consensus,
            geometry_mode=args.geometry_mode,
            method_name=args.method_name,
            reference_mode=(
                REFERENCE_MODE_AUTO
                if args.reference_mode == "auto"
                else REFERENCE_MODE_NATIVE
            ),
            reference_detector_sha256=args.reference_detector_sha256,
            execution_mode=(
                EXECUTION_FORMAL if args.formal_plan else EXECUTION_SYNTHETIC
            ),
            device=args.device,
            input_size=args.input_size,
            detector_threshold=args.detector_threshold,
            posterior_top_k=args.posterior_top_k,
            consensus_config_path=args.consensus_config,
            joint_oof_summary_path=args.joint_oof_summary,
        )
        payload = {"plan": str(result), "sha256": sha256_file(result)}
    elif args.command == "infer":
        result = run_partition(
            plan_path=args.plan,
            partition=args.partition,
            output_root=args.output_root,
            mode="formal" if args.formal else "smoke",
            smoke_groups=args.smoke_groups,
            smoke_samples_per_group=args.smoke_samples_per_group,
            torch_cpu_threads=args.torch_cpu_threads,
            joint_oof_seed=args.joint_oof_seed,
        )
        payload = {"summary": str(result), "sha256": sha256_file(result)}
    elif args.command == "freeze-joint-oof-cohort":
        result = freeze_joint_oof_cohort(
            protocol_path=args.protocol,
            output_root=args.output_root,
            input_size=args.input_size,
        )
        payload = {"joint_oof_summary": str(result), "sha256": sha256_file(result)}
    elif args.command == "preflight":
        result = run_partition(
            plan_path=args.plan,
            partition=args.partition,
            output_root=args.output_root,
            mode="smoke",
            smoke_groups=args.groups,
            smoke_samples_per_group=args.samples_per_group,
            torch_cpu_threads=args.torch_cpu_threads,
        )
        payload = {"preflight_summary": str(result), "sha256": sha256_file(result)}
    else:
        plan_path, plan = load_plan(args.plan)
        payload = {
            "plan": str(plan_path),
            "sha256": sha256_file(plan_path),
            "method": plan["method"],
            "garc": plan["garc"],
            "partitions": {
                key: {
                    "samples": value["samples"],
                    "groups": value["groups"],
                    "sha256": value["sha256"],
                }
                for key, value in plan["partitions"].items()
            },
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CONSENSUS_MODES",
    "EXECUTION_SCOPE_FULL",
    "EXECUTION_SCOPE_JOINT_SHARD",
    "EXECUTION_SCOPES",
    "GEOMETRY_MODES",
    "GEOMETRY_PROVIDERS",
    "EXPECTED_JOINT_OOF_GROUPS_BY_SEED",
    "EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY",
    "GARCFullAutoRangePipeline",
    "PLAN_PROTOCOL",
    "PREDICTION_PROTOCOL",
    "RECOGNIZER_KINDS",
    "SUPPORTED_PARTITIONS",
    "_ConsensusModeDecoder",
    "freeze_plan",
    "freeze_joint_oof_cohort",
    "load_joint_oof_cohort",
    "load_plan",
    "load_prediction_bundle",
    "run_partition",
    "validate_prediction_row",
]
