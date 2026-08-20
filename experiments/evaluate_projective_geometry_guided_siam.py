"""Inference and frozen XM2-development selection for PG-SIAM-v3.

The prediction path is label-free after an optional roster-only filter.  For
each robustness condition it builds the deployed SARN/quad two-view input,
runs the terminal PG-SIAM-v3 checkpoint, writes the ordinary paper prediction
schema, and stores parent/gate/rectification diagnostics in a separate JSONL.

The selection command is deliberately narrow: by default it accepts only the
frozen 434-row XM2-development label artifact and scores the three groups named
in ``pgsiam_v3_protocol.json``.  It never reads any external-evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments import robustness_degradations
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.projective_geometry_guided_siam import (
    ARCHITECTURE as MODEL_ARCHITECTURE,
    METHOD_PREFIX,
    PROTOCOL as MODEL_PROTOCOL,
    ProjectiveGeometryGuidedSIAM,
)
from experiments.projective_geometry_views import (
    PROTOCOL as VIEW_PROTOCOL,
    ProjectiveGeometryViews,
    build_projective_geometry_views,
)
from experiments.resnet18_direct_progress import IMAGE_SIZE, _canonical_json_bytes
from experiments.run_cagh_v5_plain_paper_batch import (
    CONDITIONS,
    OUTPUT_KEYS,
    ROBUSTNESS_SEED,
    ManifestRow,
    canonical_roi_pixel_sha256,
    load_canonical_roi,
    load_manifest,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


PREDICTION_PROTOCOL: Final[str] = "projective_geometry_guided_siam_v3_prediction_v1"
DIAGNOSTIC_PROTOCOL: Final[str] = "projective_geometry_guided_siam_v3_diagnostic_v1"
SELECTION_SUMMARY_PROTOCOL: Final[str] = (
    "projective_geometry_guided_siam_v3_xm2_selection_v1"
)
DEFAULT_FROZEN_PROTOCOL: Final[Path] = Path(__file__).with_name("pgsiam_v3_protocol.json")
DEFAULT_XM2_LABELS: Final[Path] = Path(
    r"C:\pointer_read\cagh_v5_plain_real_xm2_development_v1\labels.jsonl"
)
FROZEN_XM2_LABELS_SHA256: Final[str] = (
    "94e24f2c013b0f418f29ec7c7c8c3b0506c8c7e9c6fb88b9272de0e994e8e9a8"
)
FROZEN_XM2_ROWS: Final[int] = 434
FROZEN_XM2_GROUPS: Final[int] = 11
IDENTITY_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
SOURCE_PROVENANCE_PROTOCOL: Final[str] = (
    "pgsiam_v3_content_addressed_source_provenance_v1"
)
DEFAULT_SOURCE_PROVENANCE_REGISTRY: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "source_snapshots"
    / "pgsiam_v3"
    / "provenance_registry.json"
)


class PGSIAMEvaluationError(RuntimeError):
    """The checkpoint, artifact, or frozen selection contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PGSIAMEvaluationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Any) -> str:
    return _canonical_json_bytes(value).decode("utf-8") + "\n"


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PGSIAMEvaluationError(f"cannot read {label}: {source}") from exc
    _require(isinstance(value, Mapping), f"{label} is not a JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"{label} does not exist: {source}")
    result: list[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PGSIAMEvaluationError(
                    f"{label} line {line_number} is invalid JSON"
                ) from exc
            _require(isinstance(value, Mapping), f"{label} line {line_number} is not an object")
            result.append(value)
    _require(bool(result), f"{label} is empty")
    return result


def _validate_frozen_protocol(path: Path) -> tuple[Path, Mapping[str, Any]]:
    source = Path(path).resolve()
    protocol = _load_json(source, label="PG-SIAM frozen protocol")
    _require(
        protocol.get("protocol")
        == "projective_geometry_guided_siam_v3_single_seed_screen_v1",
        "PG-SIAM frozen protocol identity mismatch",
    )
    training = protocol.get("training")
    gate = protocol.get("selection_gate")
    _require(isinstance(training, Mapping), "frozen training contract is missing")
    _require(isinstance(gate, Mapping), "frozen selection gate is missing")
    groups = training.get("real_selection_groups")
    _require(
        isinstance(groups, list)
        and len(groups) == 3
        and len(set(map(str, groups))) == 3,
        "frozen selection group roster is invalid",
    )
    _require(training.get("real_selection_samples") == 85, "selection sample count drift")
    _require(protocol.get("external_evaluation"), "external-evaluation guard is missing")
    return source, protocol


def _sha256_string(value: Any, *, label: str) -> str:
    digest = str(value).lower()
    _require(
        len(digest) == 64 and all(character in "0123456789abcdef" for character in digest),
        f"{label} is not a SHA-256 digest",
    )
    return digest


def _load_source_provenance_registry(path: Path) -> list[Mapping[str, Any]]:
    source = Path(path).resolve()
    registry = _load_json(source, label="PG-SIAM source provenance registry")
    _require(registry.get("schema_version") == 1, "source provenance schema mismatch")
    _require(
        registry.get("protocol") == SOURCE_PROVENANCE_PROTOCOL,
        "source provenance protocol mismatch",
    )
    _require(registry.get("status") == "sealed", "source provenance registry is not sealed")
    entries = registry.get("entries")
    _require(isinstance(entries, list) and bool(entries), "source provenance entries are missing")
    identities: set[tuple[str, str, str, str]] = set()
    parsed: list[Mapping[str, Any]] = []
    for index, raw in enumerate(entries):
        _require(isinstance(raw, Mapping), f"source provenance entry {index} is invalid")
        checkpoint_digest = _sha256_string(
            raw.get("checkpoint_sha256"), label=f"source provenance entry {index} checkpoint"
        )
        source_role = str(raw.get("source_role", ""))
        recorded_path = str(raw.get("recorded_path", ""))
        bound_digest = _sha256_string(
            raw.get("bound_sha256"), label=f"source provenance entry {index} binding"
        )
        archive_digest = _sha256_string(
            raw.get("archive_sha256"), label=f"source provenance entry {index} archive"
        )
        _require(source_role and recorded_path, f"source provenance entry {index} identity is empty")
        _require(
            archive_digest == bound_digest,
            f"source provenance entry {index} substitutes a different hash",
        )
        archive_relative = Path(str(raw.get("archive_path", "")))
        _require(
            bool(str(archive_relative)) and not archive_relative.is_absolute(),
            f"source provenance entry {index} archive path must be relative",
        )
        archive_path = (source.parent / archive_relative).resolve()
        _require(
            archive_path.is_relative_to(source.parent),
            f"source provenance entry {index} escapes the archive root",
        )
        _require(
            f".sha256-{bound_digest}." in archive_path.name,
            f"source provenance entry {index} is not content-addressed",
        )
        identity = (checkpoint_digest, source_role, recorded_path, bound_digest)
        _require(identity not in identities, f"duplicate source provenance entry {index}")
        identities.add(identity)
        parsed.append(
            {
                "checkpoint_sha256": checkpoint_digest,
                "source_role": source_role,
                "recorded_path": recorded_path,
                "bound_sha256": bound_digest,
                "archive_sha256": archive_digest,
                "archive_path": archive_path,
            }
        )
    return parsed


def _validate_source_bindings(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    provenance_registry_path: Path = DEFAULT_SOURCE_PROVENANCE_REGISTRY,
) -> dict[str, Path]:
    """Resolve every source without ever weakening its checkpoint-bound hash.

    The path recorded inside the checkpoint remains the first choice.  A
    content-addressed archive is considered only when that path is missing or
    has changed, and only an exact registry row bound to this checkpoint hash,
    source role, recorded path, and source hash may relocate it.
    """

    checkpoint_digest = _sha256_string(checkpoint_sha256, label="v3 checkpoint")
    bindings = checkpoint.get("source_files")
    _require(isinstance(bindings, Mapping) and bool(bindings), "v3 source bindings are missing")
    resolved: dict[str, Path] = {}
    provenance: list[Mapping[str, Any]] | None = None
    for name, raw in bindings.items():
        _require(isinstance(raw, Mapping), f"v3 {name} source binding is invalid")
        recorded_path = str(raw.get("path", ""))
        _require(bool(recorded_path), f"v3 {name} source path is empty")
        path = Path(recorded_path).resolve()
        digest = _sha256_string(raw.get("sha256"), label=f"v3 {name} source")
        if path.is_file() and _sha256_file(path) == digest:
            resolved[str(name)] = path
            continue
        if provenance is None:
            provenance = _load_source_provenance_registry(provenance_registry_path)
        matches = [
            entry
            for entry in provenance
            if entry["checkpoint_sha256"] == checkpoint_digest
            and entry["source_role"] == str(name)
            and entry["recorded_path"] == recorded_path
            and entry["bound_sha256"] == digest
        ]
        _require(
            len(matches) == 1,
            f"v3 {name} source hash drift and no exact content-addressed relocation",
        )
        archive_path = Path(matches[0]["archive_path"])
        _require(archive_path.is_file(), f"v3 {name} archived source is missing: {archive_path}")
        _require(
            _sha256_file(archive_path) == digest == matches[0]["archive_sha256"],
            f"v3 {name} archived source hash drift",
        )
        resolved[str(name)] = archive_path
    return resolved


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device_name: str,
    provenance_registry_path: Path = DEFAULT_SOURCE_PROVENANCE_REGISTRY,
) -> tuple[str, ProjectiveGeometryGuidedSIAM, Mapping[str, Any]]:
    """Load one terminal v3 checkpoint with its source/protocol bindings."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"PG-SIAM-v3 checkpoint does not exist: {source}")
    checkpoint_sha256 = _sha256_file(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "PG-SIAM-v3 checkpoint is not an object")
    _require(checkpoint.get("protocol") == MODEL_PROTOCOL, "v3 checkpoint protocol mismatch")
    _require(
        checkpoint.get("architecture") == MODEL_ARCHITECTURE,
        "v3 checkpoint architecture mismatch",
    )
    _require(checkpoint.get("image_size") == IMAGE_SIZE, "v3 checkpoint image-size mismatch")
    seed = int(checkpoint.get("seed", -1))
    method = f"{METHOD_PREFIX}_seed_{seed}"
    _require(checkpoint.get("method") == method, "v3 checkpoint method identity mismatch")
    _validate_source_bindings(
        checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        provenance_registry_path=provenance_registry_path,
    )
    # The trainer records the frozen contract both by identity and as the
    # ``screen_protocol`` source binding.  Also accept the earlier explicit
    # ``frozen_protocol`` binding shape so already-produced pilot checkpoints
    # remain readable without weakening the file-hash check.
    expected_screen = "projective_geometry_guided_siam_v3_single_seed_screen_v1"
    _require(checkpoint.get("screen_protocol") == expected_screen, "v3 screen protocol mismatch")
    frozen = checkpoint.get("frozen_protocol")
    if isinstance(frozen, Mapping):
        frozen_path = Path(str(frozen.get("path", ""))).resolve()
        _require(frozen_path.is_file(), "v3 bound frozen protocol is missing")
        _require(frozen.get("sha256") == _sha256_file(frozen_path), "v3 frozen-protocol hash drift")
        _require(frozen.get("protocol") == expected_screen, "v3 bound frozen-protocol identity mismatch")
    else:
        source_files = checkpoint["source_files"]
        screen_binding = source_files.get("screen_protocol")
        _require(isinstance(screen_binding, Mapping), "v3 screen-protocol source binding is missing")
        screen_path = Path(str(screen_binding.get("path", ""))).resolve()
        screen_value = _load_json(screen_path, label="v3 bound screen protocol")
        _require(screen_value.get("protocol") == expected_screen, "v3 bound screen protocol identity mismatch")
    state = checkpoint.get("model_state")
    _require(isinstance(state, Mapping), "v3 model state is missing")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model = ProjectiveGeometryGuidedSIAM(imagenet_pretrained=False)
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return method, model.to(device).eval(), checkpoint


def _resize_pixel_center_transform(
    input_hw: tuple[int, int], output_hw: tuple[int, int]
) -> np.ndarray:
    in_h, in_w = map(int, input_hw)
    out_h, out_w = map(int, output_hw)
    _require(min(in_h, in_w, out_h, out_w) >= 2, "invalid resize dimensions")
    sx, sy = float(out_w) / float(in_w), float(out_h) / float(in_h)
    return np.asarray(
        [
            [sx, 0.0, 0.5 * sx - 0.5],
            [0.0, sy, 0.5 * sy - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def resize_homography_pixel_centers(
    homography_a_to_b: np.ndarray,
    *,
    input_a_hw: tuple[int, int],
    input_b_hw: tuple[int, int],
    output_a_hw: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
    output_b_hw: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> np.ndarray:
    """Conjugate native-view H through OpenCV's pixel-centre resize maps."""

    matrix = np.asarray(homography_a_to_b, dtype=np.float64).reshape(3, 3)
    a_resize = _resize_pixel_center_transform(input_a_hw, output_a_hw)
    b_resize = _resize_pixel_center_transform(input_b_hw, output_b_hw)
    resized = b_resize @ matrix @ np.linalg.inv(a_resize)
    _require(np.all(np.isfinite(resized)), "resized homography is non-finite")
    scale = float(resized[2, 2])
    _require(abs(scale) >= 1.0e-12, "resized homography has invalid scale")
    return np.ascontiguousarray(resized / scale, dtype=np.float32)


def model_batch(
    model: ProjectiveGeometryGuidedSIAM,
    views: Sequence[ProjectiveGeometryViews],
    *,
    device: torch.device,
) -> list[dict[str, float]]:
    _require(bool(views), "PG-SIAM prediction batch is empty")
    tensors_a: list[torch.Tensor] = []
    tensors_b: list[torch.Tensor] = []
    matrices: list[np.ndarray] = []
    for item in views:
        native_a_hw = tuple(map(int, item.view_a_bgr.shape[:2]))
        native_b_hw = tuple(map(int, item.view_b_bgr.shape[:2]))
        resized_a = direct_resize_whole_roi(item.view_a_bgr, size=IMAGE_SIZE)
        resized_b = direct_resize_whole_roi(item.view_b_bgr, size=IMAGE_SIZE)
        tensors_a.append(normalized_rgb_tensor(resized_a))
        tensors_b.append(normalized_rgb_tensor(resized_b))
        matrices.append(
            resize_homography_pixel_centers(
                item.homography_a_to_b,
                input_a_hw=native_a_hw,
                input_b_hw=native_b_hw,
            )
        )
    batch_a = torch.stack(tensors_a).to(device)
    batch_b = torch.stack(tensors_b).to(device)
    homographies = torch.from_numpy(np.stack(matrices)).to(device)
    confidence = torch.as_tensor(
        [float(item.confidence) for item in views], dtype=torch.float32, device=device
    )
    active = torch.as_tensor(
        [bool(item.active) for item in views], dtype=torch.bool, device=device
    )
    with torch.inference_mode():
        output = model.forward_pair(batch_a, batch_b, homographies, confidence, active)
    required = {"parent_progress", "candidate_progress", "trusted_progress", "effective_gate"}
    _require(required <= set(output), "v3 forward_pair output is incomplete")
    rows: list[dict[str, float]] = []
    for index in range(len(views)):
        row = {
            name: float(output[name][index].detach().cpu().item()) for name in sorted(required)
        }
        _require(
            all(math.isfinite(value) for value in row.values())
            and 0.0 <= row["parent_progress"] <= 1.0
            and 0.0 <= row["candidate_progress"] <= 1.0
            and 0.0 <= row["trusted_progress"] <= 1.0
            and 0.0 <= row["effective_gate"] <= 1.0,
            "v3 forward_pair returned an invalid value",
        )
        rows.append(row)
    return rows


def _diagnostic_row(
    *,
    source: ManifestRow,
    method: str,
    condition: str,
    condition_hash: str,
    views: ProjectiveGeometryViews | None,
    model_values: Mapping[str, float] | None,
    failure_code: str | None,
) -> dict[str, Any]:
    rect = views.rectification.metadata if views is not None else None
    sarn = views.sarn_decision if views is not None else None
    return {
        "schema_version": 1,
        "protocol": DIAGNOSTIC_PROTOCOL,
        "view_protocol": VIEW_PROTOCOL,
        "sample_id": source.sample_id,
        "method": method,
        "condition": condition,
        "robustness_seed": ROBUSTNESS_SEED,
        "condition_pixel_sha256": condition_hash,
        "status": "pass" if failure_code is None else "fail",
        "failure_code": failure_code,
        "geometry_active": bool(views.active) if views is not None else False,
        "geometry_confidence": float(views.confidence) if views is not None else 0.0,
        "geometry_fallback_reason": views.fallback_reason if views is not None else None,
        "homography_a_to_b": (
            np.asarray(views.homography_a_to_b, dtype=float).tolist()
            if views is not None
            else None
        ),
        "sarn_applied": bool(sarn.applied) if sarn is not None else False,
        "sarn_fallback_reason": sarn.fallback_reason if sarn is not None else None,
        "sarn_bbox_xyxy": list(sarn.bbox_xyxy) if sarn is not None and sarn.bbox_xyxy else None,
        "sarn_support_area_fraction": sarn.support_area_fraction if sarn is not None else None,
        "rectification_applied": bool(rect.applied) if rect is not None else False,
        "rectification_fallback_reason": rect.fallback_reason if rect is not None else None,
        "rectification_source_area_fraction": (
            rect.source_quad_area_fraction if rect is not None else None
        ),
        "rectification_homography_condition_number": (
            rect.homography_condition_number if rect is not None else None
        ),
        "parent_progress": model_values.get("parent_progress") if model_values else None,
        "candidate_progress": model_values.get("candidate_progress") if model_values else None,
        "trusted_progress": model_values.get("trusted_progress") if model_values else None,
        "effective_gate": model_values.get("effective_gate") if model_values else None,
    }


def _prediction_row(
    *,
    source: ManifestRow,
    method: str,
    condition: str,
    condition_hash: str,
    progress: float | None,
    failure_code: str | None,
) -> dict[str, Any]:
    passed = progress is not None and failure_code is None
    row = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "sample_id": source.sample_id,
        "method": method,
        "condition": condition,
        "robustness_seed": ROBUSTNESS_SEED,
        "status": "pass" if passed else "fail",
        "normalized_progress": progress if passed else None,
        "failure_code": None if passed else failure_code,
        "roi_png_sha256": source.roi_png_sha256,
        "roi_pixel_sha256": source.roi_pixel_sha256,
        "condition_pixel_sha256": condition_hash,
    }
    _require(set(row) == set(OUTPUT_KEYS), "prediction output schema drift")
    return row


def _selection_labels(
    labels_path: Path,
    protocol_path: Path,
    *,
    enforce_frozen_xm2: bool,
) -> tuple[dict[str, tuple[str, float]], tuple[str, ...], Path, Mapping[str, Any]]:
    protocol_file, protocol = _validate_frozen_protocol(protocol_path)
    labels_file = Path(labels_path).resolve()
    if enforce_frozen_xm2:
        _require(
            _sha256_file(labels_file) == FROZEN_XM2_LABELS_SHA256,
            "selection labels are not the frozen XM2-development artifact",
        )
    raw = _load_jsonl(labels_file, label="XM2-development labels")
    parsed: dict[str, tuple[str, float]] = {}
    all_groups: set[str] = set()
    for row in raw:
        sample_id = str(row.get("sample_id", ""))
        group_id = str(row.get("group_id", ""))
        try:
            target = float(row.get("normalized_progress"))
        except (TypeError, ValueError) as exc:
            raise PGSIAMEvaluationError(f"invalid target for {sample_id}") from exc
        _require(sample_id and group_id, "label row has empty identity")
        _require(sample_id not in parsed, f"duplicate label sample: {sample_id}")
        _require(math.isfinite(target) and 0.0 <= target <= 1.0, f"invalid target: {sample_id}")
        parsed[sample_id] = (group_id, target)
        all_groups.add(group_id)
    if enforce_frozen_xm2:
        _require(len(parsed) == FROZEN_XM2_ROWS, "XM2-development label row count drift")
        _require(len(all_groups) == FROZEN_XM2_GROUPS, "XM2-development group count drift")
    training = protocol["training"]
    groups = tuple(map(str, training["real_selection_groups"]))
    selected = {sample: value for sample, value in parsed.items() if value[0] in groups}
    _require(
        len(selected) == int(training["real_selection_samples"]),
        "frozen selection roster sample count drift",
    )
    _require({value[0] for value in selected.values()} == set(groups), "selection group missing")
    return selected, groups, protocol_file, protocol


def run_prediction(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    device_name: str = "cuda:0",
    conditions: Sequence[str] = CONDITIONS,
    selection_labels_path: Path | None = None,
    frozen_protocol_path: Path = DEFAULT_FROZEN_PROTOCOL,
    checkpoint_loader: Callable[..., tuple[str, ProjectiveGeometryGuidedSIAM, Mapping[str, Any]]] = load_checkpoint_model,
    views_builder: Callable[[np.ndarray], ProjectiveGeometryViews] = build_projective_geometry_views,
    batch_runner: Callable[..., list[dict[str, float]]] = model_batch,
) -> Mapping[str, Any]:
    """Generate OUTPUT_KEYS-compatible predictions plus diagnostic sidecars."""

    selected_conditions = tuple(conditions)
    _require(
        bool(selected_conditions)
        and len(selected_conditions) == len(set(selected_conditions))
        and set(selected_conditions) <= set(CONDITIONS),
        "invalid prediction condition roster",
    )
    _require(
        robustness_degradations.degradation_names(include_clean=True) == CONDITIONS,
        "robustness condition roster drift",
    )
    rows = list(load_manifest(manifest_path))
    selection_groups: tuple[str, ...] | None = None
    if selection_labels_path is not None:
        selected_labels, selection_groups, _, _ = _selection_labels(
            selection_labels_path,
            frozen_protocol_path,
            enforce_frozen_xm2=True,
        )
        rows = [row for row in rows if row.sample_id in selected_labels]
        _require(len(rows) == len(selected_labels), "manifest does not cover frozen selection roster")
    method, model, checkpoint = checkpoint_loader(checkpoint_path, device_name=device_name)
    device = next(model.parameters()).device
    output = Path(output_path).resolve()
    diagnostics = Path(diagnostics_path).resolve()
    summary_file = Path(summary_path).resolve()
    _require(len({output, diagnostics, summary_file}) == 3, "output artifacts must differ")
    for path in (output, diagnostics, summary_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for source in rows:
        _payload, clean = load_canonical_roi(source)
        view_items: list[ProjectiveGeometryViews | None] = []
        hashes: list[str] = []
        failures: list[str | None] = []
        for condition in selected_conditions:
            conditioned, _metadata = robustness_degradations.apply_degradation(
                clean, condition, sample_id=source.sample_id, seed=ROBUSTNESS_SEED
            )
            conditioned = np.ascontiguousarray(conditioned)
            hashes.append(canonical_roi_pixel_sha256(conditioned))
            try:
                view_items.append(views_builder(conditioned))
                failures.append(None)
            except Exception as exc:
                view_items.append(None)
                failures.append(f"view_exception:{type(exc).__name__}")
        valid_indices = [index for index, item in enumerate(view_items) if item is not None]
        model_values: list[dict[str, float] | None] = [None] * len(view_items)
        if valid_indices:
            try:
                valid_views = [view_items[index] for index in valid_indices]
                values = batch_runner(model, valid_views, device=device)
                _require(len(values) == len(valid_indices), "v3 prediction batch length mismatch")
                for index, value in zip(valid_indices, values, strict=True):
                    model_values[index] = value
            except Exception as exc:
                for index in valid_indices:
                    failures[index] = f"model_exception:{type(exc).__name__}"
        for index, condition in enumerate(selected_conditions):
            values = model_values[index]
            failure = failures[index]
            progress = values.get("trusted_progress") if values is not None and failure is None else None
            prediction_rows.append(
                _prediction_row(
                    source=source,
                    method=method,
                    condition=condition,
                    condition_hash=hashes[index],
                    progress=progress,
                    failure_code=failure,
                )
            )
            diagnostic_rows.append(
                _diagnostic_row(
                    source=source,
                    method=method,
                    condition=condition,
                    condition_hash=hashes[index],
                    views=view_items[index],
                    model_values=values,
                    failure_code=failure,
                )
            )
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(_json_line(row) for row in prediction_rows)
    with diagnostics.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(_json_line(row) for row in diagnostic_rows)

    by_condition: dict[str, Any] = {}
    for condition in selected_conditions:
        subset = [row for row in diagnostic_rows if row["condition"] == condition]
        passed = [row for row in subset if row["status"] == "pass"]
        by_condition[condition] = {
            "rows": len(subset),
            "passes": len(passed),
            "failures": len(subset) - len(passed),
            "geometry_active": sum(bool(row["geometry_active"]) for row in subset),
            "rectification_applied": sum(
                bool(row["rectification_applied"]) for row in subset
            ),
            "effective_gate_nonzero": sum(
                float(row["effective_gate"] or 0.0) > 0.0 for row in passed
            ),
            "parent_exact": sum(
                row["trusted_progress"] == row["parent_progress"] for row in passed
            ),
            "mean_effective_gate": (
                float(np.mean([float(row["effective_gate"]) for row in passed]))
                if passed
                else None
            ),
            "fallback_reasons": dict(
                sorted(Counter(str(row["geometry_fallback_reason"]) for row in subset if row["geometry_fallback_reason"]).items())
            ),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": _sha256_file(Path(checkpoint_path).resolve()),
            "model_protocol": checkpoint.get("protocol"),
            "architecture": checkpoint.get("architecture"),
            "seed": checkpoint.get("seed"),
        },
        "method": method,
        "manifest": str(Path(manifest_path).resolve()),
        "selection_only": selection_labels_path is not None,
        "selection_groups": list(selection_groups) if selection_groups else None,
        "samples": len(rows),
        "prediction_rows": len(prediction_rows),
        "output": {"path": str(output), "sha256": _sha256_file(output)},
        "diagnostics": {"path": str(diagnostics), "sha256": _sha256_file(diagnostics)},
        "conditions": by_condition,
        "external_inference_performed": False,
    }
    summary_file.write_text(_json_line(summary), encoding="utf-8", newline="\n")
    return summary


def _percentile_interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return {"low": float(low), "high": float(high)}


def _paired_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    groups: Sequence[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    _require(bool(rows), "paired metric has no rows")
    by_group: dict[str, list[float]] = defaultdict(list)
    parent_errors: list[float] = []
    trusted_errors: list[float] = []
    for row in rows:
        target = float(row["target"])
        parent_error = abs(float(row["parent_progress"]) - target)
        trusted_error = abs(float(row["trusted_progress"]) - target)
        parent_errors.append(parent_error)
        trusted_errors.append(trusted_error)
        by_group[str(row["group_id"])].append(trusted_error - parent_error)
    group_names = tuple(groups)
    _require(set(by_group) == set(group_names), "paired metric group roster mismatch")
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = np.empty(int(bootstrap_replicates), dtype=np.float64)
    for index in range(int(bootstrap_replicates)):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        values = [value for group in sampled for value in by_group[str(group)]]
        boot[index] = float(np.mean(values))
    parent_nmae = float(np.mean(parent_errors))
    trusted_nmae = float(np.mean(trusted_errors))
    return {
        "rows": len(rows),
        "parent_nmae": parent_nmae,
        "trusted_nmae": trusted_nmae,
        "delta_nmae": trusted_nmae - parent_nmae,
        "delta_nmae_group_bootstrap_ci95": _percentile_interval(boot),
        "group_delta_nmae": {
            group: float(np.mean(by_group[group])) for group in group_names
        },
    }


def score_selection(
    *,
    predictions_path: Path,
    diagnostics_path: Path,
    labels_path: Path,
    frozen_protocol_path: Path,
    output_path: Path,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    enforce_frozen_xm2: bool = True,
) -> Mapping[str, Any]:
    """Score the three frozen XM2-development groups and compute the gate."""

    _require(bootstrap_replicates >= 100, "bootstrap replicate count must be >= 100")
    labels, groups, protocol_file, protocol = _selection_labels(
        labels_path, frozen_protocol_path, enforce_frozen_xm2=enforce_frozen_xm2
    )
    predictions = _load_jsonl(predictions_path, label="v3 predictions")
    diagnostics = _load_jsonl(diagnostics_path, label="v3 diagnostics")
    selected_ids = set(labels)
    if enforce_frozen_xm2:
        _require(
            {str(row.get("sample_id", "")) for row in predictions} <= selected_ids,
            "selection prediction artifact contains non-selection samples",
        )
        _require(
            {str(row.get("sample_id", "")) for row in diagnostics} <= selected_ids,
            "selection diagnostic artifact contains non-selection samples",
        )
    pred_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    diag_map: dict[tuple[str, str], Mapping[str, Any]] = {}
    methods: set[str] = set()
    for row in predictions:
        sample_id, condition = str(row.get("sample_id", "")), str(row.get("condition", ""))
        if sample_id not in selected_ids:
            continue
        key = (sample_id, condition)
        _require(key not in pred_map, f"duplicate selected prediction: {key}")
        _require(set(row) == set(OUTPUT_KEYS), f"prediction schema drift: {key}")
        _require(row.get("protocol") == PREDICTION_PROTOCOL, f"prediction protocol drift: {key}")
        _require(row.get("robustness_seed") == ROBUSTNESS_SEED, f"robustness seed drift: {key}")
        pred_map[key] = row
        methods.add(str(row.get("method", "")))
    for row in diagnostics:
        sample_id, condition = str(row.get("sample_id", "")), str(row.get("condition", ""))
        if sample_id not in selected_ids:
            continue
        key = (sample_id, condition)
        _require(key not in diag_map, f"duplicate selected diagnostic: {key}")
        _require(row.get("protocol") == DIAGNOSTIC_PROTOCOL, f"diagnostic protocol drift: {key}")
        diag_map[key] = row
    expected = {(sample, condition) for sample in selected_ids for condition in CONDITIONS}
    _require(set(pred_map) == expected, "selected prediction roster is incomplete")
    _require(set(diag_map) == expected, "selected diagnostic roster is incomplete")
    _require(len(methods) == 1 and "" not in methods, "selection must contain one v3 method")

    paired_rows: list[dict[str, Any]] = []
    failure_count = 0
    identity_mismatches: list[dict[str, str]] = []
    for key in sorted(expected):
        prediction, diagnostic = pred_map[key], diag_map[key]
        _require(prediction.get("condition_pixel_sha256") == diagnostic.get("condition_pixel_sha256"), f"condition hash mismatch: {key}")
        _require(prediction.get("method") == diagnostic.get("method"), f"method mismatch: {key}")
        passed = prediction.get("status") == "pass" and diagnostic.get("status") == "pass"
        trusted = prediction.get("normalized_progress")
        parent = diagnostic.get("parent_progress")
        if not passed or trusted is None or parent is None:
            failure_count += 1
            continue
        trusted_value, parent_value = float(trusted), float(parent)
        _require(
            trusted_value == float(diagnostic.get("trusted_progress")),
            f"prediction/diagnostic value mismatch: {key}",
        )
        group_id, target = labels[key[0]]
        paired_rows.append(
            {
                "sample_id": key[0],
                "condition": key[1],
                "group_id": group_id,
                "target": target,
                "parent_progress": parent_value,
                "trusted_progress": trusted_value,
            }
        )
        if key[1] in IDENTITY_CONDITIONS and trusted_value != parent_value:
            identity_mismatches.append({"sample_id": key[0], "condition": key[1]})

    metrics: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
        subset = [row for row in paired_rows if row["condition"] == condition]
        if len(subset) == len(labels):
            metrics[condition] = _paired_metric(
                subset,
                groups=groups,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=int(protocol["seed"]) + offset,
            )
        else:
            metrics[condition] = None
    pooled = [row for row in paired_rows if row["condition"] in PROJECTIVE_CONDITIONS]
    pooled_metric = (
        _paired_metric(
            pooled,
            groups=groups,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=int(protocol["seed"]) + 100,
        )
        if len(pooled) == len(labels) * len(PROJECTIVE_CONDITIONS)
        else None
    )
    thresholds = protocol["selection_gate"]
    each_limit = float(thresholds["each_projective_delta_nmae_ci95_high_max"])
    pooled_limit = float(thresholds["pooled_projective_delta_nmae_ci95_high_max"])
    p90_limit = float(thresholds["group_level_excess_p90_max"])
    projective_checks = {
        condition: bool(
            metrics[condition] is not None
            and metrics[condition]["delta_nmae_group_bootstrap_ci95"]["high"] <= each_limit
        )
        for condition in PROJECTIVE_CONDITIONS
    }
    group_excess_p90 = (
        float(np.quantile(list(pooled_metric["group_delta_nmae"].values()), 0.90))
        if pooled_metric is not None
        else None
    )
    checks = {
        "complete_no_failures": failure_count == 0,
        "clean_and_blur_bit_exact_parent": not identity_mismatches,
        "each_projective_ci95_high": projective_checks,
        "pooled_projective_ci95_high": bool(
            pooled_metric is not None
            and pooled_metric["delta_nmae_group_bootstrap_ci95"]["high"] <= pooled_limit
        ),
        "group_level_excess_p90": bool(
            group_excess_p90 is not None and group_excess_p90 <= p90_limit
        ),
    }
    go = bool(
        checks["complete_no_failures"]
        and checks["clean_and_blur_bit_exact_parent"]
        and all(projective_checks.values())
        and checks["pooled_projective_ci95_high"]
        and checks["group_level_excess_p90"]
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": SELECTION_SUMMARY_PROTOCOL,
        "decision": "GO" if go else "NO-GO",
        "external_evaluation_authorized": go,
        "method": next(iter(methods)),
        "selection_scope": {
            "dataset": "XM2 development",
            "groups": list(groups),
            "samples": len(labels),
            "conditions": list(CONDITIONS),
            "forbidden_sets_read": [],
        },
        "frozen_protocol": {
            "path": str(protocol_file),
            "sha256": _sha256_file(protocol_file),
            "protocol": protocol.get("protocol"),
        },
        "inputs": {
            "predictions": {"path": str(Path(predictions_path).resolve()), "sha256": _sha256_file(Path(predictions_path).resolve())},
            "diagnostics": {"path": str(Path(diagnostics_path).resolve()), "sha256": _sha256_file(Path(diagnostics_path).resolve())},
            "labels": {"path": str(Path(labels_path).resolve()), "sha256": _sha256_file(Path(labels_path).resolve())},
        },
        "bootstrap": {
            "unit": "group_id",
            "replicates": int(bootstrap_replicates),
            "ci": "percentile_95",
            "seed_base": int(protocol["seed"]),
        },
        "paired_nmae": metrics,
        "pooled_projective": pooled_metric,
        "group_level_excess_definition": "p90 across the three pooled-projective group mean paired error deltas",
        "group_level_excess_p90": group_excess_p90,
        "failures": failure_count,
        "identity_mismatch_count": len(identity_mismatches),
        "identity_mismatch_examples": identity_mismatches[:20],
        "thresholds": dict(thresholds),
        "checks": checks,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json_line(result), encoding="utf-8", newline="\n")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--diagnostics", type=Path, required=True)
    prediction.add_argument("--summary", type=Path, required=True)
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--conditions", choices=("all", "clean"), default="all")
    prediction.add_argument("--selection-labels", type=Path)
    prediction.add_argument("--frozen-protocol", type=Path, default=DEFAULT_FROZEN_PROTOCOL)
    scoring = commands.add_parser("score-selection")
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--diagnostics", type=Path, required=True)
    scoring.add_argument("--labels", type=Path, default=DEFAULT_XM2_LABELS)
    scoring.add_argument("--frozen-protocol", type=Path, default=DEFAULT_FROZEN_PROTOCOL)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "predict":
        result = run_prediction(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_path=args.output,
            diagnostics_path=args.diagnostics,
            summary_path=args.summary,
            device_name=args.device,
            conditions=CONDITIONS if args.conditions == "all" else ("clean",),
            selection_labels_path=args.selection_labels,
            frozen_protocol_path=args.frozen_protocol,
        )
        message = {
            "status": "complete",
            "method": result["method"],
            "samples": result["samples"],
            "prediction_rows": result["prediction_rows"],
            "summary": str(Path(args.summary).resolve()),
        }
    else:
        result = score_selection(
            predictions_path=args.predictions,
            diagnostics_path=args.diagnostics,
            labels_path=args.labels,
            frozen_protocol_path=args.frozen_protocol,
            output_path=args.output,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        message = {
            "status": "complete",
            "decision": result["decision"],
            "external_evaluation_authorized": result["external_evaluation_authorized"],
            "output": str(Path(args.output).resolve()),
        }
    print(json.dumps(message, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIAGNOSTIC_PROTOCOL",
    "PREDICTION_PROTOCOL",
    "SELECTION_SUMMARY_PROTOCOL",
    "load_checkpoint_model",
    "model_batch",
    "resize_homography_pixel_centers",
    "run_prediction",
    "score_selection",
]
