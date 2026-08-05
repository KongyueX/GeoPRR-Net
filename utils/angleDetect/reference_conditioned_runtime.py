"""Audited production runtime for the paper's final pointer-meter backend.

The final method is a four-stage inference-only composition:

1. the frozen mask/geometry base calibrator;
2. the frozen probabilistic pivot/direction expert;
3. the reference-conditioned progress calibrator; and
4. the reference-conditioned safety router.

The VDN implementation and checkpoint are deliberately not used here.  VDN is
an experimental comparison only.  At runtime, the production meter detector
and reference-point detector provide the same meter box, confidence,
``start_angle``, ``range_angle`` and ``reference_branch`` fields consumed by
the paper feature extractors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch

from experiments.calibrated_progress_router import (
    FEATURE_NAMES as ROUTER_FEATURE_NAMES,
    extract_calibrated_router_features,
    feature_matrix as router_feature_matrix,
)
from experiments.pivot_direction_fallback import tensor_from_bbox
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.progress_calibrator import (
    FEATURE_NAMES as CALIBRATOR_FEATURE_NAMES,
    extract_progress_features,
    feature_matrix as calibrator_feature_matrix,
    reading_from_progress,
)
from experiments.quality_router import finite_float
from experiments.reference_conditioned_progress_calibrator import (
    REFERENCE_BRANCHES,
    REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL,
    normalize_reference_branch,
    predict_reference_conditioned_residual,
)
from experiments.reference_conditioned_router import (
    REFERENCE_CONDITIONED_ROUTER_PROTOCOL,
    deterministic_router_prediction,
)
try:
    from .residual_calibrator import (
        SELECTIVE_FEATURE_COLUMNS,
        make_calibrator_row,
        make_features,
    )
except ImportError:  # Direct execution with utils/angleDetect on sys.path.
    from residual_calibrator import (  # type: ignore[no-redef]
        SELECTIVE_FEATURE_COLUMNS,
        make_calibrator_row,
        make_features,
    )


PROJECT_DIR = Path(__file__).resolve().parents[2]
PRODUCTION_BUNDLE_PROTOCOL = "reference_conditioned_final_production_bundle_v1"
PRODUCTION_RUNTIME_PROTOCOL = "reference_conditioned_final_runtime_v1"
SOURCE_HASH_PROTOCOL = "utf8_source_newlines_lf_v1"
BACKEND_NAME = "reference_conditioned_final"

_REQUIRED_ARTIFACTS = (
    "base_calibrator",
    "probabilistic_direction",
    "probabilistic_direction_verification",
    "reference_conditioned_calibrator",
    "reference_conditioned_router",
)
_FROZEN_ARTIFACT_SHA256 = {
    "base_calibrator": (
        "7bf375c3ca0e511efc09a9032e735afa4cff5a3e8c04adaf3154d47cbab444b8"
    ),
    "probabilistic_direction": (
        "b585a5f6092af4c74fc65f6b8e999744c3d3bae5a665ea379670b066d73fe11f"
    ),
    "probabilistic_direction_verification": (
        "ff2e049925f717d69763790c47e2ec084b0155b084198813000369ae80980268"
    ),
    "reference_conditioned_calibrator": (
        "df698e4b360d1f5f942dc9311558c96811f006f43cbad7fe676924d41034682a"
    ),
    "reference_conditioned_router": (
        "5ad98bdf21e354d5ae4b0e6b2bfb293cf03f51c18943fb5496fa2d6bf54cb58f"
    ),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_VERIFIER_HASHES = {
    "formal_probabilistic_direction_run_verification_v1": (
        "b5fadf7008bb2a8a3b202886b53d2514c0dc41db0e455719f990a590bee785a6"
    ),
    "formal_probabilistic_direction_ablation_verification_v1": (
        "3e8cf11df760944c98ecaae4dcc4298f23c4b66f19a4ebe1b31346bc3f1fe542"
    ),
}


class ArtifactAuditError(RuntimeError):
    """Raised when a frozen production artifact cannot be authenticated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_source(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _expanded_path(value: Any, *, base: Path) -> Path:
    text = os.path.expanduser(os.path.expandvars(str(value or "").strip()))
    if not text:
        raise ArtifactAuditError("artifact path is empty")
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _validate_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ArtifactAuditError(f"{label} does not contain a valid SHA-256")
    return digest


def _load_bundle_manifest(
    manifest_path: str | os.PathLike[str],
) -> tuple[Path, dict[str, Any], dict[str, Path], dict[str, str]]:
    path = Path(
        os.path.expanduser(os.path.expandvars(str(manifest_path)))
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"paper artifact manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactAuditError(f"cannot read paper artifact manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactAuditError("paper artifact manifest must contain a JSON object")
    if manifest.get("protocol") != PRODUCTION_BUNDLE_PROTOCOL:
        raise ArtifactAuditError(
            "paper artifact manifest protocol mismatch: "
            f"{manifest.get('protocol')!r}"
        )
    if int(manifest.get("schema_version") or 0) != 1:
        raise ArtifactAuditError("unsupported paper artifact manifest schema")
    if manifest.get("source_hash_protocol") != SOURCE_HASH_PROTOCOL:
        raise ArtifactAuditError(
            "paper artifact manifest source hash protocol mismatch"
        )
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ArtifactAuditError("paper artifact manifest has no source hashes")
    for name, entry in sources.items():
        if not isinstance(entry, Mapping):
            raise ArtifactAuditError(
                f"paper artifact manifest source {name} is invalid"
            )
        source_path = _expanded_path(entry.get("path"), base=path.parent)
        if not source_path.is_file():
            raise FileNotFoundError(
                f"paper production source {name} is missing: {source_path}"
            )
        expected = _validate_sha256(entry.get("sha256"), label=f"source {name}")
        actual = _sha256_source(source_path)
        if actual != expected:
            raise ArtifactAuditError(
                f"paper production source {name} changed: "
                f"expected {expected}, got {actual}"
            )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ArtifactAuditError("paper artifact manifest has no artifacts object")

    paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {}
    for name in _REQUIRED_ARTIFACTS:
        entry = artifacts.get(name)
        if not isinstance(entry, Mapping):
            raise ArtifactAuditError(f"paper artifact manifest is missing {name}")
        artifact_path = _expanded_path(entry.get("path"), base=path.parent)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"paper artifact {name} is missing: {artifact_path}")
        expected = _validate_sha256(entry.get("sha256"), label=name)
        if expected != _FROZEN_ARTIFACT_SHA256[name]:
            raise ArtifactAuditError(
                f"paper artifact manifest is not the frozen publication bundle: {name}"
            )
        actual = _sha256_file(artifact_path)
        if actual != expected:
            raise ArtifactAuditError(
                f"paper artifact {name} SHA-256 mismatch: "
                f"expected {expected}, got {actual}"
            )
        paths[name] = artifact_path
        expected_hashes[name] = expected
    return path, manifest, paths, expected_hashes


def _validate_source_hashes(
    recorded: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    *,
    label: str,
    canonical_newlines: bool,
) -> None:
    for name, path in source_paths.items():
        expected = str(recorded.get(name) or "")
        actual = _sha256_source(path) if canonical_newlines else _sha256_file(path)
        if expected != actual:
            raise ArtifactAuditError(
                f"{label} source hash mismatch for {name}: "
                f"expected {expected or '<missing>'}, got {actual}"
            )


def _positive_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if model is None:
        return np.ones(matrix.shape[0], dtype=np.float64)
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        if len(classes) == 1:
            return np.full(
                matrix.shape[0],
                1.0 if classes[0] in (1, True, "1") else 0.0,
                dtype=np.float64,
            )
        if probabilities.ndim == 2 and probabilities.shape[1]:
            return probabilities[:, -1]
    return np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)


def _forest_prediction(
    model: Any,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    estimators = np.asarray(
        getattr(model, "estimators_", []),
        dtype=object,
    ).reshape(-1)
    if not len(estimators):
        return prediction, np.zeros_like(prediction)
    per_tree = np.asarray(
        [estimator.predict(matrix) for estimator in estimators],
        dtype=np.float64,
    )
    return prediction, np.std(per_tree, axis=0)


def _reading_payload(reading: Mapping[str, Any] | None) -> dict[str, Any]:
    source = reading if isinstance(reading, Mapping) else {}
    return {
        "status": bool(source.get("status")),
        "backend": source.get("backend"),
        "prediction": finite_float(source.get("resultNum")),
        "progress": finite_float(source.get("progress_ratio")),
        "pointer_angle": finite_float(source.get("pointer_angle")),
        "confidence": finite_float(source.get("confidence")),
        "message": source.get("message"),
    }


def _bbox_xyxy(value: Any) -> list[float] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape == (4, 2):
        result = [
            float(array[0, 0]),
            float(array[0, 1]),
            float(array[2, 0]),
            float(array[2, 1]),
        ]
    elif array.size == 4:
        result = [float(item) for item in array.reshape(-1)]
    else:
        return None
    if not np.isfinite(result).all() or result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def build_front_end_payload(
    training_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the prediction-only front-end payload used by paper features."""

    bbox = _bbox_xyxy(training_artifacts.get("meter_bbox"))
    start_angle = finite_float(training_artifacts.get("startAngle"))
    range_angle = finite_float(training_artifacts.get("disAngle"))
    confidence = finite_float(training_artifacts.get("meter_confidence"))
    branch = normalize_reference_branch(training_artifacts.get("branch"))
    valid = (
        bbox is not None
        and start_angle is not None
        and range_angle is not None
        and abs(range_angle) > 1e-8
    )
    return {
        "status": bool(valid),
        "meter_bbox": bbox,
        "meter_confidence": confidence,
        "start_angle": start_angle,
        "range_angle": range_angle,
        "reference_branch": branch,
        "reference_source": "production_frozen_front_end",
    }


def build_raw_payload(
    *,
    transformer_reading: Mapping[str, Any],
    geometry_reading: Mapping[str, Any],
    geometry_v2_reading: Mapping[str, Any],
    mean_fusion_reading: Mapping[str, Any],
    weighted_fusion_reading: Mapping[str, Any],
    training_artifacts: Mapping[str, Any],
    scale_start: float,
    scale_end: float,
) -> dict[str, Any]:
    """Construct the same raw row shape emitted by ``collect_predictions``."""

    feature_row = make_calibrator_row(
        geometry_reading,
        geometry_v2_reading,
        weighted_fusion_reading,
        training_artifacts,
        corrected_crop_bgr=training_artifacts.get("corrected_crop_bgr"),
        scale_start=scale_start,
        scale_end=scale_end,
    )
    return {
        "status": bool(weighted_fusion_reading.get("status")),
        "scale_start": float(scale_start),
        "scale_end": float(scale_end),
        "branch": training_artifacts.get("branch"),
        "methods": {
            "transformer": _reading_payload(transformer_reading),
            "geometry_v1": _reading_payload(geometry_reading),
            "geometry_v2": _reading_payload(geometry_v2_reading),
            "mean_fusion": _reading_payload(mean_fusion_reading),
            "weighted_fusion": _reading_payload(weighted_fusion_reading),
        },
        "features": feature_row,
    }


def apply_base_calibrator(
    raw_row: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen base model with ``selective_experiment`` semantics."""

    scale_start = finite_float(raw_row.get("scale_start"))
    scale_end = finite_float(raw_row.get("scale_end"))
    weighted = ((raw_row.get("methods") or {}).get("weighted_fusion") or {})
    base_prediction = (
        finite_float(weighted.get("prediction"))
        if weighted.get("status") is True
        else None
    )
    failed = {
        "scale_start": scale_start,
        "scale_end": scale_end,
        "predictions": {
            "weighted_fusion": base_prediction,
            "ours": None,
        },
        "gate_probability": None,
        "residual_normalized": None,
        "residual_std_normalized": None,
        "correction_applied": False,
    }
    if (
        base_prediction is None
        or scale_start is None
        or scale_end is None
        or abs(scale_end - scale_start) <= 1e-12
    ):
        return failed

    columns = list(package.get("feature_columns") or ())
    matrix = make_features(raw_row.get("features") or {}, columns).reshape(1, -1)
    residual, residual_std = _forest_prediction(package["model"], matrix)
    clip_value = abs(float(package.get("residual_clip", 0.08)))
    residual = np.clip(residual, -clip_value, clip_value)
    span = scale_end - scale_start
    low, high = min(scale_start, scale_end), max(scale_start, scale_end)
    corrected = float(np.clip(base_prediction + float(residual[0]) * span, low, high))
    gate_matrix = np.column_stack(
        (matrix, residual, np.abs(residual), residual_std)
    ).astype(np.float32)
    probability = float(_positive_probability(package.get("gate_model"), gate_matrix)[0])
    threshold = float(package.get("gate_apply_threshold", 0.5))
    max_std = package.get("max_residual_std")
    max_std_value = float("inf") if max_std is None else float(max_std)
    applied = bool(probability >= threshold and residual_std[0] <= max_std_value)
    prediction = corrected if applied else base_prediction
    return {
        "scale_start": scale_start,
        "scale_end": scale_end,
        "predictions": {
            "weighted_fusion": base_prediction,
            "residual_ungated": corrected,
            "ours": float(prediction),
        },
        "gate_probability": probability,
        "residual_normalized": float(residual[0]),
        "residual_std_normalized": float(residual_std[0]),
        "correction_applied": applied,
    }


def _prediction(row: Mapping[str, Any]) -> float | None:
    if row.get("status") is False:
        return None
    nested = row.get("vector")
    payload = nested if isinstance(nested, Mapping) else row
    return finite_float(payload.get("prediction"))


def _base_prediction(row: Mapping[str, Any]) -> float | None:
    nested = row.get("base")
    if isinstance(nested, Mapping):
        return finite_float(nested.get("prediction"))
    return finite_float((row.get("predictions") or {}).get("ours"))


def _route(
    base: float | None,
    calibrated: float | None,
    score: float,
    threshold: float,
) -> tuple[float | None, str]:
    if base is None:
        return (
            calibrated,
            (
                "reference_conditioned_hard_fallback"
                if calibrated is not None
                else "failure"
            ),
        )
    if calibrated is not None and math.isfinite(score) and score > threshold:
        return calibrated, "reference_conditioned_quality_switch"
    return base, "base"


def route_reference_conditioned_payloads(
    *,
    raw_row: Mapping[str, Any],
    base_row: Mapping[str, Any],
    vector_row: Mapping[str, Any],
    reference_row: Mapping[str, Any],
    calibrator: Mapping[str, Any],
    router: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the exact single-row calibrator/router calculation used in evaluation."""

    branch = normalize_reference_branch(reference_row)
    calibration_features = extract_progress_features(
        vector_row,
        reference_row=reference_row,
    )
    calibration_matrix = calibrator_feature_matrix([calibration_features])
    residual = predict_reference_conditioned_residual(
        calibrator,
        calibration_matrix,
        [branch],
    )
    raw_vector = _prediction(vector_row)
    vector_payload = (
        vector_row.get("vector")
        if isinstance(vector_row.get("vector"), Mapping)
        else vector_row
    )
    raw_progress = finite_float(vector_payload.get("progress"))
    corrected_progress = (
        float(
            np.clip(
                raw_progress + residual["applied_residual"][0],
                0.0,
                1.0,
            )
        )
        if raw_vector is not None and raw_progress is not None
        else None
    )
    calibrated = reading_from_progress(
        corrected_progress,
        vector_row.get("scale_start"),
        vector_row.get("scale_end"),
    )
    calibration_row = {
        "progress": corrected_progress,
        "raw_progress": raw_progress,
        "predicted_progress_residual": float(residual["model_residual"][0]),
        "applied_progress_residual": float(residual["applied_residual"][0]),
        "progress_ensemble_std": float(residual["ensemble_std"][0]),
    }
    base = _base_prediction(base_row)
    score = math.nan
    router_features: dict[str, float] | None = None
    if base is not None and calibrated is not None:
        router_features = extract_calibrated_router_features(
            raw_row=raw_row,
            base_row=base_row,
            vector_row=vector_row,
            reference_row=reference_row,
            calibration_row=calibration_row,
        )
        matrix = router_feature_matrix([router_features])
        score = float(
            deterministic_router_prediction(router["estimator"], matrix)[0]
        )
    threshold = float(router["threshold"])
    selected, route = _route(base, calibrated, score, threshold)
    selected_progress = (
        (
            (selected - float(vector_row["scale_start"]))
            / (float(vector_row["scale_end"]) - float(vector_row["scale_start"]))
        )
        if selected is not None
        and finite_float(vector_row.get("scale_start")) is not None
        and finite_float(vector_row.get("scale_end")) is not None
        and abs(
            float(vector_row["scale_end"]) - float(vector_row["scale_start"])
        )
        > 1e-12
        else None
    )
    selected_pointer_angle = (
        finite_float(vector_payload.get("pointer_angle"))
        if route != "base"
        else finite_float(
            (((raw_row.get("methods") or {}).get("weighted_fusion") or {}).get(
                "pointer_angle"
            ))
        )
    )
    status = selected is not None
    return {
        "status": status,
        "backend": BACKEND_NAME,
        "message": (
            f"{BACKEND_NAME} route={route}, result={selected:.6f}"
            if status
            else "base and probabilistic-vector branches both failed"
        ),
        "resultNum": float(selected) if selected is not None else None,
        "progress_ratio": (
            float(np.clip(selected_progress, 0.0, 1.0))
            if selected_progress is not None
            else None
        ),
        "endNum_float": (
            float(np.clip(selected_progress, 0.0, 1.0) * 100.0)
            if selected_progress is not None
            else None
        ),
        "endNum": (
            int(round(np.clip(selected_progress, 0.0, 1.0) * 100.0))
            if selected_progress is not None
            else None
        ),
        "pointer_angle": selected_pointer_angle,
        "route": route,
        "reference_branch": branch,
        "base_prediction": base,
        "raw_vector_prediction": raw_vector,
        "reference_conditioned_prediction": calibrated,
        "raw_progress": raw_progress,
        "corrected_progress": corrected_progress,
        "model_progress_residual": float(residual["model_residual"][0]),
        "applied_progress_residual": float(residual["applied_residual"][0]),
        "progress_ensemble_std": float(residual["ensemble_std"][0]),
        "correction_clip": float(residual["correction_clip"][0]),
        "deadband": float(residual["deadband"][0]),
        "calibration_applied": bool(residual["applied_residual"][0] != 0.0),
        "used_branch_model": bool(residual["used_branch_model"][0]),
        "router_score": finite_float(score),
        "router_threshold": threshold,
        "calibrator_feature_schema": list(CALIBRATOR_FEATURE_NAMES),
        "router_feature_schema": (
            list(ROUTER_FEATURE_NAMES) if router_features is not None else None
        ),
    }


def _image_angle_from_direction(direction_xy: Sequence[float]) -> float:
    dx, dy = map(float, direction_xy[:2])
    if not np.isfinite([dx, dy]).all() or math.hypot(dx, dy) <= 1e-12:
        raise ValueError("probabilistic direction is invalid")
    return (math.degrees(math.atan2(dx, -dy)) - 180.0) % 360.0


def _reading_from_pointer_angle(
    pointer_angle: float,
    *,
    start_angle: float,
    range_angle: float,
    scale_start: float,
    scale_end: float,
) -> tuple[float, float]:
    if not np.isfinite([pointer_angle, start_angle, range_angle]).all():
        raise ValueError("non-finite pointer/reference angle")
    if abs(range_angle) <= 1e-8:
        raise ValueError("dial range angle is zero")
    relative = (pointer_angle - start_angle) % 360.0
    progress = relative / range_angle
    if not 0.0 <= progress <= 1.0:
        distance_to_start = min(relative, 360.0 - relative)
        distance_to_end = abs(relative - range_angle)
        progress = 0.0 if distance_to_start <= distance_to_end else 1.0
    return scale_start + progress * (scale_end - scale_start), progress


class ReferenceConditionedFinalBackend:
    """Lazy-loaded, hash-authenticated production implementation."""

    def __init__(
        self,
        *,
        manifest_path: str | os.PathLike[str],
        device: str | torch.device,
        front_end_artifacts: Mapping[str, str | os.PathLike[str]],
    ) -> None:
        self._lock = threading.RLock()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"paper backend requested {self.device}, but CUDA is unavailable"
            )
        (
            self.manifest_path,
            self.manifest,
            paths,
            hashes,
        ) = _load_bundle_manifest(manifest_path)
        self.artifact_paths = paths
        self.artifact_hashes = hashes

        self.base_calibrator = joblib.load(paths["base_calibrator"])
        self.reference_calibrator = joblib.load(
            paths["reference_conditioned_calibrator"]
        )
        self.reference_router = joblib.load(paths["reference_conditioned_router"])
        verification = json.loads(
            paths["probabilistic_direction_verification"].read_text(
                encoding="utf-8"
            )
        )
        checkpoint = torch.load(
            paths["probabilistic_direction"],
            map_location="cpu",
            weights_only=False,
        )
        self._validate_base_calibrator(front_end_artifacts)
        self._validate_reference_calibrator()
        self._validate_reference_router()
        self._validate_direction(checkpoint, verification)

        signature = checkpoint["signature"]
        self.image_size = int(signature["image_size"])
        self.heatmap_size = int(signature["heatmap_size"])
        self.expansion = float(signature["expansion"])
        self.direction_model = build_probabilistic_pivot_direction_model(
            angle_bins=int(signature["angle_bins"]),
            imagenet_pretrained=False,
        )
        self.direction_model.load_state_dict(checkpoint["model_state"], strict=True)
        self.direction_model.to(self.device).eval()
        self.amp_enabled = self.device.type == "cuda"
        self.audit = {
            "protocol": PRODUCTION_RUNTIME_PROTOCOL,
            "bundle_protocol": self.manifest.get("protocol"),
            "manifest": str(self.manifest_path),
            "manifest_sha256": _sha256_file(self.manifest_path),
            "artifacts_sha256": dict(self.artifact_hashes),
            "direction_training_protocol": signature.get("protocol"),
            "reference_calibrator_protocol": self.reference_calibrator.get("protocol"),
            "reference_router_protocol": self.reference_router.get("protocol"),
            "source_hash_protocol": SOURCE_HASH_PROTOCOL,
            "vdn_loaded": False,
            "device": str(self.device),
        }

    def _validate_base_calibrator(
        self,
        front_end_artifacts: Mapping[str, str | os.PathLike[str]],
    ) -> None:
        package = self.base_calibrator
        if not isinstance(package, Mapping):
            raise ArtifactAuditError("base calibrator is not a mapping")
        if (
            package.get("residual_unit") != "normalized_range"
            or tuple(package.get("feature_columns") or ())
            != tuple(SELECTIVE_FEATURE_COLUMNS)
            or package.get("model") is None
            or package.get("gate_model") is None
        ):
            raise ArtifactAuditError("base calibrator schema/protocol audit failed")
        signature = (package.get("training") or {}).get(
            "prediction_cache_signature"
        )
        if not isinstance(signature, Mapping):
            raise ArtifactAuditError("base calibrator has no front-end signature")
        expected_policy = {
            "correction_mode": "off",
            "use_manifest_crop": False,
            "validate_mask_line": True,
            "include_transformer": True,
            "reading_backend": "compare",
        }
        for key, expected in expected_policy.items():
            if signature.get(key) != expected:
                raise ArtifactAuditError(
                    f"base calibrator front-end policy mismatch for {key}"
                )
        expected_weights = signature.get("weights_sha256")
        if not isinstance(expected_weights, Mapping):
            raise ArtifactAuditError("base calibrator has no weight hashes")
        for name in (
            "segmentation",
            "meter_detector",
            "meter_transformer",
            "keypoint_detector",
        ):
            value = front_end_artifacts.get(name)
            if value is None:
                raise ArtifactAuditError(
                    f"current front-end path for {name} was not provided"
                )
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"front-end artifact {name} is missing: {path}")
            actual = _sha256_file(path)
            if actual != expected_weights.get(name):
                raise ArtifactAuditError(
                    f"front-end artifact {name} differs from base training: "
                    f"expected {expected_weights.get(name)}, got {actual}"
                )
        recorded_sources = signature.get("source_sha256")
        if not isinstance(recorded_sources, Mapping):
            raise ArtifactAuditError("base calibrator has no source hashes")
        angle_dir = PROJECT_DIR / "utils" / "angleDetect"
        source_paths = {
            "geometry_baseline": angle_dir / "geometry_baseline.py",
            "residual_features": angle_dir / "residual_calibrator.py",
            "pointer_seg_inference": angle_dir / "pointerSeg" / "detectSeg.py",
            "pointer_seg_model": angle_dir / "pointerSeg" / "u2netp.py",
            "letterbox": angle_dir / "dataloader.py",
            "meter_detector": angle_dir / "yoloDetection" / "yoloDectect.py",
            "detector_geometry": angle_dir / "yoloDetection" / "pointGet.py",
            "meter_transformer_adapter": angle_dir / "detect.py",
            "meter_transformer": angle_dir / "vitTranforms" / "meterCilp.py",
            "meter_transformer_encoder": angle_dir / "vitTranforms" / "encoder.py",
            "meter_transformer_decoder": angle_dir / "vitTranforms" / "decoder.py",
            "meter_transformer_image_encoder": (
                angle_dir / "vitTranforms" / "imgencoder.py"
            ),
            "meter_transformer_text_encoder": (
                angle_dir / "vitTranforms" / "textencoder.py"
            ),
            "meter_transformer_tokenizer": (
                angle_dir / "vitTranforms" / "simple_tokenizer.py"
            ),
        }
        _validate_source_hashes(
            recorded_sources,
            source_paths,
            label="base calibrator",
            canonical_newlines=False,
        )

    def _validate_reference_calibrator(self) -> None:
        artifact = self.reference_calibrator
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("protocol") != REFERENCE_CONDITIONED_CALIBRATOR_PROTOCOL
            or artifact.get("train_only_certified") is not True
            or artifact.get("strict_nested_oof") is not True
            or tuple(artifact.get("feature_names") or ())
            != tuple(CALIBRATOR_FEATURE_NAMES)
            or tuple(artifact.get("reference_branches") or ())
            != tuple(REFERENCE_BRANCHES)
            or artifact.get("test_sets_used") != []
        ):
            raise ArtifactAuditError(
                "reference-conditioned calibrator schema/protocol audit failed"
            )
        _validate_source_hashes(
            artifact.get("source_sha256") or {},
            {
                "features": PROJECT_DIR / "experiments" / "progress_calibrator.py",
                "reference_policy": (
                    PROJECT_DIR
                    / "experiments"
                    / "reference_conditioned_progress_calibrator.py"
                ),
            },
            label="reference-conditioned calibrator",
            canonical_newlines=True,
        )

    def _validate_reference_router(self) -> None:
        artifact = self.reference_router
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("protocol") != REFERENCE_CONDITIONED_ROUTER_PROTOCOL
            or artifact.get("train_only_certified") is not True
            or artifact.get("nested_threshold_selection") is not True
            or tuple(artifact.get("feature_names") or ())
            != tuple(ROUTER_FEATURE_NAMES)
            or artifact.get("test_sets_used") != []
            or artifact.get("calibrator_sha256")
            != self.artifact_hashes["reference_conditioned_calibrator"]
        ):
            raise ArtifactAuditError(
                "reference-conditioned router schema/protocol audit failed"
            )
        _validate_source_hashes(
            artifact.get("source_sha256") or {},
            {
                "features": (
                    PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
                ),
                "quality_features": (
                    PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
                ),
                "policy": PROJECT_DIR / "experiments" / "quality_router.py",
                "router_protocol": (
                    PROJECT_DIR / "experiments" / "reference_conditioned_router.py"
                ),
            },
            label="reference-conditioned router",
            canonical_newlines=True,
        )

    def _validate_direction(
        self,
        checkpoint: Mapping[str, Any],
        verification: Mapping[str, Any],
    ) -> None:
        signature = checkpoint.get("signature") if isinstance(checkpoint, Mapping) else None
        if (
            not isinstance(signature, Mapping)
            or checkpoint.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
            or signature.get("protocol") != PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL
            or not isinstance(checkpoint.get("model_state"), Mapping)
        ):
            raise ArtifactAuditError(
                "probabilistic direction checkpoint schema/protocol audit failed"
            )
        model_source = (
            PROJECT_DIR / "experiments" / "probabilistic_pivot_direction.py"
        )
        if signature.get("model_source_sha256") != _sha256_source(model_source):
            raise ArtifactAuditError(
                "probabilistic direction model source changed after training"
            )
        if (
            verification.get("verified") is not True
            or verification.get("best_checkpoint_sha256")
            != self.artifact_hashes["probabilistic_direction"]
        ):
            raise ArtifactAuditError(
                "probabilistic direction formal verification does not match checkpoint"
            )
        protocol = str(verification.get("protocol") or "")
        expected_legacy = _LEGACY_VERIFIER_HASHES.get(protocol)
        if expected_legacy is not None:
            if verification.get("verifier_source_sha256") != expected_legacy:
                raise ArtifactAuditError(
                    "probabilistic direction legacy verifier hash mismatch"
                )
        elif not protocol.startswith("formal_probabilistic_direction_"):
            raise ArtifactAuditError(
                f"unsupported probabilistic direction verification {protocol!r}"
            )

    @torch.inference_mode()
    def predict_direction(
        self,
        image_bgr: np.ndarray,
        front_end: Mapping[str, Any],
        *,
        scale_start: float,
        scale_end: float,
    ) -> dict[str, Any]:
        if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return {
                "status": False,
                "error_code": "invalid_image",
                "message": "probabilistic direction received an invalid image",
            }
        if front_end.get("status") is not True:
            return {
                "status": False,
                "error_code": "front_end_failed",
                "message": "meter box/reference front end did not produce geometry",
            }
        bbox = _bbox_xyxy(front_end.get("meter_bbox"))
        if bbox is None:
            return {
                "status": False,
                "error_code": "invalid_meter_bbox",
                "message": "front end did not provide a valid meter bbox",
            }
        tensor = tensor_from_bbox(
            image_bgr,
            bbox,
            image_size=self.image_size,
            expansion=self.expansion,
        ).unsqueeze(0).to(self.device)
        with self._lock:
            with torch.amp.autocast(
                self.device.type,
                enabled=self.amp_enabled,
            ):
                outputs = self.direction_model(tensor)
            outputs = tuple(value.float() for value in outputs)
            prediction = decode_probabilistic_pivot_direction(*outputs)
        valid = bool(prediction.valid[0].detach().cpu().item())
        if not valid:
            return {
                "status": False,
                "error_code": "invalid_direction",
                "message": "probabilistic direction head returned an invalid vector",
                "front_end": dict(front_end),
            }
        direction = prediction.direction[0].detach().cpu().numpy().astype(np.float64)
        pointer_angle = _image_angle_from_direction(direction)
        try:
            reading, progress = _reading_from_pointer_angle(
                pointer_angle,
                start_angle=float(front_end["start_angle"]),
                range_angle=float(front_end["range_angle"]),
                scale_start=float(scale_start),
                scale_end=float(scale_end),
            )
        except (TypeError, ValueError) as exc:
            return {
                "status": False,
                "error_code": "reading_conversion_failed",
                "message": str(exc),
                "front_end": dict(front_end),
            }

        pivot_probabilities = torch.sigmoid(outputs[0][:, 0]).reshape(1, -1)
        spatial = pivot_probabilities / torch.clamp(
            pivot_probabilities.sum(dim=1, keepdim=True),
            min=1e-8,
        )
        pivot_entropy = -torch.sum(
            spatial * torch.log(torch.clamp(spatial, min=1e-12)),
            dim=1,
        ) / math.log(float(pivot_probabilities.shape[1]))
        top2 = torch.topk(pivot_probabilities, k=2, dim=1).values
        raw_norm = torch.linalg.vector_norm(outputs[1], dim=1)
        stride = float(self.image_size) / float(self.heatmap_size)
        pivot_heatmap = prediction.pivot_xy[0].detach().cpu().numpy()
        return {
            "status": True,
            "prediction": float(reading),
            "progress": float(progress),
            "pointer_angle": float(pointer_angle),
            "direction": direction.tolist(),
            "pivot_heatmap_xy": pivot_heatmap.tolist(),
            "pivot_input_xy": (pivot_heatmap * stride).tolist(),
            "pivot_peak": float(prediction.pivot_peak[0].detach().cpu().item()),
            "pivot_spatial_entropy": float(pivot_entropy[0].detach().cpu().item()),
            "pivot_top2_margin": float(
                (top2[0, 0] - top2[0, 1]).detach().cpu().item()
            ),
            "direction_raw_norm": float(raw_norm[0].detach().cpu().item()),
            "angle_std_degrees": float(
                prediction.angle_std_degrees[0].detach().cpu().item()
            ),
            "angle_log_variance": float(
                prediction.log_variance[0].detach().cpu().item()
            ),
            "angle_bin_entropy": float(
                prediction.angle_entropy[0].detach().cpu().item()
            ),
            "angle_bin_resultant_length": float(
                prediction.bin_resultant_length[0].detach().cpu().item()
            ),
            "direction_decoder": "fused",
            "front_end": dict(front_end),
            "meter_bbox": bbox,
            "start_angle": float(front_end["start_angle"]),
            "range_angle": float(front_end["range_angle"]),
            "reference_branch": str(front_end["reference_branch"]),
            "reference_source": "production_frozen_front_end",
            "scale_start": float(scale_start),
            "scale_end": float(scale_end),
        }

    def predict(
        self,
        *,
        image_bgr: np.ndarray,
        raw_row: Mapping[str, Any],
        front_end: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run all frozen paper back-end stages for one front-end result."""

        with self._lock:
            base_row = apply_base_calibrator(raw_row, self.base_calibrator)
            vector_row = self.predict_direction(
                image_bgr,
                front_end,
                scale_start=float(raw_row["scale_start"]),
                scale_end=float(raw_row["scale_end"]),
            )
            vector_row.setdefault("scale_start", float(raw_row["scale_start"]))
            vector_row.setdefault("scale_end", float(raw_row["scale_end"]))
            final = route_reference_conditioned_payloads(
                raw_row=raw_row,
                base_row=base_row,
                vector_row=vector_row,
                reference_row=front_end,
                calibrator=self.reference_calibrator,
                router=self.reference_router,
            )
        final["front_end"] = dict(front_end)
        final["uncertainty"] = {
            "angle_std_degrees": finite_float(
                vector_row.get("angle_std_degrees")
            ),
            "angle_log_variance": finite_float(
                vector_row.get("angle_log_variance")
            ),
            "angle_bin_entropy": finite_float(
                vector_row.get("angle_bin_entropy")
            ),
            "angle_bin_resultant_length": finite_float(
                vector_row.get("angle_bin_resultant_length")
            ),
            "pivot_peak": finite_float(vector_row.get("pivot_peak")),
            "pivot_spatial_entropy": finite_float(
                vector_row.get("pivot_spatial_entropy")
            ),
            "pivot_top2_margin": finite_float(
                vector_row.get("pivot_top2_margin")
            ),
            "progress_ensemble_std": final.get("progress_ensemble_std"),
            "router_score": final.get("router_score"),
            "router_threshold": final.get("router_threshold"),
            "coverage_claim": (
                "diagnostic_only; angular sigma is not a calibrated interval"
            ),
        }
        final["components"] = {
            "base": {
                "status": _base_prediction(base_row) is not None,
                "prediction": _base_prediction(base_row),
                "gate_probability": base_row.get("gate_probability"),
                "residual_normalized": base_row.get("residual_normalized"),
                "residual_std_normalized": base_row.get(
                    "residual_std_normalized"
                ),
                "correction_applied": base_row.get("correction_applied"),
            },
            "probabilistic_vector": {
                key: vector_row.get(key)
                for key in (
                    "status",
                    "prediction",
                    "progress",
                    "pointer_angle",
                    "direction",
                    "pivot_input_xy",
                    "pivot_peak",
                    "angle_std_degrees",
                    "angle_bin_entropy",
                    "angle_bin_resultant_length",
                    "error_code",
                    "message",
                )
            },
            "reference_conditioned_vector": {
                "status": final.get("reference_conditioned_prediction") is not None,
                "prediction": final.get("reference_conditioned_prediction"),
                "raw_progress": final.get("raw_progress"),
                "corrected_progress": final.get("corrected_progress"),
                "model_progress_residual": final.get(
                    "model_progress_residual"
                ),
                "applied_progress_residual": final.get(
                    "applied_progress_residual"
                ),
                "calibration_applied": final.get("calibration_applied"),
            },
        }
        final["artifact_audit"] = dict(self.audit)
        return final
