"""Evaluate VDN pointer direction with a SyncG annotation reference arc.

This is a deliberately secondary, non-deployable component diagnostic.  The
VDN network sees only the same canonical ROI pixels used by the primary paper
batch.  Offline conversion from direction to normalized progress uses the
ordered SyncG ``ScaleMark`` endpoints and ``Pointer.origin_kp`` pivot.  The
native ``vdn_official200_terminal_seed20`` automatic-reference result remains
unchanged in the primary eight-method evaluation.

For perspective conditions the three reference points are transformed by the
exact homography applied to the ROI before angles are derived.  Every sample is
emitted under every condition; invalid annotation or model output is an
explicit failure and is scored with the same error-one policy as the primary
batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from experiments import robustness_degradations
from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments import score_cagh_v5_plain_paper_batch as primary_scorer


PROTOCOL: Final[str] = "vdn_syncg_oracle_reference_component_v1"
SCORE_PROTOCOL: Final[str] = "vdn_syncg_oracle_reference_component_score_v1"
METHOD: Final[str] = "vdn_official200_terminal_seed20_oracle_reference_component"
NATIVE_AUTO_REFERENCE_METHOD: Final[str] = "vdn_official200_terminal_seed20"
EVALUATION_ROLE: Final[str] = "secondary_oracle_reference_component"
REFERENCE_SOURCE: Final[str] = (
    "SyncG Pointer.origin_kp + ordered ScaleMark.all_kp endpoints"
)
EXPECTED_SAMPLES: Final[int] = 1_625
EXPECTED_GROUPS: Final[int] = 73
EXPECTED_HOLDOUT_IDS_SHA256: Final[str] = (
    "7550cf807f6669723c8a58cf80c7e1046af4aaea8bacfebc781dcb70d899fca2"
)
EXPECTED_SOURCE_MANIFEST_SHA256: Final[str] = (
    "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
)
EXPECTED_ROI_MANIFEST_SHA256: Final[str] = (
    "b2226469463328891ad78b373fde7e94d52f9141a4a2bca6d11550f4963c734b"
)
CONDITIONS: Final[tuple[str, ...]] = primary_batch.CONDITIONS
ROBUSTNESS_SEED: Final[int] = primary_batch.ROBUSTNESS_SEED
_REFERENCE_BRANCH: Final[str] = (
    "offline_oracle:syncg_scalemark_endpoints_and_pointer_pivot"
)
_DIRECTION_ONLY_REFERENCE: Final[dict[str, Any]] = {
    "status": False,
    "start_angle": None,
    "range_angle": None,
    "reference_branch": "direction_only:no_reference_supplied_to_model_callback",
    "failure_code": "direction_only_no_reference",
}
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

OUTPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "protocol",
        "sample_id",
        "group_id",
        "method",
        "condition",
        "robustness_seed",
        "status",
        "normalized_progress",
        "failure_code",
        "roi_png_sha256",
        "roi_pixel_sha256",
        "condition_pixel_sha256",
        "evaluation_role",
        "primary_table_eligible",
        "deployable",
        "oracle_reference_used_for_offline_progress_conversion",
        "used_as_runtime_input",
        "direction_model_input",
        "reference_source",
        "oracle_reference_sha256",
        "predicted_pointer_angle_degrees",
        "native_auto_reference_result_retained_as",
    }
)


class OracleComponentError(ValueError):
    """The fixed cohort, annotation geometry, or output contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleComponentError(message)


def _canonical_json_line(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_line(value).removesuffix(b"\n")).hexdigest()


def _reference_sha256(value: Any) -> str:
    # The controlled-direction adapter hashes canonical JSON including its
    # record-terminating LF.
    return hashlib.sha256(_canonical_json_line(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, *, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OracleComponentError(f"{label} is not numeric") from exc
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _point(value: Any, *, label: str) -> tuple[float, float]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} is not a point",
    )
    _require(len(value) >= 2, f"{label} has fewer than two coordinates")
    return (
        _finite_number(value[0], label=f"{label}.x"),
        _finite_number(value[1], label=f"{label}.y"),
    )


@dataclass(frozen=True, slots=True)
class SourceReference:
    sample_id: str
    group_id: str
    bbox_xyxy: tuple[float, float, float, float]
    pivot_xy: tuple[float, float]
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DirectionRequest:
    source: primary_batch.ManifestRow
    group_id: str
    condition: str
    image_bgr: np.ndarray
    condition_pixel_sha256: str


@dataclass(frozen=True, slots=True)
class OracleRequest:
    direction: DirectionRequest
    reference: dict[str, Any]
    reference_sha256: str


class DirectionBatchPredictor(Protocol):
    native_input_size: int

    def predict_batch(
        self, requests: Sequence[DirectionRequest]
    ) -> Sequence[Mapping[str, Any]]:
        """Return one direction-only adapter record per pixel-only request."""


def _source_reference(row: Mapping[str, Any], *, index: int) -> SourceReference:
    sample_id = row.get("sample_id")
    group_id = row.get("group_id")
    _require(isinstance(sample_id, str) and bool(sample_id), f"source[{index}] sample_id")
    _require(isinstance(group_id, str) and bool(group_id), f"{sample_id}: group_id")
    metadata = row.get("metadata")
    _require(isinstance(metadata, Mapping), f"{sample_id}: metadata is absent")
    bbox = metadata.get("dial_bbox")
    _require(
        isinstance(bbox, Sequence)
        and not isinstance(bbox, (str, bytes))
        and len(bbox) >= 4,
        f"{sample_id}: dial_bbox is absent",
    )
    bbox_xyxy = tuple(
        _finite_number(bbox[position], label=f"{sample_id}.dial_bbox[{position}]")
        for position in range(4)
    )
    _require(
        bbox_xyxy[2] > bbox_xyxy[0] and bbox_xyxy[3] > bbox_xyxy[1],
        f"{sample_id}: dial_bbox is invalid",
    )

    pointers: list[Mapping[str, Any]] = []
    scale_marks: list[Mapping[str, Any]] = []
    keypoints = metadata.get("keypoints")
    _require(
        isinstance(keypoints, Sequence) and not isinstance(keypoints, (str, bytes)),
        f"{sample_id}: keypoints are absent",
    )
    for value in keypoints:
        if not isinstance(value, Mapping):
            continue
        kind = str(value.get("type") or "").casefold()
        if kind == "pointer":
            pointers.append(value)
        elif kind == "scalemark":
            scale_marks.append(value)
    _require(len(pointers) == 1, f"{sample_id}: expected exactly one Pointer")
    _require(len(scale_marks) == 1, f"{sample_id}: expected exactly one ScaleMark")
    marks = scale_marks[0].get("all_kp")
    _require(
        isinstance(marks, Sequence)
        and not isinstance(marks, (str, bytes))
        and len(marks) >= 2,
        f"{sample_id}: ordered ScaleMark endpoints are absent",
    )
    # Deliberately do not read Pointer.outside_kp: pointer direction remains
    # exclusively the VDN model's responsibility in this component cell.
    return SourceReference(
        sample_id=sample_id,
        group_id=group_id,
        bbox_xyxy=bbox_xyxy,
        pivot_xy=_point(pointers[0].get("origin_kp"), label=f"{sample_id}.pivot"),
        start_xy=_point(marks[0], label=f"{sample_id}.scale_start_point"),
        end_xy=_point(marks[-1], label=f"{sample_id}.scale_end_point"),
    )


def load_source_references(
    path: Path,
    *,
    sample_ids: Sequence[str],
    expected_manifest_sha256: str | None = EXPECTED_SOURCE_MANIFEST_SHA256,
) -> dict[str, SourceReference]:
    source = Path(path).resolve()
    _require(source.is_file(), f"SyncG manifest does not exist: {source}")
    if expected_manifest_sha256 is not None:
        _require(
            _sha256_file(source) == expected_manifest_sha256,
            "SyncG source manifest hash drift",
        )
    wanted = set(sample_ids)
    _require(len(wanted) == len(sample_ids), "ROI sample IDs are duplicated")
    result: dict[str, SourceReference] = {}
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OracleComponentError(
                    f"SyncG manifest line {index} is invalid JSON"
                ) from exc
            _require(isinstance(value, Mapping), f"source[{index}] is not an object")
            sample_id = value.get("sample_id")
            _require(
                isinstance(sample_id, str) and bool(sample_id),
                f"source[{index}] sample_id is invalid",
            )
            _require(sample_id not in seen, f"duplicate source sample_id: {sample_id}")
            seen.add(sample_id)
            if sample_id in wanted:
                result[sample_id] = _source_reference(value, index=index)
    missing = sorted(wanted - set(result))
    _require(not missing, f"ROI sample IDs absent from SyncG manifest: {missing[:5]}")
    return result


def _homography_point(point_xy: Sequence[float], matrix: np.ndarray) -> np.ndarray:
    point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0], dtype=np.float64)
    projected = np.asarray(matrix, dtype=np.float64) @ point
    _require(
        projected.shape == (3,)
        and np.isfinite(projected).all()
        and abs(float(projected[2])) > 1e-12,
        "reference homography produced an invalid point",
    )
    return projected[:2] / projected[2]


def _image_angle_from_direction(direction_xy: Sequence[float]) -> float:
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    _require(
        math.isfinite(dx) and math.isfinite(dy) and math.hypot(dx, dy) > 1e-12,
        "reference ray is degenerate",
    )
    # Exact VDN/production convention from experiments.vdn_baseline.
    return (math.degrees(math.atan2(dx, -dy)) - 180.0) % 360.0


def reference_packet_for_condition(
    source: SourceReference,
    *,
    roi_shape: Sequence[int],
    degradation_metadata: Mapping[str, Any],
    native_input_size: int,
) -> dict[str, Any]:
    """Map source GT reference geometry into the exact conditioned VDN frame."""

    _require(
        len(roi_shape) >= 2
        and type(native_input_size) is int
        and native_input_size >= 2,
        f"{source.sample_id}: invalid ROI/native shape",
    )
    height, width = int(roi_shape[0]), int(roi_shape[1])
    _require(height >= 2 and width >= 2, f"{source.sample_id}: invalid ROI shape")
    x1, y1, x2, y2 = source.bbox_xyxy
    # Match canonical_tight_roi_native's clipped crop origin.  Three fixed
    # holdout boxes extend one or two pixels beyond the 1080-pixel source
    # canvas, so the materialized ROI can be smaller than the nominal bbox.
    # Only the clipped left/top origin is needed to map source keypoints; the
    # pinned ROI pixel hash supplies the actual width/height.
    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    maximum_width = int(math.ceil(x2)) - left
    maximum_height = int(math.ceil(y2)) - top
    _require(
        maximum_width >= width and maximum_height >= height,
        f"{source.sample_id}: source bbox does not reproduce canonical ROI bounds",
    )
    points = [
        np.asarray(source.pivot_xy, dtype=np.float64) - [left, top],
        np.asarray(source.start_xy, dtype=np.float64) - [left, top],
        np.asarray(source.end_xy, dtype=np.float64) - [left, top],
    ]
    perspective = degradation_metadata.get("perspective")
    if perspective is not None:
        _require(isinstance(perspective, Mapping), "perspective metadata is invalid")
        matrix = np.asarray(perspective.get("homography"), dtype=np.float64)
        _require(
            matrix.shape == (3, 3) and np.isfinite(matrix).all(),
            "perspective homography is invalid",
        )
        points = [_homography_point(point, matrix) for point in points]

    # The terminal VDN directly resizes the complete native ROI to a square.
    # OpenCV's half-pixel offset cancels between each endpoint and the pivot;
    # the vector itself scales by output/input along each axis.
    scale = np.asarray(
        [float(native_input_size) / float(width), float(native_input_size) / float(height)],
        dtype=np.float64,
    )
    pivot, start, end = (point * scale for point in points)
    start_angle = _image_angle_from_direction(start - pivot)
    end_angle = _image_angle_from_direction(end - pivot)
    angle_range = (end_angle - start_angle) % 360.0
    _require(
        math.isfinite(angle_range) and 1e-8 < angle_range < 360.0,
        f"{source.sample_id}: invalid ordered oracle reference range",
    )
    return {
        "status": True,
        "start_angle": float(start_angle),
        "range_angle": float(angle_range),
        "reference_branch": _REFERENCE_BRANCH,
        "failure_code": None,
    }


class _VDNOracleDirectionPredictor:
    def __init__(self, *, device: str) -> None:
        # Import the strict terminal wrapper before its critical dependencies;
        # its own source-binding code intentionally records preloaded modules.
        from experiments import vdn_seed20_terminal_matched_baseline as vdn

        provider = vdn.build_seed20_terminal_matched_provider(
            device=device,
            amp_enabled=True,
        )
        from experiments.v5_unified_direction_adapters import LabelFreeInput
        from experiments.v5_unified_two_stage_retest import REFERENCE_MODE_CONTROLLED

        self._provider = provider
        self._adapter = provider.direction_adapter
        self._label_free_input = LabelFreeInput
        self._reference_mode = REFERENCE_MODE_CONTROLLED
        self.native_input_size = int(self._adapter.image_size)

    def predict_batch(
        self, requests: Sequence[DirectionRequest]
    ) -> Sequence[Mapping[str, Any]]:
        items = [
            self._label_free_input(
                sample_id=f"oracle:{request.source.sample_id}:{request.condition}",
                group_id=request.group_id,
                image_path=request.source.roi_path,
                image_sha256=request.condition_pixel_sha256,
                frame_sha256=request.condition_pixel_sha256,
                reference_input=None,
                reference_input_sha256=None,
            )
            for request in requests
        ]
        return self._adapter.predict(
            items,
            [np.ascontiguousarray(request.image_bgr).copy() for request in requests],
            [dict(_DIRECTION_ONLY_REFERENCE) for _request in requests],
            reference_mode=self._reference_mode,
            reference_detector_sha256=None,
            amp_enabled=True,
        )


def build_direction_predictor(*, device: str = "cuda:0") -> DirectionBatchPredictor:
    _require(device == "cuda:0", "oracle VDN component requires cuda:0")
    primary_batch._configure_cuda_runtime()
    return _VDNOracleDirectionPredictor(device=device)


def _failure_code(value: Any) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9_.:-]+",
        "_",
        str(value or "oracle_component_failure"),
    ).strip("_")
    return normalized[:160] or "oracle_component_failure"


def _progress_from_reference(pointer_angle: float, reference: Mapping[str, Any]) -> float:
    start = _finite_number(reference.get("start_angle"), label="oracle start angle")
    angle_range = _finite_number(reference.get("range_angle"), label="oracle angle range")
    _require(0.0 < angle_range < 360.0, "oracle angle range is invalid")
    relative = (float(pointer_angle) - start) % 360.0
    progress = relative / angle_range
    if not 0.0 <= progress <= 1.0:
        distance_to_start = min(relative, 360.0 - relative)
        distance_to_end = abs(relative - angle_range)
        progress = 0.0 if distance_to_start <= distance_to_end else 1.0
    return primary_batch._finite_progress(progress)


def _record_values(
    record: Mapping[str, Any], *, oracle_reference: Mapping[str, Any]
) -> tuple[float | None, str | None, float | None]:
    telemetry = record.get("telemetry")
    pointer_angle: float | None = None
    if isinstance(telemetry, Mapping) and telemetry.get("pointer_angle") is not None:
        try:
            candidate = float(telemetry["pointer_angle"])
        except (TypeError, ValueError):
            candidate = math.nan
        if math.isfinite(candidate):
            pointer_angle = candidate
    if record.get("reference_input_sha256") != _reference_sha256(
        _DIRECTION_ONLY_REFERENCE
    ):
        return None, "direction_only_adapter_binding_mismatch", pointer_angle
    if pointer_angle is None:
        return None, _failure_code(record.get("failure_code") or "invalid_direction"), None
    try:
        progress = _progress_from_reference(pointer_angle, oracle_reference)
    except (OracleComponentError, primary_batch.ModelPredictionFailure) as exc:
        return None, _failure_code(type(exc).__name__), pointer_angle
    return progress, None, pointer_angle


def _result_row(
    *,
    source: primary_batch.ManifestRow,
    group_id: str,
    condition: str,
    condition_pixel_sha256: str,
    reference_sha256: str | None,
    progress: float | None,
    failure_code: str | None,
    pointer_angle: float | None,
) -> dict[str, Any]:
    passed = progress is not None and failure_code is None
    row = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "sample_id": source.sample_id,
        "group_id": group_id,
        "method": METHOD,
        "condition": condition,
        "robustness_seed": ROBUSTNESS_SEED,
        "status": "pass" if passed else "fail",
        "normalized_progress": progress if passed else None,
        "failure_code": None if passed else _failure_code(failure_code),
        "roi_png_sha256": source.roi_png_sha256,
        "roi_pixel_sha256": source.roi_pixel_sha256,
        "condition_pixel_sha256": condition_pixel_sha256,
        "evaluation_role": EVALUATION_ROLE,
        "primary_table_eligible": False,
        "deployable": False,
        "oracle_reference_used_for_offline_progress_conversion": True,
        # Required paper disclosure: GT ScaleMark/pivot is not an input to the
        # deployed/final runtime system.  It is used only by this offline cell.
        "used_as_runtime_input": False,
        "direction_model_input": "canonical_roi_pixels_only",
        "reference_source": REFERENCE_SOURCE,
        "oracle_reference_sha256": reference_sha256,
        "predicted_pointer_angle_degrees": pointer_angle,
        "native_auto_reference_result_retained_as": NATIVE_AUTO_REFERENCE_METHOD,
    }
    _require(set(row) == set(OUTPUT_KEYS), "oracle output schema drift")
    return row


def run_component(
    *,
    roi_manifest_path: Path,
    syncg_manifest_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_groups: int = EXPECTED_GROUPS,
    expected_ids_sha256: str | None = EXPECTED_HOLDOUT_IDS_SHA256,
    expected_roi_manifest_sha256: str | None = EXPECTED_ROI_MANIFEST_SHA256,
    expected_source_manifest_sha256: str | None = EXPECTED_SOURCE_MANIFEST_SHA256,
    predictor_factory: Callable[..., DirectionBatchPredictor] = build_direction_predictor,
) -> int:
    """Run the complete oracle component Cartesian without sample filtering."""

    _require(type(batch_size) is int and batch_size >= 1, "batch_size must be positive")
    _require(
        NATIVE_AUTO_REFERENCE_METHOD in primary_batch.METHODS
        and METHOD not in primary_batch.METHODS,
        "oracle component must remain separate from the primary automatic roster",
    )
    _require(
        robustness_degradations.degradation_names(include_clean=True) == CONDITIONS,
        "robustness condition roster drift",
    )
    roi_manifest = Path(roi_manifest_path).resolve()
    syncg_manifest = Path(syncg_manifest_path).resolve()
    output = Path(output_path).resolve()
    _require(output not in {roi_manifest, syncg_manifest}, "output cannot overwrite an input")
    if expected_roi_manifest_sha256 is not None:
        _require(
            _sha256_file(roi_manifest) == expected_roi_manifest_sha256,
            "canonical ROI manifest hash drift",
        )
    rows = primary_batch.load_manifest(roi_manifest)
    _require(len(rows) == expected_samples, f"ROI manifest has {len(rows)} samples")
    sample_ids = tuple(row.sample_id for row in rows)
    if expected_ids_sha256 is not None:
        _require(
            _canonical_sha256(sorted(sample_ids)) == expected_ids_sha256,
            "holdout sample roster drift",
        )
    references = load_source_references(
        syncg_manifest,
        sample_ids=sample_ids,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    _require(
        len({reference.group_id for reference in references.values()}) == expected_groups,
        "holdout group count drift",
    )
    predictor = predictor_factory(device=device)
    _require(
        type(predictor.native_input_size) is int and predictor.native_input_size >= 2,
        "direction predictor native input size is invalid",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pending: list[OracleRequest] = []
    count = 0

    with output.open("w", encoding="utf-8", newline="\n") as stream:
        def write_row(value: Mapping[str, Any]) -> None:
            nonlocal count
            stream.write(_canonical_json_line(value).decode("utf-8"))
            count += 1

        def flush_pending() -> None:
            if not pending:
                return
            requests = list(pending)
            pending.clear()
            try:
                records = list(
                    predictor.predict_batch([request.direction for request in requests])
                )
                _require(len(records) == len(requests), "VDN batch output count drift")
            except Exception:
                records = []
                for request in requests:
                    try:
                        one = list(predictor.predict_batch([request.direction]))
                        _require(len(one) == 1, "VDN single output count drift")
                        records.append(one[0])
                    except Exception as exc:
                        records.append(
                            {
                                "status": False,
                                "prediction_progress": None,
                                "failure_code": f"model_exception:{type(exc).__name__}",
                                "reference_input_sha256": _reference_sha256(
                                    _DIRECTION_ONLY_REFERENCE
                                ),
                                "telemetry": {},
                            }
                        )
            for request, raw_record in zip(requests, records):
                if not isinstance(raw_record, Mapping):
                    raw_record = {
                        "status": False,
                        "prediction_progress": None,
                        "failure_code": "non_mapping_model_record",
                        "reference_input_sha256": _reference_sha256(
                            _DIRECTION_ONLY_REFERENCE
                        ),
                        "telemetry": {},
                    }
                progress, failure, pointer_angle = _record_values(
                    raw_record,
                    oracle_reference=request.reference,
                )
                write_row(
                    _result_row(
                        source=request.direction.source,
                        group_id=request.direction.group_id,
                        condition=request.direction.condition,
                        condition_pixel_sha256=request.direction.condition_pixel_sha256,
                        reference_sha256=request.reference_sha256,
                        progress=progress,
                        failure_code=failure,
                        pointer_angle=pointer_angle,
                    )
                )

        for source in rows:
            _payload, clean = primary_batch.load_canonical_roi(source)
            source_reference = references[source.sample_id]
            for condition in CONDITIONS:
                conditioned, metadata = robustness_degradations.apply_degradation(
                    clean,
                    condition,
                    sample_id=source.sample_id,
                    seed=ROBUSTNESS_SEED,
                )
                conditioned = np.ascontiguousarray(conditioned)
                condition_hash = primary_batch.canonical_roi_pixel_sha256(conditioned)
                try:
                    reference = reference_packet_for_condition(
                        source_reference,
                        roi_shape=conditioned.shape,
                        degradation_metadata=metadata,
                        native_input_size=predictor.native_input_size,
                    )
                except Exception as exc:
                    flush_pending()
                    write_row(
                        _result_row(
                            source=source,
                            group_id=source_reference.group_id,
                            condition=condition,
                            condition_pixel_sha256=condition_hash,
                            reference_sha256=None,
                            progress=None,
                            failure_code=f"oracle_reference_invalid:{type(exc).__name__}",
                            pointer_angle=None,
                        )
                    )
                    continue
                pending.append(
                    OracleRequest(
                        direction=DirectionRequest(
                            source=source,
                            group_id=source_reference.group_id,
                            condition=condition,
                            image_bgr=conditioned,
                            condition_pixel_sha256=condition_hash,
                        ),
                        reference=reference,
                        reference_sha256=_reference_sha256(reference),
                    )
                )
                if len(pending) >= batch_size:
                    flush_pending()
        flush_pending()

    expected_rows = expected_samples * len(CONDITIONS)
    _require(count == expected_rows, f"oracle output has {count} rows, expected {expected_rows}")
    return count


def score_component(
    *,
    predictions_path: Path,
    syncg_manifest_path: Path,
    validation_ids_path: Path,
    expected_samples: int = EXPECTED_SAMPLES,
    expected_groups: int = EXPECTED_GROUPS,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = ROBUSTNESS_SEED,
) -> dict[str, Any]:
    """Score the isolated oracle cell with the primary failure/metric policy."""

    result = primary_scorer.score(
        predictions_path=Path(predictions_path),
        manifest_path=Path(syncg_manifest_path),
        validation_ids_path=Path(validation_ids_path),
        methods=(METHOD,),
        conditions=CONDITIONS,
        full_seed_methods=(),
        expected_samples=expected_samples,
        expected_groups=expected_groups,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    result["protocol"] = SCORE_PROTOCOL
    result["evaluation_role"] = EVALUATION_ROLE
    result["primary_table_eligible"] = False
    result["deployable"] = False
    result["oracle_disclosure"] = {
        "reference_source": REFERENCE_SOURCE,
        "oracle_reference_used_for_offline_progress_conversion": True,
        "used_as_runtime_input": False,
        "pointer_direction_source": "VDN terminal epoch-200 checkpoint",
        "native_automatic_reference_result_retained_as": NATIVE_AUTO_REFERENCE_METHOD,
        "interpretation": (
            "direction-component upper bound and reference-error attribution only"
        ),
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="run the 1625x6 oracle component")
    predict.add_argument("--roi-manifest", type=Path, required=True)
    predict.add_argument("--syncg-manifest", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    predict.add_argument("--batch-size", type=int, default=32)
    score = subparsers.add_parser("score", help="score a completed oracle prediction file")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--syncg-manifest", type=Path, required=True)
    score.add_argument("--validation-ids", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--bootstrap-replicates", type=int, default=2_000)
    score.add_argument("--bootstrap-seed", type=int, default=ROBUSTNESS_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "predict":
        count = run_component(
            roi_manifest_path=args.roi_manifest,
            syncg_manifest_path=args.syncg_manifest,
            output_path=args.output,
            device=args.device,
            batch_size=args.batch_size,
        )
        value = {
            "status": "complete",
            "protocol": PROTOCOL,
            "prediction_rows": count,
            "output": str(Path(args.output).resolve()),
            "output_sha256": _sha256_file(Path(args.output).resolve()),
        }
    else:
        output = Path(args.output).resolve()
        _require(
            output
            not in {
                Path(args.predictions).resolve(),
                Path(args.syncg_manifest).resolve(),
                Path(args.validation_ids).resolve(),
            },
            "score output cannot overwrite an input",
        )
        result = score_component(
            predictions_path=args.predictions,
            syncg_manifest_path=args.syncg_manifest,
            validation_ids_path=args.validation_ids,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        value = {
            "status": "complete",
            "protocol": SCORE_PROTOCOL,
            "output": str(output),
            "output_sha256": _sha256_file(output),
        }
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONDITIONS",
    "EVALUATION_ROLE",
    "METHOD",
    "NATIVE_AUTO_REFERENCE_METHOD",
    "DirectionRequest",
    "OracleComponentError",
    "SourceReference",
    "load_source_references",
    "reference_packet_for_condition",
    "run_component",
    "score_component",
]
