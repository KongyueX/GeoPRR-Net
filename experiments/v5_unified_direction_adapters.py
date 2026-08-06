"""Real label-free PEPD/VDN adapters for the unified V5 retest.

Strict ``+auto-ref`` progress components receive only an already-canonical meter ROI.  The
meter detector is never invoked and the full supplied ROI is directly resized
for each direction checkpoint.  A frozen production start/end detector runs
inside that same ROI and is bound by SHA256.  No manual or ground-truth scale
geometry is accepted.  A supplied common reference packet remains available
only through an explicit secondary controlled-analysis mode.  These adapters
do not predict numeric scale labels, so their outputs are component diagnostics
and are not eligible for the complete automatic-reading primary metric.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.probabilistic_pivot_direction import (
    PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
)
from experiments.evaluate_vdn_baseline import (
    _fallback_reference,
    _load_target_detector_class,
)
from experiments.v5_unified_two_stage_retest import (
    AUTO_REFERENCE_INTERFACE,
    CANONICAL_TIGHT_ROI_CONTRACT,
    COMMON_REFERENCE_INTERFACE,
    EVIDENCE_ROLE_DIAGNOSTIC,
    EVIDENCE_ROLE_COMPONENT,
    HASH_PROTOCOL,
    INFERENCE_PROTOCOL,
    OUTPUT_MODE_PROGRESS_ONLY,
    _atomic_new,
    _is_sha256,
    _jsonl_bytes,
    _reference_packet,
    REFERENCE_MODE_AUTO,
    REFERENCE_MODE_CONTROLLED,
    assert_label_free,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
    strict_json_load,
    strict_jsonl_load,
)
from experiments.vdn_baseline import (
    build_vdn_model,
    image_angle_from_direction,
    normalized_bgr_tensor,
    predict_directions,
    reading_from_pointer_angle,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_CHECKPOINT_PROTOCOL,
    OFFICIAL200_PROTOCOL,
    OFFICIAL200_VERIFICATION_PROTOCOL,
)


PROTOCOL: Final[str] = "v5_unified_label_free_direction_adapters_v3"
AUTO_METHOD_NAMES: Final[tuple[str, str]] = (
    "PEPD+auto-ref",
    "VDN+auto-ref",
)
CONTROLLED_METHOD_NAMES: Final[tuple[str, str]] = (
    "PEPD+shared-ref(controlled)",
    "VDN+shared-ref(controlled)",
)
ROI_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    CANONICAL_TIGHT_ROI_CONTRACT
)
AUTO_REFERENCE_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    AUTO_REFERENCE_INTERFACE
)
CONTROLLED_REFERENCE_CONTRACT_SHA256: Final[str] = canonical_json_sha256(
    COMMON_REFERENCE_INTERFACE
)
BASE_INPUT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "sample_id",
        "group_id",
        "image_path",
        "image_sha256",
        "canonical_roi_sha256",
        "frame_sha256",
        "roi_contract_sha256",
        "reference_contract_sha256",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def prediction_progress_from_reference(
    pointer_angle: float,
    *,
    start_angle: float,
    range_angle: float,
) -> float:
    """Use the production angle convention without reading physical scales."""

    # Dummy 0..1 scale makes the returned reading numerically identical to
    # normalized progress while reusing the frozen production endpoint policy.
    progress, check = reading_from_pointer_angle(
        float(pointer_angle),
        start_angle=float(start_angle),
        range_angle=float(range_angle),
        scale_start=0.0,
        scale_end=1.0,
    )
    _require(
        math.isclose(progress, check, rel_tol=0.0, abs_tol=1e-12),
        "normalized progress conversion drift",
    )
    return float(check)


def direct_resize_whole_roi(image_bgr: np.ndarray, *, size: int) -> np.ndarray:
    """Directly resize all supplied pixels; no crop, padding or letterbox."""

    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] < 2
        or image_bgr.shape[1] < 2
    ):
        raise ValueError(f"invalid canonical ROI shape {image_bgr.shape}")
    if int(size) <= 0:
        raise ValueError("native input size must be positive")
    interpolation = (
        cv2.INTER_AREA
        if max(image_bgr.shape[:2]) > int(size)
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        image_bgr,
        (int(size), int(size)),
        interpolation=interpolation,
    )


def _verification_binding(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    verification_path: Path,
    expected_verification_protocols: frozenset[str],
) -> tuple[str, dict[str, Any]]:
    checkpoint = checkpoint_path.resolve(strict=True)
    verification_file = verification_path.resolve(strict=True)
    expected = str(expected_checkpoint_sha256).lower()
    _require(_is_sha256(expected), "expected checkpoint SHA256 is invalid")
    actual = sha256_file(checkpoint)
    _require(actual == expected, f"checkpoint hash mismatch: {checkpoint}")
    verification = strict_json_load(verification_file)
    _require(isinstance(verification, dict), "verification is not an object")
    _require(
        verification.get("protocol") in expected_verification_protocols,
        f"unsupported verification protocol: {verification.get('protocol')!r}",
    )
    _require(verification.get("verified") is True, "checkpoint is not verified")
    _require(
        str(verification.get("best_checkpoint_sha256") or "").lower() == actual,
        "verification belongs to another checkpoint",
    )
    return actual, verification


@dataclass(frozen=True)
class LabelFreeInput:
    sample_id: str
    group_id: str
    image_path: Path
    image_sha256: str
    frame_sha256: str
    reference_input: dict[str, Any] | None
    reference_input_sha256: str | None


def validate_label_free_inputs(
    rows: list[dict[str, Any]],
    *,
    reference_mode: str = REFERENCE_MODE_AUTO,
) -> list[LabelFreeInput]:
    """Reject label-bearing or contract-drifting rows before any I/O."""

    _require(
        reference_mode in (REFERENCE_MODE_AUTO, REFERENCE_MODE_CONTROLLED),
        "unsupported reference mode",
    )
    assert_label_free(rows, location="direction_adapter_input")
    _require(bool(rows), "direction adapter input is empty")
    seen: set[str] = set()
    result: list[LabelFreeInput] = []
    for index, row in enumerate(rows, 1):
        allowed_keys = set(BASE_INPUT_KEYS)
        # These two keys are admitted to schema parsing in both modes so auto
        # mode can reject them with an explicit leakage error below.
        allowed_keys.update({"reference_input", "reference_input_sha256"})
        unexpected = set(row) - allowed_keys
        _require(
            not unexpected,
            f"row {index}: unexpected adapter input keys {sorted(unexpected)}",
        )
        sample_id = str(row.get("sample_id") or "")
        group_id = str(row.get("group_id") or "")
        _require(bool(sample_id), f"row {index}: missing sample_id")
        _require(sample_id not in seen, f"duplicate sample_id {sample_id}")
        _require(bool(group_id), f"{sample_id}: missing group_id")
        seen.add(sample_id)
        image_sha256 = str(row.get("image_sha256") or "").lower()
        canonical_roi_sha256 = str(
            row.get("canonical_roi_sha256") or ""
        ).lower()
        frame_sha256 = str(row.get("frame_sha256") or image_sha256).lower()
        _require(_is_sha256(image_sha256), f"{sample_id}: invalid image SHA256")
        _require(_is_sha256(frame_sha256), f"{sample_id}: invalid frame SHA256")
        _require(
            canonical_roi_sha256 == image_sha256,
            f"{sample_id}: supplied file is not the canonical whole ROI",
        )
        _require(
            str(row.get("roi_contract_sha256") or "")
            == ROI_CONTRACT_SHA256,
            f"{sample_id}: ROI contract hash drift",
        )
        expected_reference_contract = (
            AUTO_REFERENCE_CONTRACT_SHA256
            if reference_mode == REFERENCE_MODE_AUTO
            else CONTROLLED_REFERENCE_CONTRACT_SHA256
        )
        _require(
            str(row.get("reference_contract_sha256") or "")
            == expected_reference_contract,
            f"{sample_id}: reference contract hash drift",
        )
        reference: dict[str, Any] | None = None
        reference_sha256: str | None = None
        if reference_mode == REFERENCE_MODE_AUTO:
            forbidden_external_reference_keys = {
                "reference_input",
                "reference_input_sha256",
                "start_angle",
                "range_angle",
                "manual_reference",
            }
            present = forbidden_external_reference_keys.intersection(row)
            _require(
                not present,
                f"{sample_id}: primary auto-reference input contains external "
                f"reference fields {sorted(present)}",
            )
        else:
            reference, reference_sha256 = _reference_packet(
                row.get("reference_input"),
                sample_id=sample_id,
            )
            _require(
                str(row.get("reference_input_sha256") or "").lower()
                == reference_sha256,
                f"{sample_id}: reference packet hash drift",
            )
        image_path = Path(str(row.get("image_path") or ""))
        _require(str(image_path) not in ("", "."), f"{sample_id}: image_path absent")
        result.append(
            LabelFreeInput(
                sample_id=sample_id,
                group_id=group_id,
                image_path=image_path.resolve(),
                image_sha256=image_sha256,
                frame_sha256=frame_sha256,
                reference_input=reference,
                reference_input_sha256=reference_sha256,
            )
        )
    return result


def _decode_bound_roi(item: LabelFreeInput) -> tuple[np.ndarray | None, str | None]:
    try:
        payload = item.image_path.read_bytes()
    except OSError:
        return None, "image_read_failed"
    actual = __import__("hashlib").sha256(payload).hexdigest()
    if actual != item.image_sha256:
        raise RuntimeError(f"{item.sample_id}: canonical ROI content hash drift")
    image = cv2.imdecode(
        np.frombuffer(payload, dtype=np.uint8),
        cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
    )
    if image is None:
        return None, "image_decode_failed"
    return image, None


def _automatic_reference_packet(
    point_detector: Any,
    image_bgr: np.ndarray,
) -> dict[str, Any]:
    """Run only the frozen start/end detector inside the whole meter ROI."""

    try:
        start_angle, range_angle, branch = _fallback_reference(
            point_detector,
            image_bgr,
        )
        branch = str(branch)
        if branch != "start_and_end":
            packet = {
                "status": False,
                "start_angle": None,
                "range_angle": None,
                "reference_branch": f"production_point_detector:{branch}",
                "failure_code": f"automatic_reference_incomplete:{branch}",
            }
            normalized, _ = _reference_packet(
                packet, sample_id="auto_reference"
            )
            return normalized
        start = _finite(start_angle)
        arc = _finite(range_angle)
        if start is None or arc is None or abs(arc) <= 1e-12:
            raise ValueError("automatic reference returned invalid angles")
        packet = {
            "status": True,
            "start_angle": start,
            "range_angle": arc,
            "reference_branch": f"production_point_detector:{branch}",
            "failure_code": None,
        }
    except Exception as error:  # Per-image failure must not abort the cohort.
        packet = {
            "status": False,
            "start_angle": None,
            "range_angle": None,
            "reference_branch": "production_point_detector:failed",
            "failure_code": f"auto_reference_exception:{type(error).__name__}",
        }
    # Reuse the strict packet normalizer so internal output cannot grow hidden
    # label/manual geometry fields.
    normalized, _ = _reference_packet(packet, sample_id="auto_reference")
    return normalized


def _safe_telemetry_value(value: Any) -> Any:
    # bool is an int subclass; preserve it before numeric coercion so protocol
    # attestations remain JSON booleans rather than 0/1 telemetry.
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return [_safe_telemetry_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_safe_telemetry_value(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    raise TypeError(f"unsupported telemetry value {type(value).__name__}")


def _method_record(
    *,
    item: LabelFreeInput,
    checkpoint_sha256: str,
    reference_mode: str,
    reference_packet: dict[str, Any],
    reference_packet_sha256: str,
    reference_detector_sha256: str | None,
    status: bool,
    prediction_progress: float | None,
    failure_code: str | None,
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    if status:
        _require(
            _finite(prediction_progress) is not None
            and 0.0 <= float(prediction_progress) <= 1.0,
            f"{item.sample_id}: successful adapter result is nonfinite",
        )
        failure_code = None
    else:
        _require(prediction_progress is None, "failed adapter result has progress")
        _require(bool(failure_code), "failed adapter result has no failure code")
    safe_telemetry = {
        str(key): _safe_telemetry_value(value) for key, value in telemetry.items()
    }
    _require(
        reference_mode in (REFERENCE_MODE_AUTO, REFERENCE_MODE_CONTROLLED),
        "unsupported reference mode",
    )
    reference_contract_sha256 = (
        AUTO_REFERENCE_CONTRACT_SHA256
        if reference_mode == REFERENCE_MODE_AUTO
        else CONTROLLED_REFERENCE_CONTRACT_SHA256
    )
    evidence_role = (
        EVIDENCE_ROLE_COMPONENT
        if reference_mode == REFERENCE_MODE_AUTO
        else EVIDENCE_ROLE_DIAGNOSTIC
    )
    record: dict[str, Any] = {
        "status": bool(status),
        "prediction_progress": (
            float(prediction_progress) if prediction_progress is not None else None
        ),
        "failure_code": failure_code,
        "checkpoint_sha256": checkpoint_sha256,
        "roi_contract_sha256": ROI_CONTRACT_SHA256,
        "reference_mode": reference_mode,
        "reference_contract_sha256": reference_contract_sha256,
        "evidence_role": evidence_role,
        "output_mode": OUTPUT_MODE_PROGRESS_ONLY,
        "canonical_roi_sha256": item.image_sha256,
        "telemetry": safe_telemetry,
    }
    if reference_mode == REFERENCE_MODE_AUTO:
        detector_hash = str(reference_detector_sha256 or "").lower()
        _require(_is_sha256(detector_hash), "invalid reference detector SHA256")
        record.update(
            {
                "reference_detector_sha256": detector_hash,
                "auto_reference": reference_packet,
                "auto_reference_sha256": reference_packet_sha256,
            }
        )
    else:
        _require(
            reference_detector_sha256 is None,
            "controlled mode must not bind an automatic detector",
        )
        record["reference_input_sha256"] = reference_packet_sha256
    assert_label_free(record, location=f"adapter_output.{item.sample_id}")
    return record


def _status_from_direction(
    *,
    item: LabelFreeInput,
    reference: dict[str, Any],
    direction_valid: bool,
    pointer_angle: float | None,
) -> tuple[bool, float | None, str | None]:
    if not direction_valid or pointer_angle is None:
        return False, None, "invalid_direction"
    if reference.get("status") is not True:
        return False, None, str(
            reference.get("failure_code") or "reference_unavailable"
        )
    start = _finite(reference.get("start_angle"))
    angle_range = _finite(reference.get("range_angle"))
    if start is None or angle_range is None or abs(angle_range) <= 1e-12:
        return False, None, "invalid_reference"
    try:
        progress = prediction_progress_from_reference(
            pointer_angle,
            start_angle=start,
            range_angle=angle_range,
        )
    except (ValueError, ArithmeticError):
        return False, None, "progress_conversion_failed"
    return True, progress, None


class PEPDLabelFreeAdapter:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        expected_checkpoint_sha256: str,
        verification_path: Path,
        device: torch.device,
    ) -> None:
        checkpoint_hash, verification = _verification_binding(
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            verification_path=verification_path,
            expected_verification_protocols=frozenset(
                {
                    "pepd_syncg_grouped_val_run_verification_v1",
                    "pepd_syncg_grouped_val_bounded_extension_verification_v2",
                }
            ),
        )
        checkpoint = torch.load(
            checkpoint_path.resolve(),
            map_location="cpu",
            weights_only=False,
        )
        _require(isinstance(checkpoint, dict), "PEPD checkpoint is not an object")
        signature = checkpoint.get("signature") or {}
        _require(
            signature.get("protocol") == PROBABILISTIC_PIVOT_DIRECTION_PROTOCOL,
            "checkpoint is not a PEPD direction artifact",
        )
        self.checkpoint_sha256 = checkpoint_hash
        self.verification_sha256 = sha256_file(verification_path.resolve())
        self.verification_protocol = verification["protocol"]
        self.image_size = int(signature.get("image_size", 256))
        self.heatmap_size = int(signature.get("heatmap_size", 64))
        self.model = build_probabilistic_pivot_direction_model(
            angle_bins=int(signature.get("angle_bins", 72)),
            imagenet_pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.device = device
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(
        self,
        items: list[LabelFreeInput],
        images: list[np.ndarray],
        references: list[dict[str, Any]],
        *,
        reference_mode: str,
        reference_detector_sha256: str | None,
        amp_enabled: bool,
    ) -> list[dict[str, Any]]:
        _require(
            len(items) == len(images) == len(references),
            "PEPD batch/reference alignment drift",
        )
        tensors = [
            normalized_rgb_tensor(
                direct_resize_whole_roi(image, size=self.image_size)
            )
            for image in images
        ]
        inputs = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.type, enabled=amp_enabled):
            outputs = self.model(inputs)
        float_outputs = tuple(value.float() for value in outputs)
        decoded = decode_probabilistic_pivot_direction(*float_outputs)
        pivot_probabilities = torch.sigmoid(float_outputs[0][:, 0]).reshape(
            len(items), -1
        )
        spatial = pivot_probabilities / pivot_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        pivot_entropy = -torch.sum(
            spatial * torch.log(spatial.clamp_min(1e-12)), dim=1
        ) / math.log(float(pivot_probabilities.shape[1]))
        top2 = torch.topk(pivot_probabilities, k=2, dim=1).values
        raw_norm = torch.linalg.vector_norm(float_outputs[1], dim=1)
        stride = float(self.image_size) / float(self.heatmap_size)
        directions = decoded.direction.detach().cpu().numpy()
        valid = decoded.valid.detach().cpu().numpy()
        pivots = decoded.pivot_xy.detach().cpu().numpy()
        records: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            direction = directions[index].astype(np.float64)
            pointer_angle: float | None = None
            if bool(valid[index]):
                try:
                    pointer_angle = image_angle_from_direction(direction)
                except ValueError:
                    pass
            status, progress, failure = _status_from_direction(
                item=item,
                reference=references[index],
                direction_valid=bool(valid[index]),
                pointer_angle=pointer_angle,
            )
            records.append(
                _method_record(
                    item=item,
                    checkpoint_sha256=self.checkpoint_sha256,
                    reference_mode=reference_mode,
                    reference_packet=references[index],
                    reference_packet_sha256=canonical_json_sha256(
                        references[index]
                    ),
                    reference_detector_sha256=reference_detector_sha256,
                    status=status,
                    prediction_progress=progress,
                    failure_code=failure,
                    telemetry={
                        "direction_xy": direction,
                        "pointer_angle": pointer_angle,
                        "pivot_heatmap_xy": pivots[index],
                        "pivot_input_xy": pivots[index] * stride,
                        "pivot_peak": float(decoded.pivot_peak[index]),
                        "pivot_spatial_entropy": float(pivot_entropy[index]),
                        "pivot_top2_margin": float(top2[index, 0] - top2[index, 1]),
                        "direction_raw_norm": float(raw_norm[index]),
                        "angle_std_degrees": float(decoded.angle_std_degrees[index]),
                        "angle_log_variance": float(decoded.log_variance[index]),
                        "angle_bin_entropy": float(decoded.angle_entropy[index]),
                        "angle_bin_resultant_length": float(
                            decoded.bin_resultant_length[index]
                        ),
                        "native_input_size": self.image_size,
                        "whole_roi_direct_resize": True,
                        "meter_bbox": [
                            0,
                            0,
                            int(images[index].shape[1]),
                            int(images[index].shape[0]),
                        ],
                        "meter_detector_not_invoked": True,
                        "second_crop_not_invoked": True,
                    },
                )
            )
        return records


class VDNOfficial200LabelFreeAdapter:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        expected_checkpoint_sha256: str,
        verification_path: Path,
        vdn_source: Path,
        device: torch.device,
    ) -> None:
        checkpoint_hash, verification = _verification_binding(
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            verification_path=verification_path,
            expected_verification_protocols=frozenset(
                {OFFICIAL200_VERIFICATION_PROTOCOL}
            ),
        )
        checkpoint = torch.load(
            checkpoint_path.resolve(),
            map_location="cpu",
            weights_only=False,
        )
        _require(isinstance(checkpoint, dict), "VDN checkpoint is not an object")
        signature = checkpoint.get("signature") or {}
        _require(signature.get("protocol") == OFFICIAL200_PROTOCOL, "not official200 VDN")
        _require(
            checkpoint.get("checkpoint_protocol") == OFFICIAL200_CHECKPOINT_PROTOCOL,
            "official200 checkpoint envelope drift",
        )
        source = vdn_source.resolve(strict=True)
        _require(
            verify_vdn_source(source) == signature.get("vdn_source_commit"),
            "VDN source commit differs from checkpoint",
        )
        self.checkpoint_sha256 = checkpoint_hash
        self.verification_sha256 = sha256_file(verification_path.resolve())
        self.verification_protocol = verification["protocol"]
        self.vdn_source = source
        self.image_size = int(signature.get("image_size", 384))
        self.model = build_vdn_model(
            source,
            image_size=self.image_size,
            imagenet_pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.device = device
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(
        self,
        items: list[LabelFreeInput],
        images: list[np.ndarray],
        references: list[dict[str, Any]],
        *,
        reference_mode: str,
        reference_detector_sha256: str | None,
        amp_enabled: bool,
    ) -> list[dict[str, Any]]:
        _require(
            len(items) == len(images) == len(references),
            "VDN batch/reference alignment drift",
        )
        tensors = [
            normalized_bgr_tensor(
                direct_resize_whole_roi(image, size=self.image_size)
            )
            for image in images
        ]
        inputs = torch.stack(tensors).to(self.device, non_blocking=True)
        with torch.amp.autocast(self.device.type, enabled=amp_enabled):
            heatmaps, vector_maps = self.model(inputs)
        heatmaps = heatmaps.float()
        vector_maps = vector_maps.float()
        directions, peaks, valid = predict_directions(heatmaps, vector_maps)
        batch, _, _, width = heatmaps.shape
        flat = heatmaps[:, 0].reshape(batch, -1)
        indices = flat.argmax(dim=1)
        y = torch.div(indices, width, rounding_mode="floor")
        x = indices % width
        batch_indices = torch.arange(batch, device=heatmaps.device)
        raw_vectors = vector_maps[batch_indices, :, y, x]
        raw_norm = torch.linalg.vector_norm(raw_vectors, dim=1)
        directions_np = directions.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        stride = float(self.image_size) / float(heatmaps.shape[-1])
        records: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            direction = directions_np[index].astype(np.float64)
            pointer_angle: float | None = None
            if bool(valid_np[index]):
                try:
                    pointer_angle = image_angle_from_direction(direction)
                except ValueError:
                    pass
            status, progress, failure = _status_from_direction(
                item=item,
                reference=references[index],
                direction_valid=bool(valid_np[index]),
                pointer_angle=pointer_angle,
            )
            records.append(
                _method_record(
                    item=item,
                    checkpoint_sha256=self.checkpoint_sha256,
                    reference_mode=reference_mode,
                    reference_packet=references[index],
                    reference_packet_sha256=canonical_json_sha256(
                        references[index]
                    ),
                    reference_detector_sha256=reference_detector_sha256,
                    status=status,
                    prediction_progress=progress,
                    failure_code=failure,
                    telemetry={
                        "direction_xy": direction,
                        "pointer_angle": pointer_angle,
                        "heatmap_peak": float(peaks[index]),
                        "tip_heatmap_xy": [int(x[index]), int(y[index])],
                        "tip_input_xy": [float(x[index]) * stride, float(y[index]) * stride],
                        "direction_raw_norm": float(raw_norm[index]),
                        "native_input_size": self.image_size,
                        "whole_roi_direct_resize": True,
                        "meter_bbox": [
                            0,
                            0,
                            int(images[index].shape[1]),
                            int(images[index].shape[0]),
                        ],
                        "meter_detector_not_invoked": True,
                        "second_crop_not_invoked": True,
                    },
                )
            )
        return records


def run_adapters(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).resolve(strict=True)
    output_path = Path(args.output).resolve()
    metadata_path = output_path.with_suffix(".metadata.json")
    rows = strict_jsonl_load(input_path)
    # This must happen before checkpoint deserialization and before image I/O.
    reference_mode = str(args.reference_mode)
    items = validate_label_free_inputs(rows, reference_mode=reference_mode)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    method_names = (
        AUTO_METHOD_NAMES
        if reference_mode == REFERENCE_MODE_AUTO
        else CONTROLLED_METHOD_NAMES
    )
    reference_contract_sha256 = (
        AUTO_REFERENCE_CONTRACT_SHA256
        if reference_mode == REFERENCE_MODE_AUTO
        else CONTROLLED_REFERENCE_CONTRACT_SHA256
    )
    point_detector: Any | None = None
    reference_detector_path: Path | None = None
    reference_detector_sha256: str | None = None
    if reference_mode == REFERENCE_MODE_AUTO:
        _require(
            args.reference_detector_weights is not None,
            "automatic mode requires --reference-detector-weights",
        )
        reference_detector_path = Path(
            args.reference_detector_weights
        ).resolve(strict=True)
        expected_detector_hash = str(
            args.reference_detector_sha256 or ""
        ).lower()
        _require(
            _is_sha256(expected_detector_hash),
            "automatic mode requires a valid --reference-detector-sha256",
        )
        actual_detector_hash = sha256_file(reference_detector_path)
        _require(
            actual_detector_hash == expected_detector_hash,
            "automatic reference detector hash mismatch",
        )
        # Import and deserialize only after every label-free input row and the
        # frozen detector bytes have passed validation.
        point_detector_class = _load_target_detector_class()
        point_detector = point_detector_class(str(reference_detector_path))
        reference_detector_sha256 = actual_detector_hash
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pepd = PEPDLabelFreeAdapter(
        checkpoint_path=Path(args.pepd_checkpoint),
        expected_checkpoint_sha256=args.pepd_checkpoint_sha256,
        verification_path=Path(args.pepd_verification),
        device=device,
    )
    vdn = VDNOfficial200LabelFreeAdapter(
        checkpoint_path=Path(args.vdn_checkpoint),
        expected_checkpoint_sha256=args.vdn_checkpoint_sha256,
        verification_path=Path(args.vdn_verification),
        vdn_source=Path(args.vdn_source),
        device=device,
    )
    amp_enabled = device.type == "cuda" and not args.no_amp
    output_rows: list[dict[str, Any]] = []
    for offset in range(0, len(items), int(args.batch_size)):
        batch_items = items[offset : offset + int(args.batch_size)]
        valid_items: list[LabelFreeInput] = []
        valid_images: list[np.ndarray] = []
        failures: dict[str, str] = {}
        for item in batch_items:
            image, failure = _decode_bound_roi(item)
            if image is None:
                failures[item.sample_id] = str(failure or "image_decode_failed")
            else:
                valid_items.append(item)
                valid_images.append(image)
        predicted: dict[str, dict[str, dict[str, Any]]] = {
            item.sample_id: {} for item in valid_items
        }
        valid_references: list[dict[str, Any]] = []
        if valid_items:
            if reference_mode == REFERENCE_MODE_AUTO:
                _require(point_detector is not None, "point detector was not loaded")
                valid_references = [
                    _automatic_reference_packet(point_detector, image)
                    for image in valid_images
                ]
            else:
                valid_references = []
                for item in valid_items:
                    _require(
                        item.reference_input is not None,
                        f"{item.sample_id}: controlled reference absent",
                    )
                    valid_references.append(item.reference_input)
            pepd_records = pepd.predict(
                valid_items,
                valid_images,
                valid_references,
                reference_mode=reference_mode,
                reference_detector_sha256=reference_detector_sha256,
                amp_enabled=amp_enabled,
            )
            vdn_records = vdn.predict(
                valid_items,
                valid_images,
                valid_references,
                reference_mode=reference_mode,
                reference_detector_sha256=reference_detector_sha256,
                amp_enabled=amp_enabled,
            )
            for item, pepd_record, vdn_record in zip(
                valid_items, pepd_records, vdn_records
            ):
                predicted[item.sample_id] = {
                    method_names[0]: pepd_record,
                    method_names[1]: vdn_record,
                }
        for item in batch_items:
            if item.sample_id in failures:
                code = failures[item.sample_id]
                if reference_mode == REFERENCE_MODE_AUTO:
                    reference_packet = {
                        "status": False,
                        "start_angle": None,
                        "range_angle": None,
                        "reference_branch": "production_point_detector:not_invoked",
                        "failure_code": "image_unavailable_for_auto_reference",
                    }
                else:
                    _require(
                        item.reference_input is not None,
                        f"{item.sample_id}: controlled reference absent",
                    )
                    reference_packet = item.reference_input
                reference_packet_sha256 = canonical_json_sha256(reference_packet)
                methods = {
                    method_names[0]: _method_record(
                        item=item,
                        checkpoint_sha256=pepd.checkpoint_sha256,
                        reference_mode=reference_mode,
                        reference_packet=reference_packet,
                        reference_packet_sha256=reference_packet_sha256,
                        reference_detector_sha256=reference_detector_sha256,
                        status=False,
                        prediction_progress=None,
                        failure_code=code,
                        telemetry={
                            "whole_roi_direct_resize": True,
                            "meter_bbox": None,
                            "meter_detector_not_invoked": True,
                            "second_crop_not_invoked": True,
                        },
                    ),
                    method_names[1]: _method_record(
                        item=item,
                        checkpoint_sha256=vdn.checkpoint_sha256,
                        reference_mode=reference_mode,
                        reference_packet=reference_packet,
                        reference_packet_sha256=reference_packet_sha256,
                        reference_detector_sha256=reference_detector_sha256,
                        status=False,
                        prediction_progress=None,
                        failure_code=code,
                        telemetry={
                            "whole_roi_direct_resize": True,
                            "meter_bbox": None,
                            "meter_detector_not_invoked": True,
                            "second_crop_not_invoked": True,
                        },
                    ),
                }
            else:
                methods = predicted[item.sample_id]
            output_row = {
                "schema_version": 1,
                "protocol": INFERENCE_PROTOCOL,
                "sample_id": item.sample_id,
                "image_sha256": item.image_sha256,
                "canonical_roi_sha256": item.image_sha256,
                "methods": methods,
            }
            if reference_mode == REFERENCE_MODE_CONTROLLED:
                output_row["reference_input"] = item.reference_input
                output_row["reference_input_sha256"] = item.reference_input_sha256
            output_rows.append(output_row)
    assert_label_free(output_rows, location="direction_adapter_output")
    payload = _jsonl_bytes(output_rows)
    _atomic_new(output_path, payload)
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hash_protocol": HASH_PROTOCOL,
        "label_files_opened": 0,
        "ground_truth_fields_read": 0,
        "scale_fields_read": 0,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": len(items),
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(output_rows),
        },
        "contracts": {
            "roi_contract_sha256": ROI_CONTRACT_SHA256,
            "reference_mode": reference_mode,
            "reference_contract_sha256": reference_contract_sha256,
            "evidence_role": (
                EVIDENCE_ROLE_COMPONENT
                if reference_mode == REFERENCE_MODE_AUTO
                else EVIDENCE_ROLE_DIAGNOSTIC
            ),
            "same_whole_roi_for_both_methods": True,
            "canonical_input_is_meter_roi": True,
            "meter_bbox_policy": "[0,0,width,height]",
            "meter_detector_invoked": False,
            "second_crop_invoked": False,
            "automatic_start_end_detector_inside_whole_roi": (
                reference_mode == REFERENCE_MODE_AUTO
            ),
            "automatic_reference_primary_success_branch": "start_and_end",
            "incomplete_reference_branches_are_primary_failures": True,
            "legacy_fallback_role": "secondary_sensitivity_only",
            "manual_or_gt_reference_used": False,
            "physical_scale_used": False,
            "complete_automatic_numeric_range_output": False,
            "full_reading_primary_metric_eligible": False,
        },
        "methods": {
            method_names[0]: {
                "checkpoint_sha256": pepd.checkpoint_sha256,
                "verification_sha256": pepd.verification_sha256,
                "verification_protocol": pepd.verification_protocol,
                "native_input_size": pepd.image_size,
                "reference_detector_sha256": reference_detector_sha256,
            },
            method_names[1]: {
                "checkpoint_sha256": vdn.checkpoint_sha256,
                "verification_sha256": vdn.verification_sha256,
                "verification_protocol": vdn.verification_protocol,
                "vdn_source": str(vdn.vdn_source),
                "vdn_source_commit": verify_vdn_source(vdn.vdn_source),
                "native_input_size": vdn.image_size,
                "reference_detector_sha256": reference_detector_sha256,
            },
        },
        "automatic_reference_detector": (
            {
                "path": str(reference_detector_path),
                "sha256": reference_detector_sha256,
                "role": "start/end detector only; meter detector not loaded",
            }
            if reference_mode == REFERENCE_MODE_AUTO
            else None
        ),
        "adapter_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    _atomic_new(metadata_path, canonical_json_bytes(metadata))
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-mode",
        choices=(REFERENCE_MODE_AUTO, REFERENCE_MODE_CONTROLLED),
        default=REFERENCE_MODE_AUTO,
    )
    parser.add_argument("--reference-detector-weights", type=Path)
    parser.add_argument("--reference-detector-sha256")
    parser.add_argument("--pepd-checkpoint", type=Path, required=True)
    parser.add_argument("--pepd-checkpoint-sha256", required=True)
    parser.add_argument("--pepd-verification", type=Path, required=True)
    parser.add_argument("--vdn-checkpoint", type=Path, required=True)
    parser.add_argument("--vdn-checkpoint-sha256", required=True)
    parser.add_argument("--vdn-verification", type=Path, required=True)
    parser.add_argument("--vdn-source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_adapters(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
