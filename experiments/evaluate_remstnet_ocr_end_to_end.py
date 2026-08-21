"""Independent full-frame deployment replay with off-the-shelf numeric OCR.

The experiment deliberately separates inference from scoring:

``predict``
    Replays the existing detector boxes, re-runs the three frozen ReMSTNet-v3
    checkpoints, and applies a frozen automatic numeric-range pipeline based on
    RapidOCR/PP-OCRv4.  This command never opens the real-photo label file.

``score``
    Joins the sealed prediction rows to the 33 available physical-reading
    labels.  The remaining 120 frames contribute to coverage only.

Two operating points are reported without field-set tuning: the decoder's
native acceptance rule (confidence >= 0.55) and the conservative 0.95
threshold selected previously on the 2,224-image public calibration roster.
The detector is a cached replay because the deployment photographs do not have
bounding-box ground truth; consequently this is an end-to-end functional
replay, not a detector recall/IoU or live latency benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import torch

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_remstnet_clean_external import _predict_checkpoint
from experiments.automatic_numeric_range import GeometryHint, GeometryProviderResult
from experiments.garc_numeric_range_bridge import GARCAutomaticNumericRangeProvider
from experiments.pretrained_numeric_ocr import load_adapter
from experiments.syncg_numeric_ocr import OCRPosteriorToken
from experiments.vdn_baseline import (
    image_angle_from_direction,
    normalize_reference_points,
)
from utils.angleDetect.yoloDetection.yoloDectect import targetDetectModel


PROTOCOL: Final[str] = "remstnet_v3_ocr_end_to_end_deployment_v1"
PREDICTION_PROTOCOL: Final[str] = f"{PROTOCOL}_predictions"
SCORE_PROTOCOL: Final[str] = f"{PROTOCOL}_score"
EXPECTED_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
DECODER_THRESHOLD: Final[float] = 0.55
FAILURE_ERROR: Final[float] = 1.0

DEFAULT_FIELD_ROOT: Final[Path] = Path(
    r"C:\pointer_read\unified_real_photo_progress_v1"
)
DEFAULT_DETECTOR_ROOT: Final[Path] = Path(
    r"C:\pointer_read\paper_syncg_only_retrain_v1\field_gauge_full_frame"
)
DEFAULT_BUNDLE: Final[Path] = Path(
    r"C:\pointer_read\pretrained_numeric_ocr_gate_v3\bundles"
    r"\ppocrv4_mobile_control.json"
)
DEFAULT_POINT_DETECTOR: Final[Path] = (
    PROJECT_ROOT
    / "utils"
    / "angleDetect"
    / "yoloDetection"
    / "result"
    / "yolo_pointbest.pt"
)
DEFAULT_V4_SCORE: Final[Path] = Path(
    r"C:\pointer_read\pretrained_numeric_ocr_gate_v3\formal_outer_garc_v4"
    r"\score.json"
)
DEFAULT_V5_SCORE: Final[Path] = Path(
    r"C:\pointer_read\pretrained_numeric_ocr_gate_v5_local_v1"
    r"\formal_outer_garc\score.json"
)
DEFAULT_V5_SERVER_DIAGNOSTIC: Final[Path] = Path(
    r"C:\pointer_read\pretrained_numeric_ocr_gate_v5_server_local_v1"
    r"\diagnostic_inner_component_100img\summary.json"
)
DEFAULT_CHECKPOINTS: Final[tuple[Path, ...]] = (
    Path(
        r"C:\pointer_read\sgca_multiview_pilot_v1"
        r"\remstnet_v3_adaptive_seed20262020_epoch5_paper.pt"
    ),
    Path(
        r"C:\pointer_read\sgca_multiview_pilot_v1"
        r"\remstnet_v3_adaptive_seed20262021_epoch5_paper.pt"
    ),
    Path(
        r"C:\pointer_read\sgca_multiview_pilot_v1"
        r"\remstnet_v3_adaptive_seed20262022_epoch5_pilot.pt"
    ),
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    r"C:\pointer_read\remstnet_ocr_end_to_end_deployment_v1"
)


class EndToEndOCRError(RuntimeError):
    """An input, model output, or metric violates the experiment protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndToEndOCRError(message)


def _read_json(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSON file does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSONL file does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(
                isinstance(value, dict),
                f"JSONL row is not an object: {source}:{line_number}",
            )
            rows.append(value)
    _require(bool(rows), f"JSONL file is empty: {source}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    output.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(not output.exists(), f"refusing to overwrite output: {output}")
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    dict(row), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                )
                + "\n"
            )


def _index(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        _require(bool(sample_id), f"{label} row lacks sample_id")
        _require(sample_id not in result, f"{label} repeats {sample_id}")
        result[sample_id] = row
    return result


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class ProductionPointGeometryProvider:
    """Existing production start/end YOLO with the production 270-degree fallback.

    The detector is run once per cropped meter ROI.  The dial pivot is the crop
    center, matching the historical production reference conversion.  When one
    endpoint is missing, the other endpoint is placed 270 degrees away; when
    both are missing, the production defaults (45 degrees, 270-degree span)
    are used.  No numeric labels or manual points enter this provider.
    """

    def __init__(self, weights_path: Path):
        weights = Path(weights_path).resolve(strict=True)
        self.detector = targetDetectModel(str(weights))
        self._identity = {
            "provider": "frozen_production_start_end_yolo_with_default_geometry",
            "weights": str(weights),
            "pivot": "canonical_crop_center",
            "single_endpoint_fallback_span_degrees": 270.0,
            "no_endpoint_fallback_start_degrees": 45.0,
            "no_endpoint_fallback_span_degrees": 270.0,
            "manual_geometry_input": False,
            "physical_scale_input": False,
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @staticmethod
    def _point_from_angle(angle_degrees: float, *, radius: float = 0.46) -> tuple[float, float]:
        raw = math.radians(float(angle_degrees) + 180.0)
        dx = math.sin(raw)
        dy = -math.cos(raw)
        return 0.5 + radius * dx, 0.5 + radius * dy

    def predict(self, image_bgr: np.ndarray) -> GeometryProviderResult:
        _require(
            isinstance(image_bgr, np.ndarray)
            and image_bgr.dtype == np.uint8
            and image_bgr.ndim == 3
            and image_bgr.shape[2] == 3,
            "point geometry input must be uint8 BGR",
        )
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        detector_input = np.stack((gray,) * 3, axis=-1)
        confidences, boxes, _crops, class_ids = self.detector.target_detection(
            detector_input
        )

        best: dict[int, tuple[float, tuple[float, float]]] = {}
        for confidence, box, class_id in zip(
            confidences, boxes, class_ids, strict=True
        ):
            class_key = int(class_id)
            if class_key not in (1, 2):
                continue
            points = np.asarray(box, dtype=np.float64).reshape(-1, 2)
            center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
            score = float(confidence)
            if class_key not in best or score > best[class_key][0]:
                best[class_key] = (score, center)

        center_end = best.get(1, (0.0, None))[1]
        center_start = best.get(2, (0.0, None))[1]
        height, width = image_bgr.shape[:2]
        center_start, center_end = normalize_reference_points(
            center_start, center_end, image_width=width
        )

        branch: str
        if center_start is not None and center_end is not None:
            start = (float(center_start[0]) / width, float(center_start[1]) / height)
            end = (float(center_end[0]) / width, float(center_end[1]) / height)
            branch = "start_and_end"
        elif center_start is not None:
            start = (float(center_start[0]) / width, float(center_start[1]) / height)
            start_angle = image_angle_from_direction(
                (start[0] - 0.5, start[1] - 0.5)
            )
            end = self._point_from_angle((start_angle + 270.0) % 360.0)
            branch = "start_only_plus_270_degree_fallback"
        elif center_end is not None:
            end = (float(center_end[0]) / width, float(center_end[1]) / height)
            end_angle = image_angle_from_direction((end[0] - 0.5, end[1] - 0.5))
            start = self._point_from_angle((end_angle - 270.0) % 360.0)
            branch = "end_only_plus_270_degree_fallback"
        else:
            start = self._point_from_angle(45.0)
            end = self._point_from_angle(315.0)
            branch = "production_default_45_plus_270"

        detected_scores = [score for score, _center in best.values()]
        confidence = (
            float(min(detected_scores)) if len(detected_scores) == 2
            else float(detected_scores[0] * 0.5) if detected_scores
            else 0.0
        )
        hint = GeometryHint(
            pivot_xy=(0.5, 0.5),
            start_xy=start,
            end_xy=end,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            source="production_start_end_yolo_or_fallback",
        ).validate()
        return GeometryProviderResult(
            hint=hint,
            telemetry={
                "branch": branch,
                "detections": len(confidences),
                "class_ids": [int(value) for value in class_ids],
                "start_detector_confidence": best.get(2, (None, None))[0],
                "end_detector_confidence": best.get(1, (None, None))[0],
                "manual_geometry_input": False,
            },
        )


class PhotometricEnsembleOCRBackend:
    """Frozen OCR under deterministic original/CLAHE views with box NMS."""

    def __init__(self, backend: Any, *, view_mode: str):
        _require(view_mode in {"original", "clahe", "ensemble"}, "unknown OCR view mode")
        self.backend = backend
        self.view_mode = view_mode
        self._identity = {
            "protocol": "frozen_ocr_photometric_view_v1",
            "view_mode": view_mode,
            "views": (
                ["original_bgr", "lab_clahe_luminance"]
                if view_mode == "ensemble"
                else [view_mode]
            ),
            "box_merge": "axis_aligned_iou_nms_0.5_keep_highest_joint_score",
            "training_or_adaptation": False,
            "backend": dict(backend.identity),
        }

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._identity

    @staticmethod
    def _clahe(image_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            luminance
        )
        return cv2.cvtColor(
            cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR
        )

    @staticmethod
    def _joint_score(token: OCRPosteriorToken) -> float:
        posterior = max(
            (float(item.beam_probability) for item in token.hypotheses),
            default=0.0,
        )
        return float(token.detector_score) * posterior

    @staticmethod
    def _iou(left: OCRPosteriorToken, right: OCRPosteriorToken) -> float:
        left_box = np.asarray(left.box, dtype=np.float64)
        right_box = np.asarray(right.box, dtype=np.float64)
        lx1, ly1 = left_box.min(axis=0)
        lx2, ly2 = left_box.max(axis=0)
        rx1, ry1 = right_box.min(axis=0)
        rx2, ry2 = right_box.max(axis=0)
        ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
        ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
        return intersection / union if union > 0.0 else 0.0

    def infer_with_posteriors(
        self, image_bgr: np.ndarray
    ) -> tuple[list[Any], list[OCRPosteriorToken], float]:
        if self.view_mode == "original":
            return self.backend.infer_with_posteriors(image_bgr)
        views = [self._clahe(image_bgr)]
        if self.view_mode == "ensemble":
            views.insert(0, image_bgr)
        elapsed = 0.0
        candidates: list[OCRPosteriorToken] = []
        for view in views:
            _top1, posteriors, seconds = self.backend.infer_with_posteriors(view)
            elapsed += float(seconds)
            candidates.extend(posteriors)
        retained: list[OCRPosteriorToken] = []
        for token in sorted(candidates, key=self._joint_score, reverse=True):
            if all(self._iou(token, existing) < 0.5 for existing in retained):
                retained.append(token)
        retained.sort(key=lambda token: float(np.asarray(token.box)[:, 1].mean()))
        return [], retained, elapsed


def _imread(path: Path) -> np.ndarray:
    payload = np.fromfile(str(Path(path).resolve()), dtype=np.uint8)
    image = cv2.imdecode(
        payload, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    _require(image is not None and image.size > 0, f"cannot decode image: {path}")
    return image


def _crop_bbox(image: np.ndarray, bbox: Sequence[Any], *, sample_id: str) -> np.ndarray:
    _require(len(bbox) >= 4, f"{sample_id}: detector bbox is absent")
    values = [_finite(value) for value in bbox[:4]]
    _require(all(value is not None for value in values), f"{sample_id}: invalid bbox")
    left, top, right, bottom = (int(round(float(value))) for value in values)
    height, width = image.shape[:2]
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    _require(right > left and bottom > top, f"{sample_id}: empty bbox crop")
    return np.ascontiguousarray(image[top:bottom, left:right])


def _metric_record(score: Mapping[str, Any]) -> dict[str, Any]:
    metrics = score.get("metrics")
    _require(isinstance(metrics, Mapping), "candidate score lacks metrics")
    prediction_summary = score.get("prediction_summary")
    _require(isinstance(prediction_summary, Mapping), "candidate score lacks summary")
    summary = _read_json(Path(str(prediction_summary.get("path") or "")))
    timing = summary.get("timing")
    _require(isinstance(timing, Mapping), "candidate prediction timing is absent")
    return {
        "candidate_id": str(score.get("candidate_id") or ""),
        "samples": int(metrics["samples"]),
        "range_pair_rounded_exact_conditional": float(
            metrics["pair_rounded_exact_conditional"]
        ),
        "range_pair_rounded_exact_full_denominator": float(
            metrics["pair_rounded_exact_full_denominator"]
        ),
        "range_coverage": float(metrics["coverage"]),
        "mean_sample_seconds": float(timing["mean_sample_seconds"]),
    }


def candidate_selection_evidence(
    *,
    v4_score_path: Path,
    v5_score_path: Path,
    v5_server_diagnostic_path: Path,
) -> dict[str, Any]:
    """Read only public/synthetic candidate evidence, never field labels."""

    v4_score = _read_json(v4_score_path)
    v5_score = _read_json(v5_score_path)
    v4 = _metric_record(v4_score)
    v5 = _metric_record(v5_score)
    _require(v4["candidate_id"] == "ppocrv4_mobile_control", "unexpected v4 candidate")
    _require(v5["candidate_id"] == "ppocrv5_mobile", "unexpected v5 candidate")
    calibration = v4_score.get("acceptance_calibration")
    _require(isinstance(calibration, Mapping), "v4 acceptance calibration is absent")
    selected = calibration.get("selected_threshold")
    _require(isinstance(selected, Mapping), "v4 selected threshold is absent")
    conservative_threshold = float(selected["threshold"])
    _require(
        0.0 <= conservative_threshold <= 1.0,
        "v4 selected threshold is invalid",
    )

    server = _read_json(v5_server_diagnostic_path)
    server_metrics = server.get("metrics")
    server_timing = server.get("timing")
    _require(
        isinstance(server_metrics, Mapping) and isinstance(server_timing, Mapping),
        "v5 server diagnostic is incomplete",
    )
    server_record = {
        "candidate_id": str(server.get("candidate_id") or ""),
        "evaluation_kind": str(server.get("evaluation_kind") or ""),
        "evaluated_tokens": int(server["evaluated_tokens"]),
        "exact_accuracy": float(server_metrics["exact_accuracy"]),
        "parseable_fraction": float(server_metrics["parseable_fraction"]),
        "mean_token_seconds": float(server_timing["mean_token_seconds"]),
        "claim_eligible": bool(server.get("claim_eligible")),
    }
    return {
        "selection_uses_field_labels": False,
        "selected_candidate": "ppocrv4_mobile_control",
        "reason": (
            "On the same 2,224-image public calibration roster, PP-OCRv4 "
            "mobile had higher conditional and full-denominator exact range-pair "
            "accuracy than PP-OCRv5 mobile while also being faster. The v5 "
            "server recognizer remained a component-only, non-claim-eligible "
            "diagnostic and was substantially slower on CPU."
        ),
        "public_calibration": [v4, v5],
        "v5_server_component_diagnostic": server_record,
        "operating_points": {
            "decoder_default": DECODER_THRESHOLD,
            "public_calibration_conservative": conservative_threshold,
        },
        "deployment_geometry_note": (
            "The public range comparison used the historical PEPD/ScaleMark "
            "geometry. The field replay uses the available production start/end "
            "YOLO plus its documented fallback because the historical PEPD "
            "checkpoint is not present. Therefore 0.95 is reported as a "
            "transferred conservative sensitivity, not recalibrated field confidence."
        ),
        "sources": {
            "v4_score": str(Path(v4_score_path).resolve()),
            "v5_score": str(Path(v5_score_path).resolve()),
            "v5_server_diagnostic": str(Path(v5_server_diagnostic_path).resolve()),
        },
    }


def _prepare_label_free_items(
    *,
    field_root: Path,
    detector_labeled_path: Path,
    detector_unlabeled_path: Path,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # ``full_frame_input_manifest.jsonl`` contains only the 33 labeled source
    # frames.  ``raw_inventory.jsonl`` is the label-free 153-image roster.
    manifest = _read_jsonl(Path(field_root) / "raw_inventory.jsonl")
    detector_rows = _read_jsonl(detector_labeled_path) + _read_jsonl(
        detector_unlabeled_path
    )
    detector_index = _index(detector_rows, label="cached detector replay")
    manifest_index: dict[str, Mapping[str, Any]] = {}
    for row in manifest:
        sample_id = str(row.get("raw_id") or "")
        _require(bool(sample_id), "raw inventory row lacks raw_id")
        _require(sample_id not in manifest_index, f"raw inventory repeats {sample_id}")
        manifest_index[sample_id] = row
    _require(len(manifest_index) == 153, "full-frame roster must contain 153 images")
    _require(set(manifest_index) == set(detector_index), "detector/manifest roster differs")

    selected_manifest = manifest
    if limit is not None:
        _require(1 <= int(limit) <= len(manifest), "invalid prediction limit")
        selected_manifest = manifest[: int(limit)]
    items: list[dict[str, Any]] = []
    selected_detector_rows: list[dict[str, Any]] = []
    for manifest_row in selected_manifest:
        sample_id = str(manifest_row["raw_id"])
        detector = dict(detector_index[sample_id])
        detector_passed = str(detector.get("status")) == "pass"
        bbox = detector.get("best_bbox_xyxy") if detector_passed else None
        if detector_passed:
            _require(isinstance(bbox, list) and len(bbox) >= 4, f"{sample_id}: bbox absent")
        image_path = (
            Path(field_root).resolve() / str(manifest_row["image_path"])
        ).resolve()
        _require(image_path.is_file(), f"{sample_id}: full-frame image is absent")
        items.append(
            {
                "sample_id": sample_id,
                "image_path": str(image_path),
                "detector_passed": detector_passed,
                "detector_confidence": detector.get("best_confidence"),
                "bbox_xyxy": bbox,
            }
        )
        selected_detector_rows.append(detector)
    return items, selected_detector_rows


def _compact_range_diagnostics(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    bridge = telemetry.get("bridge")
    consensus = telemetry.get("consensus")
    bridge_map = bridge if isinstance(bridge, Mapping) else {}
    consensus_map = consensus if isinstance(consensus, Mapping) else {}
    traces = bridge_map.get("box_trace")
    recognized: list[dict[str, Any]] = []
    if isinstance(traces, Sequence):
        for trace in traces:
            if not isinstance(trace, Mapping):
                continue
            candidates = trace.get("posterior_candidates")
            texts = []
            if isinstance(candidates, Sequence):
                texts = [
                    str(candidate.get("text"))
                    for candidate in candidates
                    if isinstance(candidate, Mapping) and candidate.get("text") is not None
                ]
            recognized.append(
                {
                    "box": trace.get("box"),
                    "texts": texts,
                    "accepted_on_arc": bool(trace.get("accepted")),
                    "automatic_arc_progress": trace.get("automatic_arc_progress"),
                    "failure_reason": trace.get("failure_reason"),
                }
            )
    selected_candidates = consensus_map.get("selected_candidates")
    return {
        "ocr_seconds": _finite(telemetry.get("ocr_seconds")),
        "range_total_seconds": _finite(telemetry.get("total_seconds")),
        "posterior_token_count": int(telemetry.get("posterior_token_count") or 0),
        "top1_numeric_token_count": int(
            telemetry.get("top1_compatibility_token_count") or 0
        ),
        "geometry": bridge_map.get("geometry"),
        "recognized_tokens": recognized,
        "selected_candidates": (
            list(selected_candidates)
            if isinstance(selected_candidates, Sequence)
            else []
        ),
    }


def run_label_free_prediction(
    *,
    field_root: Path,
    detector_labeled_path: Path,
    detector_unlabeled_path: Path,
    checkpoint_paths: Sequence[Path],
    bundle_path: Path,
    point_detector_path: Path,
    v4_score_path: Path,
    v5_score_path: Path,
    v5_server_diagnostic_path: Path,
    output_root: Path,
    device_name: str,
    batch_size: int,
    cpu_threads: int,
    ocr_view_mode: str,
    limit: int | None = None,
) -> dict[str, Any]:
    _require(len(checkpoint_paths) == 3, "exactly three ReMST checkpoints are required")
    _require(batch_size >= 1 and cpu_threads >= 1, "invalid runtime setting")
    output_dir = Path(output_root).resolve()
    _require(not output_dir.exists(), f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    selection = candidate_selection_evidence(
        v4_score_path=v4_score_path,
        v5_score_path=v5_score_path,
        v5_server_diagnostic_path=v5_server_diagnostic_path,
    )
    conservative_threshold = float(
        selection["operating_points"]["public_calibration_conservative"]
    )
    items, detector_rows = _prepare_label_free_items(
        field_root=field_root,
        detector_labeled_path=detector_labeled_path,
        detector_unlabeled_path=detector_unlabeled_path,
        limit=limit,
    )

    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(max(1, int(cpu_threads)))

    experiment_started = time.perf_counter()
    progress_started = time.perf_counter()
    predictions_by_seed: dict[int, dict[str, float]] = {}
    progress_failures: dict[int, dict[str, str]] = {}
    model_identity: dict[int, dict[str, Any]] = {}
    for checkpoint in checkpoint_paths:
        seed, predictions, failures, metadata = _predict_checkpoint(
            Path(checkpoint), items, device=device, batch_size=batch_size
        )
        _require(seed in EXPECTED_SEEDS, f"unexpected ReMST seed: {seed}")
        _require(seed not in predictions_by_seed, f"duplicate ReMST seed: {seed}")
        predictions_by_seed[seed] = predictions
        progress_failures[seed] = failures
        model_identity[seed] = {
            "checkpoint": str(Path(checkpoint).resolve()),
            "architecture": str(metadata.get("architecture") or ""),
            "source_seed": seed,
        }
    _require(tuple(sorted(predictions_by_seed)) == EXPECTED_SEEDS, "seed roster differs")
    progress_seconds = time.perf_counter() - progress_started

    base_ocr_backend = load_adapter(Path(bundle_path), cpu_threads=cpu_threads)
    _require(
        str(base_ocr_backend.identity.get("candidate_id"))
        == str(selection["selected_candidate"]),
        "runtime OCR bundle differs from the preselected candidate",
    )
    ocr_backend = PhotometricEnsembleOCRBackend(
        base_ocr_backend, view_mode=ocr_view_mode
    )
    geometry_provider = ProductionPointGeometryProvider(point_detector_path)
    range_pipeline = GARCAutomaticNumericRangeProvider(
        geometry_provider, ocr_backend, input_size=768
    )

    rows: list[dict[str, Any]] = []
    for index, (item, detector_row) in enumerate(zip(items, detector_rows, strict=True), 1):
        sample_id = str(item["sample_id"])
        progress_by_seed = {
            str(seed): predictions_by_seed[seed].get(sample_id)
            for seed in EXPECTED_SEEDS
        }
        progress_passed = all(value is not None for value in progress_by_seed.values())
        range_status = False
        pred_start: float | None = None
        pred_end: float | None = None
        confidence = 0.0
        range_failure: str | None = None
        diagnostics: dict[str, Any] = {
            "ocr_seconds": None,
            "range_total_seconds": None,
            "posterior_token_count": 0,
            "top1_numeric_token_count": 0,
            "geometry": None,
            "recognized_tokens": [],
            "selected_candidates": [],
        }
        crop_hw: list[int] | None = None
        if bool(item["detector_passed"]):
            try:
                image = _imread(Path(str(item["image_path"])))
                crop = _crop_bbox(
                    image, item["bbox_xyxy"], sample_id=sample_id
                )
                crop_hw = [int(crop.shape[0]), int(crop.shape[1])]
                range_prediction = range_pipeline.predict(crop)
                range_status = bool(range_prediction.status)
                pred_start = _finite(range_prediction.pred_start)
                pred_end = _finite(range_prediction.pred_end)
                confidence = float(range_prediction.confidence)
                range_failure = range_prediction.failure_reason
                diagnostics = _compact_range_diagnostics(
                    dict(range_prediction.telemetry)
                )
            except Exception as exc:  # Per-frame deployment failure, not cohort abort.
                range_failure = f"range_exception:{type(exc).__name__}"
        else:
            range_failure = str(
                detector_row.get("failure_code") or "detector_failure"
            )

        numeric_range_valid = bool(
            range_status
            and pred_start is not None
            and pred_end is not None
            and pred_start != pred_end
            and math.isfinite(confidence)
        )
        accepted_default = numeric_range_valid and confidence >= DECODER_THRESHOLD
        accepted_conservative = (
            numeric_range_valid and confidence >= conservative_threshold
        )
        physical_by_seed = (
            {
                seed: float(pred_start + float(progress) * (pred_end - pred_start))
                for seed, progress in progress_by_seed.items()
                if progress is not None
            }
            if numeric_range_valid
            else {}
        )
        failure_codes: list[str] = []
        if not bool(item["detector_passed"]):
            failure_codes.append(str(detector_row.get("failure_code") or "detector_failure"))
        if not progress_passed:
            failure_codes.append("remst_progress_failure")
        if not numeric_range_valid:
            failure_codes.append(range_failure or "automatic_range_failure")
        elif not accepted_conservative:
            failure_codes.append("range_below_conservative_confidence")
        failure_codes = list(dict.fromkeys(failure_codes))

        rows.append(
            {
                "schema_version": 1,
                "protocol": PREDICTION_PROTOCOL,
                "sample_id": sample_id,
                "image_path": str(item["image_path"]),
                "detector": {
                    "status": bool(item["detector_passed"]),
                    "confidence": item.get("detector_confidence"),
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "crop_hw": crop_hw,
                },
                "remst_progress_by_seed": progress_by_seed,
                "automatic_range": {
                    "decoder_status": numeric_range_valid,
                    "predicted_start": pred_start,
                    "predicted_end": pred_end,
                    "confidence": confidence,
                    "failure_reason": range_failure,
                    "accepted_decoder_default": accepted_default,
                    "accepted_public_conservative": accepted_conservative,
                },
                "physical_reading_by_seed": physical_by_seed,
                "diagnostics": diagnostics,
                "status_decoder_default": bool(progress_passed and accepted_default),
                "status_public_conservative": bool(
                    progress_passed and accepted_conservative
                ),
                "failure_codes": failure_codes,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(items):
            print(
                f"OCR deployment replay {index}/{len(items)}: "
                f"default={sum(bool(row['status_decoder_default']) for row in rows)}, "
                f"conservative={sum(bool(row['status_public_conservative']) for row in rows)}",
                flush=True,
            )

    predictions_path = output_dir / "predictions.label_free.jsonl"
    _write_jsonl(predictions_path, rows)
    summary = {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "status": "complete_label_free_deployment_replay",
        "mode": "smoke" if limit is not None else "full",
        "label_file_opened": False,
        "field_labels_used_for_model_or_ocr_selection": False,
        "task": (
            "full frame -> cached best-confidence meter box -> ReMSTNet-v3 "
            "normalized progress -> PP-OCRv4/GARC automatic range -> physical reading"
        ),
        "scope": {
            "end_to_end_functional_replay": True,
            "live_detector_latency_benchmark": False,
            "detector_recall_or_iou_claim": False,
            "training_or_adaptation_on_field_data": False,
        },
        "samples": len(rows),
        "detector_passed": sum(bool(row["detector"]["status"]) for row in rows),
        "ocr_any_token": sum(
            int(int(row["diagnostics"]["posterior_token_count"]) > 0) for row in rows
        ),
        "range_decoder_passed": sum(
            bool(row["automatic_range"]["decoder_status"]) for row in rows
        ),
        "end_to_end_passed_decoder_default": sum(
            bool(row["status_decoder_default"]) for row in rows
        ),
        "end_to_end_passed_public_conservative": sum(
            bool(row["status_public_conservative"]) for row in rows
        ),
        "candidate_selection": selection,
        "models": {
            "remstnet": model_identity,
            "ocr": dict(ocr_backend.identity),
            "automatic_geometry": dict(geometry_provider.identity),
        },
        "timing": {
            "remstnet_three_seed_seconds": progress_seconds,
            "ocr_range_total_seconds": float(
                sum(
                    float(row["diagnostics"]["range_total_seconds"] or 0.0)
                    for row in rows
                )
            ),
            "ocr_only_total_seconds": float(
                sum(
                    float(row["diagnostics"]["ocr_seconds"] or 0.0)
                    for row in rows
                )
            ),
            "whole_command_seconds": time.perf_counter() - experiment_started,
            "latency_scope_note": (
                "OCR/range and ReMST stages were measured separately; detector boxes "
                "were replayed, so no live end-to-end latency is claimed."
            ),
        },
        "artifacts": {"predictions": str(predictions_path)},
    }
    summary_path = output_dir / "prediction_summary.json"
    _write_json(summary_path, summary)
    return summary


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "cannot summarize an empty metric")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _bootstrap_ci(
    values: np.ndarray, *, seed: int, replicates: int
) -> list[float]:
    _require(values.ndim == 1 and values.size > 0, "bootstrap values are invalid")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for offset in range(0, replicates, 2000):
        count = min(2000, replicates - offset)
        indices = rng.integers(0, values.size, size=(count, values.size))
        estimates[offset : offset + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _operating_point_metrics(
    *,
    scored_rows: Sequence[Mapping[str, Any]],
    status_key: str,
) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        key = str(seed)
        errors: list[float] = []
        bounded_errors: list[float] = []
        conditional_errors: list[float] = []
        conditional_oracle_errors: list[float] = []
        oracle_errors: list[float] = []
        passed = 0
        for row in scored_rows:
            span = float(row["true_end"] - row["true_start"])
            progress = row["remst_progress_by_seed"].get(key)
            oracle_error = (
                abs(float(progress) - float(row["normalized_target"]))
                if progress is not None
                else FAILURE_ERROR
            )
            oracle_errors.append(oracle_error)
            physical = row["physical_reading_by_seed"].get(key)
            accepted = bool(row[status_key]) and physical is not None
            if accepted:
                error = abs(float(physical) - float(row["ground_truth"])) / span
                conditional_errors.append(error)
                conditional_oracle_errors.append(oracle_error)
                passed += 1
            else:
                error = FAILURE_ERROR
            errors.append(error)
            bounded_errors.append(min(error, FAILURE_ERROR))
        per_seed[key] = {
            "samples": len(scored_rows),
            "passed": passed,
            "coverage": passed / len(scored_rows),
            "nmae_full_denominator": float(np.mean(errors)),
            "bounded_nmae_full_denominator": float(np.mean(bounded_errors)),
            "nmae_conditional": (
                float(np.mean(conditional_errors)) if conditional_errors else None
            ),
            "oracle_range_nmae_on_accepted": (
                float(np.mean(conditional_oracle_errors))
                if conditional_oracle_errors
                else None
            ),
            "range_stage_delta_nmae_on_accepted": (
                float(np.mean(conditional_errors) - np.mean(conditional_oracle_errors))
                if conditional_errors
                else None
            ),
            "acc_at_5_percent_conditional": (
                float(np.mean(np.asarray(conditional_errors) <= 0.05))
                if conditional_errors
                else None
            ),
            "median_absolute_error_full_denominator": float(np.median(errors)),
            "acc_at_2_percent_full_denominator": float(
                np.mean(np.asarray(errors) <= 0.02)
            ),
            "acc_at_5_percent_full_denominator": float(
                np.mean(np.asarray(errors) <= 0.05)
            ),
            "oracle_range_nmae": float(np.mean(oracle_errors)),
        }
    scalar_keys = (
        "coverage",
        "nmae_full_denominator",
        "bounded_nmae_full_denominator",
        "median_absolute_error_full_denominator",
        "acc_at_2_percent_full_denominator",
        "acc_at_5_percent_full_denominator",
        "oracle_range_nmae",
    )
    aggregate = {
        key: _mean_sd([float(per_seed[str(seed)][key]) for seed in EXPECTED_SEEDS])
        for key in scalar_keys
    }
    conditional_values = [
        float(per_seed[str(seed)]["nmae_conditional"])
        for seed in EXPECTED_SEEDS
        if per_seed[str(seed)]["nmae_conditional"] is not None
    ]
    aggregate["nmae_conditional"] = (
        _mean_sd(conditional_values) if conditional_values else None
    )
    for conditional_key in (
        "oracle_range_nmae_on_accepted",
        "range_stage_delta_nmae_on_accepted",
        "acc_at_5_percent_conditional",
    ):
        values = [
            float(per_seed[str(seed)][conditional_key])
            for seed in EXPECTED_SEEDS
            if per_seed[str(seed)][conditional_key] is not None
        ]
        aggregate[conditional_key] = _mean_sd(values) if values else None
    return {"per_seed": per_seed, "across_seed_mean_sd": aggregate}


def score_predictions(
    *,
    prediction_root: Path,
    labels_path: Path,
    output_path: Path,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    root = Path(prediction_root).resolve()
    summary = _read_json(root / "prediction_summary.json")
    _require(summary.get("protocol") == PREDICTION_PROTOCOL, "prediction protocol differs")
    _require(summary.get("mode") == "full", "only the full prediction roster may be scored")
    predictions = _read_jsonl(root / "predictions.label_free.jsonl")
    prediction_index = _index(predictions, label="OCR deployment predictions")
    labels = _read_jsonl(labels_path)
    label_index = _index(labels, label="field physical-reading labels")
    _require(len(prediction_index) == 153, "prediction roster must contain 153 images")
    _require(len(label_index) == 33, "field score roster must contain 33 labels")
    _require(set(label_index) <= set(prediction_index), "labeled sample lacks prediction")

    scored_rows: list[dict[str, Any]] = []
    for label in labels:
        sample_id = str(label["sample_id"])
        prediction = prediction_index[sample_id]
        start = float(label["scale_start"])
        end = float(label["scale_end"])
        _require(end > start, f"{sample_id}: physical scale span is not positive")
        scored_rows.append(
            {
                "sample_id": sample_id,
                "ground_truth": float(label["ground_truth"]),
                "normalized_target": float(label["normalized_progress"]),
                "true_start": start,
                "true_end": end,
                "detector": prediction["detector"],
                "remst_progress_by_seed": prediction["remst_progress_by_seed"],
                "automatic_range": prediction["automatic_range"],
                "physical_reading_by_seed": prediction["physical_reading_by_seed"],
                "status_decoder_default": prediction["status_decoder_default"],
                "status_public_conservative": prediction[
                    "status_public_conservative"
                ],
                "recognized_tokens": prediction["diagnostics"]["recognized_tokens"],
            }
        )

    operating_points = {
        "decoder_default": _operating_point_metrics(
            scored_rows=scored_rows, status_key="status_decoder_default"
        ),
        "public_conservative": _operating_point_metrics(
            scored_rows=scored_rows, status_key="status_public_conservative"
        ),
    }

    accepted_range_rows = [
        row for row in scored_rows if bool(row["automatic_range"]["decoder_status"])
    ]
    exact_pairs = 0
    pairs_within_5_percent = 0
    pairs_within_10_percent = 0
    endpoint_errors: list[float] = []
    span_errors: list[float] = []
    for row in accepted_range_rows:
        predicted_start = float(row["automatic_range"]["predicted_start"])
        predicted_end = float(row["automatic_range"]["predicted_end"])
        true_start = float(row["true_start"])
        true_end = float(row["true_end"])
        true_span = true_end - true_start
        exact_pairs += int(
            abs(predicted_start - true_start) <= 1e-6
            and abs(predicted_end - true_end) <= 1e-6
        )
        start_relative = abs(predicted_start - true_start) / true_span
        end_relative = abs(predicted_end - true_end) / true_span
        pairs_within_5_percent += int(
            start_relative <= 0.05 and end_relative <= 0.05
        )
        pairs_within_10_percent += int(
            start_relative <= 0.10 and end_relative <= 0.10
        )
        endpoint_errors.append(
            (abs(predicted_start - true_start) + abs(predicted_end - true_end))
            / (2.0 * true_span)
        )
        span_errors.append(
            abs((predicted_end - predicted_start) - true_span) / true_span
        )

    per_image: list[dict[str, Any]] = []
    oracle_means: list[float] = []
    default_means: list[float] = []
    conservative_means: list[float] = []
    for row in scored_rows:
        span = float(row["true_end"] - row["true_start"])
        seed_oracle: dict[str, float] = {}
        seed_default: dict[str, float] = {}
        seed_conservative: dict[str, float] = {}
        for seed in EXPECTED_SEEDS:
            key = str(seed)
            progress = row["remst_progress_by_seed"].get(key)
            seed_oracle[key] = (
                abs(float(progress) - float(row["normalized_target"]))
                if progress is not None
                else FAILURE_ERROR
            )
            physical = row["physical_reading_by_seed"].get(key)
            raw_error = (
                abs(float(physical) - float(row["ground_truth"])) / span
                if physical is not None
                else FAILURE_ERROR
            )
            seed_default[key] = (
                raw_error if bool(row["status_decoder_default"]) else FAILURE_ERROR
            )
            seed_conservative[key] = (
                raw_error
                if bool(row["status_public_conservative"])
                else FAILURE_ERROR
            )
        oracle_mean = statistics.fmean(seed_oracle.values())
        default_mean = statistics.fmean(seed_default.values())
        conservative_mean = statistics.fmean(seed_conservative.values())
        oracle_means.append(oracle_mean)
        default_means.append(default_mean)
        conservative_means.append(conservative_mean)
        per_image.append(
            {
                **dict(row),
                "oracle_range_normalized_error_by_seed": seed_oracle,
                "decoder_default_normalized_error_by_seed": seed_default,
                "public_conservative_normalized_error_by_seed": seed_conservative,
                "three_seed_mean_errors": {
                    "oracle_range": oracle_mean,
                    "decoder_default": default_mean,
                    "public_conservative": conservative_mean,
                },
            }
        )

    oracle_array = np.asarray(oracle_means, dtype=np.float64)
    default_array = np.asarray(default_means, dtype=np.float64)
    conservative_array = np.asarray(conservative_means, dtype=np.float64)
    result = {
        "schema_version": 1,
        "protocol": SCORE_PROTOCOL,
        "status": "complete",
        "evaluation": "independent OCR-complete full-frame deployment replay",
        "samples": {
            "all_full_frames_for_coverage": len(predictions),
            "labeled_full_frames_for_accuracy": len(scored_rows),
            "unlabeled_full_frames": len(predictions) - len(scored_rows),
        },
        "candidate_selection": summary["candidate_selection"],
        "coverage_all_frames": {
            "detector": int(summary["detector_passed"]),
            "ocr_any_token": int(summary["ocr_any_token"]),
            "range_decoder": int(summary["range_decoder_passed"]),
            "end_to_end_decoder_default": int(
                summary["end_to_end_passed_decoder_default"]
            ),
            "end_to_end_public_conservative": int(
                summary["end_to_end_passed_public_conservative"]
            ),
            "denominator": len(predictions),
        },
        "operating_point_metrics": operating_points,
        "range_metrics_on_labeled_frames": {
            "decoder_passed": len(accepted_range_rows),
            "coverage": len(accepted_range_rows) / len(scored_rows),
            "pair_exact_tolerance_1e_6_conditional": (
                exact_pairs / len(accepted_range_rows)
                if accepted_range_rows
                else None
            ),
            "pair_within_5_percent_true_span_conditional": (
                pairs_within_5_percent / len(accepted_range_rows)
                if accepted_range_rows
                else None
            ),
            "pair_within_10_percent_true_span_conditional": (
                pairs_within_10_percent / len(accepted_range_rows)
                if accepted_range_rows
                else None
            ),
            "normalized_endpoint_mae_conditional": (
                float(np.mean(endpoint_errors)) if endpoint_errors else None
            ),
            "normalized_span_mae_conditional": (
                float(np.mean(span_errors)) if span_errors else None
            ),
        },
        "rowwise_three_seed_mean": {
            "oracle_range_nmae": float(oracle_array.mean()),
            "decoder_default_nmae": float(default_array.mean()),
            "public_conservative_nmae": float(conservative_array.mean()),
            "decoder_default_minus_oracle": float(
                (default_array - oracle_array).mean()
            ),
            "public_conservative_minus_oracle": float(
                (conservative_array - oracle_array).mean()
            ),
            "image_bootstrap_ci95": {
                "oracle_range_nmae": _bootstrap_ci(
                    oracle_array, seed=bootstrap_seed, replicates=bootstrap_replicates
                ),
                "decoder_default_nmae": _bootstrap_ci(
                    default_array,
                    seed=bootstrap_seed + 1,
                    replicates=bootstrap_replicates,
                ),
                "public_conservative_nmae": _bootstrap_ci(
                    conservative_array,
                    seed=bootstrap_seed + 2,
                    replicates=bootstrap_replicates,
                ),
                "decoder_default_minus_oracle": _bootstrap_ci(
                    default_array - oracle_array,
                    seed=bootstrap_seed + 3,
                    replicates=bootstrap_replicates,
                ),
                "public_conservative_minus_oracle": _bootstrap_ci(
                    conservative_array - oracle_array,
                    seed=bootstrap_seed + 4,
                    replicates=bootstrap_replicates,
                ),
                "replicates": bootstrap_replicates,
                "resampling_unit": "full-frame image",
            },
        },
        "failure_breakdown_all_frames": dict(
            sorted(
                Counter(
                    code
                    for row in predictions
                    for code in set(row.get("failure_codes", []))
                ).items()
            )
        ),
        "timing": summary["timing"],
        "scope_notes": [
            "OCR and automatic range recovery use no field-set fitting or threshold selection.",
            "Detector boxes are replayed; bounding-box recall/IoU and live detector latency are not claimed.",
            "Failure receives normalized error 1.0; finite wrong readings retain their uncapped normalized error, and a bounded NMAE is also reported.",
            "Accuracy is computed only on 33 labeled images; all 153 images contribute to coverage.",
        ],
        "per_image": per_image,
    }
    _write_json(output_path, result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    predict.add_argument(
        "--detector-labeled",
        type=Path,
        default=DEFAULT_DETECTOR_ROOT / "field_gauge_full_frame_labeled_predictions.jsonl",
    )
    predict.add_argument(
        "--detector-unlabeled",
        type=Path,
        default=DEFAULT_DETECTOR_ROOT / "field_gauge_full_frame_unlabeled_predictions.jsonl",
    )
    predict.add_argument(
        "--checkpoints", type=Path, nargs=3, default=list(DEFAULT_CHECKPOINTS)
    )
    predict.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    predict.add_argument(
        "--point-detector", type=Path, default=DEFAULT_POINT_DETECTOR
    )
    predict.add_argument("--v4-score", type=Path, default=DEFAULT_V4_SCORE)
    predict.add_argument("--v5-score", type=Path, default=DEFAULT_V5_SCORE)
    predict.add_argument(
        "--v5-server-diagnostic", type=Path, default=DEFAULT_V5_SERVER_DIAGNOSTIC
    )
    predict.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--batch-size", type=int, default=64)
    predict.add_argument("--cpu-threads", type=int, default=2)
    predict.add_argument(
        "--ocr-view-mode",
        choices=("original", "clahe", "ensemble"),
        default="original",
    )
    predict.add_argument("--limit", type=int)

    score = commands.add_parser("score")
    score.add_argument("--prediction-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score.add_argument(
        "--labels", type=Path, default=DEFAULT_FIELD_ROOT / "full_frame_labels.jsonl"
    )
    score.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "score.json"
    )
    score.add_argument("--bootstrap-seed", type=int, default=20260821)
    score.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "predict":
        result = run_label_free_prediction(
            field_root=args.field_root,
            detector_labeled_path=args.detector_labeled,
            detector_unlabeled_path=args.detector_unlabeled,
            checkpoint_paths=args.checkpoints,
            bundle_path=args.bundle,
            point_detector_path=args.point_detector,
            v4_score_path=args.v4_score,
            v5_score_path=args.v5_score,
            v5_server_diagnostic_path=args.v5_server_diagnostic,
            output_root=args.output_root,
            device_name=args.device,
            batch_size=args.batch_size,
            cpu_threads=args.cpu_threads,
            ocr_view_mode=args.ocr_view_mode,
            limit=args.limit,
        )
    else:
        result = score_predictions(
            prediction_root=args.prediction_root,
            labels_path=args.labels,
            output_path=args.output,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
