"""Calibrate and evaluate sealed GARC full-auto public predictions.

This scorer authenticates every prediction, model bundle, split manifest, and
source hash before opening any public annotation.  The 1,080-image/50-group
independent range cohort is reported only as component/range evidence.  An
end-to-end full-auto claim is emitted only for the independently reconstructed
412-image/19-group intersection whose assigned PEPD checkpoint also held out
that physical group.

The 1,080-row table is explicitly a fixed-fold sensitivity analysis: the
primary fold's enhanced-V5/PEPD geometry is group-unseen for only 168 images
from 8 groups and overlaps geometry fitting for 912 images from 42 groups.
The independent report includes detector box precision/recall at fixed IoU
thresholds so the frozen DBNet++ activation gate can be evaluated without a
second pass over the cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

_PROJECT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_PROJECT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_PROJECT_BOOTSTRAP))

from experiments.automatic_numeric_range_public_protocol import (
    atomic_new_json,
    canonical_sha256,
    guard_public_path,
    load_frozen_protocol,
    require,
    sha256_file,
    strict_json,
    verify_bound_file,
    wilson_interval,
)
from experiments.garc_full_auto_public import (
    EXPECTED_JOINT_OOF,
    EXPECTED_JOINT_OOF_BY_SEED,
    EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY,
    EXPECTED_RANGE_INDEPENDENT,
    PLAN_PROTOCOL,
    PREDICTION_PROTOCOL,
    load_joint_oof_cohort,
    load_plan,
    load_prediction_bundle,
)
from experiments.vdn_baseline import load_syncg_manifest


CALIBRATION_PROTOCOL: Final[str] = "garc_full_auto_public_acceptance_v1"
VALIDATION_PROTOCOL: Final[str] = "garc_full_auto_public_validation_v1"
FIT_DIAGNOSTIC_PROTOCOL: Final[str] = "garc_full_auto_public_fit_diagnostic_v1"
DBNET_GATE_RECALL: Final[float] = 0.92
BOX_IOU_THRESHOLDS: Final[tuple[float, ...]] = (0.50, 0.75)
STRONG_GARC_PAIR_GAIN_MIN: Final[float] = 0.03
STRONG_MAX_COVERAGE_REGRESSION: Final[float] = 0.01
TOPK_GARC_PAIR_GAIN_MIN: Final[float] = 0.005
TOPK_MAX_COVERAGE_REGRESSION: Final[float] = 0.01
FUSION_SCREEN_PAIR_GAIN_MIN: Final[float] = 0.02
FUSION_SCREEN_MAX_COVERAGE_REGRESSION: Final[float] = 0.02
FUSION_FINAL_PAIR_GAIN_MIN: Final[float] = 0.005
FUSION_FINAL_MAX_COVERAGE_REGRESSION: Final[float] = 0.01
EVALUATOR_SOURCE: Final[Path] = Path(__file__).resolve()
PUBLIC_ANNOTATION_ROOT: Final[Path] = (
    _PROJECT_BOOTSTRAP / "datasets/SyncG/syncG/annotations/train"
).resolve()


@dataclass(frozen=True)
class PublicTruth:
    group_id: str
    scale_start: float
    scale_end: float
    reading: float
    text_boxes: tuple[tuple[float, float, float, float], ...]


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def range_variant_sha256(plan: Mapping[str, Any]) -> str:
    """Hash the fold-invariant GARC range method family.

    Fold-routed PEPD and enhanced-V5 head checkpoint identities are replaced
    by their shared authenticated OOF-family summary and assignment hashes.
    OCR, fusion family, decoder, thresholds, fixed auxiliary artifacts, and
    code remain frozen.
    """

    garc = plan["garc"]
    progress_binding = plan["progress_component"]["binding"]
    progress_identity = dict(progress_binding["provider_identity"])
    # The three authenticated OOF folds necessarily carry different model and
    # verification digests.  Remove only those per-fold values; retain the
    # provider family, automatic-reference implementation, AMP/input contract,
    # common authoritative handoff, wrapper/factory sources, and auxiliary
    # artifact hashes.  This makes the hash fold-invariant without allowing a
    # VDN/Transformer or alternate wrapper to borrow a PEPD calibration.
    progress_identity.pop("checkpoint_sha256", None)
    progress_identity.pop("verification_sha256", None)
    progress_family = {
        "provider_protocol": progress_binding["provider_protocol"],
        "provider_identity": progress_identity,
        "artifact_sha256": {
            name: digest
            for name, digest in sorted(progress_binding["artifact_sha256"].items())
            if name not in {"checkpoint", "verification"}
        },
        "source_sha256": dict(sorted(progress_binding["source_sha256"].items())),
        "factory": {
            "sha256": plan["progress_factory"]["sha256"],
            "function": plan["progress_factory"]["function"],
        },
    }
    artifacts = garc["artifacts"]
    fold_routed_geometry = garc.get("geometry_provider") == "enhanced_v5_oof_fold"
    artifact_identity = {
        name: artifacts[name]["sha256"]
        for name in sorted(artifacts)
        if not (fold_routed_geometry and name in {"geometry", "geometry_backbone"})
    }
    geometry_family = (
        {
            "provider": garc["geometry_provider"],
            "role": garc["geometry_fold"]["role"],
            "oof_summary_sha256": garc["geometry_fold"]["oof_summary"]["sha256"],
            "sample_assignment_sha256": garc["geometry_fold"]["joint_assignment"][
                "sample_assignment_sha256"
            ],
            "group_assignment_sha256": garc["geometry_fold"]["joint_assignment"][
                "group_assignment_sha256"
            ],
        }
        if fold_routed_geometry
        else {
            "provider": garc.get("geometry_provider"),
            "role": garc.get("geometry_fold", {}).get("role"),
        }
    )
    identity = {
        "protocol": "garc_fold_invariant_range_variant_v1",
        "recognizer_kind": garc["recognizer_kind"],
        "consensus_mode": garc["consensus_mode"],
        "geometry_mode": garc["geometry_mode"],
        "geometry_family": geometry_family,
        "progress_family": progress_family,
        "reference": {
            "mode": plan["reference"]["mode"],
            "detector_sha256": plan["reference"].get("detector_sha256"),
        },
        "input_size": int(garc["input_size"]),
        "detector_threshold": float(garc["detector_threshold"]),
        "posterior_top_k": int(garc["posterior_top_k"]),
        "consensus_config": garc["consensus_config"],
        "artifacts": artifact_identity,
        "range_code": {
            name: plan["code_bindings"][name]["sha256"]
            for name in (
                "garc_bridge",
                "garc_consensus",
                "garc_geometry_fusion",
                "tiny_ocr",
                "strong_ocr",
                "geometry_provider",
                "enhanced_v5_oof_geometry_provider",
                "enhanced_v5_head",
            )
        },
    }
    return canonical_sha256(identity)


def _normalized_text_boxes(
    annotation_path: Path,
    *,
    dial_bbox: Sequence[float],
) -> tuple[tuple[float, float, float, float], ...]:
    annotation_file = guard_public_path(
        annotation_path,
        label="public SyncG annotation",
        allowed_root=PUBLIC_ANNOTATION_ROOT,
    )
    annotation = strict_json(annotation_file)
    x1, y1, x2, y2 = (float(value) for value in dial_bbox[:4])
    left, top = math.floor(x1), math.floor(y1)
    right, bottom = math.ceil(x2), math.ceil(y2)
    width, height = float(right - left), float(bottom - top)
    require(width > 0.0 and height > 0.0, "invalid public dial ROI")
    result: list[tuple[float, float, float, float]] = []
    for value in annotation.get("text_bbox_annotations") or []:
        if not isinstance(value, Mapping):
            continue
        if str(value.get("type") or "").casefold() != "text":
            continue
        box = value.get("bbox")
        if not isinstance(box, Sequence) or len(box) < 4:
            continue
        bx1, by1, bx2, by2 = (float(item) for item in box[:4])
        if not all(math.isfinite(item) for item in (bx1, by1, bx2, by2)):
            continue
        ix1, iy1 = max(float(left), bx1), max(float(top), by1)
        ix2, iy2 = min(float(right), bx2), min(float(bottom), by2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        result.append(
            (
                (ix1 - left) / width,
                (iy1 - top) / height,
                (ix2 - left) / width,
                (iy2 - top) / height,
            )
        )
    require(bool(result), f"public sample has no valid text boxes: {annotation_path}")
    return tuple(result)


def load_public_truth_after_authentication(
    *,
    protocol_path: Path,
    roster: Sequence[Mapping[str, Any]],
    include_text_boxes: bool,
) -> dict[str, PublicTruth]:
    """Open public values only after the caller has authenticated predictions."""

    _, protocol = load_frozen_protocol(protocol_path)
    manifest_path = verify_bound_file(
        protocol, "source_bindings", "syncg_train_manifest"
    )
    verify_bound_file(protocol, "source_bindings", "syncg_train_manifest_protocol")
    samples, _ = load_syncg_manifest(manifest_path, expected_split="train")
    requested = {str(row["sample_id"]): row for row in roster}
    result: dict[str, PublicTruth] = {}
    for sample in samples:
        sample_id = str(sample.sample_id)
        roster_row = requested.get(sample_id)
        if roster_row is None:
            continue
        require(
            str(sample.group_id) == str(roster_row["group_id"]),
            f"public truth group drift: {sample_id}",
        )
        start = float(sample.scale_start)
        end = float(sample.scale_end)
        reading = float(sample.ground_truth)
        require(
            all(math.isfinite(value) for value in (start, end, reading)) and end > start,
            f"invalid public numeric truth: {sample_id}",
        )
        boxes: tuple[tuple[float, float, float, float], ...] = ()
        if include_text_boxes:
            annotation_value = sample.metadata.get("annotation_path")
            require(
                isinstance(annotation_value, str) and bool(annotation_value),
                f"missing public annotation path: {sample_id}",
            )
            boxes = _normalized_text_boxes(
                Path(annotation_value), dial_bbox=roster_row["dial_bbox"]
            )
        result[sample_id] = PublicTruth(
            group_id=str(sample.group_id),
            scale_start=start,
            scale_end=end,
            reading=reading,
            text_boxes=boxes,
        )
    require(set(result) == set(requested), "public truth roster is incomplete")
    return result


def _pair_correct(row: Mapping[str, Any], truth: PublicTruth) -> bool:
    return bool(
        row["range_status"]
        and row["predicted_scale_start"] is not None
        and row["predicted_scale_end"] is not None
        and int(round(float(row["predicted_scale_start"])))
        == int(round(truth.scale_start))
        and int(round(float(row["predicted_scale_end"])))
        == int(round(truth.scale_end))
    )


def threshold_table(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, PublicTruth],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    require(bool(rows), "cannot calibrate an empty prediction table")
    total = len(rows)
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            row
            for row in rows
            if row["range_status"]
            and float(row["range_confidence"]) >= float(threshold)
        ]
        correct = sum(
            _pair_correct(row, truth[str(row["sample_id"])]) for row in accepted
        )
        interval = wilson_interval(correct, len(accepted))
        table.append(
            {
                "threshold": float(threshold),
                "accepted": len(accepted),
                "coverage": len(accepted) / total,
                "pair_rounded_exact_accepted": correct,
                "pair_rounded_exact_conditional": (
                    correct / len(accepted) if accepted else None
                ),
                "pair_rounded_exact_wilson95": list(interval),
            }
        )
    return table


def select_threshold(
    table: Sequence[Mapping[str, Any]],
    *,
    minimum_accepted: int,
    target_precision: float,
) -> tuple[dict[str, Any], str]:
    require(bool(table), "empty threshold table")
    eligible = [row for row in table if int(row["accepted"]) >= minimum_accepted]
    if not eligible:
        return dict(table[0]), "insufficient_range_coverage_threshold_zero"
    target = [
        row
        for row in eligible
        if row["pair_rounded_exact_conditional"] is not None
        and float(row["pair_rounded_exact_conditional"]) >= target_precision
    ]
    if target:
        selected = max(
            target,
            key=lambda row: (
                float(row["coverage"]),
                float(row["pair_rounded_exact_wilson95"][0]),
                -float(row["threshold"]),
            ),
        )
        return dict(selected), "target_precision_met_maximum_coverage"
    selected = max(
        eligible,
        key=lambda row: (
            float(row["pair_rounded_exact_wilson95"][0]),
            float(row["coverage"]),
            -float(row["threshold"]),
        ),
    )
    return dict(selected), "target_unmet_maximum_wilson_lower_bound"


def score_range_rows(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, PublicTruth],
    *,
    threshold: float,
) -> dict[str, Any]:
    require(bool(rows), "cannot score an empty range cohort")
    total = len(rows)
    accepted = [
        row
        for row in rows
        if row["range_status"] and float(row["range_confidence"]) >= threshold
    ]
    correct = sum(
        _pair_correct(row, truth[str(row["sample_id"])]) for row in accepted
    )
    start_errors: list[float] = []
    end_errors: list[float] = []
    span_errors: list[float] = []
    endpoint_errors: list[float] = []
    numeric_exact = 0
    for row in accepted:
        target = truth[str(row["sample_id"])]
        start_error = abs(float(row["predicted_scale_start"]) - target.scale_start)
        end_error = abs(float(row["predicted_scale_end"]) - target.scale_end)
        start_errors.append(start_error)
        end_errors.append(end_error)
        endpoint_errors.append(0.5 * (start_error + end_error))
        span_errors.append(
            abs(
                (float(row["predicted_scale_end"]) - float(row["predicted_scale_start"]))
                - (target.scale_end - target.scale_start)
            )
        )
        numeric_exact += int(start_error <= 1e-6 and end_error <= 1e-6)
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(row)
    group_coverages: list[float] = []
    group_pair_full: list[float] = []
    for group_rows in by_group.values():
        group_accepted = [
            row
            for row in group_rows
            if row["range_status"] and float(row["range_confidence"]) >= threshold
        ]
        group_coverages.append(len(group_accepted) / len(group_rows))
        group_pair_full.append(
            sum(
                _pair_correct(row, truth[str(row["sample_id"])])
                for row in group_accepted
            )
            / len(group_rows)
        )
    failures: Counter[str] = Counter()
    for row in rows:
        if not row["range_status"]:
            failures[f"pipeline:{row['range_failure_reason'] or 'unspecified'}"] += 1
        elif float(row["range_confidence"]) < threshold:
            failures["acceptance:below_frozen_confidence"] += 1
        elif _pair_correct(row, truth[str(row["sample_id"])]):
            failures["accepted:pair_correct"] += 1
        else:
            failures["accepted:pair_incorrect"] += 1
    return {
        "samples": total,
        "groups": len(by_group),
        "minimum_confidence": float(threshold),
        "accepted": len(accepted),
        "coverage": len(accepted) / total,
        "coverage_wilson95": list(wilson_interval(len(accepted), total)),
        "pair_rounded_exact_full_denominator": correct / total,
        "pair_rounded_exact_full_denominator_wilson95": list(
            wilson_interval(correct, total)
        ),
        "pair_rounded_exact_conditional": correct / len(accepted) if accepted else None,
        "pair_rounded_exact_conditional_wilson95": list(
            wilson_interval(correct, len(accepted))
        ),
        "pair_numeric_exact_tolerance_1e_6_conditional": (
            numeric_exact / len(accepted) if accepted else None
        ),
        "start_mae_conditional": _mean(start_errors),
        "end_mae_conditional": _mean(end_errors),
        "range_endpoint_mae_conditional": _mean(endpoint_errors),
        "range_span_mae_conditional": _mean(span_errors),
        "group_macro_coverage": _mean(group_coverages),
        "group_macro_pair_rounded_exact_full_denominator": _mean(group_pair_full),
        "failure_breakdown": dict(sorted(failures.items())),
        "interpretation": "component/range-only; no end-to-end reading claim",
    }


def score_full_auto_rows(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, PublicTruth],
    *,
    threshold: float,
) -> dict[str, Any]:
    require(bool(rows), "cannot score an empty joint OOF cohort")
    total = len(rows)
    accepted = [
        row
        for row in rows
        if row["joint_oof_eligible"]
        and row["full_status"]
        and float(row["range_confidence"]) >= threshold
    ]
    absolute_errors: list[float] = []
    normalized_errors: list[float] = []
    progress_errors: list[float] = []
    for row in accepted:
        target = truth[str(row["sample_id"])]
        span = target.scale_end - target.scale_start
        absolute = abs(float(row["predicted_reading"]) - target.reading)
        absolute_errors.append(absolute)
        normalized_errors.append(absolute / span)
        expected_progress = (target.reading - target.scale_start) / span
        progress_errors.append(abs(float(row["prediction_progress"]) - expected_progress))
    full_absolute: list[float] = []
    full_normalized: list[float] = []
    accepted_by_id = {str(row["sample_id"]): row for row in accepted}
    for row in rows:
        target = truth[str(row["sample_id"])]
        span = target.scale_end - target.scale_start
        accepted_row = accepted_by_id.get(str(row["sample_id"]))
        if accepted_row is None:
            full_absolute.append(span)
            full_normalized.append(1.0)
        else:
            absolute = abs(float(accepted_row["predicted_reading"]) - target.reading)
            full_absolute.append(absolute)
            full_normalized.append(absolute / span)
    failures: Counter[str] = Counter()
    for row in rows:
        if not row["joint_oof_eligible"]:
            failures["audit:not_jointly_unseen"] += 1
        elif not row["full_status"]:
            failures[f"pipeline:{row['failure_reason'] or 'unspecified'}"] += 1
        elif float(row["range_confidence"]) < threshold:
            failures["acceptance:below_frozen_range_confidence"] += 1
        else:
            failures["accepted:full_reading"] += 1
    return {
        "samples": total,
        "groups": len({str(row["group_id"]) for row in rows}),
        "minimum_range_confidence": float(threshold),
        "accepted": len(accepted),
        "coverage": len(accepted) / total,
        "coverage_wilson95": list(wilson_interval(len(accepted), total)),
        "reading_mae_full_denominator_failure_penalty_one_range": _mean(full_absolute),
        "reading_nmae_full_denominator_failure_penalty_1": _mean(full_normalized),
        "reading_mae_conditional": _mean(absolute_errors),
        "reading_nmae_conditional": _mean(normalized_errors),
        "progress_mae_conditional": _mean(progress_errors),
        "failure_breakdown": dict(sorted(failures.items())),
    }


def _axis_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left[:4])
    rx1, ry1, rx2, ry2 = (float(value) for value in right[:4])
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _predicted_boxes(row: Mapping[str, Any]) -> list[tuple[float, float, float, float]]:
    prediction = row.get("range_prediction")
    if not isinstance(prediction, Mapping):
        return []
    telemetry = prediction.get("telemetry")
    if not isinstance(telemetry, Mapping):
        return []
    shape = telemetry.get("ocr_input_shape")
    bridge = telemetry.get("bridge")
    if (
        not isinstance(shape, Sequence)
        or len(shape) < 2
        or not isinstance(bridge, Mapping)
    ):
        return []
    height, width = float(shape[0]), float(shape[1])
    if height <= 0.0 or width <= 0.0:
        return []
    result: list[tuple[float, float, float, float]] = []
    for trace in bridge.get("box_trace") or []:
        if not isinstance(trace, Mapping):
            continue
        box = trace.get("box")
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        try:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
        except (TypeError, ValueError, IndexError):
            continue
        value = (
            max(0.0, min(xs) / width),
            max(0.0, min(ys) / height),
            min(1.0, max(xs) / width),
            min(1.0, max(ys) / height),
        )
        if value[2] > value[0] and value[3] > value[1]:
            result.append(value)
    return result


def _match_boxes(
    predicted: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> int:
    candidates = sorted(
        (
            (_axis_iou(prediction, target), pred_index, target_index)
            for pred_index, prediction in enumerate(predicted)
            for target_index, target in enumerate(expected)
        ),
        reverse=True,
    )
    used_predicted: set[int] = set()
    used_expected: set[int] = set()
    matched = 0
    for iou, pred_index, target_index in candidates:
        if iou < threshold:
            break
        if pred_index in used_predicted or target_index in used_expected:
            continue
        used_predicted.add(pred_index)
        used_expected.add(target_index)
        matched += 1
    return matched


def detector_box_metrics(
    rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, PublicTruth],
) -> dict[str, Any]:
    totals = {
        threshold: {"matched": 0, "predicted": 0, "expected": 0}
        for threshold in BOX_IOU_THRESHOLDS
    }
    for row in rows:
        predicted = _predicted_boxes(row)
        expected = truth[str(row["sample_id"])].text_boxes
        for threshold in BOX_IOU_THRESHOLDS:
            totals[threshold]["matched"] += _match_boxes(
                predicted, expected, threshold=threshold
            )
            totals[threshold]["predicted"] += len(predicted)
            totals[threshold]["expected"] += len(expected)
    metrics: dict[str, Any] = {
        "samples": len(rows),
        "groups": len({str(row["group_id"]) for row in rows}),
        "detector_family": "frozen Tiny OCR MobileNetV3Small-FPN detector shared by Tiny/Strong recognizers",
        "matching": "one-to-one greedy maximum IoU on all internally predicted boxes",
    }
    for threshold, values in totals.items():
        suffix = str(threshold).replace(".", "_")
        matched = values["matched"]
        predicted = values["predicted"]
        expected = values["expected"]
        metrics[f"iou_{suffix}"] = {
            "matched": matched,
            "predicted_boxes": predicted,
            "public_text_boxes": expected,
            "precision": matched / predicted if predicted else None,
            "recall": matched / expected if expected else None,
            "f1": (
                2.0 * matched / (predicted + expected)
                if predicted + expected
                else None
            ),
        }
    recall = metrics["iou_0_5"]["recall"]
    metrics["dbnet_plus_plus_gate"] = {
        "threshold": DBNET_GATE_RECALL,
        "observed_detector_box_recall_iou_0_5": recall,
        "triggered": recall is None or float(recall) < DBNET_GATE_RECALL,
        "rule": "activate DBNet++ iff independent detector box recall at IoU 0.5 is below 0.92",
    }
    return metrics


def calibrate(
    *,
    plan_path: Path,
    prediction_root: Path,
    output_path: Path,
    allow_smoke: bool = False,
) -> Path:
    # Authentication completes before public values open.
    summary, rows, roster, bundle = load_prediction_bundle(
        plan_path=plan_path,
        prediction_root=prediction_root,
        expected_partition="calibration",
        allow_smoke=allow_smoke,
    )
    plan_file, plan = load_plan(plan_path)
    protocol_path = Path(plan["parent_protocol"]["path"])
    truth = load_public_truth_after_authentication(
        protocol_path=protocol_path, roster=roster, include_text_boxes=False
    )
    protocol = strict_json(protocol_path)
    settings = protocol["acceptance_calibration"]
    thresholds = [float(value) for value in settings["candidate_thresholds"]]
    table = threshold_table(rows, truth, thresholds)
    formal = summary["mode"] == "formal"
    minimum_accepted = (
        max(
            int(settings["minimum_acceptance_count_formal"]),
            int(
                math.ceil(
                    float(settings["minimum_acceptance_fraction"]) * len(rows)
                )
            ),
        )
        if formal
        else 1
    )
    selected, reason = select_threshold(
        table,
        minimum_accepted=minimum_accepted,
        target_precision=float(settings["target_conditional_pair_rounded_exact"]),
    )
    artifact = {
        "schema_version": 1,
        "protocol": CALIBRATION_PROTOCOL,
        "status": "frozen_range_acceptance",
        "mode": summary["mode"],
        "claim_eligible": formal,
        "parent_plan": {"path": str(plan_file), "sha256": sha256_file(plan_file)},
        "parent_protocol": dict(plan["parent_protocol"]),
        "prediction_bundle": {
            "path": str(Path(prediction_root).resolve(strict=True)),
            "summary_sha256": sha256_file(Path(prediction_root) / "summary.json"),
            "predictions_sha256": summary["artifacts"]["predictions"]["sha256"],
            "range_binding_sha256": bundle.range_binding_sha256,
            "sample_ids_sha256": summary["sample_ids_sha256"],
            "samples": len(rows),
            "groups": len({str(row["group_id"]) for row in rows}),
        },
        "garc_variant": {
            "recognizer_kind": plan["garc"]["recognizer_kind"],
            "consensus_mode": plan["garc"]["consensus_mode"],
            "posterior_top_k": plan["garc"]["posterior_top_k"],
            "range_binding_sha256": bundle.range_binding_sha256,
            "range_variant_sha256": range_variant_sha256(plan),
        },
        "frozen_acceptance": {
            "minimum_confidence": selected["threshold"],
            "selection_reason": reason,
            "minimum_accepted": minimum_accepted,
            "target_conditional_pair_rounded_exact": settings[
                "target_conditional_pair_rounded_exact"
            ],
            "selected_calibration_metrics": selected,
            "predicted_numeric_values_modified": False,
        },
        "candidate_table": table,
        "audit": {
            "sealed_predictions_verified_before_public_values_opened": True,
            "independent_validation_values_opened": 0,
            "restricted_namespace_images_opened": 0,
            "rounding_is_evaluation_only": True,
        },
        "code": {"path": str(EVALUATOR_SOURCE), "sha256": sha256_file(EVALUATOR_SOURCE)},
    }
    output = guard_public_path(
        output_path, label="GARC calibration output", must_exist=False
    )
    atomic_new_json(output, artifact)
    return output


def load_calibration(
    path: Path,
    *,
    range_binding_sha256: str,
    expected_range_variant_sha256: str | None = None,
    allow_smoke: bool,
) -> tuple[dict[str, Any], float]:
    calibration_path = guard_public_path(path, label="GARC frozen calibration")
    artifact = strict_json(calibration_path)
    require(artifact.get("protocol") == CALIBRATION_PROTOCOL, "calibration drift")
    require(artifact.get("status") == "frozen_range_acceptance", "calibration not frozen")
    require(allow_smoke or artifact.get("mode") == "formal", "smoke calibration ineligible")
    require(
        artifact.get("garc_variant", {}).get("range_binding_sha256")
        == range_binding_sha256,
        "calibration range variant drift",
    )
    if expected_range_variant_sha256 is not None:
        require(
            artifact.get("garc_variant", {}).get("range_variant_sha256")
            == expected_range_variant_sha256,
            "calibration range-family variant drift",
        )
    threshold = _finite(
        artifact.get("frozen_acceptance", {}).get("minimum_confidence")
    )
    require(threshold is not None and 0.0 <= threshold <= 1.0, "bad threshold")
    return artifact, threshold


def _load_joint_runs(
    *,
    primary_plan: Path,
    primary_root: Path,
    extra_joint_runs: Sequence[tuple[Path, Path]],
    expected_range_binding: str,
    expected_range_variant: str,
    allow_smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bundles = [(Path(primary_plan), Path(primary_root)), *extra_joint_runs]
    merged: dict[str, dict[str, Any]] = {}
    seed_counts: Counter[int] = Counter()
    mapping_identity: str | None = None
    for plan_path, prediction_root in bundles:
        _, plan = load_plan(plan_path)
        if plan["joint_oof"]["status"] != "bound_for_overlap_audit":
            continue
        joint_path = Path(plan["joint_oof"]["summary"]["path"])
        _, _, mapping = load_joint_oof_cohort(joint_path)
        current_identity = sha256_file(joint_path)
        if mapping_identity is None:
            mapping_identity = current_identity
        require(mapping_identity == current_identity, "joint OOF mapping differs across runs")
        mapping_by_id = {str(row["sample_id"]): row for row in mapping}
        summary, rows, _, bundle = load_prediction_bundle(
            plan_path=plan_path,
            prediction_root=prediction_root,
            expected_partition="independent_validation",
            allow_smoke=allow_smoke,
        )
        current_variant = range_variant_sha256(plan)
        require(current_variant == expected_range_variant, "joint run range family drift")
        if plan["garc"].get("geometry_provider") != "enhanced_v5_oof_fold":
            require(
                bundle.range_binding_sha256 == expected_range_binding,
                "joint run range variant drift",
            )
        for row in rows:
            if not row["joint_oof_eligible"]:
                continue
            sample_id = str(row["sample_id"])
            require(sample_id not in merged, f"duplicate eligible joint row: {sample_id}")
            route = mapping_by_id[sample_id]
            seed_counts[int(route["pepd_seed"])] += 1
            merged[sample_id] = row
    rows = [merged[key] for key in sorted(merged)]
    groups = {str(row["group_id"]) for row in rows}
    complete = (
        len(rows) == EXPECTED_JOINT_OOF[0]
        and len(groups) == EXPECTED_JOINT_OOF[1]
        and dict(seed_counts) == EXPECTED_JOINT_OOF_BY_SEED
    )
    return rows, {
        "mapping_sha256": mapping_identity,
        "samples": len(rows),
        "groups": len(groups),
        "samples_by_pepd_seed": {
            str(seed): seed_counts[seed] for seed in sorted(EXPECTED_JOINT_OOF_BY_SEED)
        },
        "expected_samples": EXPECTED_JOINT_OOF[0],
        "expected_groups": EXPECTED_JOINT_OOF[1],
        "complete": complete,
    }


def validate(
    *,
    plan_path: Path,
    prediction_root: Path,
    calibration_path: Path,
    output_path: Path,
    extra_joint_runs: Sequence[tuple[Path, Path]] = (),
    allow_smoke: bool = False,
) -> Path:
    # Authenticate the full 1,080-row range bundle and calibration first.
    summary, rows, roster, bundle = load_prediction_bundle(
        plan_path=plan_path,
        prediction_root=prediction_root,
        expected_partition="independent_validation",
        allow_smoke=allow_smoke,
    )
    plan_file, plan = load_plan(plan_path)
    range_variant = range_variant_sha256(plan)
    calibration, threshold = load_calibration(
        calibration_path,
        range_binding_sha256=bundle.range_binding_sha256,
        expected_range_variant_sha256=range_variant,
        allow_smoke=allow_smoke,
    )
    protocol_path = Path(plan["parent_protocol"]["path"])
    # This is the first operation allowed to expose independent public values.
    truth = load_public_truth_after_authentication(
        protocol_path=protocol_path, roster=roster, include_text_boxes=True
    )
    range_inventory = (
        len(rows), len({str(row["group_id"]) for row in rows})
    )
    require(
        summary["mode"] != "formal" or range_inventory == EXPECTED_RANGE_INDEPENDENT,
        "formal range independent-validation inventory drift",
    )
    joint_rows, joint_audit = _load_joint_runs(
        primary_plan=plan_file,
        primary_root=prediction_root,
        extra_joint_runs=extra_joint_runs,
        expected_range_binding=bundle.range_binding_sha256,
        expected_range_variant=range_variant,
        allow_smoke=allow_smoke,
    )
    formal = summary["mode"] == "formal" and calibration["mode"] == "formal"
    fixed_fold_sensitivity = summary.get("overlap_audit", {}).get(
        "primary_fixed_fold_sensitivity"
    )
    if formal:
        require(
            isinstance(fixed_fold_sensitivity, Mapping),
            "formal primary run lacks fixed-fold sensitivity audit",
        )
        for key, expected in EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY.items():
            require(
                int(fixed_fold_sensitivity.get(key, -1)) == expected,
                f"formal primary fixed-fold audit drift: {key}",
            )
        require(
            fixed_fold_sensitivity.get("full_1080_all_component_unseen") is False,
            "formal primary run falsely claims full 1080 unseen geometry",
        )
    joint_metrics = (
        score_full_auto_rows(joint_rows, truth, threshold=threshold)
        if joint_rows
        else None
    )
    joint_range_metrics = (
        score_range_rows(joint_rows, truth, threshold=threshold)
        if joint_rows
        else None
    )
    result = {
        "schema_version": 1,
        "protocol": VALIDATION_PROTOCOL,
        "status": "independent_validation_complete",
        "mode": "formal" if formal else "smoke",
        "claim_eligible": formal,
        "claim_boundaries": {
            "range_fixed_fold_sensitivity": (
                "1,080 public SyncG/train images, 50 entity groups; OCR is split-unseen, "
                "but the primary enhanced-V5/PEPD geometry fold is unseen for only "
                "168 images/8 groups and has fit overlap for 912 images/42 groups"
            ),
            "end_to_end": (
                "412 images, 19 entity groups only when progress, enhanced-V5 head, "
                "and PEPD geometry backbone are all routed to the assigned OOF fold"
            ),
            "full_1080_end_to_end_claim_allowed": False,
        },
        "parent_plan": {"path": str(plan_file), "sha256": sha256_file(plan_file)},
        "parent_protocol": dict(plan["parent_protocol"]),
        "validation_predictions": {
            "path": str(Path(prediction_root).resolve(strict=True)),
            "summary_sha256": sha256_file(Path(prediction_root) / "summary.json"),
            "predictions_sha256": summary["artifacts"]["predictions"]["sha256"],
            "range_binding_sha256": bundle.range_binding_sha256,
        },
        "frozen_calibration": {
            "path": str(Path(calibration_path).resolve(strict=True)),
            "sha256": sha256_file(Path(calibration_path)),
            "minimum_confidence": threshold,
        },
        "overlap_audit": {
            "range_independent_validation_samples": EXPECTED_RANGE_INDEPENDENT[0],
            "range_independent_validation_groups": EXPECTED_RANGE_INDEPENDENT[1],
            "strict_progress_oof_union_samples": 4_380,
            "strict_progress_oof_union_groups": 197,
            "jointly_unseen_samples": joint_audit["samples"],
            "jointly_unseen_groups": joint_audit["groups"],
            "jointly_unseen_samples_by_pepd_seed": joint_audit[
                "samples_by_pepd_seed"
            ],
            "expected_jointly_unseen_samples": EXPECTED_JOINT_OOF[0],
            "expected_jointly_unseen_groups": EXPECTED_JOINT_OOF[1],
            "joint_cohort_complete": joint_audit["complete"],
            "joint_mapping_sha256": joint_audit["mapping_sha256"],
            "range_rows_not_jointly_unseen": (
                EXPECTED_RANGE_INDEPENDENT[0] - joint_audit["samples"]
            ),
            "primary_fixed_fold_sensitivity": fixed_fold_sensitivity,
        },
        "metrics": {
            "range_component_ungated": score_range_rows(rows, truth, threshold=0.0),
            "range_component_frozen_acceptance": score_range_rows(
                rows, truth, threshold=threshold
            ),
            "detector_independent_validation": detector_box_metrics(rows, truth),
            "joint_oof_range_frozen_acceptance": joint_range_metrics,
            "joint_oof_end_to_end_frozen_acceptance": joint_metrics,
        },
        "evidence_eligibility": {
            "range_component_claim": False,
            "ocr_and_fixed_fold_geometry_sensitivity": formal
            and range_inventory == EXPECTED_RANGE_INDEPENDENT,
            "joint_oof_end_to_end_claim": formal and bool(joint_audit["complete"]),
            "full_1080_end_to_end_claim": False,
            "under_pressure_range_comparison_allowed": False,
            "under_pressure_same_cohort_sensitivity_allowed": formal
            and bool(joint_audit["complete"]),
        },
        "audit": {
            "validation_predictions_verified_before_public_values_opened": True,
            "calibration_frozen_before_validation_values_opened": True,
            "validation_metrics_not_consumed_by_model": True,
            "numeric_range_values_supplied_to_model": 0,
            "manual_geometry_supplied_to_model": 0,
            "restricted_namespace_images_opened": 0,
        },
        "code": {"path": str(EVALUATOR_SOURCE), "sha256": sha256_file(EVALUATOR_SOURCE)},
    }
    output = guard_public_path(
        output_path, label="GARC validation output", must_exist=False
    )
    atomic_new_json(output, result)
    return output


def diagnose_algorithm_fit(
    *,
    plan_path: Path,
    prediction_root: Path,
    output_path: Path,
    allow_smoke: bool = False,
) -> Path:
    summary, rows, roster, bundle = load_prediction_bundle(
        plan_path=plan_path,
        prediction_root=prediction_root,
        expected_partition="algorithm_fit",
        allow_smoke=allow_smoke,
    )
    _, plan = load_plan(plan_path)
    truth = load_public_truth_after_authentication(
        protocol_path=Path(plan["parent_protocol"]["path"]),
        roster=roster,
        include_text_boxes=False,
    )
    result = {
        "schema_version": 1,
        "protocol": FIT_DIAGNOSTIC_PROTOCOL,
        "status": "fit_diagnostic_complete",
        "claim_eligible": False,
        "reason": "algorithm_fit values may tune the method and are never independent evidence",
        "parent_plan_sha256": sha256_file(Path(plan_path)),
        "prediction_summary_sha256": sha256_file(Path(prediction_root) / "summary.json"),
        "range_binding_sha256": bundle.range_binding_sha256,
        "metrics": score_range_rows(rows, truth, threshold=0.0),
        "audit": {
            "sealed_predictions_verified_before_public_values_opened": True,
            "independent_validation_values_opened": 0,
            "restricted_namespace_images_opened": 0,
        },
        "code": {"path": str(EVALUATOR_SOURCE), "sha256": sha256_file(EVALUATOR_SOURCE)},
    }
    output = guard_public_path(
        output_path, label="GARC fit diagnostic output", must_exist=False
    )
    atomic_new_json(output, result)
    return output


def _calibration_candidate(
    plan_path: Path,
    calibration_path: Path,
    *,
    allow_smoke: bool,
) -> dict[str, Any]:
    plan_file, plan = load_plan(plan_path)
    calibration_file = guard_public_path(
        calibration_path, label="GARC calibration candidate"
    )
    artifact = strict_json(calibration_file)
    require(artifact.get("protocol") == CALIBRATION_PROTOCOL, "candidate calibration drift")
    require(artifact.get("status") == "frozen_range_acceptance", "candidate not frozen")
    require(allow_smoke or artifact.get("mode") == "formal", "smoke candidate ineligible")
    require(
        artifact.get("parent_plan", {}).get("sha256") == sha256_file(plan_file),
        "candidate plan/calibration drift",
    )
    require(
        artifact.get("garc_variant", {}).get("range_variant_sha256")
        == range_variant_sha256(plan),
        "candidate range-family drift",
    )
    require(
        artifact.get("audit", {}).get("independent_validation_values_opened") == 0,
        "candidate selection touched independent validation",
    )
    selected = artifact.get("frozen_acceptance", {}).get(
        "selected_calibration_metrics"
    )
    require(isinstance(selected, Mapping), "candidate calibration metrics absent")
    samples = int(artifact.get("prediction_bundle", {}).get("samples", 0))
    accepted = int(selected.get("accepted", -1))
    correct = int(selected.get("pair_rounded_exact_accepted", -1))
    coverage = _finite(selected.get("coverage"))
    require(
        samples > 0
        and 0 <= correct <= accepted <= samples
        and coverage is not None
        and math.isclose(coverage, accepted / samples, abs_tol=1e-12),
        "candidate calibration metric drift",
    )
    return {
        "plan": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "calibration": str(calibration_file),
        "calibration_sha256": sha256_file(calibration_file),
        "mode": artifact["mode"],
        "sample_ids_sha256": artifact["prediction_bundle"]["sample_ids_sha256"],
        "samples": samples,
        "groups": int(artifact["prediction_bundle"]["groups"]),
        "recognizer_kind": plan["garc"]["recognizer_kind"],
        "consensus_mode": plan["garc"]["consensus_mode"],
        "geometry_mode": plan["garc"]["geometry_mode"],
        "range_variant_sha256": range_variant_sha256(plan),
        "checkpoints": {
            "detector": dict(plan["garc"]["artifacts"]["detector"]),
            "recognizer": dict(plan["garc"]["artifacts"]["recognizer"]),
        },
        "minimum_confidence": float(
            artifact["frozen_acceptance"]["minimum_confidence"]
        ),
        "coverage": coverage,
        "pair_rounded_exact_full_denominator": correct / samples,
        "pair_rounded_exact_conditional": (
            correct / accepted if accepted else None
        ),
    }


def _same_calibration_roster(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for key in ("mode", "sample_ids_sha256", "samples", "groups"):
        require(left[key] == right[key], f"calibration candidate {key} drift")


def select_recognizer(
    *,
    tiny_top1: tuple[Path, Path],
    tiny_topk: tuple[Path, Path],
    strong_topk: tuple[Path, Path] | None,
    training_evidence_path: Path,
    activation_decision_path: Path,
    strong_qualification_path: Path | None,
    output_path: Path,
) -> Path:
    """Select consensus/recognizer strictly from formal calibration artifacts."""

    top1 = _calibration_candidate(*tiny_top1, allow_smoke=False)
    topk = _calibration_candidate(*tiny_topk, allow_smoke=False)
    require(
        (top1["recognizer_kind"], top1["consensus_mode"])
        == ("tiny", "top1"),
        "Tiny-top1 candidate identity drift",
    )
    require(
        (topk["recognizer_kind"], topk["consensus_mode"])
        == ("tiny", "topk"),
        "Tiny-topK candidate identity drift",
    )
    require(top1["geometry_mode"] == topk["geometry_mode"] == "v5", "base geometry drift")
    _same_calibration_roster(top1, topk)
    topk_gain = (
        topk["pair_rounded_exact_full_denominator"]
        - top1["pair_rounded_exact_full_denominator"]
    )
    topk_coverage_regression = top1["coverage"] - topk["coverage"]
    retain_topk = bool(
        topk_gain >= TOPK_GARC_PAIR_GAIN_MIN
        and topk_coverage_regression <= TOPK_MAX_COVERAGE_REGRESSION
    )
    selected_tiny = topk if retain_topk else top1

    strong: dict[str, Any] | None = None
    retain_strong = False
    if strong_topk is not None:
        strong = _calibration_candidate(*strong_topk, allow_smoke=False)
        require(
            (strong["recognizer_kind"], strong["consensus_mode"])
            == ("strong", "topk"),
            "Strong-topK candidate identity drift",
        )
        require(strong["geometry_mode"] == "v5", "Strong base geometry drift")
        _same_calibration_roster(selected_tiny, strong)
        strong_gain = (
            strong["pair_rounded_exact_full_denominator"]
            - selected_tiny["pair_rounded_exact_full_denominator"]
        )
        strong_coverage_regression = selected_tiny["coverage"] - strong["coverage"]
        retain_strong = bool(
            strong_gain >= STRONG_GARC_PAIR_GAIN_MIN
            and strong_coverage_regression <= STRONG_MAX_COVERAGE_REGRESSION
        )
        strong = {
            **strong,
            "pair_gain_over_selected_tiny": strong_gain,
            "coverage_regression_from_selected_tiny": strong_coverage_regression,
            "retention_pass": retain_strong,
        }
    evidence_file = guard_public_path(
        training_evidence_path, label="GARC-aligned OCR training evidence"
    )
    activation_file = guard_public_path(
        activation_decision_path, label="outer-calibration OCR activation decision"
    )
    evidence = strict_json(evidence_file)
    activation = strict_json(activation_file)
    require(
        evidence.get("protocol") == "garc_aligned_ocr_training_evidence_v1"
        and evidence.get("status") == "verified_garc_aligned_ocr_training_evidence",
        "OCR training evidence drift",
    )
    require(
        activation.get("protocol") == "syncg_strong_numeric_ocr_gate_decision_v2"
        and activation.get("selection_partition") == "garc_outer_calibration",
        "OCR activation decision drift",
    )
    strong_trained = bool(evidence.get("strong", {}).get("available"))
    require(
        strong_trained == bool(activation.get("train_strong_recognizer")),
        "Strong training evidence differs from the activation decision",
    )
    qualification: Mapping[str, Any] | None = None
    qualification_binding: dict[str, str] | None = None
    if strong_trained:
        require(strong_qualification_path is not None, "activated Strong lacks qualification")
        qualification_file = guard_public_path(
            strong_qualification_path, label="Strong OCR candidate qualification"
        )
        qualification = strict_json(qualification_file)
        require(
            qualification.get("protocol")
            == "syncg_strong_numeric_ocr_candidate_qualification_v1",
            "Strong qualification protocol drift",
        )
        require(
            qualification.get("status")
            in {"strong_candidate_qualified", "strong_candidate_rejected"},
            "Strong qualification status drift",
        )
        require(
            qualification.get("selection_role")
            == "candidate_eligibility_only_not_final_selection",
            "Strong qualification role drift",
        )
        require(
            qualification.get("activation_decision", {}).get("sha256")
            == sha256_file(activation_file),
            "Strong qualification/activation binding drift",
        )
        qualifier_code = Path(
            str(qualification.get("code", {}).get("path") or "")
        ).resolve(strict=True)
        require(
            qualification.get("code", {}).get("sha256") == sha256_file(qualifier_code),
            "Strong qualification code hash drift",
        )
        tiny_report_file = guard_public_path(
            Path(str(qualification.get("tiny_report", {}).get("path") or "")),
            label="Tiny outer-calibration component report",
        )
        strong_report_file = guard_public_path(
            Path(str(qualification.get("strong_report", {}).get("path") or "")),
            label="Strong outer-calibration component report",
        )
        require(
            qualification.get("tiny_report", {}).get("sha256")
            == sha256_file(tiny_report_file)
            and qualification.get("strong_report", {}).get("sha256")
            == sha256_file(strong_report_file),
            "Strong qualification report hash drift",
        )
        tiny_report = strict_json(tiny_report_file)
        strong_report = strict_json(strong_report_file)
        require(
            tiny_report.get("recognizer_checkpoint", {}).get("sha256")
            == evidence["tiny"]["components"]["recognizer"]["sha256"]
            and strong_report.get("recognizer_checkpoint", {}).get("sha256")
            == evidence["strong"]["recognizer"]["sha256"],
            "Strong qualification report/checkpoint binding drift",
        )
        from experiments.qualify_syncg_strong_numeric_ocr_candidate import qualify

        recomputed = qualify(
            tiny_report_path=tiny_report_file,
            strong_report_path=strong_report_file,
            activation_path=activation_file,
            protocol_path=Path(str(qualification["upgrade_protocol"]["path"])),
        )
        require(
            canonical_sha256(recomputed) == canonical_sha256(qualification),
            "Strong qualification artifact differs from trusted recomputation",
        )
        eligible = bool(
            qualification.get("strong_candidate_eligible_for_garc_calibration")
        )
        require(
            eligible == (qualification.get("status") == "strong_candidate_qualified"),
            "Strong qualification eligibility drift",
        )
        for key in (
            "images_opened_by_qualifier",
            "annotations_opened_by_qualifier",
            "independent_validation_opened",
            "development_excluded_opened",
            "joint_oof_412_19_opened",
            "field_test_sealed_confirmatory_opened",
        ):
            require(
                int(qualification.get("data_access_audit", {}).get(key, -1)) == 0,
                f"Strong qualification crossed forbidden data boundary: {key}",
            )
        qualification_binding = {
            "path": str(qualification_file),
            "sha256": sha256_file(qualification_file),
            "status": str(qualification["status"]),
        }
    else:
        require(
            strong_qualification_path is None,
            "non-activated Strong must not supply a qualification artifact",
        )
        eligible = False
    require(
        (strong is not None) == eligible,
        "Strong candidate presence differs from frozen qualification eligibility",
    )
    tiny_detector_sha = str(
        evidence.get("tiny", {}).get("components", {}).get("detector", {}).get("sha256")
        or ""
    )
    tiny_recognizer_sha = str(
        evidence.get("tiny", {}).get("components", {}).get("recognizer", {}).get("sha256")
        or ""
    )
    for candidate in (top1, topk):
        require(
            candidate["checkpoints"]["detector"]["sha256"] == tiny_detector_sha
            and candidate["checkpoints"]["recognizer"]["sha256"] == tiny_recognizer_sha,
            "Tiny selection checkpoint binding drift",
        )
    require(
        activation.get("tiny_checkpoint", {}).get("sha256") == tiny_recognizer_sha,
        "activation/Tiny recognizer binding drift",
    )
    if strong is not None:
        require(
            strong["checkpoints"]["detector"]["sha256"] == tiny_detector_sha
            and strong["checkpoints"]["recognizer"]["sha256"]
            == evidence["strong"]["recognizer"]["sha256"],
            "Strong selection checkpoint binding drift",
        )
    selected = strong if retain_strong and strong is not None else selected_tiny
    evidence_binding = {"path": str(evidence_file), "sha256": sha256_file(evidence_file)}
    activation_binding = {
        "path": str(activation_file),
        "sha256": sha256_file(activation_file),
    }
    decision = {
        "schema_version": 1,
        "protocol": "garc_numeric_recognizer_selection_v2",
        "status": "frozen_calibration_only_selection",
        "selected_recognizer": selected["recognizer_kind"],
        "selected_consensus": selected["consensus_mode"],
        "selected_plan": selected["plan"],
        "selected_calibration": selected["calibration"],
        "tiny_top1": top1,
        "tiny_topk": {
            **topk,
            "pair_gain_over_top1": topk_gain,
            "coverage_regression_from_top1": topk_coverage_regression,
            "retention_pass": retain_topk,
        },
        "strong_topk": strong,
        "ocr_training_evidence": evidence_binding,
        "outer_calibration_activation_decision": activation_binding,
        "strong_candidate_qualification": qualification_binding,
        "selection_input_sha256": canonical_sha256(
            {
                "tiny_top1": top1,
                "tiny_topk": topk,
                "strong_topk": strong,
                "ocr_training_evidence": evidence_binding,
                "outer_calibration_activation_decision": activation_binding,
                "strong_candidate_qualification": qualification_binding,
            }
        ),
        "thresholds": {
            "topk_pair_gain_min": TOPK_GARC_PAIR_GAIN_MIN,
            "topk_max_coverage_regression": TOPK_MAX_COVERAGE_REGRESSION,
            "strong_pair_gain_min": STRONG_GARC_PAIR_GAIN_MIN,
            "strong_max_coverage_regression": STRONG_MAX_COVERAGE_REGRESSION,
        },
        "audit": {
            "selection_data_partition": "calibration",
            "independent_validation_artifacts_opened": 0,
            "formal_calibration_results_only": True,
            "fallback_is_tiny_top1": True,
            "restricted_namespace_images_opened": 0,
        },
        "code": {"path": str(EVALUATOR_SOURCE), "sha256": sha256_file(EVALUATOR_SOURCE)},
    }
    output = guard_public_path(
        output_path, label="GARC recognizer selection output", must_exist=False
    )
    atomic_new_json(output, decision)
    return output


def select_geometry(
    *,
    base: tuple[Path, Path],
    fusions: Sequence[tuple[Path, Path]],
    stage: str,
    output_path: Path,
) -> Path:
    """Calibration-only gate for the optional PEPD geometry fusion."""

    require(stage in ("screen", "final"), "unknown geometry-selection stage")
    require(bool(fusions), "no geometry-fusion candidate supplied")
    allow_smoke = stage == "screen"
    baseline = _calibration_candidate(*base, allow_smoke=allow_smoke)
    require(baseline["geometry_mode"] == "v5", "geometry baseline is not V5")
    if stage == "screen":
        require(baseline["mode"] == "smoke", "screen baseline must be bounded")
        gain_min = FUSION_SCREEN_PAIR_GAIN_MIN
        max_regression = FUSION_SCREEN_MAX_COVERAGE_REGRESSION
    else:
        require(baseline["mode"] == "formal", "final baseline must be formal")
        gain_min = FUSION_FINAL_PAIR_GAIN_MIN
        max_regression = FUSION_FINAL_MAX_COVERAGE_REGRESSION
    evaluated: list[dict[str, Any]] = []
    allowed_modes = {"v5_pepd_fusion", "v5_pepd_base_fusion"}
    for fusion in fusions:
        candidate = _calibration_candidate(*fusion, allow_smoke=allow_smoke)
        require(candidate["geometry_mode"] in allowed_modes, "fusion identity drift")
        require(
            (baseline["recognizer_kind"], baseline["consensus_mode"])
            == (candidate["recognizer_kind"], candidate["consensus_mode"]),
            "geometry candidates use different OCR/consensus variants",
        )
        _same_calibration_roster(baseline, candidate)
        require(
            candidate["mode"] == ("smoke" if stage == "screen" else "formal"),
            "geometry candidate calibration mode drift",
        )
        pair_gain = (
            candidate["pair_rounded_exact_full_denominator"]
            - baseline["pair_rounded_exact_full_denominator"]
        )
        coverage_regression = baseline["coverage"] - candidate["coverage"]
        retained = bool(pair_gain >= gain_min and coverage_regression <= max_regression)
        evaluated.append(
            {
                **candidate,
                "pair_full_denominator_gain": pair_gain,
                "coverage_regression": coverage_regression,
                "retention_pass": retained,
            }
        )
    retained_candidates = [row for row in evaluated if row["retention_pass"]]
    selected = (
        max(
            retained_candidates,
            key=lambda row: (
                row["pair_rounded_exact_full_denominator"],
                row["coverage"],
                row["geometry_mode"],
            ),
        )
        if retained_candidates
        else baseline
    )
    decision = {
        "schema_version": 1,
        "protocol": "garc_geometry_fusion_calibration_gate_v1",
        "status": "frozen_calibration_only_selection",
        "stage": stage,
        "fusion_retained": bool(retained_candidates),
        "selected_geometry": selected["geometry_mode"],
        "selected_plan": selected["plan"],
        "selected_calibration": selected["calibration"],
        "baseline": baseline,
        "fusion_candidates": evaluated,
        "thresholds": {
            "pair_full_denominator_gain_min": gain_min,
            "maximum_coverage_regression": max_regression,
        },
        "audit": {
            "selection_data_partition": "calibration",
            "independent_validation_artifacts_opened": 0,
            "restricted_namespace_images_opened": 0,
        },
        "code": {"path": str(EVALUATOR_SOURCE), "sha256": sha256_file(EVALUATOR_SOURCE)},
    }
    output = guard_public_path(
        output_path, label="GARC geometry selection output", must_exist=False
    )
    atomic_new_json(output, decision)
    return output


def _joint_arg(value: str) -> tuple[Path, Path]:
    parts = value.split("::", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("joint run must be PLAN::PREDICTION_ROOT")
    return Path(parts[0]), Path(parts[1])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--plan", type=Path, required=True)
    calibration.add_argument("--prediction-root", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--allow-smoke", action="store_true")
    validation = commands.add_parser("validate")
    validation.add_argument("--plan", type=Path, required=True)
    validation.add_argument("--prediction-root", type=Path, required=True)
    validation.add_argument("--calibration", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument(
        "--joint-run",
        type=_joint_arg,
        action="append",
        default=[],
        metavar="PLAN::PREDICTION_ROOT",
    )
    validation.add_argument("--allow-smoke", action="store_true")
    fit = commands.add_parser("diagnose-fit")
    fit.add_argument("--plan", type=Path, required=True)
    fit.add_argument("--prediction-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--allow-smoke", action="store_true")
    selection = commands.add_parser("select-recognizer")
    selection.add_argument(
        "--tiny-top1", type=_joint_arg, required=True, metavar="PLAN::CALIBRATION"
    )
    selection.add_argument(
        "--tiny-topk", type=_joint_arg, required=True, metavar="PLAN::CALIBRATION"
    )
    selection.add_argument(
        "--strong-topk", type=_joint_arg, metavar="PLAN::CALIBRATION"
    )
    selection.add_argument("--training-evidence", type=Path, required=True)
    selection.add_argument("--activation-decision", type=Path, required=True)
    selection.add_argument("--strong-qualification", type=Path)
    selection.add_argument("--output", type=Path, required=True)
    geometry = commands.add_parser("select-geometry")
    geometry.add_argument(
        "--base", type=_joint_arg, required=True, metavar="PLAN::CALIBRATION"
    )
    geometry.add_argument(
        "--fusion",
        type=_joint_arg,
        action="append",
        required=True,
        metavar="PLAN::CALIBRATION",
    )
    geometry.add_argument("--stage", choices=("screen", "final"), required=True)
    geometry.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "calibrate":
        output = calibrate(
            plan_path=args.plan,
            prediction_root=args.prediction_root,
            output_path=args.output,
            allow_smoke=args.allow_smoke,
        )
    elif args.command == "validate":
        output = validate(
            plan_path=args.plan,
            prediction_root=args.prediction_root,
            calibration_path=args.calibration,
            output_path=args.output,
            extra_joint_runs=args.joint_run,
            allow_smoke=args.allow_smoke,
        )
    elif args.command == "diagnose-fit":
        output = diagnose_algorithm_fit(
            plan_path=args.plan,
            prediction_root=args.prediction_root,
            output_path=args.output,
            allow_smoke=args.allow_smoke,
        )
    elif args.command == "select-recognizer":
        output = select_recognizer(
            tiny_top1=args.tiny_top1,
            tiny_topk=args.tiny_topk,
            strong_topk=args.strong_topk,
            training_evidence_path=args.training_evidence,
            activation_decision_path=args.activation_decision,
            strong_qualification_path=args.strong_qualification,
            output_path=args.output,
        )
    else:
        output = select_geometry(
            base=args.base,
            fusions=args.fusion,
            stage=args.stage,
            output_path=args.output,
        )
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CALIBRATION_PROTOCOL",
    "DBNET_GATE_RECALL",
    "FIT_DIAGNOSTIC_PROTOCOL",
    "PublicTruth",
    "VALIDATION_PROTOCOL",
    "detector_box_metrics",
    "score_full_auto_rows",
    "score_range_rows",
    "select_geometry",
    "select_recognizer",
    "select_threshold",
    "threshold_table",
    "range_variant_sha256",
]
