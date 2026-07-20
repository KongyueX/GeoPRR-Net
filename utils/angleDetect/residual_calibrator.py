# -*- coding: utf-8 -*-
"""Optional residual calibrator for geometry_fusion readings.

The calibrator uses only features that are available during normal inference.
GT-derived fields are intentionally excluded.
"""
import math
from pathlib import Path

import cv2
import numpy as np


FEATURE_COLUMNS = [
    "p_geom",
    "p_geom_v2",
    "p_fusion",
    "v1_v2_result_delta",
    "endNum",
    "mask_point_count",
    "mask_candidate_count",
    "mask_center_distance",
    "mask_axis_threshold",
    "startAngle",
    "endAngle",
    "disAngle",
    "ellipse_ratio",
    "ellipse_angle",
    "ellipse_area_ratio",
]

SELECTIVE_FEATURE_COLUMNS = [
    "p_geom",
    "p_geom_v2",
    "p_fusion",
    "v1_v2_progress_delta",
    "endNum",
    "v1_confidence",
    "v2_confidence",
    "fusion_weight_v1",
    "fusion_weight_v2",
    "v1_axis_score",
    "v1_support_ratio",
    "v1_direction_consistency",
    "v1_tip_support_ratio",
    "v2_axis_score",
    "v2_support_ratio",
    "v2_vote_concentration",
    "v2_side_separation",
    "mask_component_area_ratio",
    "mask_candidate_ratio",
    "mask_center_distance_ratio",
    "mask_axis_threshold_ratio",
    "seg_probability_max",
    "seg_probability_p99",
    "seg_probability_mean",
    "seg_foreground_ratio",
    "startAngle",
    "endAngle",
    "disAngle",
    "ellipse_ratio",
    "ellipse_angle",
    "ellipse_area_ratio",
]


def as_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def fit_ellipse_proxy(crop_bgr):
    if crop_bgr is None:
        return {}
    h, w = crop_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return {}
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    v_thr = np.percentile(val, 55)
    mask = ((val >= v_thr) & (sat <= 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {}
    contour = max(cnts, key=cv2.contourArea)
    if len(contour) < 20:
        return {}
    ellipse = cv2.fitEllipse(contour)
    (_, _), (d1, d2), angle = ellipse
    major = max(float(d1), float(d2))
    minor = min(float(d1), float(d2))
    if major <= 1e-6:
        return {}
    return {
        "ellipse_ratio": minor / major,
        "ellipse_angle": float(angle),
        "ellipse_area_ratio": float(math.pi * (d1 / 2.0) * (d2 / 2.0) / max(1.0, h * w)),
    }


def make_features(row, feature_columns=None):
    feature_columns = feature_columns or FEATURE_COLUMNS
    values = []
    for column in feature_columns:
        value = as_float(row.get(column), 0.0)
        if column == "endNum":
            value /= 100.0
        elif column in ("mask_point_count", "mask_candidate_count"):
            value /= 2000.0
        elif column in ("mask_center_distance", "mask_axis_threshold"):
            value /= 200.0
        elif column in ("startAngle", "endAngle", "disAngle", "ellipse_angle"):
            value /= 360.0
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def make_calibrator_row(
    geometry_reading,
    geometry_v2_reading,
    geometry_fusion_reading,
    training_artifacts,
    corrected_crop_bgr=None,
    scale_start=0.0,
    scale_end=1.6,
):
    geometry_reading = geometry_reading or {}
    geometry_v2_reading = geometry_v2_reading or {}
    geometry_fusion_reading = geometry_fusion_reading or {}
    training_artifacts = training_artifacts or {}
    tip_info = geometry_reading.get("tip_info") or {}
    tip_info_v2 = geometry_v2_reading.get("tip_info") or {}
    segmentation_summary = training_artifacts.get("segmentation_summary") or {}
    ellipse_proxy = fit_ellipse_proxy(corrected_crop_bgr)
    mask_height = 0
    mask_width = 0
    pointer_mask = training_artifacts.get("pointer_mask")
    if pointer_mask is not None and getattr(pointer_mask, "ndim", 0) >= 2:
        mask_height, mask_width = pointer_mask.shape[:2]
    elif corrected_crop_bgr is not None:
        mask_height, mask_width = corrected_crop_bgr.shape[:2]
    mask_area = max(1.0, float(mask_height * mask_width))
    mask_short_side = max(1.0, float(min(mask_height, mask_width)))
    point_count = as_float(tip_info.get("point_count"), 0.0)
    candidate_count = as_float(tip_info.get("candidate_count"), 0.0)

    p_geom = geometry_reading.get("progress_ratio")
    p_geom_v2 = geometry_v2_reading.get("progress_ratio")
    p_geom = as_float(p_geom, as_float(geometry_fusion_reading.get("progress_ratio"), 0.0))
    p_geom_v2 = as_float(p_geom_v2, p_geom)
    p_fusion = as_float(geometry_fusion_reading.get("progress_ratio"), (p_geom + p_geom_v2) / 2.0)
    v1_v2_progress_delta = abs(p_geom - p_geom_v2)
    result_v1 = geometry_reading.get("resultNum")
    result_v2 = geometry_v2_reading.get("resultNum")
    v1_v2_delta = 0.0
    if result_v1 is not None and result_v2 is not None:
        v1_v2_delta = abs(as_float(result_v1) - as_float(result_v2))
    fusion_weights = geometry_fusion_reading.get("fusion_source_weights") or {}
    v1_confidence = as_float(
        geometry_reading.get("confidence"),
        as_float((geometry_reading.get("tip_info") or {}).get("confidence"), 0.0),
    )
    v2_confidence = as_float(
        geometry_v2_reading.get("confidence"),
        as_float((geometry_v2_reading.get("tip_info") or {}).get("confidence"), 0.0),
    )

    return {
        "scaleStart": as_float(scale_start, 0.0),
        "scaleEnd": as_float(scale_end, 1.6),
        "p_geom": p_geom,
        "p_geom_v2": p_geom_v2,
        "p_fusion": p_fusion,
        "v1_v2_result_delta": v1_v2_delta,
        "v1_v2_progress_delta": v1_v2_progress_delta,
        "v1_confidence": v1_confidence,
        "v2_confidence": v2_confidence,
        "fusion_weight_v1": as_float(fusion_weights.get("geometry_direct"), 0.5),
        "fusion_weight_v2": as_float(fusion_weights.get("geometry_direct_v2"), 0.5),
        "v1_axis_score": tip_info.get("axis_score"),
        "v1_support_ratio": tip_info.get("support_ratio"),
        "v1_direction_consistency": tip_info.get("direction_consistency"),
        "v1_tip_support_ratio": tip_info.get("tip_support_ratio"),
        "v2_axis_score": tip_info_v2.get("axis_score"),
        "v2_support_ratio": tip_info_v2.get("support_ratio"),
        "v2_vote_concentration": tip_info_v2.get("vote_concentration"),
        "v2_side_separation": tip_info_v2.get("side_separation"),
        "endNum": geometry_reading.get("endNum"),
        "mask_point_count": tip_info.get("point_count"),
        "mask_candidate_count": tip_info.get("candidate_count"),
        "mask_center_distance": tip_info.get("center_distance"),
        "mask_axis_threshold": tip_info.get("threshold"),
        "mask_component_area_ratio": point_count / mask_area,
        "mask_candidate_ratio": candidate_count / max(1.0, point_count),
        "mask_center_distance_ratio": (
            as_float(tip_info.get("center_distance"), 0.0) / mask_short_side
        ),
        "mask_axis_threshold_ratio": (
            as_float(tip_info.get("threshold"), 0.0) / mask_short_side
        ),
        "seg_probability_max": segmentation_summary.get("probability_max"),
        "seg_probability_p99": segmentation_summary.get("probability_p99"),
        "seg_probability_mean": segmentation_summary.get("probability_mean"),
        "seg_foreground_ratio": segmentation_summary.get("foreground_ratio"),
        "startAngle": training_artifacts.get("startAngle"),
        "endAngle": training_artifacts.get("endAngle"),
        "disAngle": training_artifacts.get("disAngle"),
        "ellipse_ratio": ellipse_proxy.get("ellipse_ratio"),
        "ellipse_angle": ellipse_proxy.get("ellipse_angle"),
        "ellipse_area_ratio": ellipse_proxy.get("ellipse_area_ratio"),
    }



def predict_residual_with_uncertainty(model, x):
    prediction = float(model.predict(x)[0])
    tree_predictions = []
    fitted_named_estimators = getattr(model, "named_estimators_", None)
    if fitted_named_estimators is not None:
        for estimator in fitted_named_estimators.values():
            if hasattr(estimator, "predict"):
                tree_predictions.append(float(estimator.predict(x)[0]))
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and not tree_predictions:
        flat_estimators = np.asarray(estimators, dtype=object).reshape(-1)
        for estimator in flat_estimators:
            if hasattr(estimator, "predict"):
                tree_predictions.append(float(estimator.predict(x)[0]))
    if tree_predictions:
        return prediction, float(np.std(tree_predictions)), int(len(tree_predictions))
    return prediction, None, 0


def build_base_fallback_reading(
    base_reading,
    reason,
    residual=None,
    residual_std=None,
    tree_count=0,
    package=None,
    backend_name=None,
):
    package = package or {}
    learned_gate = package.get("_last_learned_gate") or {}
    backend_name = backend_name or package.get("backend_name") or "geometry_fusion_calibrated"
    fallback = dict(base_reading)
    fallback.update(
        backend=backend_name,
        calibration={
            "applied": False,
            "reason": reason,
            "residual": residual,
            "residual_std": residual_std,
            "residual_unit": package.get("residual_unit", "reading"),
            "tree_count": tree_count,
            "base_backend": base_reading.get("backend"),
            "base_resultNum": base_reading.get("resultNum"),
            "min_abs_residual": package.get("min_abs_residual"),
            "boundary_margin": package.get("boundary_margin"),
            "max_abs_residual": package.get("max_abs_residual"),
            "max_residual_std": package.get("max_residual_std"),
            "learned_gate_probability": learned_gate.get("probability"),
            "learned_gate_threshold": learned_gate.get("threshold"),
            "learned_gate_model_type": package.get("gate_model_type"),
        },
    )
    return fallback


def make_learned_gate_features(feature_row, feature_columns, residual, residual_std):
    values = list(make_features(feature_row, feature_columns))
    residual_std = 0.0 if residual_std is None else float(residual_std)
    values.extend([float(residual), abs(float(residual)), residual_std])
    return np.asarray(values, dtype=np.float32)


def gate_probability(model, x):
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(x)[0], dtype=np.float64).reshape(-1)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return float(probabilities[classes.index(1)])
        if len(classes) == 1:
            return 1.0 if classes[0] in (1, True, "1") else 0.0
        if probabilities.size:
            return float(probabilities[-1])
        return None
    return float(model.predict(x)[0])


def load_calibrator(path):
    import joblib

    package = joblib.load(Path(path))
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("invalid residual calibrator package")
    package.setdefault("feature_columns", FEATURE_COLUMNS)
    package.setdefault("residual_clip", 0.08)
    package.setdefault("min_abs_residual", None)
    package.setdefault("boundary_margin", 0.0)
    package.setdefault("gate_apply_threshold", 0.5)
    package.setdefault("residual_unit", "reading")
    package.setdefault("_last_learned_gate", None)
    return package


def apply_calibrator(package, feature_row, base_reading, backend_name=None):
    if not package or not base_reading or not base_reading.get("status"):
        return None
    model = package.get("model")
    if model is None:
        return None
    # This field is inference-local diagnostic state. Clear it before any
    # early uncertainty/size guard so a fallback cannot report the previous
    # sample's learned-gate probability.
    package["_last_learned_gate"] = None

    feature_columns = package.get("feature_columns") or FEATURE_COLUMNS
    x = make_features(feature_row, feature_columns).reshape(1, -1)
    residual_model, residual_std_model, tree_count = predict_residual_with_uncertainty(model, x)
    backend_name = backend_name or package.get("backend_name") or "geometry_fusion_calibrated"

    min_abs_residual = package.get("min_abs_residual")
    if min_abs_residual is not None and abs(residual_model) < abs(float(min_abs_residual)):
        return build_base_fallback_reading(
            base_reading,
            "residual_min_abs_gate",
            residual=residual_model,
            residual_std=residual_std_model,
            tree_count=tree_count,
            package=package,
            backend_name=backend_name,
        )

    max_abs_residual = package.get("max_abs_residual")
    if max_abs_residual is not None and abs(residual_model) > abs(float(max_abs_residual)):
        return build_base_fallback_reading(
            base_reading,
            "residual_abs_gate",
            residual=residual_model,
            residual_std=residual_std_model,
            tree_count=tree_count,
            package=package,
            backend_name=backend_name,
        )

    max_residual_std = package.get("max_residual_std")
    if residual_std_model is not None and max_residual_std is not None and residual_std_model > float(max_residual_std):
        return build_base_fallback_reading(
            base_reading,
            "residual_std_gate",
            residual=residual_model,
            residual_std=residual_std_model,
            tree_count=tree_count,
            package=package,
            backend_name=backend_name,
        )

    residual_clip = package.get("residual_clip")
    if residual_clip is not None:
        residual_model = float(
            np.clip(
                residual_model,
                -abs(float(residual_clip)),
                abs(float(residual_clip)),
            )
        )

    gate_model = package.get("gate_model")
    if gate_model is not None:
        gate_x = make_learned_gate_features(
            feature_row,
            feature_columns,
            residual_model,
            residual_std_model,
        ).reshape(1, -1)
        probability = gate_probability(gate_model, gate_x)
        threshold = as_float(package.get("gate_apply_threshold"), 0.5)
        package["_last_learned_gate"] = {
            "probability": probability,
            "threshold": threshold,
        }
        if probability is None or probability < threshold:
            return build_base_fallback_reading(
                base_reading,
                "learned_gate_reject",
                residual=residual_model,
                residual_std=residual_std_model,
                tree_count=tree_count,
                package=package,
                backend_name=backend_name,
            )
    else:
        package["_last_learned_gate"] = None

    scale_start = as_float(feature_row.get("scaleStart"), 0.0)
    scale_end = as_float(feature_row.get("scaleEnd"), 1.6)
    base_result = as_float(base_reading.get("resultNum"), 0.0)

    denom = scale_end - scale_start
    residual_unit = str(package.get("residual_unit") or "reading").strip().lower()
    if residual_unit == "normalized_range":
        residual = residual_model * denom
        residual_std = None if residual_std_model is None else abs(denom) * residual_std_model
    else:
        residual = residual_model
        residual_std = residual_std_model

    boundary_margin = as_float(package.get("boundary_margin"), 0.0)
    if boundary_margin > 0:
        low_scale = min(scale_start, scale_end)
        high_scale = max(scale_start, scale_end)
        if base_result <= low_scale + boundary_margin and residual < 0:
            return build_base_fallback_reading(
                base_reading,
                "low_boundary_guard",
                residual=residual_model,
                residual_std=residual_std_model,
                tree_count=tree_count,
                package=package,
                backend_name=backend_name,
            )
        if base_result >= high_scale - boundary_margin and residual > 0:
            return build_base_fallback_reading(
                base_reading,
                "high_boundary_guard",
                residual=residual_model,
                residual_std=residual_std_model,
                tree_count=tree_count,
                package=package,
                backend_name=backend_name,
            )

    result_num = base_result + residual
    low_scale = min(scale_start, scale_end)
    high_scale = max(scale_start, scale_end)
    result_num = float(np.clip(result_num, low_scale, high_scale))

    if abs(denom) > 1e-9:
        progress_ratio = float(np.clip((result_num - scale_start) / denom, 0.0, 1.0))
    else:
        progress_ratio = as_float(base_reading.get("progress_ratio"), 0.0)
    end_num_float = progress_ratio * 100.0

    calibrated = dict(base_reading)
    calibrated.update(
        status=True,
        backend=backend_name,
        message=f"geometry fusion calibrated residual={residual:.4f}, result={result_num:.4f}",
        resultNum=result_num,
        progress_ratio=progress_ratio,
        endNum_float=end_num_float,
        endNum=int(round(min(max(end_num_float, 0.0), 100.0))),
        calibration={
            "applied": True,
            "reason": "applied",
            "model_type": package.get("model_type"),
            "feature_columns": feature_columns,
            "residual": residual,
            "residual_std": residual_std,
            "residual_model": residual_model,
            "residual_std_model": residual_std_model,
            "residual_unit": residual_unit,
            "tree_count": tree_count,
            "residual_clip": residual_clip,
            "min_abs_residual": package.get("min_abs_residual"),
            "boundary_margin": package.get("boundary_margin"),
            "max_abs_residual": package.get("max_abs_residual"),
            "max_residual_std": package.get("max_residual_std"),
            "base_backend": base_reading.get("backend"),
            "base_resultNum": base_result,
            "learned_gate_probability": (package.get("_last_learned_gate") or {}).get("probability"),
            "learned_gate_threshold": (package.get("_last_learned_gate") or {}).get("threshold"),
            "learned_gate_model_type": package.get("gate_model_type"),
        },
    )
    return calibrated

