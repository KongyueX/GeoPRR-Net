import math

import cv2
import numpy as np


def estimate_pointer_tip(mask, center, threshold_ratio=0.10):
    """Estimate a pointer tip point from a binary mask.

    The baseline intentionally stays simple:
    1. Fit a line to the non-zero mask pixels.
    2. Reject masks whose main axis is too far from the dial center.
    3. Prefer pixels close to the fitted axis.
    4. Pick the farthest candidate from the dial center as the pointer tip.

    Returns a dict with status/message/tip_point and debug fields.
    """
    result = {
        "status": False,
        "message": "",
        "tip_point": None,
        "point_count": 0,
        "candidate_count": 0,
        "fit_line": None,
        "center_distance": None,
        "threshold": None,
        "band_threshold": None,
        "axis_score": 0.0,
        "support_ratio": 0.0,
        "direction_consistency": 0.0,
        "tip_support_ratio": 0.0,
        "confidence": 0.0,
    }

    if mask is None or center is None:
        result["message"] = "mask or center is None"
        return result

    if len(mask.shape) == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = mask

    points = cv2.findNonZero((gray_mask > 0).astype(np.uint8))
    if points is None or len(points) < 2:
        result["message"] = "mask has too few points"
        return result

    points_xy = points.reshape(-1, 2).astype(np.float32)
    result["point_count"] = int(points_xy.shape[0])

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    direction_norm = math.hypot(float(vx), float(vy))
    if direction_norm < 1e-6:
        result["message"] = "invalid fitted line"
        return result

    try:
        ratio = float(threshold_ratio)
    except (TypeError, ValueError):
        ratio = 0.10

    height, width = gray_mask.shape[:2]
    threshold = min(height, width) * max(0.0, ratio)
    band_threshold = max(2.0, threshold * 1.5)

    cx, cy = float(center[0]), float(center[1])
    center_distance = abs(float(vx) * (cy - float(y0)) - float(vy) * (cx - float(x0))) / direction_norm
    result["center_distance"] = float(center_distance)
    result["threshold"] = float(threshold)
    result["band_threshold"] = float(band_threshold)
    result["fit_line"] = {
        "vx": float(vx),
        "vy": float(vy),
        "x0": float(x0),
        "y0": float(y0),
    }

    if center_distance > threshold:
        result["message"] = f"mask axis too far from center: {center_distance:.2f} > {threshold:.2f}"
        return result

    perp_dist = np.abs(
        float(vx) * (points_xy[:, 1] - float(y0)) - float(vy) * (points_xy[:, 0] - float(x0))
    ) / direction_norm
    candidates = points_xy[perp_dist <= band_threshold]
    if candidates.size == 0:
        candidates = points_xy

    result["candidate_count"] = int(candidates.shape[0])

    center_vec = np.array([cx, cy], dtype=np.float32)
    distances = np.linalg.norm(candidates - center_vec, axis=1)
    tip_idx = int(np.argmax(distances))
    tip_point = candidates[tip_idx]
    tip_point = (int(round(float(tip_point[0]))), int(round(float(tip_point[1]))))
    support_ratio = float(result["candidate_count"]) / max(1.0, float(result["point_count"]))
    axis_score = max(0.0, 1.0 - center_distance / max(threshold, 1e-6))
    max_distance = float(np.max(distances))
    tip_support_ratio = float(np.mean(distances >= max_distance * 0.85))
    outer_cutoff = float(np.percentile(distances, 90))
    outer_rel = candidates[distances >= outer_cutoff] - center_vec
    outer_norms = np.linalg.norm(outer_rel, axis=1)
    valid_outer = outer_norms > 1e-6
    direction_consistency = 0.0
    if np.any(valid_outer) and max_distance > 1e-6:
        tip_direction = (candidates[tip_idx] - center_vec) / max_distance
        unit_outer = outer_rel[valid_outer] / outer_norms[valid_outer, None]
        alignment = np.clip(unit_outer @ tip_direction, 0.0, 1.0)
        direction_consistency = float(
            np.average(
                alignment,
                weights=np.maximum(outer_norms[valid_outer], 1.0),
            )
        )

    result.update(
        status=True,
        message="OK",
        tip_point=tip_point,
        axis_score=float(axis_score),
        support_ratio=float(support_ratio),
        direction_consistency=direction_consistency,
        tip_support_ratio=tip_support_ratio,
        confidence=float(
            np.clip(
                0.45 * axis_score
                + 0.15 * support_ratio
                + 0.25 * direction_consistency
                + 0.15 * min(1.0, tip_support_ratio * 10.0),
                0.0,
                1.0,
            )
        ),
    )
    return result


def estimate_pointer_tip_v2(mask, center, threshold_ratio=0.10):
    """Estimate a pointer tip with a more robust directional vote.

    Compared with ``estimate_pointer_tip`` this version does not trust a single
    farthest mask pixel. It fits the main axis, chooses the side that extends
    farther from the dial center, then averages the outer points on that side to
    suppress burrs, highlights, and one-pixel mask spikes.
    """
    result = {
        "status": False,
        "message": "",
        "tip_point": None,
        "point_count": 0,
        "candidate_count": 0,
        "outer_candidate_count": 0,
        "fit_line": None,
        "center_distance": None,
        "threshold": None,
        "band_threshold": None,
        "selected_side": None,
        "axis_score": 0.0,
        "support_ratio": 0.0,
        "vote_concentration": 0.0,
        "side_separation": 0.0,
        "confidence": 0.0,
    }

    if mask is None or center is None:
        result["message"] = "mask or center is None"
        return result

    if len(mask.shape) == 3:
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_mask = mask

    binary = (gray_mask > 0).astype(np.uint8)
    points = cv2.findNonZero(binary)
    if points is None or len(points) < 8:
        result["message"] = "mask has too few points"
        return result

    points_xy = points.reshape(-1, 2).astype(np.float32)
    result["point_count"] = int(points_xy.shape[0])

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    axis = np.array([float(vx), float(vy)], dtype=np.float32)
    direction_norm = float(np.linalg.norm(axis))
    if direction_norm < 1e-6:
        result["message"] = "invalid fitted line"
        return result
    axis /= direction_norm

    try:
        ratio = float(threshold_ratio)
    except (TypeError, ValueError):
        ratio = 0.10

    height, width = gray_mask.shape[:2]
    threshold = min(height, width) * max(0.0, ratio)
    band_threshold = max(2.0, threshold * 1.2)

    cx, cy = float(center[0]), float(center[1])
    center_vec = np.array([cx, cy], dtype=np.float32)
    center_distance = abs(float(vx) * (cy - float(y0)) - float(vy) * (cx - float(x0))) / direction_norm
    result["center_distance"] = float(center_distance)
    result["threshold"] = float(threshold)
    result["band_threshold"] = float(band_threshold)
    result["fit_line"] = {
        "vx": float(vx),
        "vy": float(vy),
        "x0": float(x0),
        "y0": float(y0),
    }

    if center_distance > threshold:
        result["message"] = f"mask axis too far from center: {center_distance:.2f} > {threshold:.2f}"
        return result

    rel = points_xy - center_vec
    distances = np.linalg.norm(rel, axis=1)
    projection = rel @ axis
    perp_dist = np.abs(
        float(vx) * (points_xy[:, 1] - float(y0)) - float(vy) * (points_xy[:, 0] - float(x0))
    ) / direction_norm
    on_axis = perp_dist <= band_threshold

    side_stats = []
    for side in (1.0, -1.0):
        side_mask = on_axis & ((projection * side) > 0)
        side_distances = distances[side_mask]
        if side_distances.size == 0:
            continue
        side_stats.append(
            {
                "side": side,
                "count": int(side_distances.size),
                "p90": float(np.percentile(side_distances, 90)),
                "p95": float(np.percentile(side_distances, 95)),
                "max": float(np.max(side_distances)),
            }
        )

    if not side_stats:
        result["message"] = "no valid points along fitted axis"
        return result

    side_stats.sort(key=lambda item: (item["p95"], item["p90"], item["count"]), reverse=True)
    selected_side = side_stats[0]["side"]
    result["selected_side"] = int(selected_side)
    if len(side_stats) > 1:
        side_separation = max(
            0.0,
            (side_stats[0]["p95"] - side_stats[1]["p95"])
            / max(side_stats[0]["p95"], 1e-6),
        )
    else:
        side_separation = 1.0

    candidate_mask = on_axis & ((projection * selected_side) > 0)
    candidate_points = points_xy[candidate_mask]
    candidate_distances = distances[candidate_mask]
    result["candidate_count"] = int(candidate_points.shape[0])
    if candidate_points.size == 0:
        result["message"] = "selected side has no candidates"
        return result

    outer_cutoff = max(
        float(np.percentile(candidate_distances, 70)),
        min(height, width) * 0.12,
    )
    outer_mask = candidate_distances >= outer_cutoff
    outer_points = candidate_points[outer_mask]
    outer_distances = candidate_distances[outer_mask]
    if outer_points.size == 0:
        outer_points = candidate_points
        outer_distances = candidate_distances
    result["outer_candidate_count"] = int(outer_points.shape[0])

    outer_rel = outer_points - center_vec
    outer_norms = np.linalg.norm(outer_rel, axis=1)
    valid = outer_norms > 1e-6
    if not np.any(valid):
        result["message"] = "outer candidates are too close to center"
        return result

    unit_vectors = outer_rel[valid] / outer_norms[valid, None]
    weights = np.maximum(outer_norms[valid], 1.0) ** 2
    direction = np.average(unit_vectors, axis=0, weights=weights)
    direction_norm = float(np.linalg.norm(direction))
    vote_concentration = direction_norm
    if direction_norm < 1e-6:
        farthest_idx = int(np.argmax(candidate_distances))
        direction = candidate_points[farthest_idx] - center_vec
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            result["message"] = "invalid voted direction"
            return result
    direction = direction / direction_norm

    tip_distance = float(np.percentile(outer_distances, 95))
    tip = center_vec + direction.astype(np.float32) * tip_distance
    tip_point = (int(round(float(tip[0]))), int(round(float(tip[1]))))

    support_ratio = float(result["outer_candidate_count"]) / max(1.0, float(result["point_count"]))
    axis_score = max(0.0, 1.0 - center_distance / max(threshold, 1e-6))
    result.update(
        status=True,
        message="OK",
        tip_point=tip_point,
        axis_score=float(axis_score),
        support_ratio=float(support_ratio),
        vote_concentration=float(np.clip(vote_concentration, 0.0, 1.0)),
        side_separation=float(np.clip(side_separation, 0.0, 1.0)),
        confidence=float(
            np.clip(
                0.35 * axis_score
                + 0.30 * np.clip(vote_concentration, 0.0, 1.0)
                + 0.20 * np.clip(side_separation, 0.0, 1.0)
                + 0.15 * min(1.0, support_ratio * 5.0),
                0.0,
                1.0,
            )
        ),
    )
    return result
