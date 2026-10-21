"""Joint tick and height-aware material fitting in source-pixel coordinates."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from experiments.geoprr_v2_height_profile import fit_height_geometry
from experiments.geoprr_v2_tick_calibration import _canonical, _dlt, fit_ordered_tick_calibration


TAU = 2 * np.pi
PROJECTION_EPS = 1e-12


def _matrix(parameters):
    vector = parameters[6:8]
    norm = np.sqrt(1 + vector @ vector)
    bottom = vector / norm
    derivative = np.eye(2) / norm - np.outer(vector, vector) / norm**3
    return np.array([parameters[:3], parameters[3:6], [*bottom, 1.]]), derivative


def _projection(parameters, points):
    matrix, bottom_derivative = _matrix(parameters)
    mapped = points @ matrix.T
    depth = mapped[:, 2]
    divisor = np.where(np.abs(depth) < PROJECTION_EPS, np.copysign(PROJECTION_EPS, depth), depth)
    prediction = mapped[:, :2] / divisor[:, None]
    jacobian = np.zeros((len(points), 2, 8))
    jacobian[:, 0, :3] = points / divisor[:, None]
    jacobian[:, 1, 3:6] = points / divisor[:, None]
    jacobian[:, :, 6:8] = (-prediction[:, :, None]
                           * (points[:, :2] @ bottom_derivative)[:, None] / divisor[:, None, None])
    point_derivative = (matrix[None, :2] - prediction[:, :, None] * matrix[None, 2]) / divisor[:, None, None]
    return prediction, jacobian, point_derivative, depth


def _joint_prediction(parameters, xyz, count):
    span, body = parameters[8], parameters[9:]
    fractions = np.arange(count) / (count - 1)
    angles = -fractions * span
    ticks = _canonical(count, span)
    c, sine, tx, ty, ax, ay, b = body
    x, y, z = xyz.T
    material = np.column_stack((c*x - sine*y + tx - ax*z,
                                sine*x + c*y + ty - ay*z, 1 - b*z))
    prediction, homography_jacobian, point_jacobian, depth = _projection(parameters, np.vstack((ticks, material)))
    jacobian = np.zeros((count + len(xyz), 2, 16))
    jacobian[:, :, :8] = homography_jacobian
    tick_derivative = np.column_stack((fractions*np.sin(angles), -fractions*np.cos(angles), np.zeros(count)))
    jacobian[:count, :, 8] = np.einsum("nij,nj->ni", point_jacobian[:count], tick_derivative)
    zeros, ones = np.zeros(len(xyz)), np.ones(len(xyz))
    body_derivative = np.stack((np.column_stack((x, -y, ones, zeros, -z, zeros, zeros)),
                                np.column_stack((y, x, zeros, ones, zeros, -z, zeros)),
                                np.column_stack((zeros, zeros, zeros, zeros, zeros, zeros, -z))), axis=1)
    jacobian[count:, :, 9:] = np.einsum("nij,njk->nik", point_jacobian[count:], body_derivative)
    return prediction, jacobian.reshape(-1, 16), depth


def _homography_parameters(matrix):
    matrix = matrix / matrix[2, 2]
    bottom = matrix[2, :2]
    if bottom @ bottom >= 1:
        raise ValueError("homography does not keep the full circle in front")
    return np.r_[matrix[:2].ravel(), bottom / np.sqrt(1 - bottom @ bottom)]


def _condition(jacobian):
    norms = np.maximum(np.linalg.norm(jacobian, axis=0), 1e-12)
    singular = np.linalg.svd(jacobian / norms, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-12)) if singular[0] > 0 else 0
    return rank, float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")


def _projection_failure_reason(depth):
    if not np.isfinite(depth).all():
        return "nonfinite_projection_depth"
    if np.any(np.abs(depth) < PROJECTION_EPS):
        return "final_projection_requires_numerical_clamp"
    return "nonpositive_projection_depth" if np.any(depth <= 0) else None


def fit_joint_geometry(material_xyz, material_pixels, ordered_tick_pixels) -> dict:
    """Fit one shared image model without camera, homography, span or reading inputs.

    Corresponding material vertices retain their own heights. Tick order contains
    consecutive identities and both scale endpoints. Inputs are raw annotation
    pixels; every residual uses pixel centers and the same source-pixel unit.
    """
    xyz, pixels, ticks = (np.asarray(value, dtype=np.float64)
                          for value in (material_xyz, material_pixels, ordered_tick_pixels))
    failure = {"status": "failed", "success": False, "prediction": None, "raw_prediction": None}
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or pixels.shape != (len(xyz), 2):
        return {**failure, "reason": "requires_corresponding_material_xyz_and_pixels"}
    if ticks.ndim != 2 or ticks.shape[1] != 2 or len(ticks) < 5:
        return {**failure, "reason": "requires_at_least_five_ordered_ticks"}
    if not all(np.isfinite(value).all() for value in (xyz, pixels, ticks)):
        return {**failure, "reason": "nonfinite_observations"}
    max_z = float(xyz[:, 2].max())
    if max_z <= 0 or np.any(xyz[:, 2] < 0):
        return {**failure, "reason": "requires_nonnegative_material_heights_with_positive_maximum"}
    bmax = (1 - 1e-6) / max_z
    centers = ticks + .5
    origin = centers.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((centers - origin)**2, axis=1))))
    if radius == 0:
        return {**failure, "reason": "coincident_tick_pixels"}
    scale = np.sqrt(2) / radius
    transform = np.array([[scale, 0., -scale*origin[0]], [0., scale, -scale*origin[1]], [0., 0., 1.]])
    target = (np.vstack((ticks, pixels)) + .5 - origin) * scale
    serial = fit_ordered_tick_calibration(ticks)
    starts = [("serial", serial.get("span_radians"), serial.get("H"))]
    normalized_ticks = (centers - origin) * scale
    for fraction in (.2, .5, .8):
        span = fraction * TAU
        initial = _dlt(normalized_ticks, span)
        if initial is None:
            starts.append((f"span_{fraction}", span, None))
            continue
        try:
            initial_h = _homography_parameters(initial[0])
            canonical = _canonical(len(ticks), span)
            refined = least_squares(
                lambda values: ((_projection(values, canonical)[0] - normalized_ticks) / scale).ravel(),
                initial_h, jac=lambda values: _projection(values, canonical)[1].reshape(-1, 8) / scale,
                x_scale="jac", max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            starts.append((f"span_{fraction}", span, np.linalg.solve(transform, _matrix(refined.x)[0])))
        except (ValueError, np.linalg.LinAlgError):
            starts.append((f"span_{fraction}", span, None))
    source_centers = np.vstack((ticks, pixels)) + .5
    low, high = source_centers.min(axis=0) - 1, source_centers.max(axis=0) + 1
    bbox = np.r_[low, high]
    crop = np.array([[high[0]-low[0], 0., low[0]], [0., high[1]-low[1], low[1]], [0., 0., 1.]])
    plane_to_canonical = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
    lower = np.r_[np.full(8, -np.inf), np.finfo(float).eps, np.full(6, -np.inf), 0.]
    upper = np.r_[np.full(8, np.inf), np.nextafter(TAU, 0.), np.full(6, np.inf), bmax]
    solutions = []
    for label, span, homography in starts:
        if homography is None:
            solutions.append({"start": label, "status": "failed", "reason": "no_physical_tick_initialization"})
            continue
        try:
            bridge = plane_to_canonical @ np.linalg.inv(homography) @ crop
            fitted_body = fit_height_geometry(xyz, pixels, bbox, bridge)
            body = np.r_[fitted_body["positive_scale"] * np.array([np.cos(fitted_body["theta_radians"]),
                                                                 np.sin(fitted_body["theta_radians"])]),
                         fitted_body["translation"], fitted_body["camera_xy_over_z"], fitted_body["camera_inverse_z"]]
            initial = np.r_[_homography_parameters(transform @ homography), span, body]

            def residual(values):
                return ((_joint_prediction(values, xyz, len(ticks))[0] - target) / scale).ravel()

            def jacobian(values):
                return _joint_prediction(values, xyz, len(ticks))[1] / scale

            result = least_squares(residual, initial, jac=jacobian, bounds=(lower, upper),
                                   x_scale="jac", max_nfev=300, ftol=1e-11, xtol=1e-11, gtol=1e-10)
            matrix, _ = _matrix(result.x)
            source_h = np.linalg.solve(transform, matrix)
            _, _, depth = _joint_prediction(result.x, xyz, len(ticks))
            errors = residual(result.x).reshape(-1, 2)
            span, c, sine = result.x[8:11]
            theta = float(np.arctan2(sine, c))
            delta = float((-theta) % TAU)
            unwrapped = float((delta - span/2 + np.pi) % TAU - np.pi + span/2)
            raw = unwrapped / span
            denominator = 1 - result.x[-1] * xyz[:, 2]
            rank, condition = _condition(jacobian(result.x))
            projection_failure = _projection_failure_reason(depth)
            valid = (result.success and 0 < span < TAU and np.isfinite(errors).all() and projection_failure is None
                     and np.all(denominator > 0) and np.hypot(c, sine) > 0
                     and np.isfinite(raw) and np.linalg.matrix_rank(source_h) == 3)
            tick_sse = float(np.sum(errors[:len(ticks)]**2))
            body_sse = float(np.sum(errors[len(ticks):]**2))
            solutions.append({"start": label, "initial_span_radians": float(initial[8]),
                              "status": "ok" if valid else "failed", "success": bool(valid),
                              "reason": None if valid else projection_failure or "optimizer_or_physical_constraint_failed",
                              "optimizer_success": bool(result.success), "optimizer_message": result.message,
                              "nfev": int(result.nfev), "parameters": result.x.tolist(), "H": source_h.tolist(),
                              "span_radians": float(span), "theta_radians": theta, "delta_radians": delta,
                              "raw_prediction": float(raw), "prediction": float(np.clip(raw, 0., 1.)),
                              "pixel_sse": tick_sse + body_sse, "tick_pixel_sse": tick_sse,
                              "material_pixel_sse": body_sse, "tick_pixel_rmse": float(np.sqrt(tick_sse/len(ticks))),
                              "material_pixel_rmse": float(np.sqrt(body_sse/len(xyz))),
                              "pixel_rmse": float(np.sqrt((tick_sse+body_sse)/(len(ticks)+len(xyz)))),
                              "pixel_max": float(np.linalg.norm(errors, axis=1).max()),
                              "front_circle_margin": float(1-np.linalg.norm(matrix[2, :2])),
                              "camera_denominator_min": float(denominator.min()), "projection_depth_min": float(depth.min()),
                              "positive_scale": float(np.hypot(c, sine)), "homography_condition": float(np.linalg.cond(source_h)),
                              "normalized_jacobian_rank": rank, "normalized_jacobian_condition": condition})
        except (ValueError, np.linalg.LinAlgError) as error:
            solutions.append({"start": label, "status": "failed", "reason": str(error)})
    valid = [item for item in solutions if item["status"] == "ok"]
    if not valid:
        return {**failure, "reason": "no_valid_joint_solution", "starts": solutions}
    best = min(valid, key=lambda item: item["pixel_sse"])
    same = [item for item in valid if item["pixel_sse"] <= best["pixel_sse"] + 1e-8]
    rotation_spread = max(abs((item["theta_radians"]-best["theta_radians"]+np.pi) % TAU-np.pi) for item in same)
    return {**best, "selected_start": best["start"], "starts": solutions, "tick_count": len(ticks),
            "material_count": len(xyz), "pixel_center_offset": .5,
            "same_cost_sse_tolerance": 1e-8, "same_cost_rotation_spread_radians": float(rotation_spread),
            "same_cost_prediction_spread": float(max(item["prediction"] for item in same)-min(item["prediction"] for item in same)),
            "same_cost_raw_prediction_spread": float(max(item["raw_prediction"] for item in same)
                                                     - min(item["raw_prediction"] for item in same))}
