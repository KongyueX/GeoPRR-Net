"""Conditional material geometry fitted by source-pixel reprojection error.

The three starts diagnose local solution ambiguity; they do not certify global
uniqueness. Dial G and corresponding material XYZ are explicit inputs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from experiments.geoprr_v2_physical_geometry import source_to_world_plane


def _pixel_prediction(parameters, xyz, pixel_matrix):
    c, sine, tx, ty, ax, ay, b = parameters
    x, y, z = xyz.T
    material = np.column_stack((c * x - sine * y + tx - ax * z,
                                sine * x + c * y + ty - ay * z, 1 - b * z))
    homogeneous = material @ pixel_matrix.T
    denominator = homogeneous[:, 2]
    denominator = np.where(np.abs(denominator) < 1e-12, np.copysign(1e-12, denominator), denominator)
    prediction = homogeneous[:, :2] / denominator[:, None]
    zeros, ones = np.zeros(len(xyz)), np.ones(len(xyz))
    derivative = np.stack((np.column_stack((x, -y, ones, zeros, -z, zeros, zeros)),
                           np.column_stack((y, x, zeros, ones, zeros, -z, zeros)),
                           np.column_stack((zeros, zeros, zeros, zeros, zeros, zeros, -z))), axis=1)
    dh = np.einsum("ij,njk->nik", pixel_matrix, derivative)
    jacobian = (dh[:, :2] - prediction[:, :, None] * dh[:, 2, None]) / denominator[:, None, None]
    return prediction, jacobian.reshape(-1, 7)


def fit_height_geometry(material_xyz, pixels, bbox, matrix) -> dict:
    """Fit unknown camera nuisance and oriented material rotation without GT C or y."""
    xyz, pixels = np.asarray(material_xyz, dtype=np.float64), np.asarray(pixels, dtype=np.float64)
    bbox, matrix = np.asarray(bbox, dtype=np.float64), np.asarray(matrix, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or pixels.shape != (len(xyz), 2) or len(xyz) < 3:
        raise ValueError("expected at least three corresponding material XYZ and source pixels")
    if not all(np.isfinite(value).all() for value in (xyz, pixels, bbox, matrix)):
        raise ValueError("nonfinite geometry")
    if bbox.shape != (4,) or np.any(bbox[2:] <= bbox[:2]) or matrix.shape != (3, 3):
        raise ValueError("invalid integer-bbox geometry contract")
    plane = source_to_world_plane(pixels, bbox, matrix)
    crop_to_pixel = np.array([[bbox[2] - bbox[0], 0., bbox[0]],
                              [0., bbox[3] - bbox[1], bbox[1]], [0., 0., 1.]])
    world_to_canonical = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
    pixel_matrix = crop_to_pixel @ np.linalg.inv(matrix) @ world_to_canonical
    x, y, z = xyz.T
    zeros, ones = np.zeros(len(xyz)), np.ones(len(xyz))
    design = np.stack((np.column_stack((x, -y, ones, zeros, -z, zeros, plane[:, 0] * z)),
                       np.column_stack((y, x, zeros, ones, zeros, -z, plane[:, 1] * z))), axis=1)

    def linear_fit(a, target):
        norms = np.linalg.norm(a, axis=0)
        norms[norms == 0] = 1.
        return np.linalg.lstsq(a / norms, target, rcond=1e-12)[0] / norms

    linear = linear_fit(design.reshape(-1, 7), plane.reshape(-1))
    max_z = float(z.max())
    bmax = (1 - 1e-6) / max_z if max_z > 0 else 0.
    if bmax == 0 and np.any(z != 0):
        raise ValueError("this camera bound requires positive material height or z identically zero")
    starts = []
    for b in (0., bmax / 2, float(np.clip(linear[-1], 0., bmax))):
        if not any(abs(b - previous) <= 1e-12 * max(1., bmax) for previous in starts):
            starts.append(b)
    h = np.column_stack((plane, ones)) @ pixel_matrix.T
    local_pixel_jacobian = (pixel_matrix[None, :2, :2] - (h[:, :2] / h[:, 2, None])[:, :, None]
                            * pixel_matrix[None, 2, :2]) / h[:, 2, None, None]
    observed_centers = pixels + .5
    solutions = []
    for b in starts:
        denominator = 1 - b * z
        weight = local_pixel_jacobian / denominator[:, None, None]
        a = np.einsum("nij,njk->nik", weight, design[:, :, :6]).reshape(-1, 6)
        target = np.einsum("nij,nj->ni", weight, plane * denominator[:, None]).reshape(-1)
        initial = np.append(linear_fit(a, target), b)
        active = 7 if bmax > 0 else 6

        def expand(values):
            return values if active == 7 else np.append(values, 0.)

        def residual(values):
            return (_pixel_prediction(expand(values), xyz, pixel_matrix)[0] - observed_centers).reshape(-1)

        def jacobian(values):
            return _pixel_prediction(expand(values), xyz, pixel_matrix)[1][:, :active]

        lower, upper = np.full(active, -np.inf), np.full(active, np.inf)
        if active == 7:
            lower[-1], upper[-1] = 0., bmax
        fit = least_squares(residual, initial[:active], jac=jacobian, bounds=(lower, upper),
                            x_scale="jac", max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-10)
        parameters = expand(fit.x)
        errors = residual(fit.x).reshape(-1, 2)
        fit_jacobian = jacobian(fit.x)
        singular = np.linalg.svd(fit_jacobian / np.maximum(np.linalg.norm(fit_jacobian, axis=0), 1e-12), compute_uv=False)
        solutions.append({"start_inverse_camera_z": b, "success": bool(fit.success), "nfev": int(fit.nfev),
                          "parameters": parameters.tolist(), "theta_radians": float(np.arctan2(parameters[1], parameters[0])),
                          "positive_scale": float(np.hypot(*parameters[:2])), "pixel_sse": float(np.sum(errors ** 2)),
                          "pixel_rmse": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
                          "pixel_max": float(np.linalg.norm(errors, axis=1).max()),
                          "normalized_pixel_jacobian_rank": int(np.sum(singular > singular[0] * 1e-12))})
    best = min(solutions, key=lambda item: item["pixel_sse"])
    equal_cost = [item for item in solutions if item["pixel_sse"] <= best["pixel_sse"] + 1e-8]
    ambiguity = max(abs((item["theta_radians"] - best["theta_radians"] + np.pi) % (2 * np.pi) - np.pi) for item in equal_cost)
    parameters = best["parameters"]
    return {"theta_radians": best["theta_radians"], "positive_scale": best["positive_scale"],
            "translation": parameters[2:4], "camera_xy_over_z": parameters[4:6], "camera_inverse_z": parameters[6],
            "camera_denominator_min": float(np.min(1 - parameters[6] * z)),
            "pixel_sse": best["pixel_sse"], "pixel_rmse": best["pixel_rmse"], "pixel_max": best["pixel_max"],
            "success": best["success"], "starts": solutions,
            "same_cost_rotation_spread_radians": float(ambiguity), "readout_rotation_ambiguous": bool(ambiguity > 1e-4),
            "same_cost_pixel_sse_tolerance": 1e-8, "camera_constraints_applied": True}
