"""Fit metric circle coordinates from ordered, equally angular tick pixels."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


TAU = 2 * np.pi


def _canonical(count: int, span: float) -> np.ndarray:
    angles = -np.arange(count) * span / (count - 1)
    return np.column_stack((np.cos(angles), np.sin(angles), np.ones(count)))


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    mapped = points @ matrix.T
    return mapped[:, :2] / mapped[:, 2, None]


def _cross_ratio_start(points: np.ndarray) -> dict:
    count = len(points)
    stride = (count - 1) // 4
    indices = np.arange(4) * stride
    remaining = np.array([i for i in range(count) if i not in indices])
    distances = np.linalg.norm(points[remaining, None] - points[indices], axis=2)
    reference = int(remaining[np.argmax(distances.min(axis=1))])
    vectors = points[indices] - points[reference]
    determinant = lambda a, b: a[0] * b[1] - a[1] * b[0]
    denominator = determinant(vectors[0], vectors[3]) * determinant(vectors[1], vectors[2])
    result = {"stride": stride, "indices": indices.tolist(), "reference": reference,
              "cross_ratio": None, "span_radians": None}
    if denominator == 0:
        return result
    ratio = determinant(vectors[0], vectors[2]) * determinant(vectors[1], vectors[3]) / denominator
    result["cross_ratio"] = float(ratio)
    if not np.isfinite(ratio) or ratio == 1:
        return result
    square = ratio / (4 * (ratio - 1))
    if not 0 < square < 1:
        return result
    span = 2 * np.arccos(np.sqrt(square)) * (count - 1) / stride
    if 0 < span < TAU:
        result["span_radians"] = float(span)
    return result


def _dlt(points: np.ndarray, span: float) -> tuple | None:
    source = _canonical(len(points), span)
    center = source[:, :2].mean(axis=0)
    scale = np.sqrt(2 / np.mean(np.sum((source[:, :2] - center) ** 2, axis=1)))
    transform = np.array([[scale, 0, -scale*center[0]],
                          [0, scale, -scale*center[1]], [0, 0, 1.]])
    source = source @ transform.T
    rows = []
    for (u, v, _), (x, y) in zip(source, points):
        rows.extend(([u, v, 1, 0, 0, 0, -x*u, -x*v, -x],
                     [0, 0, 0, u, v, 1, -y*u, -y*v, -y]))
    _, singular, vectors = np.linalg.svd(np.asarray(rows), full_matrices=False)
    matrix = vectors[-1].reshape(3, 3) @ transform
    if singular[-2] <= np.finfo(float).eps * singular[0] * len(rows):
        return None
    if np.linalg.norm(matrix[2, :2]) >= abs(matrix[2, 2]):
        return None
    matrix /= matrix[2, 2]
    return matrix, float(singular[0] / singular[-2])


def _unpack(parameters: np.ndarray) -> tuple[np.ndarray, float]:
    vector = parameters[6:8]
    bottom = vector / np.sqrt(1 + vector @ vector)
    matrix = np.array([parameters[:3], parameters[3:6], [*bottom, 1.]])
    return matrix, float(parameters[8])


def fit_ordered_tick_calibration(pixels: np.ndarray) -> dict:
    """Accept raw annotation pixels; return G mapping pixel centers to unit-circle coordinates.

    Input order must contain consecutive physical tick identities, including the range
    endpoints. No per-image angle, camera, target reading or homography is an input.
    All candidates use full-N pixel reprojection error; this is a geometric fit.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    count = len(pixels) if pixels.ndim else 0
    failure = {"status": "failed", "tick_count": count, "G": None, "H": None}
    if pixels.ndim != 2 or pixels.shape[1] != 2 or count < 5:
        return {**failure, "reason": "requires_at_least_five_ordered_2d_ticks"}
    if not np.isfinite(pixels).all():
        return {**failure, "reason": "nonfinite_tick_pixels"}
    centers = pixels + .5
    origin = centers.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((centers - origin) ** 2, axis=1))))
    if radius == 0:
        return {**failure, "reason": "coincident_tick_pixels"}
    scale = np.sqrt(2) / radius
    transform = np.array([[scale, 0, -scale*origin[0]],
                          [0, scale, -scale*origin[1]], [0, 0, 1.]])
    normalized = (centers - origin) * scale
    initializer = _cross_ratio_start(normalized)

    def at_span(span):
        fitted = _dlt(normalized, span)
        if fitted is None:
            return None
        matrix, condition = fitted
        errors = (_project(matrix, _canonical(count, span)) - normalized) / scale
        return matrix, condition, float(np.sum(errors ** 2))

    def cost(span):
        fitted = at_span(span)
        return np.inf if fitted is None else fitted[2]

    grid = [TAU * fraction / 10 for fraction in range(1, 10)]
    if initializer["span_radians"] is not None:
        grid.append(initializer["span_radians"])
    grid = sorted(set(grid))
    values = [cost(span) for span in grid]
    starts = []
    for i, (span, value) in enumerate(zip(grid, values)):
        left = values[i - 1] if i else np.inf
        right = values[i + 1] if i + 1 < len(grid) else np.inf
        if not np.isfinite(value) or value > min(left, right):
            continue
        low = grid[i - 1] if i else 0.
        high = grid[i + 1] if i + 1 < len(grid) else TAU
        optimum = minimize_scalar(cost, bounds=(low, high), method="bounded",
                                  options={"xatol": 1e-10, "maxiter": 100})
        refined = float(optimum.x)
        starts.append(refined if np.isfinite(cost(refined)) else span)
    if not starts:
        return {**failure, "reason": "no_physical_circle_dlt_candidate", "initializer": initializer,
                "coarse_spans": grid, "coarse_sse_pixels": values}

    candidates = []
    for span in starts:
        matrix, condition, _ = at_span(span)
        bottom = matrix[2, :2]
        vector = bottom / np.sqrt(1 - bottom @ bottom)
        initial = np.r_[matrix[:2].ravel(), vector, span]

        def residual(parameters):
            candidate, arc = _unpack(parameters)
            return ((_project(candidate, _canonical(count, arc)) - normalized) / scale).ravel()

        lower = np.r_[np.full(8, -np.inf), np.finfo(float).eps]
        upper = np.r_[np.full(8, np.inf), TAU - np.finfo(float).eps]
        result = least_squares(residual, initial, bounds=(lower, upper), method="trf",
                               x_scale="jac", max_nfev=160, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        fitted, fitted_span = _unpack(result.x)
        errors = residual(result.x).reshape(count, 2)
        if not np.isfinite(errors).all() or not result.success:
            continue
        homography = np.linalg.solve(transform, fitted)
        inverse = np.linalg.inv(homography)
        singular = np.linalg.svd(result.jac, compute_uv=False)
        candidates.append({"H": homography, "G": inverse, "span_radians": fitted_span,
                           "alpha_radians": fitted_span / (count - 1),
                           "sse_pixels": float(np.sum(errors ** 2)),
                           "rmse_pixels": float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))),
                           "max_error_pixels": float(np.linalg.norm(errors, axis=1).max()),
                           "dlt_condition": condition,
                           "jacobian_condition": float(singular[0] / singular[-1]),
                           "front_circle_margin": float(1 - np.linalg.norm(fitted[2, :2])),
                           "optimizer_evaluations": int(result.nfev)})
    candidates.sort(key=lambda candidate: candidate["sse_pixels"])
    unique = []
    for candidate in candidates:
        if all(abs(candidate["span_radians"] - old["span_radians"]) > 1e-6 for old in unique):
            unique.append(candidate)
    if not unique:
        return {**failure, "reason": "geometric_refinement_failed", "initializer": initializer,
                "coarse_spans": grid, "coarse_sse_pixels": values}
    best = unique[0]
    alternatives = [candidate for candidate in unique[1:]
                    if candidate["sse_pixels"] <= best["sse_pixels"] + count * .5**2]
    return {"status": "ok", "reason": None, "tick_count": count, **best,
            "initializer": initializer, "coarse_spans": grid, "coarse_sse_pixels": values,
            "candidates": unique, "near_fit_alternatives": alternatives,
            "near_fit_delta_sse_per_point_pixels": .5**2,
            "pixel_center_offset": .5, "geometric_refinement": True}
