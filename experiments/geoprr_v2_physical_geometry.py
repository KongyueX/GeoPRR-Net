"""Recover relative needle state from ordered source annotation geometry."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_templates(path: Path) -> dict[int, dict]:
    inventory = {int(key): value for key, value in json.loads(path.read_text()).items()}
    if set(inventory) != {76, 274, 472}:
        raise ValueError("expected the three frozen source needle templates")
    for count, template in inventory.items():
        vertices = np.asarray(template["scaled_vertices"], dtype=np.float64)
        if vertices.shape != (count, 3) or not np.isfinite(vertices).all():
            raise ValueError("invalid frozen template vertices")
        template["scaled_vertices"] = vertices
    return inventory


def source_to_world_plane(points: np.ndarray, box: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    crop = (points + .5 - box[:2]) / (box[2:] - box[:2])
    mapped = np.column_stack((crop, np.ones(len(crop)))) @ matrix.T
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 2]) <= 1e-12):
        raise ValueError("undefined annotation homography coordinate")
    normalized = mapped[:, :2] / mapped[:, 2, None]
    return (normalized - .5) * np.array([2., -2.])


def relative_state(annotation: dict, inventory: dict[int, dict]) -> dict:
    """Use mesh/tick observations only; reading and rotation fields are not inputs."""
    pointers = [entry for entry in annotation["seg_annotations"] if entry["type"] == "Pointer"]
    scales = [entry for entry in annotation["keypoints_annotations"] if entry["type"] == "ScaleMark"]
    if len(pointers) != 1 or len(scales) != 1:
        raise ValueError("expected one ordered pointer mesh and tick sequence")
    points = np.asarray(pointers[0]["mask"], dtype=np.float64)
    ticks = np.asarray(scales[0]["all_kp"], dtype=np.float64)
    camera = np.asarray(annotation["ann_camera_location"], dtype=np.float64)
    matrix = np.asarray(annotation["homo_matrix"], dtype=np.float64)
    box = np.asarray(annotation["dial_bbox_annotations"], dtype=np.float64)
    if camera.shape != (3,) or camera[2] <= 1e-12 or matrix.shape != (3, 3):
        raise ValueError("invalid source camera or plane homography")
    if box.shape != (4,) or np.any(box[2:] <= box[:2]):
        raise ValueError("invalid source casing box")
    if points.ndim != 2 or points.shape[1] != 2 or ticks.ndim != 2 or ticks.shape[1] != 2 or len(ticks) < 3:
        raise ValueError("invalid ordered point arrays")
    if not all(np.isfinite(value).all() for value in (points, ticks, camera, matrix, box)):
        raise ValueError("nonfinite annotation geometry")
    template = inventory[len(points)]
    vertices = template["scaled_vertices"]
    plane = source_to_world_plane(points, box, matrix)
    lifted = plane + (camera[:2] - plane) * vertices[:, 2, None] / camera[2]
    source = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    destination = lifted - lifted.mean(axis=0)
    dot = float(np.sum(source * destination))
    cross = float(np.sum(source[:, 0] * destination[:, 1] - source[:, 1] * destination[:, 0]))
    if np.hypot(dot, cross) <= 1e-12:
        raise ValueError("unidentified rigid needle rotation")
    theta = float(np.arctan2(cross, dot))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = vertices[:, :2] @ rotation.T
    translation = lifted.mean(axis=0) - rotated.mean(axis=0)
    residual = lifted - rotated - translation
    tick_world = source_to_world_plane(ticks, box, matrix)
    spacing = float(annotation["long_interval_degree"]) * np.pi / 180
    span = float((len(ticks) - 1) * spacing)
    if not 0 < span < 2 * np.pi:
        raise ValueError("invalid ordered tick arc")
    basis = np.exp(-1j * np.arange(len(ticks)) * spacing)
    observed = tick_world[:, 0] + 1j * tick_world[:, 1]
    centered_basis = basis - basis.mean()
    amplitude = np.vdot(centered_basis, observed - observed.mean()) / np.vdot(centered_basis, centered_basis).real
    if abs(amplitude) <= 1e-12:
        raise ValueError("unidentified ordered tick orientation")
    center = observed.mean() - amplitude * basis.mean()
    tick_residual = observed - (center + amplitude * basis)
    phi0 = float(np.angle(amplitude))
    delta = float((phi0 - theta) % (2 * np.pi))
    unwrapped_delta = float((phi0 - theta - span / 2 + np.pi) % (2 * np.pi) - np.pi + span / 2)
    raw = unwrapped_delta / span
    return {"delta_radians": delta, "span_radians": span,
            "unwrapped_delta_radians": unwrapped_delta,
            "raw_progress": float(raw), "prediction": float(np.clip(raw, 0., 1.)),
            "theta_radians": theta, "first_tick_phi0_radians": phi0,
            "template_family": template["family"], "template_vertex_count": len(points),
            "tick_count": len(ticks), "tick_spacing_radians": float(spacing),
            "fitted_tick_radius_world": float(abs(amplitude)),
            "fitted_tick_center_world": [float(center.real), float(center.imag)],
            "tick_fit_rmse_world": float(np.sqrt(np.mean(np.abs(tick_residual) ** 2))),
            "tick_fit_max_world": float(np.abs(tick_residual).max()),
            "rigid_translation_world": translation.tolist(),
            "rigid_centered_rmse_world": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
            "rigid_centered_max_world": float(np.linalg.norm(residual, axis=1).max()),
            "clipped_first": bool(raw < 0), "clipped_last": bool(raw > 1)}
