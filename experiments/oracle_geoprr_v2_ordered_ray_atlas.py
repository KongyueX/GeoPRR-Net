"""Compare a GT ordered ray atlas with saved GT homography, without inference."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
started = time.perf_counter()
previous = json.loads((args.directory / "gt_calibration_full_q_oracle.json").read_text())
maps = np.load(args.directory / "component_maps.npz")
report = {"protocol": "GT_ordered_common_center_ray_atlas_K65_oracle_v1", "diagnostic_only": True,
          "K": 65, "new_model_forwards": 0, "new_dataset_items_loaded": 0,
          "algorithm": {
              "knots": "s_j=j/64; canonical unit-circle p_j=(cos(a*s_j),sin(a*s_j)); x_j=project(H^-1*[p_j,1])",
              "center": "c=project(H^-1*[0,0,1]); all ray knots pass through this image center",
              "bearings": "phi_j=unwrap(atan2(x_j.y-c.y,x_j.x-c.x)); require strictly monotonic phi_j",
              "grid": "192x192 output-cell centers mapped to 768x768 ROI pixels with (index+.5)*4-.5",
              "chart": "unwrap image angle around (phi_0+phi_64)/2; t=sign*(phi-phi_0); knots=sign*(phi_j-phi_0)",
              "inverse_interpolation": "linear interpolation t -> s_j, clipping t outside [0,knots[-1]] to the corresponding endpoint",
              "readout": "expectation of atlas coordinate over the entire saved normalized Q; no support M, argmax, threshold or direct",
              "reference": "continuous saved-GT-H canonical angle, unwrapped around canonical arc midpoint and clamped to [0,1]",
              "error_units": "coordinate_difference_percent_fs=100*abs(atlas_progress-reference_progress); reading_error_percent_fs=100*abs(reading-target)",
              "seam_decomposition": "separate both-in-arc interpolation from both-out-of-arc endpoint selection; record membership mismatches explicitly"},
          "limitations": ["No atlas has been predicted or trained; these are GT geometry oracle diagnostics, not new model results.",
                          "GT oracle is not a formal performance upper bound.",
                          "Image-angle and canonical-angle complementary-gap midpoints may select different endpoints outside the measured arc.",
                          "Out-of-arc endpoint differences are chart seam conventions, not within-arc interpolation errors.",
                          "Common-center ray chart requires finite center and knots, strict ray order and an image bearing span below 2*pi; failures are reported without skipping layouts."],
          "layouts": {}, "samples": []}
grids = {}
for sample_id, geometry in previous["geometry"].items():
    h = np.asarray(geometry["roi_to_canonical"], dtype=np.float64)
    inverse = np.linalg.inv(h)
    span = geometry["arc_span_radians"]
    s = np.linspace(0, 1, 65)
    homogeneous_knots = np.stack((np.cos(span * s), np.sin(span * s), np.ones_like(s)), axis=1) @ inverse.T
    center_h = inverse @ np.array([0., 0., 1.])
    result = {"canonical_arc_span_degrees": float(np.rad2deg(span)),
              "projected_center_homogeneous": center_h.tolist(),
              "projected_knots_denominator_range": [float(homogeneous_knots[:, 2].min()), float(homogeneous_knots[:, 2].max())]}
    if abs(center_h[2]) <= 1e-10 or np.any(np.abs(homogeneous_knots[:, 2]) <= 1e-10):
        result.update(feasible=False, failure="center or projected circle knots have non-finite image coordinates")
        report["layouts"][sample_id] = result
        continue
    center = center_h[:2] / center_h[2]
    points = homogeneous_knots[:, :2] / homogeneous_knots[:, 2, None]
    phi = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    increments = np.diff(phi)
    sign = 1 if np.all(increments > 0) else -1 if np.all(increments < 0) else 0
    result.update(center_roi_pixels=center.tolist(), projected_knots_roi_pixels=points.tolist(),
                  progress_knots=s.tolist(), bearing_knots_radians=phi.tolist(),
                  bearing_increment_range_radians=[float(increments.min()), float(increments.max())], direction_sign=sign)
    if not sign or not 0 < abs(phi[-1] - phi[0]) < 2 * np.pi:
        result.update(feasible=False, failure="projected ray bearings are not strictly ordered on a single image-angle arc")
        report["layouts"][sample_id] = result
        continue
    knots = sign * (phi - phi[0])
    midpoint = (phi[0] + phi[-1]) / 2
    coordinates = (np.arange(192, dtype=np.float64) + 0.5) * 4 - 0.5
    xx, yy = np.meshgrid(coordinates, coordinates)
    image_angle = np.arctan2(yy - center[1], xx - center[0])
    image_unwrapped = (image_angle - midpoint + np.pi) % (2 * np.pi) - np.pi + midpoint
    t = sign * (image_unwrapped - phi[0])
    atlas = np.interp(t, knots, s)
    canonical_h = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ h.T
    canonical = canonical_h[..., :2] / canonical_h[..., 2, None]
    theta = np.arctan2(canonical[..., 1], canonical[..., 0])
    theta_unwrapped = (theta - span / 2 + np.pi) % (2 * np.pi) - np.pi + span / 2
    raw_reference = theta_unwrapped / span
    reference = np.clip(raw_reference, 0, 1)
    atlas_inside = (t >= 0) & (t <= knots[-1])
    reference_inside = (raw_reference >= 0) & (raw_reference <= 1)
    both_inside = atlas_inside & reference_inside
    both_outside = ~atlas_inside & ~reference_inside
    mismatch = atlas_inside != reference_inside
    seam = both_outside & (np.abs(atlas - reference) > 0.5)
    mask = maps[f"supervised__{sample_id}__m_gt"].astype(bool)
    differences = atlas - reference
    result.update(feasible=True, image_arc_span_degrees=float(np.rad2deg(knots[-1])),
                  signed_increasing_bearing_knots_radians=knots.tolist(),
                  image_chart_midpoint_radians=float(midpoint),
                  GT_support_pixels=int(mask.sum()),
                  GT_support_max_coordinate_difference_percent_fs=float(np.abs(differences[mask]).max() * 100),
                  GT_support_mean_coordinate_difference_percent_fs=float(np.abs(differences[mask]).mean() * 100),
                  GT_support_inside_both_charts_pixels=int((mask & both_inside).sum()),
                  whole_grid_arc_membership_mismatch_pixels=int(mismatch.sum()),
                  whole_grid_out_of_arc_endpoint_seam_difference_pixels=int(seam.sum()),
                  whole_grid_in_arc_max_coordinate_difference_percent_fs=float(np.abs(differences[both_inside]).max() * 100),
                  whole_grid_in_arc_mean_coordinate_difference_percent_fs=float(np.abs(differences[both_inside]).mean() * 100))
    report["layouts"][sample_id] = result
    grids[sample_id] = (atlas, reference, t, knots[-1], raw_reference, both_inside, both_outside, mismatch, seam)
for sample in previous["samples"]:
    model, sample_id = sample["model"], sample["sample_id"]
    record = {"model": model, "sample_id": sample_id, "target": sample["target"]}
    if sample_id not in grids:
        record.update(feasible=False, failure=report["layouts"][sample_id]["failure"])
        report["samples"].append(record)
        continue
    atlas, reference, t, end, raw_reference, both_inside, both_outside, mismatch, seam = grids[sample_id]
    q = maps[f"{model}__{sample_id}__q_pred"].astype(np.float64)
    q /= q.sum()
    reading = float((q * atlas).sum())
    reference_reading = float((q * reference).sum())
    differences = atlas - reference
    delta_inside = float((q * differences * both_inside).sum())
    delta_outside = float((q * differences * both_outside).sum())
    delta_mismatch = float((q * differences * mismatch).sum())
    assert abs(reference_reading - sample["GT_calibration_full_Q_oracle_reading"]) < 1e-12
    assert abs(reading - reference_reading - delta_inside - delta_outside - delta_mismatch) < 1e-12
    record.update(feasible=True, GT_atlas_full_Q_oracle_reading=reading,
                  GT_atlas_full_Q_oracle_error_percent_fs=100 * abs(reading - sample["target"]),
                  previous_GT_H_full_Q_oracle_reading=reference_reading,
                  previous_GT_H_full_Q_oracle_error_percent_fs=sample["GT_calibration_full_Q_oracle_error_percent_fs"],
                  atlas_minus_H_reading=reading - reference_reading,
                  in_arc_interpolation_reading_delta=delta_inside,
                  out_of_arc_endpoint_selection_reading_delta=delta_outside,
                  arc_membership_mismatch_reading_delta=delta_mismatch,
                  Q_mass_in_out_of_arc_endpoint_seam_difference=float(q[seam].sum()),
                  Q_mass_arc_membership_mismatch=float(q[mismatch].sum()),
                  atlas_Q_mass_clipped_first=float(q[t < 0].sum()), atlas_Q_mass_clipped_last=float(q[t > end].sum()),
                  previous_H_Q_mass_clipped_first=float(q[raw_reference < 0].sum()),
                  previous_H_Q_mass_clipped_last=float(q[raw_reference > 1].sum()))
    report["samples"].append(record)
report["resources"] = {"wall_seconds": time.perf_counter() - started, "device": "cpu",
                       "new_model_forwards": 0, "new_dataset_items_loaded": 0,
                       "environment": {key: os.environ.get(key) for key in
                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
output = args.directory / "gt_ordered_ray_atlas_oracle.json"
output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "resources": report["resources"],
                  "layouts": {key: {field: value for field, value in layout.items() if field not in
                                       ("projected_knots_roi_pixels", "progress_knots", "bearing_knots_radians", "signed_increasing_bearing_knots_radians")}
                              for key, layout in report["layouts"].items()}, "samples": report["samples"]}, indent=2))
