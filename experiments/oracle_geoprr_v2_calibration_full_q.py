"""Push saved full predicted Q through GT calibration with zero model forwards."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


def pushforward(q, progress, bins=256):
    q = q / q.sum()
    positions = progress * (bins - 1)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, bins - 1)
    fraction = positions - lower
    distribution = np.zeros(bins, dtype=np.float64)
    np.add.at(distribution, lower.ravel(), (q * (1 - fraction)).ravel())
    np.add.at(distribution, upper.ravel(), (q * fraction).ravel())
    return float((q * progress).sum()), distribution.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    sys.path.insert(0, str(args.root / "code"))
    from experiments.geoprr_v2_data import GeoPRRV2Dataset

    diagnosis = json.loads((args.directory / "diagnosis.json").read_text())
    maps = np.load(args.directory / "component_maps.npz")
    original = diagnosis["models"]["supervised"]
    dataset = GeoPRRV2Dataset(original["manifest"], dataset_root=args.root / "datasets",
                             image_size=768, output_size=192, require_geometry=True, augment=False)
    geometry = {}
    for sample_id in diagnosis["sample_ids"]:
        index = dataset.by_id[sample_id]
        sample = dataset[index]
        calibration = dataset._calibrations[index]
        h = np.asarray(calibration["roi_to_canonical"], dtype=np.float64)
        angles = np.asarray(calibration["angles"], dtype=np.float64)
        span = float(angles[-1])
        coordinates = (np.arange(192, dtype=np.float64) + 0.5) * 768 / 192 - 0.5
        xx, yy = np.meshgrid(coordinates, coordinates)
        homogeneous = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        transformed = homogeneous @ h.T
        denominator = transformed[..., 2]
        assert np.all(np.abs(denominator) > 1e-10)
        canonical = transformed[..., :2] / denominator[..., None]
        theta = np.arctan2(canonical[..., 1], canonical[..., 0])
        midpoint = span / 2
        unwrapped = (theta - midpoint + np.pi) % (2 * np.pi) - np.pi + midpoint
        unclipped = unwrapped / span
        progress = np.clip(unclipped, 0, 1)
        geometry[sample_id] = (progress, unclipped, denominator)
        assert np.allclose(angles / span, calibration["progress_anchors"], atol=1e-12, rtol=0)
        geometry[sample_id] += ({"roi_to_canonical": h.tolist(), "arc_span_radians": span,
                                 "arc_span_degrees": float(np.rad2deg(span)),
                                 "arc_midpoint_radians": midpoint,
                                 "progress_anchors": calibration["progress_anchors"].tolist(),
                                 "angles_radians": angles.tolist(),
                                 "original_source_validation_index": index,
                                 "GT_geometry_audit": sample["geometry_audit"]},)
    report = {"protocol": "GT_calibration_full_Q_pushforward_oracle_v1", "diagnostic_only": True,
              "new_model_forwards": 0, "device": args.device,
              "sample_ids": diagnosis["sample_ids"], "source_manifest": original["manifest"],
              "algorithm": {
                  "grid": "192x192 output-cell centers mapped to 768x768 ROI coordinates: (index+.5)*4-.5, in ROI pixels",
                  "canonical_map": "[X,Y,W]=H*[x,y,1]; canonical=(X/W,Y/W), using the original float64 GT roi_to_canonical",
                  "angle": "theta=atan2(Y/W,X/W) in radians; the ordered first tick is angle 0",
                  "unwrap": "a=GT arc span; theta_u=((theta-a/2+pi) mod 2pi)-pi+a/2, then u=clip(theta_u/a,0,1)",
                  "readout": "normalize the entire saved Q to sum 1, then E_Q[u]; no M multiplication, point selection, support threshold, or direct fallback",
                  "distribution": "push all Q mass to adjacent progress bins with linear weights, preserving its continuous expected progress",
                  "target": "normalized meter progress; error_percent_fs=100*abs(reading-target)",
                  "GT_anchors": "all three layouts have equally spaced angle and progress anchors, so angle/arc_span equals the original within-arc GT calibration",
                  "comparison_GT_U_GT_M": "sum(Qpred*original_GT_U*GT_M)/sum(Qpred*GT_M), retaining the original GT map angle convention",
                  "seam": "arc-midpoint unwrap maps small negative angles near the first tick to progress 0 rather than to the last tick"},
              "interpretation": ["GT-calibration oracle is a candidate upper-reference diagnostic conditional on saved predicted Q, not a formal error bound.",
                                 "GT H/ordered ticks/arc are unavailable to an image-only deployed reader unless predicted; this does not establish learned calibration accuracy.",
                                 "No new method result or novelty claim follows from known homography-to-angle composition.",
                                 "All Q tails are retained; clamping maps out-of-arc angles to the corresponding endpoint and its probability mass is reported."],
              "geometry": {key: value[3] for key, value in geometry.items()}, "samples": []}
    for model, detail in diagnosis["models"].items():
        for sample in detail["samples"]:
            sample_id = sample["sample_id"]
            progress, unclipped, denominator, _ = geometry[sample_id]
            prefix = f"{model}__{sample_id}__"
            q, qg, mg, ug = [maps[prefix + key].astype(np.float64) for key in ("q_pred", "q_gt", "m_gt", "u_gt")]
            q /= q.sum()
            reading, distribution = pushforward(q, progress)
            gt_q_reading, _ = pushforward(qg, progress)
            masked_mass = float((q * mg).sum())
            gt_u_m_reading = float((q * mg * ug).sum() / masked_mass)
            raw_gt_u_reading = float((q * ug).sum())
            target = sample["target"]
            forced = sample["conditional_and_oracle_diagnostics"]["predicted_U_Q_M"]["forced_conditional_reading"]
            report["samples"].append({
                "model": model, "sample_id": sample_id, "target": target,
                "GT_calibration_full_Q_oracle_reading": reading,
                "GT_calibration_full_Q_oracle_error_percent_fs": 100 * abs(reading - target),
                "GT_calibration_full_Q_distribution": distribution,
                "all_GT_calibration_and_GT_Q_reading": gt_q_reading,
                "original_GT_U_GT_M_with_pred_Q_reading": gt_u_m_reading,
                "original_GT_U_GT_M_with_pred_Q_error_percent_fs": 100 * abs(gt_u_m_reading - target),
                "original_GT_U_GT_M_support_mass_after_normalizing_Q": masked_mass,
                "original_GT_U_without_M_with_pred_Q_reading": raw_gt_u_reading,
                "existing_predicted_U_Q_M_forced_reading": forced,
                "existing_predicted_U_Q_M_forced_error_percent_fs": 100 * abs(forced - target),
                "oracle_minus_original_GT_U_GT_M_reading": reading - gt_u_m_reading,
                "Q_mass_clamped_to_first_endpoint": float(q[unclipped < 0].sum()),
                "Q_mass_clamped_to_last_endpoint": float(q[unclipped > 1].sum()),
                "Q_mass_with_negative_H_denominator": float(q[denominator < 0].sum()),
                "H_denominator_range_on_grid": [float(denominator.min()), float(denominator.max())]})
    report["resources"] = {"wall_seconds": time.perf_counter() - started,
                           "new_model_forwards": 0, "cuda_initialized": torch.cuda.is_initialized(),
                           "torch_threads": torch.get_num_threads(),
                           "environment": {key: os.environ.get(key) for key in
                                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}
    output = args.directory / "gt_calibration_full_q_oracle.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "resources": report["resources"],
                      "samples": [{key: value for key, value in row.items() if not key.endswith("distribution")}
                                  for row in report["samples"]]}, indent=2))


if __name__ == "__main__":
    main()
