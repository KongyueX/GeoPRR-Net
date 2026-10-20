"""Summarize the six saved CPU component maps without model inference."""
import argparse
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
diagnosis = json.loads((args.directory / "diagnosis.json").read_text())
maps = np.load(args.directory / "component_maps.npz")
results = []
for model, details in diagnosis["models"].items():
    for sample in details["samples"]:
        prefix = f"{model}__{sample['sample_id']}__"
        q, m, u, qg, mg, ug = [maps[prefix + key].astype(np.float64) for key in
                               ("q_pred", "m_pred", "u_mean_pred", "q_gt", "m_gt", "u_gt")]
        yy, xx = np.indices(q.shape)
        gt_x, gt_y = sample["Q"]["gt_virtual_point_grid_xy"]
        distance = np.hypot(xx - gt_x, yy - gt_y)
        qm = q * m
        effective = qm / qm.sum()
        effective_mean_x = float((effective * xx).sum())
        effective_mean_y = float((effective * yy).sum())
        inside = mg.astype(bool)
        record = {"model": model, "sample_id": sample["sample_id"], "diagnostic_only": True,
                  "predicted_support_soft_area_over_gt_area": float(m.sum() / mg.sum()),
                  "nearest_M_ge_0p5_distance_from_gt_point_grid_px": float(distance[m >= 0.5].min()),
                  "effective_QM_mean_distance_from_gt_point_grid_px": float((effective * distance).sum()),
                  "effective_QM_mean_grid_xy": [effective_mean_x, effective_mean_y],
                  "effective_QM_mean_position_distance_from_gt_point_grid_px": float(np.hypot(effective_mean_x - gt_x, effective_mean_y - gt_y)),
                  "Qpred_mean_distance_from_gt_point_grid_px": float((q * distance).sum()),
                  "Qpred_and_effective_QM_local_mass": {
                      f"r{radius}": {"Qpred_mass": float(q[distance <= radius].sum()),
                                      "normalized_QM_mass": float(effective[distance <= radius].sum()),
                                      "GT_support_local_recall_M_ge_0p5": float(np.mean(m[(distance <= radius) & inside] >= 0.5))}
                      for radius in (2, 4, 8, 16)},
                  "U_pred_MAE_in_GT_support_percent_fs": float(np.mean(np.abs(u[inside] - ug[inside])) * 100),
                  "U_pred_MAE_weighted_by_GT_QM_percent_fs": float((np.abs(u - ug) * qg * mg).sum() / (qg * mg).sum() * 100),
                  "U_pred_distribution_with_GT_QM": {}}
        distribution = sample["conditional_and_oracle_diagnostics"]["replace_Q_M_with_GT"]["forced_conditional_distribution"]
        record["U_pred_distribution_with_GT_QM"] = {
            "bin_0_probability": distribution[0], "bin_255_probability": distribution[-1],
            "bin_0_to_16_probability": sum(distribution[:17]),
            "bin_239_to_255_probability": sum(distribution[-17:]),
            "top_five_bins": sorted(enumerate(distribution), key=lambda item: -item[1])[:5]}
        results.append(record)
output = args.directory / "saved_map_summary.json"
output.write_text(json.dumps({"diagnostic_only": True, "new_model_forwards": 0, "samples": results}, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "samples": results}, indent=2))
