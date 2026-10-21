"""Decode joint tick/material geometry without camera or reading labels."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_joint_geometry import fit_joint_geometry
from experiments.geoprr_v2_physical_geometry import load_templates


FIELDS = ("sample_id", "group_id", "ordered_long_tick_pixels", "pointer_projected_vertices_pixels")


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    for name in ("geometry", "templates", "selection", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    templates = load_templates(args.templates)
    selection = json.loads(args.selection.read_text())
    expected = {"76": list(range(64, 76)), "274": list(range(256, 274)), "472": list(range(408))}
    if {key: value["indices"] for key, value in selection["groups"].items()} != expected:
        raise ValueError("requires frozen nonring material indices")
    source = [{key: row[key] for key in FIELDS} for row in
              (json.loads(line) for line in args.geometry.read_text().splitlines() if line.strip())]
    if len(source) != 1506 or len({row["sample_id"] for row in source}) != 1506:
        raise ValueError("expected complete original1506")
    protocol = {
        "scope": "GT material/tick correspondences and GT template identity; conditional geometry, not RGB recognition",
        "input_fields": FIELDS, "camera_calibration_or_angle_labels": "none",
        "selection": selection, "pixel_rule": "raw integer cell centers +0.5 exactly once inside joint fitter",
        "joint_fit": "H8 + span1 + height-aware body7; shared full source-pixel SSE across all tick and material residuals",
        "starts": "best retained serial tick-only fit and fixed 0.2/0.5/0.8 times 2pi spans; observation-only initialization",
        "prediction_selection": "minimum joint source-pixel SSE among converged physically valid final iterates; retain all four starts; no target selection or global optimum certificate",
        "physical_constraints": "whole canonical circle in front; positive material camera denominator and projected depths",
        "readout": "canonical first tick angle0, delta=-theta; fixed arc-midpoint unwrap/clip; no fallback or r8",
        "failures": "retain each invalid row for error1; no deletion",
        "observation_limitations": "projected mesh vertices lack verified per-image visibility and observable vertex identity",
        "target_join": "separate scoring only after all predictions.jsonl rows are saved",
        "new_model_forwards": 0, "model_optimizer_updates": 0, "seed_expansion": False}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    failures = []
    with (args.output / "predictions.jsonl").open("w") as handle:
        for index, row in enumerate(source):
            result = {"sample_id": row["sample_id"], "group_id": row["group_id"], "valid": False,
                      "prediction": None, "raw_progress": None}
            try:
                points = np.asarray(row["pointer_projected_vertices_pixels"], dtype=np.float64)
                body = selection["groups"][str(len(points))]
                xyz = templates[len(points)]["scaled_vertices"][body["indices"]]
                fit = fit_joint_geometry(xyz, points[body["indices"]], row["ordered_long_tick_pixels"])
                valid = fit["status"] == "ok"
                result.update(valid=valid, prediction=fit["prediction"] if valid else None,
                              raw_progress=fit["raw_prediction"] if valid else None,
                              template_family=body["family"], material_points=len(body["indices"]),
                              tick_count=len(row["ordered_long_tick_pixels"]), fit=fit,
                              failure="" if valid else fit["reason"])
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                result["failure"] = f"{type(exc).__name__}: {exc}"
            if not result["valid"]:
                failures.append(row["sample_id"])
            handle.write(json.dumps(serializable(result), allow_nan=False) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed": index+1, "failed": len(failures),
                                  "wall_seconds": time.perf_counter()-started}), flush=True)
    (args.output / "completed.json").write_text(json.dumps({
        "samples": 1506, "failed_ids": failures, "wall_seconds": time.perf_counter()-started,
        "numpy": np.__version__, "model_forwards": 0, "model_optimizer_updates": 0,
        "targets_read": False, "GT_H_C_spacing_span_read": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
