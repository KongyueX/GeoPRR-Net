"""Decode geometry observations without per-image camera or calibration labels."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_height_profile import fit_height_geometry
from experiments.geoprr_v2_physical_geometry import load_templates
from experiments.geoprr_v2_tick_calibration import fit_ordered_tick_calibration


FIELDS = ("sample_id", "group_id", "dial_bbox_integer", "ordered_long_tick_pixels", "pointer_projected_vertices_pixels")


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
    source = [{key: row[key] for key in FIELDS} for row in
              (json.loads(line) for line in args.geometry.read_text().splitlines() if line.strip())]
    if len(source) != 1506 or len({r["sample_id"] for r in source}) != 1506:
        raise ValueError("expected complete original1506")
    normalize = np.array([[.5, 0., .5], [0., -.5, .5], [0., 0., 1.]])
    protocol = {"scope": "GT material/tick observations; estimated calibration; not RGB recognition",
                "input_fields": FIELDS, "camera_calibration_or_angle_labels": "none",
                "selection": selection, "pixel_rule": "raw integer cell centers +0.5",
                "calibration": "ordered tick-only whole-N pixel SSE; nine fixed span starts plus legal CR; geometric refinement",
                "material_fit": "full source-pixel reprojection; unknown camera nuisance; fixed three starts; GT family/XYZ correspondences",
                "prediction_selection": "lowest tick SSE calibration then lowest body SSE; no reading target selection",
                "alternative_diagnostics": "same procedure for near-fit tick calibrations; their readings never used for target-based choice",
                "readout": "canonical first tick angle0,delta=-theta; fixed arc-midpoint unwrap/clip",
                "failures": "retain each invalid row for error1; no deletion",
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
                calibration = fit_ordered_tick_calibration(row["ordered_long_tick_pixels"])
                result["calibration"] = calibration
                if calibration["status"] != "ok":
                    raise ValueError(calibration["reason"])
                box = np.asarray(row["dial_bbox_integer"], dtype=np.float64)
                crop = np.array([[box[2]-box[0], 0., box[0]], [0., box[3]-box[1], box[1]], [0., 0., 1.]])
                readings = []
                for frame in [calibration, *calibration["near_fit_alternatives"]]:
                    matrix = normalize @ frame["G"] @ crop
                    fit = fit_height_geometry(xyz, points[body["indices"]], box, matrix)
                    span = float(frame["span_radians"])
                    raw = ((-fit["theta_radians"] - span/2 + math.pi) % (2*math.pi) - math.pi + span/2) / span
                    readings.append({"raw_progress": float(raw), "prediction": float(np.clip(raw, 0., 1.)),
                                     "span_radians": span, "tick_sse_pixels": frame["sse_pixels"], "fit": fit})
                chosen = readings[0]
                valid = bool(chosen["fit"]["success"] and math.isfinite(chosen["raw_progress"])
                             and chosen["fit"]["positive_scale"] > 1e-12)
                result.update(valid=valid, prediction=chosen["prediction"] if valid else None,
                              raw_progress=chosen["raw_progress"] if valid else None,
                              template_family=body["family"], material_points=len(body["indices"]),
                              readout_candidates=readings, prediction_range_across_near_tick_fits=
                              float(np.ptp([r["prediction"] for r in readings])),
                              failure="" if valid else "material_fit_nonconvergence_or_zero_scale")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                result["failure"] = f"{type(exc).__name__}: {exc}"
            if not result["valid"]:
                failures.append(row["sample_id"])
            handle.write(json.dumps(serializable(result), allow_nan=False) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed": index+1, "failed": len(failures), "wall_seconds": time.perf_counter()-started}), flush=True)
    (args.output / "completed.json").write_text(json.dumps({"samples": 1506, "failed_ids": failures,
        "wall_seconds": time.perf_counter()-started, "numpy": np.__version__, "model_forwards": 0,
        "model_optimizer_updates": 0, "targets_read": False, "GT_H_C_spacing_span_read": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
