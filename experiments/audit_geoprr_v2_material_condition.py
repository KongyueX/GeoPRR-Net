"""Decode one complete target-blind material-geometry conditional population."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_height_profile import fit_height_geometry
from experiments.geoprr_v2_physical_geometry import load_templates, source_to_world_plane


def main():
    parser = argparse.ArgumentParser()
    for name in ("geometry", "templates", "selection", "spacing", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    templates = load_templates(args.templates)
    selection = json.loads(args.selection.read_text())
    spacing = json.loads(args.spacing.read_text())
    source = [json.loads(line) for line in args.geometry.read_text().splitlines() if line.strip()]
    if len(source) != 1506 or len({r["sample_id"] for r in source}) != 1506:
        raise ValueError("expected the complete original1506")
    protocol = {"scope": "GT-G and known-spacing conditional geometry; material correspondences only, no GT camera or reading input",
                "population": 1506, "selection": selection, "pixel_rule": "integer cell centers +0.5",
                "fit": "full source-pixel reprojection SSE; fixed starts b=0,bmax/2,clipped linear b; minSSE only",
                "solver": "scipy1.15.3 least_squares analyticJac; max_nfev300,ftol=xtol1e-12,gtol1e-10",
                "camera": "a,b nuisance fitted; 0<=b<=(1-1e-6)/max_material_z; no GT C input",
                "readout": "free tick center knownstep complexLS; fixed arc-midpoint unwrap and clip; no y branch selection",
                "failures": "nonconvergence/nonfinite/numerical failures retained invalid for error1 scoring",
                "neural_model_forwards": 0, "model_optimizer_updates": 0, "seed_expansion": False,
                "information_boundary": "GT G, material identities/family and tick spacing are oracle supervision inputs; not RGB deployment"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    failures = []
    with (args.output / "predictions.jsonl").open("w") as handle:
        for index, row in enumerate(source):
            result = {"sample_id": row["sample_id"], "group_id": row["group_id"], "valid": False,
                      "prediction": None, "raw_progress": None}
            try:
                points = np.asarray(row["pointer_projected_vertices_pixels"], dtype=np.float64)
                body = selection["groups"][str(len(points))]
                ids = body["indices"]
                matrix, bbox = row["homography_bbox_to_canonical_z0"], row["dial_bbox_integer"]
                fit = fit_height_geometry(templates[len(points)]["scaled_vertices"][ids], points[ids], bbox, matrix)
                ticks = source_to_world_plane(np.asarray(row["ordered_long_tick_pixels"], dtype=np.float64),
                                              np.asarray(bbox), np.asarray(matrix))
                step = math.radians(float(spacing[row["sample_id"]]))
                span = (len(ticks) - 1) * step
                basis = np.exp(-1j * np.arange(len(ticks)) * step)
                observed = ticks[:, 0] + 1j * ticks[:, 1]
                centered = basis - basis.mean()
                amplitude = np.vdot(centered, observed - observed.mean()) / np.vdot(centered, centered).real
                if not 0 < span < 2 * math.pi or abs(amplitude) <= 1e-12:
                    raise ValueError("invalid conditional tick frame")
                phi0, theta = float(np.angle(amplitude)), fit["theta_radians"]
                raw = ((phi0 - theta - span / 2 + math.pi) % (2 * math.pi) - math.pi + span / 2) / span
                valid = bool(fit["success"] and math.isfinite(raw) and fit["positive_scale"] > 1e-12)
                result.update(valid=valid, prediction=float(np.clip(raw, 0, 1)) if valid else None,
                              raw_progress=float(raw) if valid else None, template_family=body["family"],
                              material_points=len(ids), first_tick_phi0_radians=phi0, span_radians=span,
                              fit=fit, failure="" if valid else "geometry_nonconvergence_or_zero_scale")
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                result["failure"] = f"{type(exc).__name__}: {exc}"
            if not result["valid"]:
                failures.append(row["sample_id"])
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            handle.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed": index + 1, "failed": len(failures), "wall_seconds": time.perf_counter() - started}), flush=True)
    (args.output / "completed.json").write_text(json.dumps({"samples": 1506, "failed_ids": failures,
        "wall_seconds": time.perf_counter() - started, "numpy": np.__version__,
        "model_forwards": 0, "model_optimizer_updates": 0, "targets_read": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
