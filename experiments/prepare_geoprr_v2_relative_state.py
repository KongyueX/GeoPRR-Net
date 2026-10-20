"""Export target-blind physical relative-state labels for the frozen source split."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np

from experiments.geoprr_v2_physical_geometry import load_templates, relative_state


VERSION = "physical_relative_state_v1_source_role_centered_mesh_and_tick_fit"


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate manifest IDs: {path}")
    return rows


def cohort_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid"]]
    result = {"population": len(rows), "valid": len(valid),
              "invalid": len(rows) - len(valid),
              "invalid_rows": [{"sample_id": row["sample_id"], "failure": row["failure"]}
                               for row in rows if not row["valid"]],
              "template_counts": dict(Counter(row["template_family"] for row in valid)),
              "clipped_first": sum(row["clipped_first"] for row in valid),
              "clipped_last": sum(row["clipped_last"] for row in valid)}
    if valid:
        for key in ("delta_radians", "span_radians", "raw_progress", "rigid_centered_rmse_world", "tick_fit_rmse_world"):
            values = [row[key] for row in valid]
            result[key] = {"minimum": min(values), "maximum": max(values), "mean": float(np.mean(values))}
    return result


def export_split(rows: list[dict], output: Path, roles: dict[str, str], inventory: dict[int, dict]) -> list[dict]:
    exported = []
    with output.open("w") as handle:
        for row in rows:
            result = {"sample_id": row["sample_id"], "group_id": row["group_id"],
                      "source_role": roles[row["sample_id"]],
                      "valid": False, "failure": "", "delta_radians": None, "span_radians": None}
            try:
                annotation = json.loads(Path(row["metadata"]["annotation_path"]).read_text())
                result.update(relative_state(annotation, inventory), valid=True)
            except (KeyError, OSError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                result["failure"] = str(exc)
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            exported.append(result)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--original-train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    train = read_manifest(args.train_manifest)
    original = read_manifest(args.original_train_manifest)
    validation = read_manifest(args.validation_manifest)
    original_ids = {row["sample_id"] for row in original}
    train_ids = {row["sample_id"] for row in train}
    validation_ids = {row["sample_id"] for row in validation}
    if len(train) != 18590 or len(original) != 14494 or len(validation) != 1506:
        raise ValueError("expected the frozen 18590/14494/1506 source populations")
    if not original_ids <= train_ids or train_ids & validation_ids:
        raise ValueError("source split identity overlap or missing original training IDs")
    roles = {sample_id: "original" for sample_id in original_ids}
    roles.update({sample_id: "supplement" for sample_id in train_ids - original_ids})
    roles.update({sample_id: "validation" for sample_id in validation_ids})
    inventory = load_templates(args.templates)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"version": VERSION, "train_manifest": str(args.train_manifest),
                "original_train_manifest": str(args.original_train_manifest),
                "validation_manifest": str(args.validation_manifest), "templates": str(args.templates),
                "annotation_fields_used": ["seg_annotations.Pointer.mask", "keypoints_annotations.ScaleMark.all_kp",
                                           "ann_camera_location", "homo_matrix", "dial_bbox_annotations", "long_interval_degree"],
                "excluded_information": ["ground_truth", "pointer_angle", "pointer_rotate_degree",
                                         "pointer_rotation_euler", "normalized_state", "factors.r", "RGB", "model_weights"],
                "pixel_rule": "original integer annotation cell center +0.5; original integer casing bbox",
                "needle_fit": "individual vertex-height ray lift; centered 2D orthogonal Procrustes",
                "tick_fit": "centered complex least squares with free center/amplitude; no assumed tick height",
                "delta": "(first_tick_phi0 - object_theta) modulo 2*pi, range [0,2*pi)",
                "span": "(ordered tick count - 1) * annotation long_interval_degree",
                "readout": "fixed span-midpoint unwrap and [0,1] clip; no target-derived angle",
                "source_role_values": ["original", "supplement", "validation"],
                "row_policy": "preserve manifest order and all rows; invalid rows have null labels and an explicit failure",
                "new_model_forwards": 0, "training_steps": 0, "seed_expansion": False}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    training_labels = export_split(train, args.output / "train.jsonl", roles, inventory)
    validation_labels = export_split(validation, args.output / "validation.jsonl", roles, inventory)
    all_labels = training_labels + validation_labels
    summary = {**protocol, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "training": cohort_summary(training_labels),
               "validation": cohort_summary(validation_labels),
               "source_role_counts": dict(Counter(row["source_role"] for row in all_labels)),
               "source_roles": {role: cohort_summary([row for row in all_labels if row["source_role"] == role])
                                for role in ("original", "supplement", "validation")},
               "resources": {"wall_seconds": time.perf_counter() - started, "numpy_version": np.__version__,
                             "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                             "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                             "max_rss_linux_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("source_role_counts", "source_roles", "resources")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
