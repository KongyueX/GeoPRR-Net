"""Export aggregate ROI results without local paths or per-image records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def export(input_dir: Path, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _read(input_dir / "roi_comparison_zero_shot.json")
    transfer = {
        "schema_version": 1,
        "status": raw["status"],
        "generated_at_utc": raw["generated_at_utc"],
        "units": "NMAE is percent full scale; accuracy and coverage are percentages; SD is across training seeds, ddof=1.",
        "seeds": raw["seeds"],
        "source_domain": {key: raw["source_domain"][key] for key in ("name", "train_samples", "validation_samples", "selection")},
        "protocol": raw["protocol"],
        "reference_kind": raw["reference_kind"],
        "reference_geometry": {
            "model": ("YOLO11s-Pose-4KP trained on SyncG" if raw["reference_kind"] == "source_only_pose4kp_reference" else raw["reference_kind"]),
            "keypoint_indices": [0, 2, 3],
            "predicted_pointer_tip_used": False,
            "scope": "Industrial supplied ROI; RF100 keeps annotated pivot/start/end",
            "training_manifest_audit": (
                "Each seed has 12866 training and 1576 validation IDs in the SyncG train manifest, with no train/validation ID overlap."
                if raw["reference_kind"] == "source_only_pose4kp_reference"
                else "Reference training provenance has not been established."
            ),
            "per_seed": [
                {
                    "seed": item["checkpoint_metadata_verified_on_cpu"]["seed"],
                    "rows": item["industrial_geometry"]["rows"],
                    "passes": item["industrial_geometry"]["passes"],
                    "failures": item["industrial_geometry"]["failures"],
                    "detector_configuration": item["industrial_geometry"]["detector_configuration"],
                }
                for item in raw["geometry_provenance"]["caches"]
            ],
        },
        "datasets": {},
        "verification": {
            "rows_recomputed": raw["verification"]["rows_recomputed"],
            "stored_error_max_abs_difference": raw["verification"]["stored_error_max_abs_difference"],
            "checks": raw["verification"]["checks"],
        },
    }
    for key, dataset in raw["datasets"].items():
        target = {name: dataset[name] for name in ("name", "samples", "groups", "rows_per_seed")}
        target["methods"] = {
            method: {
                "name": value["name"],
                "aggregate": value["aggregate"],
                "per_seed": [
                    {name: record[name] for name in ("seed", "clean", "all_conditions")}
                    for record in value["per_seed"]
                ],
            }
            for method, value in dataset["methods"].items()
        }
        transfer["datasets"][key] = target
    efficiency_raw = _read(input_dir / "roi_comparison_efficiency.json")
    efficiency = {
        "schema_version": 1,
        "status": efficiency_raw["status"],
        "measurement_date": efficiency_raw["measurement_date"],
        "reports": {},
    }
    for arm, value in efficiency_raw["reports"].items():
        configuration = value["configuration"]
        efficiency["reports"][arm] = {
            "method": value["method"],
            "display_name": value["display_name"],
            "protocol": value["protocol"],
            "environment": value["environment"],
            "configuration": {key: configuration[key] for key in (
                "seed", "precision", "condition", "image_size", "preprocessing",
                "evaluation_scope", "success_definition", "observed_precision",
                "reference_detector_kind", "reference_configuration",
                "geometry_failure_behavior",
            ) if key in configuration},
            "input": {key: value["input"][key] for key in ("manifest_rows", "selected_rows", "selection", "ground_truth_read")},
            "parameters": value["parameters"],
            "measurement": {key: item for key, item in value["measurement"].items() if key != "raw_latency_ms"},
            "flops": value["flops"],
        }
    _write(output_dir / "roi_comparison_zero_shot_public.json", transfer)
    _write(output_dir / "roi_comparison_efficiency_public.json", efficiency)
    # The input CSVs contain only cohort/seed aggregate statistics.
    for name in ("roi_comparison_zero_shot.csv", "roi_comparison_efficiency.csv"):
        with (input_dir / name).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        with (output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return transfer, efficiency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("docs/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/data"))
    args = parser.parse_args()
    transfer, efficiency = export(args.input_dir, args.output_dir)
    print(json.dumps({"transfer_datasets": len(transfer["datasets"]), "efficiency_arms": len(efficiency["reports"])}))


if __name__ == "__main__":
    main()
