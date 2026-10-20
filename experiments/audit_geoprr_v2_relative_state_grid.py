"""Check the fixed probability readout using saved target-blind state labels."""
import argparse
import json
from pathlib import Path
import time

import torch

from experiments.geoprr_v2_relative_state import RelativeStateReadout, relative_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    labels = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    if len(labels) != 1506 or len({r["sample_id"] for r in labels}) != 1506 or not all(r["valid"] for r in labels):
        raise ValueError("grid audit requires the complete valid1506 state labels")
    bins = 256
    decoder = RelativeStateReadout(bins).eval()
    records, maximum_mass_error = [], 0.
    with torch.no_grad():
        for start in range(0, len(labels), 16):
            batch = labels[start:start + 16]
            delta = torch.tensor([r["delta_radians"] for r in batch])
            span = torch.tensor([r["span_radians"] for r in batch])
            dl, du, dw, ll, lu, lw = relative_targets(delta, span, bins)
            a, b = torch.zeros(len(batch), bins), torch.zeros(len(batch), bins)
            for index, weight in ((dl, 1 - dw), (du, dw)):
                a.scatter_add_(1, index[:, None], weight[:, None])
            for index, weight in ((ll, 1 - lw), (lu, lw)):
                b.scatter_add_(1, index[:, None], weight[:, None])
            joint = a[:, :, None] * b[:, None, :]
            result = decoder(joint)
            maximum_mass_error = max(maximum_mass_error, float((result["distribution"].sum(1) - 1).abs().max()))
            for i, row in enumerate(batch):
                records.append({"sample_id": row["sample_id"], "group_id": row["group_id"], "valid": True,
                                "raw_progress": row["raw_progress"], "prediction": float(result["reading"][i]),
                                "physical_prediction": row["prediction"],
                                "delta_radians": row["delta_radians"], "span_radians": row["span_radians"],
                                "invalid_mass": float(result["invalid_mass"][i]),
                                "outside_arc_mass": float(result["outside_arc_mass"][i])})
    with (args.output / "predictions_geometry.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    metadata = {"protocol": "relative_state_grid256_ordered_span_v1", "population": len(records),
                "input": str(args.labels), "labels_used": ["delta_radians", "span_radians"],
                "scalar_targets_or_baseline_read": False, "pretrained_model_loaded": False,
                "network_forward_images": 0, "training_steps": 0, "gpu_calls": 0,
                "bins": bins, "delta_grid": "circular:2*pi*j/K", "span_grid": "ordered:2*pi*j/(K-1)",
                "invalid_rule": "both ordered-span endpoints retain all mass as uniform progress",
                "maximum_distribution_mass_error": maximum_mass_error,
                "maximum_invalid_mass": max(r["invalid_mass"] for r in records),
                "cpu_wall_seconds": time.perf_counter() - started}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
