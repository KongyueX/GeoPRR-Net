"""Convert one rendered state pilot and check its source geometry composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.geoprr_v2_data import GeoPRRV2Dataset, load_controlled_pairs, normalized_target
from experiments.prepare_geoprr_v2_data import syncg_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True)
    args = parser.parse_args()
    root = args.group.resolve()
    group = json.loads((root / "group.json").read_text())
    rows = []
    for image in group["images"]:
        row = syncg_row(Path(image["annotation_path"]), root / "images")
        row.update(split="train", group_id=group["layout_id"], diagnostic_only=True)
        row["metadata"]["role"] = "geometry_diagnostic"
        rows.append(row)
    output = root / "manifests"
    output.mkdir(exist_ok=True)
    manifest = output / "source_group_diagnostic.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pairs = []
    for index, row in enumerate(rows):
        partner = rows[(index + 1) % len(rows)]
        pairs.append({"sample_a": row["sample_id"], "sample_b": partner["sample_id"],
                      "kind": "state", "layout_id": group["layout_id"],
                      "factors_a": row["metadata"]["factors"], "factors_b": partner["metadata"]["factors"]})
    pair_path = output / "source_group_pairs_diagnostic.jsonl"
    pair_path.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
    loaded_pairs = load_controlled_pairs(pair_path, rows)
    dataset = GeoPRRV2Dataset(manifest, augment=False)
    samples = [dataset[index] for index in range(len(dataset))]
    for sample in samples[1:]:
        for key in ("field_target", "field_mask", "support_target", "roi_to_canonical"):
            if not np.array_equal(sample[key].numpy(), samples[0][key].numpy()):
                raise ValueError(f"the fixed-scene {key} changed across pointer states")
    composed = []
    for donor, a in enumerate(samples):
        for recipient, b in enumerate(samples):
            weights = b["point_target"].numpy()[0] * a["field_mask"].numpy()[0]
            mass = float(weights.sum())
            prediction = float((a["field_target"].numpy() * weights).sum() / mass)
            target = normalized_target(rows[recipient])
            composed.append({"donor": rows[donor]["sample_id"], "recipient": rows[recipient]["sample_id"],
                             "target": target, "gt_composed_prediction": prediction,
                             "absolute_error": abs(prediction - target), "gt_support_mass": mass})
    report = {
        "diagnostic_only": True, "role": "geometry_diagnostic", "layout_id": group["layout_id"],
        "samples": len(samples), "source_pairs": len(loaded_pairs), "gt_combinations": len(composed),
        "fixed_field_and_camera_equal": True,
        "render_identity_checks": group["identity_checks"],
        "minimum_gt_support_mass": min(row["gt_support_mass"] for row in composed),
        "maximum_gt_composed_error_percent_fs": 100 * max(row["absolute_error"] for row in composed),
        "geometry_audits": [{"sample_id": row["sample_id"], **sample["geometry_audit"]}
                            for row, sample in zip(rows, samples)],
        "gt_compositions": composed,
        "scope": "source generation and label contract only; no learned model, training or generalization result",
    }
    (root / "geometry_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("samples", "source_pairs", "gt_combinations",
                       "fixed_field_and_camera_equal", "maximum_gt_composed_error_percent_fs")}), flush=True)


if __name__ == "__main__":
    main()
