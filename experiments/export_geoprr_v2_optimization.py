"""Export completed refinement evaluations and seed-level summaries."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def export(run: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((run / "evaluation").glob("*/summary.json")):
        result = json.loads(path.read_text())
        if result["smoke_only"] or result["summary"]["samples"] != result["declared_samples"]:
            raise ValueError(f"incomplete/diagnostic evaluation: {path}")
        label, seed_and_mode = path.parent.name.rsplit("_s", 1)
        seed, mode = seed_and_mode.split("_", 1)
        metrics = result["summary"]
        rows.append({"configuration": label, "seed": int(seed), "mode": mode,
                     "checkpoint_epoch": result["checkpoint_epoch"], "samples": metrics["samples"],
                     "nmae_percent_fs": metrics["nmae_percent_fs"],
                     "acc_at_1_percent": 100 * metrics["acc_at_1"],
                     "acc_at_2_percent": 100 * metrics["acc_at_2"],
                     "acc_at_5_percent": 100 * metrics["acc_at_5"],
                     "maximum_error_percent_fs": 100 * metrics["maximum_error"],
                     "tail10_nmae_percent_fs": 100 * metrics["tail10_nmae"],
                     "errors_above_2_percent_fs": metrics["errors_above_2_percent_fs"],
                     "errors_above_5_percent_fs": metrics["errors_above_5_percent_fs"],
                     "failures": metrics["failed"], "parameters": result["parameters"],
                     "reader_parameters": result["reader_parameters"], "router_parameters": result["router_parameters"],
                     "selected_field": result["selected_experts"]["field"],
                     "selected_direct": result["selected_experts"]["direct"],
                     "selected_mean": result["selected_experts"]["mean"],
                     "batched_pipeline_ms_per_image": result["timing"]["pipeline_ms_per_image"],
                     "batched_forward_ms_per_image": result["timing"]["forward_ms_per_image"],
                     "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"]})
    if not rows:
        raise ValueError("no complete evaluation results")
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["configuration"], row["mode"]].append(row)
    seed_summary = []
    for (label, mode), records in sorted(grouped.items()):
        if len(records) == 3:
            seed_summary.append({"configuration": label, "mode": mode, "seed_count": 3,
                                 "nmae_percent_fs_mean": statistics.mean(r["nmae_percent_fs"] for r in records),
                                 "nmae_percent_fs_sample_sd": statistics.stdev(r["nmae_percent_fs"] for r in records),
                                 "acc_at_2_percent_mean": statistics.mean(r["acc_at_2_percent"] for r in records),
                                 "acc_at_5_percent_mean": statistics.mean(r["acc_at_5_percent"] for r in records)})
    result = {"protocol": json.loads((run / "protocol.json").read_text()),
              "evaluations": len(rows), "per_seed": rows, "three_seed": seed_summary,
              "scope": "native SyncG exploratory comparison; source-validation selection only"}
    if (run / "validation_decision.json").exists():
        result["validation_decision"] = json.loads((run / "validation_decision.json").read_text())
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if seed_summary:
        with (output / "three_seed_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_summary[0]))
            writer.writeheader()
            writer.writerows(seed_summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run, args.output)
    print(json.dumps({"evaluations": result["evaluations"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
