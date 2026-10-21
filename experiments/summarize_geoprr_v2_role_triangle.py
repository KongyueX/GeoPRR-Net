"""Recompute the complete three-role source result and export scalar-only records."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics


def read_rows(path):
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1506 or len({row["sample_id"] for row in rows}) != 1506:
        raise ValueError("the original1506 unique source population is required")
    return rows


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    index = int(position)
    return ordered[index] + (ordered[min(index + 1, len(ordered) - 1)] - ordered[index]) * (position - index)


def metrics(errors, population):
    values = [errors[key] for key in population]
    return {"samples": len(values), "nmae_percent_fs": statistics.mean(values) * 100,
            "median_percent_fs": quantile(values, .5) * 100,
            "p90_percent_fs": quantile(values, .9) * 100,
            "p95_percent_fs": quantile(values, .95) * 100,
            "tail10_percent_fs": statistics.mean(sorted(values, reverse=True)[:math.ceil(.1 * len(values))]) * 100,
            "maximum_percent_fs": max(values) * 100, "maximum_error_id": max(population, key=errors.get),
            **{f"acc_at_{t}_percent": sum(error <= t / 100 for error in values) / len(values) * 100 for t in (1, 2, 5)},
            **{f"above_{t}_percent_fs": sum(error > t / 100 for error in values) for t in (1, 2, 5)}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with gzip.open(path, "wt", newline="") if path.suffix == ".gz" else path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("refine-logs/geoprr_method_value_20261021/network_role_triangle_v1/completed_full"))
    parser.add_argument("--anchor", type=Path, default=Path("docs/data/geoprr_v2_reference_feedback_20261021/formal_paired_rows.csv.gz"))
    parser.add_argument("--output", type=Path, default=Path("docs/data/geoprr_v2_role_triangle_20261021"))
    args = parser.parse_args(argv)
    raw, output = args.raw, args.output
    current, anchors = read_rows(raw / "best_predictions.csv.gz"), read_rows(args.anchor)
    by_id, old_by_id = ({row["sample_id"]: row for row in rows} for rows in (current, anchors))
    ids = [row["sample_id"] for row in current]
    if set(ids) != set(old_by_id):
        raise ValueError("candidate and frozen reference IDs differ")
    errors = {name: {} for name in ("native", "r8", "best57", "best48", "candidate")}
    predictions = {name: {} for name in errors}
    saved_error_difference = 0.0
    for key in ids:
        row, old = by_id[key], old_by_id[key]
        if (row["group_id"], float(row["target"])) != (old["group_id"], float(old["target"])):
            raise ValueError("candidate ID/group/target differs from the frozen reference")
        if row["valid"] != "True" or row["input_valid"] != "True":
            raise ValueError("this completed result must retain all1506 valid rows")
        target = float(row["target"])
        predictions["candidate"][key] = float(row["prediction"])
        errors["candidate"][key] = abs(predictions["candidate"][key] - target)
        saved_error_difference = max(saved_error_difference, abs(errors["candidate"][key] - float(row["absolute_error"])))
        for name in ("native", "r8", "best57", "best48"):
            source_name = "candidate" if name == "best48" else name
            predictions[name][key] = float(old[f"{source_name}_prediction"])
            errors[name][key] = abs(predictions[name][key] - target)
            if abs(errors[name][key] * 100 - float(old[f"{source_name}_error_percent_fs"])) > 1e-10:
                raise ValueError("frozen anchor prediction/error differs")
    if saved_error_difference > 1e-12:
        raise ValueError("saved candidate prediction/error differs")
    common = [key for key in ids if max(errors["native"][key], errors["r8"][key]) <= .01]
    r8_ordinary = [key for key in ids if errors["r8"][key] <= .01]
    severe = [key for key in ids if max(errors["native"][key], errors["r8"][key]) > .02]
    if (len(common), len(r8_ordinary), len(severe)) != (1495, 1497, 6):
        raise ValueError("frozen ordinary or severe populations differ")
    full = {name: metrics(values, ids) for name, values in errors.items()}
    ordinary = {label: {name: metrics(values, population) for name, values in errors.items()}
                for label, population in (("common1495", common), ("r8ordinary1497", r8_ordinary))}
    paired, transitions = {}, []
    for name in ("native", "r8", "best57", "best48"):
        old, new = errors[name], errors["candidate"]
        paired[name] = {"better": sum(new[key] < old[key] for key in ids),
                        "worse": sum(new[key] > old[key] for key in ids),
                        "equal": sum(new[key] == old[key] for key in ids)}
        for threshold in (1, 2, 5):
            rescued = [key for key in ids if new[key] <= threshold / 100 < old[key]]
            added = [key for key in ids if old[key] <= threshold / 100 < new[key]]
            paired[name][f"above{threshold}"] = {"rescued_ids": rescued, "new_error_ids": added}
            transitions.extend({"reference": name, "threshold_percent_fs": threshold, "change": change,
                                "sample_id": key, "old_error_percent_fs": old[key] * 100,
                                "new_error_percent_fs": new[key] * 100}
                               for change, population in (("rescued", rescued), ("new", added)) for key in population)
    checks = {"all1506_valid": True, "nmae_no_worse_r8": full["candidate"]["nmae_percent_fs"] <= full["r8"]["nmae_percent_fs"],
              "tail10_no_worse_r8": full["candidate"]["tail10_percent_fs"] <= full["r8"]["tail10_percent_fs"],
              "maximum_below_both": full["candidate"]["maximum_percent_fs"] < min(full[name]["maximum_percent_fs"] for name in ("native", "r8")),
              "old_severe_union_rescued": all(errors["candidate"][key] <= .02 for key in severe),
              "no_new_above1_2_5": all(not paired[name][f"above{t}"]["new_error_ids"] for name in ("native", "r8") for t in (1, 2, 5)),
              **{f"ordinary_{metric}_protected": ordinary["common1495"]["candidate"][metric] <= min(ordinary["common1495"][name][metric] for name in ("native", "r8"))
                 for metric in ("nmae_percent_fs", "median_percent_fs", "p90_percent_fs")},
              "r8_ordinary_mean_protected": ordinary["r8ordinary1497"]["candidate"]["nmae_percent_fs"] <= ordinary["r8ordinary1497"]["r8"]["nmae_percent_fs"]}
    protection = {"nmae_no_worse_best48": full["candidate"]["nmae_percent_fs"] <= full["best48"]["nmae_percent_fs"],
                  "no_new_above2_5_vs_best48": all(not paired["best48"][f"above{t}"]["new_error_ids"] for t in (2, 5)),
                  "r8_ordinary_mean_protected_vs_best48": ordinary["r8ordinary1497"]["candidate"]["nmae_percent_fs"] <= ordinary["r8ordinary1497"]["best48"]["nmae_percent_fs"],
                  **{f"ordinary_{metric}_protected_vs_best48": ordinary["common1495"]["candidate"][metric] <= ordinary["common1495"]["best48"][metric]
                     for metric in ("nmae_percent_fs", "median_percent_fs", "p90_percent_fs")}}
    decision = json.loads((raw / "validation_decision.json").read_text())
    if checks != decision["checks"] or protection != decision["best48_protection"]["checks"]:
        raise ValueError("recomputed gates differ from the frozen controller")
    for name, controller_name in (("candidate", "candidate"), ("native", "ordinary_native"), ("r8", "ordinary_r8")):
        if any(abs(full[name][key] - value) > 1e-11 for key, value in decision["summary"][controller_name].items()):
            raise ValueError("recomputed complete1506 metrics differ from the controller")
    for name in ("native", "r8"):
        if paired[name] != decision["paired"][name]:
            raise ValueError("recomputed threshold IDs differ from the controller")
    for name, controller_name in (("candidate", "candidate"), ("native", "ordinary_native"), ("r8", "ordinary_r8")):
        if any(abs(ordinary["common1495"][name][key] - value) > 1e-11
               for key, value in decision["ordinary"][controller_name].items()):
            raise ValueError("recomputed common1495 metrics differ from the controller")
    if any(abs(ordinary["r8ordinary1497"]["candidate"][key] - value) > 1e-11
           for key, value in decision["ordinary"]["r8_cohort_candidate"].items()):
        raise ValueError("recomputed r8ordinary1497 metrics differ from the controller")
    if any(paired["best48"][f"above{t}"] != decision["best48_comparison"]["paired"][f"above{t}"] for t in (1, 2, 5)):
        raise ValueError("recomputed best48 paired IDs differ from the controller")
    history = [json.loads(line) for line in (raw / "history.jsonl").read_text().splitlines()]
    complete = json.loads((raw / "completed.json").read_text())
    config = json.loads((raw / "config.json").read_text())
    protocol = json.loads((raw / "protocol.json").read_text())
    queue = json.loads((raw / "queue_state.json").read_text())
    run = json.loads((raw / "run.json").read_text())
    if complete["epochs"] != 65 or complete["seed"] != 20262020:
        raise ValueError("formal epochs or unique authorized seed differs")
    if not protocol["teacher_label_preflight"]["numerical_gate_passed"] or not decision["original_G1"]["G0_label_gate_passed"]:
        raise ValueError("reused original G0 did not pass")
    if [row["epoch"] for row in history] != list(range(1, 66)):
        raise ValueError("complete65 history is required")
    if min(history, key=lambda row: row["validation"]["nmae"])["epoch"] != complete["best_epoch"] or complete["best_epoch"] != 64:
        raise ValueError("checkpoint is not the earliest complete1506 minimum")
    if abs(complete["best_nmae"] * 100 - full["candidate"]["nmae_percent_fs"]) > 1e-11:
        raise ValueError("best checkpoint NMAE differs from saved predictions")
    for field in ("train_batches", "optimizer_updates", "amp_skipped_updates", "donor_images", "recipient_images"):
        if sum(row[field] for row in history) != complete["totals"][field]:
            raise ValueError("history and completed counters differ")
    for field, total_field in (("training_seconds", "training_wall_seconds"), ("validation_seconds", "validation_wall_seconds")):
        if abs(sum(row[field] for row in history) - complete["totals"][total_field]) > 1e-8:
            raise ValueError("history and completed timings differ")
    if any((row["train_batches"], row["donor_images"], row["recipient_images"]) != (709, 22686, 8192) for row in history):
        raise ValueError("per-epoch exposure differs")
    if sum(row["total_image_forwards"] for row in history) != 2007070 or complete["totals"]["validation_images"] != 65 * 1506:
        raise ValueError("training or validation forward count differs")
    if run["parameters"] != 7461316 or run["config"] != config or config != protocol["configs"]["full"]:
        raise ValueError("completed model configuration or parameter count differs")
    if queue["status"] != "validation_rejected" or set(queue["steps"]) != {"train_full_s20262020"}:
        raise ValueError("separable must remain skipped after rejection")
    step = queue["steps"]["train_full_s20262020"]
    if step["status"] != "complete" or step["exit_code"] != 0:
        raise ValueError("formal full arm did not complete normally")
    cost = {**complete, "training_image_forwards": 2007070,
            "epoch_elapsed_sum_seconds": sum(row["elapsed_seconds"] for row in history),
            "peak_cuda_allocated_bytes": max(row["peak_cuda_memory_bytes"] for row in history),
            "peak_cuda_allocated_epoch": max(history, key=lambda row: row["peak_cuda_memory_bytes"])["epoch"],
            "peak_cuda_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in history),
            "started_at_utc": step["started_at"], "completed_at_utc": step["finished_at"]}
    counterfactual = {key: abs(float(by_id[key]["counterfactual_reading"]) - float(by_id[key]["target"])) for key in ids}
    counterfactual_summary = {"scope": "same trained triangle checkpoint, product of relative-J marginals; not the factor-separable trained control",
                              "metrics": metrics(counterfactual, ids),
                              "maximum_normalized_reading_difference": max(abs(float(row["counterfactual_reading"]) - float(row["prediction"])) for row in current),
                              "threshold_changes": {str(t): sum((counterfactual[key] > t / 100) != (errors["candidate"][key] > t / 100) for key in ids) for t in (1, 2, 5)}}
    rows = []
    scalar_fields = ("valid", "input_valid", "reference_entropy", "reference_max", "joint_vs_product_state_l1", "counterfactual_reading")
    for key in ids:
        row = by_id[key]
        rows.append({**{field: row[field] for field in ("sample_id", "group_id", "target")},
                     **{f"{name}_prediction": predictions[name][key] for name in predictions},
                     **{f"{name}_error_percent_fs": errors[name][key] * 100 for name in errors},
                     **{field: row[field] for field in scalar_fields}, "common_ordinary1495": key in common,
                     "r8_ordinary1497": key in r8_ordinary, "old_severe_union6": key in severe})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "formal_paired_rows.csv.gz", rows)
    write_csv(output / "formal_threshold_transitions.csv", transitions)
    write_csv(output / "formal_old_severe.csv", [row for row in rows if row["old_severe_union6"]])
    write_csv(output / "formal_worst151.csv", sorted(rows, key=lambda row: -row["candidate_error_percent_fs"])[:151])
    prediction_fields = ("sample_id", "group_id", "target", "prediction", "valid", "input_valid", "absolute_error",
                         "invalid_mass", "outside_arc_mass", "failure", "reference_entropy", "reference_max",
                         "joint_vs_product_state_l1", "counterfactual_reading")
    write_csv(output / "formal_predictions.csv.gz", [{field: row[field] for field in prediction_fields} for row in current])
    (output / "formal_history.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))
    write_json(output / "formal_config.json", config)
    write_json(output / "formal_completed.json", complete)
    run_fields = ("protocol", "train_unique_images", "train_items", "expected_full_epoch", "validation_images", "model_selection",
                  "initialization", "parameters", "shared_parameters", "head_parameters", "device", "gpu", "torch", "cuda",
                  "inference_input", "recipient_batching", "comparison")
    write_json(output / "formal_run_metadata.json", {field: run[field] for field in run_fields})
    protocol_fields = ("protocol", "seed", "epochs_per_arm", "configs", "code_revision", "arms", "primary_readout",
                       "checkpoint_selection", "initial_full_best48_checkpoint", "parameters_per_arm", "source_images_unique",
                       "epoch_donors", "epoch_recipients", "epoch_images", "epoch_optimizer_batches", "original_G1",
                       "best48_protection", "counterfactual", "reference_summary", "scope", "continuation", "private_artifacts")
    public_protocol = {field: protocol[field] for field in protocol_fields}
    public_protocol.update(teacher_label_gate_passed=protocol["teacher_label_preflight"]["numerical_gate_passed"],
                           runtime_preflight="runtime_preflight.json")
    write_json(output / "formal_protocol.json", public_protocol)
    summary = {"status": "complete_gate_rejected", "seed": 20262020, "epochs": 65, "best_epoch": 64,
               "code_revision": protocol["code_revision"], "scope": "single-seed complete1506 source development; repeatedly viewed; no official test or real transfer",
               "metrics": full, "ordinary": ordinary, "paired": paired,
               "gate": {"checks": checks, "RGB_passed": sum(checks.values()), "RGB_total": 10,
                        "G0_label_gate_passed": True, "combined_passed": sum(checks.values()) + 1, "combined_total": 11, "passed": all(checks.values())},
               "best48_protection": {"checks": protection, "passed_count": sum(protection.values()), "total": 6, "passed": all(protection.values())},
               "old_severe_union6": [{"sample_id": key, **{f"{name}_error_percent_fs": values[key] * 100 for name, values in errors.items()}} for key in severe],
               "relative_improvement_percent": {reference: {metric: (full[reference][metric] - full["candidate"][metric]) / full[reference][metric] * 100
                                                           for metric in ("nmae_percent_fs", "tail10_percent_fs", "maximum_percent_fs")}
                                                for reference in ("r8", "best57", "best48")},
               "control": "separable skipped by predeclared originalG1 and best48 protection; no matched training comparison",
               "cost": cost, "same_checkpoint_counterfactual": counterfactual_summary,
               "limitations": ["continued optimization and exposure differ from best48; trained factor-separable control was skipped, so association causality is not identified",
                               "one seed, repeatedly viewed source development only; no significance or real transfer",
                               "old severe6 rescued at2%, but sync3172 is a new above2 error relative to best48 and ordinary r8"]}
    write_json(output / "formal_summary.json", summary)
    write_json(output / "formal_numeric_verification.json", {
        "status": "passed", "scope": "deterministic recomputation from saved complete1506 predictions and65history; not an independent semantic audit",
        "samples": 1506, "ID_group_target": "exact", "saved_error_max_difference": saved_error_difference,
        "all_valid": True, "ordinary_populations": [1495, 1497], "old_severe_population": 6,
        "best_selection": "epoch64 earliest strict minimum among65", "controller_metrics_gates_pairedIDs": "exact within1e-11 scalar tolerance",
        "history_completed_counters": "exact", "public_run_data_contract_exported": False,
        "separable_started": False, "CNN_forwards": 0, "optimizer_steps": 0})
    print(json.dumps({"metrics": full["candidate"], "gate": summary["gate"], "best48_protection": summary["best48_protection"],
                      "cost": cost, "counterfactual": counterfactual_summary}, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
