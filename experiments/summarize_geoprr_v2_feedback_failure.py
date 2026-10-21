"""Publish scalar-only summaries of the frozen feedback failure diagnosis."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


PREDICTION_FIELDS = ("sample_id", "group_id", "readout", "target", "prediction", "valid", "input_valid",
                     "absolute_error", "invalid_mass", "outside_arc_mass")
DERIVED_FIELDS = ("sample_id", "center_error_ROI_normalized", "SPD_relative_Frobenius") + tuple(
    f"{branch}_{name}" for branch in ("native", "teacher_rectifier") for name in (
        "A_teacher_neighbor_mass", "GT_A_conditional_reading", "GT_A_conditional_error_percent_fs",
        "P_correct_mass_at_GT_A", "B_correct_mass_at_GT_A", "local_A_teacher_neighbor_mass", "A_global_local_L1"))
REFERENCE_FIELDS = ("sample_id", "branch", "reference_probability_rank", "probability", "is_teacher_adjacent",
                    "conditional_reading", "local_probability", "reference_unary", "log_Z_P", "log_Z_B",
                    "conditional_error_percent_fs", "P_correct_mass", "B_correct_mass", "invalid_mass",
                    "outside_low_mass", "outside_high_mass", "signed_joint_product_contribution")
PROTOCOL_FIELDS = ("checkpoint_epoch", "seed", "population", "batch_size", "optimizer_updates", "teacher_scope",
                   "reader_class", "reference_normalization", "capture_reference_feedback", "focus_ids",
                   "code_revision", "teacher_role_subsets")
COMPLETED_FIELDS = ("status", "images", "checkpoint_epoch", "reader_class", "reference_normalization",
                    "capture_reference_feedback", "CNN_image_forwards", "native_head_image_calls",
                    "teacher_rectifier_head_image_calls", "optimizer_updates", "backward_calls", "readout_calls",
                    "readout_image_equivalents", "readouts_per_image", "replay_max_difference", "identity_max_residual",
                    "probability_mass_max_error", "input_failures", "process_wall_seconds",
                    "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "scope")
AUDIT_FIELDS = ("status", "scope", "population", "numeric_max_residuals", "focus_PMF_reconstruction_residuals",
                "minimum_saved_probability", "saved_error_recompute_max", "root_summary_metric_max_difference",
                "readout_metrics", "focus", "cached_GT_A_all1506", "limits", "audit_wall_seconds")


def read_rows(path):
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, fields):
    with gzip.open(path, "wt", newline="") if path.suffix == ".gz" else path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(error):
    error = np.asarray(error, dtype=np.float64)
    return {"samples": len(error), "nmae_percent_fs": float(100*error.mean()),
            "median_percent_fs": float(100*np.median(error)), "p90_percent_fs": float(100*np.quantile(error, .9)),
            "p95_percent_fs": float(100*np.quantile(error, .95)),
            "tail10_percent_fs": float(100*np.sort(error)[-math.ceil(.1*len(error)):].mean()),
            "maximum_percent_fs": float(100*error.max()),
            **{f"acc_at_{k}_percent": float(100*np.mean(error <= k/100)) for k in (1, 2, 5)},
            **{f"above_{k}_percent_fs": int(np.sum(error > k/100)) for k in (1, 2, 5)}}


def summarize(raw, output, reference):
    paired = read_rows(reference)
    ids = [row["sample_id"] for row in paired]
    if len(ids) != 1506 or len(set(ids)) != 1506:
        raise ValueError("requires the frozen original1506 population")
    target = np.array([float(row["target"]) for row in paired])
    masks = {"all1506": np.ones(1506, dtype=bool),
             "common1495": np.array([row["common_ordinary1495"] == "True" for row in paired]),
             "r8ordinary1497": np.array([row["r8_ordinary1497"] == "True" for row in paired]),
             "old_severe6": np.array([row["old_severe_union6"] == "True" for row in paired])}
    if [int(mask.sum()) for mask in masks.values()] != [1506, 1495, 1497, 6]:
        raise ValueError("frozen protection cohorts changed")
    baseline = {name: np.array([float(row[f"{name}_error_percent_fs"])/100 for row in paired])
                for name in ("native", "r8", "best57", "candidate")}
    baseline = dict(zip(("ordinary_native", "ordinary_r8", "v5_best57", "v6_formal_best48"), baseline.values()))
    predictions = read_rows(raw/"predictions.csv.gz")
    names = sorted({row["readout"] for row in predictions})
    if len(names) != 13 or len(predictions) != 13*1506:
        raise ValueError("requires all13 complete diagnostic readouts")
    supplementary = read_rows(raw/"supplementary_cached_predictions.csv.gz")
    all_rows = predictions + supplementary
    readouts, errors, maximum_error_residual = {}, {}, 0.
    for name in sorted({row["readout"] for row in all_rows}):
        lookup = {row["sample_id"]: row for row in all_rows if row["readout"] == name}
        if set(lookup) != set(ids):
            raise ValueError("diagnostic population differs")
        rows = [lookup[key] for key in ids]
        for row, original in zip(rows, paired):
            if row["group_id"] != original["group_id"] or float(row["target"]) != float(original["target"]):
                raise ValueError("diagnostic labels differ")
            if row.get("valid", "True") != "True" or row.get("input_valid", "True") != "True":
                raise ValueError("available full-population diagnosis unexpectedly contains failures")
        prediction = np.array([float(row["prediction"]) for row in rows])
        error = np.abs(prediction-target)
        maximum_error_residual = max(maximum_error_residual,
            float(np.max(np.abs(error-np.array([float(row["absolute_error"]) for row in rows])))))
        errors[name] = error
        readouts[name] = {"kind": "runtime_intervention" if name in names else "cached_posthoc_intervention",
                          "cohorts": {cohort: metrics(error[mask]) for cohort, mask in masks.items()},
                          "maximum_error_id": ids[int(error.argmax())]}
    if maximum_error_residual > 1e-12:
        raise ValueError("saved prediction/error mismatch")
    transition_rows = []
    for name, error in errors.items():
        readouts[name]["paired"] = {}
        for label, old in baseline.items():
            readouts[name]["paired"][label] = {}
            for threshold in (1, 2, 5):
                rescued = [key for key, a, b in zip(ids, error, old) if a <= threshold/100 < b]
                new = [key for key, a, b in zip(ids, error, old) if b <= threshold/100 < a]
                readouts[name]["paired"][label][str(threshold)] = {"rescued": rescued, "new": new}
                for kind, selected in (("rescued", rescued), ("new", new)):
                    transition_rows.extend({"readout": name, "reference": label, "threshold_percent_fs": threshold,
                                            "transition": kind, "sample_id": key} for key in selected)
    derived = read_rows(raw/"derived_rows.csv")
    derived_lookup = {row["sample_id"]: row for row in derived}
    if set(derived_lookup) != set(ids):
        raise ValueError("derived scalar population differs")
    focus_ids = ["sync_2810", "sync_3663", "sync_10304", "sync_3172"]
    focus = {key: {"target": float(target[ids.index(key)]),
                   "errors_percent_fs": {name: float(100*error[ids.index(key)]) for name, error in errors.items()},
                   "derived": {field: float(derived_lookup[key][field]) for field in DERIVED_FIELDS[1:]}}
             for key in focus_ids}
    protocol = json.loads((raw/"protocol.json").read_text())
    completed = json.loads((raw/"completed.json").read_text())
    audit = json.loads((raw/"independent_diagnosis_audit.json").read_text())
    summary = {"status": "complete", "scope": "frozen best48 zero-training conditional diagnosis",
               "formal_RGB_results_unchanged": {name: {c: metrics(e[m]) for c, m in masks.items()}
                                                 for name, e in baseline.items()},
               "readouts": readouts, "focus": focus, "saved_error_recompute_max": maximum_error_residual,
               "reading_scope": {"teacher": "GT coordinate/role interventions; not deployable RGB predictions",
                                  "local_reference": "same frozen weights and conditional evidence, localA normalization deletion",
                                  "MAP_A": "posthoc predicted-reference maximum; no fitted threshold or per-image GT selection",
                                  "native_GT_A_cached": "GT-A reweighting of native per-reference cached readings"}}
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output/"intervention_predictions.csv.gz", predictions, PREDICTION_FIELDS)
    write_rows(output/"cached_posthoc_predictions.csv.gz", supplementary,
               ("sample_id", "group_id", "target", "readout", "prediction", "absolute_error"))
    write_rows(output/"derived_rows.csv.gz", derived, DERIVED_FIELDS)
    write_rows(output/"focus_reference_rows.csv", read_rows(raw/"focus_reference_rows.csv"), REFERENCE_FIELDS)
    write_rows(output/"threshold_transitions.csv.gz", transition_rows,
               ("readout", "reference", "threshold_percent_fs", "transition", "sample_id"))
    write_json(output/"summary.json", summary)
    write_json(output/"focus.json", focus)
    write_json(output/"protocol.json", {key: protocol[key] for key in PROTOCOL_FIELDS})
    write_json(output/"completed.json", {key: completed[key] for key in COMPLETED_FIELDS})
    write_json(output/"independent_numeric_audit.json", {key: audit[key] for key in AUDIT_FIELDS})
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path("docs/data/geoprr_v2_reference_feedback_20261021/formal_paired_rows.csv.gz"))
    args = parser.parse_args()
    result = summarize(args.raw, args.output, args.reference)
    print(json.dumps({"status": result["status"], "runtime_readouts": 13, "population": 1506,
                      "saved_error_recompute_max": result["saved_error_recompute_max"]}))


if __name__ == "__main__":
    main()
