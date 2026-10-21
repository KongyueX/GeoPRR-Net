"""Score saved target-blind geometry predictions against frozen source rows."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


def summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {"samples": len(errors), "nmae_percent_fs": float(100 * errors.mean()),
            "median_percent_fs": float(100 * np.median(errors)),
            "p90_percent_fs": float(100 * np.quantile(errors, .9)),
            "p95_percent_fs": float(100 * np.quantile(errors, .95)),
            "tail10_percent_fs": float(100 * np.sort(errors)[-math.ceil(.1 * len(errors)):].mean()),
            "maximum_percent_fs": float(100 * errors.max()),
            **{f"acc_at_{k}_percent": float(100 * np.mean(errors <= k / 100)) for k in (1, 2, 5)},
            **{f"above_{k}_percent_fs": int(np.sum(errors > k / 100)) for k in (1, 2, 5)}}


def score(predictions, reference, output):
    with gzip.open(reference, "rt", newline="") if reference.suffix == ".gz" else reference.open(newline="") as f:
        original = list(csv.DictReader(f))
    refs = {r["sample_id"]: r for r in original}
    predicted = [json.loads(l) for l in predictions.read_text().splitlines() if l.strip()]
    pred = {r["sample_id"]: r for r in predicted}
    if len(original) != 1506 or len(refs) != 1506 or len(predicted) != 1506 or set(pred) != set(refs):
        raise ValueError("saved target-blind output differs from full1506 reference")
    records = []
    for ref in original:
        row = pred[ref["sample_id"]]
        target = float(ref["target"])
        valid = bool(row["valid"])
        records.append({**row, "target": target,
                        "absolute_error": abs(float(row["prediction"]) - target) if valid else 1.,
                        "raw_absolute_error": abs(float(row["raw_progress"]) - target) if valid else 1.,
                        "native_error": float(ref["native_reference_absolute_error"]),
                        "r8_error": float(ref["r8_reference_absolute_error"])})
    e = np.array([r["absolute_error"] for r in records])
    raw = np.array([r["raw_absolute_error"] for r in records])
    native = np.array([r["native_error"] for r in records])
    r8 = np.array([r["r8_error"] for r in records])
    common = (native <= .01) & (r8 <= .01)
    own = r8 <= .01
    if common.sum() != 1495 or own.sum() != 1497:
        raise ValueError("ordinary population differs from preregistered source cohort")
    sums = {"candidate": summary(e), "native": summary(native), "r8": summary(r8)}
    ordinary = {"candidate": summary(e[common]), "native": summary(native[common]), "r8": summary(r8[common]),
                "r8_ordinary_candidate": summary(e[own]), "r8_ordinary_reference": summary(r8[own])}
    pairs = {}
    for name, base in (("native", native), ("r8", r8)):
        pairs[name] = {"better": int(np.sum(e < base)), "worse": int(np.sum(e > base)),
                       "equal": int(np.sum(e == base)),
                       "relative_nmae_change_percent": float(100 * (e.mean() / base.mean() - 1))}
        for k in (1, 2, 5):
            pairs[name][f"above{k}"] = {
                "rescued": [r["sample_id"] for r, a, b in zip(records, e, base) if a <= k / 100 < b],
                "new": [r["sample_id"] for r, a, b in zip(records, e, base) if b <= k / 100 < a]}
    tolerance = 1e-12
    checks = {"valid1506": all(r["valid"] for r in records), "raw_error_at_most1_percent": bool(raw.max() <= .01),
              "full_nmae_no_worse_r8": bool(e.mean() <= r8.mean() + tolerance),
              "full_tail_no_worse_r8": sums["candidate"]["tail10_percent_fs"] <= sums["r8"]["tail10_percent_fs"] + tolerance,
              "maximum_below_both": bool(e.max() < min(native.max(), r8.max())),
              "all_old_severe_union_rescued": bool(np.all(e[(native > .02) | (r8 > .02)] <= .02)),
              "no_new_threshold_errors": all(not pairs[name][f"above{k}"]["new"] for name in pairs for k in (1, 2, 5)),
              "common_mean_no_worse_both": ordinary["candidate"]["nmae_percent_fs"] <= min(ordinary["native"]["nmae_percent_fs"], ordinary["r8"]["nmae_percent_fs"]) + tolerance,
              "common_median_no_worse_both": ordinary["candidate"]["median_percent_fs"] <= min(ordinary["native"]["median_percent_fs"], ordinary["r8"]["median_percent_fs"]) + tolerance,
              "common_p90_no_worse_both": ordinary["candidate"]["p90_percent_fs"] <= min(ordinary["native"]["p90_percent_fs"], ordinary["r8"]["p90_percent_fs"]) + tolerance,
              "r8_ordinary_mean_protected": ordinary["r8_ordinary_candidate"]["nmae_percent_fs"] <= ordinary["r8_ordinary_reference"]["nmae_percent_fs"] + tolerance}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    result = {"source_validation_development_only": True, "target_join_after_saved_predictions": True,
              "summary": sums, "ordinary": ordinary, "paired": pairs, "checks": checks,
              "numerical_gate_passed": all(checks.values()),
              "interpretation": "geometry-label solver only; does not establish RGB model accuracy or novelty",
              "model_forwards": 0, "training_steps": 0}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    score(a.predictions, a.reference, a.output)
