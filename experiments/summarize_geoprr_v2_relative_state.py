"""Compare saved full-population relative-state source predictions."""
import csv
import gzip
import math
from pathlib import Path

import numpy as np


def read_predictions(path: Path):
    with gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1506 or len({r["sample_id"] for r in rows}) != 1506:
        raise ValueError("comparison requires all original1506 unique images")
    return rows


def metrics(error):
    a = np.asarray(error, dtype=np.float64)
    return {"samples": len(a), "nmae_percent_fs": float(100 * a.mean()),
            "median_percent_fs": float(100 * np.median(a)), "p90_percent_fs": float(100 * np.quantile(a, .9)),
            "p95_percent_fs": float(100 * np.quantile(a, .95)),
            "tail10_percent_fs": float(100 * np.sort(a)[-math.ceil(.1 * len(a)):].mean()),
            "maximum_percent_fs": float(100 * a.max()),
            **{f"acc_at_{k}_percent": float(100 * np.mean(a <= k / 100)) for k in (1, 2, 5)},
            **{f"above_{k}_percent_fs": int(np.sum(a > k / 100)) for k in (1, 2, 5)}}


def compare_to_ordinary(candidate: Path, reference: Path):
    records, baseline = read_predictions(candidate), read_predictions(reference)
    lookup = {r["sample_id"]: r for r in records}
    if set(lookup) != {r["sample_id"] for r in baseline}:
        raise ValueError("candidate and ordinary source IDs differ")
    records = [lookup[r["sample_id"]] for r in baseline]
    for a, b in zip(records, baseline):
        if a["group_id"] != b["group_id"] or float(a["target"]) != float(b["target"]):
            raise ValueError("candidate target/group differs from ordinary source reference")
        valid = a["valid"].lower() == "true"
        expected = abs(float(a["prediction"]) - float(a["target"])) if valid else 1.
        if abs(expected - float(a["absolute_error"])) > 1e-10:
            raise ValueError("candidate saved prediction/error disagrees")
    error = np.array([float(r["absolute_error"]) for r in records])
    native = np.array([float(r["native_reference_absolute_error"]) for r in baseline])
    modal = np.array([float(r["r8_reference_absolute_error"]) for r in baseline])
    common = (native <= .01) & (modal <= .01)
    modal_ordinary = modal <= .01
    if common.sum() != 1495 or modal_ordinary.sum() != 1497:
        raise ValueError("ordinary precision cohorts changed")
    full = {"candidate": metrics(error), "ordinary_native": metrics(native), "ordinary_r8": metrics(modal)}
    ordinary = {"candidate": metrics(error[common]), "ordinary_native": metrics(native[common]),
                "ordinary_r8": metrics(modal[common]), "r8_cohort_candidate": metrics(error[modal_ordinary]),
                "r8_cohort_reference": metrics(modal[modal_ordinary])}
    paired = {}
    for name, old in (("native", native), ("r8", modal)):
        paired[name] = {"better": int(np.sum(error < old)), "worse": int(np.sum(error > old)),
                        "equal": int(np.sum(error == old))}
        for k in (1, 2, 5):
            paired[name][f"above{k}"] = {
                "rescued_ids": [r["sample_id"] for r, a, b in zip(records, error, old) if a <= k / 100 < b],
                "new_error_ids": [r["sample_id"] for r, a, b in zip(records, error, old) if b <= k / 100 < a]}
    checks = {"all1506_valid": all(r["valid"].lower() == "true" for r in records),
              "nmae_no_worse_r8": error.mean() <= modal.mean(),
              "tail10_no_worse_r8": full["candidate"]["tail10_percent_fs"] <= full["ordinary_r8"]["tail10_percent_fs"],
              "maximum_below_both": error.max() < min(native.max(), modal.max()),
              "old_severe_union_rescued": np.all(error[(native > .02) | (modal > .02)] <= .02),
              "no_new_above1_2_5": all(not paired[name][f"above{k}"]["new_error_ids"] for name in paired for k in (1, 2, 5)),
              **{f"ordinary_{key}_protected": ordinary["candidate"][key] <= min(ordinary["ordinary_native"][key], ordinary["ordinary_r8"][key])
                 for key in ("nmae_percent_fs", "median_percent_fs", "p90_percent_fs")},
              "r8_ordinary_mean_protected": ordinary["r8_cohort_candidate"]["nmae_percent_fs"] <= ordinary["r8_cohort_reference"]["nmae_percent_fs"]}
    checks = {key: bool(value) for key, value in checks.items()}
    return {"source_development_only": True, "candidate_predictions": str(candidate),
            "reference_predictions": str(reference), "summary": full, "ordinary": ordinary,
            "paired": paired, "checks": checks, "gate_passed": all(checks.values()),
            "candidate_severe_ids": [r["sample_id"] for r, e in zip(records, error) if e > .02]}
