"""Checks for matched-roster scoring, ensemble semantics, and saved metadata."""
import csv
import json

import numpy as np
import pytest
import torch

from experiments.revision_diagnostics import (
    CONDITIONS, SEEDS, audit_saved_files, load_predictions, main,
    routing_diagnostics, summarize,
)


def prediction_rows():
    rows = []
    for method in ("full", "candidate"):
        for seed_index, seed in enumerate(SEEDS):
            for sample in range(2):
                for condition in CONDITIONS:
                    prediction = 0.52 if method == "full" else (
                        (0.4, 0.5, 0.6)[seed_index] if sample == 0 else 0.5)
                    rows.append({"method": method, "seed": seed,
                                 "sample_id": f"image_{sample}", "group_id": f"group_{sample}",
                                 "condition": condition, "target": 0.5,
                                 "prediction": prediction, "error": abs(prediction - 0.5),
                                 "status": "success"})
    return rows


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_seed_metrics_ensemble_and_cluster_intervals(tmp_path):
    path = write_predictions(tmp_path / "predictions.csv", prediction_rows())
    result = summarize(load_predictions([path]), replicates=1000)
    summary = {(r["method"], r["aggregation"], r["scope"]): r for r in result["summary"]}
    fitted = summary["candidate", "seed_mean", "all_conditions"]
    ensemble = summary["candidate", "equal_ensemble", "all_conditions"]
    assert fitted["nmae_pct_fs"] == pytest.approx(10 / 3)
    assert fitted["nmae_pct_fs_sample_sd"] == pytest.approx(np.std([5, 0, 5], ddof=1))
    assert fitted["acc_at_5_pct"] == pytest.approx(200 / 3)
    assert (fitted["rows"], fitted["images"], fitted["groups"]) == (12, 2, 2)
    assert ensemble["nmae_pct_fs"] == pytest.approx(0)
    assert ensemble["acc_at_5_pct"] == 100
    assert ensemble["ensemble_members"] == 3
    assert ensemble["nmae_pct_fs_sample_sd"] is None
    pairs = {(r["aggregation"], r["scope"], r["metric"]): r for r in result["paired"]}
    nmae = pairs["seed_mean", "all_conditions", "nmae_pct_fs"]
    assert nmae["candidate_minus_reference"] == pytest.approx(4 / 3)
    assert [nmae["ci95_low"], nmae["ci95_high"]] == pytest.approx([-2, 14 / 3])
    acc = pairs["seed_mean", "all_conditions", "acc_at_5_pct"]
    assert [acc["ci95_low"], acc["ci95_high"]] == pytest.approx([-200 / 3, 0])
    # Six correlated views retain the same group uncertainty as one repeated view.
    for condition in CONDITIONS:
        cell = pairs["seed_mean", condition, "nmae_pct_fs"]
        assert [cell["ci95_low"], cell["ci95_high"]] == pytest.approx([nmae["ci95_low"], nmae["ci95_high"]])


def test_failures_remain_in_denominator_and_fail_ensemble(tmp_path):
    rows = prediction_rows()
    for row in rows:
        if row["method"] == "candidate" and row["seed"] == SEEDS[0] and row["sample_id"] == "image_0":
            row.update(prediction="", error=1.0, status="failure")
    path = write_predictions(tmp_path / "predictions.csv", rows)
    result = summarize(load_predictions([path]), replicates=100)
    ensemble = next(r for r in result["summary"] if r["method"] == "candidate"
                    and r["aggregation"] == "equal_ensemble" and r["scope"] == "all_conditions")
    assert ensemble["nmae_pct_fs"] == 50
    assert ensemble["acc_at_5_pct"] == 50
    assert ensemble["coverage_pct"] == 50
    assert ensemble["rows"] == 12


@pytest.mark.parametrize("change,match", [
    ("missing_row", "roster|six conditions"),
    ("missing_seed", "three source seeds"),
    ("target", "group or target"),
    ("group", "group or target"),
    ("duplicate", "duplicate"),
    ("bad_failure", "error 1.0"),
    ("bad_error", "supplied error"),
])
def test_rejects_unmatched_or_invalid_predictions(tmp_path, change, match):
    rows = prediction_rows()
    if change == "missing_row":
        rows.pop()
    elif change == "missing_seed":
        rows = [r for r in rows if not (r["method"] == "candidate" and r["seed"] == SEEDS[0])]
    elif change == "target":
        rows[-1].update(target=0.6, error=0.1)
    elif change == "group":
        rows[-1]["group_id"] = "different"
    elif change == "duplicate":
        rows.append(dict(rows[-1]))
    elif change == "bad_failure":
        rows[-1].update(status="failure", prediction="", error=0)
    else:
        rows[-1]["error"] = 0.4
    path = write_predictions(tmp_path / "predictions.csv", rows)
    with pytest.raises(ValueError, match=match):
        load_predictions([path])


def test_routing_entropy_is_not_polar_entropy():
    payload = {"scope": "evaluation", "seed": SEEDS[0], "cache": {
        "routing_weights": torch.tensor([[1 / 3] * 3, [1.0, 0.0, 0.0]]),
        "polar_entropy": torch.zeros(2),
        "moment_residual": torch.tensor([-0.002, 0.004]),
    }}
    report = routing_diagnostics(payload)
    assert report["routing"]["normalized_entropy_mean"] == pytest.approx(0.5)
    assert report["routing"]["fraction_max_weight_gt_095"] == 0.5
    assert report["moment_residual"]["signed_mean"] == pytest.approx(0.001)
    assert report["moment_residual"]["absolute_mean"] == pytest.approx(0.003)
    assert report["moment_residual"]["absolute_max"] == pytest.approx(0.004)
    assert routing_diagnostics({"cache": {}})["missing_fields"] == ["routing_weights", "moment_residual"]


def test_audit_preserves_actual_metadata_without_inventing_fields(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"seed": SEEDS[1], "warm_initialization": {"seed": SEEDS[0]},
                "model_state": {"weight": torch.zeros(2, 3)}}, checkpoint)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"outer_test_groups": ["g2"], "train_groups": ["g0", "g1"]}))
    report = audit_saved_files([checkpoint], [split])
    actual = report["checkpoints"][0]["actual_fields"]
    assert actual["seed"] == SEEDS[1]
    assert actual["warm_initialization"]["seed"] == SEEDS[0]
    assert "training_scope" in report["checkpoints"][0]["not_present_at_top_level"]
    assert "inner_validation_groups" not in report["split_files"][0]["actual_fields"]
    assert actual["model_state"]["tensor_fields"]["weight"]["shape"] == [2, 3]


def test_cli_writes_summary_and_audit_outputs(tmp_path):
    path = write_predictions(tmp_path / "predictions.csv", prediction_rows())
    output = tmp_path / "summary"
    assert main(["summarize", "--predictions", str(path), "--output-dir", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["protocol"]["bootstrap_replicates"] == 20000
    assert len(report["summary"]) == 2 * 2 * 7
    assert {p.name for p in output.iterdir()} == {"per_seed.csv", "summary.csv", "paired.csv", "summary.json"}
    split = tmp_path / "split.json"
    split.write_text('{"outer_folds":5}')
    audit = tmp_path / "audit.json"
    assert main(["audit", "--splits", str(split), "--output", str(audit)]) == 0
    assert json.loads(audit.read_text())["split_files"][0]["actual_fields"] == {"outer_folds": 5}


def test_routing_diagnostics_retains_condition_specific_allocations():
    result = routing_diagnostics({'conditions':['clean','blur_moderate'], 'cache':{
        'routing_weights':np.array([[1.,0.,0.],[0.,.5,.5]]),
        'moment_residual':np.array([0.,.002]),'condition_index':np.array([0,1])}})
    assert result['per_condition']['clean']['routing']['normalized_entropy_mean']==0
    assert result['per_condition']['blur_moderate']['moment_residual']['absolute_max']==.002
