"""Run the first-seed equal-exposure state-composition comparison."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from experiments.prepare_geoprr_v2_compositional_data import PROTOCOL
from experiments.run_geoprr_v2_decoupled import evaluate
from experiments.run_geoprr_v2_optimization import OptimizationRun, read_jsonl, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def validation_decision(composition: dict, consistency: dict, supervised: dict, original: dict) -> dict:
    for result in (composition, consistency, supervised, original):
        if result["smoke_only"] or result["summary"]["samples"] != 1506 or result.get("modal_radius", 0):
            raise ValueError("the decision requires complete native fixed source-validation evaluations")
    a, b = composition["summary"], consistency["summary"]
    checks = {"native_reduction_at_least_5_percent_vs_consistency": a["nmae"] <= 0.95 * b["nmae"],
              "severe_count_not_increased_vs_consistency": a["errors_above_5_percent_fs"] <= b["errors_above_5_percent_fs"],
              "tail10_not_increased_vs_consistency": a["tail10_nmae"] <= b["tail10_nmae"],
              "native_strictly_better_than_supervised": a["nmae"] < supervised["summary"]["nmae"],
              "native_strictly_better_than_original": a["nmae"] < original["summary"]["nmae"]}
    return {"decision": "promising_source_validation" if all(checks.values()) else "validation_rejected",
            "checks": checks, "test_used_for_decision": False, "automatic_seed_expansion": False,
            "composition": a, "consistency": b, "supervised": supervised["summary"], "original": original["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    source = args.home / "rendering/compositional_source_v1"
    output = args.home / "runs/geoprr_v2_tail/iteration_v3_compositional"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, [20262020])
        run.state.update(protocol=PROTOCOL, status="waiting_for_source")
        run.save()
        try:
            while True:
                state = json.loads((source / "queue_state.json").read_text())
                if state["status"] == "complete":
                    break
                if state["status"] not in ("running", "waiting_for_gpu"):
                    raise RuntimeError(f"source-generation queue is {state['status']}; inspect before training")
                time.sleep(30)
            prepared = json.loads((source / "prepared.json").read_text())
            expected_counts = {"groups": 1024, "new_unique_images": 4096, "unique_training_images": 18590,
                               "training_items_per_epoch": 22686, "recipient_forwards_per_epoch": 8192,
                               "image_forwards_per_epoch": 30878, "training_edges": 8192, "heldout_edges": 4096}
            if any(prepared[key] != value for key, value in expected_counts.items()):
                raise ValueError("prepared source data differs from the frozen composition protocol")
            fixed = json.loads((args.home / "runs/geoprr_v2_optimization/manifests/source_ids.json").read_text())
            current = {name: [row["sample_id"] for row in read_jsonl(run.manifests / f"{name}.jsonl")]
                       for name in ("train", "validation", "test")}
            if current != fixed:
                raise ValueError("original source/validation/test identities changed")
            train_rows = read_jsonl(source / "manifests/train.jsonl")
            if train_rows[:14494] != read_jsonl(run.manifests / "train.jsonl"):
                raise ValueError("the supplemental training manifest must preserve all original training rows")
            identities = {"train": [row["sample_id"] for row in train_rows], "validation": fixed["validation"]}
            write_json(output / "protocol.json", {"protocol": PROTOCOL, "seed": 20262020,
                       "arms": ["supervised", "consistency", "composition"], "epochs": 65,
                       "source_data": str(source), "source_counts": prepared, "modal_radius": 8,
                       "selection": "native original source-validation NMAE",
                       "scope": "first seed; no automatic expansion or official-test evaluation",
                       "continuation": "all native comparison checks in validation_decision.json must pass"})
            run.state["status"] = "running"
            run.save()
            results = {}
            for arm in ("supervised", "consistency", "composition"):
                config = run.repo / f"experiments/configs/geoprr_v2_compositional_{arm}.json"
                reader = output / "readers" / f"{arm}_s20262020"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2", "--config", str(config),
                           "--train-manifest", str(source / "manifests/train.jsonl"),
                           "--pair-manifest", str(source / "manifests/train_edges.jsonl"),
                           "--val-manifest", str(run.manifests / "validation.jsonl"), "--seed", "20262020", "--output", str(reader)]
                name = f"train_{arm}_s20262020"
                if (reader / "latest.pt").exists():
                    command += ["--resume", str(reader / "latest.pt")]
                if run.state["steps"].get(name, {}).get("status") not in ("running", "complete"):
                    while compute_processes():
                        run.state["status"] = "waiting_for_gpu"
                        run.save()
                        time.sleep(30)
                    run.state["status"] = "running"
                    run.save()
                run.step(name, command, reader / "completed.json")
                complete = json.loads((reader / "completed.json").read_text())
                if (complete["config"] != json.loads(config.read_text()) or complete["epochs"] != 65
                        or complete["smoke_only"] or complete["sample_ids"] != identities):
                    raise ValueError(f"{arm} completion differs from the fixed source protocol")
                for radius in (0, 8):
                    tag = "modal_r8" if radius else "native"
                    result = evaluate(run, reader / "best.pt", output / "validation_evaluation" / f"{arm}_{tag}",
                                      f"validation_{arm}_{tag}", radius)
                    if not radius:
                        results[arm] = result
                heldout = output / "heldout_combinations" / arm
                run.step(f"heldout_{arm}", [sys.executable, "-m", "experiments.evaluate_geoprr_v2_composition",
                         "--checkpoint", str(reader / "best.pt"), "--manifest", str(source / "manifests/source_groups.jsonl"),
                         "--pair-manifest", str(source / "manifests/heldout_edges.jsonl"), "--output", str(heldout)], heldout / "summary.json")
            original = json.loads((args.home / "runs/geoprr_v2_tail/iteration_v1/validation_evaluation/baseline_s20262020_native/summary.json").read_text())
            decision = validation_decision(results["composition"], results["consistency"], results["supervised"], original)
            write_json(output / "validation_decision.json", decision)
            run.state.update(status="complete", validation_decision=decision["decision"])
            run.save()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
