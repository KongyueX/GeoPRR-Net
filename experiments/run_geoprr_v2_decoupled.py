"""Finish the frozen readout comparison, then test separate Q/U supervision."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from experiments.run_geoprr_v2_optimization import OptimizationRun, SEEDS, read_jsonl, write_json


def validation_decision(candidate: dict, baseline: dict) -> dict:
    for result in (candidate, baseline):
        if result["smoke_only"] or result["summary"]["samples"] != 1506:
            raise ValueError("the decision requires complete fixed source-validation evaluations")
    a, b = candidate["summary"], baseline["summary"]
    promising = (a["nmae"] <= 0.95 * b["nmae"]
                 and a["errors_above_5_percent_fs"] <= b["errors_above_5_percent_fs"]
                 and a["tail10_nmae"] <= b["tail10_nmae"])
    return {"decision": "promising_source_validation" if promising else "validation_rejected",
            "rule": "at least 5% lower native source-validation NMAE; no increase in severe-error count or tail10 NMAE",
            "test_used_for_decision": False, "candidate": a, "baseline": b}


def evaluate(run: OptimizationRun, checkpoint: Path, output: Path, name: str, radius: int = 0) -> dict:
    command = [sys.executable, "-m", "experiments.evaluate_geoprr_v2", "--checkpoint", str(checkpoint),
               "--manifest", str(run.manifests / "validation.jsonl"), "--output", str(output)]
    if radius:
        command += ["--modal-radius", str(radius)]
    run.step(name, command, output / "summary.json")
    return json.loads((output / "summary.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    parent = args.home / "runs/geoprr_v2_tail/iteration_v1"
    output = args.home / "runs/geoprr_v2_tail/iteration_v2_decoupled"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, [SEEDS[0]])
        run.state.update(protocol="geoprr_v2_decoupled_qu_source_validation_v1", status="waiting_for_parent")
        run.save()
        print(json.dumps({"event": "wait_for_parent", "parent": str(parent)}), flush=True)
        try:
            while True:
                state = json.loads((parent / "queue_state.json").read_text())
                if state["status"] == "complete":
                    break
                if state["status"] != "running":
                    raise RuntimeError(f"parent queue is {state['status']}; inspect it before continuing")
                time.sleep(30)
            run.state["status"] = "running"
            run.save()
            fixed = json.loads((args.home / "runs/geoprr_v2_optimization/manifests/source_ids.json").read_text())
            identities = {name: [row["sample_id"] for row in read_jsonl(run.manifests / f"{name}.jsonl")]
                          for name in ("train", "validation", "test")}
            if identities != fixed:
                raise ValueError("source-validation iteration must preserve the frozen sample identities")
            write_json(output / "protocol.json", {
                "protocol": run.state["protocol"], "seed": SEEDS[0], "epochs": 65,
                "new_trainable_parameters": 0, "parent": str(parent), "modal_radius": 8,
                "selection": "native source-validation NMAE; no test-based choices",
                "continuation": "at least 5% lower native source-validation NMAE; no increase in severe-error count or tail10 NMAE",
                "scope": "first-seed training candidate; no automatic expansion or official-test evaluation",
            })
            baseline_validation = None
            for seed in SEEDS:
                checkpoint = (args.home / "runs/geoprr_v2/mobilenetv4_conv_medium/field/seed_20262020/best.pt"
                              if seed == SEEDS[0] else parent / "readers" / f"baseline_s{seed}" / "best.pt")
                for radius in (0, 8):
                    tag = "modal_r8" if radius else "native"
                    result = evaluate(run, checkpoint, parent / "validation_evaluation" / f"baseline_s{seed}_{tag}",
                                      f"validation_baseline_s{seed}_{tag}", radius)
                    if seed == SEEDS[0] and radius == 0:
                        baseline_validation = result
            config = run.repo / "experiments/configs/geoprr_v2_tail_decoupled.json"
            reader = output / "readers" / "decoupled_s20262020"
            command = [sys.executable, "-m", "experiments.train_geoprr_v2", "--config", str(config),
                       "--train-manifest", str(run.manifests / "train.jsonl"),
                       "--val-manifest", str(run.manifests / "validation.jsonl"),
                       "--seed", str(SEEDS[0]), "--output", str(reader)]
            if (reader / "latest.pt").exists():
                command += ["--resume", str(reader / "latest.pt")]
            run.step("train_decoupled_s20262020", command, reader / "completed.json")
            complete = json.loads((reader / "completed.json").read_text())
            if (complete["config"] != json.loads(config.read_text()) or complete["epochs"] != 65
                    or complete["smoke_only"] or complete["sample_ids"] != {key: fixed[key] for key in ("train", "validation")}):
                raise ValueError("candidate completion differs from the formal fixed configuration")
            native = evaluate(run, reader / "best.pt", output / "validation_evaluation/decoupled_s20262020_native",
                              "validation_decoupled_s20262020_native")
            evaluate(run, reader / "best.pt", output / "validation_evaluation/decoupled_s20262020_modal_r8",
                     "validation_decoupled_s20262020_modal_r8", 8)
            decision = validation_decision(native, baseline_validation)
            write_json(output / "validation_decision.json", decision)
            run.state.update(status="complete", validation_decision=decision["decision"])
            run.save()
            print(json.dumps({"event": "source_validation_decision", "decision": decision["decision"]}), flush=True)
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
