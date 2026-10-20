"""Run the single-seed relative-state candidate and gated matched control."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary


PROTOCOL = "geoprr_v2_relative_state_matched_source_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    home = args.home
    output = home / "runs/geoprr_v2_tail/iteration_v4_relative_state"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=PROTOCOL)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        source = home / "rendering/compositional_source_v1/manifests"
        labels = home / "diagnostics/relative_state_labels_v1_20261021"
        reference = home / "diagnostics/gt_atlas_full1506_v1/predictions.csv"
        initial = home / "runs/geoprr_v2_tail/iteration_v3_compositional/readers/supervised_s20262020/best.pt"
        preflight = json.loads((home / "diagnostics/physical_interface_v2_20261021/relative_state_gpu_preflight.json").read_text())
        grid = json.loads((home / "diagnostics/physical_interface_v2_20261021/relative_state_grid_ordered256_scored_v1/summary.json").read_text())
        if preflight["status"] != "passed" or preflight["optimizer_steps"] != 0 or not grid["numerical_gate_passed"]:
            raise ValueError("required interface or runtime preflight did not pass")
        configs = {arm: json.loads((run.repo / f"experiments/configs/geoprr_v2_relative_state_{arm}.json").read_text())
                   for arm in ("joint", "independent")}
        protocol_path = output / "protocol.json"
        previous_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if previous_protocol and previous_protocol["configs"] != configs:
            raise ValueError("formal configurations differ from the frozen queue")
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for filename in ("geoprr_v2_relative_state.py", "train_geoprr_v2_relative_state.py",
                             "summarize_geoprr_v2_relative_state.py", "run_geoprr_v2_relative_state.py"):
                shutil.copy2(run.repo / "experiments" / filename, snapshot / filename)
        write_json(output / "protocol.json", {"protocol": PROTOCOL, "seed": 20262020, "epochs_per_arm": 65,
                   "configs": configs, "code_revision": os.environ.get("GEOPRR_CODE_REVISION", previous_protocol.get("code_revision")),
                   "arms": ["joint", "independent_if_joint_gate_passes"], "primary_readout": "full_native_probability_mean",
                   "checkpoint_selection": "complete1506 source NMAE, earliest strict minimum",
                   "initial_shared_checkpoint": str(initial), "labels": str(labels),
                   "source_images_unique": 18590, "epoch_donors": 22686, "epoch_recipients": 8192,
                   "epoch_images": 30878, "epoch_optimizer_batches": 709,
                   "scope": "source-development only, no official test or automatic seed expansion",
                   "continuation": "joint must pass predeclared ordinary/tail/new-error gates before matched control"})
        results = {}
        try:
            for arm in ("joint", "independent"):
                reader = output / "readers" / f"{arm}_s20262020"
                config = run.repo / f"experiments/configs/geoprr_v2_relative_state_{arm}.json"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2_relative_state", "--config", str(config),
                           "--train-manifest", str(source / "train.jsonl"),
                           "--val-manifest", str(run.manifests / "validation.jsonl"),
                           "--train-labels", str(labels / "train.jsonl"), "--val-labels", str(labels / "validation.jsonl"),
                           "--pair-manifest", str(source / "train_edges.jsonl"),
                           "--init-checkpoint", str(initial), "--output", str(reader)]
                if (reader / "latest.pt").exists():
                    command += ["--resume", str(reader / "latest.pt")]
                step = f"train_{arm}_s20262020"
                previous = run.state["steps"].get(step, {})
                if previous.get("status") not in ("running", "complete"):
                    while compute_processes():
                        run.state.update(status="waiting_for_gpu", current_step=step)
                        run.save()
                        time.sleep(30)
                run.state.update(status="running", current_step=step)
                run.save()
                run.step(step, command, reader / "completed.json")
                complete = json.loads((reader / "completed.json").read_text())
                metadata = json.loads((reader / "run.json").read_text())
                if complete["epochs"] != 65 or complete["seed"] != 20262020 or metadata["config"] != json.loads(config.read_text()):
                    raise ValueError("completed reader differs from formal configuration")
                if complete["totals"]["train_batches"] != 65 * 709 or complete["totals"]["donor_images"] != 65 * 22686 or complete["totals"]["recipient_images"] != 65 * 8192:
                    raise ValueError("completed reader exposure differs")
                result = compare_to_ordinary(reader / "best_predictions.csv.gz", reference)
                result.update(best_epoch=complete["best_epoch"], parameters=metadata["parameters"], costs=complete)
                write_json(output / "validation_evaluation" / arm / "summary.json", result)
                results[arm] = result
                if arm == "joint" and not result["gate_passed"]:
                    run.state.update(status="validation_rejected", deferred_work=["matched_control", "other_seeds", "official_test"],
                                     next_action="preserve negative result and reassess mechanism without loss/window/seed expansion")
                    write_json(output / "validation_decision.json", {"decision": "joint_validation_rejected", **result})
                    run.save()
                    return
            joint, independent = results["joint"]["summary"]["candidate"], results["independent"]["summary"]["candidate"]
            checks = {"joint_nmae_at_least5_percent_better": joint["nmae_percent_fs"] <= .95 * independent["nmae_percent_fs"],
                      "joint_tail_not_worse": joint["tail10_percent_fs"] <= independent["tail10_percent_fs"],
                      "joint_above5_not_worse": joint["above_5_percent_fs"] <= independent["above_5_percent_fs"]}
            decision = {"decision": "promising_source_mechanism" if all(checks.values()) else "coupling_not_supported",
                        "checks": checks, "joint": joint, "independent": independent,
                        "test_used": False, "seed_expansion": False,
                        "interpretation": "single-seed source-development screen, not significance or real-domain evidence"}
            write_json(output / "validation_decision.json", decision)
            run.state.update(status="complete", validation_decision=decision["decision"])
            run.save()
        except Exception as exc:
            run.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            run.save()
            raise


if __name__ == "__main__":
    main()
