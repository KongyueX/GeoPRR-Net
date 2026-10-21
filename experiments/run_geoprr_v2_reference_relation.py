"""Run the single-seed reference-conditioned reader and gated matched control."""
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


PROTOCOL = "geoprr_v2_reference_relation_matched_source_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    home = args.home
    output = home / "runs/geoprr_v2_tail/iteration_v5_reference_relation"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=PROTOCOL)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        source = home / "rendering/compositional_source_v1/manifests"
        reference = home / "diagnostics/gt_atlas_full1506_v1/predictions.csv"
        initial = home / "runs/geoprr_v2_tail/iteration_v3_compositional/readers/supervised_s20262020/best.pt"
        preflight_root = home / "diagnostics/reference_relation_preflight_v1_20261021"
        preflight = json.loads((preflight_root / "runtime_preflight.json").read_text())
        teacher = json.loads((preflight_root / "teacher_gate.json").read_text())
        if preflight["status"] != "passed" or preflight["optimizer_steps"] != 0 or not teacher["numerical_gate_passed"]:
            raise ValueError("required teacher-label or RGB-runtime preflight did not pass")
        configs = {arm: json.loads((run.repo / f"experiments/configs/geoprr_v2_reference_relation_{arm}.json").read_text())
                   for arm in ("joint", "product")}
        if preflight["configs"] != configs or not preflight["same_initial_parameters"]:
            raise ValueError("runtime/config/initialization differs from frozen paired control")
        left, right = configs["joint"].copy(), configs["product"].copy()
        left.pop("name"); right.pop("name")
        left = dict(left, model={key: value for key, value in left["model"].items() if key != "coupling"})
        right = dict(right, model={key: value for key, value in right["model"].items() if key != "coupling"})
        if left != right:
            raise ValueError("candidate and control differ beyond aggregation")
        protocol_path = output / "protocol.json"
        previous_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if previous_protocol and previous_protocol["configs"] != configs:
            raise ValueError("formal configurations differ from the frozen queue")
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for filename in ("geoprr_v2_reference_relation.py", "train_geoprr_v2_reference_relation.py",
                             "summarize_geoprr_v2_relative_state.py", "run_geoprr_v2_reference_relation.py",
                             "preflight_geoprr_v2_reference_relation.py"):
                shutil.copy2(run.repo / "experiments" / filename, snapshot / filename)
        write_json(output / "protocol.json", {"protocol": PROTOCOL, "seed": 20262020, "epochs_per_arm": 65,
                   "configs": configs, "code_revision": os.environ.get("GEOPRR_CODE_REVISION", previous_protocol.get("code_revision")),
                   "arms": ["joint", "product_if_joint_gate_passes"], "primary_readout": "full_native_probability_mean",
                   "checkpoint_selection": "complete1506 source NMAE, earliest strict minimum",
                   "initial_shared_checkpoint": str(initial), "teacher_label_preflight": teacher,
                   "runtime_preflight": preflight,
                   "source_images_unique": 18590, "epoch_donors": 22686, "epoch_recipients": 8192,
                   "epoch_images": 30878, "epoch_optimizer_batches": 709,
                   "scope": "source-development only, no official test or automatic seed expansion",
                   "continuation": "joint must pass predeclared ordinary/tail/new-error gates before matched control"})
        results = {}
        try:
            for arm in ("joint", "product"):
                reader = output / "readers" / f"{arm}_s20262020"
                config = run.repo / f"experiments/configs/geoprr_v2_reference_relation_{arm}.json"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2_reference_relation", "--config", str(config),
                           "--train-manifest", str(source / "train.jsonl"),
                           "--val-manifest", str(run.manifests / "validation.jsonl"),
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
            joint, product = results["joint"]["summary"]["candidate"], results["product"]["summary"]["candidate"]
            checks = {"joint_nmae_at_least5_percent_better_than_product": joint["nmae_percent_fs"] <= .95 * product["nmae_percent_fs"],
                      "joint_tail_not_worse": joint["tail10_percent_fs"] <= product["tail10_percent_fs"],
                      "joint_above5_not_worse": joint["above_5_percent_fs"] <= product["above_5_percent_fs"]}
            decision = {"decision": "promising_source_mechanism" if all(checks.values()) else "coupling_not_supported",
                        "checks": checks, "joint": joint, "product": product,
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
