"""Run the single-seed reference feedback reader and gated local-normalization control."""
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
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary, metrics, read_predictions


PROTOCOL = "geoprr_v2_reference_feedback_matched_source_v1"


def compare_best57(candidate, anchor, ordinary_reference):
    rows, baseline, ordinary = read_predictions(candidate), read_predictions(anchor), read_predictions(ordinary_reference)
    by_id = {row["sample_id"]: row for row in rows}
    old_by_id = {row["sample_id"]: row for row in baseline}
    ordered = [row["sample_id"] for row in ordinary]
    if set(by_id) != set(ordered) or set(old_by_id) != set(ordered):
        raise ValueError("best57 comparison population differs")
    for key in ordered:
        if (by_id[key]["group_id"], float(by_id[key]["target"])) != (old_by_id[key]["group_id"], float(old_by_id[key]["target"])):
            raise ValueError("best57 comparison group or target differs")
    errors = [float(by_id[key]["absolute_error"]) for key in ordered]
    old = [float(old_by_id[key]["absolute_error"]) for key in ordered]
    common = [i for i, row in enumerate(ordinary) if float(row["native_reference_absolute_error"]) <= .01
              and float(row["r8_reference_absolute_error"]) <= .01]
    r8_ordinary = [i for i, row in enumerate(ordinary) if float(row["r8_reference_absolute_error"]) <= .01]
    paired = {f"above{k}": {"rescued_ids": [key for key, a, b in zip(ordered, errors, old) if a <= k/100 < b],
                             "new_error_ids": [key for key, a, b in zip(ordered, errors, old) if b <= k/100 < a]}
              for k in (1, 2, 5)}
    return {"candidate": metrics(errors), "best57": metrics(old), "paired": paired,
            "ordinary_candidate": metrics([errors[i] for i in common]),
            "ordinary_best57": metrics([old[i] for i in common]),
            "r8_ordinary_candidate": metrics([errors[i] for i in r8_ordinary]),
            "r8_ordinary_best57": metrics([old[i] for i in r8_ordinary])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    home = args.home
    output = home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=PROTOCOL)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        source = home / "rendering/compositional_source_v1/manifests"
        reference = home / "diagnostics/gt_atlas_full1506_v1/predictions.csv"
        initial = home / "runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020/best.pt"
        preflight_root = home / "diagnostics/reference_feedback_preflight_v1_20261021"
        preflight = json.loads((preflight_root / "runtime_preflight.json").read_text())
        teacher = json.loads((home / "diagnostics/reference_relation_preflight_v1_20261021/teacher_gate.json").read_text())
        if preflight["status"] != "passed" or preflight["optimizer_steps"] != 0 or not teacher["numerical_gate_passed"]:
            raise ValueError("required teacher-label or RGB-runtime preflight did not pass")
        configs = {arm: json.loads((run.repo / f"experiments/configs/geoprr_v2_reference_feedback_{arm}.json").read_text())
                   for arm in ("feedback", "local")}
        if preflight["configs"] != configs or not preflight["same_initial_parameters"]:
            raise ValueError("runtime/config/initialization differs from frozen paired control")
        left, right = configs["feedback"].copy(), configs["local"].copy()
        left.pop("name"); right.pop("name")
        left = dict(left, model={key: value for key, value in left["model"].items() if key != "reference_normalization"})
        right = dict(right, model={key: value for key, value in right["model"].items() if key != "reference_normalization"})
        if left != right:
            raise ValueError("candidate and control differ beyond reference normalization")
        protocol_path = output / "protocol.json"
        previous_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if previous_protocol and previous_protocol["configs"] != configs:
            raise ValueError("formal configurations differ from the frozen queue")
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for filename in ("geoprr_v2_reference_relation.py", "train_geoprr_v2_reference_relation.py",
                             "summarize_geoprr_v2_relative_state.py", "run_geoprr_v2_reference_feedback.py",
                             "preflight_geoprr_v2_reference_feedback.py"):
                shutil.copy2(run.repo / "experiments" / filename, snapshot / filename)
        write_json(output / "protocol.json", {"protocol": PROTOCOL, "seed": 20262020, "epochs_per_arm": 65,
                   "configs": configs, "code_revision": os.environ.get("GEOPRR_CODE_REVISION", previous_protocol.get("code_revision")),
                   "arms": ["feedback", "local_if_feedback_gate_passes"], "primary_readout": "full_native_probability_mean",
                   "checkpoint_selection": "complete1506 source NMAE, earliest strict minimum",
                   "initial_full_best57_checkpoint": str(initial), "teacher_label_preflight": teacher,
                   "runtime_preflight": preflight,
                   "source_images_unique": 18590, "epoch_donors": 22686, "epoch_recipients": 8192,
                   "epoch_images": 30878, "epoch_optimizer_batches": 709,
                   "scope": "source-development only; other seeds and official test are forbidden",
                   "continuation": "feedback must pass unchanged original G1 before local control"})
        results = {}
        try:
            for arm in ("feedback", "local"):
                reader = output / "readers" / f"{arm}_s20262020"
                config = run.repo / f"experiments/configs/geoprr_v2_reference_feedback_{arm}.json"
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
                result["best57_comparison"] = compare_best57(reader / "best_predictions.csv.gz", initial.parent / "best_predictions.csv.gz", reference)
                result.update(best_epoch=complete["best_epoch"], parameters=metadata["parameters"], costs=complete)
                write_json(output / "validation_evaluation" / arm / "summary.json", result)
                results[arm] = result
                if arm == "feedback" and not result["gate_passed"]:
                    run.state.update(status="validation_rejected", deferred_work=["matched_local_control"],
                                     forbidden_work=["other_seeds", "official_test"],
                                     next_action="preserve negative result and reassess mechanism without loss/window/seed expansion")
                    write_json(output / "validation_decision.json", {"decision": "feedback_validation_rejected", **result})
                    run.save()
                    return
            feedback, local = results["feedback"]["summary"]["candidate"], results["local"]["summary"]["candidate"]
            checks = {"feedback_nmae_at_least5_percent_better_than_local": feedback["nmae_percent_fs"] <= .95 * local["nmae_percent_fs"],
                      "feedback_tail_not_worse": feedback["tail10_percent_fs"] <= local["tail10_percent_fs"],
                      "feedback_above5_not_worse": feedback["above_5_percent_fs"] <= local["above_5_percent_fs"]}
            anchor_comparison = results["feedback"]["best57_comparison"]
            checks["feedback_nmae_no_worse_best57"] = feedback["nmae_percent_fs"] <= anchor_comparison["best57"]["nmae_percent_fs"]
            checks["no_new_above2_5_vs_best57"] = all(not anchor_comparison["paired"][f"above{k}"]["new_error_ids"] for k in (2, 5))
            for key in ("nmae_percent_fs", "median_percent_fs", "p90_percent_fs"):
                checks[f"ordinary_{key}_protected_vs_best57"] = anchor_comparison["ordinary_candidate"][key] <= anchor_comparison["ordinary_best57"][key]
            checks["r8_ordinary_mean_protected_vs_best57"] = anchor_comparison["r8_ordinary_candidate"]["nmae_percent_fs"] <= anchor_comparison["r8_ordinary_best57"]["nmae_percent_fs"]
            decision = {"decision": "promising_source_mechanism" if all(checks.values()) else "feedback_not_supported",
                        "checks": checks, "feedback": feedback, "local": local,
                        "best57_comparison": anchor_comparison,
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
