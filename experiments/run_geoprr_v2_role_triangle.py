"""Run the single-seed three-role association and its gated matched deletion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_reference_feedback import compare_best57
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary


PROTOCOL = "geoprr_v2_role_triangle_matched_source_v1"


def compare_best48(candidate, anchor, ordinary_reference):
    comparison = compare_best57(candidate, anchor, ordinary_reference)
    return {key.replace("best57", "best48"): value for key, value in comparison.items()}


def best48_protection(comparison):
    checks = {
        "nmae_no_worse_best48": comparison["candidate"]["nmae_percent_fs"] <= comparison["best48"]["nmae_percent_fs"],
        "no_new_above2_5_vs_best48": all(not comparison["paired"][f"above{k}"]["new_error_ids"] for k in (2, 5)),
        "r8_ordinary_mean_protected_vs_best48": comparison["r8_ordinary_candidate"]["nmae_percent_fs"] <= comparison["r8_ordinary_best48"]["nmae_percent_fs"],
    }
    for key in ("nmae_percent_fs", "median_percent_fs", "p90_percent_fs"):
        checks[f"ordinary_{key}_protected_vs_best48"] = comparison["ordinary_candidate"][key] <= comparison["ordinary_best48"][key]
    return {"checks": checks, "gate_passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    home = args.home
    output = home / "runs/geoprr_v2_tail/iteration_v7_role_triangle"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=PROTOCOL)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        source = home / "rendering/compositional_source_v1/manifests"
        reference = home / "diagnostics/gt_atlas_full1506_v1/predictions.csv"
        initial = home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt"
        preflight_root = home / "diagnostics/role_triangle_preflight_v1_20261021"
        preflight = json.loads((preflight_root / "runtime_preflight.json").read_text())
        teacher = json.loads((home / "diagnostics/reference_relation_preflight_v1_20261021/teacher_gate.json").read_text())
        if (preflight["status"] != "passed" or preflight["optimizer_steps"] != 0
                or not preflight["teacher_numerical_gate_passed"] or not teacher["numerical_gate_passed"]
                or preflight["initialization"]["source_epoch"] != 48
                or preflight["max_effective_batch_size"] != 64
                or not preflight["shared_finetune_backward_passed"]
                or preflight["parameters_each_arm"] != 7461316):
            raise ValueError("required label, initialization or RGB resource preflight did not pass")
        configs = {arm: json.loads((run.repo / f"experiments/configs/geoprr_v2_role_triangle_{arm}.json").read_text())
                   for arm in ("full", "separable")}
        if preflight["configs"] != configs or not preflight["same_initial_parameters"]:
            raise ValueError("runtime/config/initialization differs from the frozen paired control")
        comparable = []
        for arm in ("full", "separable"):
            config = configs[arm]
            if (config["protocol"] != "geoprr_v2_role_triangle_source_only_v1"
                    or config["initialization"] != "full_best48"
                    or config["model"]["association"] != arm
                    or config["training"]["seed"] != 20262020):
                raise ValueError("triangle protocol, initialization, seed or association differs")
            comparable.append({key: value for key, value in config.items() if key not in ("name", "model")}
                              | {"model": {key: value for key, value in config["model"].items() if key != "association"}})
        if comparable[0] != comparable[1]:
            raise ValueError("candidate and control differ beyond factor association")
        protocol_path = output / "protocol.json"
        previous_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if previous_protocol and previous_protocol["configs"] != configs:
            raise ValueError("formal configurations differ from the frozen queue")
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for filename in ("geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py",
                             "train_geoprr_v2_reference_relation.py", "train_geoprr_v2_role_triangle.py",
                             "summarize_geoprr_v2_relative_state.py", "run_geoprr_v2_reference_feedback.py",
                             "run_geoprr_v2_role_triangle.py", "preflight_geoprr_v2_role_triangle.py"):
                shutil.copy2(run.repo / "experiments" / filename, snapshot / filename)
        write_json(protocol_path, {
            "protocol": PROTOCOL, "seed": 20262020, "epochs_per_arm": 65, "configs": configs,
            "code_revision": os.environ.get("GEOPRR_CODE_REVISION", previous_protocol.get("code_revision")),
            "arms": ["full", "separable_if_original_G1_and_best48_protection_pass"],
            "primary_readout": "full_native_probability_mean",
            "checkpoint_selection": "complete1506 source NMAE, earliest strict minimum",
            "initial_full_best48_checkpoint": str(initial), "parameters_per_arm": 7461316,
            "teacher_label_preflight": teacher, "runtime_preflight": preflight,
            "source_images_unique": 18590, "epoch_donors": 22686, "epoch_recipients": 8192,
            "epoch_images": 30878, "epoch_optimizer_batches": 709,
            "original_G1": "unchanged compare_to_ordinary ten RGB checks plus reused G0 teacher label gate",
            "best48_protection": "separate additional gate: full mean, common1495 mean/median/p90, r8ordinary1497 mean no worse; no new >2/5",
            "counterfactual": "same checkpoint product of the new relative J marginals; diagnosis only",
            "reference_summary": "true triangle posterior A marginal entropy and maximum",
            "scope": "source-development only; other seeds, official test and old controls are forbidden",
            "continuation": "full must pass original G1 and separate best48 protection before separable",
            "private_artifacts": "run/data_contract contains original geometry; public export requires explicit allowlist",
        })
        results = {}
        try:
            for arm in ("full", "separable"):
                reader = output / "readers" / f"{arm}_s20262020"
                config = run.repo / f"experiments/configs/geoprr_v2_role_triangle_{arm}.json"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2_role_triangle", "--config", str(config),
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
                if (complete["epochs"] != 65 or complete["seed"] != 20262020
                        or metadata["config"] != json.loads(config.read_text())
                        or metadata["parameters"] != 7461316):
                    raise ValueError("completed reader differs from formal configuration or parameters")
                if (complete["totals"]["train_batches"] != 65 * 709
                        or complete["totals"]["donor_images"] != 65 * 22686
                        or complete["totals"]["recipient_images"] != 65 * 8192):
                    raise ValueError("completed reader exposure differs")
                result = compare_to_ordinary(reader / "best_predictions.csv.gz", reference)
                result["original_G1"] = {"G0_label_gate_passed": True,
                                         "RGB_checks": result["checks"], "gate_passed": result["gate_passed"]}
                result["best48_comparison"] = compare_best48(reader / "best_predictions.csv.gz",
                                                             initial.parent / "best_predictions.csv.gz", reference)
                result["best48_protection"] = best48_protection(result["best48_comparison"])
                result.update(best_epoch=complete["best_epoch"], parameters=metadata["parameters"], costs=complete)
                write_json(output / "validation_evaluation" / arm / "summary.json", result)
                results[arm] = result
                if arm == "full" and not (result["gate_passed"] and result["best48_protection"]["gate_passed"]):
                    run.state.update(status="validation_rejected", deferred_work=["matched_separable_control"],
                                     forbidden_work=["other_seeds", "official_test", "old_local_product_controls"],
                                     next_action="preserve complete negative result and reassess the single mechanism")
                    write_json(output / "validation_decision.json", {"decision": "triangle_validation_rejected", **result})
                    run.save()
                    return
            full, separable = results["full"]["summary"]["candidate"], results["separable"]["summary"]["candidate"]
            checks = {
                "full_nmae_at_least5_percent_better_than_separable": full["nmae_percent_fs"] <= .95 * separable["nmae_percent_fs"],
                "full_tail_not_worse": full["tail10_percent_fs"] <= separable["tail10_percent_fs"],
                "full_above5_not_worse": full["above_5_percent_fs"] <= separable["above_5_percent_fs"],
                **results["full"]["best48_protection"]["checks"],
            }
            decision = {"decision": "promising_source_mechanism" if all(checks.values()) else "triangle_not_supported",
                        "checks": checks, "full": full, "separable": separable,
                        "original_G1": results["full"]["original_G1"],
                        "best48_protection": results["full"]["best48_protection"],
                        "best48_comparison": results["full"]["best48_comparison"],
                        "test_used": False, "seed_expansion": False,
                        "interpretation": "single-seed source-development screen; no significance or real-transfer claim"}
            write_json(output / "validation_decision.json", decision)
            run.state.update(status="complete", validation_decision=decision["decision"])
            run.save()
        except Exception as exc:
            run.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            run.save()
            raise


if __name__ == "__main__":
    main()
