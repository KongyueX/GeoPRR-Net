"""Run the authorized reference update and its gated matched global control."""
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
from experiments.run_geoprr_v2_role_triangle import compare_best48, best48_protection
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary


PROTOCOL = "geoprr_v2_reference_pair_update_matched_source_v1"
ARMS = {"preserve": "pair_preserving", "global": "global"}


def compare_best64(candidate, anchor, ordinary_reference):
    comparison = compare_best48(candidate, anchor, ordinary_reference)
    return {key.replace("best48", "best64"): value for key, value in comparison.items()}


def best64_protection(comparison):
    as_best48 = {key.replace("best64", "best48"): value for key, value in comparison.items()}
    protection = best48_protection(as_best48)
    return {"checks": {key.replace("best48", "best64"): value for key, value in protection["checks"].items()},
            "gate_passed": protection["gate_passed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    home = parser.parse_args().home
    output = home / "runs/geoprr_v2_tail/iteration_v8_reference_pair_update"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=PROTOCOL)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        source = home / "rendering/compositional_source_v1/manifests"
        ordinary = home / "diagnostics/gt_atlas_full1506_v1/predictions.csv"
        initial = home / "runs/geoprr_v2_tail/iteration_v7_role_triangle/readers/full_s20262020/best.pt"
        best48 = home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best_predictions.csv.gz"
        preflight_path = home / "diagnostics/reference_pair_update_preflight_20261022_v1/runtime_preflight.json"
        preflight = json.loads(preflight_path.read_text())
        teacher = json.loads((home / "diagnostics/reference_relation_preflight_v1_20261021/teacher_gate.json").read_text())
        if (preflight["status"] != "passed" or preflight["optimizer_updates"] != 0
                or preflight["CNN_image_forwards"] != 132 or preflight["backward_completed"] != 4
                or preflight["max_effective_batch"] != 64 or preflight["parameters_each_arm"] != 7465988
                or preflight["states_each_arm"] != 500 or not preflight["same_initial_tensors"]
                or preflight["initial_epoch"] != 64 or preflight["model_seed"] != 20262020
                or preflight["initialization"]["loaded_full_tensors"] != 497
                or not teacher["numerical_gate_passed"]):
            raise ValueError("required initialization, label or bounded RGB resource preflight did not pass")
        source_config = json.loads((initial.parent / "config.json").read_text())
        configs = {arm: json.loads((run.repo / f"experiments/configs/geoprr_v2_reference_pair_update_{arm}.json").read_text())
                   for arm in ARMS}
        comparable = []
        for arm, mode in ARMS.items():
            config = configs[arm]
            expected_model = {key: value for key, value in source_config["model"].items() if key != "association"}
            expected_model["reference_update"] = mode
            if (config["protocol"] != "geoprr_v2_reference_pair_update_source_only_v1"
                    or config["initialization"] != "full_best64" or config["model"] != expected_model
                    or config["training"]["seed"] != 20262020
                    or any(config[key] != source_config[key] for key in
                           ("training", "data", "exposure", "state_grid", "objective", "inference_input"))):
                raise ValueError("formal inputs, exposure or model differ from the passed best64 resource arrangement")
            frozen = preflight["stages"][f"{mode}_frozen_v0_RGB"]
            stress = preflight["stages"][f"{mode}_shared_nonzero_probe_maxbatch"]
            if (not frozen["gradients_finite"] or not stress["gradients_finite"]
                    or frozen["best64_reading_replay_max"] > 1e-5 or frozen["original_objective_replay_max"] > 1e-5
                    or frozen["evidence_abs_max"] != 0 or stress["evidence_abs_max"] == 0
                    or stress["effective_batch"] != 64):
                raise ValueError("matched zero-initialization or nonzero shared-backward preflight did not pass")
            comparable.append({key: value for key, value in config.items() if key not in ("name", "model")}
                              | {"model": {key: value for key, value in config["model"].items() if key != "reference_update"}})
        if comparable[0] != comparable[1]:
            raise ValueError("candidate and control differ beyond reference update normalization")
        protocol_path = output / "protocol.json"
        previous_protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if previous_protocol and previous_protocol["configs"] != configs:
            raise ValueError("formal configurations differ from the frozen queue")
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for filename in ("geoprr_v2_reference_pair_reader.py", "geoprr_v2_reference_pair_update.py",
                             "train_geoprr_v2_reference_pair_update.py", "run_geoprr_v2_reference_pair_update.py",
                             "preflight_geoprr_v2_reference_pair_update.py", "geoprr_v2_role_triangle.py",
                             "geoprr_v2_reference_relation.py", "train_geoprr_v2_reference_relation.py",
                             "train_geoprr_v2.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
                             "run_geoprr_v2_optimization.py", "run_geoprr_v2_render_probe.py",
                             "run_geoprr_v2_role_triangle.py", "run_geoprr_v2_reference_feedback.py",
                             "summarize_geoprr_v2_relative_state.py"):
                shutil.copy2(run.repo / "experiments" / filename, snapshot / filename)
            for arm in ARMS:
                filename = f"geoprr_v2_reference_pair_update_{arm}.json"
                shutil.copy2(run.repo / "experiments/configs" / filename, snapshot / filename)
        write_json(protocol_path, {
            "protocol": PROTOCOL, "seed": 20262020, "epochs_per_arm": 65, "configs": configs,
            "code_revision": os.environ.get("GEOPRR_CODE_REVISION", previous_protocol.get("code_revision")),
            "arms": ["preserve", "global_if_original_G1_and_best48_and_best64_protections_pass"],
            "human_authorization": "explicit latest instruction in this thread: candidate65; matched global65 only after all three gate groups; no new loss or seed",
            "primary_readout": "full_native_probability_mean",
            "checkpoint_selection": "complete1506 source NMAE, earliest strict minimum",
            "initial_full_best64_checkpoint": str(initial), "best48_reference": str(best48),
            "parameters_per_arm": 7465988, "shared_parameters": 7431248, "head_parameters": 34740,
            "teacher_label_preflight": teacher, "runtime_preflight": preflight,
            "source_images_unique": 18590, "epoch_donors": 22686, "epoch_recipients": 8192,
            "epoch_images": 30878, "epoch_optimizer_batches": 709,
            "original_G1": "unchanged compare_to_ordinary ten RGB checks plus reused G0 teacher label gate",
            "best48_protection": "separate six checks: mean, common1495 mean/median/p90 and r8ordinary1497 mean no worse; no new >2/5",
            "best64_protection": "separate same six checks against retained full best64",
            "matched_control": "same ordered evidence, parameters, initialization, labels, executed batches and exposure; pair-preserving versus global update",
            "optimizer_accounting": "same planned schedule; actual effective updates and AMP skips recorded independently",
            "counterfactual": "same checkpoint product of updated relative-J marginals; not the trained global control",
            "reference_summary": "true updated posterior A marginal entropy and maximum",
            "continuation": "preserve must pass original G1 and both separate anchor protections before global65",
            "G2": "preserve mean at least5% lower than global; tail151 and above5 no worse; both anchor protections retained",
            "anchor_comparison_limit": "fresh optimizer and continued exposure differ from best64; anchor improvements alone cannot identify an algorithm effect",
            "scope": "one seed, repeatedly viewed source development; no official test, new geometry teacher, other seed or old control",
            "private_artifacts": "run/data_contract contains original geometry; only explicit allowlist normalization may be published",
        })
        results = {}
        try:
            for arm in ARMS:
                reader = output / "readers" / f"{arm}_s20262020"
                config = run.repo / f"experiments/configs/geoprr_v2_reference_pair_update_{arm}.json"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2_reference_pair_update", "--config", str(config),
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
                        or metadata["config"] != json.loads(config.read_text()) or metadata["parameters"] != 7465988
                        or metadata["shared_parameters"] != 7431248 or metadata["head_parameters"] != 34740):
                    raise ValueError("completed reader differs from formal configuration or parameters")
                if (complete["totals"]["train_batches"] != 65 * 709
                        or complete["totals"]["donor_images"] != 65 * 22686
                        or complete["totals"]["recipient_images"] != 65 * 8192
                        or complete["totals"]["validation_images"] != 65 * 1506):
                    raise ValueError("completed reader exposure differs")
                result = compare_to_ordinary(reader / "best_predictions.csv.gz", ordinary)
                result["original_G1"] = {"G0_label_gate_passed": True,
                                         "RGB_checks": result["checks"], "gate_passed": result["gate_passed"]}
                result["best48_comparison"] = compare_best48(reader / "best_predictions.csv.gz", best48, ordinary)
                result["best48_protection"] = best48_protection(result["best48_comparison"])
                result["best64_comparison"] = compare_best64(reader / "best_predictions.csv.gz",
                                                             initial.parent / "best_predictions.csv.gz", ordinary)
                result["best64_protection"] = best64_protection(result["best64_comparison"])
                result.update(best_epoch=complete["best_epoch"], parameters=metadata["parameters"], costs=complete)
                write_json(output / "validation_evaluation" / arm / "summary.json", result)
                results[arm] = result
                if arm == "preserve" and not all((result["gate_passed"], result["best48_protection"]["gate_passed"],
                                                  result["best64_protection"]["gate_passed"])):
                    run.state.update(status="validation_rejected", deferred_work=["matched_global_control"],
                                     forbidden_work=["other_seeds", "official_test", "old_local_product_separable_controls"],
                                     next_action="preserve complete results and costs; reassess the single reference mechanism")
                    write_json(output / "validation_decision.json", {"decision": "reference_pair_update_validation_rejected", **result})
                    run.save()
                    return
            preserve = results["preserve"]["summary"]["candidate"]
            global_control = results["global"]["summary"]["candidate"]
            checks = {
                "preserve_nmae_at_least5_percent_better_than_global": preserve["nmae_percent_fs"] <= .95 * global_control["nmae_percent_fs"],
                "preserve_tail_not_worse": preserve["tail10_percent_fs"] <= global_control["tail10_percent_fs"],
                "preserve_above5_not_worse": preserve["above_5_percent_fs"] <= global_control["above_5_percent_fs"],
                **results["preserve"]["best48_protection"]["checks"],
                **results["preserve"]["best64_protection"]["checks"],
            }
            decision = {"decision": "promising_source_mechanism" if all(checks.values()) else "reference_pair_update_not_supported",
                        "checks": checks, "preserve": preserve, "global": global_control,
                        **{key: results["preserve"][key] for key in ("original_G1", "best48_protection", "best64_protection",
                                                                  "best48_comparison", "best64_comparison")},
                        "test_used": False, "seed_expansion": False,
                        "interpretation": "single-seed source-development comparison; anchor improvement alone is not mechanism evidence"}
            write_json(output / "validation_decision.json", decision)
            run.state.update(status="complete", validation_decision=decision["decision"])
            run.save()
        except Exception as exc:
            run.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            run.save()
            raise


if __name__ == "__main__":
    main()
