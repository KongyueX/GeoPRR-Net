"""Run source-only semantic-conditioned local relation training and frozen transfer."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_semantic_relation import CONFIG, PROTOCOL, SOURCE_RELATIVE
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def load(path):
    return json.loads(path.read_text())


def run(home):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required; use only existing cached ImageNet weights")
    repo = Path(__file__).resolve().parents[1]
    output = home / "runs/geoprr_generalization/iteration_v20_semantic_relation_20261023/source_only_training"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = load(home / "runs/geoprr_generalization/iteration_v19_c4_orbit_20261023/source_only_training/queue_state.json")
        if previous["status"] != "complete" or process_running(previous["pipeline_pid"]):
            raise RuntimeError("the previous C4 orbit experiment must be complete and its controller inactive")
        if load(home / SOURCE_RELATIVE / "queue_state.json")["status"] != "complete":
            raise ValueError("existing physical endpoint source is incomplete")
        if compute_processes():
            raise RuntimeError("GPU has an active compute process; do not duplicate a live job")
        witness = load(home / "diagnostics/geoprr_generalization_v20_semantic_relation_preflight_20261023/result.json")
        analytic = load(home / "diagnostics/geoprr_generalization_v18_analytic_source_contract_20261023/result.json")
        feasibility = load(home / "diagnostics/geoprr_v19_source_role_pushforward_feasibility_20261023/result.json")
        if (witness["status"] != "passed" or witness["config"] != CONFIG or witness["counts"]["optimizer_updates"] != 0
                or witness["validation_input_matches_declared256_pipeline"] is not True
                or witness["source_analytic_contract_verified"] is not True
                or witness["semantic_context_1280_used"] is not True
                or witness["deployed_relation_readout_verified"] is not True
                or witness["GT_isolated_from_model_input"] is not True
                or witness["strict_model_reload"] is not True
                or analytic["status"] != "passed" or analytic["samples"] != 18590
                or analytic["data_config"] != CONFIG["source"]["analytic_data"]
                or analytic["geometry_config"] != CONFIG["source"]["analytic_geometry_config"]
                or analytic["source_manifest_relative"] != f"{SOURCE_RELATIVE}/manifests/train.jsonl"
                or analytic["label_contract"] != CONFIG["source"]["analytic_label_contract"]
                or analytic["GT_Q_constructed"] is not False
                or analytic["old_Q_support_readout_integrity_claimed"] is not False
                or feasibility["status"] != "complete" or feasibility["source_population"] != 18590
                or feasibility["angular_bins"] != 256 or feasibility["target_datasets_used"] is not False
                or feasibility["model_or_weights_loaded"] is not False
                or feasibility["all_source_rows_retained"] is not True
                or feasibility["summary"]["errors_above_original_1_percent_fs"] != 0):
            raise ValueError("the existing analytic labels and ideal-role transport proof plus exact zero-update semantic relation witness are required")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                    "source_train_manifest": str(home / SOURCE_RELATIVE / "manifests/train.jsonl"),
                    "source_validation_manifest": str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
                    "initialization": "fresh complete cached ImageNet encoder, new FPN and semantic-conditioned local relation; no point head or trained task weights",
                    "old_task_checkpoint_or_teacher_used": False, "new_weight_download_authorized": False,
                    "model_inputs": CONFIG["inference_input"], "checkpoint_selection": CONFIG["checkpoint_selection"],
                    "source_loss_labels": "analytic U/M/H/span and virtual A/P/B role supervision; labels never model inputs",
                    "deployed_readout": "predicted local conditional J through unchanged CircularProgressPushforward256",
                    "expected_training": {"images": 1115400, "CNN_calls": 17460, "backward_attempts": 17460,
                                          "relation_forward_calls": 17460, "relation_backward_attempts": 17460, "readout_forward_calls": 17460},
                    "expected_validation": {"images": 90360, "CNN_calls": 1440, "relation_forward_calls": 1440, "readout_forward_calls": 1440},
                    "expected_external": {"images": 1546, "CNN_calls": 25, "relation_forward_calls": 25, "readout_forward_calls": 25, "backward_calls": 0, "optimizer_updates": 0},
                    "expected_complete_CNN": {"images": 1207306, "calls": 18925},
                    "external_scope": "previously viewed Industrial1395/52 and RF100151/35 development; source-selected deployed relation best after complete60source epochs",
                    "training_or_adaptation_on_target": False,
                    "comparison_scope": "same source RGB population/256 preprocessing/60epoch CNN exposure asv16; semantic-conditioned relation replaces point and incurs deployed decoder/readout compute",
                    "witness": {"status": witness["status"], "counts": witness["counts"],
                                "validation_input_matches_declared256_pipeline": True, "source_analytic_contract_verified": True,
                                "semantic_context_1280_used": True,
                                "deployed_relation_readout_verified": True, "GT_isolated_from_model_input": True, "strict_model_reload": True},
                    "analytic_contract_audit": {"status": analytic["status"], "samples": analytic["samples"]},
                    "ideal_role_transport_proof": {"status": feasibility["status"], "source_population": feasibility["source_population"],
                        "maximum_error_percent_fs": feasibility["summary"]["maximum_error_percent_fs"],
                        "errors_above_original_1_percent_fs": 0, "target_datasets_used": False}}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored semantic relation protocol or execution revision differs; retain the partial run")
        write_json(path, protocol)
        write_json(output / "config.json", CONFIG)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("geoprr_v2_semantic_relation.py", "run_geoprr_v2_semantic_relation.py", "test_geoprr_v2_semantic_relation.py",
                         "geoprr_v2_role_field_data.py", "test_geoprr_v2_role_field_data.py", "geoprr_v2_semantic_direct.py", "geoprr_v2_semantic_aux.py",
                         "geoprr_v2_reference_relation.py", "geoprr_v2_model.py", "geoprr_v2_data.py",
                         "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py", "train_geoprr_v2_reference_relation.py",
                         "geoprr_v2_endpoint_source.py"):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            reader = output / "reader"
            stages = [("train_source", ["train", "--output", str(reader)], reader / "completed.json")]
            for dataset in ("industrial", "rf100"):
                destination = output / "evaluations" / dataset
                stages.append((f"eval_{dataset}", ["evaluate", "--checkpoint", str(reader / "best.pt"),
                               "--dataset-name", dataset, "--output", str(destination)], destination / "summary.json"))
            completed = {}
            for name, arguments, expected in stages:
                if compute_processes():
                    raise RuntimeError("GPU became busy before the next serialized source/evaluation stage")
                queue.state["current_step"] = name
                queue.save()
                command = [sys.executable, "-m", "experiments.geoprr_v2_semantic_relation", *arguments, "--home", str(home)]
                queue.step(name, command, expected)
                completed[name] = load(expected)
            training = completed["train_source"]["totals"]
            external = [completed[f"eval_{dataset}"]["forward_counts"] for dataset in ("industrial", "rf100")]
            image_count = training["train_cnn_image_forwards"] + training["validation_images"] + sum(value["cnn_image_forwards"] for value in external)
            call_count = training["train_cnn_forward_calls"] + training["validation_cnn_forward_calls"] + sum(value["cnn_forward_calls"] for value in external)
            if (image_count, call_count, training["backward_calls"], training["relation_forward_calls"], training["relation_backward_attempts"], training["readout_forward_calls"],
                    training["validation_relation_forward_calls"], training["validation_readout_forward_calls"],
                    sum(value["relation_forward_calls"] for value in external),
                    sum(value["readout_forward_calls"] for value in external)) != (1207306, 18925, 17460, 17460, 17460, 17460, 1440, 1440, 25, 25):
                raise RuntimeError("completed semantic relation source training or frozen deployed evaluation counts differ")
            write_json(output / "screen_summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
                       "stages": completed, "counts": {"cnn_image_forwards": image_count, "cnn_forward_calls": call_count,
                           "source_backward_calls": training["backward_calls"], "source_optimizer_updates": training["optimizer_updates"],
                           "source_amp_skipped_updates": training["amp_skipped_updates"],
                           "source_relation_forward_calls": training["relation_forward_calls"],
                           "source_relation_backward_attempts": training["relation_backward_attempts"],
                           "source_readout_forward_calls": training["readout_forward_calls"],
                           "validation_relation_forward_calls": training["validation_relation_forward_calls"],
                           "validation_readout_forward_calls": training["validation_readout_forward_calls"],
                           "target_relation_forward_calls": sum(value["relation_forward_calls"] for value in external),
                           "target_readout_forward_calls": sum(value["readout_forward_calls"] for value in external), "target_backward_calls": 0, "target_optimizer_updates": 0},
                       "scope": "complete new source-only semantic-conditioned local relation experiment; deployed geometry, viewed external development, no automatic promotion"})
            queue.state.update(status="complete", current_step=None)
            queue.save()
        except Exception as error:
            queue.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    run(parser.parse_args().home)


if __name__ == "__main__":
    main()
