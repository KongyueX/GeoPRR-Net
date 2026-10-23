"""Run complete semantic MobileNetV4 source-only scalar training and frozen transfer."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_semantic_direct import CONFIG, PROTOCOL, SOURCE_RELATIVE
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def load(path):
    return json.loads(path.read_text())


def run(home):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required; this run may use only existing cached ImageNet weights")
    repo = Path(__file__).resolve().parents[1]
    output = home / "runs/geoprr_generalization/iteration_v16_full_semantic_direct_20261023/source_only_training"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = load(home / "runs/geoprr_generalization/iteration_v15_direct_readout_20261023/source_only_probe/queue_state.json")
        if previous["status"] != "complete" or process_running(previous["pipeline_pid"]):
            raise RuntimeError("the previous frozen-feature probe must be complete and its controller inactive")
        if load(home / SOURCE_RELATIVE / "queue_state.json")["status"] != "complete":
            raise ValueError("existing physical endpoint source generation is incomplete")
        if compute_processes():
            raise RuntimeError("GPU has an active compute process; do not duplicate a live job")
        witness = load(home / "diagnostics/geoprr_generalization_v16_semantic_direct_preflight_20261023/result.json")
        if (witness["status"] != "passed" or witness["config"] != CONFIG or witness["counts"]["optimizer_updates"] != 0
                or witness["validation_input_matches_declared256_pipeline"] is not True):
            raise ValueError("the complete semantic encoder and declared256 RGB pipeline need their exact zero-update witness")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                    "source_train_manifest": str(home / SOURCE_RELATIVE / "manifests/train.jsonl"),
                    "source_validation_manifest": str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
                    "initialization": "fresh complete cached ImageNet encoder, num_classes0, including conv_head and norm_head; new scalar point head",
                    "old_task_checkpoint_or_teacher_used": False, "new_weight_download_authorized": False,
                    "model_inputs": CONFIG["inference_input"], "checkpoint_selection": CONFIG["checkpoint_selection"],
                    "expected_training": {"images": 1115400, "CNN_calls": 17460, "backward_attempts": 17460},
                    "expected_validation": {"images": 90360, "CNN_calls": 1440},
                    "expected_external": {"images": 1546, "CNN_calls": 25, "backward_calls": 0, "optimizer_updates": 0},
                    "expected_complete_CNN": {"images": 1207306, "calls": 18925},
                    "external_scope": "previously viewed Industrial1395/52 and RF100151/35 development; source-selected best only after complete60 source epochs",
                    "training_or_adaptation_on_target": False,
                    "comparison_scope": "independent full semantic representation, scalar loss and256 RGB pipeline; not matched old768 geometry training",
                    "witness": {"status": witness["status"], "counts": witness["counts"],
                                "validation_input_matches_declared256_pipeline": True}}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored source semantic protocol or execution revision differs; retain the partial run")
        write_json(path, protocol)
        write_json(output / "config.json", CONFIG)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("geoprr_v2_semantic_direct.py", "run_geoprr_v2_semantic_direct.py", "test_geoprr_v2_semantic_direct.py",
                         "geoprr_v2_data.py", "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
                         "train_geoprr_v2_reference_relation.py", "geoprr_v2_endpoint_source.py"):
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
                command = [sys.executable, "-m", "experiments.geoprr_v2_semantic_direct", *arguments, "--home", str(home)]
                queue.step(name, command, expected)
                completed[name] = load(expected)
            training = completed["train_source"]["totals"]
            external = [completed[f"eval_{dataset}"]["forward_counts"] for dataset in ("industrial", "rf100")]
            image_count = training["train_cnn_image_forwards"] + training["validation_images"] + sum(value["cnn_image_forwards"] for value in external)
            call_count = training["train_cnn_forward_calls"] + training["validation_cnn_forward_calls"] + sum(value["cnn_forward_calls"] for value in external)
            if (image_count, call_count, training["backward_calls"]) != (1207306, 18925, 17460):
                raise RuntimeError("completed full semantic source training or frozen external CNN counts differ")
            write_json(output / "screen_summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
                       "stages": completed, "counts": {"cnn_image_forwards": image_count, "cnn_forward_calls": call_count,
                           "source_backward_calls": training["backward_calls"], "source_optimizer_updates": training["optimizer_updates"],
                           "source_amp_skipped_updates": training["amp_skipped_updates"],
                           "target_backward_calls": 0, "target_optimizer_updates": 0},
                       "scope": "complete new source-only semantic scalar experiment; viewed external development, no automatic promotion"})
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
