"""Run the frozen semantic parent with a new source-only spatial residual."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_semantic_spatial import CONFIG, PARENT_RELATIVE, PROTOCOL
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def load(path):
    return json.loads(path.read_text())


def run(home):
    repo = Path(__file__).resolve().parents[1]
    output = home / "runs/geoprr_generalization/iteration_v17_semantic_spatial_20261023/source_only_training"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_queue = load(home / PARENT_RELATIVE / "queue_state.json")
        if parent_queue["status"] != "complete" or process_running(parent_queue["pipeline_pid"]):
            raise RuntimeError("the full semantic parent queue must be complete and its controller inactive")
        if compute_processes():
            raise RuntimeError("GPU has an active compute process; do not duplicate a live job")
        witness = load(home / "diagnostics/geoprr_generalization_v17_semantic_spatial_preflight_20261023/result.json")
        if (witness["status"] != "passed" or witness["config"] != CONFIG or witness["counts"]["optimizer_updates"] != 0
                or witness["validation_input_matches_declared256_pipeline"] is not True):
            raise ValueError("the exact spatial residual and unchanged declared256 RGB inputs require a zero-update witness")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                    "parent_checkpoint": str(home / PARENT_RELATIVE / "reader/best.pt"),
                    "selection": "source minimum over saved exact parent incumbent0 and all8 complete new source epochs; earliest tie",
                    "inference": "one frozen full semantic encoder pass; native parent logit plus learned generic spatial residual",
                    "parent_parameters_and_buffers": "all464 state tensors frozen including77 BN counters; compare directly each epoch",
                    "trainable_branch_parameters": 129057, "source_only_training": True,
                    "expected_training": {"images": 148720, "CNN_calls": 2328, "branch_backward_attempts": 2328, "CNN_backward_calls": 0},
                    "expected_validation": {"images": 12048, "CNN_calls": 192},
                    "expected_incumbent0_extra_source_CNN_calls": 0,
                    "expected_complete_CNN_if_parent_selected": {"images": 160768, "calls": 2520},
                    "expected_complete_CNN_if_new_branch_selected": {"images": 162314, "calls": 2545},
                    "external_scope": "previously viewed Industrial1395/52 and RF100151/35 development; no fitting or adaptation; parent0 reuses exact saved external artifacts",
                    "comparison_scope": "new source-only spatial branch on a frozen scalar parent; no GT role or geometry input",
                    "witness": {"status": witness["status"], "counts": witness["counts"],
                                "validation_input_matches_declared256_pipeline": True}}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored spatial residual protocol or execution revision differs; retain the partial run")
        write_json(path, protocol)
        write_json(output / "config.json", CONFIG)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("geoprr_v2_semantic_spatial.py", "run_geoprr_v2_semantic_spatial.py", "test_geoprr_v2_semantic_spatial.py",
                         "geoprr_v2_semantic_direct.py", "geoprr_v2_data.py", "evaluate_geoprr_v2_role_external.py",
                         "train_geoprr_v2.py", "train_geoprr_v2_reference_relation.py", "geoprr_v2_endpoint_source.py"):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            reader = output / "reader"
            queue.state["current_step"] = "train_source"
            queue.save()
            queue.step("train_source", [sys.executable, "-m", "experiments.geoprr_v2_semantic_spatial", "train",
                       "--home", str(home), "--output", str(reader)], reader / "completed.json")
            training = load(reader / "completed.json")
            completed = {"train_source": training}
            for dataset in ("industrial", "rf100"):
                if compute_processes():
                    raise RuntimeError("GPU became busy before the next serialized spatial external stage")
                name = f"eval_{dataset}"
                destination = output / "evaluations" / dataset
                queue.state["current_step"] = name
                queue.save()
                command = [sys.executable, "-m", "experiments.geoprr_v2_semantic_spatial", "evaluate", "--home", str(home),
                           "--checkpoint", str(reader / "best.pt"), "--dataset-name", dataset, "--output", str(destination)]
                queue.step(name, command, destination / "summary.json",
                           environment_overrides={"CUDA_VISIBLE_DEVICES": ""} if training["best_epoch"] == 0 else None)
                completed[name] = load(destination / "summary.json")
            totals = training["totals"]
            external = [completed[f"eval_{dataset}"]["forward_counts"] for dataset in ("industrial", "rf100")]
            images = totals["train_cnn_image_forwards"] + totals["validation_images"] + sum(value["cnn_image_forwards"] for value in external)
            calls = totals["train_cnn_forward_calls"] + totals["validation_cnn_forward_calls"] + sum(value["cnn_forward_calls"] for value in external)
            expected = (160768, 2520) if training["best_epoch"] == 0 else (162314, 2545)
            if ((images, calls) != expected or totals["branch_backward_calls"] != 2328 or totals["cnn_backward_calls"] != 0
                    or totals["optimizer_updates"] + totals["amp_skipped_updates"] != 2328):
                raise RuntimeError("completed spatial source training, frozen parent or external counts differ")
            write_json(output / "screen_summary.json", {"protocol": PROTOCOL, "config": CONFIG, "status": "complete",
                       "stages": completed, "counts": {"cnn_image_forwards": images, "cnn_forward_calls": calls,
                           "source_branch_backward_calls": totals["branch_backward_calls"], "source_cnn_backward_calls": 0,
                           "source_optimizer_updates": totals["optimizer_updates"], "source_amp_skipped_updates": totals["amp_skipped_updates"],
                           "target_backward_calls": 0, "target_optimizer_updates": 0},
                       "scope": "complete source-only structural spatial residual on viewed external development; no automatic promotion"})
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
