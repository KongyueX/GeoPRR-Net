"""Run the frozen-CNN, source-only direct scalar readout probe serially."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_direct_readout import CONFIG, PARENT_RELATIVE, PROTOCOL, SOURCE_RELATIVE
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def load(path):
    return json.loads(path.read_text())


def run(home):
    repo = Path(__file__).resolve().parents[1]
    output = home / "runs/geoprr_generalization/iteration_v15_direct_readout_20261023/source_only_probe"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for relative in ("iteration_v13_cnn_weight_freeze_20261023/matched_training",
                         "iteration_v14_anchor_half_20261023/frozen_evaluation"):
            previous = load(home / "runs/geoprr_generalization" / relative / "queue_state.json")
            if previous["status"] != "complete" or process_running(previous["pipeline_pid"]):
                raise RuntimeError("previous CNN-freeze and interpolation queues must be complete and inactive")
        source = home / SOURCE_RELATIVE
        if load(source / "queue_state.json")["status"] != "complete":
            raise ValueError("physical endpoint source is incomplete")
        if compute_processes():
            raise RuntimeError("GPU has an active compute process; do not duplicate a live job")
        witness = load(home / "diagnostics/geoprr_generalization_v15_direct_readout_preflight_20261023/result.json")
        if (witness["status"] != "passed" or witness["config"] != CONFIG
                or witness["counts"]["optimizer_updates"] != 0 or witness["validation_input_exact_identity"] is not True):
            raise ValueError("the exact new readout recipe needs its actual-input zero-update witness")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                    "parent_checkpoint": str(home / PARENT_RELATIVE), "source_manifest": str(source / "manifests/train.jsonl"),
                    "source_validation_manifest": str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
                    "expected_source_cache": {"images": 20096, "CNN_calls": 676},
                    "expected_complete_CNN": {"images": 21642, "calls": 774, "backward": 0, "optimizer_updates": 0},
                    "expected_head_fit": {"feature_epochs": 20, "feature_presentations": 371800,
                                          "CPU_backward_calls": 1460, "CPU_optimizer_updates": 1460},
                    "comparison_scope": "independent frozen-feature capacity probe; no claim of matched donor/recipient exposure or end-to-end improvement",
                    "source_selection": "minimum direct1506 NMAE, earliest tie, complete20 head epochs; same selected head for all readouts",
                    "external_scope": "previously viewed Industrial1395/52 and RF100151/35 development, no fitting or adaptation",
                    "inference": "one native ROI RGB CNN pass shared by geometry, direct sigmoid and their fixed half; no target GT geometry input",
                    "witness": {"status": witness["status"], "counts": witness["counts"],
                                "validation_input_exact_identity": True}}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored probe protocol or code revision differs; preserve the partial experiment")
        write_json(path, protocol)
        write_json(output / "config.json", CONFIG)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("geoprr_v2_direct_readout.py", "run_geoprr_v2_direct_readout.py", "test_geoprr_v2_direct_readout.py",
                         "geoprr_v2_reference_relation.py", "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py",
                         "evaluate_geoprr_v2_generalization.py", "evaluate_geoprr_v2_role_external.py",
                         "train_geoprr_v2_reference_relation.py"):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            cache, fitting = output / "source_cache", output / "head"
            stages = [("cache_source", ["cache_source", "--output", str(cache)], cache / "completed.json", True),
                      ("fit_head", ["fit_head", "--cache", str(cache), "--output", str(fitting)], fitting / "completed.json", False)]
            for dataset in ("industrial", "rf100"):
                directory = output / "evaluations" / dataset
                stages.append((f"eval_{dataset}", ["evaluate", "--head-checkpoint", str(fitting / "best.pt"),
                               "--dataset-name", dataset, "--output", str(directory)], directory / "summary.json", True))
            completed = {}
            for name, arguments, expected, gpu in stages:
                if gpu and compute_processes():
                    raise RuntimeError("GPU became busy before the next serialized stage")
                queue.state["current_step"] = name
                queue.save()
                command = [sys.executable, "-m", "experiments.geoprr_v2_direct_readout", *arguments, "--home", str(home)]
                queue.step(name, command, expected, environment_overrides=None if gpu else {"CUDA_VISIBLE_DEVICES": ""})
                completed[name] = load(expected)
            counts = [completed[name]["counts"] for name in ("cache_source", "eval_industrial", "eval_rf100")]
            total_images = sum(item["cnn_image_forwards"] for item in counts)
            total_calls = sum(item["cnn_forward_calls"] for item in counts)
            head_counts = completed["fit_head"]["counts"]
            if ((total_images, total_calls) != (21642, 774)
                    or (head_counts["head_optimizer_updates"], head_counts["source_feature_presentations"]) != (1460, 371800)):
                raise RuntimeError("formal source cache, CPU head fitting or frozen evaluation counts differ")
            write_json(output / "screen_summary.json", {
                "protocol": PROTOCOL, "status": "complete", "config": CONFIG, "stages": completed,
                "counts": {"cnn_image_forwards": total_images, "cnn_forward_calls": total_calls,
                           "cnn_backward_calls": 0, "cnn_optimizer_updates": 0,
                           "CPU_head_backward_calls": head_counts["head_backward_calls"],
                           "CPU_head_optimizer_updates": head_counts["head_optimizer_updates"],
                           "source_feature_presentations": head_counts["source_feature_presentations"]},
                "scope": "complete source-only frozen-feature readout probe on viewed external development; no automatic model promotion"})
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
