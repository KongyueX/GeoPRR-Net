"""Render and prepare the fixed single-view camera-roll source population."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.prepare_geoprr_v2_roll_source import generation_plan
from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


SOURCE_RELATIVE = "rendering/generalization_roll_v22_20261024"
RENDER_SETTINGS = {"width": 1920, "height": 1080, "samples": 200, "threads": 4,
                   "device": "CUDA", "image_format": "JPEG", "image_quality": 95}


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(home):
    revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
    output = home / SOURCE_RELATIVE
    old = home / "rendering/compositional_source_v1"
    current = home / "rendering/generalization_endpoints_v11_20261023"
    original_validation = home / "runs/geoprr_v2/manifests/validation.jsonl"
    if load(current / "queue_state.json")["status"] != "complete":
        raise ValueError("the retained physical endpoint source must be complete")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = generation_plan(load(old / "generation_plan.json"),
                               rows(current / "manifests/train.jsonl"), rows(original_validation))
        if len(plan) != 1152 or sum(item["split"] == "train" for item in plan) != 1024:
            raise ValueError("roll source must contain1024 training and128 held-out layouts")
        protocol = {"protocol": "geoprr_v2_camera_roll_source_v22",
                    "implementation_revision": revision, "render_settings": RENDER_SETTINGS,
                    "training_layouts": 1024, "training_new_RGB": 4096,
                    "heldout_layouts": 128, "heldout_new_RGB": 512,
                    "retained_training_RGB": 18590,
                    "source_intervention": "independent continuous camera-local optical-axis roll; one RGB per inference",
                    "heldout_scope": "disjoint new layout IDs and render seeds; shared source-training HDR background",
                    "population_operation": "append new physical RGB; retain all18590 source and original1506 validation unchanged",
                    "failure_policy": "stop and preserve partial output; no state/seed resampling or relaxed geometry thresholds",
                    "target_data_used": False}
        for name, value in (("generation_plan.json", plan), ("protocol.json", protocol)):
            path = output / name
            if path.exists() and load(path) != value:
                raise ValueError(f"stored roll source {name} differs from the frozen experiment")
            if not path.exists():
                write_json(path, value)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=protocol["protocol"], implementation_revision=revision,
                           planned_groups=1152, planned_new_RGB=4608)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in ("run_geoprr_v2_roll_source_data.py", "render_geoprr_v2_roll_group.py",
                     "prepare_geoprr_v2_roll_source.py", "render_geoprr_v2_state_group.py",
                     "prepare_geoprr_v2_compositional_data.py", "geoprr_v2_endpoint_source.py",
                     "geoprr_v2_data.py", "prepare_geoprr_v2_data.py"):
            source, destination = queue.repo / "experiments" / name, snapshot / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError(f"frozen source snapshot changed: {name}")
            if not destination.exists():
                shutil.copy2(source, destination)
        renderer = home / "rendering/tools/blender-4.5.14-linux-x64/blender"
        vendor = home / "rendering/vendor/SyncG-337b868f0f5495b31acc383433cac44040247bf2"
        render_environment = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1",
                              "PYTHONPATH": str(home / "rendering/python_packages"),
                              "LD_LIBRARY_PATH": str(home / "rendering/tools/sysroot/usr/lib/x86_64-linux-gnu")
                                  + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
        prepare = [sys.executable, "-m", "experiments.prepare_geoprr_v2_roll_source",
                   "--root", str(output), "--original-train", str(current / "manifests/train.jsonl"),
                   "--validation", str(original_validation)]
        try:
            for group in plan:
                directory = output / "groups" / group["layout_id"]
                render_step = f"render_g{group['index']:04d}"
                if queue.state["steps"].get(render_step, {}).get("status") not in ("running", "complete"):
                    while compute_processes():
                        queue.state.update(status="waiting_for_gpu", current_step=render_step)
                        queue.save()
                        time.sleep(30)
                queue.state.update(status="running", current_step=render_step)
                queue.save()
                command = [str(renderer), "--python-use-system-env", "--background", "--factory-startup",
                           "--python", str(queue.repo / "experiments/render_geoprr_v2_roll_group.py"), "--",
                           "--source-root", str(vendor), "--output", str(directory), "--device", "CUDA",
                           "--role", "source_training", "--state-policy", group["state_policy"],
                           "--split", group["split"], "--roll-degrees", str(group["roll_degrees"]),
                           "--layout-id", group["layout_id"], "--seed", str(group["render_seed"]),
                           "--states", *map(str, group["states"]), "--width", "1920", "--height", "1080",
                           "--samples", "200", "--threads", "4", "--image-format", "JPEG"]
                queue.step(render_step, command, directory / "roll_group.json", render_environment)
                queue.state["current_step"] = f"prepare_g{group['index']:04d}"
                queue.save()
                queue.step(queue.state["current_step"], prepare + ["--group-index", str(group["index"])],
                           directory / "roll_prepared.json", {"CUDA_VISIBLE_DEVICES": ""})
                queue.state.update(completed_groups=group["index"] + 1,
                                   completed_new_RGB=4 * (group["index"] + 1))
                queue.save()
            queue.state.update(current_step="assemble")
            queue.save()
            queue.step("assemble", prepare, output / "prepared.json", {"CUDA_VISIBLE_DEVICES": ""})
            queue.state.update(status="complete", current_step=None, completed_groups=1152,
                               completed_new_RGB=4608)
            queue.save()
        except Exception as error:
            queue.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    run(parser.parse_args().home.resolve())


if __name__ == "__main__":
    main()
