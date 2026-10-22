"""Serially render and verify the frozen 128-group endpoint source replacement."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.geoprr_v2_endpoint_source import (
    DATA_PROTOCOL, RENDER_SETTINGS, REPLACEMENT, endpoint_generation_plan, read_json, write_json,
)
from experiments.run_geoprr_v2_optimization import OptimizationRun
from experiments.run_geoprr_v2_render_probe import compute_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    revision = os.environ.get("GEOPRR_ENDPOINT_REVISION")
    if not revision:
        raise ValueError("GEOPRR_ENDPOINT_REVISION must record the frozen implementation revision")
    home = args.home.resolve()
    output = home / "rendering/generalization_endpoints_v11_20261023"
    old = home / "rendering/compositional_source_v1"
    if read_json(old / "queue_state.json")["status"] != "complete":
        raise RuntimeError("the original source generation is not complete")
    prior = home / "runs/geoprr_generalization/iteration_v10_roi_padding_20261023/matched_training/queue_state.json"
    if read_json(prior)["status"] != "complete":
        raise RuntimeError("v10 must complete before endpoint source generation")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = endpoint_generation_plan(read_json(old / "generation_plan.json"))
        protocol = {"protocol": DATA_PROTOCOL, "replacement": REPLACEMENT, "render_settings": RENDER_SETTINGS,
                    "implementation_revision": revision, "original_source_root": str(old),
                    "groups": 128, "new_rgb_images": 512, "population_operation": "replace selected original groups in a new manifest",
                    "source_subset": "first 128 groups in the immutable original 1024-group generation plan",
                    "upstream_revision": "337b868f0f5495b31acc383433cac44040247bf2", "blender": "4.5.14",
                    "failure_policy": "stop and retain partial output; no seed/state resampling or weakened geometry thresholds",
                    "training_launched": False}
        for name, value in (("generation_plan.json", plan), ("protocol.json", protocol)):
            path = output / name
            if path.exists() and read_json(path) != value:
                raise ValueError(f"stored endpoint {name} differs")
            if not path.exists():
                write_json(path, value)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=DATA_PROTOCOL, implementation_revision=revision, status="running", current_step=None)
        run.save()
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in ("render_geoprr_v2_state_group.py", "geoprr_v2_endpoint_source.py", "run_geoprr_v2_endpoint_source.py"):
            destination = snapshot / name
            source = run.repo / "experiments" / name
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("endpoint source snapshot differs from the frozen implementation")
            if not destination.exists():
                shutil.copyfile(source, destination)
        renderer = home / "rendering/tools/blender-4.5.14-linux-x64/blender"
        vendor = home / "rendering/vendor/SyncG-337b868f0f5495b31acc383433cac44040247bf2"
        environment = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1",
                       "PYTHONPATH": str(home / "rendering/python_packages"),
                       "LD_LIBRARY_PATH": str(home / "rendering/tools/sysroot/usr/lib/x86_64-linux-gnu")
                           + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
        common = [sys.executable, "-m", "experiments.geoprr_v2_endpoint_source", "--root", str(output),
                  "--old-source", str(old), "--original-train", str(run.manifests / "train.jsonl"),
                  "--validation", str(run.manifests / "validation.jsonl"), "--data-root", str(home / "datasets")]
        try:
            for group in plan:
                directory = output / "groups" / group["layout_id"]
                name = f"render_g{group['index']:04d}"
                previous = run.state["steps"].get(name, {})
                if previous.get("status") not in ("running", "complete"):
                    while compute_processes():
                        run.state.update(status="waiting_for_gpu", current_step=name)
                        run.save()
                        time.sleep(30)
                run.state.update(status="running", current_step=name)
                run.save()
                command = [str(renderer), "--python-use-system-env", "--background", "--factory-startup",
                           "--python", str(run.repo / "experiments/render_geoprr_v2_state_group.py"), "--",
                           "--source-root", str(vendor), "--output", str(directory), "--device", "CUDA",
                           "--role", "source_training", "--state-policy", "endpoint_v11",
                           "--layout-id", group["layout_id"], "--seed", str(group["render_seed"]),
                           "--states", *map(str, group["states"]), "--width", "1920", "--height", "1080",
                           "--samples", "200", "--threads", "4", "--image-format", "JPEG"]
                run.step(name, command, directory / "group.json", environment_overrides=environment)
                run.state["current_step"] = f"prepare_g{group['index']:04d}"
                run.save()
                run.step(run.state["current_step"], common + ["--group-index", str(group["index"])],
                         directory / "endpoint_prepared.json", environment_overrides={"CUDA_VISIBLE_DEVICES": ""})
                run.state["completed_groups"] = group["index"] + 1
                run.save()
            run.state["current_step"] = "assemble"
            run.save()
            run.step("assemble", common, output / "prepared.json", environment_overrides={"CUDA_VISIBLE_DEVICES": ""})
            run.state.update(status="complete", current_step=None, completed_groups=128)
            run.save()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
