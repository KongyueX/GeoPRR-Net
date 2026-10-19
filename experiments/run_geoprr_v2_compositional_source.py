"""Generate the frozen supplemental source groups without launching readers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from experiments.prepare_geoprr_v2_compositional_data import PROTOCOL, generation_plan
from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    output = args.home / "rendering/compositional_source_v1"
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, [20262020])
        run.state.update(protocol=PROTOCOL, status="running")
        run.save()
        try:
            plan = generation_plan()
            plan_path = output / "generation_plan.json"
            if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
                raise ValueError("the stored generation plan differs from this protocol")
            write_json(plan_path, plan)
            protocol = {"protocol": PROTOCOL, "role": "source_training", "groups": 1024,
                        "states_per_group": 4, "render_seed_start": 20262020,
                        "state_rng": "numpy.default_rng(20262020), one continuous state in each quartile",
                        "edges": "randomly permuted bidirectional C4 train; bidirectional diagonals held out",
                        "source_revision": "337b868f0f5495b31acc383433cac44040247bf2",
                        "blender": "4.5.14", "backend": "CUDA", "width": 1920, "height": 1080,
                        "samples": 200, "image_format": "JPEG", "quality": 95, "threads": 4,
                        "model_training_launched": False,
                        "failure_policy": "stop and retain failed group; no seed or state resampling"}
            protocol_path = output / "protocol.json"
            if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
                raise ValueError("the stored source-generation protocol differs")
            write_json(protocol_path, protocol)
            renderer = args.home / "rendering/tools/blender-4.5.14-linux-x64/blender"
            source = args.home / "rendering/vendor/SyncG-337b868f0f5495b31acc383433cac44040247bf2"
            environment = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1",
                           "PYTHONPATH": str(args.home / "rendering/python_packages"),
                           "LD_LIBRARY_PATH": str(args.home / "rendering/tools/sysroot/usr/lib/x86_64-linux-gnu")
                           + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
            for group in plan:
                directory = output / "groups" / group["layout_id"]
                name = f"render_g{group['index']:04d}"
                previous = run.state["steps"].get(name, {})
                if previous.get("status") not in ("running", "complete"):
                    while compute_processes():
                        run.state["status"] = "waiting_for_gpu"
                        run.save()
                        time.sleep(30)
                run.state["status"] = "running"
                run.save()
                command = [str(renderer), "--python-use-system-env", "--background", "--factory-startup",
                           "--python", str(run.repo / "experiments/render_geoprr_v2_state_group.py"), "--",
                           "--source-root", str(source), "--output", str(directory), "--device", "CUDA",
                           "--role", "source_training", "--layout-id", group["layout_id"],
                           "--seed", str(group["render_seed"]), "--states", *map(str, group["states"]),
                           "--image-format", "JPEG", "--width", "1920", "--height", "1080",
                           "--samples", "200", "--threads", "4"]
                run.step(name, command, directory / "group.json",
                         environment_overrides=environment)
                run.step(f"prepare_g{group['index']:04d}", [sys.executable, "-m",
                         "experiments.prepare_geoprr_v2_compositional_data", "--root", str(output),
                         "--original-train", str(run.manifests / "train.jsonl"),
                         "--validation", str(run.manifests / "validation.jsonl"),
                         "--group-index", str(group["index"])], directory / "prepared.json")
                run.state["completed_groups"] = group["index"] + 1
                run.save()
            run.step("assemble", [sys.executable, "-m", "experiments.prepare_geoprr_v2_compositional_data",
                     "--root", str(output), "--original-train", str(run.manifests / "train.jsonl"),
                     "--validation", str(run.manifests / "validation.jsonl")], output / "prepared.json")
            run.state["status"] = "complete"
            run.save()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
