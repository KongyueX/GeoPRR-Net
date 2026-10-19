"""Measure one GPU source-generation group after the current reader finishes."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time

from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json


def compute_processes() -> list[str]:
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    output = args.home / "rendering/gpu_quality_probe_seed20262020_v1"
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, [20262020])
        run.state.update(protocol="geoprr_v2_gpu_render_quality_probe_v1", status="waiting_for_reader")
        run.save()
        try:
            parent = args.home / "runs/geoprr_v2_tail/iteration_v2_decoupled/queue_state.json"
            while True:
                state = json.loads(parent.read_text())
                if state["status"] == "complete":
                    break
                if state["status"] != "running":
                    raise RuntimeError(f"reader queue is {state['status']}; inspect before rendering")
                time.sleep(30)
            runtime = json.loads((args.home / "rendering/gpu_runtime_state.json").read_text())
            if runtime["status"] != "ready_for_future_gpu_measurement":
                raise RuntimeError("the separate GPU rendering runtime is not ready")
            while compute_processes():
                run.state["status"] = "waiting_for_gpu"
                run.save()
                time.sleep(30)
            run.state["status"] = "running"
            run.save()
            write_json(output / "protocol.json", {"protocol": run.state["protocol"], "diagnostic_only": True,
                       "role": "render_quality_and_throughput", "seed": 20262020,
                       "states": [.15, .35, .65, .85], "width": 1920, "height": 1080,
                       "samples": 200, "image_format": "JPEG", "quality": 95, "threads": 4,
                       "blender": "4.5.14", "backend": "CUDA", "training_or_formal_data_generation": False})
            renderer = args.home / "rendering/tools/blender-4.5.14-linux-x64/blender"
            source = args.home / "rendering/vendor/SyncG-337b868f0f5495b31acc383433cac44040247bf2"
            environment = {"CUDA_VISIBLE_DEVICES": "0", "PYTHONNOUSERSITE": "1",
                           "PYTHONPATH": str(args.home / "rendering/python_packages"),
                           "LD_LIBRARY_PATH": str(args.home / "rendering/tools/sysroot/usr/lib/x86_64-linux-gnu")
                           + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
            command = [str(renderer), "--python-use-system-env", "--background", "--factory-startup",
                       "--python", str(run.repo / "experiments/render_geoprr_v2_state_group.py"), "--",
                       "--source-root", str(source), "--output", str(output), "--device", "CUDA",
                       "--image-format", "JPEG", "--width", "1920", "--height", "1080",
                       "--samples", "200", "--threads", "4"]
            memory_samples = []
            stop = threading.Event()

            def sample_memory() -> None:
                while not stop.wait(1):
                    query = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid,used_memory",
                                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    pid = run.state["steps"].get("render", {}).get("pid")
                    for line in query.stdout.splitlines():
                        values = [value.strip() for value in line.split(",")]
                        if len(values) == 2 and values[0] == str(pid) and values[1].isdigit():
                            memory_samples.append(int(values[1]))

            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()
            try:
                run.step("render", command, output / "group.json", environment_overrides=environment)
            finally:
                stop.set()
                sampler.join()
            memory_path = output / "gpu_memory.json"
            if memory_samples:
                write_json(memory_path, {"sampled_gpu_memory_max_mib": max(memory_samples),
                                        "gpu_memory_sample_interval_seconds": 1,
                                        "gpu_memory_samples": len(memory_samples)})
            memory = json.loads(memory_path.read_text()) if memory_path.exists() else {
                "sampled_gpu_memory_max_mib": None, "gpu_memory_sample_interval_seconds": 1, "gpu_memory_samples": 0}
            run.step("geometry", [sys.executable, "-m", "experiments.prepare_geoprr_v2_state_group",
                                  "--group", str(output)], output / "geometry_diagnostic.json")
            group = json.loads((output / "group.json").read_text())
            log = Path(run.state["steps"]["render"]["log"]).read_text()
            elapsed = re.findall(r"^Time:\s+([\d:.]+)\s+\(Saving:", log, re.MULTILINE)
            seconds = [sum(float(part) * 60 ** index for index, part in enumerate(reversed(value.split(":"))))
                       for value in elapsed]
            if len(group["images"]) != 4 or len(seconds) != 4:
                raise RuntimeError("the four-image render timing record is incomplete")
            record = run.state["steps"]["render"]
            total = (datetime.fromisoformat(record["finished_at"]) - datetime.fromisoformat(record["started_at"])).total_seconds()
            write_json(output / "throughput.json", {
                "diagnostic_only": True, "provenance": group["provenance"], "render_seconds": seconds,
                "first_render_seconds": seconds[0], "subsequent_render_seconds_mean": statistics.mean(seconds[1:]),
                "render_process_wall_seconds": total,
                "process_boundary": "Blender startup, scene construction, four renders, identity checks and labels",
                **memory,
                "image_bytes": [Path(image["image_path"]).stat().st_size for image in group["images"]],
            })
            run.state["status"] = "complete"
            run.save()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
