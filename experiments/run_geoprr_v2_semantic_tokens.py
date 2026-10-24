"""Run the semantic-token witness, source training and frozen external evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_semantic_tokens import CONFIG, OUTPUT_RELATIVE, PROTOCOL, SOURCE_RELATIVE
from experiments.geoprr_v2_roll_source import OUTPUT_RELATIVE as V22_OUTPUT_RELATIVE, PROTOCOL as V22_PROTOCOL
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


PREFLIGHT_RELATIVE = "diagnostics/geoprr_v23_semantic_tokens_preflight_20261024/result.json"


def load(path):
    return json.loads(path.read_text())


def run(home):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required; only cached ImageNet initialization is permitted")
    previous = home / V22_OUTPUT_RELATIVE
    if (load(previous / "queue_state.json")["status"] != "complete"
            or load(previous / "screen_summary.json")["status"] != "complete"
            or load(previous / "reader/completed.json")["protocol"] != V22_PROTOCOL
            or load(previous / "reader/completed.json")["status"] != "complete"):
        raise ValueError("the retained V22 source experiment must complete before the new architecture")
    source = home / SOURCE_RELATIVE
    if (load(source / "prepared.json")["status"] != "complete"
            or load(source / "queue_state.json")["status"] != "complete"):
        raise ValueError("the existing V22 physical-roll source must already be complete")
    output = home / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
            "source_data": str(home / SOURCE_RELATIVE),
            "source_train_manifest": str(home / SOURCE_RELATIVE / "manifests/train_combined.jsonl"),
            "source_original_validation_manifest": str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
            "source_roll_validation_manifest": str(home / SOURCE_RELATIVE / "manifests/validation_roll.jsonl"),
            "initialization": "fresh complete cached ImageNet1280 semantic encoder and four generic multiscale semantic-token readouts",
            "old_task_checkpoint_or_teacher_used": False, "new_weight_download_authorized": False,
            "model_inputs": CONFIG["inference_input"], "checkpoint_selection": CONFIG["checkpoint_selection"],
            "epoch_exposure": {"unique_train_images":22686, "old_images_once":18590, "roll_images_four_times":16384,
                "train_images":34974, "train_CNN_calls":547, "validation_images":2018, "validation_CNN_calls":32},
            "source_stopping": "complete epoch; >=60 and20 epochs without strict source score improvement, maximum160",
            "expected_external": {"images":1546, "CNN_calls":25, "backward_calls":0, "optimizer_updates":0},
            "preflight_budget": {"CNN_images":8, "CNN_calls":3, "backward_calls":1, "optimizer_updates":0},
            "roll_validation_background_independent": False, "original_source_validation_scene_guard_retained": True,
            "external_scope": "viewed Industrial1395/52 and RF100151/35; source-selected best once each after complete source training",
            "training_or_adaptation_on_target": False, "automatic_promotion": False}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored semantic-token protocol or execution revision differs")
        write_json(path, protocol)
        write_json(output / "config.json", CONFIG)
        repo = Path(__file__).resolve().parents[1]
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("geoprr_v2_semantic_tokens.py", "run_geoprr_v2_semantic_tokens.py",
                "test_geoprr_v2_semantic_tokens.py", "geoprr_v2_roll_source.py", "geoprr_v2_semantic_direct.py",
                "geoprr_v2_data.py", "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
                "train_geoprr_v2_reference_relation.py", "geoprr_v2_endpoint_source.py", "run_geoprr_v2_optimization.py"):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        reader = output / "reader"
        witness = home / PREFLIGHT_RELATIVE
        stages = [
            ("preflight", "experiments.geoprr_v2_semantic_tokens", ["preflight", "--output", str(witness)], witness),
            ("train_source", "experiments.geoprr_v2_semantic_tokens", ["train", "--output", str(reader)], reader / "completed.json"),
        ]
        for dataset in ("industrial", "rf100"):
            destination = output / "evaluations" / dataset
            stages.append((f"eval_{dataset}", "experiments.geoprr_v2_semantic_tokens",
                ["evaluate", "--checkpoint", str(reader / "best.pt"), "--dataset-name", dataset,
                 "--output", str(destination)], destination / "summary.json"))
        completed = {}
        try:
            for name, module, arguments, expected in stages:
                command = [sys.executable, "-m", module, *arguments, "--home", str(home)]
                if name == "preflight" and expected.exists():
                    saved = load(expected)
                    if (saved["status"] != "passed" or saved["protocol"] != PROTOCOL or saved["config"] != CONFIG
                            or saved["counts"] != {"CNN_calls":3, "CNN_images":8, "CNN_backward_calls":1,
                                "combined_backward_calls":1, "optimizer_updates":0}):
                        raise ValueError("saved new-source execution witness differs from this run")
                    if name not in queue.state["steps"]:
                        queue.state["steps"][name] = {"status":"complete", "command":command,
                            "expected":str(expected), "exit_code":None, "reused_existing_result":True,
                            "finished_at":datetime.now(timezone.utc).isoformat()}
                        queue.save()
                else:
                    previous = queue.state["steps"].get(name, {})
                    adopting = previous.get("status") == "running" and process_running(previous.get("pid", -1))
                    if compute_processes() and not adopting:
                        raise RuntimeError("GPU has an active unrelated compute process before the next serialized stage")
                    queue.state["current_step"] = name
                    queue.save()
                    queue.step(name, command, expected)
                completed[name] = load(expected)
            trained = completed["train_source"]; totals = trained["totals"]; epochs = trained["epochs"]
            external = [completed[f"eval_{name}"]["forward_counts"] for name in ("industrial", "rf100")]
            images = totals["train_cnn_image_forwards"] + totals["validation_images"] + sum(e["cnn_image_forwards"] for e in external)
            calls = totals["train_cnn_forward_calls"] + totals["validation_cnn_forward_calls"] + sum(e["cnn_forward_calls"] for e in external)
            if (images, calls, totals["backward_calls"]) != (36992*epochs+1546, 579*epochs+25, 547*epochs):
                raise RuntimeError("completed source and frozen external exposure differs from actual epoch budget")
            write_json(output / "screen_summary.json", {"protocol":PROTOCOL, "config":CONFIG, "status":"complete",
                "stages":completed, "epochs":epochs, "counts":{"cnn_image_forwards":images,"cnn_forward_calls":calls,
                    "source_backward_calls":totals["backward_calls"], "source_optimizer_updates":totals["optimizer_updates"],
                    "source_amp_skipped_updates":totals["amp_skipped_updates"], "target_backward_calls":0,"target_optimizer_updates":0},
                "preflight_counts_separate":completed["preflight"]["counts"],
                "scope":"complete source-only multiscale semantic-token reader; no automatic promotion"})
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
