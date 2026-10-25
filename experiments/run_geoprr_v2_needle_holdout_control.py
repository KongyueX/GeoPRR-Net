"""Run the frozen source needle-control preflight and fresh five-epoch control."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_needle_holdout_control import (
    CONFIG, OUTPUT_RELATIVE, PREFLIGHT_RELATIVE, PREPARED_RELATIVE, PROTOCOL,
    validate_prepared, validate_totals,
)
from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


def load(path):
    return json.loads(path.read_text())


def run(home):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required for existing cached ImageNet weights")
    revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
    if not revision.strip():
        raise ValueError("GEOPRR_GENERALIZATION_REVISION must identify the frozen code")
    prepared = validate_prepared(home)
    output = home / OUTPUT_RELATIVE
    witness = home / PREFLIGHT_RELATIVE
    if witness.parent.exists() and any(witness.parent.iterdir()):
        raise FileExistsError("the source-ID preflight already has complete or partial outputs; explicit root handling is required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.name != "queue.lock" for path in output.iterdir()):
            raise FileExistsError("the source control has complete or partial outputs; no rerun or automatic resume is permitted")
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            protocol = {"protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                "source_contract": prepared["source_contract"],
                "prepared_relative": PREPARED_RELATIVE,
                "kind": "bounded_source_diagnostic_not_V32_or_generalization_candidate",
                "stages": ["preflight", "control"], "automatic_resume": False,
                "preflight_budget": {"CNN_images": 8, "CNN_calls": 3,
                                     "backward_calls": 1, "optimizer_updates": 0},
                "formal_budget": {"epochs": 5, "train_images": 27480,
                    "train_CNN_calls": 430, "backward_attempts": 430,
                    "ID_gate_fails_images": 27772, "ID_gate_fails_CNN_calls": 435,
                    "ID_gate_passes_images": 27992, "ID_gate_passes_CNN_calls": 439},
                "external_counts": {"images": 0, "CNN_calls": 0,
                                    "backward_calls": 0, "optimizer_updates": 0}}
            write_json(output / "protocol.json", protocol)
            write_json(output / "config.json", CONFIG)
            repo = Path(__file__).resolve().parents[1]
            snapshot = output / "code_snapshot"
            snapshot.mkdir()
            for name in (
                "geoprr_v2_needle_holdout_control.py", "run_geoprr_v2_needle_holdout_control.py",
                "test_geoprr_v2_needle_holdout_control.py", "geoprr_v2_high_resolution_direct.py",
                "geoprr_v2_roll_source.py", "geoprr_v2_semantic_tokens.py", "geoprr_v2_semantic_direct.py",
                "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
                "geoprr_v2_objectives.py", "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py",
                "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
                "train_geoprr_v2_reference_relation.py", "run_geoprr_v2_optimization.py",
                "run_geoprr_v2_render_probe.py",
            ):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
            completed = {}
            reader = output / "reader"
            for name, stage, destination, expected in (
                ("preflight", "preflight", witness, witness),
                ("source_control", "control", reader, reader / "completed.json"),
            ):
                if compute_processes():
                    raise RuntimeError("GPU has an active compute process before the serialized source stage")
                command = [sys.executable, "-m", "experiments.geoprr_v2_needle_holdout_control", stage,
                           "--home", str(home), "--output", str(destination)]
                queue.state["current_step"] = name
                queue.save()
                queue.step(name, command, expected)
                saved = load(expected)
                if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG
                        or saved["source_contract"] != prepared["source_contract"]):
                    raise ValueError("completed source stage changed the frozen configuration or source contract")
                completed[name] = saved
                if name == "preflight":
                    if (saved["status"] != "passed" or saved["counts"] != {
                        "CNN_calls": 3, "CNN_images": 8, "CNN_backward_calls": 1,
                        "combined_backward_calls": 1, "optimizer_updates": 0}):
                        raise ValueError("the source-ID preflight did not satisfy its zero-update budget")
            control = completed["source_control"]
            passed = control["learning_gate"]["passed"]
            if (control["status"] != "complete" or control["epochs"] != 5
                    or control["completion_reason"] != (
                        "source_ood_observed" if passed else "learning_gate_failed")):
                raise ValueError("source control did not complete its frozen five-epoch gate")
            validate_totals(control["totals"], passed)
            totals = control["totals"]
            counts = {
                "cnn_image_forwards": totals["train_cnn_image_forwards"] + totals["validation_images"],
                "cnn_forward_calls": totals["train_cnn_forward_calls"] + totals["validation_cnn_forward_calls"],
                "point_image_forwards": totals["train_point_image_forwards"] + totals["validation_point_image_forwards"],
                "point_forward_calls": totals["train_point_forward_calls"] + totals["validation_point_forward_calls"],
                "source_backward_calls": totals["backward_calls"],
                "source_optimizer_updates": totals["optimizer_updates"],
                "source_amp_skipped_updates": totals["amp_skipped_updates"],
                "external_images": 0, "external_CNN_calls": 0,
                "external_backward_calls": 0, "external_optimizer_updates": 0,
            }
            write_json(output / "screen_summary.json", {"protocol": PROTOCOL, "config": CONFIG,
                "source_contract": prepared["source_contract"], "status": "complete", "epochs": 5,
                "completion_reason": control["completion_reason"], "learning_gate": control["learning_gate"],
                "counts": counts, "preflight_counts_separate": completed["preflight"]["counts"],
                "scope": "one bounded source needle control; no automatic promotion"})
            queue.state.update(status="complete", current_step=None,
                               completion_reason=control["completion_reason"])
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
