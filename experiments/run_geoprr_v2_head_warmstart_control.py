"""Run one frozen source head-warmstart preflight and LP100 plus FT5 control."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_head_warmstart_control import (
    CONFIG, OUTPUT_RELATIVE, PREFLIGHT_COUNTS, PREFLIGHT_RELATIVE, PREPARED_RELATIVE,
    PROTOCOL, count_summary, validate_prepared, validate_totals,
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
    output, witness = home / OUTPUT_RELATIVE, home / PREFLIGHT_RELATIVE
    if witness.parent.exists() and any(witness.parent.iterdir()):
        raise FileExistsError("the head-warmstart preflight already has complete or partial outputs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(path.name != "queue.lock" for path in output.iterdir()):
            raise FileExistsError("the source control has complete or partial outputs; no rerun or resume is permitted")
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision, automatic_resume=False)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            protocol = {
                "protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                "source_contract": prepared["source_contract"],
                "preparation_protocol": prepared["protocol"], "prepared_relative": PREPARED_RELATIVE,
                "kind": "one bounded source procedure control, not formalV32 or LP-FT paper replication",
                "stages": ["preflight", "control"], "automatic_resume": False,
                "preflight_budget": PREFLIGHT_COUNTS,
                "formal_budget": {
                    "LP_cache_CNN_images": 5496, "LP_cache_CNN_calls": 86, "LP_cache_CNN_backward_calls": 0,
                    "LP_point_calls": 100, "LP_point_feature_examples": 549600,
                    "LP_point_only_backward_calls": 100, "LP_updates_plus_AMP_skips": 100,
                    "FT_epochs": 5, "FT_CNN_images": 27480, "FT_CNN_calls": 430,
                    "FT_point_calls": 430, "FT_point_feature_examples": 27480,
                    "FT_full_CNN_and_point_backward_calls": 430, "FT_updates_plus_AMP_skips": 430,
                    "source_ID_images": 292, "source_ID_CNN_calls": 5,
                    "gate_fail_CNN_images": 33268, "gate_fail_CNN_calls": 521,
                    "gate_fail_point_feature_examples": 577372, "gate_fail_point_calls": 535,
                    "gate_pass_CNN_images": 33488, "gate_pass_CNN_calls": 525,
                    "gate_pass_point_feature_examples": 577592, "gate_pass_point_calls": 539,
                    "total_backward_calls": 530, "CNN_backward_calls": 430,
                    "source_feature_values_exported": False, "preflight_counts_separate": True,
                },
                "external_counts": {"images": 0, "CNN_calls": 0, "point_calls": 0,
                                    "backward_calls": 0, "optimizer_updates": 0},
                "supervisor_process_started_before_torch_import": True,
                "supervisor_seconds_at_protocol": time.perf_counter() - PROCESS_STARTED,
            }
            write_json(output / "protocol.json", protocol)
            write_json(output / "config.json", CONFIG)
            repo = Path(__file__).resolve().parents[1]
            snapshot = output / "code_snapshot"
            snapshot.mkdir()
            for name in (
                "geoprr_v2_head_warmstart_control.py", "run_geoprr_v2_head_warmstart_control.py",
                "test_geoprr_v2_head_warmstart_control.py", "geoprr_v2_needle_holdout_control.py",
                "geoprr_v2_high_resolution_direct.py", "geoprr_v2_roll_source.py",
                "geoprr_v2_semantic_tokens.py", "geoprr_v2_semantic_direct.py",
                "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
                "geoprr_v2_objectives.py", "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py",
                "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
                "train_geoprr_v2_reference_relation.py", "run_geoprr_v2_optimization.py",
                "run_geoprr_v2_render_probe.py",
            ):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
            for relative in (
                "refine-logs/geoprr_research_reset_20261025/head_warmstart_control_protocol.json",
                "refine-logs/geoprr_research_reset_20261025/prior_transfer_source_evidence.json",
                "docs/GEOPRR_V2_HEAD_WARMSTART_SOURCE_CONTROL_CN.md",
            ):
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, target)
            completed, stage_seconds = {}, {}
            reader = output / "reader"
            for name, stage, destination, expected in (
                ("preflight", "preflight", witness, witness),
                ("source_control", "control", reader, reader / "completed.json"),
            ):
                if compute_processes():
                    raise RuntimeError("GPU has an active compute process before the serialized source stage")
                command = [sys.executable, "-m", "experiments.geoprr_v2_head_warmstart_control", stage,
                           "--home", str(home), "--output", str(destination)]
                queue.state["current_step"] = name
                queue.save()
                tick = time.perf_counter()
                queue.step(name, command, expected)
                stage_seconds[name] = time.perf_counter() - tick
                if queue.state["steps"][name]["exit_code"] != 0:
                    raise RuntimeError("the new source stage requires an observed exit code zero")
                saved = load(expected)
                if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG
                        or saved["source_contract"] != prepared["source_contract"]):
                    raise ValueError("completed source stage changed the frozen configuration or source contract")
                completed[name] = saved
                queue.state["supervisor_stage_seconds"] = dict(stage_seconds)
                queue.state["supervisor_process_seconds"] = time.perf_counter() - PROCESS_STARTED
                queue.save()
                if name == "preflight" and (saved["status"] != "passed" or saved["counts"] != PREFLIGHT_COUNTS):
                    raise ValueError("the new source-ID preflight did not satisfy its zero-update budget")
            control = completed["source_control"]
            passed = control["learning_gate"]["passed"]
            if (control["status"] != "complete" or control["epochs"] != 5
                    or control["completion_reason"] != (
                        "source_ood_observed" if passed else "learning_gate_failed")):
                raise ValueError("source control did not complete its frozen LP100 plus FT5 gate")
            validate_totals(control["totals"], passed)
            write_json(output / "screen_summary.json", {
                "protocol": PROTOCOL, "config": CONFIG, "source_contract": prepared["source_contract"],
                "status": "complete", "epochs": 5, "completion_reason": control["completion_reason"],
                "learning_gate": control["learning_gate"], "primary_comparison": control["primary_comparison"],
                "counts": count_summary(control["totals"]),
                "preflight_counts_separate": completed["preflight"]["counts"],
                "supervisor_stage_seconds": stage_seconds,
                "supervisor_process_seconds": time.perf_counter() - PROCESS_STARTED,
                "scope": "one bounded source head-warmstart control; no automatic promotion or full training",
            })
            queue.state.update(status="complete", current_step=None,
                               completion_reason=control["completion_reason"],
                               supervisor_process_seconds=time.perf_counter() - PROCESS_STARTED)
            queue.save()
        except Exception as error:
            queue.state.update(status="failed", error=f"{type(error).__name__}: {error}",
                               supervisor_process_seconds=time.perf_counter() - PROCESS_STARTED)
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    run(parser.parse_args().home)


if __name__ == "__main__":
    main()
