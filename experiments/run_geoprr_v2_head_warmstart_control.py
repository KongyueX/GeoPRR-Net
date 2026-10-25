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
from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


SNAPSHOT_METADATA = (
    "refine-logs/geoprr_research_reset_20261025/head_warmstart_control_protocol.json",
    "refine-logs/geoprr_research_reset_20261025/prior_transfer_source_evidence.json",
    "docs/GEOPRR_V2_HEAD_WARMSTART_SOURCE_CONTROL_CN.md",
)


def load(path):
    return json.loads(path.read_text())


def recover_unstarted_bootstrap(output, witness, prepared):
    """Preserve one failed bootstrap whose preflight and model stages never began."""
    artifacts = ("queue_state.json", "protocol.json", "config.json", "pipeline.pid", "code_snapshot")
    if set(path.name for path in output.iterdir()) != {"queue.lock", *artifacts}:
        raise FileExistsError("recovery requires only the original zero-stage bootstrap artifacts")
    queue, protocol = load(output / "queue_state.json"), load(output / "protocol.json")
    if (queue.get("status") != "failed" or queue.get("steps") != {}
            or queue.get("current_step") is not None or queue.get("protocol") != PROTOCOL
            or protocol.get("protocol") != PROTOCOL or protocol.get("config") != CONFIG
            or load(output / "config.json") != CONFIG
            or protocol.get("source_contract") != prepared["source_contract"]
            or queue.get("code_revision") != protocol.get("code_revision")):
        raise ValueError("only the same failed, unstarted protocol/config/source contract can be recovered")
    if ((output / "reader").exists() or (witness.parent.exists() and any(witness.parent.iterdir()))
            or not (output / "code_snapshot").is_dir()):
        raise FileExistsError("reader or preflight artifacts prohibit zero-stage bootstrap recovery")
    pid = int((output / "pipeline.pid").read_text().strip())
    if pid <= 0 or queue.get("pipeline_pid") != pid or process_running(pid):
        raise RuntimeError("the original bootstrap parent must be recorded and actually gone")
    if compute_processes():
        raise RuntimeError("GPU must have no compute process before zero-stage bootstrap recovery")
    failure = output / "bootstrap_failure"
    failure.mkdir(exist_ok=False)
    for name in artifacts:
        (output / name).rename(failure / name)
    return {"code_revision": queue["code_revision"], "pipeline_pid": pid,
            "preserved_relative": "bootstrap_failure", "training_resume": False}


def run(home, *, recover_unstarted=False):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required for existing cached ImageNet weights")
    revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
    if not revision.strip():
        raise ValueError("GEOPRR_GENERALIZATION_REVISION must identify the frozen code")
    repo = Path(__file__).resolve().parents[1]
    for relative in SNAPSHOT_METADATA:
        if not (repo / relative).is_file():
            raise FileNotFoundError(f"required bootstrap snapshot metadata is missing: {relative}")
    prepared = validate_prepared(home)
    output, witness = home / OUTPUT_RELATIVE, home / PREFLIGHT_RELATIVE
    if witness.parent.exists() and any(witness.parent.iterdir()):
        raise FileExistsError("the head-warmstart preflight already has complete or partial outputs")
    if recover_unstarted:
        if not output.is_dir():
            raise FileNotFoundError("explicit zero-stage recovery requires the existing failed bootstrap output")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recovered_from = recover_unstarted_bootstrap(output, witness, prepared) if recover_unstarted else None
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision, automatic_resume=False,
                           recovered_from=recovered_from)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            protocol = {
                "protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                "recovered_from": recovered_from,
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
            for relative in SNAPSHOT_METADATA:
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
                "code_revision": revision, "recovered_from": recovered_from,
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
    parser.add_argument("--recover-unstarted", action="store_true",
                        help="explicitly preserve and recover one failed bootstrap with no started stage")
    args = parser.parse_args()
    run(args.home, recover_unstarted=args.recover_unstarted)


if __name__ == "__main__":
    main()
