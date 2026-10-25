"""Run one frozen fifteen-epoch source mean-tail objective control."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

from experiments.geoprr_v2_mean_tail_control import (
    CONFIG, OUTPUT_RELATIVE, PREFLIGHT_COUNTS, PREFLIGHT_RELATIVE, PREPARED_RELATIVE,
    PROTOCOL, count_summary, validate_prepared, validate_totals,
)
from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


SNAPSHOT_METADATA = (
    "refine-logs/geoprr_research_reset_20261025/mean_tail_source_control_protocol.json",
    "docs/GEOPRR_V2_MEAN_TAIL_SOURCE_CONTROL_CN.md",
    "docs/data/geoprr_v2_mean_tail_source_control_20261026/protocol.json",
)
SNAPSHOT_CODE = (
    "geoprr_v2_mean_tail_control.py", "run_geoprr_v2_mean_tail_control.py",
    "test_geoprr_v2_mean_tail_control.py", "geoprr_v2_matched_identity_control.py",
    "geoprr_v2_cyclic_color_control.py",
    "geoprr_v2_head_warmstart_control.py",
    "geoprr_v2_needle_holdout_control.py", "geoprr_v2_high_resolution_direct.py",
    "geoprr_v2_roll_source.py", "geoprr_v2_semantic_tokens.py", "geoprr_v2_semantic_direct.py",
    "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
    "geoprr_v2_objectives.py", "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py",
    "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
    "train_geoprr_v2_reference_relation.py", "run_geoprr_v2_optimization.py",
    "run_geoprr_v2_render_probe.py",
)


def load(path):
    return json.loads(path.read_text())


def run(home):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required for existing cached ImageNet weights")
    revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
    if not revision.strip():
        raise ValueError("GEOPRR_GENERALIZATION_REVISION must identify the frozen code")
    repo = Path(__file__).resolve().parents[1]
    for relative in SNAPSHOT_METADATA:
        if not (repo / relative).is_file():
            raise FileNotFoundError(f"required bootstrap snapshot metadata is missing: {relative}")
    for name in SNAPSHOT_CODE:
        if not (repo / "experiments" / name).is_file():
            raise FileNotFoundError(f"required bootstrap snapshot code is missing: experiments/{name}")
    prepared = validate_prepared(home)
    output, witness = home / OUTPUT_RELATIVE, home / PREFLIGHT_RELATIVE
    if witness.parent.exists() and any(witness.parent.iterdir()):
        raise FileExistsError("the mean-tail preflight already has complete or partial outputs")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=PROTOCOL, code_revision=revision, automatic_resume=False)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        try:
            write_json(output / "protocol.json", {
                "protocol": PROTOCOL, "code_revision": revision, "config": CONFIG,
                "source_contract": prepared["source_contract"],
                "preparation_protocol": prepared["protocol"], "prepared_relative": PREPARED_RELATIVE,
                "kind": "one matched fifteen-epoch source loss-aggregation control; not formal V32",
                "stages": ["preflight", "control"], "automatic_resume": False,
                "preflight_budget": PREFLIGHT_COUNTS, "objective_config": CONFIG["objective"],
                "formal_budget": {
                    "epochs": 15, "train_images": 82440, "train_CNN_calls": 1290,
                    "train_point_calls": 1290, "backward_attempts": 1290,
                    "updates_plus_AMP_skips": 1290,
                    "native_images": 292, "native_CNN_calls": 5, "native_point_calls": 5,
                    "counterfactual_if_eligible_images": 292,
                    "counterfactual_if_eligible_CNN_calls": 5,
                    "counterfactual_if_eligible_point_calls": 5,
                    "native_failure_CNN_images": 82732, "native_failure_CNN_calls": 1295,
                    "counterfactual_done_CNN_images": 83024, "counterfactual_done_CNN_calls": 1300,
                    "preflight_counts_separate": True,
                    "budget_qualification": "same identity-reference fifteen-epoch scheduled train images and backward attempts; actual AMP skips may differ",
                },
                "excluded_counts": {
                    "source_ood_needle03_RGB_images": 0, "source_ood_needle03_CNN_calls": 0,
                    "industrial_RGB_images": 0, "industrial_CNN_calls": 0,
                    "RF_RGB_images": 0, "RF_CNN_calls": 0,
                    "external_backward_calls": 0, "external_optimizer_updates": 0,
                },
                "supervisor_process_started_before_torch_import": True,
                "supervisor_seconds_at_protocol": time.perf_counter() - PROCESS_STARTED,
            })
            write_json(output / "config.json", CONFIG)
            snapshot = output / "code_snapshot"
            snapshot.mkdir()
            for name in SNAPSHOT_CODE:
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
                command = [sys.executable, "-m", "experiments.geoprr_v2_mean_tail_control", stage,
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
                if name == "preflight":
                    objective = saved.get("source_preflight_mean_tail_objective")
                    if (saved["status"] != "passed" or saved["counts"] != PREFLIGHT_COUNTS
                            or saved.get("identity_actual_rgb") is not True
                            or saved.get("train_nominal_slots") != [0,1,2,0]
                            or saved.get("train_actual_orders") != [[0,1,2]] * 4
                            or saved.get("objective_config") != CONFIG["objective"]
                            or type(objective) not in (int,float) or not math.isfinite(objective)
                            or "source_smooth_l1" in saved):
                        raise ValueError("the mean-tail preflight differs from its fixed objective and identity-input witness")
            control = completed["source_control"]
            eligible = control["counterfactual_eligible"]
            if (control["status"] != "complete" or control["epochs"] != 15
                    or eligible != bool(control["native_learning_gate"]["passed"] and control["native_primary"]["passed"])
                    or control["completion_reason"] != (
                        "counterfactual_observed" if eligible else "native_learning_or_primary_failed")):
                raise ValueError("source control did not complete its frozen native and conditional-color gate")
            if (control["source_counterfactual"] is None) != (not eligible):
                raise ValueError("counterfactual result presence differs from its native eligibility")
            expected_verdict = "MEAN_TAIL_SOURCE_CONTROL_PASSED" if (
                eligible and control["counterfactual_gate"]["passed"]) else "NOT_RETAINED"
            if control["verdict"] != expected_verdict:
                raise ValueError("mean-tail verdict differs from its native and counterfactual joint predicates")
            validate_totals(control["totals"], eligible)
            write_json(output / "screen_summary.json", {
                "protocol": PROTOCOL, "config": CONFIG, "source_contract": prepared["source_contract"],
                "code_revision": revision, "status": "complete", "epochs": 15,
                "completion_reason": control["completion_reason"], "verdict": control["verdict"],
                "objective_config": CONFIG["objective"],
                "native_learning_gate": control["native_learning_gate"], "native_primary": control["native_primary"],
                "counterfactual_eligible": eligible, "counterfactual_gate": control["counterfactual_gate"],
                "source_id": control["source_id"], "source_counterfactual": control["source_counterfactual"],
                "counts": count_summary(control["totals"]),
                "preflight_counts_separate": completed["preflight"]["counts"],
                "supervisor_stage_seconds": stage_seconds,
                "supervisor_process_seconds": time.perf_counter() - PROCESS_STARTED,
                "scope": "one source mean-tail objective control; no automatic promotion or external evaluation",
                "budget_qualification": "same scheduled15E/82440 train images/1290 backward attempts as the saved identity control",
            })
            queue.state.update(status="complete", current_step=None,
                               completion_reason=control["completion_reason"], verdict=control["verdict"],
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
