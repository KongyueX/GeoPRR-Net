"""Serialize the frozen source conditional-view control and its conditional stages."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROTOCOL = "geoprr_v2_conditional_view_adversary_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_conditional_view_adversary_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_conditional_view_adversary_preflight_20261026/result.json"
MODULE = "experiments.geoprr_v2_conditional_view_adversary_control"
SNAPSHOT_METADATA = (
    "refine-logs/geoprr_research_reset_20261025/conditional_view_adversary_source_control_protocol.json",
    "docs/GEOPRR_V2_CONDITIONAL_VIEW_ADVERSARY_SOURCE_CONTROL_CN.md",
    "docs/data/geoprr_v2_conditional_view_adversary_source_control_20261026/protocol.json",
)
SNAPSHOT_CODE = (
    "geoprr_v2_conditional_view_adversary_control.py",
    "run_geoprr_v2_conditional_view_adversary_control.py",
    "test_geoprr_v2_conditional_view_adversary_control.py",
    "geoprr_v2_matched_identity_control.py", "geoprr_v2_cyclic_color_control.py",
    "geoprr_v2_head_warmstart_control.py", "geoprr_v2_needle_holdout_control.py",
    "geoprr_v2_high_resolution_direct.py", "geoprr_v2_roll_source.py",
    "geoprr_v2_semantic_tokens.py", "geoprr_v2_semantic_direct.py",
    "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
    "geoprr_v2_objectives.py", "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py",
    "evaluate_geoprr_v2_role_external.py", "train_geoprr_v2.py",
    "train_geoprr_v2_reference_relation.py", "run_geoprr_v2_optimization.py",
    "run_geoprr_v2_render_probe.py",
)
STAGES = ("metadata", "preflight", "detached_control", "control_cf", "reversal", "reversal_cf", "finalize")


def load(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def precheck(home, output=None):
    """Read code and frozen documents before any run-output mutation."""
    repo = Path(__file__).resolve().parents[1]
    output = Path(output) if output is not None else Path(home) / OUTPUT_RELATIVE
    relative_paths = tuple("experiments/" + name for name in SNAPSHOT_CODE) + SNAPSHOT_METADATA
    for relative in relative_paths:
        if not (repo / relative).is_file():
            raise FileNotFoundError("required frozen source file is missing: " + relative)
    private_protocol = repo / SNAPSHOT_METADATA[0]
    public_protocol = repo / SNAPSHOT_METADATA[2]
    if private_protocol.read_bytes() != public_protocol.read_bytes():
        raise ValueError("private and public frozen protocol snapshots differ")
    if load(private_protocol).get("protocol_id") != PROTOCOL:
        raise ValueError("the frozen protocol ID differs from this source control")
    snapshot = output / "code"
    if output.exists():
        if not snapshot.is_dir():
            raise FileExistsError("an existing run has no complete frozen code snapshot")
        for relative in relative_paths:
            saved = snapshot / relative
            if not saved.is_file() or saved.read_bytes() != (repo / relative).read_bytes():
                raise ValueError("run code snapshot is missing or differs: " + relative)
    return repo, relative_paths


def expected_path(home, output, stage):
    if stage == "metadata":
        return output / "source_metadata/completed.json"
    if stage == "preflight":
        return home / PREFLIGHT_RELATIVE
    if stage in {"control_cf", "reversal_cf"}:
        arm = "detached_control" if stage == "control_cf" else "reversal"
        return output / arm / "counterfactual_completed.json"
    if stage in {"detached_control", "reversal"}:
        return output / stage / "completed.json"
    return output / "completed.json"


def stage_action(home, output, stage):
    """Choose reuse or an explicit invocation; partial observations never replay."""
    expected = expected_path(home, output, stage)
    if stage in {"detached_control", "reversal", "control_cf", "reversal_cf"}:
        view = "source_counterfactual" if stage in {"control_cf", "reversal_cf"} else "source_id"
        if (expected.parent / (view + "_active.json")).exists():
            raise FileExistsError("evaluation has an unfinished observation marker")
    if expected.exists():
        saved = load(expected)
        if saved.get("protocol") != PROTOCOL or saved.get("status") not in {"complete", "passed"}:
            raise ValueError("existing stage summary is incomplete or belongs to another protocol")
        return "reuse"
    if stage == "preflight":
        if expected.parent.exists() and any(expected.parent.iterdir()):
            raise FileExistsError("preflight has partial outputs without its passed result")
    elif stage in {"detached_control", "reversal"}:
        arm = expected.parent
        if arm.exists() and any(arm.iterdir()):
            if (arm / "active_epoch.json").exists() or not (arm / "latest.pt").is_file():
                raise FileExistsError("arm has partial outputs without a complete-epoch checkpoint")
            if any((arm / ("source_id_" + name)).exists() for name in ("predictions.csv", "groups.csv")) and not (arm / "source_id_summary.json").is_file():
                raise FileExistsError("native prediction CSV exists without its complete summary")
    elif stage in {"control_cf", "reversal_cf"}:
        if any((expected.parent / ("source_counterfactual_" + name)).exists() for name in ("predictions.csv", "groups.csv")) and not (expected.parent / "source_counterfactual_summary.json").is_file():
            raise FileExistsError("counterfactual prediction CSV exists without its complete summary")
    elif stage == "metadata" and expected.parent.exists() and any(expected.parent.iterdir()):
        raise FileExistsError("metadata has partial outputs without its complete marker")
    return "run"


def process_running(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    return True


def compute_processes():
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def run(home):
    home = Path(home)
    output = home / OUTPUT_RELATIVE
    repo, relative_paths = precheck(home, output)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required for the existing cached encoder")
    revision = os.environ.get("GEOPRR_GENERALIZATION_REVISION", "").strip()
    if not revision:
        raise ValueError("GEOPRR_GENERALIZATION_REVISION must identify the frozen code")
    fresh = not output.exists()
    if fresh:
        witness = home / PREFLIGHT_RELATIVE
        if witness.parent.exists() and any(witness.parent.iterdir()):
            raise FileExistsError("preflight output exists outside this new run")
        output.mkdir(parents=True, exist_ok=False)
        for relative in relative_paths:
            target = output / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, target)
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "queue_state.json"
        state = load(state_path) if state_path.exists() else {
            "protocol": PROTOCOL, "code_revision": revision, "steps": {}, "invocations": [],
        }
        if state["protocol"] != PROTOCOL or state["code_revision"] != revision:
            raise ValueError("resume protocol or frozen code revision differs")
        prior_pid = state.get("pipeline_pid")
        if prior_pid and prior_pid != os.getpid() and process_running(prior_pid):
            raise RuntimeError("the recorded pipeline parent is still alive")
        for record in state["steps"].values():
            if record.get("status") == "running" and record.get("pid") and process_running(record["pid"]):
                raise RuntimeError("a recorded stage child is still alive")
        invocation = {"pipeline_pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
                      "explicit_resume": not fresh, "module_entry_before_torch_import": True}
        state["invocations"].append(invocation)
        state.update(status="running", pipeline_pid=os.getpid(), automatic_retry=False)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        frozen_config = None

        def validate_stage(saved, stage):
            if saved.get("protocol") != PROTOCOL or saved.get("status") not in {"complete", "passed"}:
                raise ValueError("stage returned an incomplete or different protocol summary")
            if not isinstance(saved.get("config"), dict) or (frozen_config is not None and saved["config"] != frozen_config):
                raise ValueError("stage configuration differs from the frozen metadata configuration")
            if stage in {"detached_control", "control_cf", "reversal", "reversal_cf"}:
                arm = "detached_control" if stage in {"detached_control", "control_cf"} else "reversal"
                if saved.get("arm") != arm or saved.get("alpha") != (0.0 if arm == "detached_control" else 0.05):
                    raise ValueError("stage arm or gradient coefficient differs from the frozen arm")
            return saved

        def save():
            precheck(home, output)
            state["current_invocation_seconds"] = time.perf_counter() - PROCESS_STARTED
            write_json(state_path, state)

        def step(stage):
            precheck(home, output)
            action = stage_action(home, output, stage)
            expected = expected_path(home, output, stage)
            if action == "reuse":
                return validate_stage(load(expected), stage)
            if stage not in {"metadata", "finalize"} and compute_processes():
                raise RuntimeError("GPU has an active compute process before the serialized stage")
            command = [sys.executable, "-m", MODULE, stage, "--home", str(home),
                       "--output", str(expected if stage == "preflight" else output)]
            record = {"status": "running", "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                      "expected": str(expected), "exit_code": None}
            if stage == "metadata":
                record.update(timeout_seconds=30, environment_overrides={"CUDA_VISIBLE_DEVICES": ""})
            state["steps"][stage] = record
            state["current_step"] = stage
            save()
            log_path = output / "logs" / (stage + ".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            tick = time.perf_counter()
            print(json.dumps({"event": "start", "step": stage}), flush=True)
            with log_path.open("ab") as log:
                environment = dict(os.environ)
                if stage == "metadata":
                    environment["CUDA_VISIBLE_DEVICES"] = ""
                process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=environment)
                record["pid"] = process.pid
                save()
                try:
                    code = process.wait(timeout=30 if stage == "metadata" else None)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                    record["timed_out"] = True
            record.update(exit_code=code, elapsed_parent_seconds=time.perf_counter()-tick,
                          finished_at=datetime.now(timezone.utc).isoformat(),
                          status="complete" if code == 0 and expected.is_file() else "failed")
            save()
            if record["status"] != "complete":
                raise RuntimeError(stage + " failed; inspect its private stage log")
            saved = load(expected)
            validate_stage(saved, stage)
            print(json.dumps({"event": "complete", "step": stage, "exit_code": code}), flush=True)
            return saved

        try:
            save()
            metadata = step("metadata")
            frozen_config = metadata["config"]
            if metadata.get("admitted") is True:
                pref = step("preflight")
                if pref.get("status") != "passed":
                    raise ValueError("the single source preflight did not pass")
                control = step("detached_control")
                if control.get("prerequisite_passed") is True:
                    step("control_cf")
                    reversal = step("reversal")
                    if reversal.get("native_joint_passed") is True:
                        step("reversal_cf")
                final = step("finalize")
            else:
                if metadata.get("admitted") is not False:
                    raise ValueError("metadata summary lacks its explicit admission decision")
                final = load(output / "completed.json")
            if final.get("protocol") != PROTOCOL or final.get("status") != "complete":
                raise ValueError("pipeline completion summary is missing or has a different protocol")
            if final.get("verdict") not in {"NOT_ADMITTED_SOURCE_VIEW_METADATA", "NOT_RETAINED",
                                            "CONDITIONAL_VIEW_SOURCE_CONTROL_PASSED"}:
                raise ValueError("pipeline completion has an invalid frozen verdict")
            invocation.update(elapsed_seconds=time.perf_counter()-PROCESS_STARTED,
                              finished_at=datetime.now(timezone.utc).isoformat(), status="complete")
            screen = {key: final[key] for key in (
                "protocol", "config", "status", "verdict", "completion_reason", "counts", "totals",
                "detached_control", "reversal", "joint_acceptance", "critic_effect", "runtime") if key in final}
            screen.update(code_revision=revision,
                supervisor_invocations=state["invocations"],
                supervisor_process_seconds=sum(item["elapsed_seconds"] for item in state["invocations"]
                                               if "elapsed_seconds" in item),
                supervisor_boundary="explicit runner module entry before torch import through finalized child; excludes downtime, screen/queue final writes and caller lifetime",
                screen_queue_boundary="screen and final queue writes follow measured supervisor interval",
                all_actual_parents_gone_and_GPU_empty_verified=False,
                audit="requires one fresh independent nonauthor saved-number CPU audit after all actual parents exit and GPU is empty")
            precheck(home, output)
            write_json(output / "screen_summary.json", screen)
            state.update(status="complete", current_step=None, verdict=final["verdict"])
            save()
            print(json.dumps({"event": "pipeline_complete", "verdict": final["verdict"]}), flush=True)
        except Exception as error:
            invocation.update(elapsed_seconds=time.perf_counter()-PROCESS_STARTED,
                              finished_at=datetime.now(timezone.utc).isoformat(), status="failed")
            state.update(status="failed", error=type(error).__name__)
            save()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    run(parser.parse_args(argv).home)


if __name__ == "__main__":
    main()
