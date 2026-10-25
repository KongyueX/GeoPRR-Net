"""Run the frozen source state-correspondence control in its admitted order."""
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


PROTOCOL = "geoprr_v2_state_change_supervision_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_state_change_supervision_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_state_change_supervision_preflight_20261026/result.json"
MODULE = "experiments.geoprr_v2_state_change_supervision_control"
SNAPSHOT_METADATA = (
    "refine-logs/geoprr_research_reset_20261025/state_change_supervision_source_control_protocol.json",
    "docs/GEOPRR_V2_STATE_CHANGE_SUPERVISION_SOURCE_CONTROL_CN.md",
    "docs/data/geoprr_v2_state_change_supervision_source_control_20261026/protocol.json",
)
SNAPSHOT_CODE = (
    "geoprr_v2_state_change_supervision_control.py",
    "run_geoprr_v2_state_change_supervision_control.py",
    "test_geoprr_v2_state_change_supervision_control.py",
    "geoprr_v2_matched_identity_control.py", "geoprr_v2_cyclic_color_control.py",
    "geoprr_v2_mean_tail_control.py",
    "geoprr_v2_head_warmstart_control.py", "geoprr_v2_needle_holdout_control.py",
    "geoprr_v2_high_resolution_direct.py", "geoprr_v2_roll_source.py",
    "geoprr_v2_semantic_tokens.py", "geoprr_v2_semantic_direct.py",
    "geoprr_v2_endpoint_source.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
    "geoprr_v2_objectives.py", "geoprr_v2_reference_relation.py",
    "geoprr_v2_role_triangle.py", "evaluate_geoprr_v2_role_external.py",
    "train_geoprr_v2.py", "train_geoprr_v2_reference_relation.py",
    "run_geoprr_v2_optimization.py",
)


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def precheck(home):
    """Check the full deployment before either supervisor or child writes output."""
    repo = Path(__file__).resolve().parents[1]
    for name in SNAPSHOT_CODE:
        if not (repo / "experiments" / name).is_file():
            raise FileNotFoundError("required state-change code snapshot is missing: experiments/" + name)
    for relative in SNAPSHOT_METADATA:
        if not (repo / relative).is_file():
            raise FileNotFoundError("required state-change metadata snapshot is missing: " + relative)
    frozen = load(repo / SNAPSHOT_METADATA[0])
    if frozen.get("protocol_id") != PROTOCOL or load(repo / SNAPSHOT_METADATA[2]) != frozen:
        raise ValueError("the two frozen state-change protocol snapshots differ")
    if frozen["locations"]["root_output"] != OUTPUT_RELATIVE or frozen["locations"]["pref"] != PREFLIGHT_RELATIVE:
        raise ValueError("the frozen state-change output locations differ")
    return frozen


def compute_processes():
    query = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return [line.strip() for line in query.stdout.splitlines() if line.strip()]


def require_idle_gpu():
    if compute_processes():
        raise RuntimeError("the source stage requires no active GPU compute process")


def _numeric_notice(**values):
    if any(type(value) not in (int, float, bool, type(None)) for value in values.values()):
        raise TypeError("supervisor notices admit only numeric values, booleans and null")
    print(json.dumps(values, allow_nan=False), flush=True)


def _identity(saved, preparation=None):
    if saved.get("protocol") != PROTOCOL or saved.get("status") != "complete":
        raise ValueError("the completed child has an incompatible protocol or status")
    if preparation is not None:
        for key in ("config", "source_contract", "target_contract"):
            if saved.get(key) != preparation.get(key):
                raise ValueError("the completed child changed its frozen " + key)


def _validate_arm(saved, arm):
    if saved.get("epochs") != 15:
        raise ValueError("the source arm must finish exactly fifteen complete epochs")
    native = saved.get("source_id")
    if native is None or native["summary"]["samples"] != 292:
        raise ValueError("the source arm lacks its complete native292 observation")
    native_passed = bool(saved["native_learning_gate"]["passed"] and saved["native_joint"]["passed"])
    cf = saved.get("source_counterfactual")
    expected_cf = native_passed if arm == "correct" else True
    if saved.get("counterfactual_eligible") is not expected_cf or (cf is not None) != expected_cf:
        raise ValueError("the source arm violated its frozen conditional counterfactual stage")
    if expected_cf and (cf["summary"]["samples"] != 292 or saved.get("counterfactual_joint") is None):
        raise ValueError("the source arm lacks its complete counterfactual292 observation")
    if not expected_cf and saved.get("counterfactual_joint") is not None:
        raise ValueError("an unrun counterfactual must retain a null result")
    if arm == "correct":
        expected_cyclic = bool(native_passed and expected_cf and saved["counterfactual_joint"]["passed"])
        if saved.get("cyclic_eligible") is not expected_cyclic:
            raise ValueError("cyclic eligibility differs from the complete correct-arm joint gate")


def run(home, *, newcpu_passed=False):
    frozen = precheck(home)
    if newcpu_passed is not True:
        raise ValueError("the explicitly completed necessary NEWCPU checks are required")
    revision = os.environ.get("GEOPRR_GENERALIZATION_REVISION", "")
    if not revision.strip():
        raise ValueError("GEOPRR_GENERALIZATION_REVISION must identify the frozen deployed code")
    home = Path(home)
    output, witness = home / OUTPUT_RELATIVE, home / PREFLIGHT_RELATIVE
    if output.exists():
        raise FileExistsError("the state-change supervisor already has complete or partial outputs")
    if witness.parent.exists() and any(witness.parent.iterdir()):
        raise FileExistsError("the state-change preflight already has complete or partial outputs")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output, 0o700)
    state = {"protocol": PROTOCOL, "code_revision": revision, "status": "running", "steps": {},
             "pipeline_pid": os.getpid(), "current_step": None, "automatic_resume": False,
             "necessary_NEWCPU_passed_explicitly": True, "supervisor_started_before_torch_import": True}

    def save_state():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["supervisor_process_seconds"] = time.perf_counter() - PROCESS_STARTED
        write_json(output / "queue_state.json", state)

    with (output / "queue.lock").open("x") as lock:
        os.chmod(output / "queue.lock", 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_state()
        try:
            (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
            os.chmod(output / "pipeline.pid", 0o600)
            write_json(output / "protocol.json", frozen)
            snapshot = output / "code_snapshot"
            snapshot.mkdir(mode=0o700)
            repo = Path(__file__).resolve().parents[1]
            for name in SNAPSHOT_CODE:
                shutil.copy2(repo / "experiments" / name, snapshot / name)
            for relative in SNAPSHOT_METADATA:
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(repo / relative, destination)

            def step(index, name, stage, destination, expected, *, arm=None, cpu_only=False):
                if expected.exists():
                    raise FileExistsError("the prospective child completion output already exists")
                command = [sys.executable, "-m", MODULE, stage, "--home", str(home), "--output", str(destination)]
                if arm is not None:
                    command.extend(["--arm", arm])
                if stage in {"preflight", "control"}:
                    command.append("--newcpu-passed")
                child_environment = os.environ.copy()
                if cpu_only:
                    child_environment["CUDA_VISIBLE_DEVICES"] = ""
                log_path = output / "logs" / (name + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                record = {"status": "running", "command": command, "log": str(log_path),
                          "expected": str(expected), "started_at": datetime.now(timezone.utc).isoformat(),
                          "CPU_only": cpu_only, "CUDA_visibility_overridden": cpu_only}
                state["current_step"] = name
                state["steps"][name] = record
                save_state()
                _numeric_notice(stage_index=index, started=True)
                tick = time.perf_counter()
                with log_path.open("x") as log:
                    os.chmod(log_path, 0o600)
                    process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                               env=child_environment)
                    record["pid"] = process.pid
                    save_state()
                    exit_code = process.wait()
                record.update(exit_code=exit_code, seconds=time.perf_counter() - tick,
                              finished_at=datetime.now(timezone.utc).isoformat(),
                              status="complete" if exit_code == 0 and expected.is_file() else "failed")
                save_state()
                if record["status"] != "complete":
                    raise RuntimeError("the state-change child failed; retain its private partial output and log")
                saved = load(expected)
                _numeric_notice(stage_index=index, completed=True, exit_code=exit_code,
                                seconds=record["seconds"])
                return saved

            prep_dir = output / "preparation"
            preparation = step(0, "preparation", "prepare", prep_dir, prep_dir / "completed.json", cpu_only=True)
            _identity(preparation)
            admitted = preparation.get("admitted")
            if type(admitted) is not bool:
                raise ValueError("preparation must record an explicit observation admission boolean")
            if admitted:
                if preparation.get("verdict") != "SOURCE_OBSERVATION_ADMITTED":
                    raise ValueError("admitted preparation has an incompatible verdict")
                if os.environ.get("HF_HUB_OFFLINE") != "1":
                    raise RuntimeError("HF_HUB_OFFLINE=1 is required before opening existing cached weights")
                require_idle_gpu()
                pref = step(1, "preflight", "preflight", witness, witness)
                if pref.get("protocol") != PROTOCOL or pref.get("status") != "passed":
                    raise ValueError("the one zero-update preflight did not pass")
                for key in ("config", "source_contract", "target_contract"):
                    if pref.get(key) != preparation.get(key):
                        raise ValueError("the preflight changed its frozen " + key)
                require_idle_gpu()
                correct_dir = output / "matched_state"
                correct = step(2, "correct", "control", correct_dir, correct_dir / "completed.json", arm="correct")
                _identity(correct, preparation)
                _validate_arm(correct, "correct")
                require_idle_gpu()
                if correct["cyclic_eligible"]:
                    cyclic_dir = output / "cycled_state"
                    cyclic = step(3, "cyclic", "control", cyclic_dir, cyclic_dir / "completed.json", arm="cyclic")
                    _identity(cyclic, preparation)
                    _validate_arm(cyclic, "cyclic")
                    require_idle_gpu()
            elif preparation.get("verdict") != "NOT_ADMITTED_SOURCE_OBSERVATION":
                raise ValueError("rejected preparation has an incompatible scientific stopping verdict")
            complete = step(4, "finalize", "finalize", output, output / "completed.json", cpu_only=True)
            _identity(complete, preparation)
            if complete.get("verdict") not in {
                "STATE_CHANGE_SOURCE_CONTROL_PASSED", "NOT_RETAINED", "NOT_ADMITTED_SOURCE_OBSERVATION",
            }:
                raise ValueError("the completed state-change control has an incompatible verdict")
            if not admitted and complete["verdict"] != "NOT_ADMITTED_SOURCE_OBSERVATION":
                raise ValueError("source rejection cannot produce a trained-model verdict")
            state.update(status="complete", current_step=None, verdict=complete["verdict"],
                         source_observation_admitted=admitted,
                         independent_nonauthor_saved_number_CPU_audit_required=True,
                         audit_requires_supervisor_and_all_actual_parents_exited=True)
            save_state()
            _numeric_notice(complete=True, source_observation_admitted=admitted,
                            preflight_executed=admitted, cyclic_executed="cyclic" in state["steps"],
                            passed=complete["verdict"] == "STATE_CHANGE_SOURCE_CONTROL_PASSED",
                            seconds=time.perf_counter() - PROCESS_STARTED)
            return complete
        except BaseException as error:
            state.update(status="failed", error=type(error).__name__ + ": " + str(error),
                         partial_outputs_retained=True)
            save_state()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--newcpu-passed", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.home, newcpu_passed=args.newcpu_passed)
    except BaseException:
        _numeric_notice(failed=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
