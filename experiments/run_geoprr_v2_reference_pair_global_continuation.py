"""Run the frozen global65 control under the paper-evidence continuation policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
from experiments.run_geoprr_v2_reference_pair_update import compare_best64, best64_protection
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.run_geoprr_v2_role_triangle import compare_best48, best48_protection
from experiments.summarize_geoprr_v2_relative_state import compare_to_ordinary
PROTOCOL = "geoprr_v2_reference_pair_global_paper_continuation_v1"
REVISION = "docs/GEOPRR_V2_PAPER_EVIDENCE_AMENDMENT_CN.md"
SNAPSHOT_FILES = (
    "geoprr_v2_reference_pair_reader.py", "geoprr_v2_reference_pair_update.py",
    "train_geoprr_v2_reference_pair_update.py", "train_geoprr_v2_reference_relation.py",
    "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py", "geoprr_v2_data.py", "geoprr_v2_model.py",
    "train_geoprr_v2.py", "run_geoprr_v2_optimization.py",
    "run_geoprr_v2_render_probe.py", "run_geoprr_v2_role_triangle.py",
    "run_geoprr_v2_reference_feedback.py", "summarize_geoprr_v2_relative_state.py",
    "run_geoprr_v2_reference_pair_update.py",
)
def load(path: Path) -> dict:
    return json.loads(path.read_text())
def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
def verify_snapshot(repo: Path, original: Path, protocol: dict, config: Path) -> list[str]:
    snapshot = original / "code_snapshot"
    checked = []
    for name in (*SNAPSHOT_FILES, "geoprr_v2_reference_pair_update_global.json"):
        current = config if name.endswith(".json") else repo / "experiments" / name
        frozen = snapshot / name
        require(current.is_file() and frozen.is_file() and current.read_bytes() == frozen.read_bytes(),
                f"current file differs from frozen snapshot: {name}")
        checked.append(name)
    require(load(config) == protocol["configs"]["global"], "global config differs from formal protocol")
    return checked
def verify_completed(reader: Path, expected_config: dict) -> tuple[dict, dict]:
    complete, metadata = load(reader / "completed.json"), load(reader / "run.json")
    require(complete["epochs"] == 65 and complete["seed"] == 20262020, "reader epoch or seed differs")
    require(complete["protocol"] == expected_config["protocol"] == metadata["protocol"],
            "reader protocol differs")
    require(metadata["config"] == expected_config, "reader config differs from frozen config")
    require((metadata["parameters"], metadata["shared_parameters"], metadata["head_parameters"])
            == (7465988, 7431248, 34740), "reader parameter counts differ")
    require((metadata["train_unique_images"], metadata["train_items"], metadata["validation_images"])
            == (18590, 22686, 1506), "reader populations differ")
    require((metadata["initialization"]["source_epoch"], metadata["initialization"]["loaded_full_tensors"],
             metadata["initialization"]["preserved_all_trained_weights"]) == (64, 497, True),
            "reader initialization differs from full best64")
    totals = complete["totals"]
    require((totals["train_batches"], totals["donor_images"], totals["recipient_images"],
             totals["validation_images"]) == (65 * 709, 65 * 22686, 65 * 8192, 65 * 1506),
            "reader exposure totals differ")
    require((reader / "best_predictions.csv.gz").is_file(), "reader best predictions are missing")
    return complete, metadata
def evaluate_global(original: Path, continuation: Path, global_reader: Path, preserve: dict,
                    ordinary: Path, best48: Path, best64: Path) -> dict:
    result = compare_to_ordinary(global_reader / "best_predictions.csv.gz", ordinary)
    result["original_G1"] = {"G0_label_gate_passed": True, "RGB_checks": result["checks"],
                             "gate_passed": result["gate_passed"]}
    result["best48_comparison"] = compare_best48(global_reader / "best_predictions.csv.gz", best48, ordinary)
    result["best48_protection"] = best48_protection(result["best48_comparison"])
    result["best64_comparison"] = compare_best64(global_reader / "best_predictions.csv.gz", best64, ordinary)
    result["best64_protection"] = best64_protection(result["best64_comparison"])
    complete, metadata = verify_completed(global_reader, load(original / "protocol.json")["configs"]["global"])
    result.update(best_epoch=complete["best_epoch"], parameters=metadata["parameters"], costs=complete)
    write_json(continuation / "validation_evaluation/global/summary.json", result)
    p, g = preserve["summary"]["candidate"], result["summary"]["candidate"]
    mechanism_checks = {
        "preserve_nmae_at_least5_percent_better_than_global": p["nmae_percent_fs"] <= .95 * g["nmae_percent_fs"],
        "preserve_tail_not_worse": p["tail10_percent_fs"] <= g["tail10_percent_fs"],
        "preserve_above5_not_worse": p["above_5_percent_fs"] <= g["above_5_percent_fs"],
    }
    checks = {
        **mechanism_checks,
        **preserve["best48_protection"]["checks"], **preserve["best64_protection"]["checks"],
    }
    decision = {
        "decision": "matched_paper_evidence_complete",
        "original_G2_result": "not_passed_under_original_protocol_due_failed_G1_or_anchor_eligibility",
        "mechanism_screen_checks": mechanism_checks, "screen_passed": all(mechanism_checks.values()),
        "recomputed_original_G2_checks": checks,
        "recomputed_original_G2_all_checks_passed": all(checks.values()), "publication_gate": False,
        "interpretation": "single-seed repeatedly viewed source development; report G2 and anchors without treating them as a paper veto",
        "preserve": p, "global": g, "original_G1": preserve["original_G1"],
        "best48_protection": preserve["best48_protection"],
        "best64_protection": preserve["best64_protection"],
        "costs": {"preserve": preserve["costs"], "global": complete},
        "test_used": False, "seed_expansion": False,
    }
    write_json(continuation / "paper_evidence_decision.json", decision)
    return decision
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    home = parser.parse_args().home
    repo = Path(__file__).resolve().parents[1]
    original = home / "runs/geoprr_v2_tail/iteration_v8_reference_pair_update"
    continuation = original / "paper_continuation/global65_20261022"
    continuation.mkdir(parents=True, exist_ok=True)
    with (continuation / "paper_continuation.lock").open("w") as continuation_lock:
        fcntl.flock(continuation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (original / "queue.lock").open("r") as original_lock:
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old_queue, old_decision, protocol = (load(original / name) for name in
                                                  ("queue_state.json", "validation_decision.json", "protocol.json"))
            require(protocol["code_revision"] == "d6376a0", "formal code revision differs")
            preserve_step = old_queue["steps"].get("train_preserve_s20262020", {})
            require(old_queue["status"] == "validation_rejected", "original queue is not validation_rejected")
            require(old_decision["decision"] == "reference_pair_update_validation_rejected",
                    "original validation rejection is missing")
            require(preserve_step.get("status") == "complete", "preserve step is not complete")
            require(not any("global" in name for name in old_queue["steps"]), "old controller already recorded global")
            for pid in (old_queue.get("pipeline_pid"), preserve_step.get("pid")):
                require(not pid or not process_running(pid), f"original process is still running: {pid}")
            require(not compute_processes(), "GPU0 still has a compute process")
            preserve_reader = original / "readers/preserve_s20262020"
            verify_completed(preserve_reader, protocol["configs"]["preserve"])
            preserve = load(original / "validation_evaluation/preserve/summary.json")
            config = repo / "experiments/configs/geoprr_v2_reference_pair_update_global.json"
            checked = verify_snapshot(repo, original, protocol, config)
            require((repo / REVISION).is_file(), "paper-evidence policy revision is missing")
            global_reader = original / "readers/global_s20262020"
            completed, latest = global_reader / "completed.json", global_reader / "latest.pt"
            decision_path, state_path = continuation / "decision.json", continuation / "queue_state.json"
            if completed.exists() or latest.exists():
                require(decision_path.exists() and state_path.exists(),
                        "global artifacts exist without paper-continuation provenance")
            require(completed.exists() or latest.exists() or not any(
                (global_reader / name).exists() for name in ("history.jsonl", "run.json", "config.json")),
                "partial global artifacts exist without a resumable checkpoint")
            source = home / "rendering/compositional_source_v1/manifests"
            initial = home / "runs/geoprr_v2_tail/iteration_v7_role_triangle/readers/full_s20262020/best.pt"
            command = [str(home / "code/../.venv/bin/python"), "-m", "experiments.train_geoprr_v2_reference_pair_update",
                       "--config", str(config),
                       "--train-manifest", str(source / "train.jsonl"),
                       "--val-manifest", str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
                       "--pair-manifest", str(source / "train_edges.jsonl"),
                       "--init-checkpoint", str(initial), "--output", str(global_reader)]
            base_command = list(command)
            if latest.exists() and not completed.exists():
                command += ["--resume", str(latest)]
            continuation_revision = os.environ.get("GEOPRR_CONTINUATION_REVISION")
            require(bool(continuation_revision), "GEOPRR_CONTINUATION_REVISION is required")
            record = {
                "protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
                "policy_revision": REVISION, "code_revision": protocol["code_revision"],
                "continuation_code_revision": continuation_revision,
                "authoritative_old_gate": {key: old_decision[key] for key in
                                            ("decision", "original_G1", "best48_protection", "best64_protection")},
                "eligibility_revision": "run the one frozen matched global65 after preserving the original strict-gate rejection",
                "original_queue": str(original / "queue_state.json"), "cwd": str(repo),
                "original_validation_decision": str(original / "validation_decision.json"),
                "base_command": base_command, "global_output": str(global_reader),
                "expected": str(completed), "resume_checkpoint": str(latest),
                "snapshot_files_equal": checked,
                "eligibility_checks": {"original_validation_rejected": True, "preserve_complete": True,
                                       "old_parent_child_exited": True, "GPU0_idle": True,
                                       "old_global_step_absent": True, "frozen_snapshot_equal": True},
                "scope": "one frozen global65 only; no model, loss, seed, epoch or matrix extension",
            }
            if decision_path.exists():
                prior = load(decision_path)
                require(all(prior[key] == record[key] for key in record if key != "created_at"),
                        "continuation decision differs from its first recorded policy")
            else:
                write_json(decision_path, record)
            run = OptimizationRun(home, continuation, [20262020])
            run.state.update(protocol=PROTOCOL, status="running", current_step="train_global_s20262020",
                             original_queue=str(original / "queue_state.json"),
                             original_validation_decision=str(original / "validation_decision.json"))
            (continuation / "pipeline.pid").write_text(str(os.getpid()) + "\n")
            run.save()
            try:
                run.step("train_global_s20262020", command, completed)
                paper_decision = evaluate_global(
                    original, continuation, global_reader, preserve,
                    home / "diagnostics/gt_atlas_full1506_v1/predictions.csv",
                    home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best_predictions.csv.gz",
                    initial.parent / "best_predictions.csv.gz")
                run.state.update(status="complete", current_step=None,
                                 original_G2_result=paper_decision["original_G2_result"])
                run.save()
            except Exception as exc:
                run.state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                run.save()
                raise
if __name__ == "__main__":
    main()
