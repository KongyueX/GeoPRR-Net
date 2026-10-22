"""Run a matched four-epoch source-only generalization screen."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from experiments.run_geoprr_v2_optimization import OptimizationRun, write_json
from experiments.run_geoprr_v2_render_probe import compute_processes


PROTOCOL = "geoprr_v2_generalization_matched_screen_v1"
ARMS = ("control", "bnfixed", "mixstyle")


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--study", choices=("normalization", "roi_padding"), default="normalization")
    args = parser.parse_args()
    home = args.home
    roi_study = args.study == "roi_padding"
    arms = ("roi", "roi_bnfixed") if roi_study else ARMS
    config_prefix = "geoprr_v2_generalization_v10" if roi_study else "geoprr_v2_generalization_v9"
    iteration = "iteration_v10_roi_padding_20261023" if roi_study else "iteration_v9_20261023"
    protocol_name = "geoprr_v2_roi_generalization_matched_screen_v1" if roi_study else PROTOCOL
    repo = Path(__file__).resolve().parents[1]
    output = home / f"runs/geoprr_generalization/{iteration}/matched_training"
    anchor = home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if compute_processes():
            raise RuntimeError("GPU has an active compute process")
        configs = {arm: load(repo / f"experiments/configs/{config_prefix}_{arm}.json") for arm in arms}
        preflight_name = "geoprr_generalization_v10_roi_preflight_20261023" if roi_study else "geoprr_generalization_v9_preflight_20261023"
        preflight = load(home / f"diagnostics/{preflight_name}/result.json")
        if preflight["status"] != "passed" or preflight["configs"] != configs or preflight["counts"]["optimizer_updates"]:
            raise ValueError("targeted runtime witness or frozen configs differ")
        identity_key = "validation_input_exact_identity" if roi_study else "eval_MixStyle_exact_identity"
        if preflight[identity_key] is not True:
            raise ValueError("evaluation identity witness failed")
        common = []
        for config in configs.values():
            common.append({key: value for key, value in config.items() if key not in ("name", "generalization")})
        if not all(value == common[0] for value in common):
            raise ValueError("matched arms differ beyond the generalization policy")
        family = load(home / "runs/geoprr_generalization/iteration_v9_20261023/family_screen/queue_state.json")
        if family["status"] != "complete":
            raise ValueError("existing-family screen is incomplete")
        if roi_study and load(home / "runs/geoprr_generalization/iteration_v9_20261023/matched_training/queue_state.json")["status"] != "complete":
            raise ValueError("normalization/style study is incomplete")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": protocol_name, "code_revision": revision, "anchor": str(anchor),
                    "anchor_selection": "viewed external development: feedback48 improves both domains versus v7best64; local57 retains RF100 advantage",
                    "configs": configs, "arms": list(arms), "epochs_each": 4, "seed": 20262020,
                    "study": args.study,
                    "prior_controls": "v9 control/bnfixed reused; same initialization, optimizer, seed and source exposures" if roi_study else None,
                    "checkpoint_selection": "within run source1506 NMAE; evaluate source-best and last externally",
                    "external_scope": "viewed development, not blind or target-supervised training",
                    "external_training_labels_used": False, "official_test_used": False,
                    "exposure_each_epoch": {"unique": 18590, "donors": 22686, "recipients": 8192, "batches": 709},
                    "preflight": {"status": preflight["status"], "counts": preflight["counts"],
                                  identity_key: preflight[identity_key]}}
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and load(protocol_path) != protocol:
            raise ValueError("stored protocol differs; do not overwrite a partial experiment")
        write_json(protocol_path, protocol)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            names = ["train_geoprr_v2_generalization.py", "geoprr_v2_generalization_style.py",
                         "evaluate_geoprr_v2_generalization.py", "run_geoprr_v2_generalization.py",
                         "geoprr_v2_reference_relation.py", "geoprr_v2_role_triangle.py", "geoprr_v2_data.py"]
            if roi_study:
                names.append("geoprr_v2_roi_generalization_data.py")
            for name in names:
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        run = OptimizationRun(home, output, [20262020])
        run.state.update(protocol=protocol_name, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        run.save()
        evaluations, aliases = {}, {}
        try:
            for arm in arms:
                reader = output / "readers" / arm
                config = repo / f"experiments/configs/{config_prefix}_{arm}.json"
                command = [sys.executable, "-m", "experiments.train_geoprr_v2_generalization", "--config", str(config),
                           "--train-manifest", str(home / "rendering/compositional_source_v1/manifests/train.jsonl"),
                           "--val-manifest", str(home / "runs/geoprr_v2/manifests/validation.jsonl"),
                           "--pair-manifest", str(home / "rendering/compositional_source_v1/manifests/train_edges.jsonl"),
                           "--init-checkpoint", str(anchor), "--output", str(reader)]
                if (reader / "latest.pt").exists() and not (reader / "completed.json").exists():
                    command += ["--resume", str(reader / "latest.pt")]
                if compute_processes():
                    raise RuntimeError("GPU became busy before training")
                run.state["current_step"] = "train_" + arm
                run.save()
                run.step("train_" + arm, command, reader / "completed.json")
                complete = load(reader / "completed.json")
                totals = complete["totals"]
                if (complete["epochs"], totals["train_batches"], totals["donor_images"], totals["recipient_images"],
                        totals["validation_images"]) != (4, 2836, 90744, 32768, 6024):
                    raise ValueError("completed arm exposure differs")
                evaluations[arm] = {"training": complete, "checkpoints": {}}
                for kind, filename in (("best_source", "best.pt"), ("last", "latest.pt")):
                    if kind == "last" and complete["best_epoch"] == 4:
                        aliases[arm + ":last"] = arm + ":best_source (same epoch4 weights)"
                        evaluations[arm]["checkpoints"][kind] = {"reuses": "best_source", "epoch": 4}
                        continue
                    evaluations[arm]["checkpoints"][kind] = {}
                    for dataset, folder in (("industrial", "Industrial-1395"), ("rf100", "RF100-VL")):
                        if compute_processes():
                            raise RuntimeError("GPU became busy before external evaluation")
                        step = f"eval_{arm}_{kind}_{dataset}"
                        destination = output / "evaluations" / step
                        run.state["current_step"] = step
                        run.save()
                        command = [sys.executable, "-m", "experiments.evaluate_geoprr_v2_generalization",
                                   "--checkpoint", str(reader / filename), "--dataset-root", str(home / "datasets" / folder),
                                   "--dataset-name", dataset, "--output", str(destination),
                                   "--batch-size", "16", "--workers", "4", "--device", "cuda"]
                        run.step(step, command, destination / "summary.json")
                        result = load(destination / "summary.json")
                        evaluations[arm]["checkpoints"][kind][dataset] = {
                            "summary": result["summary"], "checkpoint_epoch": result["checkpoint_epoch"],
                            "forward_counts": result["forward_counts"], "timing": result["timing"]}
                write_json(output / "screen_summary.json", {"scope": "partial viewed external development screen",
                                                           "arms": evaluations, "aliases": aliases})
            write_json(output / "screen_summary.json", {"scope": "complete viewed external development screen; no automatic model promotion",
                                                       "arms": evaluations, "aliases": aliases})
            run.state.update(status="complete", current_step=None, next_action="review continuous per-domain effects and plan next distinct iteration")
            run.save()
        except Exception as error:
            run.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            run.save()
            raise


if __name__ == "__main__":
    main()
