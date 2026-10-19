"""Compare a source-selected mode readout after rejecting the dual-expert route."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from experiments.run_geoprr_v2_optimization import OptimizationRun, SEEDS, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("GEOPRR_HOME", "/root/autodl-tmp/geoprrnet")))
    args = parser.parse_args()
    decision_path = args.home / "runs/geoprr_v2_optimization/validation_decision.json"
    if not decision_path.exists():
        raise RuntimeError("await the current main reader's full source-validation decision")
    decision = json.loads(decision_path.read_text())
    if decision["decision"] != "return_to_original_mobilenet":
        raise RuntimeError("the source-validation decision has not activated the original-MobileNet tail route")
    output = args.home / "runs/geoprr_v2_tail/iteration_v1"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = OptimizationRun(args.home, output, list(SEEDS))
        fixed_ids = json.loads((args.home / "runs/geoprr_v2_optimization/manifests/source_ids.json").read_text())
        for name in ("train", "validation", "test"):
            ids = [row["sample_id"] for row in read_jsonl(run.manifests / f"{name}.jsonl")]
            if ids != fixed_ids[name]:
                raise ValueError(f"tail iteration must preserve the fixed {name} sample identities")
        run.state["protocol"] = "geoprr_v2_original_mobile_modal_readout_v1"
        run.save()
        write_json(output / "protocol.json", {
            "protocol": run.state["protocol"], "seeds": list(SEEDS),
            "parent_validation_decision": decision,
            "backbone_and_training": "unchanged original MobileNetV4 field recipe",
            "training_samples": 14494, "validation_samples": 1506, "test_samples": 4000,
            "selected_modal_radius": 8, "selection_population": "source_validation_seed_20262020",
            "selection_evidence": str(args.home / "runs/geoprr_v2_tail/diagnosis_seed_20262020/summary.json"),
            "test_used_for_radius_selection": False, "new_trainable_parameters": 0,
            "conditions": ["native"], "scope": "exploratory comparison on historically used official test images",
        })
        try:
            for seed in SEEDS:
                checkpoint = run.reader("baseline", seed)
                run.evaluate("baseline", seed, checkpoint)
                run.evaluate("baseline", seed, checkpoint, modal_radius=8)
            from experiments.export_geoprr_v2_optimization import export
            export(output, output / "exports")
            run.state["status"] = "complete"
            run.save()
        except Exception:
            run.state["status"] = "failed"
            run.save()
            raise


if __name__ == "__main__":
    main()
