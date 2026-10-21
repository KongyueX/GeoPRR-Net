"""Check full-best57 initialization and the paired RGB feedback runtime."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from experiments.geoprr_v2_data import normalized_target
from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationDataset, ReferenceRelationReader, reference_relation_losses,
)
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, seed_everything
from experiments.train_geoprr_v2_reference_relation import validation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    started, home = time.perf_counter(), args.home
    output = home / "diagnostics/reference_feedback_preflight_v1_20261021/runtime_preflight.json"
    if output.exists():
        raise FileExistsError("runtime preflight is already complete")
    if compute_processes():
        raise RuntimeError("GPU is occupied; do not overlap the runtime check")
    repo = Path(__file__).resolve().parents[1]
    configs = {arm: json.loads((repo / f"experiments/configs/geoprr_v2_reference_feedback_{arm}.json").read_text())
               for arm in ("feedback", "local")}
    comparable = []
    for arm, normalization in (("feedback", "global"), ("local", "local")):
        config = configs[arm]
        if (config["protocol"] != "geoprr_v2_reference_feedback_source_only_v1"
                or config["initialization"] != "full_best57"
                or config["training"]["seed"] != 20262020
                or config["model"]["coupling"] != "joint"
                or config["model"]["reference_normalization"] != normalization):
            raise ValueError("feedback protocol, initialization or normalization differs")
        comparable.append({key: value for key, value in config.items() if key not in ("name", "model")}
                          | {"model": {key: value for key, value in config["model"].items()
                                       if key != "reference_normalization"}})
    if comparable[0] != comparable[1]:
        raise ValueError("paired configurations differ beyond reference normalization")
    initial = home / "runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020/best.pt"
    checkpoint = torch.load(initial, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 57 or checkpoint["protocol"] != "geoprr_v2_reference_relation_source_only_v1":
        raise ValueError("initial checkpoint is not the frozen reference-relation best57")
    config, contract = configs["feedback"], checkpoint["data_contract"]
    if config["training"] != checkpoint["config"]["training"] or config["data"] != checkpoint["config"]["data"]:
        raise ValueError("training arrangement or data configuration differs from best57")
    expected_exposure = {"donor_items": 22686, "recipient_images": 8192,
                         "total_images": 30878, "optimizer_batches": 709}
    if config["exposure"] != expected_exposure or any(
            contract["train"][key] != value for key, value in
            (("unique_images", 18590), ("donor_items", 22686), ("recipient_items", 8192))):
        raise ValueError("frozen training exposure differs")
    validation = ReferenceRelationDataset(home / "runs/geoprr_v2/manifests/validation.jsonl",
                                         data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=False)
    if len(validation) != 1506 or validation.supervision_contract() != contract["validation"]:
        raise ValueError("validation inputs or geometry supervision differ from best57")
    references = read_predictions(initial.parent / "best_predictions.csv.gz")
    expected = {row["sample_id"]: (row["group_id"], float(row["target"])) for row in references}
    actual = {row["sample_id"]: (row["group_id"], float(torch.tensor(normalized_target(row), dtype=torch.float32)))
              for row in validation.rows}
    if actual != expected:
        raise ValueError("original1506 validation ID/group/float32 target differs")
    teacher_path = home / "diagnostics/reference_relation_preflight_v1_20261021/teacher_gate.json"
    teacher = json.loads(teacher_path.read_text())
    if teacher["numerical_gate_passed"] is not True:
        raise ValueError("existing teacher-label gate did not pass")
    by_id = {row["sample_id"]: index for index, row in enumerate(validation.rows)}
    samples = [validation[by_id[key]] for key in ("sync_3172", "sync_10")]
    if any(sample["input_failure"] for sample in samples):
        raise ValueError("fixed validation RGB input failed")
    batch = device_batch(default_collate(samples), torch.device("cuda"))
    metadata, initial_state = {}, None
    for arm in ("feedback", "local"):
        arm_started = time.perf_counter()
        seed_everything(20262020)
        model = ReferenceRelationReader(**configs[arm]["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        state = model.state_dict()
        if not all(torch.equal(value, checkpoint["model"][key]) for key, value in state.items()):
            raise ValueError("full best57 initialization differs")
        if initial_state is None:
            initial_state = {key: value.clone() for key, value in state.items()}
        elif any(not torch.equal(value, initial_state[key]) for key, value in state.items()):
            raise ValueError("paired initial parameters or buffers differ")
        model = model.cuda()
        model.set_shared_trainable(False)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(batch["image"])
        terms = reference_relation_losses(result, batch)
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError("runtime preflight loss is nonfinite")
        probability = result["distribution"]
        mass_error = float((probability.sum(-1)-1).abs().max().detach())
        if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()) or mass_error > 1e-4:
            raise ValueError("runtime readout loses probability mass")
        csv_rows = validation_rows(result, batch)
        if len(csv_rows) != 2 or not all(row["valid"] for row in csv_rows):
            raise ValueError("validation CSV interface rejects runtime predictions")
        local_replay_max = None
        if arm == "local":
            saved = {row["sample_id"]: row for row in references}
            local_replay_max = max(abs(row["prediction"]-float(saved[row["sample_id"]]["prediction"]))
                                   for row in csv_rows)
            if local_replay_max > 1e-5:
                raise ValueError("local normalization does not reproduce the frozen best57")
        if arm == "feedback":
            terms["total"].backward()
            gradients = [parameter.grad for parameter in model.state_head.parameters()]
            if any(value is None or not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError("head gradient is absent or nonfinite")
            if any(parameter.grad is not None for parameter in model.shared_parameters()):
                raise ValueError("frozen shared module received a gradient")
            del gradients
        if any(not torch.equal(value.cpu(), checkpoint["model"][key]) for key, value in model.state_dict().items()
               if key.startswith(("backbone.", "lateral.", "refine."))):
            raise ValueError("frozen shared parameters or statistics changed")
        torch.cuda.synchronize()
        metadata[arm] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                         "head_parameters": sum(parameter.numel() for parameter in model.state_head.parameters()),
                         "reference_normalization": configs[arm]["model"]["reference_normalization"],
                         "probability_mass_max_error": mass_error,
                         "losses": {key: float(value.detach()) for key, value in terms.items()},
                         "validation_CSV_interface_passed": True, "frozen_shared_statistics_unchanged": True,
                         "local_best57_replay_max_difference": local_replay_max,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "wall_seconds": time.perf_counter()-arm_started}
        del probability, result, terms, model, state
        gc.collect()
        torch.cuda.empty_cache()
    if any(arm["parameters"] != 7461060 for arm in metadata.values()):
        raise ValueError("paired architecture parameter count changed")
    report = {"status": "passed", "process_pid": os.getpid(), "configs": configs,
              "code_revision": os.environ.get("GEOPRR_CODE_REVISION"), "same_initial_parameters": True,
              "initialization": {"checkpoint": str(initial), "source_epoch": 57,
                                 "source_protocol": checkpoint["protocol"], "loaded_all_tensors": len(initial_state)},
              "reused_teacher_gate": str(teacher_path), "teacher_numerical_gate_passed": True,
              "original_train_images": 14494, "supplementary_images": 4096, "validation_images": 1506,
              **expected_exposure, "fixed_sample_ids": [sample["sample_id"] for sample in samples],
              "unique_RGB_images_forwarded": 2, "CNN_image_forwards": 4,
              "backward_calls": 1, "optimizer_steps": 0, "arms": metadata,
              "wall_seconds": time.perf_counter()-started,
              "scope": "two-image RGB runtime and full-best57 initialization; no model accuracy evaluation"}
    write_json(output, report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
