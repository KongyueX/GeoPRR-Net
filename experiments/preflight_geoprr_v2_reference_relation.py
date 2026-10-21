"""Check frozen source contracts and the RGB-only paired-control runtime."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from experiments.geoprr_v2_data import normalized_target, read_rows
from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationDataset, ReferenceRelationReader, reference_relation_losses,
)
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, seed_everything, source_split_check
from experiments.train_geoprr_v2_reference_relation import validation_rows


def roster(rows):
    return {row["sample_id"]: (row["group_id"], float(torch.tensor(normalized_target(row), dtype=torch.float32)))
            for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    started = time.perf_counter()
    home = args.home
    output = home / "diagnostics/reference_relation_preflight_v1_20261021"
    if (output / "runtime_preflight.json").exists():
        raise FileExistsError("runtime preflight is already complete")
    if compute_processes():
        raise RuntimeError("GPU is occupied; do not overlap the runtime check")
    repo = Path(__file__).resolve().parents[1]
    configs = {arm: json.loads((repo / f"experiments/configs/geoprr_v2_reference_relation_{arm}.json").read_text())
               for arm in ("joint", "product")}
    source = home / "rendering/compositional_source_v1/manifests"
    frozen = home / "runs/geoprr_v2/manifests"
    train = ReferenceRelationDataset(source / "train.jsonl", data_root=DEFAULT_DATA_ROOT,
                                    data_config=configs["joint"]["data"], augment=True,
                                    pair_manifest=source / "train_edges.jsonl")
    validation = ReferenceRelationDataset(frozen / "validation.jsonl", data_root=DEFAULT_DATA_ROOT,
                                         data_config=configs["joint"]["data"], augment=False)
    source_split_check(train.rows, validation.rows)
    original = roster(read_rows(frozen / "train.jsonl"))
    actual = roster(train.rows)
    extra = {f"geoprr_compositional_v1_g{group:04d}_s{state:02d}" for group in range(1024) for state in range(4)}
    if len(original) != 14494 or set(actual) != set(original) | extra or any(actual[key] != value for key, value in original.items()):
        raise ValueError("original14494 or supplementary4096 roster/labels changed")
    references = read_predictions(home / "diagnostics/gt_atlas_full1506_v1/predictions.csv")
    expected = {row["sample_id"]: (row["group_id"], float(row["target"])) for row in references}
    if roster(validation.rows) != expected:
        raise ValueError("original1506 validation roster/targets/groups changed")
    if (len(train.rows), len(train), len(train.edges), len(validation)) != (18590, 22686, 8192, 1506):
        raise ValueError("source exposure counts changed")
    seed_everything(20262020)
    index = next(i for i, (_, recipient) in enumerate(train.items) if recipient is not None)
    batch = device_batch(default_collate([train[index]]), torch.device("cuda"))
    images = torch.cat((batch["image"], batch["pair_image"]))
    supervision = {"image": images, "y": torch.cat((batch["y"], batch["pair_y"]))}
    for key in ("roi_to_canonical", "field_target", "field_mask", "span_radians"):
        supervision[key] = torch.cat((batch[key], batch[f"pair_{key}"]))
    csv_batch = {"y": supervision["y"],
                 **{key: batch[key]+batch[f"pair_{key}"] for key in ("sample_id", "group_id", "input_failure")}}
    # Read one original validation item to check its target interface, without another CNN pass.
    val_item = validation[0]
    if val_item["input_failure"]:
        raise ValueError(val_item["input_failure"])
    initial = home / "runs/geoprr_v2_tail/iteration_v3_compositional/readers/supervised_s20262020/best.pt"
    metadata, initial_state, same = {}, None, True
    for arm in ("joint", "product"):
        seed_everything(20262020)
        model = ReferenceRelationReader(**configs[arm]["model"])
        initialization = model.initialize_shared(initial)
        if initial_state is None:
            initial_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            same = set(initial_state) == set(model.state_dict()) and all(
                torch.equal(initial_state[key], value) for key, value in model.state_dict().items())
            if not same:
                raise ValueError("joint/product initial parameters or buffers differ")
        model = model.cuda()
        model.set_shared_trainable(False)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            result = model(images)
        terms = reference_relation_losses(result, supervision)
        if not all(torch.isfinite(value).all() for value in terms.values()):
            raise FloatingPointError("runtime preflight loss is nonfinite")
        probability = result["distribution"]
        mass_error = float((probability.sum(-1)-1).abs().max().detach())
        if not bool(torch.isfinite(probability).all()) or bool((probability < 0).any()) or mass_error > 1e-4:
            raise ValueError("runtime readout loses probability mass")
        csv_rows = validation_rows(result, csv_batch)
        if len(csv_rows) != 2 or not all(row["valid"] for row in csv_rows):
            raise ValueError("new validation CSV interface rejects valid runtime predictions")
        if arm == "joint":
            terms["total"].backward()
            gradients = [parameter.grad for parameter in model.state_head.parameters()]
            if any(value is None or not torch.isfinite(value).all() for value in gradients):
                raise FloatingPointError("runtime head gradient is absent or nonfinite")
            if any(parameter.grad is not None for parameter in model.shared_parameters()):
                raise ValueError("frozen shared module received a gradient")
        metadata[arm] = {"parameters": sum(parameter.numel() for parameter in model.parameters()),
                         "head_parameters": sum(parameter.numel() for parameter in model.state_head.parameters()),
                         "initialization": initialization, "probability_mass_max_error": mass_error,
                         "losses": {key: float(value.detach()) for key, value in terms.items()},
                         "validation_CSV_interface_passed": True,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "reference_max": result["reference_max"].detach().cpu().tolist(),
                         "joint_product_state_l1": result["joint_vs_product_state_l1"].detach().cpu().tolist()}
        del result, terms, model
        gc.collect()
        torch.cuda.empty_cache()
    if metadata["joint"]["parameters"] != metadata["product"]["parameters"]:
        raise ValueError("paired architectures have different parameter counts")
    report = {"status": "passed", "process_pid": os.getpid(), "configs": configs,
              "code_revision": os.environ.get("GEOPRR_CODE_REVISION"), "same_initial_parameters": same,
              "original_train_images": 14494, "supplementary_images": 4096, "validation_images": 1506,
              "donor_items": 22686, "recipient_items": 8192,
              "unique_RGB_images_forwarded": 2, "CNN_image_forwards": 4,
              "original_validation_images_read_without_forward": 1,
              "backward_calls": 1, "optimizer_steps": 0, "arms": metadata,
              "wall_seconds": time.perf_counter()-started,
              "scope": "RGB runtime, initialization and label-interface preflight; no model accuracy evaluation"}
    write_json(output / "runtime_preflight.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
