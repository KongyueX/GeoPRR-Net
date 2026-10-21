"""Inspect one frozen reference-relation checkpoint without optimization."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.geoprr_v2_reference_diagnostics import diagnose_state, role_interventions
from experiments.geoprr_v2_reference_relation import (
    ReferenceRelationDataset, ReferenceRelationReader, reference_geometry_targets,
)
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, make_loader, seed_everything
from experiments.train_geoprr_v2_reference_relation import validation_rows


FOCUS = {"sync_3172", "sync_12540", "sync_338", "sync_1999"}


def resample_state(head, features, center, spd):
    """Reuse frozen image features with the specified angular sampling coordinates."""
    center, spd = center.float(), spd.float()
    rays = torch.linalg.solve(spd, head.directions.T.expand(len(center), -1, -1)).transpose(1, 2)
    rays = F.normalize(rays, dim=-1)
    divisor = torch.where(rays != 0, rays, torch.ones_like(rays))
    distances = torch.where(rays > 0, (1-center[:, None])/divisor, -center[:, None]/divisor)
    distances = torch.where(rays != 0, distances, torch.full_like(distances, float("inf")))
    points = center[:, None, None]+head.radii[None, None, :, None]*distances.amin(-1)[:, :, None, None]*rays[:, :, None]
    sampled = F.grid_sample(features.float(), 2*points-1, mode="bilinear", padding_mode="border", align_corners=False)
    tokens = torch.einsum("bckr,rcd->bkd", sampled, head.radial_embedding)/16
    return head.relation(tokens)


def score_rows(name, result, batch):
    values = {key: result[key].float().cpu().numpy() for key in
              ("reading", "distribution", "invalid_mass", "outside_arc_mass")}
    rows = []
    for i, sample_id in enumerate(batch["sample_id"]):
        probability = values["distribution"][i]
        prediction = float(values["reading"][i])
        valid = (not batch["input_failure"][i] and np.isfinite(probability).all()
                 and (probability >= 0).all() and abs(float(probability.sum())-1) <= 1e-4
                 and np.isfinite(prediction) and 0 <= prediction <= 1)
        rows.append({"sample_id": sample_id, "group_id": batch["group_id"][i], "readout": name,
                     "target": float(batch["y"][i]), "prediction": prediction if valid else None,
                     "valid": bool(valid), "input_valid": not batch["input_failure"][i],
                     "absolute_error": abs(prediction-float(batch["y"][i])) if valid else 1.,
                     "invalid_mass": float(values["invalid_mass"][i]),
                     "outside_arc_mass": float(values["outside_arc_mass"][i])})
    return rows


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any((args.output/name).exists() for name in ("batches", "predictions.csv.gz", "completed.json")):
        raise FileExistsError("diagnostic artifacts already exist; preserve them and inspect the running process")
    if compute_processes():
        raise RuntimeError("GPU occupied; do not overlap the frozen checkpoint diagnostic")
    started = time.perf_counter()
    reader_path = args.home/"runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020"
    reference = read_predictions(reader_path/"best_predictions.csv.gz")
    checkpoint = torch.load(reader_path/"best.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 57 or checkpoint["config"]["training"]["seed"] != 20262020:
        raise ValueError("requires the frozen best57 seed20262020 checkpoint")
    config = checkpoint["config"]
    dataset = ReferenceRelationDataset(args.home/"runs/geoprr_v2/manifests/validation.jsonl",
                                       data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=False)
    if dataset.supervision_contract() != checkpoint["data_contract"]["validation"]:
        raise ValueError("source validation inputs or labels differ from the trained checkpoint")
    if [row["sample_id"] for row in dataset.rows] != [row["sample_id"] for row in reference]:
        raise ValueError("frozen validation order changed")
    seed_everything(20262020)
    device = torch.device("cuda")
    model = ReferenceRelationReader(**config["model"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    readout_counts = {"calls": 0, "image_equivalents": 0}

    def count_readout(_module, inputs, _output):
        readout_counts["calls"] += 1
        readout_counts["image_equivalents"] += len(inputs[0])

    model.readout.register_forward_hook(count_readout)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"batches").mkdir()
    (args.output/"focus").mkdir()
    (args.output/"diagnostic.pid").write_text(str(os.getpid())+"\n")
    write_json(args.output/"protocol.json", {"checkpoint_epoch": 57, "seed": 20262020,
        "population": 1506, "batch_size": 16, "optimizer_updates": 0,
        "teacher_scope": "conditional coordinate/role intervention, not deployable RGB performance",
        "focus_ids": sorted(FOCUS), "code_revision": os.environ.get("GEOPRR_CODE_REVISION"),
        "teacher_role_subsets": ["A", "P", "B", "PB", "AP", "AB", "APB"]})
    loader = make_loader(dataset, batch_size=16, workers=8, shuffle=False, seed=20262020, device=device)
    rows, native_rows, position = [], [], 0
    replay_max, identities_max, mass_max = 0., 0., 0.
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = device_batch(raw_batch, device)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(batch["image"])
            native = validation_rows(output, batch)
            expected = reference[position:position+len(native)]
            for actual, saved in zip(native, expected, strict=True):
                if (actual["sample_id"], actual["group_id"], actual["target"], actual["valid"]) != (
                    saved["sample_id"], saved["group_id"], float(saved["target"]), saved["valid"].lower() == "true"):
                    raise ValueError("native replay population, target or validity differs")
                if actual["valid"]:
                    replay_max = max(replay_max, abs(actual["prediction"]-float(saved["prediction"])),
                                     abs(actual["counterfactual_reading"]-float(saved["counterfactual_reading"])))
            if replay_max > 1e-5:
                raise ValueError(f"native replay exceeds frozen tolerance: {replay_max}")
            native_rows.extend(native)
            targets = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"])
            labels = targets["role_twohot"].float()
            teacher = resample_state(model.state_head, output["decoded_features"], targets["center"], targets["spd"])
            branches = {"native": output, "teacher_rectifier": teacher}
            arrays = {"sample_id": np.asarray(batch["sample_id"]),
                      "diagnostic_available": np.asarray([not x for x in batch["input_failure"]]),
                      "teacher_center": targets["center"].cpu().numpy(), "teacher_spd": targets["spd"].cpu().numpy(),
                      "predicted_center": output["calibration"]["center"].cpu().numpy(),
                      "predicted_spd": output["calibration"]["spd"].cpu().numpy(),
                      "teacher_role_twohot": labels.cpu().numpy()}
            for branch_name, state in branches.items():
                stats = diagnose_state(state, model.readout, labels)
                for key, value in stats.items():
                    arrays[f"{branch_name}__{key}"] = value.float().cpu().numpy()
                    if key.endswith("residual"):
                        identities_max = max(identities_max, float(value.abs().max()))
                for coupling, joint in (("joint", state["J"]), ("product", state["product"])):
                    result = model.readout(joint)
                    rows.extend(score_rows(f"{branch_name}_{coupling}", result, batch))
                    mass_max = max(mass_max, float((result["distribution"].sum(-1)-1).abs().max()))
                for i, sample_id in enumerate(batch["sample_id"]):
                    if sample_id in FOCUS:
                        np.savez_compressed(args.output/"focus"/f"{sample_id}_{branch_name}.npz",
                                            A=state["A"][i].cpu().numpy(), P_rel=state["P_rel"][i].cpu().numpy(),
                                            B_rel=state["B_rel"][i].cpu().numpy())
            for name, result in role_interventions(teacher, model.readout, labels).items():
                rows.extend(score_rows(f"teacher_{name}", result, batch))
                arrays[f"teacher_{name}__progress"] = result["distribution"].cpu().numpy()
                mass_max = max(mass_max, float((result["distribution"].sum(-1)-1).abs().max()))
            if identities_max > 1e-5 or mass_max > 1e-4:
                raise ValueError("conditional contribution identity or probability mass check failed")
            np.savez_compressed(args.output/"batches"/f"batch_{batch_index:04d}.npz", **arrays)
            position += len(native)
            print(json.dumps({"event": "diagnostic_batch", "batch": batch_index+1, "images": position,
                              "replay_max": replay_max, "identity_max": identities_max}), flush=True)
    save_rows(args.output/"native_replay.csv.gz", native_rows)
    save_rows(args.output/"predictions.csv.gz", rows)
    torch.cuda.synchronize(device)
    write_json(args.output/"completed.json", {"status": "complete", "images": position,
        "CNN_image_forwards": position, "native_head_image_calls": position,
        "teacher_rectifier_head_image_calls": position, "optimizer_updates": 0, "backward_calls": 0,
        "readout_calls": readout_counts["calls"], "readout_image_equivalents": readout_counts["image_equivalents"],
        "readouts_per_image": len({row["readout"] for row in rows}), "replay_max_difference": replay_max,
        "identity_max_residual": identities_max, "probability_mass_max_error": mass_max,
        "input_failures": sum(not row["input_valid"] for row in native_rows),
        "process_wall_seconds": time.perf_counter()-started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "scope": "frozen checkpoint diagnostic; teacher interventions are not model performance"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
