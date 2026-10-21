"""Diagnose one frozen triangle checkpoint with exact conditional summaries."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.diagnose_geoprr_v2_reference_relation import save_rows
from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset, reference_geometry_targets
from experiments.geoprr_v2_role_triangle import RoleTriangleReader, pair_log_factor
from experiments.geoprr_v2_triangle_diagnostic import summarize_triangle
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, make_loader, seed_everything
from experiments.train_geoprr_v2_reference_relation import validation_rows


FOCUS = {"sync_3172", "sync_13360", "sync_1860", "sync_2810", "sync_3663", "sync_10304", "sync_12540", "sync_9323"}


def teacher_star(head, features, center, spd):
    center, spd = center.float(), spd.float()
    rays = torch.linalg.solve(spd, head.directions.T.expand(len(center), -1, -1)).transpose(1, 2)
    rays = F.normalize(rays, dim=-1)
    divisor = torch.where(rays != 0, rays, torch.ones_like(rays))
    distance = torch.where(rays > 0, (1-center[:, None])/divisor, -center[:, None]/divisor)
    distance = torch.where(rays != 0, distance, torch.full_like(distance, float("inf")))
    points = center[:, None, None]+head.radii[None, None, :, None]*distance.amin(-1)[:, :, None, None]*rays[:, :, None]
    samples = F.grid_sample(features.float(), 2*points-1, mode="bilinear", padding_mode="border", align_corners=False)
    tokens = torch.einsum("bckr,rcd->bkd", samples, head.radial_embedding)/16
    return {**head.relation(tokens), "role_tokens": tokens}


def scalar_rows(name, summary, batch, gt_a=False):
    prefix = "GTA_" if gt_a else ""
    values = {key: summary[prefix+key].float().cpu().numpy() for key in ("reading", "invalid_mass", "outside_arc_mass")}
    values["reading"] = np.clip(values["reading"], 0, 1)
    mass_error = summary["mass_error"].float().cpu().numpy()
    rows = []
    for index, key in enumerate(batch["sample_id"]):
        prediction = float(values["reading"][index])
        invalid, outside = (float(values[k][index]) for k in ("invalid_mass", "outside_arc_mass"))
        valid = (not batch["input_failure"][index] and math.isfinite(prediction) and 0 <= prediction <= 1
                 and mass_error[index] <= 1e-4 and all(math.isfinite(x) and -1e-4 <= x <= 1+1e-4 for x in (invalid, outside)))
        target = float(batch["y"][index])
        rows.append({"sample_id": key, "group_id": batch["group_id"][index], "readout": name,
                     "target": target, "prediction": prediction if valid else None, "valid": bool(valid),
                     "input_valid": not batch["input_failure"][index], "absolute_error": abs(prediction-target) if valid else 1.,
                     "invalid_mass": invalid, "outside_arc_mass": outside,
                     "probability_mass_error": float(mass_error[index])})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any((args.output/name).exists() for name in ("batches", "predictions.csv.gz", "completed.json")):
        raise FileExistsError("diagnostic artifacts exist; inspect the job and preserve partial outputs")
    if compute_processes():
        raise RuntimeError("GPU is occupied; do not duplicate or overlap the diagnostic")
    started = time.perf_counter()
    reader = args.home/"runs/geoprr_v2_tail/iteration_v7_role_triangle/readers/full_s20262020"
    checkpoint = torch.load(reader/"best.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint["epoch"] != 64 or config["training"]["seed"] != 20262020 or config["model"]["association"] != "full":
        raise ValueError("requires the frozen full best64 seed20262020 checkpoint")
    reference = read_predictions(reader/"best_predictions.csv.gz")
    dataset = ReferenceRelationDataset(args.home/"runs/geoprr_v2/manifests/validation.jsonl",
        data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=False)
    if dataset.supervision_contract() != checkpoint["data_contract"]["validation"]:
        raise ValueError("original validation input/label contract differs")
    if [r["sample_id"] for r in dataset.rows] != [r["sample_id"] for r in reference]:
        raise ValueError("original1506 validation order differs")
    seed_everything(20262020)
    device = torch.device("cuda")
    model = RoleTriangleReader(**config["model"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("batches", "focus"):
        (args.output/name).mkdir()
    (args.output/"diagnostic.pid").write_text(str(os.getpid())+"\n")
    counts = {"CNN_image_forwards": 0, "backbone_calls": 0, "native_state_head_image_calls": 0,
              "teacher_sampling_relation_head_image_calls": 0, "triangle_summary_calls": 0,
              "reference_blocks": 0, "summary_image_equivalents": 0,
              "readout_module_calls": 0, "readout_module_image_equivalents": 0}

    def count_backbone(_module, inputs, _result):
        counts["backbone_calls"] += 1
        counts["CNN_image_forwards"] += len(inputs[0])

    def count_head(_module, inputs, _result):
        counts["native_state_head_image_calls"] += len(inputs[0])

    def count_readout(_module, inputs, _result):
        counts["readout_module_calls"] += 1
        counts["readout_module_image_equivalents"] += len(inputs[0])

    model.backbone.register_forward_hook(count_backbone)
    model.state_head.register_forward_hook(count_head)
    model.readout.register_forward_hook(count_readout)
    transport = model.readout
    reading_atoms = (transport.coordinates[transport.lower]*(1-transport.fraction)
                     +transport.coordinates[transport.upper]*transport.fraction)
    reading_atoms = reading_atoms.masked_fill(transport.invalid_atoms, float(transport.coordinates.mean())).reshape(256, 256)
    masks = {"invalid": transport.invalid_atoms.reshape(256, 256), "outside": transport.outside_atoms.reshape(256, 256)}
    np.savez_compressed(args.output/"readout_atoms.npz", reading=reading_atoms.cpu().numpy(),
                        invalid=masks["invalid"].cpu().numpy(), outside=masks["outside"].cpu().numpy())
    protocol = {"checkpoint_epoch": 64, "seed": 20262020, "population": 1506, "batch_size": 16,
                "model": config["model"], "code_revision": os.environ.get("GEOPRR_CODE_REVISION"),
                "focus_ids": sorted(FOCUS), "readouts_per_image": 8, "optimizer_updates": 0,
                "native_replay_tolerance": 1e-5, "probability_mass_tolerance": 1e-4,
                "GT_scope": "conditional diagnosis only; no GT inference input or new model accuracy claim",
                "factor_scope": "same best64 weights:full/Fsep/W0; Jproduct is read from existing cache",
                "reference_scope": "exact triangle conditionals normalized per A; no independent P/B replacement"}
    write_json(args.output/"protocol.json", protocol)
    loader = make_loader(dataset, batch_size=16, workers=8, shuffle=False, seed=20262020, device=device)
    rows, native_rows, position = [], [], 0
    replay_max, summary_replay_max, reference_residual, mass_max = 0., 0., 0., 0.
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for batch_index, raw in enumerate(loader):
                batch = device_batch(raw, device)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(batch["image"])
                native = validation_rows(output, batch)
                for actual, saved in zip(native, reference[position:position+len(native)], strict=True):
                    if (actual["sample_id"], actual["group_id"], actual["target"]) != (saved["sample_id"], saved["group_id"], float(saved["target"])):
                        raise ValueError("native replay ID/group/target differs")
                    if actual["valid"] and saved["valid"].lower() == "true":
                        replay_max = max(replay_max, abs(actual["prediction"]-float(saved["prediction"])),
                                         abs(actual["counterfactual_reading"]-float(saved["counterfactual_reading"])))
                if replay_max > 1e-5:
                    raise ValueError(f"native replay exceeds frozen tolerance: {replay_max}")
                native_rows.extend(native)
                with torch.autocast("cuda", enabled=False):
                    targets = reference_geometry_targets(batch["roi_to_canonical"], batch["y"], batch["span_radians"])
                    labels = targets["role_twohot"].float()
                    teacher = teacher_star(model.state_head, output["decoded_features"], targets["center"], targets["spd"])
                    counts["teacher_sampling_relation_head_image_calls"] += len(native)
                    p, b = (model.state_head.relation.projections[i](teacher["role_tokens"]) for i in (1, 2))
                    teacher_factor = pair_log_factor(p, b, model.state_head.Wpb)
                    states = {"native": (output["star_log_A"], output["log_P_rel"], output["log_B_rel"], output["log_F_abs"]),
                              "teacher_rectifier": (teacher["log_A"], teacher["log_P_rel"], teacher["log_B_rel"], teacher_factor)}
                    arrays = {"sample_id": np.asarray(batch["sample_id"]), "diagnostic_available": np.asarray([not x for x in batch["input_failure"]]),
                              "teacher_role_twohot": labels.cpu().numpy(),
                              "center_error_ROI_normalized": (output["calibration"]["center"]-targets["center"]).norm(dim=-1).float().cpu().numpy(),
                              "SPD_relative_Frobenius": ((output["calibration"]["spd"]-targets["spd"]).flatten(1).norm(dim=-1)
                                  /targets["spd"].flatten(1).norm(dim=-1)).float().cpu().numpy()}
                    batch_rows = []
                    for branch, (log_a, log_p, log_b, full_factor) in states.items():
                        factors = {"full": full_factor,
                                   "Fsep": full_factor.logsumexp(-1)[:, :, None]+full_factor.logsumexp(-2)[:, None, :],
                                   "W0": torch.full_like(full_factor, -2*math.log(256))}
                        for variant, factor in factors.items():
                            summary = summarize_triangle(log_a, log_p, log_b, factor, reading_atoms, labels,
                                                         tile=16, atom_masks=masks)
                            for key, value in summary["counts"].items():
                                counts[key] += value
                            mass_max = max(mass_max, float(summary["mass_error"].max()))
                            for key, value in summary.items():
                                if isinstance(value, torch.Tensor):
                                    arrays[f"{branch}__{variant}__{key}"] = value.float().cpu().numpy()
                            name = f"{branch}_{variant}"
                            batch_rows.extend(scalar_rows(name, summary, batch))
                            if variant == "full":
                                batch_rows.extend(scalar_rows(f"{branch}_GT_A", summary, batch, gt_a=True))
                            if branch == "native" and variant == "full":
                                summary_replay_max = max(summary_replay_max, float((summary["reading"]-output["reading"]).abs().max()))
                                reference_residual = max(reference_residual, float((summary["A"]-output["A"]).abs().max()))
                        for i, sample_id in enumerate(batch["sample_id"]):
                            if sample_id in FOCUS:
                                np.savez_compressed(args.output/"focus"/f"{sample_id}_{branch}.npz",
                                    star_log_A=log_a[i].cpu().numpy(), star_log_P_rel=log_p[i].cpu().numpy(),
                                    star_log_B_rel=log_b[i].cpu().numpy(), log_F_abs=full_factor[i].cpu().numpy(),
                                    role_twohot=labels[i].cpu().numpy())
                if summary_replay_max > 1e-5 or reference_residual > 1e-5:
                    raise ValueError("exact triangle native conditional replay failed")
                rows.extend(batch_rows)
                save_rows(args.output/"batches"/f"batch_{batch_index:04d}_predictions.csv.gz", batch_rows)
                np.savez_compressed(args.output/"batches"/f"batch_{batch_index:04d}.npz", **arrays)
                position += len(native)
                write_json(args.output/"progress.json", {"status": "running", "images_saved": position, **counts,
                    "replay_max": replay_max, "summary_replay_max": summary_replay_max,
                    "reference_residual": reference_residual, "probability_mass_max_error": mass_max})
                print(json.dumps({"batch": batch_index+1, "images_saved": position, "replay_max": replay_max,
                                  "summary_replay_max": summary_replay_max}), flush=True)
        save_rows(args.output/"predictions.csv.gz", rows)
        save_rows(args.output/"native_replay.csv.gz", native_rows)
        torch.cuda.synchronize()
        complete = {"status": "complete", "checkpoint_epoch": 64, "images": position, **counts,
                    "optimizer_updates": 0, "backward_calls": 0, "readouts_per_image": len({r["readout"] for r in rows}),
                    "saved_readout_rows": len(rows), "summary_scalar_reading_computations": 2*counts["summary_image_equivalents"],
                    "replay_max_difference": replay_max, "summary_reading_replay_max": summary_replay_max,
                    "reference_posterior_replay_max": reference_residual, "probability_mass_max_error": mass_max,
                    "input_failures": sum(not row["input_valid"] for row in native_rows),
                    "invalid_diagnostic_rows": sum(not row["valid"] for row in rows),
                    "process_wall_seconds": time.perf_counter()-started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "scope": "frozen best64 component diagnosis; GT and factor deletions are not new trained models"}
        if position != 1506 or len(rows) != 8*1506:
            raise ValueError("complete original1506 and all8 readouts are required")
        write_json(args.output/"completed.json", complete)
        print(json.dumps(complete), flush=True)
    except Exception as error:
        write_json(args.output/"failed.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "images_saved": position, **counts, "process_wall_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
