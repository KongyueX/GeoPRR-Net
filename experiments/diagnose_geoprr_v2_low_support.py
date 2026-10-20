"""Diagnose frozen U/Q/M components for three fixed source-validation examples."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
import torch


IDS = ("sync_3663", "sync_12540", "sync_10")


def scalar(value):
    return float(value)


def region_stats(values, region):
    selected = values[region]
    return {"pixels": int(selected.numel()), "mean": scalar(selected.mean()),
            "min": scalar(selected.min()), "max": scalar(selected.max()),
            "sum": scalar(selected.sum())}


def conditional(field, q, m, coordinates, target, threshold, gt_mask, direct):
    weights = q.double() * m.double()
    mass = weights.sum()
    distribution = (field.double() * weights).sum((1, 2)) / mass
    reading = (distribution * coordinates.double()).sum()
    peak = int(distribution.argmax())
    return {"diagnostic_only": True, "support_mass": scalar(mass),
            "field_valid_at_original_threshold": bool(mass >= threshold),
            "forced_conditional_reading": scalar(reading),
            "forced_absolute_error_percent_fs": 100 * abs(scalar(reading) - target),
            "thresholded_reading_with_original_direct_reference": scalar(reading) if mass >= threshold else direct,
            "qm_fraction_inside_gt_support": scalar((weights * gt_mask).sum() / mass),
            "distribution_sum": scalar(distribution.sum()),
            "distribution_peak_bin": peak,
            "distribution_peak_progress": scalar(coordinates[peak]),
            "distribution_entropy_nats": scalar(-(distribution * distribution.clamp_min(1e-30).log()).sum()),
            "forced_conditional_distribution": distribution.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    sys.path.insert(0, str(args.root / "code"))
    from experiments.geoprr_v2_data import GeoPRRV2Dataset, read_rows
    from experiments.geoprr_v2_model import GeoPRRV2, two_hot_targets
    from experiments.train_geoprr_v2 import PROTOCOL

    base = args.root / "runs/geoprr_v2_tail/iteration_v3_compositional"
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "fixed_three_source_validation_component_diagnosis_v1",
              "diagnostic_only": True, "sample_ids": list(IDS), "device": args.device,
              "inference_dtype": "float32; manual conditional accumulation float64",
              "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "environment": {key: os.environ.get(key) for key in
                              ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
              "pid": os.getpid(), "total_model_forward_calls": 0,
              "notes": ["No threshold change, training, adaptation, test-set evaluation, or main-score recomputation.",
                        "Forced conditional distribution is manually normalized even below the original threshold.",
                        "Oracle component substitutions use ground-truth geometry and are diagnostics, not method results.",
                        "U_GT is continuous progress converted to the existing two-adjacent-bin soft target; outside GT support its values have no supervised meaning.",
                        "CPU float32 single-image results may differ from the existing GPU AMP batch-16 CSV."],
              "models": {}}
    maps = {}
    for name in ("supervised", "consistency"):
        checkpoint_path = base / "readers" / f"{name}_s20262020" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["protocol"] == PROTOCOL and not checkpoint.get("smoke_only", False)
        config = checkpoint["config"]
        manifest = Path(checkpoint["manifests"]["validation"])
        rows = read_rows(manifest)
        assert len(rows) == 1506
        selected = [next(i for i, row in enumerate(rows) if row["sample_id"] == sample_id) for sample_id in IDS]
        dataset = GeoPRRV2Dataset(manifest, dataset_root=args.root / "datasets",
                                 **config["data"], require_geometry=True, augment=False)
        gpu_dir = base / "validation_evaluation" / f"{name}_native"
        with (gpu_dir / "predictions.csv").open() as handle:
            gpu_rows = {row["sample_id"]: row for row in csv.DictReader(handle) if row["sample_id"] in IDS}
        summary = json.loads((gpu_dir / "summary.json").read_text())
        assert summary["manifest"] == str(manifest) and summary["timing"]["amp"]
        model = GeoPRRV2(**dict(config["model"], pretrained=False)).to(torch.device("cpu"))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        model_report = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
                        "manifest": str(manifest), "declared_validation_population": len(rows),
                        "source_csv": str(gpu_dir / "predictions.csv"),
                        "source_csv_timing": summary["timing"], "support_threshold": model.support_threshold,
                        "samples": []}
        with torch.inference_mode():
            for index in selected:
                sample = dataset[index]
                image = sample["image"].unsqueeze(0).to(torch.device("cpu"))
                forward_start = time.perf_counter()
                output = model(image)
                forward_seconds = time.perf_counter() - forward_start
                report["total_model_forward_calls"] += 1
                target = scalar(sample["y"])
                q, m, field = (output[key][0] for key in ("point_probs", "support_probs", "field_probs"))
                qg, mg = (sample[key] for key in ("point_target", "support_target"))
                ug = two_hot_targets(sample["field_target"], model.bins).permute(2, 0, 1)
                coordinates = model.bin_coordinates
                u = (field * coordinates[:, None, None]).sum(0)
                q2, qg2, m2, mg2 = q[0], qg[0], m[0], mg[0]
                height, width = q2.shape
                yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
                gt_point = (sample["virtual_point_xy"] + 0.5) * height / config["data"]["image_size"] - 0.5
                peak = int(q2.argmax()); peak_xy = torch.tensor([peak % width, peak // width]).float()
                mean_xy = torch.stack(((q2 * xx).sum(), (q2 * yy).sum()))
                dist = ((xx - gt_point[0]) ** 2 + (yy - gt_point[1]) ** 2).sqrt()
                inside = mg2.bool()
                thresholded_m = m2 >= 0.5
                direct = scalar(output["appearance_reading"][0])
                cpu = {"prediction": scalar(output["reading"][0]), "direct": direct,
                       "absolute_error_percent_fs": 100 * abs(scalar(output["reading"][0]) - target),
                       "support_mass": scalar(output["support_mass"][0]),
                       "field_valid": bool(output["field_valid"][0]),
                       "legacy_field_reading": scalar(output["field_reading"][0]),
                       "legacy_field_distribution_sum": scalar(output["field_distribution"][0].sum())}
                gpu = gpu_rows[sample["sample_id"]]
                gpu_values = {"prediction": float(gpu["prediction"]), "support_mass": float(gpu["support_mass"]),
                              "field_valid": gpu["field_valid"], "absolute_error_percent_fs": 100 * float(gpu["absolute_error"])}
                oracle = {}
                for label, uf, qf, mf in (("predicted_U_Q_M", field, q, m),
                                         ("replace_Q_with_GT", field, qg, m),
                                         ("replace_M_with_GT", field, q, mg),
                                         ("replace_U_with_GT", ug, q, m),
                                         ("replace_Q_M_with_GT", field, qg, mg),
                                         ("all_GT", ug, qg, mg)):
                    oracle[label] = conditional(uf, qf, mf, coordinates, target,
                                                model.support_threshold, mg, direct)
                record = {"sample_id": sample["sample_id"], "source_validation_index": index,
                          "target": target, "forward_seconds": forward_seconds,
                          "native_cpu_float32": cpu, "existing_gpu_amp_csv": gpu_values,
                          "cpu_minus_gpu": {key: cpu[key] - gpu_values[key] for key in ("prediction", "support_mass")},
                          "cross_support_mass": {
                              "Qpred_Mpred": scalar((q.double() * m.double()).sum()),
                              "Qpred_Mgt": scalar((q.double() * mg.double()).sum()),
                              "Qgt_Mpred": scalar((qg.double() * m.double()).sum()),
                              "Qgt_Mgt": scalar((qg.double() * mg.double()).sum())},
                          "Q": {"gt_virtual_point_grid_xy": gt_point.tolist(),
                                "pred_peak_grid_xy": peak_xy.tolist(), "pred_mean_grid_xy": mean_xy.tolist(),
                                "peak_distance_to_gt_grid_px": scalar(torch.linalg.vector_norm(peak_xy - gt_point)),
                                "mean_distance_to_gt_grid_px": scalar(torch.linalg.vector_norm(mean_xy - gt_point)),
                                "max_probability": scalar(q.max()),
                                "entropy_nats": scalar(-(q * q.clamp_min(1e-30).log()).sum()),
                                "gt_peak_grid_xy": [int(qg2.argmax()) % width, int(qg2.argmax()) // width],
                                "local_mass_about_gt": {f"r{r}": scalar(q2[dist <= r].sum()) for r in (2, 4, 8)}},
                          "M": {"pred_inside_gt_support": region_stats(m2, inside),
                                "pred_outside_gt_support": region_stats(m2, ~inside),
                                "pred_gt_point_neighborhood": {f"r{r}": region_stats(m2, dist <= r) for r in (2, 4, 8)},
                                "soft_area_sum": scalar(m.sum()),
                                "gt_area_pixels": int(inside.sum()),
                                "pred_area_at_0p5_pixels": int(thresholded_m.sum()),
                                "thresholded_0p5_IoU": scalar((thresholded_m & inside).sum() / (thresholded_m | inside).sum()),
                                "soft_mass_fraction_inside_gt_support": scalar(m2[inside].sum() / m2.sum())},
                          "conditional_and_oracle_diagnostics": oracle,
                          "gt_geometry_audit": sample["geometry_audit"]}
                model_report["samples"].append(record)
                prefix = f"{name}__{sample['sample_id']}"
                for label, value in (("q_pred", q2), ("m_pred", m2), ("u_mean_pred", u),
                                     ("q_gt", qg2), ("m_gt", mg2), ("u_gt", sample["field_target"])):
                    maps[f"{prefix}__{label}"] = value.numpy()
                print(json.dumps({"model": name, "sample_id": sample["sample_id"],
                                  "forward_seconds": forward_seconds, "cpu": cpu,
                                  "cross_support_mass": record["cross_support_mass"]}), flush=True)
        report["models"][name] = model_report
        del model, checkpoint, output, field, ug, image, dataset
        gc.collect()
    report["resources"] = {"wall_seconds": time.perf_counter() - started,
                           "forward_seconds": sum(row["forward_seconds"] for model in report["models"].values() for row in model["samples"]),
                           "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                           "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                           "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "cuda_initialized": torch.cuda.is_initialized()}
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "component_maps.npz", **maps)
    print(json.dumps({"report": str(args.output / "diagnosis.json"), "resources": report["resources"]}), flush=True)


if __name__ == "__main__":
    main()
