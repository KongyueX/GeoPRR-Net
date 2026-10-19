"""Diagnose frozen MobileNetV4 readout failures on source validation only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.evaluate_geoprr_v2 import FullPopulationDataset
from experiments.geoprr_v2_data import GeoPRRV2Dataset, read_rows
from experiments.geoprr_v2_model import GeoPRRV2, masked_field_readout
from experiments.train_geoprr_v2 import (
    DEFAULT_DATA_ROOT, PROTOCOL, device_batch, make_loader, prediction_rows,
    source_split_check, summarize_rows, write_json,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-30).log()).flatten(1).sum(1)


def compose(u: np.ndarray, q: np.ndarray, m: np.ndarray, threshold: float) -> dict:
    mass = float((q * m).sum())
    value = float((u * q * m).sum() / mass) if mass > 0 else None
    return {"conditional_reading": value, "support_mass": mass,
            "field_valid": mass >= threshold}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("smoke_only", False):
        raise ValueError("diagnosis requires a formal source reader")
    config = checkpoint["config"]
    rows = read_rows(args.manifest)
    source_split_check(read_rows(checkpoint["manifests"]["train"]), rows)
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in read_rows(checkpoint["manifests"]["validation"])]:
        raise ValueError("diagnosis must use the frozen reader's original source validation population")
    device = torch.device(args.device)
    model = GeoPRRV2(**dict(config["model"], pretrained=False)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = GeoPRRV2Dataset(args.manifest, dataset_root=args.data_root,
                             **config["data"], require_geometry=False, augment=False)
    population = FullPopulationDataset(dataset)
    loader = make_loader(population, batch_size=args.batch_size, workers=args.workers,
                         shuffle=False, seed=config["training"]["seed"], device=device)
    names = (["native", "distribution_map", "distribution_discrete_median"]
             + [f"distribution_peak_r{r}" for r in (2, 4, 8)] + [f"q_peak_r{r}" for r in (2, 4, 8)])
    predictions = {name: [] for name in names}
    statistics, distributions = [], []
    with torch.inference_mode():
        for raw in loader:
            batch = device_batch(raw, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(batch["image"])
            distribution = output["field_distribution"]
            coordinates = model.bin_coordinates
            bins = distribution.shape[1]
            peak = distribution.argmax(1)
            candidates = {"native": output}
            median_bin = (distribution.cumsum(1) >= 0.5).long().argmax(1)
            for name, indices in (("distribution_map", peak), ("distribution_discrete_median", median_bin)):
                reading = torch.where(output["field_valid"], coordinates[indices], output["appearance_reading"])
                candidates[name] = dict(output, reading=reading)
            for radius in (2, 4, 8):
                mask = (torch.arange(bins, device=device)[None] - peak[:, None]).abs() <= radius
                local = distribution * mask
                reading = (local * coordinates).sum(1) / local.sum(1).clamp_min(1e-30)
                candidates[f"distribution_peak_r{radius}"] = dict(output, reading=torch.where(output["field_valid"], reading, output["appearance_reading"]))
            q = output["point_probs"]
            height, width = q.shape[-2:]
            peak_q = q.flatten(1).argmax(1)
            yy, xx = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
            distance = ((xx[None] - (peak_q % width)[:, None, None]) ** 2
                        + (yy[None] - (peak_q // width)[:, None, None]) ** 2)
            for radius in (2, 4, 8):
                localized = q * (distance <= radius ** 2)[:, None]
                localized /= localized.sum((1, 2, 3), keepdim=True).clamp_min(1e-30)
                decoded = masked_field_readout(output["field_probs"], localized,
                                               output["support_probs"], model.support_threshold,
                                               output["appearance_probs"])
                candidates[f"q_peak_r{radius}"] = dict(output, **decoded)
            for name, candidate in candidates.items():
                records = prediction_rows(candidate, batch)
                for i, record in enumerate(records):
                    if not bool(raw["input_valid"][i]):
                        record.update(prediction=None, absolute_error=1.0, valid=False,
                                      field_valid=False, failure=raw["input_failure"][i])
                predictions[name].extend(records)
            q_entropy = entropy(q)
            d_entropy = entropy(distribution[:, :, None, None])
            q_x = (q[:, 0] * xx).sum((1, 2))
            q_y = (q[:, 0] * yy).sum((1, 2))
            for i, sample_id in enumerate(raw["sample_id"]):
                statistics.append({"sample_id": sample_id, "q_entropy_nats": float(q_entropy[i]),
                                   "q_max_probability": float(q[i].max()),
                                   "q_peak_x": int(peak_q[i] % width), "q_peak_y": int(peak_q[i] // width),
                                   "q_mean_x": float(q_x[i]), "q_mean_y": float(q_y[i]),
                                   "distribution_entropy_nats": float(d_entropy[i]),
                                   "distribution_peak_bin": int(peak[i]),
                                   "distribution_max_probability": float(distribution[i].max())})
            distributions.append(distribution.cpu().numpy())
    if len(predictions["native"]) != len(rows):
        raise RuntimeError("source validation denominator changed")
    args.output.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(predictions["native"]):
        record.update(statistics[i])
        for name in names[1:]:
            record[f"{name}_prediction"] = predictions[name][i]["prediction"]
            record[f"{name}_absolute_error"] = predictions[name][i]["absolute_error"]
    write_csv(args.output / "predictions.csv", predictions["native"])
    np.savez_compressed(args.output / "field_distributions.npz",
                        sample_ids=np.asarray([r["sample_id"] for r in rows]),
                        probabilities=np.concatenate(distributions),
                        bin_coordinates=coordinates.cpu().numpy())

    # Annotation replacement is a diagnostic counterfactual, never inference.
    geometry = GeoPRRV2Dataset(args.manifest, dataset_root=args.data_root,
                              **config["data"], require_geometry=True, augment=False)
    ranked = sorted(range(len(rows)), key=lambda i: predictions["native"][i]["absolute_error"], reverse=True)[:args.top]
    counterfactuals = []
    with torch.inference_mode():
        for index in ranked:
            record = {"sample_id": rows[index]["sample_id"], "target": predictions["native"][index]["target"],
                      "native_prediction": predictions["native"][index]["prediction"],
                      "native_error": predictions["native"][index]["absolute_error"]}
            try:
                gt = geometry[index]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    output = model(gt["image"][None].to(device))
                u = (output["field_probs"][0] * coordinates[:, None, None]).sum(0).cpu().numpy()
                q = output["point_probs"][0, 0].cpu().numpy()
                m = output["support_probs"][0, 0].cpu().numpy()
                ug, qg, mg = gt["field_target"].numpy(), gt["point_target"][0].numpy(), gt["support_target"][0].numpy()
                replacements = {"pred_u_gt_q_gt_m": (u, qg, mg),
                                "gt_u_pred_q_gt_m": (ug, q, mg),
                                "pred_u_pred_q_gt_m": (u, q, mg),
                                "all_gt": (ug, qg, mg),
                                "pred_u_gt_q_pred_m": (u, qg, m)}
                record["counterfactuals"] = {}
                for name, values in replacements.items():
                    result = compose(*values, model.support_threshold)
                    result["conditional_absolute_error"] = abs(result["conditional_reading"] - record["target"]) if result["conditional_reading"] is not None else None
                    record["counterfactuals"][name] = result
                py, px = np.unravel_index(q.argmax(), q.shape)
                gy, gx = np.unravel_index(qg.argmax(), qg.shape)
                record["pred_gt_q_peak_distance_output_pixels"] = float(np.hypot(px - gx, py - gy))
                record["q_probability_within_4px_gt_peak"] = float(q[(xx.cpu().numpy() - gx) ** 2 + (yy.cpu().numpy() - gy) ** 2 <= 16].sum())
                record["geometry_audit"] = gt["geometry_audit"]
            except (OSError, ValueError, KeyError) as exc:
                record["geometry_diagnostic_failure"] = f"{type(exc).__name__}: {exc}"
            counterfactuals.append(record)
    write_json(args.output / "counterfactuals.json", counterfactuals)
    summary = {"protocol": "geoprr_v2_source_validation_tail_diagnosis_v1",
               "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
               "seed": config["training"]["seed"], "manifest": str(args.manifest),
               "declared_samples": len(rows), "model_inputs": "RGB ROI only",
               "candidate_scope": "source validation diagnosis; radii not selected using official test",
               "counterfactual_scope": "annotation-assisted top-error diagnosis, not deployable performance",
               "summaries": {name: summarize_rows(records) for name, records in predictions.items()},
               "top_error_sample_ids": [rows[i]["sample_id"] for i in ranked]}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
