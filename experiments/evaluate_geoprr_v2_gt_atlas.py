"""Evaluate fixed K65 GT calibration charts with one frozen source-validation Q."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


K = 65
POPULATION = 1506


def calibration_charts(h: np.ndarray, span: float, image_size: int, output_size: int) -> dict:
    """Use ROI pixels, ordered common-center rays and midpoint endpoint clipping."""
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or not 0 < abs(span) < 2 * np.pi:
        raise ValueError("invalid GT homography or canonical arc")
    inverse = np.linalg.inv(h)
    center_h = inverse @ np.array([0., 0., 1.])
    progress = np.linspace(0., 1., K)
    angles = span * progress
    projected = np.stack((np.cos(angles), np.sin(angles), np.ones(K)), axis=1) @ inverse.T
    # Check the complete circular arc for a projective pole, including extrema
    # between knots. A finite center may lie on the other side of the horizon.
    low, high = sorted((0., span))
    phase = math.atan2(inverse[2, 1], inverse[2, 0])
    critical = [low, high] + [phase + j * np.pi for j in range(-3, 4) if low <= phase + j * np.pi <= high]
    denominators = inverse[2, 0] * np.cos(critical) + inverse[2, 1] * np.sin(critical) + inverse[2, 2]
    if abs(center_h[2]) <= 1e-10 or (denominators.min() <= 1e-10 and denominators.max() >= -1e-10):
        raise ValueError("GT center is infinite or circular arc crosses a projective pole")
    center = center_h[:2] / center_h[2]
    points = projected[:, :2] / projected[:, 2, None]
    phi = np.unwrap(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    increments = np.diff(phi)
    sign = 1 if np.all(increments > 0) else -1 if np.all(increments < 0) else 0
    if not sign or not 0 < abs(phi[-1] - phi[0]) < 2 * np.pi:
        raise ValueError("GT image bearings do not form one strictly ordered arc")
    knots = sign * (phi - phi[0])
    midpoint = (phi[0] + phi[-1]) / 2
    cells = (np.arange(output_size, dtype=np.float64) + .5) * image_size / output_size - .5
    xx, yy = np.meshgrid(cells, cells)
    image_angle = np.arctan2(yy - center[1], xx - center[0])
    unwrapped = (image_angle - midpoint + np.pi) % (2 * np.pi) - np.pi + midpoint
    t = sign * (unwrapped - phi[0])
    atlas = np.interp(t, knots, progress)
    canonical_h = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ h.T
    if not np.isfinite(canonical_h).all() or np.any(np.abs(canonical_h[..., 2]) <= 1e-10):
        raise ValueError("GT homography has an undefined output-cell coordinate")
    canonical = canonical_h[..., :2] / canonical_h[..., 2, None]
    theta = np.arctan2(canonical[..., 1], canonical[..., 0])
    raw_reference = ((theta - span / 2 + np.pi) % (2 * np.pi) - np.pi + span / 2) / span
    reference = np.clip(raw_reference, 0, 1)
    atlas_inside = (t >= 0) & (t <= knots[-1])
    reference_inside = (raw_reference >= 0) & (raw_reference <= 1)
    both_inside = atlas_inside & reference_inside
    both_outside = ~atlas_inside & ~reference_inside
    mismatch = atlas_inside != reference_inside
    difference = atlas - reference
    seam = both_outside & (np.abs(difference) > .5)
    return {"atlas": atlas, "reference": reference, "difference": difference,
            "both_inside": both_inside, "both_outside": both_outside, "mismatch": mismatch, "seam": seam,
            "atlas_first": t < 0, "atlas_last": t > knots[-1], "h_first": raw_reference < 0,
            "h_last": raw_reference > 1, "negative_h_denominator": canonical_h[..., 2] < 0,
            "exact_center_cell": (xx == center[0]) & (yy == center[1]),
            "direction_sign": sign, "canonical_span_radians": float(span),
            "image_span_radians": float(knots[-1]), "minimum_bearing_gap_radians": float(np.abs(increments).min()),
            "minimum_bearing_gap_fraction": float(np.abs(increments).min() / knots[-1]),
            "arc_denominator_min": float(denominators.min()), "arc_denominator_max": float(denominators.max()),
            "center_denominator": float(center_h[2]),
            "minimum_grid_distance_to_center_roi_pixels": float(np.hypot(xx - center[0], yy - center[1]).min()),
            "center_and_arc_denominator_same_sign": bool(center_h[2] * denominators[0] > 0)}


def readout(q: np.ndarray, chart: dict) -> tuple[float, float, dict]:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != chart["atlas"].shape or not np.isfinite(q).all() or np.any(q < 0) or q.sum() <= 0:
        raise ValueError("invalid full predicted Q")
    original_sum = float(q.sum())
    q = q / original_sum
    atlas, reference = (float(np.sum(q * chart[key])) for key in ("atlas", "reference"))
    delta = {label: float(np.sum(q * chart["difference"] * chart[key])) for label, key in
             (("in_arc_interpolation_reading_delta", "both_inside"),
              ("out_of_arc_endpoint_reading_delta", "both_outside"), ("arc_membership_reading_delta", "mismatch"))}
    if abs(atlas - reference - sum(delta.values())) > 1e-12:
        raise ValueError("atlas/H reading decomposition failed")
    diagnostic = {key: value for key, value in chart.items() if not isinstance(value, np.ndarray)}
    diagnostic.update(delta, atlas_minus_h_reading=atlas - reference, q_original_sum=original_sum)
    for label, key in (("q_atlas_clipped_first", "atlas_first"), ("q_atlas_clipped_last", "atlas_last"),
                       ("q_h_clipped_first", "h_first"), ("q_h_clipped_last", "h_last"),
                       ("q_endpoint_seam", "seam"), ("q_membership_mismatch", "mismatch"),
                       ("q_negative_h_denominator", "negative_h_denominator"), ("q_exact_center_cell", "exact_center_cell")):
        diagnostic[label] = float(q[chart[key]].sum())
    inside_error = np.abs(chart["difference"][chart["both_inside"]])
    diagnostic.update(in_arc_cells=int(inside_error.size),
                      in_arc_max_interpolation_error_percent_fs=float(100 * inside_error.max()) if inside_error.size else None,
                      in_arc_mean_interpolation_error_percent_fs=float(100 * inside_error.mean()) if inside_error.size else None,
                      endpoint_seam_cells=int(chart["seam"].sum()), membership_mismatch_cells=int(chart["mismatch"].sum()))
    return atlas, reference, diagnostic


def roi_calibration(row: dict, image_size: int) -> dict:
    from experiments.geoprr_v2_data import GeometryConfig, dial_calibration

    metadata = row["metadata"]
    width, height = int(metadata["image_width"]), int(metadata["image_height"])
    bbox = None if "roi_path" in row else metadata.get("dial_bbox")
    left, top, right, bottom = (0, 0, width, height) if bbox is None else (
        max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1])),
        min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3])))
    if right <= left or bottom <= top:
        raise ValueError("GT ROI does not intersect annotated image dimensions")
    sx, sy = image_size / (right - left), image_size / (bottom - top)
    original_to_roi = np.array([[sx, 0, sx * (.5 - left) - .5],
                                [0, sy, sy * (.5 - top) - .5], [0, 0, 1]], dtype=np.float64)
    return dial_calibration(row, original_to_roi, GeometryConfig())


def reference_rows(path: Path, declared: list[dict]) -> list[dict]:
    from experiments.geoprr_v2_data import normalized_target

    with path.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in source}
    if len(source) != POPULATION or len(by_id) != POPULATION or set(by_id) != {str(row["sample_id"]) for row in declared}:
        raise ValueError("formal reference CSV differs from the frozen 1506 IDs")
    result = []
    for row in declared:
        ref = by_id[str(row["sample_id"])]
        target = float(np.float32(normalized_target(row)))
        if float(ref["target"]) != target or str(ref["group_id"]) != str(row.get("group_id", row["sample_id"])):
            raise ValueError("formal reference target or group differs from the manifest")
        valid = ref["valid"].lower() == "true"
        result.append(dict(ref, target=target, prediction=float(ref["prediction"]) if valid else None,
                           absolute_error=float(ref["absolute_error"]), valid=valid,
                           field_valid=ref["field_valid"].lower() == "true"))
    return result


def paired(candidate: list[dict], reference: list[dict]) -> dict:
    result = {"nmae_delta_percent_fs": 100 * np.mean([a["absolute_error"] - b["absolute_error"]
                                                   for a, b in zip(candidate, reference, strict=True)])}
    for percent in (1, 2, 5):
        threshold = percent / 100
        result[f"above_{percent}_percent_fs"] = {
            "rescued_ids": [a["sample_id"] for a, b in zip(candidate, reference, strict=True)
                            if a["absolute_error"] <= threshold < b["absolute_error"]],
            "new_error_ids": [a["sample_id"] for a, b in zip(candidate, reference, strict=True)
                              if b["absolute_error"] <= threshold < a["absolute_error"]]}
    return result


def evaluate(args: argparse.Namespace) -> dict:
    import torch
    from experiments.evaluate_geoprr_v2 import FullPopulationDataset
    from experiments.geoprr_v2_data import GeoPRRV2Dataset, normalized_target, read_rows
    from experiments.geoprr_v2_model import GeoPRRV2, apply_modal_field_readout
    from experiments.train_geoprr_v2 import PROTOCOL, device_batch, make_loader, prediction_rows, summarize_rows, write_json

    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("smoke_only", False) or checkpoint["epoch"] != 63:
        raise ValueError("evaluation requires the full supervised best epoch 63 checkpoint")
    if config["training"]["seed"] != 20262020 or config["model"]["variant"] != "field":
        raise ValueError("evaluation requires the frozen supervised seed 20262020 field reader")
    state_weights = config["training"].get("loss_weights", config.get("loss_weights", {}))
    if any(float(state_weights.get(key, 0)) for key in ("state_field_l1", "state_support_l1", "state_reading_cdf", "state_mass_l1")):
        raise ValueError("the checkpoint has active state supervision")
    declared = read_rows(args.manifest)
    frozen = read_rows(checkpoint["manifests"]["validation"])
    if len(declared) != POPULATION or [(r["sample_id"], normalized_target(r)) for r in declared] != [
            (r["sample_id"], normalized_target(r)) for r in frozen]:
        raise ValueError("manifest differs from the checkpoint's complete frozen validation roster")
    references = {"native_reference": reference_rows(args.reference_native, declared),
                  "r8_reference": reference_rows(args.reference_r8, declared)}
    source = GeoPRRV2Dataset(args.manifest, dataset_root=args.root / "datasets", **config["data"],
                            require_geometry=False, augment=False)
    dataset = FullPopulationDataset(source)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("the formal replay requires CUDA batch-16 AMP")
    model = GeoPRRV2(**dict(config["model"], pretrained=False)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 7695954 or source.image_size != 768 or source.output_size != 192:
        raise ValueError("reader structure differs from the frozen supervised protocol")
    args.output.mkdir(parents=True, exist_ok=False)
    loader = make_loader(dataset, batch_size=16, workers=4, shuffle=False, seed=20262020, device=device)
    populations = {"native_replay": [], "r8_replay": [], **references}
    q_cache = np.empty((POPULATION, source.output_size, source.output_size), dtype=np.float32)
    input_valid, input_failures = [], []
    forward_seconds = r8_seconds = 0.
    offset = batches = 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            batch = device_batch(raw_batch, device)
            torch.cuda.synchronize(device)
            step = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch["image"])
            torch.cuda.synchronize(device)
            forward_seconds += time.perf_counter() - step
            step = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                modal = apply_modal_field_readout(output, 8)
            torch.cuda.synchronize(device)
            r8_seconds += time.perf_counter() - step
            for label, value in (("native_replay", output), ("r8_replay", modal)):
                rows = prediction_rows(value, batch)
                for index, row in enumerate(rows):
                    if not bool(raw_batch["input_valid"][index]):
                        row.update(prediction=None, valid=False, absolute_error=1., field_valid=False,
                                   support_mass=None, failure=raw_batch["input_failure"][index])
                populations[label].extend(rows)
            q = output["point_probs"].detach().float().cpu().numpy()[:, 0]
            q_cache[offset:offset + len(q)] = q
            offset += len(q)
            batches += 1
            input_valid.extend(bool(value) for value in raw_batch["input_valid"])
            input_failures.extend(raw_batch["input_failure"])
    inference_seconds = time.perf_counter() - inference_started
    peak_cuda = torch.cuda.max_memory_allocated(device)
    if offset != POPULATION:
        raise RuntimeError("the replay lost source-validation rows")
    del output, modal, model, checkpoint, batch
    torch.cuda.empty_cache()
    geometry_started = time.perf_counter()
    populations.update(gt_atlas_full_q=[], gt_h_full_q=[])
    geometry_records = []
    homographies = np.zeros((POPULATION, 3, 3), dtype=np.float64)
    spans = np.zeros(POPULATION, dtype=np.float64)
    for index, row in enumerate(declared):
        base = {key: populations["native_replay"][index][key] for key in ("sample_id", "group_id", "target")}
        diagnostic = dict(base, geometry_valid=False, geometry_failure="")
        readings = (None, None)
        try:
            calibration = roi_calibration(row, source.image_size)
            h, span = np.asarray(calibration["roi_to_canonical"], dtype=np.float64), float(calibration["angles"][-1])
            homographies[index], spans[index] = h, span
            chart = calibration_charts(h, span, source.image_size, source.output_size)
            atlas, reference, detail = readout(q_cache[index], chart)
            diagnostic.update(detail, geometry_valid=True)
            if input_valid[index]:
                readings = (atlas, reference)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            diagnostic["geometry_failure"] = f"{type(exc).__name__}: {exc}"
        for label, reading in zip(("gt_atlas_full_q", "gt_h_full_q"), readings, strict=True):
            valid = reading is not None and math.isfinite(reading) and 0 <= reading <= 1
            failure = input_failures[index] if not input_valid[index] else diagnostic["geometry_failure"]
            populations[label].append(dict(base, prediction=reading if valid else None, valid=valid,
                                           absolute_error=abs(reading - base["target"]) if valid else 1.,
                                           failure=failure if not valid else ""))
        geometry_records.append(diagnostic)
    geometry_seconds = time.perf_counter() - geometry_started
    flattened = []
    for index, detail in enumerate(geometry_records):
        record = dict(detail)
        for label, population in populations.items():
            for key in ("prediction", "valid", "absolute_error", "failure", "field_valid", "support_mass"):
                if key in population[index]:
                    record[f"{label}_{key}"] = population[index][key]
        flattened.append(record)
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (args.output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    cache_started = time.perf_counter()
    np.savez_compressed(args.output / "frozen_q_geometry.npz", q=q_cache, roi_to_canonical=homographies,
                        arc_span_radians=spans, sample_id=np.array([r["sample_id"] for r in declared]),
                        target=np.array([r["target"] for r in geometry_records]), input_valid=np.array(input_valid),
                        geometry_valid=np.array([r["geometry_valid"] for r in geometry_records]),
                        image_size=np.array(source.image_size), output_size=np.array(source.output_size), K=np.array(K))
    cache_seconds = time.perf_counter() - cache_started
    replay = {}
    for label in ("native", "r8"):
        current, formal = populations[f"{label}_replay"], populations[f"{label}_reference"]
        deltas = [abs(a["prediction"] - b["prediction"]) for a, b in zip(current, formal, strict=True) if a["valid"] and b["valid"]]
        replay[label] = {"maximum_prediction_difference": max(deltas) if deltas else None,
                         "validity_disagreement_ids": [a["sample_id"] for a, b in zip(current, formal, strict=True) if a["valid"] != b["valid"]],
                         **paired(current, formal)}
    summaries = {label: summarize_rows(rows) for label, rows in populations.items()}
    for label in ("gt_atlas_full_q", "gt_h_full_q"):
        del summaries[label]["field_coverage"]
    result = {"protocol": "geoprr_v2_full_source_validation_gt_atlas_frozen_q_K65_v1", "diagnostic_only": True,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": 63, "seed": 20262020,
              "manifest": str(args.manifest.resolve()), "declared_samples": POPULATION, "K": K,
              "reference_native": str(args.reference_native.resolve()), "reference_r8": str(args.reference_r8.resolve()),
              "training_or_adaptation": False, "reader_parameters": parameters,
              "model_inputs": "RGB ROI only; GT geometry used exclusively after frozen Q inference",
              "readout": "full normalized predicted Q expectation of GT chart; no M, window, threshold or fallback",
              "coordinate_rule": "output cell (j+.5)*image_size/output_size-.5; float64 ROI-to-canonical H",
              "chart_rule": "65 projected circle rays; strictly ordered CW/CCW; arc-midpoint unwrap; endpoint clipping",
              "exact_center_rule": "atan2(0,0)=0; exact-center Q mass is retained and reported",
              "denominator": "all original 1506 rows; input/geometry/readout failures retain normalized error 1",
              "limitations": ["GT atlas and GT H are conditional oracle diagnostics, not predicted model performance or upper bounds.",
                              "Source validation is development data; no new seed, training, official test or real-domain transfer.",
                              "Within-arc interpolation and out-of-arc chart endpoint seams are reported separately."],
              "summary": summaries, "replay_vs_formal": replay,
              "paired": {f"{label}_vs_{reference}": paired(populations[label], populations[reference])
                         for label in ("gt_atlas_full_q", "gt_h_full_q")
                         for reference in ("native_reference", "r8_reference", "native_replay", "r8_replay")},
              "geometry": {"feasible": sum(r["geometry_valid"] for r in geometry_records),
                           "failed_ids": [r["sample_id"] for r in geometry_records if not r["geometry_valid"]],
                           "positive_image_bearing_direction": sum(r.get("direction_sign") == 1 for r in geometry_records),
                           "negative_image_bearing_direction": sum(r.get("direction_sign") == -1 for r in geometry_records)},
              "resources": {"device": str(device), "gpu": torch.cuda.get_device_name(device), "batch_size": 16,
                            "workers": 4, "amp": True, "autocast_dtype": "float16", "q_cache_dtype": "float32",
                            "geometry_and_expectation_dtype": "float64", "model_forward_batches": batches,
                            "model_forward_images": offset, "valid_input_images": sum(input_valid),
                            "forward_seconds": forward_seconds, "r8_readout_seconds": r8_seconds,
                            "inference_pipeline_seconds": inference_seconds,
                            "inference_boundary": "image IO/ROI/normalization/transfer/model/native+r8 collection/Q CPU cache; batch throughput, no warmup exclusion",
                            "cpu_geometry_seconds": geometry_seconds, "cache_compression_seconds": cache_seconds,
                            "peak_cuda_allocated_bytes": peak_cuda, "peak_cuda_boundary": "model resident plus frozen inference/native+r8; reset after checkpoint/model loading",
                            "cache_bytes": (args.output / "frozen_q_geometry.npz").stat().st_size,
                            "process_wall_seconds_before_summary_write": time.perf_counter() - started,
                            "process_user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                            "process_system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    write_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-native", type=Path, required=True)
    parser.add_argument("--reference-r8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = evaluate(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("summary", "replay_vs_formal", "geometry", "resources")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
