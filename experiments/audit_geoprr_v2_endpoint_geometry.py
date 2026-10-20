"""Audit frozen source-validation endpoint geometry without images or models."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


POPULATION = 1506
PROTOCOL = "geoprr_v2_full1506_gt_endpoint_geometry_v1"


def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) <= 1e-12):
        raise ValueError("undefined endpoint homography coordinate")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def source_to_roi(row: dict, image_size: int) -> np.ndarray:
    metadata = row["metadata"]
    width, height = int(metadata["image_width"]), int(metadata["image_height"])
    bbox = None if "roi_path" in row else metadata.get("dial_bbox")
    left, top, right, bottom = (0, 0, width, height) if bbox is None else (
        max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1])),
        min(width, math.ceil(bbox[2])), min(height, math.ceil(bbox[3])))
    if right <= left or bottom <= top:
        raise ValueError("empty source ROI")
    sx, sy = image_size / (right - left), image_size / (bottom - top)
    return np.array([[sx, 0, sx * (.5 - left) - .5],
                     [0, sy, sy * (.5 - top) - .5], [0, 0, 1]], dtype=np.float64)


def circular_difference(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def metrics(rows: list[dict], column: str) -> dict:
    errors = np.array([r[column] for r in rows], dtype=np.float64)
    tail_count = math.ceil(.1 * len(errors))
    return {"samples": len(rows), "nmae_percent_fs": float(100 * errors.mean()),
            **{f"acc_at_{k}_percent": float(100 * np.mean(errors <= k / 100)) for k in (1, 2, 5)},
            **{f"above_{k}_percent_fs": int(np.sum(errors > k / 100)) for k in (1, 2, 5)},
            "median_percent_fs": float(100 * np.median(errors)),
            "p90_percent_fs": float(100 * np.quantile(errors, .9)),
            "p95_percent_fs": float(100 * np.quantile(errors, .95)),
            "tail10_count": tail_count, "tail10_percent_fs": float(100 * np.sort(errors)[-tail_count:].mean()),
            "maximum_percent_fs": float(100 * errors.max()),
            "maximum_ids": [r["sample_id"] for r in rows if r[column] == errors.max()]}


def paired(rows: list[dict], candidate: str, reference: str) -> dict:
    result = {"nmae_delta_percent_fs": float(100 * np.mean([r[candidate] - r[reference] for r in rows]))}
    for k in (1, 2, 5):
        cutoff = k / 100
        result[f"above_{k}_percent_fs"] = {
            "rescued_ids": [r["sample_id"] for r in rows if r[candidate] <= cutoff < r[reference]],
            "new_error_ids": [r["sample_id"] for r in rows if r[reference] <= cutoff < r[candidate]]}
    return result


def audit(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"protocol": PROTOCOL, "diagnostic_only": True, "population": POPULATION,
                "manifest": str(args.manifest), "geometry_cache": str(args.cache),
                "reference": str(args.reference), "image_or_model_loading": False,
                "new_model_forwards": 0, "training_steps": 0, "seed_expansion": False,
                "coordinates": "cached float64 ROI-to-canonical H composed with original half-pixel source-to-ROI",
                "readout": "transform both annotated points; difference; fixed arc-midpoint unwrap; endpoint clip",
                "target_rule": "original float32 reference for paired scores; manifest float64 y for circular residual",
                "denominator": "all original 1506 IDs; failed geometry has error 1",
                "quantization": "fixed conservative +/-1 source-pixel endpoint boxes; excludes H/tick-frame error",
                "raw_geometry_gate": "all 1506 valid and raw circular direction error <= existing 1%FS geometry tolerance",
                "height_contract": "must be established from source provenance; numerical agreement alone cannot prove equal height"}
    write_json(args.output / "protocol.json", protocol)
    declared = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    reference = read_csv(args.reference)
    refs = {r["sample_id"]: r for r in reference}
    ids = [r["sample_id"] for r in declared]
    if len(ids) != POPULATION or len(set(ids)) != POPULATION or len(refs) != POPULATION or len(reference) != POPULATION or set(ids) != set(refs):
        raise ValueError("manifest/reference differ from frozen 1506 population")
    with np.load(args.cache, allow_pickle=False) as cache:
        cached_ids = [str(s) for s in cache["sample_id"]]
        matrices = cache["roi_to_canonical"].astype(np.float64)
        spans = cache["arc_span_radians"].astype(np.float64)
        targets = cache["target"].astype(np.float64)
        image_size = int(cache["image_size"])
        cached_valid = cache["geometry_valid"].astype(bool)
    if cached_ids != ids:
        raise ValueError("cached geometry does not preserve manifest order")
    rows = []
    for i, row in enumerate(declared):
        ref = refs[row["sample_id"]]
        y64 = (float(row["ground_truth"]) - float(row["scale_start"])) / (float(row["scale_end"]) - float(row["scale_start"]))
        target = float(np.float32(y64))
        if target != float(ref["target"]) or target != targets[i] or str(row["group_id"]) != ref["group_id"]:
            raise ValueError(f"{row['sample_id']}: reference target/group mismatch")
        result = {"sample_id": row["sample_id"], "group_id": row["group_id"], "target": target,
                  "physical_target_float64": y64, "geometry_valid": False, "failure": "",
                  "pointer_entries": 0, "endpoint_integer_coordinates": False,
                  "canonical_origin_x": None, "canonical_origin_y": None,
                  "canonical_outside_x": None, "canonical_outside_y": None,
                  "source_endpoint_distance_pixels": None, "canonical_endpoint_distance": None,
                  "canonical_arc_radians": float(spans[i]), "direction_radians": None,
                  "raw_progress": None, "prediction": None, "absolute_error": 1.,
                  "raw_absolute_error": 1., "circular_direction_error": 1.,
                  "clipped_first": False, "clipped_last": False,
                  "quantization_angle_bound": None, "quantization_angle_unidentified": False,
                  "native_prediction": float(ref["native_reference_prediction"]),
                  "native_absolute_error": float(ref["native_reference_absolute_error"]),
                  "r8_prediction": float(ref["r8_reference_prediction"]),
                  "r8_absolute_error": float(ref["r8_reference_absolute_error"])}
        for prefix in ("native", "r8"):
            if ref[f"{prefix}_reference_valid"].lower() != "true" or abs(abs(result[f"{prefix}_prediction"] - target) - result[f"{prefix}_absolute_error"]) > 1e-12:
                raise ValueError("reference prediction/validity/error disagreement")
        try:
            entries = [entry for entry in row["metadata"]["keypoints"] if entry.get("type", "").lower() == "pointer"]
            result["pointer_entries"] = len(entries)
            if len(entries) != 1 or not cached_valid[i]:
                raise ValueError("missing unique pointer or cached invalid geometry")
            points = np.array([entries[0]["origin_kp"], entries[0]["outside_kp"]], dtype=np.float64)
            if points.shape != (2, 2) or not np.isfinite(points).all():
                raise ValueError("invalid endpoint labels")
            matrix = matrices[i] @ source_to_roi(row, image_size)
            canonical = transform(points, matrix)
            displacement = canonical[1] - canonical[0]
            length = float(np.linalg.norm(displacement))
            if length <= 1e-12 * max(1., float(np.linalg.norm(canonical[0])), float(np.linalg.norm(canonical[1]))):
                raise ValueError("degenerate endpoint displacement")
            span = float(spans[i])
            if not 0 < span < 2 * np.pi:
                raise ValueError("invalid calibrated arc")
            angle = math.atan2(displacement[1], displacement[0])
            raw = ((angle - span / 2 + np.pi) % (2 * np.pi) - np.pi + span / 2) / span
            prediction = float(np.clip(raw, 0., 1.))
            offsets = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
            movement = [float(np.linalg.norm(transform(p + offsets, matrix) - c, axis=1).max())
                        for p, c in zip(points, canonical, strict=True)]
            uncertainty = sum(movement) / length
            result.update(geometry_valid=True, endpoint_integer_coordinates=bool(np.all(points == np.floor(points))),
                          canonical_origin_x=float(canonical[0, 0]), canonical_origin_y=float(canonical[0, 1]),
                          canonical_outside_x=float(canonical[1, 0]), canonical_outside_y=float(canonical[1, 1]),
                          source_endpoint_distance_pixels=float(np.linalg.norm(points[1] - points[0])),
                          canonical_endpoint_distance=length, direction_radians=angle,
                          raw_progress=float(raw), prediction=prediction, absolute_error=abs(prediction - target),
                          raw_absolute_error=abs(raw - target),
                          circular_direction_error=abs(circular_difference(angle - span * y64)) / span,
                          clipped_first=bool(raw < 0), clipped_last=bool(raw > 1),
                          quantization_angle_bound=math.asin(uncertainty) / span if uncertainty < 1 else None,
                          quantization_angle_unidentified=bool(uncertainty >= 1))
        except (KeyError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            result["failure"] = str(exc)
        rows.append(result)
    write_csv(args.output / "predictions.csv", rows)
    severe = [r for r in rows if max(r["absolute_error"], r["native_absolute_error"], r["r8_absolute_error"]) > .02]
    if severe:
        write_csv(args.output / "severe_cases.csv", severe)
    group_rows = []
    for group in sorted({str(r["group_id"]) for r in rows}):
        subset = [r for r in rows if str(r["group_id"]) == group]
        group_rows.append({"group_id": group, **metrics(subset, "absolute_error")})
    write_csv(args.output / "group_metrics.csv", group_rows)
    valid = [r for r in rows if r["geometry_valid"]]
    result = {**protocol,
              "summary": {"gt_endpoint_pair": metrics(rows, "absolute_error"),
                          "gt_endpoint_pair_unclipped": metrics(rows, "raw_absolute_error"),
                          "raw_circular_direction": metrics(rows, "circular_direction_error"),
                          "supervised_native": metrics(rows, "native_absolute_error"),
                          "supervised_r8": metrics(rows, "r8_absolute_error")},
              "paired": {"vs_native": paired(rows, "absolute_error", "native_absolute_error"),
                         "vs_r8": paired(rows, "absolute_error", "r8_absolute_error")},
              "geometry": {"valid": len(valid), "failed_ids": [r["sample_id"] for r in rows if not r["geometry_valid"]],
                           "unique_pointer": sum(r["pointer_entries"] == 1 for r in rows),
                           "integer_endpoint_labels": sum(r["endpoint_integer_coordinates"] for r in rows),
                           "clipped_first": sum(r["clipped_first"] for r in rows), "clipped_last": sum(r["clipped_last"] for r in rows),
                           "raw_circular_above_percent_fs": {str(k): sum(r["circular_direction_error"] > k / 100 for r in rows) for k in (.1, .5, 1, 2, 5)},
                           "minimum_source_endpoint_distance_pixels": min(r["source_endpoint_distance_pixels"] for r in valid),
                           "minimum_canonical_endpoint_distance": min(r["canonical_endpoint_distance"] for r in valid),
                           "quantization_angle_unidentified": sum(r["quantization_angle_unidentified"] for r in rows),
                           "endpoint_quantization_bound_max_percent_fs": max(100 * r["quantization_angle_bound"] for r in valid if r["quantization_angle_bound"] is not None),
                           "raw_circular_beyond_endpoint_only_quantization_bound": sum(r["circular_direction_error"] > r["quantization_angle_bound"] for r in valid if r["quantization_angle_bound"] is not None)},
              "numerical_gate_passed": len(valid) == POPULATION and max(r["circular_direction_error"] for r in rows) <= .01,
              "height_hypothesis_status": "not established by numerical audit; inspect source provenance separately",
              "resources": {"cpu_wall_seconds": time.perf_counter() - started,
                            "user_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                            "system_cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "numpy_version": np.__version__, "new_forward_images": 0, "gpu_calls": 0}}
    write_json(args.output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    print(json.dumps({key: result[key] for key in ("summary", "geometry", "numerical_gate_passed", "resources")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
