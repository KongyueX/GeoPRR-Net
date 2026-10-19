"""Audit released SyncG geometry and discrete GT readout before training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.geoprr_v2_data import GeoPRRV2Dataset, GeometryConfig, geometry_targets


def audit_geometry(manifest: Path, *, dataset_root: Path | None = None,
                   image_size: int = 768, limit: int | None = None,
                   config: GeometryConfig | None = None,
                   probe_endpoints: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = config or GeometryConfig()
    dataset = GeoPRRV2Dataset(manifest, dataset_root=dataset_root, image_size=image_size,
                             geometry_config=config, augment=False)
    count = len(dataset) if limit is None else min(len(dataset), limit)
    records = []
    probes = []
    for index in range(count):
        row = dataset.rows[index]
        record: dict[str, Any] = {"sample_id": row["sample_id"], "group_id": row.get("group_id", "")}
        try:
            sample = dataset[index]
            record.update(status="pass", **sample["geometry_audit"])
            if probe_endpoints:
                # Numerical label probes reuse the measured layout/camera.
                # They are not rendered interventions or model evaluations.
                for y in (0.0, 0.001, 0.01, 0.99, 0.999, 1.0):
                    calibration = dict(dataset._calibrations[index], y=y)
                    probe = {"sample_id": row["sample_id"], "progress": y}
                    try:
                        target = geometry_targets(calibration, image_size, image_size // 4, config)
                        probe.update(status="pass", gt_readout_error=target["geometry_audit"]["gt_readout_error"])
                    except ValueError as exc:
                        probe.update(status="fail", reason=str(exc))
                    probes.append(probe)
        except (ValueError, OSError, np.linalg.LinAlgError) as exc:
            record.update(status="fail", reason=str(exc))
        records.append(record)
    passed = [row for row in records if row["status"] == "pass"]
    metric_names = list(passed[0].keys() - {"sample_id", "group_id", "status"}) if passed else []
    metrics = {}
    for name in sorted(metric_names):
        values = np.asarray([row[name] for row in passed])
        metrics[name] = {"mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)), "max": float(values.max())}
    summary = {"protocol": "geoprr_v2_dial_plane_geometry_audit_v2", "manifest": str(manifest.resolve()),
               "calibration": "known_tick_angles_similarity_v1",
               "diagnostic_only": count != len(dataset) or "diagnostic" in manifest.name or any(row.get("diagnostic_only", False) for row in dataset.rows),
               "image_size": image_size, "output_size": image_size // 4,
               "pixel_centers": "align_corners_false_half_pixel",
               "virtual_point": "physical_progress_on_rectified_ordered_major_tick_ring",
               "point_kernel": "canonical_gaussian_sigma=2*point_sigma/output_size_with_projective_jacobian",
               "homography_input": "source_pixels_normalized_by_annotation_bbox",
               "thresholds": asdict(config), "manifest_samples": len(dataset), "checked_samples": count,
               "passed": len(passed), "failed": count - len(passed),
               "contract_passed": count == len(dataset) and len(passed) == count and all(row["status"] == "pass" for row in probes),
               "metrics_for_passed_samples": metrics}
    if probe_endpoints:
        probe_errors = [row["gt_readout_error"] for row in probes if row["status"] == "pass"]
        summary["endpoint_label_probes"] = {"checked": len(probes), "failed": sum(row["status"] == "fail" for row in probes),
                                            "max_passed_readout_error": max(probe_errors) if probe_errors else None,
                                            "rendered_interventions": False, "records": probes}
    return summary, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-endpoints", action="store_true",
                        help="numerical label probes at both arc ends, not extra training images")
    parser.add_argument("--max-readout-error", type=float, default=0.01,
                        help="maximum GT field readout error in normalized full-scale units")
    parser.add_argument("--max-anchor-progress-error", type=float, default=0.01)
    parser.add_argument("--max-radial-error", type=float, default=0.02)
    parser.add_argument("--min-support-mass", type=float, default=0.15)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = GeometryConfig(max_readout_error=args.max_readout_error,
                            max_anchor_progress_error=args.max_anchor_progress_error,
                            max_radial_error=args.max_radial_error,
                            min_support_mass=args.min_support_mass)
    summary, records = audit_geometry(args.manifest, dataset_root=args.dataset_root,
                                      image_size=args.image_size, limit=args.limit, config=config,
                                      probe_endpoints=args.probe_endpoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["failed"] or summary.get("endpoint_label_probes", {}).get("failed", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
