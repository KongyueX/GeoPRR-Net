"""Recompute frozen field-transfer metrics from YOLO, DeepLab, and VDN rows.

This does not rerun inference. Industrial supports the original legacy reference
detector or a source-trained pose reference. RF100 retains annotated reference
geometry and establishes only geometry-assisted component transfer.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MISSING_ROOT = ROOT / "artifacts/runs/roi_missing_baselines_three_seed"
SEEDS = (20262020, 20262021, 20262022)
CONDITIONS = (
    "clean", "blur_moderate", "blur_severe", "perspective_moderate",
    "perspective_severe", "combined_severe",
)
METHODS = {
    "yolo11s_pose4kp": "YOLO11s-Pose-4KP",
    "deeplabv3plus_roi": "DeepLabV3+-ROI",
    "vdn": "VDN",
}
COHORTS = {
    "industrial_pooled": ("Industrial-1395", (
        "field_gauge_roi_test_a", "field_gauge_roi_test_b",
        "field_gauge_external_roi",
    ), 1395, 52),
    "rf100": ("RF100-VL", ("rf100",), 151, 35),
}
METRICS = {
    "nmae_percent_fs": ("nmae_percent_fs", 1),
    "acc_at_2_percent": ("acc_at_2_percent_fs", 100),
    "acc_at_5_percent": ("acc_at_5_percent_fs", 100),
    "coverage_percent": ("coverage", 100),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _run(seed: int, method: str) -> Path:
    return ROOT / "artifacts/runs/roi_comparison_pilot" / f"seed_{seed}" / method


def _field_dir(seed: int, method: str, slug: str, missing_root: Path,
               rf100_missing_root: Path = ORIGINAL_MISSING_ROOT) -> Path:
    if method == "vdn" or (method == "deeplabv3plus_roi" and slug != "rf100"):
        name = "vdn" if method == "vdn" else "deeplabv3plus_roi_auto_geometry"
        selected_root = rf100_missing_root if slug == "rf100" else missing_root
        return selected_root / f"seed_{seed}" / name / "field" / slug
    return _run(seed, method) / "field" / slug


def _training_evidence() -> list[str]:
    evidence = ["experiments/train_yolo11s_pose4kp.py",
                "experiments/train_deeplabv3plus_roi.py"]
    for seed in SEEDS:
        for method in ("yolo11s_pose4kp", "deeplabv3plus_roi"):
            path = _run(seed, method) / "training_summary.json"
            value = _read(path)
            if (value["training_samples"], value["validation_samples"],
                    value["subset_run"]) != (12866, 1576, False):
                raise ValueError(f"Unexpected training counts in {path}")
            evidence.append(_relative(path))
        path = (ROOT / "artifacts/runs/geoprr_vdn_matched"
                / f"seed_{seed}" / "summary.json")
        signature = _read(path)["signature"]
        if (signature["train_samples"], signature["validation_samples"],
                signature["epochs"]) != (12866, 1576, 200):
            raise ValueError(f"Unexpected VDN training counts in {path}")
        evidence.append(_relative(path))
    return evidence


def _geometry_provenance(missing_root: Path) -> dict[str, Any]:
    import torch

    cache_paths = set()
    for seed in SEEDS:
        for method in ("deeplabv3plus_roi", "vdn"):
            for slug in COHORTS["industrial_pooled"][1]:
                metadata = _read(_field_dir(seed, method, slug, missing_root) / "summary.json")
                cache_paths.add(Path(metadata["configuration"]["geometry_cache"]).resolve())
    known_source = {
        (_run(seed, "yolo11s_pose4kp") / "ultralytics/weights/best.pt").resolve(): seed
        for seed in SEEDS
    }
    legacy = (ROOT / "utils/angleDetect/yoloDetection/result/yolo_pointbest.pt").resolve()
    caches, kinds = [], set()
    for cache in sorted(cache_paths):
        summary_path = cache.with_suffix(".summary.json")
        geometry = _read(summary_path)
        path = Path(geometry["reference_detector_weights"]).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        args = checkpoint.get("train_args", {})
        if path == legacy:
            kind = "legacy_reference"
        elif path in known_source:
            kind = "source_only_pose4kp_reference"
        else:
            kind = "unverified_reference"
        kinds.add(kind)
        entry = {
            "predictions": str(cache), "summary": str(summary_path),
            "reference_kind": kind, "checkpoint": str(path),
            "checkpoint_metadata_verified_on_cpu": {
                "date": checkpoint.get("date"), "ultralytics_version": checkpoint.get("version"),
                **{key: args.get(key) for key in ("data", "epochs", "imgsz", "seed")},
            },
            "industrial_geometry": {key: geometry[key] for key in (
                "rows", "passes", "failures", "failure_codes", "detector_configuration",
            ) if key in geometry},
            "geometry_role": geometry.get("geometry_role"),
            "reference_detector_kind": geometry.get("reference_detector_kind"),
            "reference_detector_training": geometry.get("reference_detector_training"),
        }
        if kind == "source_only_pose4kp_reference":
            entry["training_summary"] = _relative(
                _run(known_source[path], "yolo11s_pose4kp") / "training_summary.json")
        caches.append(entry)
    kind = next(iter(kinds)) if len(kinds) == 1 else "mixed_reference"
    return {
        "reference_kind": kind,
        "training_roster": ("Source-only YOLO11s-Pose-4KP training summaries identified."
                            if kind == "source_only_pose4kp_reference" else
                            "Legacy or unknown reference training roster unavailable; target overlap cannot be ruled out."),
        "scope": "Provided ROI only; RF100 DeepLab/VDN still use annotated reference geometry.",
        "legacy_provenance_evidence": "docs/THIRD_PARTY_PROVENANCE_AUDIT_CN.md:220",
        "caches": caches,
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    errors = [row["_error"] for row in rows]
    failures = sum(row["status"] == "fail" for row in rows)
    return {
        "rows": count, "failures": failures,
        "nmae_percent_fs": 100 * statistics.fmean(errors),
        "acc_at_2_percent": 100 * sum(error <= 0.02 for error in errors) / count,
        "acc_at_5_percent": 100 * sum(error <= 0.05 for error in errors) / count,
        "coverage_percent": 100 * (count - failures) / count,
    }


def summarize(missing_root: Path, reference_path: Path | None,
              rf100_missing_root: Path = ORIGINAL_MISSING_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = _read(reference_path) if reference_path is not None else None
    geometry_provenance = _geometry_provenance(missing_root)
    output: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(), "status": "complete",
        "description": "Frozen source-trained component transfer recomputed from stored per-row predictions; this summarizer itself performs no inference. Prediction source paths identify the evaluated runs.",
        "seeds": list(SEEDS),
        "source_domain": {
            "name": "SyncG", "train_samples": 12866, "validation_samples": 1576,
            "selection": "YOLO: best inner-validation fitness, 30-epoch budget; DeepLab: best inner-validation Dice, 20-epoch budget; VDN: terminal epoch 200 matched checkpoint.",
            "evidence": _training_evidence(),
        },
        "protocol": {
            "conditions": list(CONDITIONS),
            "clean": "Single undegraded condition; one row per image per seed.",
            "all_conditions": "Six conditions pooled by image-condition rows; each image has equal weight; no source-level macro average.",
            "error": "abs(normalized_progress - normalized_target); all failed rows receive 1.0.",
            "acc_thresholds": "Inclusive absolute error <= 0.02 / <= 0.05; failed rows remain in denominator.",
            "units": "Metrics are percentages; NMAE is %FS. SD is sample standard deviation across three seeds (ddof=1).",
            "field_use": "Retrospective test only for the three retrained reader components; previously viewed field data are not a new confirmatory set.",
            "strict_zero_shot_scope": {
                "yolo11s_pose4kp": "Source-trained reader evaluated frozen within a supplied ROI; no target geometry used at inference. Full-frame ROI acquisition is outside scope.",
                "deeplabv3plus_roi": "Industrial: source-trained segmentation plus legacy automatic geometry detector of unverified training provenance. RF100: annotated pivot/minimum/maximum at inference; geometry-assisted component transfer.",
                "vdn": "Industrial: source-trained direction plus legacy automatic geometry detector of unverified training provenance. RF100: annotated pivot/minimum/maximum at inference; geometry-assisted component transfer.",
            },
        },
        "reference_kind": geometry_provenance["reference_kind"],
        "missing_root": str(missing_root.resolve()),
        "rf100_missing_root": str(rf100_missing_root.resolve()),
        "geometry_provenance": geometry_provenance, "datasets": {},
        "verification": {
            "rows_recomputed": 0, "stored_error_max_abs_difference": 0.0,
            "published_metric_max_abs_difference": 0.0,
            "matched_existing_summary": str(reference_path.resolve()) if reference_path else None,
            "checks": [
                "Pass errors recomputed from prediction and target; failed errors verified as 1.0.",
                "Expected sample, group and condition counts checked for each seed and cohort.",
                "Sample-condition identities, groups and targets agree across methods and seeds.",
            ],
        },
    }
    if geometry_provenance["reference_kind"] == "source_only_pose4kp_reference":
        for method, component in (("deeplabv3plus_roi", "segmentation"), ("vdn", "direction")):
            output["protocol"]["strict_zero_shot_scope"][method] = (
                f"Industrial: source-trained {component} plus source-trained YOLO11s-Pose-4KP reference geometry, evaluated frozen within supplied ROI. "
                "RF100: annotated pivot/minimum/maximum at inference; geometry-assisted component transfer."
            )
    if reference is not None:
        output["verification"]["checks"].append(
            "Per-seed and aggregate metrics agree with the supplied summary within 1e-10 percentage points.")
    audit = output["verification"]
    flat: list[dict[str, Any]] = []
    for cohort, (name, slugs, samples, groups) in COHORTS.items():
        dataset: dict[str, Any] = {
            "name": name, "samples": samples, "groups": groups,
            "rows_per_seed": samples * 6, "methods": {},
        }
        expected_roster = None
        for method, label in METHODS.items():
            records = []
            for seed_index, seed in enumerate(SEEDS):
                rows, sources, checkpoints = [], [], []
                for slug in slugs:
                    directory = _field_dir(seed, method, slug, missing_root, rf100_missing_root)
                    path = directory / "predictions.jsonl"
                    sources.append(_relative(path))
                    metadata = _read(directory / "summary.json")
                    checkpoint = Path(metadata["configuration"]["checkpoint"])
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    checkpoints.append(_relative(checkpoint))
                    rows.extend(json.loads(line) for line in
                                path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
                roster = {}
                for row in rows:
                    key = (row["dataset_slug"], row["sample_id"], row["condition"])
                    if key in roster or row["status"] not in ("pass", "fail"):
                        raise ValueError(f"Duplicate row or unknown status: {key}")
                    roster[key] = (row["group_id"], float(row["normalized_target"]))
                    error = (abs(float(row["normalized_progress"]) - float(row["normalized_target"]))
                             if row["status"] == "pass" else 1.0)
                    difference = abs(error - float(row["absolute_error"]))
                    if not math.isfinite(error) or difference > 1e-12:
                        raise ValueError(f"Invalid or inconsistent error: {key}")
                    audit["stored_error_max_abs_difference"] = max(
                        audit["stored_error_max_abs_difference"], difference)
                    row["_error"] = error
                if expected_roster is None:
                    expected_roster = roster
                elif roster != expected_roster:
                    raise ValueError(f"Method/seed target rosters differ for {cohort}")
                if (len(rows) != samples * 6
                        or len({key[:2] for key in roster}) != samples
                        or len({(key[0], value[0]) for key, value in roster.items()}) != groups
                        or Counter(row["condition"] for row in rows)
                        != Counter({condition: samples for condition in CONDITIONS})):
                    raise ValueError(f"Unexpected sample/group/condition counts for {cohort}")
                record: dict[str, Any] = {
                    "seed": seed, "checkpoints": sorted(set(checkpoints)),
                    "predictions": sources,
                    "failure_codes": dict(Counter(str(row["failure_code"])
                                                  for row in rows if row["status"] == "fail")),
                }
                for scope in ("clean", "all_conditions"):
                    selected = rows if scope == "all_conditions" else [
                        row for row in rows if row["condition"] == "clean"]
                    record[scope] = _metrics(selected)
                    if reference is None:
                        continue
                    previous = reference["datasets"][cohort]["methods"][method]
                    previous = (previous["all_conditions"] if scope == "all_conditions"
                                else previous["per_condition"]["clean"])
                    for metric, (alias, scale) in METRICS.items():
                        difference = abs(record[scope][metric]
                                         - previous[alias]["per_seed"][seed_index] * scale)
                        audit["published_metric_max_abs_difference"] = max(
                            audit["published_metric_max_abs_difference"], difference)
                        if difference > 1e-10:
                            raise ValueError(f"Published metric differs: {cohort}/{method}/{seed}/{scope}/{metric}")
                audit["rows_recomputed"] += len(rows)
                records.append(record)
            method_result: dict[str, Any] = {"name": label, "per_seed": records, "aggregate": {}}
            for scope in ("clean", "all_conditions"):
                aggregate: dict[str, Any] = {"rows_per_seed": samples * (6 if scope == "all_conditions" else 1)}
                csv_row = {"dataset": name, "method": label, "condition_scope": scope,
                           "samples": samples, "groups": groups, **aggregate}
                previous = None
                if reference is not None:
                    previous = reference["datasets"][cohort]["methods"][method]
                    previous = (previous["all_conditions"] if scope == "all_conditions"
                                else previous["per_condition"]["clean"])
                for metric, (alias, scale) in METRICS.items():
                    values = [record[scope][metric] for record in records]
                    aggregate[metric] = {"mean": statistics.fmean(values),
                                         "sample_sd": statistics.stdev(values),
                                         "per_seed": dict(zip(map(str, SEEDS), values))}
                    for stat in ("mean", "sample_sd"):
                        if previous is not None and not math.isclose(aggregate[metric][stat], previous[alias][stat] * scale,
                                            rel_tol=0, abs_tol=1e-10):
                            raise ValueError(f"Published aggregate differs: {cohort}/{method}/{scope}/{metric}")
                        csv_row[f"{metric}_{stat}"] = aggregate[metric][stat]
                    for seed, value in zip(SEEDS, values):
                        csv_row[f"{metric}_seed_{seed}"] = value
                method_result["aggregate"][scope] = aggregate
                flat.append(csv_row)
            dataset["methods"][method] = method_result
        output["datasets"][cohort] = dataset
    return output, flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/data")
    parser.add_argument("--missing-root", type=Path, default=ORIGINAL_MISSING_ROOT,
                        help="Industrial DeepLab/VDN prediction root.")
    parser.add_argument("--rf100-missing-root", type=Path, default=ORIGINAL_MISSING_ROOT,
                        help="RF100 VDN prediction root; annotation-assisted results remain unchanged.")
    parser.add_argument("--reference-summary", type=Path,
                        help="Optional previously generated summary to cross-check against.")
    args = parser.parse_args()
    output, rows = summarize(args.missing_root, args.reference_summary, args.rf100_missing_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "roi_comparison_zero_shot.json"
    csv_path = args.output_dir / "roi_comparison_zero_shot.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path.resolve()), "csv": str(csv_path.resolve()),
                      "verification": output["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
