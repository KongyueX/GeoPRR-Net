"""Aggregate DeepLabV3+-ROI and YOLO11s-Pose-4KP evaluation summaries."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.roi_geometry_comparison import write_json


METRICS = (
    "nmae",
    "nmae_percent_fs",
    "rmse",
    "acc_at_1_percent_fs",
    "acc_at_2_percent_fs",
    "acc_at_5_percent_fs",
    "coverage",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"summary root is not an object: {path}")
    return value


def _aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        array = np.asarray([float(value[metric]) for value in values], dtype=np.float64)
        result[metric] = {
            "mean": float(np.mean(array)),
            "sample_sd": float(np.std(array, ddof=1)) if len(array) > 1 else None,
        }
    return result


def _method(paths: Sequence[Path]) -> dict[str, Any] | None:
    if not paths:
        return None
    summaries = [_load(path) for path in paths]
    seeds = [int(value["seed"]) for value in summaries]
    all_values = [value["all_conditions"] for value in summaries]
    conditions = tuple(summaries[0]["per_condition"])
    return {
        "method": str(summaries[0]["method"]).rsplit("_seed_", 1)[0],
        "seeds": seeds,
        "runs": {str(seed): value for seed, value in zip(seeds, summaries, strict=True)},
        "all_conditions": _aggregate(all_values),
        "per_condition": {
            condition: _aggregate(
                [value["per_condition"][condition] for value in summaries]
            )
            for condition in conditions
        },
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    methods = {
        "deeplabv3plus_roi": _method(args.deeplab),
        "yolo11s_pose4kp": _method(args.yolo),
    }
    methods = {key: value for key, value in methods.items() if value is not None}
    if not methods:
        raise ValueError("at least one evaluation summary is required")
    result: dict[str, Any] = {
        "schema_version": 1,
        "comparison": "matched canonical-ROI geometry baselines",
        "methods": methods,
        "fairness": {
            "outer_holdout": "same 1,558 SyncG scene-disjoint samples",
            "conditions": "same clean/blur/perspective six-condition pixels",
            "failure_policy": "failed row receives normalized error 1.0",
            "deeplab_role": "annotation-assisted pointer segmentation component; GT pivot/endpoints used offline",
            "yolo_role": "ROI-localized four-keypoint model; no GT geometry used for prediction",
        },
    }
    if args.geoprr_vdn is not None:
        reference = _load(args.geoprr_vdn)
        result["existing_geoprr_vdn_reference"] = {
            "source": str(Path(args.geoprr_vdn).resolve()),
            "all_conditions": reference.get("scopes", {}).get("all_conditions"),
        }
    write_json(Path(args.output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeplab", type=Path, action="append", default=[])
    parser.add_argument("--yolo", type=Path, action="append", default=[])
    parser.add_argument("--geoprr-vdn", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize(args)
    print(json.dumps({"status": "complete", "methods": list(result["methods"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "summarize"]
