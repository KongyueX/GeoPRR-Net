"""Three-seed aggregate for PG-SIAM-v3 versus same-seed DB-GAR18+SARN-v2.

Each training seed is scored independently with failures assigned normalized
error 1.0 on the full denominator.  Primary metrics are then averaged across
seeds, while paired group bootstrap uses each sample's mean three-seed error
difference.  The utility also reports per-seed deltas and verifies exact
clean/blur fallback for every seed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.summarize_support_normalized_cbam_pilot import (
    CONDITIONS,
    FAILURE_ERROR,
    load_predictions,
    load_real_targets,
    load_scene_targets,
    paired_group_bootstrap,
)


PROTOCOL: Final[str] = "projective_geometry_guided_siam_v3_three_seed_summary_v1"
SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
IDENTITY_CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
)
PROJECTIVE_CONDITIONS: Final[tuple[str, ...]] = (
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _method_rows(
    path: Path, *, sample_ids: set[str]
) -> tuple[str, dict[tuple[str, str], tuple[float, bool]]]:
    methods, _bindings = load_predictions((path,), sample_ids=sample_ids)
    _require(len(methods) == 1, "prediction file must contain one method")
    method = next(iter(methods))
    return method, methods[method]


def summarize_multiseed(
    *,
    dataset: str,
    targets: Mapping[str, tuple[float, str]],
    candidate_paths: Sequence[Path],
    parent_paths: Sequence[Path],
    seeds: Sequence[int] = SEEDS,
    replicates: int = 20_000,
    bootstrap_seed: int = 20262050,
    group_unit: str = "dataset group_id cluster",
) -> dict[str, Any]:
    _require(len(candidate_paths) == len(parent_paths) == len(seeds) >= 2, "seed/path count mismatch")
    sample_ids = list(targets)
    sample_set = set(sample_ids)
    aligned_methods, prediction_bindings = load_predictions(
        tuple(candidate_paths) + tuple(parent_paths),
        sample_ids=sample_set,
    )
    candidate_by_seed: list[dict[tuple[str, str], tuple[float, bool]]] = []
    parent_by_seed: list[dict[tuple[str, str], tuple[float, bool]]] = []
    methods: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []
    for seed, candidate_path, parent_path in zip(seeds, candidate_paths, parent_paths, strict=True):
        candidate_method, _candidate_rows = _method_rows(
            candidate_path, sample_ids=sample_set
        )
        parent_method, _parent_rows = _method_rows(
            parent_path, sample_ids=sample_set
        )
        _require(candidate_method.endswith(f"seed_{seed}"), f"candidate seed mismatch: {seed}")
        _require(parent_method.endswith(f"seed_{seed}"), f"parent seed mismatch: {seed}")
        candidate = aligned_methods[candidate_method]
        parent = aligned_methods[parent_method]
        candidate_by_seed.append(candidate)
        parent_by_seed.append(parent)
        methods.append({"seed": int(seed), "candidate": candidate_method, "parent": parent_method})
        for condition in IDENTITY_CONDITIONS:
            for sample_id in sample_ids:
                c = candidate[(sample_id, condition)]
                p = parent[(sample_id, condition)]
                if c != p:
                    identity_mismatches.append(
                        {"seed": seed, "sample_id": sample_id, "condition": condition}
                    )

    groups = [targets[sample_id][1] for sample_id in sample_ids]
    condition_metrics: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    direction_counts: dict[str, int] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        candidate_errors: list[float] = []
        parent_errors: list[float] = []
        candidate_acc5 = 0
        parent_acc5 = 0
        candidate_coverage = 0.0
        parent_coverage = 0.0
        for sample_id in sample_ids:
            target = targets[sample_id][0]
            candidate_values = [rows[(sample_id, condition)] for rows in candidate_by_seed]
            parent_values = [rows[(sample_id, condition)] for rows in parent_by_seed]
            candidate_seed_errors = [
                abs(value - target) if passed else FAILURE_ERROR
                for value, passed in candidate_values
            ]
            parent_seed_errors = [
                abs(value - target) if passed else FAILURE_ERROR
                for value, passed in parent_values
            ]
            candidate_error = sum(candidate_seed_errors) / len(candidate_seed_errors)
            parent_error = sum(parent_seed_errors) / len(parent_seed_errors)
            candidate_errors.append(candidate_error)
            parent_errors.append(parent_error)
            candidate_coverage += sum(int(value[1]) for value in candidate_values) / len(candidate_values)
            parent_coverage += sum(int(value[1]) for value in parent_values) / len(parent_values)
            candidate_acc5 += sum(
                int(passed and error <= 0.05)
                for error, (_value, passed) in zip(
                    candidate_seed_errors, candidate_values, strict=True
                )
            ) / len(candidate_values)
            parent_acc5 += sum(
                int(passed and error <= 0.05)
                for error, (_value, passed) in zip(
                    parent_seed_errors, parent_values, strict=True
                )
            ) / len(parent_values)
        candidate_nmae = sum(candidate_errors) / len(candidate_errors)
        parent_nmae = sum(parent_errors) / len(parent_errors)
        condition_metrics.append(
            {
                "condition": condition,
                "samples": len(sample_ids),
                "groups": len(set(groups)),
                "candidate_nmae": candidate_nmae,
                "parent_nmae": parent_nmae,
                "candidate_acc_at_5": candidate_acc5 / len(sample_ids),
                "parent_acc_at_5": parent_acc5 / len(sample_ids),
                "candidate_coverage": candidate_coverage / len(sample_ids),
                "parent_coverage": parent_coverage / len(sample_ids),
            }
        )
        bootstrap = paired_group_bootstrap(
            candidate_errors,
            parent_errors,
            groups,
            replicates=replicates,
            seed=bootstrap_seed + condition_index,
        )
        paired.append({"condition": condition, **bootstrap})

        seed_deltas: list[dict[str, Any]] = []
        for seed, candidate, parent in zip(seeds, candidate_by_seed, parent_by_seed, strict=True):
            c_errors: list[float] = []
            p_errors: list[float] = []
            for sample_id in sample_ids:
                target = targets[sample_id][0]
                c_value, c_pass = candidate[(sample_id, condition)]
                p_value, p_pass = parent[(sample_id, condition)]
                c_errors.append(abs(c_value - target) if c_pass else FAILURE_ERROR)
                p_errors.append(abs(p_value - target) if p_pass else FAILURE_ERROR)
            delta = sum(c - p for c, p in zip(c_errors, p_errors, strict=True)) / len(c_errors)
            seed_deltas.append({"seed": int(seed), "delta_nmae": delta})
        per_seed.append({"condition": condition, "values": seed_deltas})
        direction_counts[condition] = sum(value["delta_nmae"] <= 0.0 for value in seed_deltas)

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": dataset,
        "seeds": list(map(int, seeds)),
        "methods": methods,
        "samples": len(sample_ids),
        "groups": len(set(groups)),
        "aggregation": (
            "score each training seed with failure error 1.0, then average per-sample "
            "errors/accuracy/coverage across seeds before group-bootstrap"
        ),
        "paired_input_pixel_identity": True,
        "prediction_bindings": prediction_bindings,
        "condition_metrics": condition_metrics,
        "paired_comparisons": paired,
        "per_seed_deltas": per_seed,
        "nonpositive_seed_counts": direction_counts,
        "identity_mismatch_count": len(identity_mismatches),
        "identity_mismatch_examples": identity_mismatches[:20],
        "bootstrap": {
            "replicates": int(replicates),
            "seed_base": int(bootstrap_seed),
            "unit": group_unit,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("xm2", "scene", "rf100"), required=True)
    parser.add_argument("--candidate", type=Path, nargs=3, required=True)
    parser.add_argument("--parent", type=Path, nargs=3, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset == "scene":
        _require(args.manifest is not None and args.split is not None, "scene targets missing")
        targets = load_scene_targets(args.manifest, args.split)
    else:
        _require(args.labels is not None and args.sample_ids is not None, "real targets missing")
        targets = load_real_targets(args.labels, args.sample_ids)
    value = summarize_multiseed(
        dataset=args.dataset,
        targets=targets,
        candidate_paths=args.candidate,
        parent_paths=args.parent,
        replicates=args.bootstrap_replicates,
    )
    output = Path(args.output).resolve()
    _require(not output.exists(), f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(value))
    print(json.dumps({"status": "complete", "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "summarize_multiseed"]
