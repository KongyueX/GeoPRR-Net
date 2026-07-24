"""Aggregate audited VDN retraining replicates for paper-facing tables.

The original VDN comparison script intentionally operates on one checkpoint so
that paired bootstrap intervals are tied to an exact run.  This companion
summary verifies several complete runs, reports run-to-run mean and sample
standard deviation, and compares the frozen final method with the mean
per-sample VDN error across the supplied seeds.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.vdn_baseline import (
    group_bootstrap_ci,
    normalized_error,
    sha256_file,
)

PROTOCOL = "vdn_three_seed_end_to_end_summary_v1"
VDN_EVALUATION_PROTOCOL = "vdn_syncg_external_baseline_e2e_v1"
FINAL_EVALUATION_PROTOCOL = "frozen_calibrated_progress_router_evaluation_v1"
CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
    "rpm10k",
)
DEFAULT_RUN_ROOTS = tuple(
    Path(f"artifacts/runs/vdn_syncg/seed_{seed}")
    for seed in (20260720, 20260721, 20260722)
)
DEFAULT_FINAL_ROOT = Path(
    "artifacts/runs/calibrated_progress_router_syncg/evaluations"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("mean/std requires finite values")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def _prediction_index(
    rows: Sequence[dict[str, Any]],
    *,
    source: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{source} contains an empty sample_id")
        if sample_id in result:
            raise ValueError(f"{source} contains duplicate sample_id {sample_id}")
        result[sample_id] = row
    return result


def _row_error(row: dict[str, Any]) -> float:
    return normalized_error(
        row.get("prediction"),
        float(row["ground_truth"]),
        float(row["scale_start"]),
        float(row["scale_end"]),
    )


def _assert_same_target(
    sample_id: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    for field in ("ground_truth", "scale_start", "scale_end"):
        if not math.isclose(
            float(first[field]),
            float(second[field]),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{sample_id}: inconsistent {field}")
    first_group = str(first.get("group_id") or first.get("meter_id") or sample_id)
    second_group = str(second.get("group_id") or second.get("meter_id") or sample_id)
    if first_group != second_group:
        raise ValueError(f"{sample_id}: inconsistent group_id")


def paired_final_vs_vdn_replicates(
    vdn_runs: Sequence[Sequence[dict[str, Any]]],
    final_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Compare final errors with each VDN run and their per-image mean.

    Averaging errors rather than predictions avoids inventing a VDN ensemble.
    It estimates the expected result of an independently trained VDN model.
    """

    if not vdn_runs:
        raise ValueError("at least one VDN run is required")
    final_index = _prediction_index(final_rows, source=Path("<final>"))
    vdn_indexes = [
        _prediction_index(rows, source=Path(f"<vdn-run-{index}>"))
        for index, rows in enumerate(vdn_runs)
    ]
    sample_ids = sorted(final_index)
    for index in vdn_indexes:
        if sorted(index) != sample_ids:
            raise ValueError("VDN and final runs use different sample sets")

    final_errors: list[float] = []
    mean_vdn_errors: list[float] = []
    groups: list[str] = []
    per_run_deltas: list[list[float]] = [[] for _ in vdn_indexes]
    for sample_id in sample_ids:
        final = final_index[sample_id]
        vdn_rows = [index[sample_id] for index in vdn_indexes]
        for row in vdn_rows:
            _assert_same_target(sample_id, final, row)
        final_error = _row_error(final)
        run_errors = [_row_error(row) for row in vdn_rows]
        final_errors.append(final_error)
        mean_vdn_errors.append(float(np.mean(run_errors)))
        groups.append(
            str(final.get("group_id") or final.get("meter_id") or sample_id)
        )
        for values, run_error in zip(per_run_deltas, run_errors):
            values.append(final_error - run_error)

    delta = np.asarray(final_errors) - np.asarray(mean_vdn_errors)
    per_seed_delta = [float(np.mean(values)) for values in per_run_deltas]
    return {
        "samples": len(sample_ids),
        "groups": len(set(groups)),
        "comparison_unit": "per-sample mean VDN error across independent seeds",
        "final_nmae_recomputed": float(np.mean(final_errors)),
        "vdn_nmae_mean_error_recomputed": float(np.mean(mean_vdn_errors)),
        "delta_nmae_final_minus_vdn": float(np.mean(delta)),
        "delta_nmae_group_bootstrap_95ci": group_bootstrap_ci(
            delta.tolist(),
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "per_seed_delta_nmae_final_minus_vdn": per_seed_delta,
        "per_seed_delta_nmae_mean_std": _mean_std(per_seed_delta),
        "final_better_rate": float(np.mean(delta < 0.0)),
        "tie_rate": float(np.mean(delta == 0.0)),
    }


def _load_vdn_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    summary_path = root / "summary.json"
    verification_path = root / "verification.json"
    checkpoint_path = root / "best.pt"
    for path in (summary_path, verification_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = _read_json(summary_path)
    verification = _read_json(verification_path)
    if summary.get("status") != "complete":
        raise ValueError(f"{root}: VDN training is incomplete")
    if verification.get("verified") is not True:
        raise ValueError(f"{root}: VDN verification failed or is missing")
    checkpoint_hash = sha256_file(checkpoint_path)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"{root}: checkpoint hash disagrees with verification")
    if verification.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError(f"{root}: training summary changed after verification")

    signature = summary.get("signature") or {}
    seed = int(signature["seed"])
    evaluations: dict[str, Any] = {}
    for condition in CONDITIONS:
        prediction_path = root / "evaluations" / f"{condition}.jsonl"
        evaluation_path = (
            root / "evaluations" / f"{condition}.summary.json"
        )
        if not prediction_path.is_file() or not evaluation_path.is_file():
            raise FileNotFoundError(
                f"{root}: missing formal {condition} evaluation"
            )
        evaluation = _read_json(evaluation_path)
        evaluation_signature = evaluation.get("signature") or {}
        expected_evaluation_condition = (
            "clean" if condition == "rpm10k" else condition
        )
        if (
            evaluation.get("status") != "complete"
            or evaluation.get("protocol") != VDN_EVALUATION_PROTOCOL
            or evaluation.get("condition") != expected_evaluation_condition
        ):
            raise ValueError(f"{evaluation_path}: invalid formal evaluation")
        if evaluation_signature.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"{evaluation_path}: checkpoint identity mismatch")
        rows = _read_jsonl(prediction_path)
        if len(rows) != int((evaluation.get("metrics") or {}).get("samples", -1)):
            raise ValueError(f"{prediction_path}: row count mismatch")
        evaluations[condition] = {
            "summary": evaluation,
            "rows": rows,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "summary_path": str(evaluation_path),
            "summary_sha256": sha256_file(evaluation_path),
        }
    return {
        "root": str(root),
        "seed": seed,
        "summary": summary,
        "verification": verification,
        "checkpoint_sha256": checkpoint_hash,
        "evaluations": evaluations,
    }


def _load_final_condition(root: Path, condition: str) -> dict[str, Any]:
    condition_root = root.resolve() / condition
    prediction_path = condition_root / "predictions.jsonl"
    summary_path = condition_root / "summary.json"
    if not prediction_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{condition_root}: final evaluation is missing")
    summary = _read_json(summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("protocol") != FINAL_EVALUATION_PROTOCOL
        or summary.get("condition") != condition
    ):
        raise ValueError(f"{summary_path}: invalid final evaluation")
    prediction_hash = sha256_file(prediction_path)
    if summary.get("predictions_sha256") != prediction_hash:
        raise ValueError(f"{prediction_path}: hash disagrees with final summary")
    rows = _read_jsonl(prediction_path)
    if len(rows) != int(summary.get("samples", -1)):
        raise ValueError(f"{prediction_path}: row count mismatch")
    return {
        "summary": summary,
        "rows": rows,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_hash,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }


def _aggregate_metric(
    runs: Sequence[dict[str, Any]],
    condition: str,
    section: str,
    metric: str,
) -> dict[str, float]:
    return _mean_std(
        [
            float(
                run["evaluations"][condition]["summary"][section][metric]
            )
            for run in runs
        ]
    )


def _aggregate_optional_metric(
    runs: Sequence[dict[str, Any]],
    condition: str,
    section: str,
    metric: str,
) -> dict[str, float] | None:
    sections = [
        run["evaluations"][condition]["summary"].get(section)
        for run in runs
    ]
    if all(value is None for value in sections):
        return None
    if any(not isinstance(value, dict) or metric not in value for value in sections):
        raise ValueError(
            f"{condition}: inconsistent optional {section}.{metric}"
        )
    return _mean_std([float(value[metric]) for value in sections])


def build_summary(
    run_roots: Sequence[Path],
    final_root: Path,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(run_roots) < 2:
        raise ValueError("replicate summary requires at least two VDN runs")
    runs = [_load_vdn_run(root) for root in run_roots]
    seeds = [int(run["seed"]) for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("VDN replicate seeds must be unique")

    conditions: dict[str, Any] = {}
    for offset, condition in enumerate(CONDITIONS):
        final = _load_final_condition(final_root, condition)
        final_metrics = final["summary"]["metrics"][
            "calibrated_progress_router"
        ]
        paired = paired_final_vs_vdn_replicates(
            [
                run["evaluations"][condition]["rows"]
                for run in runs
            ],
            final["rows"],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + offset,
        )
        if not math.isclose(
            paired["final_nmae_recomputed"],
            float(final_metrics["nmae"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{condition}: final NMAE failed recomputation")
        conditions[condition] = {
            "samples": int(final_metrics["samples"]),
            "vdn": {
                "nmae": _aggregate_metric(
                    runs, condition, "metrics", "nmae"
                ),
                "acc_1pct": _aggregate_metric(
                    runs, condition, "metrics", "acc_1pct"
                ),
                "acc_2pct": _aggregate_metric(
                    runs, condition, "metrics", "acc_2pct"
                ),
                "coverage": _aggregate_metric(
                    runs, condition, "metrics", "coverage"
                ),
                "direction_angle_mae_degrees": _aggregate_optional_metric(
                    runs,
                    condition,
                    "direction_component",
                    "angle_mae_degrees_success_only",
                ),
                "direction_acc_3deg": _aggregate_optional_metric(
                    runs,
                    condition,
                    "direction_component",
                    "angle_acc_3deg_all",
                ),
            },
            "final": {
                key: final_metrics[key]
                for key in (
                    "nmae",
                    "acc_1pct",
                    "acc_2pct",
                    "acc_5pct",
                    "coverage",
                    "samples",
                    "successful",
                )
            },
            "paired_final_vs_vdn": paired,
            "sources": {
                "vdn_predictions": [
                    {
                        "seed": run["seed"],
                        "path": run["evaluations"][condition][
                            "prediction_path"
                        ],
                        "sha256": run["evaluations"][condition][
                            "prediction_sha256"
                        ],
                    }
                    for run in runs
                ],
                "final_prediction": {
                    "path": final["prediction_path"],
                    "sha256": final["prediction_sha256"],
                },
            },
        }

    validation_mae = _mean_std(
        [
            float(run["summary"]["best_validation_angle_mae_degrees"])
            for run in runs
        ]
    )
    return {
        "protocol": PROTOCOL,
        "status": "complete",
        "replicates": len(runs),
        "seeds": seeds,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": seed,
        "training": {
            "best_validation_angle_mae_degrees": validation_mae,
            "runs": [
                {
                    "seed": run["seed"],
                    "best_epoch": int(run["summary"]["best_epoch"]),
                    "best_validation_angle_mae_degrees": float(
                        run["summary"][
                            "best_validation_angle_mae_degrees"
                        ]
                    ),
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "run_root": run["root"],
                }
                for run in runs
            ],
        },
        "conditions": conditions,
        "interpretation": (
            "Run-to-run values are mean ± sample standard deviation. "
            "Paired intervals compare the frozen final method against the "
            "per-sample mean VDN error, not an ensemble prediction."
        ),
        "source_sha256": {
            "script": sha256_file(Path(__file__).resolve()),
        },
    }


def _fmt_mean_std(value: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{float(value['mean']):.{digits}f} ± "
        f"{float(value['std']):.{digits}f}"
    )


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VDN three-seed end-to-end summary",
        "",
        (
            "All VDN runs use the pinned architecture and the same SyncG "
            "training protocol. Values are mean ± sample standard deviation "
            "over independent training seeds."
        ),
        "",
        "## Training stability",
        "",
        "| Seed | Best epoch | Validation direction MAE (degrees) |",
        "|---:|---:|---:|",
    ]
    for run in payload["training"]["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | "
            f"{run['best_validation_angle_mae_degrees']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Validation direction MAE: "
            + _fmt_mean_std(
                payload["training"][
                    "best_validation_angle_mae_degrees"
                ]
            )
            + " degrees.",
            "",
            "## Frozen end-to-end comparison",
            "",
            (
                "| Condition | N | VDN NMAE | VDN Acc@2% | VDN coverage | "
                "Final NMAE | Final Acc@2% | Final-VDN Delta NMAE (95% CI) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        result = payload["conditions"][condition]
        vdn = result["vdn"]
        final = result["final"]
        paired = result["paired_final_vs_vdn"]
        interval = paired["delta_nmae_group_bootstrap_95ci"]
        interval_text = (
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
            if interval is not None
            else "n/a"
        )
        lines.append(
            f"| {condition} | {result['samples']} | "
            f"{_fmt_mean_std(vdn['nmae'])} | "
            f"{_fmt_mean_std(vdn['acc_2pct'])} | "
            f"{_fmt_mean_std(vdn['coverage'])} | "
            f"{float(final['nmae']):.4f} | "
            f"{float(final['acc_2pct']):.4f} | "
            f"{paired['delta_nmae_final_minus_vdn']:+.4f} "
            f"{interval_text} |"
        )
    lines.extend(
        [
            "",
            (
                "The paired comparison averages the three VDN errors for each "
                "image before grouped bootstrap. It therefore estimates an "
                "independent VDN training run and is not a prediction ensemble."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        help="audited VDN run root; repeat for each seed",
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=DEFAULT_FINAL_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/vdn_syncg/vdn_replicate_summary.json"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    run_roots = (
        tuple(args.run_root)
        if args.run_root
        else DEFAULT_RUN_ROOTS
    )
    payload = build_summary(
        run_roots,
        args.final_root,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(
        render_markdown(payload) + "\n",
        encoding="utf-8",
    )
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
