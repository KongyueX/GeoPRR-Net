"""Audit and aggregate three pivot-direction training/evaluation seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.pivot_direction_fallback import PIVOT_DIRECTION_PROTOCOL
from experiments.vdn_baseline import (
    group_bootstrap_ci,
    normalized_error,
    sha256_file,
    summarize_scalar_predictions,
)


EVALUATION_PROTOCOL = "pivot_direction_fallback_e2e_v1"
PROTOCOL = "pivot_direction_three_seed_stability_v1"
CONDITIONS = ("clean", "rpm10k")
METRICS = ("nmae", "acc_2pct", "coverage", "successful_nmae")
RPM_ENVIRONMENT_SUBGROUPS = {
    "blur": lambda values: "blur" in values,
    "tilted": lambda values: "tilted" in values,
    "blur_and_tilted": lambda values: "blur" in values and "tilted" in values,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/runs/pivot_direction_syncg"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260720, 20260721, 20260722],
    )
    parser.add_argument(
        "--vdn-root",
        type=Path,
        default=Path("artifacts/runs/vdn_syncg/seed_20260720"),
    )
    parser.add_argument(
        "--internal-robustness-root",
        type=Path,
        default=Path("artifacts/runs/robustness"),
    )
    parser.add_argument(
        "--internal-rpm-root",
        type=Path,
        default=Path("artifacts/runs/rpm10k_single_pointer_zero_shot"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _prediction_map(
    rows: Sequence[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    result = {str(row.get("sample_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return result


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    payload = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def route_prediction(
    base_prediction: float | None,
    fallback_prediction: float | None,
) -> tuple[float | None, str]:
    """Apply the predeclared, label-free hard-failure route."""
    if base_prediction is not None:
        return float(base_prediction), "base"
    if fallback_prediction is not None:
        return float(fallback_prediction), "fallback"
    return None, "unresolved"


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("replicate statistic is empty or non-finite")
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _method_summary(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    result = summarize_scalar_predictions(
        rows,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    successful_errors = [
        normalized_error(
            row.get("prediction"),
            row["ground_truth"],
            row["scale_start"],
            row["scale_end"],
        )
        for row in rows
        if row.get("prediction") is not None
    ]
    result["successful_nmae"] = (
        float(np.mean(successful_errors)) if successful_errors else None
    )
    return result


def _paired_delta(
    candidate_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    if len(candidate_rows) != len(baseline_rows):
        raise ValueError("paired methods have different sample counts")
    deltas: list[float] = []
    groups: list[str] = []
    candidate_better: list[bool] = []
    for candidate, baseline in zip(candidate_rows, baseline_rows, strict=True):
        if candidate["sample_id"] != baseline["sample_id"]:
            raise ValueError("paired methods have different sample order")
        truth = float(candidate["ground_truth"])
        start = float(candidate["scale_start"])
        end = float(candidate["scale_end"])
        candidate_error = normalized_error(
            candidate.get("prediction"), truth, start, end
        )
        baseline_error = normalized_error(
            baseline.get("prediction"), truth, start, end
        )
        deltas.append(candidate_error - baseline_error)
        groups.append(str(candidate.get("group_id") or candidate["sample_id"]))
        candidate_better.append(candidate_error < baseline_error)
    return {
        "paired_samples": len(deltas),
        "delta_nmae": float(np.mean(deltas)),
        "delta_nmae_group_bootstrap_95ci": group_bootstrap_ci(
            deltas,
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
        "candidate_better_rate": float(np.mean(candidate_better)),
    }


def _row(
    source: dict[str, Any], prediction: float | None
) -> dict[str, Any]:
    return {
        "sample_id": str(source["sample_id"]),
        "group_id": str(
            source.get("group_id")
            or source.get("meter_id")
            or source["sample_id"]
        ),
        "meter_id": source.get("meter_id"),
        "ground_truth": float(source["ground_truth"]),
        "scale_start": float(source["scale_start"]),
        "scale_end": float(source["scale_end"]),
        "prediction": prediction,
        "metadata": source.get("metadata") or {},
    }


def _condition_paths(args: argparse.Namespace, condition: str) -> tuple[Path, Path]:
    if condition == "clean":
        return (
            args.internal_robustness_root / "clean" / "predictions.jsonl",
            args.vdn_root / "evaluations" / "clean.jsonl",
        )
    return (
        args.internal_rpm_root / "predictions.jsonl",
        args.vdn_root / "evaluations" / "rpm10k.jsonl",
    )


def _format(value: dict[str, Any], digits: int = 4) -> str:
    return f"{value['mean']:.{digits}f} ± {value['sample_std']:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    seed_count = len(payload["seeds"])
    lines = [
        "# Pivot-direction three-seed stability",
        "",
        f"Values are mean ± sample standard deviation over {seed_count} independently ",
        "trained SyncG seeds. The Base Ours and VDN predictions are frozen and shared.",
        "",
        "| Condition | Method | NMAE | Acc@2% | Coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        for method in ("Direction fallback", "Dual route"):
            metrics = aggregate[condition][method]
            lines.append(
                f"| {condition} | {method} | {_format(metrics['nmae'])} | "
                f"{_format(metrics['acc_2pct'])} | {_format(metrics['coverage'])} |"
            )
    training = aggregate["training"]
    clean_component = aggregate["clean_direction_component"]
    rpm_comparison = aggregate["rpm10k_comparison"]
    lines.extend(
        [
            "",
            "Training and component diagnostics:",
            "",
            f"- Validation direction MAE: {_format(training['validation_angle_mae_degrees'])}°",
            f"- Validation pivot error: {_format(training['validation_pivot_error_fraction'], 5)} of input width",
            f"- Clean direction MAE: {_format(clean_component['angle_mae_degrees'], 3)}°",
            f"- Clean Acc@3°: {_format(clean_component['angle_acc_3deg'])}",
            "",
            "RPM-10K paired deltas (negative NMAE is better):",
            "",
            f"- Dual − Base Ours: {_format(rpm_comparison['dual_minus_base_nmae'])}",
            f"- Dual − VDN: {_format(rpm_comparison['dual_minus_vdn_nmae'])}",
            "",
        ]
    )
    subgroup_aggregate = aggregate.get("rpm10k_environment_subgroups") or {}
    if subgroup_aggregate:
        lines.extend(
            [
                "RPM-10K environment subgroups:",
                "",
                "| Subgroup | Dual NMAE | VDN NMAE | Dual−VDN NMAE | Dual / VDN Acc@2% |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for subgroup_name in RPM_ENVIRONMENT_SUBGROUPS:
            subgroup = subgroup_aggregate[subgroup_name]
            lines.append(
                f"| {subgroup_name} | "
                f"{_format(subgroup['Dual route']['nmae'])} | "
                f"{_format(subgroup['VDN']['nmae'])} | "
                f"{_format(subgroup['dual_minus_vdn_nmae'])} | "
                f"{_format(subgroup['Dual route']['acc_2pct'])} / "
                f"{_format(subgroup['VDN']['acc_2pct'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    args.vdn_root = args.vdn_root.resolve()
    args.internal_robustness_root = args.internal_robustness_root.resolve()
    args.internal_rpm_root = args.internal_rpm_root.resolve()
    output = (args.output or (args.root / "replicate_stability.json")).resolve()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("at least two distinct seeds are required")
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")

    source_root = Path(__file__).resolve().parent
    expected_sources = {
        "model": sha256_file(source_root / "pivot_direction_fallback.py"),
        "evaluation": sha256_file(
            source_root / "evaluate_pivot_direction_fallback.py"
        ),
        "degradation": sha256_file(source_root / "robustness_degradations.py"),
    }
    expected_trainer_hash = sha256_file(source_root / "train_pivot_direction_syncg.py")
    expected_verifier_hash = sha256_file(source_root / "verify_pivot_direction_run.py")
    shared: dict[str, Any] = {}
    shared_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for condition in CONDITIONS:
        internal_path, vdn_path = _condition_paths(args, condition)
        internal_metrics_path = internal_path.with_name("metrics.json")
        vdn_summary_path = vdn_path.with_suffix(".summary.json")
        internal_metrics = _read_json(internal_metrics_path)
        vdn_summary = _read_json(vdn_summary_path)
        if internal_metrics.get("front_end_signature_verified") is not True:
            raise ValueError(f"{condition}: internal front-end signature is unverified")
        if vdn_summary.get("status") != "complete":
            raise ValueError(f"{condition}: VDN evaluation is incomplete")
        internal_rows = _read_jsonl(internal_path)
        vdn_rows = _read_jsonl(vdn_path)
        internal_map = _prediction_map(internal_rows, internal_path)
        vdn_map = _prediction_map(vdn_rows, vdn_path)
        if set(internal_map) != set(vdn_map):
            raise ValueError(f"{condition}: Base Ours and VDN sample IDs differ")
        ordered_ids = [str(row["sample_id"]) for row in internal_rows]
        base_method: list[dict[str, Any]] = []
        vdn_method: list[dict[str, Any]] = []
        for sample_id in ordered_ids:
            internal = internal_map[sample_id]
            vdn = vdn_map[sample_id]
            metadata = (
                float(internal["ground_truth"]),
                float(internal["scale_start"]),
                float(internal["scale_end"]),
            )
            if metadata != (
                float(vdn["ground_truth"]),
                float(vdn["scale_start"]),
                float(vdn["scale_end"]),
            ):
                raise ValueError(f"{condition}/{sample_id}: target metadata mismatch")
            base_method.append(
                _row(internal, (internal.get("predictions") or {}).get("ours"))
            )
            vdn_method.append(_row(internal, vdn.get("prediction")))
        shared[condition] = {
            "internal_predictions": str(internal_path),
            "internal_predictions_sha256": sha256_file(internal_path),
            "internal_metrics": str(internal_metrics_path),
            "internal_metrics_sha256": sha256_file(internal_metrics_path),
            "vdn_predictions": str(vdn_path),
            "vdn_predictions_sha256": sha256_file(vdn_path),
            "vdn_summary": str(vdn_summary_path),
            "vdn_summary_sha256": sha256_file(vdn_summary_path),
            "sample_ids_sha256": _sample_ids_sha256(ordered_ids),
            "samples": len(ordered_ids),
            "Base Ours": _method_summary(
                base_method,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
            "VDN": _method_summary(
                vdn_method,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed,
            ),
        }
        shared_rows[condition] = {
            "base": base_method,
            "vdn": vdn_method,
        }

    raw: dict[str, Any] = {}
    first_eval_identity: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        seed_root = args.root / f"seed_{seed}"
        summary_path = seed_root / "summary.json"
        verification_path = seed_root / "verification.json"
        checkpoint_path = seed_root / "best.pt"
        summary = _read_json(summary_path)
        verification = _read_json(verification_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        signature = summary.get("signature") or {}
        if summary.get("status") != "complete":
            raise ValueError(f"seed {seed}: training is incomplete")
        if signature.get("protocol") != PIVOT_DIRECTION_PROTOCOL:
            raise ValueError(f"seed {seed}: training protocol mismatch")
        if int(signature.get("seed", -1)) != seed:
            raise ValueError(f"seed {seed}: training signature seed mismatch")
        if signature.get("model_source_sha256") != expected_sources["model"]:
            raise ValueError(f"seed {seed}: model source signature changed")
        if signature.get("trainer_source_sha256") != expected_trainer_hash:
            raise ValueError(f"seed {seed}: trainer source signature changed")
        if summary.get("best_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"seed {seed}: checkpoint hash mismatch")
        if verification.get("verified") is not True:
            raise ValueError(f"seed {seed}: verification failed")
        if verification.get("best_checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"seed {seed}: verification checkpoint mismatch")
        if verification.get("verifier_source_sha256") != expected_verifier_hash:
            raise ValueError(f"seed {seed}: verifier source signature changed")

        seed_value: dict[str, Any] = {
            "run_dir": str(seed_root),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "verification": str(verification_path),
            "verification_sha256": sha256_file(verification_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "training": {
                "best_epoch": int(summary["best_epoch"]),
                "validation_angle_mae_degrees": float(
                    summary["best_validation_angle_mae_degrees"]
                ),
                "validation_pivot_error_fraction": float(
                    summary["best_validation_pivot_error_fraction"]
                ),
                "optimizer_steps": int(verification["optimizer_steps"]),
                "skipped_optimizer_steps": int(
                    verification["skipped_optimizer_steps"]
                ),
                "group_overlap": int(verification["group_overlap"]),
            },
            "conditions": {},
        }
        for condition in CONDITIONS:
            prediction_path = seed_root / "evaluations" / f"{condition}.jsonl"
            eval_summary_path = prediction_path.with_suffix(".summary.json")
            eval_summary = _read_json(eval_summary_path)
            eval_signature = eval_summary.get("signature") or {}
            if eval_summary.get("status") != "complete":
                raise ValueError(f"seed {seed}/{condition}: evaluation incomplete")
            if eval_summary.get("protocol") != EVALUATION_PROTOCOL:
                raise ValueError(f"seed {seed}/{condition}: evaluation protocol mismatch")
            if eval_summary.get("output_sha256") != sha256_file(prediction_path):
                raise ValueError(f"seed {seed}/{condition}: output hash mismatch")
            if eval_signature.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError(f"seed {seed}/{condition}: checkpoint mismatch")
            if eval_signature.get("training_protocol") != PIVOT_DIRECTION_PROTOCOL:
                raise ValueError(f"seed {seed}/{condition}: training protocol mismatch")
            if eval_signature.get("checkpoint_training_signature") != signature:
                raise ValueError(
                    f"seed {seed}/{condition}: checkpoint training signature mismatch"
                )
            if eval_signature.get("verification_sha256") != sha256_file(
                verification_path
            ):
                raise ValueError(f"seed {seed}/{condition}: verification hash mismatch")
            if eval_signature.get("source_sha256") != expected_sources:
                raise ValueError(f"seed {seed}/{condition}: source signature changed")
            if eval_signature.get("reference_predictions_sha256") != shared[condition][
                "vdn_predictions_sha256"
            ]:
                raise ValueError(
                    f"seed {seed}/{condition}: frozen VDN/front-end hash mismatch"
                )

            fallback_rows = _read_jsonl(prediction_path)
            fallback_map = _prediction_map(fallback_rows, prediction_path)
            base_rows = shared_rows[condition]["base"]
            vdn_rows = shared_rows[condition]["vdn"]
            ids = [str(row["sample_id"]) for row in base_rows]
            if set(ids) != set(fallback_map):
                raise ValueError(f"seed {seed}/{condition}: sample IDs differ")
            evaluation_identity = {
                "manifest_sha256": eval_signature.get("manifest_sha256"),
                "reference_predictions_sha256": eval_signature.get(
                    "reference_predictions_sha256"
                ),
                "degradation_seed": eval_signature.get("degradation_seed"),
                "condition": eval_signature.get("condition"),
            }
            expected_identity = first_eval_identity.setdefault(
                condition, evaluation_identity
            )
            if evaluation_identity != expected_identity:
                raise ValueError(
                    f"seed {seed}/{condition}: evaluation protocol inputs differ"
                )

            direction_rows: list[dict[str, Any]] = []
            dual_rows: list[dict[str, Any]] = []
            routing = {"base": 0, "fallback": 0, "unresolved": 0}
            for base in base_rows:
                sample_id = str(base["sample_id"])
                fallback = fallback_map[sample_id]
                metadata = (
                    float(base["ground_truth"]),
                    float(base["scale_start"]),
                    float(base["scale_end"]),
                )
                if metadata != (
                    float(fallback["ground_truth"]),
                    float(fallback["scale_start"]),
                    float(fallback["scale_end"]),
                ):
                    raise ValueError(
                        f"seed {seed}/{condition}/{sample_id}: target metadata mismatch"
                    )
                fallback_prediction = fallback.get("prediction")
                dual_prediction, route = route_prediction(
                    base.get("prediction"), fallback_prediction
                )
                routing[route] += 1
                direction_rows.append(_row(base, fallback_prediction))
                dual_rows.append(_row(base, dual_prediction))

            method_metrics = {
                "Direction fallback": _method_summary(
                    direction_rows,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + seed,
                ),
                "Dual route": _method_summary(
                    dual_rows,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + seed,
                ),
            }
            comparisons = {
                "Dual vs Base Ours": _paired_delta(
                    dual_rows,
                    base_rows,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + seed,
                ),
                "Dual vs VDN": _paired_delta(
                    dual_rows,
                    vdn_rows,
                    bootstrap_iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + seed,
                ),
            }
            environment_subgroups: dict[str, Any] = {}
            if condition == "rpm10k":
                for subgroup_name, member in RPM_ENVIRONMENT_SUBGROUPS.items():
                    indices = [
                        index
                        for index, base in enumerate(base_rows)
                        if member(
                            set(
                                (base.get("metadata") or {}).get(
                                    "environment_conditions"
                                )
                                or []
                            )
                        )
                    ]
                    subgroup_methods = {
                        "Base Ours": [base_rows[index] for index in indices],
                        "Direction fallback": [
                            direction_rows[index] for index in indices
                        ],
                        "Dual route": [dual_rows[index] for index in indices],
                        "VDN": [vdn_rows[index] for index in indices],
                    }
                    environment_subgroups[subgroup_name] = {
                        "samples": len(indices),
                        "overlaps_other_environment_subgroups": True,
                        "metrics": {
                            method: _method_summary(
                                rows,
                                bootstrap_iterations=args.bootstrap_iterations,
                                seed=args.bootstrap_seed + seed,
                            )
                            for method, rows in subgroup_methods.items()
                        },
                        "comparisons": {
                            "Dual vs Base Ours": _paired_delta(
                                subgroup_methods["Dual route"],
                                subgroup_methods["Base Ours"],
                                bootstrap_iterations=args.bootstrap_iterations,
                                seed=args.bootstrap_seed + seed,
                            ),
                            "Dual vs VDN": _paired_delta(
                                subgroup_methods["Dual route"],
                                subgroup_methods["VDN"],
                                bootstrap_iterations=args.bootstrap_iterations,
                                seed=args.bootstrap_seed + seed,
                            ),
                        },
                    }
            direction_component = eval_summary.get("direction_component")
            seed_value["conditions"][condition] = {
                "predictions": str(prediction_path),
                "predictions_sha256": sha256_file(prediction_path),
                "evaluation_summary": str(eval_summary_path),
                "evaluation_summary_sha256": sha256_file(eval_summary_path),
                "evaluation_identity": evaluation_identity,
                "routing": {
                    **routing,
                    "fallback_requested": routing["fallback"] + routing["unresolved"],
                    "fallback_recovery_rate": routing["fallback"]
                    / max(routing["fallback"] + routing["unresolved"], 1),
                },
                "metrics": method_metrics,
                "comparisons": comparisons,
                "environment_subgroups": environment_subgroups,
                "direction_component": direction_component,
            }
        raw[str(seed)] = seed_value

    aggregate: dict[str, Any] = {
        "training": {
            metric: _stats(
                [float(raw[str(seed)]["training"][metric]) for seed in args.seeds]
            )
            for metric in (
                "validation_angle_mae_degrees",
                "validation_pivot_error_fraction",
            )
        },
        "clean_direction_component": {},
        "rpm10k_comparison": {},
        "rpm10k_environment_subgroups": {},
    }
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for method in ("Direction fallback", "Dual route"):
            aggregate[condition][method] = {}
            for metric in METRICS:
                values = [
                    raw[str(seed)]["conditions"][condition]["metrics"][method].get(
                        metric
                    )
                    for seed in args.seeds
                ]
                finite = [float(value) for value in values if value is not None]
                aggregate[condition][method][metric] = (
                    _stats(finite) if len(finite) == len(values) else None
                )
    for output_name, source_name in (
        ("angle_mae_degrees", "angle_mae_degrees_success_only"),
        ("angle_acc_3deg", "angle_acc_3deg_all"),
    ):
        aggregate["clean_direction_component"][output_name] = _stats(
            [
                float(
                    raw[str(seed)]["conditions"]["clean"]["direction_component"][
                        source_name
                    ]
                )
                for seed in args.seeds
            ]
        )
    for output_name, comparison in (
        ("dual_minus_base_nmae", "Dual vs Base Ours"),
        ("dual_minus_vdn_nmae", "Dual vs VDN"),
    ):
        aggregate["rpm10k_comparison"][output_name] = _stats(
            [
                float(
                    raw[str(seed)]["conditions"]["rpm10k"]["comparisons"][
                        comparison
                    ]["delta_nmae"]
                )
                for seed in args.seeds
            ]
        )
    for subgroup_name in RPM_ENVIRONMENT_SUBGROUPS:
        subgroup_aggregate: dict[str, Any] = {}
        for method in ("Base Ours", "Direction fallback", "Dual route", "VDN"):
            subgroup_aggregate[method] = {
                metric: _stats(
                    [
                        float(
                            raw[str(seed)]["conditions"]["rpm10k"][
                                "environment_subgroups"
                            ][subgroup_name]["metrics"][method][metric]
                        )
                        for seed in args.seeds
                    ]
                )
                for metric in ("nmae", "acc_2pct", "coverage")
            }
        subgroup_aggregate["dual_minus_vdn_nmae"] = _stats(
            [
                float(
                    raw[str(seed)]["conditions"]["rpm10k"][
                        "environment_subgroups"
                    ][subgroup_name]["comparisons"]["Dual vs VDN"]["delta_nmae"]
                )
                for seed in args.seeds
            ]
        )
        aggregate["rpm10k_environment_subgroups"][subgroup_name] = (
            subgroup_aggregate
        )

    payload = {
        "protocol": PROTOCOL,
        "seeds": args.seeds,
        "training_protocol": PIVOT_DIRECTION_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "route_policy": "base Ours if available; pivot-direction fallback only on failure",
        "uses_target_for_routing": False,
        "confidence_threshold": None,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "shared_frozen_predictions": shared,
        "evaluation_identity": first_eval_identity,
        "dependency_sha256": {
            **expected_sources,
            "trainer": expected_trainer_hash,
            "verifier": expected_verifier_hash,
        },
        "raw": raw,
        "aggregate": aggregate,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, payload)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(output)
    print(markdown_path)


if __name__ == "__main__":
    main()
