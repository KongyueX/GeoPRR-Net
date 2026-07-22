"""Aggregate failure-triggered mask/vector routing across all frozen tests."""
from __future__ import annotations

import argparse
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


PROTOCOL = "failure_triggered_dual_representation_route_v1"
SYNCG_CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
ALL_CONDITIONS = SYNCG_CONDITIONS + ("rpm10k",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fallback-root",
        type=Path,
        default=Path("artifacts/runs/pivot_direction_syncg/seed_20260722"),
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
    parser.add_argument("--seed", type=int, default=20260722)
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


def _paths(args: argparse.Namespace, condition: str) -> dict[str, Path]:
    fallback_name = "rpm10k" if condition == "rpm10k" else condition
    vdn_name = fallback_name
    internal_root = (
        args.internal_rpm_root
        if condition == "rpm10k"
        else args.internal_robustness_root / condition
    )
    return {
        "internal_predictions": internal_root / "predictions.jsonl",
        "internal_metrics": internal_root / "metrics.json",
        "fallback_predictions": args.fallback_root / "evaluations" / f"{fallback_name}.jsonl",
        "fallback_summary": args.fallback_root
        / "evaluations"
        / f"{fallback_name}.summary.json",
        "vdn_predictions": args.vdn_root / "evaluations" / f"{vdn_name}.jsonl",
        "vdn_summary": args.vdn_root / "evaluations" / f"{vdn_name}.summary.json",
    }


def _prediction_map(rows: Sequence[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result = {str(row.get("sample_id")): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return result


def route_prediction(
    base_prediction: float | None,
    fallback_prediction: float | None,
) -> tuple[float | None, str]:
    """Apply the predeclared label-free hard-failure routing policy."""
    if base_prediction is not None:
        return float(base_prediction), "base"
    if fallback_prediction is not None:
        return float(fallback_prediction), "fallback"
    return None, "unresolved"


def _method_summary(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    metrics = summarize_scalar_predictions(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    successful_errors = [
        normalized_error(
            row["prediction"],
            row["ground_truth"],
            row["scale_start"],
            row["scale_end"],
        )
        for row in rows
        if row.get("prediction") is not None
    ]
    result = {
        key: metrics.get(key)
        for key in (
            "samples",
            "successful",
            "coverage",
            "nmae",
            "nmae_group_bootstrap_95ci",
            "successful_nmae",
            "acc_1pct",
            "acc_2pct",
        )
    }
    result["successful_nmae"] = (
        float(np.mean(successful_errors)) if successful_errors else None
    )
    return result


def _paired(
    candidate_errors: Sequence[float],
    baseline_errors: Sequence[float],
    groups: Sequence[str],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    delta = candidate - baseline
    return {
        "paired_samples": int(delta.size),
        "delta_nmae": float(np.mean(delta)),
        "delta_nmae_group_bootstrap_95ci": group_bootstrap_ci(
            delta,
            groups,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "candidate_better_rate": float(np.mean(candidate < baseline)),
        "ties_rate": float(np.mean(candidate == baseline)),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Failure-triggered dual-representation routing",
        "",
        "The route uses the mask/residual method whenever it returns a reading and ",
        "invokes the independent pivot-direction head only on a hard front-end failure.",
        "",
        "| Condition | Base NMAE | Direction NMAE | Dual NMAE | VDN NMAE | Base / Dual / VDN coverage | Dual−VDN NMAE (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ALL_CONDITIONS:
        result = payload["conditions"][name]
        methods = result["metrics"]
        paired = result["paired"]["Dual vs VDN"]
        interval = paired["delta_nmae_group_bootstrap_95ci"]
        interval_text = (
            f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "—"
        )
        lines.append(
            f"| {name} | {methods['Base Ours']['nmae']:.4f} | "
            f"{methods['Direction fallback']['nmae']:.4f} | "
            f"{methods['Dual route']['nmae']:.4f} | {methods['VDN']['nmae']:.4f} | "
            f"{methods['Base Ours']['coverage']:.4f} / "
            f"{methods['Dual route']['coverage']:.4f} / {methods['VDN']['coverage']:.4f} | "
            f"{paired['delta_nmae']:.4f} {interval_text} |"
        )
    rpm_subgroups = payload["conditions"]["rpm10k"].get("environment_subgroups") or {}
    if rpm_subgroups:
        lines.extend(
            [
                "",
                "## RPM-10K environment subgroups",
                "",
                "Environment tags are used only for frozen reporting, never for routing.",
                "",
                "| Subgroup | N | Base NMAE | Direction NMAE | Dual NMAE | VDN NMAE | Dual / VDN Acc@2% | Dual−VDN NMAE (95% CI) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for subgroup_name in ("blur", "tilted", "blur_and_tilted"):
            subgroup = rpm_subgroups[subgroup_name]
            metrics = subgroup["metrics"]
            paired = subgroup["paired"]["Dual vs VDN"]
            interval = paired["delta_nmae_group_bootstrap_95ci"]
            interval_text = (
                f"[{interval[0]:.4f}, {interval[1]:.4f}]" if interval else "—"
            )
            lines.append(
                f"| {subgroup_name} | {subgroup['samples']} | "
                f"{metrics['Base Ours']['nmae']:.4f} | "
                f"{metrics['Direction fallback']['nmae']:.4f} | "
                f"{metrics['Dual route']['nmae']:.4f} | "
                f"{metrics['VDN']['nmae']:.4f} | "
                f"{metrics['Dual route']['acc_2pct']:.4f} / "
                f"{metrics['VDN']['acc_2pct']:.4f} | "
                f"{paired['delta_nmae']:.4f} {interval_text} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.fallback_root = args.fallback_root.resolve()
    args.vdn_root = args.vdn_root.resolve()
    args.internal_robustness_root = args.internal_robustness_root.resolve()
    args.internal_rpm_root = args.internal_rpm_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else args.fallback_root / "dual_route_comparison.json"
    )
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    verification_path = args.fallback_root / "verification.json"
    verification = _read_json(verification_path)
    if verification.get("verified") is not True:
        raise ValueError("fallback training verification is missing or failed")
    checkpoint_path = args.fallback_root / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if verification.get("best_checkpoint_sha256") != checkpoint_hash:
        raise ValueError("fallback verification belongs to another checkpoint")

    conditions: dict[str, Any] = {}
    expected_fallback_sources = {
        "model": sha256_file(Path(__file__).resolve().with_name("pivot_direction_fallback.py")),
        "evaluation": sha256_file(
            Path(__file__).resolve().with_name("evaluate_pivot_direction_fallback.py")
        ),
        "degradation": sha256_file(
            Path(__file__).resolve().with_name("robustness_degradations.py")
        ),
    }
    for condition in ALL_CONDITIONS:
        paths = _paths(args, condition)
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        internal_metrics = _read_json(paths["internal_metrics"])
        if internal_metrics.get("front_end_signature_verified") is not True:
            raise ValueError(f"{condition}: internal front-end signature is unverified")
        fallback_summary = _read_json(paths["fallback_summary"])
        fallback_signature = fallback_summary.get("signature") or {}
        if fallback_summary.get("status") != "complete":
            raise ValueError(f"{condition}: fallback evaluation is incomplete")
        if fallback_signature.get("training_protocol") != PIVOT_DIRECTION_PROTOCOL:
            raise ValueError(f"{condition}: fallback training protocol mismatch")
        if fallback_signature.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"{condition}: fallback checkpoint mismatch")
        if fallback_signature.get("source_sha256") != expected_fallback_sources:
            raise ValueError(f"{condition}: fallback source signature changed")
        if fallback_summary.get("output_sha256") != sha256_file(
            paths["fallback_predictions"]
        ):
            raise ValueError(f"{condition}: fallback output hash mismatch")
        vdn_summary = _read_json(paths["vdn_summary"])
        if vdn_summary.get("status") != "complete":
            raise ValueError(f"{condition}: VDN evaluation is incomplete")

        internal_rows = _read_jsonl(paths["internal_predictions"])
        fallback_rows = _read_jsonl(paths["fallback_predictions"])
        vdn_rows = _read_jsonl(paths["vdn_predictions"])
        fallback_map = _prediction_map(fallback_rows, paths["fallback_predictions"])
        vdn_map = _prediction_map(vdn_rows, paths["vdn_predictions"])
        ids = [str(row.get("sample_id")) for row in internal_rows]
        if len(ids) != len(set(ids)) or set(ids) != set(fallback_map) or set(ids) != set(vdn_map):
            raise ValueError(f"{condition}: method sample IDs are not identical")

        method_rows: dict[str, list[dict[str, Any]]] = {
            "Base Ours": [],
            "Direction fallback": [],
            "Dual route": [],
            "VDN": [],
        }
        errors: dict[str, list[float]] = {name: [] for name in method_rows}
        groups: list[str] = []
        route_counts = {"base": 0, "fallback": 0, "unresolved": 0}
        fallback_recovered_errors: list[float] = []
        for internal in internal_rows:
            sample_id = str(internal.get("sample_id"))
            fallback = fallback_map[sample_id]
            vdn = vdn_map[sample_id]
            truth = float(internal["ground_truth"])
            start = float(internal["scale_start"])
            end = float(internal["scale_end"])
            for external in (fallback, vdn):
                if (
                    float(external["ground_truth"]) != truth
                    or float(external["scale_start"]) != start
                    or float(external["scale_end"]) != end
                ):
                    raise ValueError(f"{condition}/{sample_id}: target metadata mismatch")
            base_prediction = (internal.get("predictions") or {}).get("ours")
            fallback_prediction = fallback.get("prediction")
            vdn_prediction = vdn.get("prediction")
            dual_prediction, route_source = route_prediction(
                base_prediction,
                fallback_prediction,
            )
            route_counts[route_source] += 1
            if route_source == "fallback":
                fallback_recovered_errors.append(
                    normalized_error(fallback_prediction, truth, start, end)
                )
            predictions = {
                "Base Ours": base_prediction,
                "Direction fallback": fallback_prediction,
                "Dual route": dual_prediction,
                "VDN": vdn_prediction,
            }
            group = str(internal.get("group_id") or internal.get("meter_id") or sample_id)
            groups.append(group)
            for method, prediction in predictions.items():
                method_rows[method].append(
                    {
                        "sample_id": sample_id,
                        "group_id": group,
                        "meter_id": internal.get("meter_id"),
                        "ground_truth": truth,
                        "scale_start": start,
                        "scale_end": end,
                        "prediction": prediction,
                    }
                )
                errors[method].append(normalized_error(prediction, truth, start, end))
        metrics = {
            method: _method_summary(rows, args) for method, rows in method_rows.items()
        }
        paired = {
            "Dual vs Base Ours": _paired(
                errors["Dual route"], errors["Base Ours"], groups, args=args
            ),
            "Dual vs Direction fallback": _paired(
                errors["Dual route"], errors["Direction fallback"], groups, args=args
            ),
            "Dual vs VDN": _paired(
                errors["Dual route"], errors["VDN"], groups, args=args
            ),
            "Direction fallback vs VDN": _paired(
                errors["Direction fallback"], errors["VDN"], groups, args=args
            ),
        }
        environment_subgroups: dict[str, Any] = {}
        if condition == "rpm10k":
            subgroup_memberships = {
                "blur": lambda values: "blur" in values,
                "tilted": lambda values: "tilted" in values,
                "blur_and_tilted": lambda values: (
                    "blur" in values and "tilted" in values
                ),
            }
            for subgroup_name, member in subgroup_memberships.items():
                indices = [
                    index
                    for index, internal in enumerate(internal_rows)
                    if member(
                        set(
                            (internal.get("metadata") or {}).get(
                                "environment_conditions"
                            )
                            or []
                        )
                    )
                ]
                subgroup_method_rows = {
                    method: [rows[index] for index in indices]
                    for method, rows in method_rows.items()
                }
                subgroup_errors = {
                    method: [values[index] for index in indices]
                    for method, values in errors.items()
                }
                subgroup_groups = [groups[index] for index in indices]
                environment_subgroups[subgroup_name] = {
                    "samples": len(indices),
                    "overlaps_other_environment_subgroups": True,
                    "metrics": {
                        method: _method_summary(rows, args)
                        for method, rows in subgroup_method_rows.items()
                    },
                    "paired": {
                        "Dual vs Base Ours": _paired(
                            subgroup_errors["Dual route"],
                            subgroup_errors["Base Ours"],
                            subgroup_groups,
                            args=args,
                        ),
                        "Dual vs VDN": _paired(
                            subgroup_errors["Dual route"],
                            subgroup_errors["VDN"],
                            subgroup_groups,
                            args=args,
                        ),
                    },
                }
        conditions[condition] = {
            "metrics": metrics,
            "paired": paired,
            "routing": {
                **route_counts,
                "fallback_requested": route_counts["fallback"] + route_counts["unresolved"],
                "fallback_recovery_rate": route_counts["fallback"]
                / max(route_counts["fallback"] + route_counts["unresolved"], 1),
                "fallback_recovered_nmae": (
                    float(np.mean(fallback_recovered_errors))
                    if fallback_recovered_errors
                    else None
                ),
            },
            "environment_subgroups": environment_subgroups,
            "sources": {name: str(path) for name, path in paths.items()},
            "source_sha256": {name: sha256_file(path) for name, path in paths.items()},
        }

    payload = {
        "protocol": PROTOCOL,
        "route_policy": "base Ours if available; pivot-direction fallback only on failure",
        "uses_target_for_routing": False,
        "confidence_threshold": None,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "fallback_training_verification": str(verification_path),
        "fallback_training_verification_sha256": sha256_file(verification_path),
        "fallback_checkpoint_sha256": checkpoint_hash,
        "conditions": conditions,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, payload)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)
    print(markdown_path)


if __name__ == "__main__":
    main()
