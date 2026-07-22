"""Build the external VDN comparison and paired paper-facing tables."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.vdn_baseline import (
    group_bootstrap_ci,
    normalized_error,
    sha256_file,
)


SYNCG_CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
ALL_CONDITIONS = SYNCG_CONDITIONS + ("rpm10k",)
INTERNAL_METHODS = (
    "Geometry-v1",
    "Original Transformer",
    "Quality-weighted Fusion",
    "Ours",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--seed", type=int, default=20260720)
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


def _vdn_paths(root: Path, condition: str) -> tuple[Path, Path]:
    prediction = root / "evaluations" / f"{condition}.jsonl"
    summary = prediction.with_name(prediction.stem + ".summary.json")
    return prediction, summary


def _internal_root(args: argparse.Namespace, condition: str) -> Path:
    return (
        args.internal_rpm_root
        if condition == "rpm10k"
        else args.internal_robustness_root / condition
    )


def _metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "samples",
            "successful",
            "coverage",
            "nmae",
            "nmae_group_bootstrap_95ci",
            "acc_1pct",
            "acc_2pct",
            "negative_transfer_rate",
            "correction_coverage",
            "dialbench_ref_successful",
            "dialbench_rel_successful",
            "dialbench_acc_epsilon_e2e",
            "dialbench_acc_theta_e2e",
        )
        if key in metrics
    }


def _error(row: dict[str, Any], prediction: Any) -> float:
    return normalized_error(
        prediction,
        float(row["ground_truth"]),
        float(row["scale_start"]),
        float(row["scale_end"]),
    )


def _paired_comparison(
    vdn_rows: Sequence[dict[str, Any]],
    internal_rows: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    vdn_by_id = {str(row.get("sample_id")): row for row in vdn_rows}
    internal_by_id = {str(row.get("sample_id")): row for row in internal_rows}
    if len(vdn_by_id) != len(vdn_rows) or len(internal_by_id) != len(internal_rows):
        raise ValueError("comparison inputs contain duplicate sample identifiers")
    if set(vdn_by_id) != set(internal_by_id):
        raise ValueError("VDN and internal predictions do not cover identical samples")

    deltas: list[float] = []
    groups: list[str] = []
    ours_errors: list[float] = []
    vdn_errors: list[float] = []
    common_successes = 0
    for sample_id in sorted(vdn_by_id):
        vdn = vdn_by_id[sample_id]
        internal = internal_by_id[sample_id]
        for field in ("ground_truth", "scale_start", "scale_end"):
            if not math.isclose(
                float(vdn[field]),
                float(internal[field]),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise ValueError(f"{sample_id}: mismatched {field}")
        vdn_prediction = vdn.get("prediction")
        ours_prediction = (internal.get("predictions") or {}).get("ours")
        vdn_error = _error(vdn, vdn_prediction)
        ours_error = _error(internal, ours_prediction)
        vdn_errors.append(vdn_error)
        ours_errors.append(ours_error)
        deltas.append(ours_error - vdn_error)
        groups.append(str(vdn.get("group_id") or vdn.get("meter_id") or sample_id))
        common_successes += vdn_prediction is not None and ours_prediction is not None
    delta_array = np.asarray(deltas, dtype=np.float64)
    return {
        "paired_samples": len(deltas),
        "common_successes": common_successes,
        "delta_nmae_ours_minus_vdn": float(np.mean(delta_array)),
        "delta_nmae_group_bootstrap_95ci": group_bootstrap_ci(
            deltas,
            groups,
            iterations=iterations,
            seed=seed,
        ),
        "ours_better_rate": float(np.mean(delta_array < 0.0)),
        "ties_rate": float(np.mean(delta_array == 0.0)),
        "raw_ours_nmae": float(np.mean(ours_errors)),
        "raw_vdn_nmae": float(np.mean(vdn_errors)),
    }


def _robustness_from_clean(conditions: dict[str, Any]) -> list[dict[str, Any]]:
    clean = conditions["clean"]
    rows = []
    for condition in SYNCG_CONDITIONS[1:]:
        current = conditions[condition]
        row: dict[str, Any] = {"condition": condition}
        for method in ("Original Transformer", "VDN", "Ours"):
            clean_nmae = float(clean["metrics"][method]["nmae"])
            current_nmae = float(current["metrics"][method]["nmae"])
            row[method] = {
                "clean_nmae": clean_nmae,
                "condition_nmae": current_nmae,
                "absolute_increase": current_nmae - clean_nmae,
                "relative_increase": (
                    current_nmae / clean_nmae - 1.0 if clean_nmae > 0.0 else None
                ),
            }
        rows.append(row)
    return rows


def _format(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# External VDN comparison",
        "",
        "VDN uses the pinned official architecture retrained only on SyncG train. ",
        "All methods share the frozen dial/reference and scale-conversion protocol.",
        "",
        "| Condition | N | Geometry NMAE | Transformer NMAE | VDN NMAE | "
        "Ours NMAE | Geometry Acc@2% | Transformer Acc@2% | VDN Acc@2% | "
        "Ours Acc@2% | "
        "Ours−VDN NMAE (95% CI) |",
        "|---|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---:|",
    ]
    for condition in ALL_CONDITIONS:
        value = payload["conditions"][condition]
        metrics = value["metrics"]
        paired = value["paired_ours_vs_vdn"]
        ci = paired["delta_nmae_group_bootstrap_95ci"]
        ci_text = (
            f"{paired['delta_nmae_ours_minus_vdn']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            if ci is not None
            else _format(paired["delta_nmae_ours_minus_vdn"])
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    condition,
                    str(metrics["VDN"]["samples"]),
                    _format(metrics["Geometry-v1"]["nmae"]),
                    _format(metrics["Original Transformer"]["nmae"]),
                    _format(metrics["VDN"]["nmae"]),
                    _format(metrics["Ours"]["nmae"]),
                    _format(metrics["Geometry-v1"]["acc_2pct"]),
                    _format(metrics["Original Transformer"]["acc_2pct"]),
                    _format(metrics["VDN"]["acc_2pct"]),
                    _format(metrics["Ours"]["acc_2pct"]),
                    ci_text,
                )
            )
            + " |"
        )
    training = payload["training"]
    lines.extend(
        [
            "",
            "VDN component and end-to-end coverage diagnostics:",
            "",
            "| Condition | VDN E2E coverage | Ours E2E coverage | "
            "VDN direction coverage | VDN direction MAE | VDN Acc@3° |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in ALL_CONDITIONS:
        value = payload["conditions"][condition]
        component = value.get("vdn_direction_component") or {}
        lines.append(
            "| "
            + " | ".join(
                (
                    condition,
                    _format(value["metrics"]["VDN"].get("coverage")),
                    _format(value["metrics"]["Ours"].get("coverage")),
                    _format(component.get("coverage")),
                    _format(component.get("angle_mae_degrees_success_only")),
                    _format(component.get("angle_acc_3deg_all")),
                )
            )
            + " |"
        )
    rpm = payload["conditions"]["rpm10k"]
    rpm_vdn = rpm.get("vdn_dialbench_metrics") or {}
    rpm_ours = rpm["metrics"]["Ours"]
    lines.extend(
        [
            "",
            "RPM-10K single-pointer diagnostics under the official Ref/Rel formulas:",
            "",
            "| Method | Ref (success only) | Rel (success only) | "
            "Accε Ref≤1% E2E | Accθ Rel<5% E2E | coverage |",
            "|---|---:|---:|---:|---:|---:|",
            "| VDN | "
            + " | ".join(
                (
                    _format(rpm_vdn.get("ref_successful")),
                    _format(rpm_vdn.get("rel_successful")),
                    _format(rpm_vdn.get("acc_epsilon_ref_le_1pct_e2e")),
                    _format(rpm_vdn.get("acc_theta_rel_lt_5pct_e2e")),
                    _format(rpm["metrics"]["VDN"].get("coverage")),
                )
            )
            + " |",
            "| Ours | "
            + " | ".join(
                (
                    _format(rpm_ours.get("dialbench_ref_successful")),
                    _format(rpm_ours.get("dialbench_rel_successful")),
                    _format(rpm_ours.get("dialbench_acc_epsilon_e2e")),
                    _format(rpm_ours.get("dialbench_acc_theta_e2e")),
                    _format(rpm_ours.get("coverage")),
                )
            )
            + " |",
        ]
    )
    lines.extend(
        [
            "",
            "Absolute NMAE increase relative to each method's clean result:",
            "",
            "| Condition | Transformer ΔNMAE | VDN ΔNMAE | Ours ΔNMAE |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["robustness_from_clean"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["condition"],
                    _format(row["Original Transformer"]["absolute_increase"]),
                    _format(row["VDN"]["absolute_increase"]),
                    _format(row["Ours"]["absolute_increase"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "VDN training audit:",
            "",
            f"- Best epoch: {training['best_epoch']} / {training['epochs']}",
            f"- Validation direction MAE: {training['best_validation_angle_mae_degrees']:.4f}°",
            f"- Train/validation samples: {training['train_samples']} / {training['validation_samples']}",
            f"- Train/validation groups: {training['train_groups']} / {training['validation_groups']}",
            f"- Optimizer/skipped AMP steps: {training['optimizer_steps']} / {training['skipped_optimizer_steps']}",
            f"- External source commit: `{training['vdn_source_commit']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.vdn_root = args.vdn_root.resolve()
    args.internal_robustness_root = args.internal_robustness_root.resolve()
    args.internal_rpm_root = args.internal_rpm_root.resolve()
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    output = (
        args.output.resolve()
        if args.output
        else args.vdn_root / "vdn_comparison.json"
    )
    training_summary_path = args.vdn_root / "summary.json"
    verification_path = args.vdn_root / "verification.json"
    training_summary = _read_json(training_summary_path)
    if training_summary.get("status") != "complete":
        raise ValueError("VDN training is not complete")
    verification = _read_json(verification_path)
    if verification.get("verified") is not True:
        raise ValueError("VDN formal training verification is missing or failed")
    signature = training_summary.get("signature") or {}
    checkpoint_sha256 = sha256_file(args.vdn_root / "best.pt")
    if verification.get("best_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("VDN verification belongs to a different best checkpoint")
    if verification.get("summary_sha256") != sha256_file(training_summary_path):
        raise ValueError("VDN training summary changed after formal verification")
    verifier_path = Path(__file__).resolve().with_name("verify_vdn_run.py")
    if verification.get("verifier_source_sha256") != sha256_file(verifier_path):
        raise ValueError("VDN verifier source changed after formal verification")
    expected_evaluation_sources = {
        "adapter": sha256_file(Path(__file__).resolve().with_name("vdn_baseline.py")),
        "evaluation": sha256_file(
            Path(__file__).resolve().with_name("evaluate_vdn_baseline.py")
        ),
        "degradation": sha256_file(
            Path(__file__).resolve().with_name("robustness_degradations.py")
        ),
    }
    training = {
        "summary": str(training_summary_path),
        "summary_sha256": sha256_file(training_summary_path),
        "verification": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "best_epoch": int(training_summary["best_epoch"]),
        "epochs": int(signature["epochs"]),
        "best_validation_angle_mae_degrees": float(
            training_summary["best_validation_angle_mae_degrees"]
        ),
        "train_samples": int(signature["train_samples"]),
        "validation_samples": int(signature["validation_samples"]),
        "train_groups": int(verification["train_groups"]),
        "validation_groups": int(verification["validation_groups"]),
        "optimizer_steps": int(verification["optimizer_steps"]),
        "skipped_optimizer_steps": int(verification["skipped_optimizer_steps"]),
        "vdn_source_commit": signature["vdn_source_commit"],
        "checkpoint_sha256": checkpoint_sha256,
    }

    conditions: dict[str, Any] = {}
    for index, condition in enumerate(ALL_CONDITIONS):
        vdn_prediction_path, vdn_summary_path = _vdn_paths(args.vdn_root, condition)
        internal_root = _internal_root(args, condition)
        internal_metrics_path = internal_root / "metrics.json"
        internal_predictions_path = internal_root / "predictions.jsonl"
        for path in (
            vdn_prediction_path,
            vdn_summary_path,
            internal_metrics_path,
            internal_predictions_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        vdn_summary = _read_json(vdn_summary_path)
        if vdn_summary.get("status") != "complete":
            raise ValueError(f"incomplete VDN evaluation: {vdn_summary_path}")
        vdn_signature = vdn_summary.get("signature") or {}
        expected_condition = "clean" if condition == "rpm10k" else condition
        if (
            vdn_summary.get("condition") != expected_condition
            or vdn_signature.get("condition") != expected_condition
        ):
            raise ValueError(f"{condition}: VDN degradation condition mismatch")
        if vdn_signature.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"{condition}: VDN evaluation used a different checkpoint")
        if float(vdn_signature.get("failure_nmae_penalty", -1.0)) != 1.0:
            raise ValueError(f"{condition}: VDN evaluation used a different failure penalty")
        if vdn_signature.get("source_sha256") != expected_evaluation_sources:
            raise ValueError(f"{condition}: VDN evaluation source signature changed")
        internal_metrics = _read_json(internal_metrics_path)
        if internal_metrics.get("front_end_signature_verified") is not True:
            raise ValueError(f"unverified internal metrics: {internal_metrics_path}")
        shared_prediction_path = Path(
            str(internal_metrics.get("source_predictions") or "")
        ).resolve()
        if not shared_prediction_path.is_file():
            raise FileNotFoundError(shared_prediction_path)
        if vdn_signature.get("shared_predictions_sha256") != sha256_file(
            shared_prediction_path
        ):
            raise ValueError(f"{condition}: methods used different frozen predictions")
        if vdn_signature.get("shared_predictions_signature") != internal_metrics.get(
            "prediction_cache_signature"
        ):
            raise ValueError(f"{condition}: frozen prediction signatures differ")
        metrics = {
            method: _metric_view(internal_metrics["metrics"][method])
            for method in INTERNAL_METHODS
        }
        metrics["VDN"] = _metric_view(vdn_summary["metrics"])
        paired = _paired_comparison(
            _read_jsonl(vdn_prediction_path),
            _read_jsonl(internal_predictions_path),
            iterations=args.bootstrap_iterations,
            seed=args.seed + index,
        )
        if not math.isclose(
            paired["raw_vdn_nmae"],
            float(metrics["VDN"]["nmae"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{condition}: VDN raw/summary NMAE mismatch")
        if not math.isclose(
            paired["raw_ours_nmae"],
            float(metrics["Ours"]["nmae"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{condition}: Ours raw/summary NMAE mismatch")
        conditions[condition] = {
            "metrics": metrics,
            "paired_ours_vs_vdn": paired,
            "vdn_dialbench_metrics": vdn_summary.get("dialbench_metrics"),
            "vdn_direction_component": vdn_summary.get("direction_component"),
            "vdn_subgroups": vdn_summary.get("subgroups"),
            "sources": {
                "vdn_predictions": str(vdn_prediction_path),
                "vdn_predictions_sha256": sha256_file(vdn_prediction_path),
                "vdn_summary": str(vdn_summary_path),
                "vdn_summary_sha256": sha256_file(vdn_summary_path),
                "internal_metrics": str(internal_metrics_path),
                "internal_metrics_sha256": sha256_file(internal_metrics_path),
            },
        }

    payload = {
        "protocol": "external_vdn_same_data_same_adapter_comparison_v1",
        "training": training,
        "conditions": conditions,
        "robustness_from_clean": _robustness_from_clean(conditions),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": args.seed,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, payload)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(output)
    print(markdown_path)


if __name__ == "__main__":
    main()
