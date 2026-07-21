"""Create paper-ready controlled and real hard-subset robustness summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.robustness_degradations import degradation_names


CONDITION_ORDER = degradation_names()
CONDITION_LABELS = {
    "clean": "Clean",
    "blur_moderate": "Moderate blur",
    "blur_severe": "Severe blur",
    "perspective_moderate": "Moderate perspective",
    "perspective_severe": "Severe perspective",
    "combined_severe": "Severe blur + perspective",
}
METHODS = (
    "Original Transformer",
    "Quality-weighted Fusion",
    "Residual without Gate",
    "Ours",
)
METHOD_KEYS = {
    "Original Transformer": "transformer",
    "Quality-weighted Fusion": "weighted_fusion",
    "Residual without Gate": "residual_ungated",
    "Ours": "ours",
}
METHOD_STYLES = {
    "Original Transformer": {"marker": "o", "linestyle": "--"},
    "Quality-weighted Fusion": {"marker": "s", "linestyle": ":"},
    "Residual without Gate": {"marker": "^", "linestyle": "-."},
    "Ours": {"marker": "D", "linestyle": "-"},
}


def _condition_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must use NAME=METRICS_JSON")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if name not in CONDITION_ORDER:
        raise argparse.ArgumentTypeError(
            f"unknown condition {name!r}; expected one of {', '.join(CONDITION_ORDER)}"
        )
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("condition metrics path cannot be empty")
    return name, Path(raw_path.strip())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no prediction rows")
    return rows


def _rows_by_sample_id(
    rows: list[dict[str, Any]], path: Path
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{path} contains a row without sample_id")
        if sample_id in indexed:
            raise ValueError(f"{path} contains duplicate sample_id {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def _normalized_error(row: dict[str, Any], method_key: str) -> float:
    span = abs(float(row["scale_end"]) - float(row["scale_start"]))
    if span <= 0.0:
        raise ValueError(f"sample {row.get('sample_id')!r} has a non-positive scale span")
    prediction = (row.get("predictions") or {}).get(method_key)
    if prediction is None:
        return 1.0
    prediction = float(prediction)
    if not np.isfinite(prediction):
        return 1.0
    return abs(prediction - float(row["ground_truth"])) / span


def _bootstrap_group_ci(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> list[float] | None:
    if iterations <= 0 or values.size == 0:
        return None
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return None
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        sample = np.concatenate([values[indices[group]] for group in sampled_groups])
        estimates.append(float(np.mean(sample)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def _paired_degradation_statistics(
    clean_rows: list[dict[str, Any]],
    degraded_rows: list[dict[str, Any]],
    *,
    clean_path: Path,
    degraded_path: Path,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    clean = _rows_by_sample_id(clean_rows, clean_path)
    degraded = _rows_by_sample_id(degraded_rows, degraded_path)
    if clean.keys() != degraded.keys():
        missing = sorted(clean.keys() - degraded.keys())[:3]
        extra = sorted(degraded.keys() - clean.keys())[:3]
        raise ValueError(
            f"paired prediction sample mismatch for {degraded_path}: "
            f"missing={missing}, extra={extra}"
        )

    changes: dict[str, list[float]] = {method: [] for method in METHODS}
    groups: list[str] = []
    for sample_id in sorted(clean):
        clean_row = clean[sample_id]
        degraded_row = degraded[sample_id]
        for field in ("ground_truth", "scale_start", "scale_end"):
            if not np.isclose(
                float(clean_row[field]), float(degraded_row[field]), rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"paired sample {sample_id!r} changed {field} in {degraded_path}"
                )
        clean_group = str(
            clean_row.get("group_id") or clean_row.get("meter_id") or sample_id
        )
        degraded_group = str(
            degraded_row.get("group_id") or degraded_row.get("meter_id") or sample_id
        )
        if clean_group != degraded_group:
            raise ValueError(
                f"paired sample {sample_id!r} changed group_id in {degraded_path}"
            )
        groups.append(clean_group)
        for method, method_key in METHOD_KEYS.items():
            changes[method].append(
                _normalized_error(degraded_row, method_key)
                - _normalized_error(clean_row, method_key)
            )

    group_array = np.asarray(groups, dtype=object)
    method_means = {
        method: float(np.mean(np.asarray(values, dtype=np.float64)))
        for method, values in changes.items()
    }
    interaction = np.asarray(changes["Ours"], dtype=np.float64) - np.asarray(
        changes["Original Transformer"], dtype=np.float64
    )
    return {
        "paired_samples": len(clean),
        "failure_penalty_nmae": 1.0,
        "method_nmae_change_from_clean": method_means,
        "ours_vs_transformer_degradation_difference_nmae": float(
            np.mean(interaction)
        ),
        "ours_vs_transformer_degradation_difference_group_bootstrap_95ci": (
            _bootstrap_group_ci(
                interaction,
                group_array,
                iterations=bootstrap_iterations,
                seed=seed,
            )
        ),
    }


def _method_summary(rows: list[dict[str, Any]], method_key: str) -> dict[str, Any]:
    errors = np.asarray(
        [_normalized_error(row, method_key) for row in rows], dtype=np.float64
    )
    successful = np.asarray(
        [
            (row.get("predictions") or {}).get(method_key) is not None
            and np.isfinite(float((row.get("predictions") or {}).get(method_key)))
            for row in rows
        ],
        dtype=bool,
    )
    return {
        "samples": len(rows),
        "nmae": float(np.mean(errors)),
        "acc_2pct": float(np.mean(successful & (errors <= 0.02))),
        "coverage": float(np.mean(successful)),
    }


def _real_subset_summary(
    rows: list[dict[str, Any]],
    required_tags: set[str],
    *,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    subset = [
        row
        for row in rows
        if required_tags.issubset(
            {
                str(tag).strip().lower()
                for tag in (row.get("metadata") or {}).get(
                    "environment_conditions", []
                )
            }
        )
    ]
    if not subset:
        return None
    original = _method_summary(subset, "transformer")
    ours = _method_summary(subset, "ours")
    delta = np.asarray(
        [
            _normalized_error(row, "ours")
            - _normalized_error(row, "transformer")
            for row in subset
        ],
        dtype=np.float64,
    )
    groups = np.asarray(
        [
            str(row.get("group_id") or row.get("meter_id") or row.get("sample_id"))
            for row in subset
        ],
        dtype=object,
    )
    return {
        "samples": len(subset),
        "Original Transformer": original,
        "Ours": ours,
        "delta_nmae": float(np.mean(delta)),
        "delta_nmae_group_bootstrap_95ci": _bootstrap_group_ci(
            delta,
            groups,
            iterations=bootstrap_iterations,
            seed=seed,
        ),
    }


def _validate_condition(name: str, path: Path, payload: dict[str, Any]) -> None:
    if payload.get("protocol") != "frozen_model_evaluation":
        raise ValueError(f"{path} is not a frozen evaluation result")
    signature = payload.get("prediction_cache_signature")
    degradation = signature.get("input_degradation") if isinstance(signature, dict) else None
    # Formal clean metrics produced before this protocol remain usable because
    # their images were not transformed.  Every non-clean cache must be signed.
    if name != "clean":
        if not isinstance(degradation, dict) or degradation.get("condition") != name:
            raise ValueError(
                f"{path} is not signed for controlled degradation {name!r}"
            )
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"{path} has no metrics object")
    if payload.get("front_end_signature_verified") is not True:
        raise ValueError(f"{path} does not have a verified frozen front-end signature")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _delta_with_ci(payload: dict[str, Any]) -> tuple[float | None, list[float] | None]:
    comparison = (payload.get("paired_comparisons") or {}).get(
        "Ours vs Original Transformer"
    ) or {}
    delta = comparison.get("delta_nmae")
    interval = comparison.get("delta_nmae_group_bootstrap_95ci")
    return delta, interval if isinstance(interval, list) and len(interval) == 2 else None


def build_report_data(
    conditions: list[tuple[str, Path]],
    rpm_metrics: Path | None = None,
    expected_samples: int | None = None,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    paths_by_name: dict[str, Path] = {}
    for name, path in conditions:
        if name in by_name:
            raise ValueError(f"duplicate condition: {name}")
        payload = _load_json(path)
        _validate_condition(name, path, payload)
        by_name[name] = payload
        paths_by_name[name] = path
    missing = [name for name in CONDITION_ORDER if name not in by_name]
    if missing:
        raise ValueError(f"missing controlled conditions: {', '.join(missing)}")

    sample_counts = {int(payload.get("samples_total", -1)) for payload in by_name.values()}
    if len(sample_counts) != 1 or min(sample_counts) <= 0:
        raise ValueError("all controlled conditions must contain the same positive sample count")
    actual_samples = next(iter(sample_counts))
    if expected_samples is not None and actual_samples != expected_samples:
        raise ValueError(
            f"controlled evaluation contains {actual_samples} samples; "
            f"expected {expected_samples}"
        )
    calibrators = {str(payload.get("calibrator") or "") for payload in by_name.values()}
    if len(calibrators) != 1 or not next(iter(calibrators)):
        raise ValueError("all controlled conditions must use the same frozen calibrator")
    signed_protocols = set()
    signed_seeds = set()
    signed_sources = set()
    for payload in by_name.values():
        signature = payload.get("prediction_cache_signature") or {}
        degradation = signature.get("input_degradation")
        if isinstance(degradation, dict):
            signed_protocols.add(str(degradation.get("protocol") or ""))
            signed_seeds.add(int(degradation.get("seed")))
            signed_sources.add(str(signature.get("input_degradation_source_sha256") or ""))
    if len(signed_protocols) != 1 or "" in signed_protocols:
        raise ValueError("controlled conditions do not share one degradation protocol")
    if len(signed_seeds) != 1:
        raise ValueError("controlled conditions do not share one degradation seed")
    if len(signed_sources) != 1 or "" in signed_sources:
        raise ValueError("controlled conditions do not share one degradation implementation")

    prediction_paths = {
        name: paths_by_name[name].parent / "predictions.jsonl"
        for name in CONDITION_ORDER
    }
    missing_predictions = [
        str(path) for path in prediction_paths.values() if not path.is_file()
    ]
    if missing_predictions:
        raise ValueError(
            "paired robustness reporting requires sibling predictions.jsonl files; "
            f"missing: {', '.join(missing_predictions)}"
        )
    prediction_rows = {
        name: _load_jsonl(path) for name, path in prediction_paths.items()
    }
    for name, rows in prediction_rows.items():
        if len(rows) != actual_samples:
            raise ValueError(
                f"{prediction_paths[name]} contains {len(rows)} rows; "
                f"expected {actual_samples}"
            )
    clean_rows = prediction_rows["clean"]
    degradation_seed = next(iter(signed_seeds))

    controlled = []
    for condition_index, name in enumerate(CONDITION_ORDER):
        payload = by_name[name]
        delta, interval = _delta_with_ci(payload)
        paired_degradation = _paired_degradation_statistics(
            clean_rows,
            prediction_rows[name],
            clean_path=prediction_paths["clean"],
            degraded_path=prediction_paths[name],
            bootstrap_iterations=bootstrap_iterations,
            seed=degradation_seed + condition_index,
        )
        controlled.append(
            {
                "condition": name,
                "label": CONDITION_LABELS[name],
                "samples": int(payload["samples_total"]),
                "metrics": {method: payload["metrics"].get(method, {}) for method in METHODS},
                "ours_vs_transformer_delta_nmae": delta,
                "ours_vs_transformer_delta_nmae_group_bootstrap_95ci": interval,
                "paired_degradation_from_clean": paired_degradation,
                "front_end_signature_verified": payload.get("front_end_signature_verified"),
            }
        )

    clean = controlled[0]["metrics"]
    for row in controlled:
        for method in METHODS:
            current = row["metrics"][method].get("nmae")
            baseline = clean[method].get("nmae")
            row["metrics"][method]["nmae_increase_from_clean"] = (
                None if current is None or baseline is None else float(current) - float(baseline)
            )

    real_subsets: list[dict[str, Any]] = []
    if rpm_metrics is not None:
        rpm = _load_json(rpm_metrics)
        rpm_predictions = rpm_metrics.parent / "predictions.jsonl"
        if rpm_predictions.is_file():
            rpm_rows = _load_jsonl(rpm_predictions)
            subset_specs = (
                ("blur", {"blur"}),
                ("tilted", {"tilted"}),
                ("blur + tilted", {"blur", "tilted"}),
            )
            for subset_index, (condition, required_tags) in enumerate(subset_specs):
                summary = _real_subset_summary(
                    rpm_rows,
                    required_tags,
                    bootstrap_iterations=bootstrap_iterations,
                    seed=degradation_seed + 100 + subset_index,
                )
                if summary is not None:
                    summary["condition"] = condition
                    real_subsets.append(summary)
        else:
            groups = (rpm.get("subgroups") or {}).get("environment_condition") or {}
            for condition in ("blur", "tilted"):
                methods = groups.get(condition)
                if not isinstance(methods, dict):
                    continue
                original = methods.get("Original Transformer") or {}
                ours = methods.get("Ours") or {}
                real_subsets.append(
                    {
                        "condition": condition,
                        "samples": int(
                            ours.get("samples") or original.get("samples") or 0
                        ),
                        "Original Transformer": original,
                        "Ours": ours,
                        "delta_nmae": (
                            None
                            if ours.get("nmae") is None
                            or original.get("nmae") is None
                            else float(ours["nmae"]) - float(original["nmae"])
                        ),
                        "delta_nmae_group_bootstrap_95ci": None,
                    }
                )

    return {
        "protocol": "robustness_report_v2",
        "controlled_degradation_seed": degradation_seed,
        "bootstrap_iterations": bootstrap_iterations,
        "relative_degradation_definition": (
            "(Ours degraded - Ours clean) - "
            "(Original Transformer degraded - Original Transformer clean); "
            "negative values mean Ours degrades less"
        ),
        "controlled": controlled,
        "real_rpm10k_multilabel_subsets": real_subsets,
    }


def _markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Blur and perspective robustness",
        "",
        "## Controlled SyncG evaluation",
        "",
        "| Condition | N | Transformer NMAE ↓ | Geometry fusion NMAE ↓ | "
        "Residual (no gate) NMAE ↓ | Ours NMAE ↓ | Ours−Transformer ΔNMAE "
        "[group-bootstrap 95% CI] ↓ | Transformer Acc@2% ↑ | Ours Acc@2% ↑ | "
        "Ours coverage ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["controlled"]:
        metrics = row["metrics"]
        delta = row["ours_vs_transformer_delta_nmae"]
        interval = row["ours_vs_transformer_delta_nmae_group_bootstrap_95ci"]
        delta_text = _fmt(delta)
        if interval is not None:
            delta_text += f" [{_fmt(interval[0])}, {_fmt(interval[1])}]"
        lines.append(
            "| "
            + " | ".join(
                (
                    row["label"],
                    str(row["samples"]),
                    _fmt(metrics["Original Transformer"].get("nmae")),
                    _fmt(metrics["Quality-weighted Fusion"].get("nmae")),
                    _fmt(metrics["Residual without Gate"].get("nmae")),
                    _fmt(metrics["Ours"].get("nmae")),
                    delta_text,
                    _fmt(metrics["Original Transformer"].get("acc_2pct")),
                    _fmt(metrics["Ours"].get("acc_2pct")),
                    _fmt(metrics["Ours"].get("coverage")),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "_NMAE and Acc@2% use the full evaluable denominator. A missing "
            "reading receives NMAE penalty 1.0 and is incorrect for Acc@2%. "
            "All degraded inputs are deterministic and the calibrator remains "
            "frozen from clean SyncG train._",
        )
    )

    lines.extend(
        (
            "",
            "## Paired degradation relative to clean",
            "",
            "| Condition | Transformer ΔNMAE from clean | Ours ΔNMAE from clean | "
            "ΔΔNMAE (Ours−Transformer) [group-bootstrap 95% CI] |",
            "|---|---:|---:|---:|",
        )
    )
    for row in data["controlled"]:
        paired = row["paired_degradation_from_clean"]
        method_change = paired["method_nmae_change_from_clean"]
        interaction = paired[
            "ours_vs_transformer_degradation_difference_nmae"
        ]
        interval = paired.get(
            "ours_vs_transformer_degradation_difference_group_bootstrap_95ci"
        )
        interaction_text = _fmt(interaction)
        if interval is not None:
            interaction_text += f" [{_fmt(interval[0])}, {_fmt(interval[1])}]"
        lines.append(
            "| "
            + " | ".join(
                (
                    row["label"],
                    _fmt(method_change.get("Original Transformer")),
                    _fmt(method_change.get("Ours")),
                    interaction_text,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "_ΔΔNMAE compares the paired change from clean. Negative values mean "
            "Ours degrades less; positive values mean its absolute advantage remains "
            "but narrows under that corruption._",
        )
    )

    real = data.get("real_rpm10k_multilabel_subsets") or []
    if real:
        lines.extend(
            (
                "",
                "## RPM-10K real hard subsets (frozen zero-shot evaluation)",
                "",
                "| Official condition | N | Transformer NMAE ↓ | Ours NMAE ↓ | "
                "ΔNMAE [group-bootstrap 95% CI] ↓ | Transformer Acc@2% ↑ | "
                "Ours Acc@2% ↑ | Coverage ↑ |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in real:
            original = row["Original Transformer"]
            ours = row["Ours"]
            delta_text = _fmt(row.get("delta_nmae"))
            interval = row.get("delta_nmae_group_bootstrap_95ci")
            if interval is not None:
                delta_text += f" [{_fmt(interval[0])}, {_fmt(interval[1])}]"
            lines.append(
                "| "
                + " | ".join(
                    (
                        row["condition"],
                        str(row["samples"]),
                        _fmt(original.get("nmae")),
                        _fmt(ours.get("nmae")),
                        delta_text,
                        _fmt(original.get("acc_2pct")),
                        _fmt(ours.get("acc_2pct")),
                        _fmt(ours.get("coverage")),
                    )
                )
                + " |"
            )
        lines.extend(
            (
                "",
                "_RPM-10K environment tags are multi-label, so blur and tilted "
                "subsets may overlap. These are supporting real-image results, "
                "not independent datasets._",
            )
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, data: dict[str, Any]) -> None:
    rows = []
    for condition in data["controlled"]:
        paired = condition["paired_degradation_from_clean"]
        method_change = paired["method_nmae_change_from_clean"]
        interaction_interval = paired.get(
            "ours_vs_transformer_degradation_difference_group_bootstrap_95ci"
        )
        for method in METHODS:
            metric = condition["metrics"][method]
            rows.append(
                {
                    "condition": condition["condition"],
                    "samples": condition["samples"],
                    "method": method,
                    "nmae": metric.get("nmae"),
                    "nmae_increase_from_clean": metric.get("nmae_increase_from_clean"),
                    "paired_nmae_change_from_clean": method_change.get(method),
                    "ours_vs_transformer_degradation_difference_nmae": paired.get(
                        "ours_vs_transformer_degradation_difference_nmae"
                    ),
                    "degradation_difference_ci_low": (
                        None
                        if interaction_interval is None
                        else interaction_interval[0]
                    ),
                    "degradation_difference_ci_high": (
                        None
                        if interaction_interval is None
                        else interaction_interval[1]
                    ),
                    "acc_2pct": metric.get("acc_2pct"),
                    "coverage": metric.get("coverage"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_robustness(data: dict[str, Any], output: Path) -> None:
    by_name = {row["condition"]: row for row in data["controlled"]}
    branches = (
        ("Blur", ("clean", "blur_moderate", "blur_severe")),
        ("Perspective", ("clean", "perspective_moderate", "perspective_severe")),
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex="col")
    for column, (title, names) in enumerate(branches):
        labels = ["Clean", "Moderate", "Severe"]
        for method in METHODS:
            style = METHOD_STYLES[method]
            nmae = [by_name[name]["metrics"][method].get("nmae") for name in names]
            accuracy = [
                by_name[name]["metrics"][method].get("acc_2pct") for name in names
            ]
            axes[0, column].plot(labels, nmae, label=method, linewidth=2.0, **style)
            axes[1, column].plot(labels, accuracy, label=method, linewidth=2.0, **style)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("End-to-end NMAE" if column == 0 else "")
        axes[1, column].set_ylabel("Acc@2%" if column == 0 else "")
        axes[1, column].set_xlabel("Degradation severity")
    for axis in axes.ravel():
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        type=_condition_argument,
        required=True,
        metavar="NAME=METRICS_JSON",
    )
    parser.add_argument("--rpm-metrics", type=Path)
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    data = build_report_data(
        args.condition,
        args.rpm_metrics,
        expected_samples=args.expected_samples,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output.with_suffix(".csv"), data)
    if args.plot is not None:
        plot_robustness(data, args.plot)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
