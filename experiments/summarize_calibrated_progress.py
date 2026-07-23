"""Create paper-facing JSON and Markdown tables for progress calibration."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.vdn_baseline import sha256_file


CONDITIONS = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
    "rpm10k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrator-root",
        type=Path,
        default=Path("artifacts/runs/progress_calibrator_syncg"),
    )
    parser.add_argument(
        "--router-root",
        type=Path,
        default=Path("artifacts/runs/calibrated_progress_router_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/calibrated_progress_router_syncg/"
            "calibrated_progress_comparison.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    calibrator_root = args.calibrator_root.resolve()
    router_root = args.router_root.resolve()
    output = args.output.resolve()
    markdown = output.with_suffix(".md")
    for path in (output, markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    calibrator_training_path = calibrator_root / "model" / "training_summary.json"
    router_training_path = router_root / "model" / "training_summary.json"
    calibrator_training = _load(calibrator_training_path)
    router_training = _load(router_training_path)
    evaluations: dict[str, Any] = {}
    for condition in CONDITIONS:
        calibration_path = calibrator_root / "evaluations" / condition / "summary.json"
        router_path = router_root / "evaluations" / condition / "summary.json"
        calibration = _load(calibration_path)
        router = _load(router_path)
        evaluations[condition] = {
            "samples": int(router["samples"]),
            "metrics": {
                name: {
                    "nmae": float(values["nmae"]),
                    "acc_2pct": float(values["acc_2pct"]),
                    "coverage": float(values["coverage"]),
                }
                for name, values in router["metrics"].items()
            },
            "calibrator_comparisons": calibration["paired_comparisons"],
            "router_comparisons": router["paired_comparisons"],
            "calibrator_transfers": calibration["transfers"],
            "router_routing": router["routing"],
            "calibrator_summary": str(calibration_path),
            "calibrator_summary_sha256": sha256_file(calibration_path),
            "router_summary": str(router_path),
            "router_summary_sha256": sha256_file(router_path),
        }
    payload = {
        "schema_version": 1,
        "protocol": "calibrated_progress_paper_summary_v2",
        "calibrator_training": {
            "metrics": calibrator_training["metrics"],
            "paired_vs_raw_vector": calibrator_training["paired_vs_raw_vector"],
            "transfers": calibrator_training["transfers"],
            "group_leakage_count": calibrator_training["group_leakage_count"],
            "test_samples_used": calibrator_training["test_samples_used"],
            "summary": str(calibrator_training_path),
            "summary_sha256": sha256_file(calibrator_training_path),
        },
        "router_training": {
            "metrics": router_training["metrics"],
            "paired_vs_hard_fallback": router_training["paired_vs_hard_fallback"],
            "routing": router_training["routing"],
            "group_leakage_count": router_training["group_leakage_count"],
            "test_samples_used": router_training["test_samples_used"],
            "summary": str(router_training_path),
            "summary_sha256": sha256_file(router_training_path),
        },
        "evaluations": evaluations,
        "interpretation": {
            "all_syncg_router_improvements_significant_vs_quality_v1": all(
                evaluations[name]["router_comparisons"]["router_vs_quality_router_v1"]
                ["group_bootstrap_95ci"][1] < 0.0
                for name in CONDITIONS if name != "rpm10k"
            ),
            "all_conditions_calibrator_improves_raw_vector": all(
                evaluations[name]["calibrator_comparisons"]["calibrated_vs_raw_vector"]
                ["delta_nmae"] < 0.0
                for name in CONDITIONS
            ),
            "rpm10k_router_point_estimate_improves_quality_v1": evaluations[
                "rpm10k"
            ]["router_comparisons"]["router_vs_quality_router_v1"]
            ["delta_nmae"] < 0.0,
            "rpm10k_router_improvement_significant_vs_quality_v1": evaluations[
                "rpm10k"
            ]["router_comparisons"]["router_vs_quality_router_v1"]
            ["group_bootstrap_95ci"][1] < 0.0,
            "all_conditions_router_improves_vdn_nmae": all(
                evaluations[name]["router_comparisons"]["router_vs_vdn"]
                ["group_bootstrap_95ci"][1] < 0.0
                for name in CONDITIONS
            ),
            "conditions_router_acc2_below_vdn": [
                name
                for name in CONDITIONS
                if evaluations[name]["metrics"]["calibrated_progress_router"]
                ["acc_2pct"] < evaluations[name]["metrics"]["vdn"]["acc_2pct"]
            ],
        },
    }
    _atomic_write(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    labels = {
        "clean": "Clean",
        "blur_moderate": "Blur-M",
        "blur_severe": "Blur-S",
        "perspective_moderate": "Perspective-M",
        "perspective_severe": "Perspective-S",
        "combined_severe": "Combined-S",
        "rpm10k": "RPM-10K",
    }
    lines = [
        "# Frozen perspective-aware progress calibration comparison",
        "",
        "Both the calibrator and router are fitted from SyncG/train grouped OOF rows only.",
        "",
        "| Condition | Base | Raw vector | Calibrated vector | Quality v1 | Previous router | Final router | Final Acc@2% | Final vs quality v1 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = evaluations[condition]
        metrics = result["metrics"]
        comparison = result["router_comparisons"]["router_vs_quality_router_v1"]
        interval = comparison["group_bootstrap_95ci"]
        lines.append(
            f"| {labels[condition]} | {metrics['base_mask']['nmae']:.4f} | "
            f"{metrics['raw_probabilistic_vector']['nmae']:.4f} | "
            f"{metrics['calibrated_vector']['nmae']:.4f} | "
            f"{metrics['quality_router_v1']['nmae']:.4f} | "
            f"{metrics['uncertainty_router_v2']['nmae']:.4f} | "
            f"{metrics['calibrated_progress_router']['nmae']:.4f} | "
            f"{metrics['calibrated_progress_router']['acc_2pct']:.4f} | "
            f"{comparison['delta_nmae']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "A negative delta means the final router improves over the previous train-only quality router. "
            "RPM-10K has only six groups, so its interval must be reported with the point estimate.",
            "",
            "## External architecture under the same protocol",
            "",
            "VDN is the published Vector Detection Network architecture retrained on the same SyncG/train split. "
            "It shares the frozen meter crop, start/end references, known-range adapter, degraded inputs, and "
            "full-denominator failure scoring with the final method.",
            "",
            "| Condition | Samples | VDN NMAE | Final NMAE | VDN Acc@2% | Final Acc@2% | Final vs VDN ΔNMAE (95% CI) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        result = evaluations[condition]
        metrics = result["metrics"]
        comparison = result["router_comparisons"]["router_vs_vdn"]
        interval = comparison["group_bootstrap_95ci"]
        lines.append(
            f"| {labels[condition]} | {result['samples']} | "
            f"{metrics['vdn']['nmae']:.4f} | "
            f"{metrics['calibrated_progress_router']['nmae']:.4f} | "
            f"{metrics['vdn']['acc_2pct']:.4f} | "
            f"{metrics['calibrated_progress_router']['acc_2pct']:.4f} | "
            f"{comparison['delta_nmae']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "The final method has a lower NMAE with a below-zero grouped-bootstrap interval in all seven "
            "conditions. This does not imply metric-wide dominance: VDN has the higher Acc@2% on severe "
            "perspective and RPM-10K.",
        ]
    )
    _atomic_write(markdown, "\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
