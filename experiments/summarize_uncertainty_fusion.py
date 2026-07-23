"""Create paper-facing JSON and Markdown tables for uncertainty fusion."""
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
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/uncertainty_fusion_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/uncertainty_fusion_comparison.json"
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
    root = args.run_root.resolve()
    output = args.output.resolve()
    markdown = output.with_suffix(".md")
    for path in (output, markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    training_path = root / "model" / "training_summary.json"
    training = _load(training_path)
    feature_ablation_path = root / "feature_ablation.json"
    feature_ablation = _load(feature_ablation_path)
    evaluations: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = root / "evaluations" / condition / "summary.json"
        summary = _load(path)
        metrics = summary["metrics"]
        evaluations[condition] = {
            "samples": int(summary["samples"]),
            "metrics": {
                name: {
                    "nmae": float(values["nmae"]),
                    "acc_2pct": float(values["acc_2pct"]),
                    "coverage": float(values["coverage"]),
                }
                for name, values in metrics.items()
            },
            "paired_comparisons": summary["paired_comparisons"],
            "routing": summary["routing"],
            "calibration": summary["calibration"],
            "subgroups": summary.get("subgroups") or {},
            "summary": str(path),
            "summary_sha256": sha256_file(path),
        }
    payload = {
        "schema_version": 1,
        "protocol": "uncertainty_fusion_paper_summary_v1",
        "training": {
            "nested_group_oof_metrics": training["metrics"],
            "group_leakage_count": training["group_leakage_count"],
            "test_samples_used": training["test_samples_used"],
            "selected_final_epochs": training["selected_final_epochs"],
            "summary": str(training_path),
            "summary_sha256": sha256_file(training_path),
        },
        "feature_ablation": {
            **feature_ablation,
            "summary": str(feature_ablation_path),
            "summary_sha256": sha256_file(feature_ablation_path),
        },
        "evaluations": evaluations,
        "interpretation": {
            "syncg_all_point_estimates_improve_base": all(
                evaluations[name]["paired_comparisons"]["fusion_vs_base_mask"]
                ["delta_nmae"]
                < 0.0
                for name in CONDITIONS
                if name != "rpm10k"
            ),
            "syncg_all_improvements_significant_vs_base": all(
                evaluations[name]["paired_comparisons"]["fusion_vs_base_mask"]
                ["group_bootstrap_95ci"]
                and evaluations[name]["paired_comparisons"]["fusion_vs_base_mask"]
                ["group_bootstrap_95ci"][1]
                < 0.0
                for name in CONDITIONS
                if name != "rpm10k"
            ),
            "rpm10k_point_estimate_improves_base": evaluations["rpm10k"]
            ["paired_comparisons"]["fusion_vs_base_mask"]["delta_nmae"]
            < 0.0,
            "rpm10k_point_estimate_improves_quality_v1": evaluations["rpm10k"]
            ["paired_comparisons"].get("fusion_vs_quality_router_v1", {})
            .get("delta_nmae", 0.0)
            < 0.0,
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
        "# Frozen probabilistic direction and uncertainty-fusion comparison",
        "",
        "The variance model is fitted from SyncG/train grouped nested OOF predictions only.",
        "",
        "| Condition | Base mask NMAE | Prob. vector | Hard fallback | Quality v1 | Soft fusion | VDN | Fusion Acc@2% | Δ vs base (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = evaluations[condition]
        metrics = result["metrics"]
        comparison = result["paired_comparisons"]["fusion_vs_base_mask"]
        interval = comparison["group_bootstrap_95ci"]
        interval_text = (
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "n/a"
        )
        quality = metrics.get("quality_router_v1", {"nmae": float("nan")})
        lines.append(
            f"| {labels[condition]} | {metrics['base_mask']['nmae']:.4f} | "
            f"{metrics['probabilistic_vector']['nmae']:.4f} | "
            f"{metrics['hard_fallback']['nmae']:.4f} | {quality['nmae']:.4f} | "
            f"{metrics['uncertainty_fusion']['nmae']:.4f} | {metrics['vdn']['nmae']:.4f} | "
            f"{metrics['uncertainty_fusion']['acc_2pct']:.4f} | "
            f"{comparison['delta_nmae']:+.4f} {interval_text} |"
        )
    lines.extend(
        [
            "",
            "A negative delta means the frozen soft fusion improves on the mask expert. "
            "Statistical claims should be made only when the whole paired group-bootstrap interval is below zero.",
            "",
            "## Train-only nested-OOF fusion feature ablation",
            "",
            "| Features | NMAE | Acc@2% | Coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, metrics in feature_ablation["results"].items():
        lines.append(
            f"| {name} | {metrics['nmae']:.4f} | {metrics['acc_2pct']:.4f} | "
            f"{metrics['coverage']:.4f} |"
        )
    _atomic_write(markdown, "\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
