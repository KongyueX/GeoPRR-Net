"""Create paper-facing JSON/Markdown summaries for the quality router."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.quality_router import sha256_file


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
        default=Path("artifacts/runs/quality_router_syncg"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/quality_router_syncg/quality_router_comparison.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    args.output = args.output.resolve()
    markdown = args.output.with_suffix(".md")
    for path in (args.output, markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    training_path = args.run_root / "model" / "training_summary.json"
    ablation_path = args.run_root / "feature_ablation.json"
    training = _load(training_path)
    ablation = _load(ablation_path)
    evaluations: dict[str, Any] = {}
    for condition in CONDITIONS:
        path = args.run_root / "evaluations" / condition / "summary.json"
        summary = _load(path)
        metrics = summary["metrics"]
        comparison = summary["paired_comparisons"]["quality_vs_hard_fallback"]
        evaluations[condition] = {
            "samples": summary["samples"],
            "metrics": {
                name: {
                    "nmae": values["nmae"],
                    "acc_2pct": values["acc_2pct"],
                    "coverage": values["coverage"],
                }
                for name, values in metrics.items()
            },
            "quality_vs_hard_fallback": comparison,
            "routing": summary["routing"],
            "subgroups": summary.get("subgroups") or {},
            "summary": str(path),
            "summary_sha256": sha256_file(path),
        }

    stability: dict[str, Any] = {}
    for condition in ("clean", "rpm10k"):
        values = []
        for seed in (20260720, 20260721, 20260722):
            root = (
                args.run_root / "evaluations"
                if seed == 20260722
                else args.run_root / f"evaluations_seed_{seed}"
            )
            summary = _load(root / condition / "summary.json")
            values.append(
                {
                    "seed": seed,
                    "nmae": summary["metrics"]["quality_router"]["nmae"],
                    "acc_2pct": summary["metrics"]["quality_router"]["acc_2pct"],
                    "quality_switches": summary["routing"]["quality_switches"],
                    "delta_vs_hard": summary["paired_comparisons"]
                    ["quality_vs_hard_fallback"]["delta_nmae"],
                }
            )
        stability[condition] = {
            "seeds": values,
            "mean": {
                key: float(np.mean([value[key] for value in values]))
                for key in ("nmae", "acc_2pct", "quality_switches", "delta_vs_hard")
            },
            "sample_std": {
                key: float(np.std([value[key] for value in values], ddof=1))
                for key in ("nmae", "acc_2pct", "quality_switches", "delta_vs_hard")
            },
        }

    payload = {
        "schema_version": 1,
        "protocol": "quality_router_paper_summary_v1",
        "training": {
            "train_only_nested_oof_metrics": training["metrics"],
            "paired_vs_hard_fallback": training["paired_vs_hard_fallback"],
            "threshold": training["threshold"],
            "routing": training["routing"],
            "test_samples_used": training["test_samples_used"],
            "summary": str(training_path),
            "summary_sha256": sha256_file(training_path),
        },
        "feature_ablation": ablation,
        "evaluations_seed_20260722": evaluations,
        "three_seed_stability": stability,
        "interpretation": {
            "syncg_all_conditions_improve_hard_fallback": all(
                evaluations[name]["quality_vs_hard_fallback"]["group_bootstrap_95ci"][1]
                < 0.0
                for name in CONDITIONS
                if name != "rpm10k"
            ),
            "rpm10k_point_estimate_improves_hard_fallback": evaluations["rpm10k"]
            ["quality_vs_hard_fallback"]["delta_nmae"]
            < 0.0,
            "rpm10k_improvement_statistically_significant": evaluations["rpm10k"]
            ["quality_vs_hard_fallback"]["group_bootstrap_95ci"][1]
            < 0.0,
        },
    }
    _atomic_write(
        args.output,
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
        "# Frozen quality-router comparison",
        "",
        "The router and threshold are fitted from SyncG/train grouped OOF pairs only.",
        "",
        "| Condition | Base mask | Hard fallback | Quality router | Vector | VDN | Quality Acc@2% | Δ vs hard (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        result = evaluations[condition]
        metrics = result["metrics"]
        comparison = result["quality_vs_hard_fallback"]
        interval = comparison["group_bootstrap_95ci"]
        lines.append(
            f"| {labels[condition]} | {metrics['base_mask']['nmae']:.4f} | "
            f"{metrics['hard_fallback']['nmae']:.4f} | "
            f"{metrics['quality_router']['nmae']:.4f} | {metrics['vector']['nmae']:.4f} | "
            f"{metrics['vdn']['nmae']:.4f} | {metrics['quality_router']['acc_2pct']:.4f} | "
            f"{comparison['delta_nmae']:+.4f} [{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Three-seed stability",
            "",
            "| Dataset | NMAE mean ± sample std | Acc@2% mean ± sample std | Δ vs hard mean ± sample std |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in ("clean", "rpm10k"):
        value = stability[condition]
        lines.append(
            f"| {labels[condition]} | {value['mean']['nmae']:.4f} ± "
            f"{value['sample_std']['nmae']:.4f} | {value['mean']['acc_2pct']:.4f} ± "
            f"{value['sample_std']['acc_2pct']:.4f} | {value['mean']['delta_vs_hard']:+.4f} ± "
            f"{value['sample_std']['delta_vs_hard']:.4f} |"
        )
    lines.extend(
        [
            "",
            "RPM-10K improves in point estimate, but the seed-20260722 paired 95% CI crosses zero; it must not be reported as a statistically significant gain.",
        ]
    )
    _atomic_write(markdown, "\n".join(lines) + "\n")
    print(args.output)
    print(markdown)


if __name__ == "__main__":
    main()
