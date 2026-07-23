"""Compare v2 direction, v1 direction, and targeted ablations with paired CIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.quality_router import finite_float, normalized_error, read_jsonl
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
ABLATION_CONDITIONS = (
    "clean",
    "blur_severe",
    "perspective_severe",
    "combined_severe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-run",
        type=Path,
        default=Path(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260722"
        ),
    )
    parser.add_argument(
        "--v1-run",
        type=Path,
        default=Path("artifacts/runs/pivot_direction_syncg/seed_20260722"),
    )
    parser.add_argument(
        "--training-ablation-root",
        type=Path,
        default=Path("artifacts/runs/probabilistic_direction_ablations"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/probabilistic_pivot_direction_syncg/seed_20260722/"
            "probabilistic_direction_comparison.json"
        ),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _by_id(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    identifiers = [str(row.get("sample_id")) for row in rows]
    mapping = dict(zip(identifiers, rows))
    if len(mapping) != len(rows):
        raise ValueError(f"{path} contains duplicate sample IDs")
    return identifiers, mapping


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    predictions = [
        finite_float(row.get("prediction")) if row.get("status") is True else None
        for row in rows
    ]
    errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, predictions)],
        dtype=np.float64,
    )
    angles = [
        finite_float(row.get("direction_angle_error_degrees"))
        for row in rows
        if row.get("status") is True
    ]
    angles = [value for value in angles if value is not None]
    return {
        "samples": len(rows),
        "coverage": float(np.mean([value is not None for value in predictions])),
        "nmae": float(np.mean(errors)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "angle_mae_degrees_success_only": (
            float(np.mean(angles)) if angles else float("nan")
        ),
        "angle_acc_3deg_all": float(
            sum(value <= 3.0 for value in angles) / max(len(rows), 1)
        ),
    }


def _errors(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            normalized_error(
                row,
                finite_float(row.get("prediction"))
                if row.get("status") is True
                else None,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _paired(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        selected_indices = np.concatenate([indices[group] for group in selected])
        deltas[iteration] = float(
            np.mean(candidate[selected_indices]) - np.mean(baseline[selected_indices])
        )
    return {
        "delta_nmae": float(np.mean(candidate) - np.mean(baseline)),
        "group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "iterations": iterations,
        "groups": len(unique),
    }


def _load_aligned(
    candidate_path: Path,
    baseline_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    identifiers, candidate = _by_id(candidate_path)
    _, baseline = _by_id(baseline_path)
    if set(candidate) != set(baseline):
        raise ValueError(f"unaligned evaluations: {candidate_path}, {baseline_path}")
    candidate_rows = [candidate[sample_id] for sample_id in identifiers]
    baseline_rows = [baseline[sample_id] for sample_id in identifiers]
    groups = np.asarray(
        [
            str(
                row.get("group_id")
                or row.get("meter_id")
                or row.get("sample_id")
            )
            for row in candidate_rows
        ],
        dtype=object,
    )
    return candidate_rows, baseline_rows, groups


def _comparison(
    candidate_path: Path,
    baseline_path: Path,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    candidate, baseline, groups = _load_aligned(candidate_path, baseline_path)
    return {
        "candidate": _metrics(candidate),
        "baseline": _metrics(baseline),
        "paired": _paired(
            _errors(candidate),
            _errors(baseline),
            groups,
            seed=seed,
            iterations=iterations,
        ),
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    v2 = args.v2_run.resolve()
    v1 = args.v1_run.resolve()
    ablation_root = args.training_ablation_root.resolve()
    output = args.output.resolve()
    markdown = output.with_suffix(".md")
    if args.bootstrap_iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    for path in (output, markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    v2_vs_v1 = {
        condition: _comparison(
            v2 / "evaluations" / f"{condition}.jsonl",
            v1 / "evaluations" / f"{condition}.jsonl",
            seed=args.seed + index,
            iterations=args.bootstrap_iterations,
        )
        for index, condition in enumerate(CONDITIONS)
    }
    decoder_ablations: dict[str, Any] = {}
    for decoder_index, decoder in enumerate(("direct", "circular")):
        decoder_ablations[decoder] = {
            condition: _comparison(
                v2 / "decoder_ablations" / decoder / f"{condition}.jsonl",
                v2 / "evaluations" / f"{condition}.jsonl",
                seed=args.seed + 100 + decoder_index * 10 + index,
                iterations=args.bootstrap_iterations,
            )
            for index, condition in enumerate(ABLATION_CONDITIONS)
        }
    training_ablations: dict[str, Any] = {}
    for ablation_index, name in enumerate(
        ("no_equivariance_loss", "no_projective_pair")
    ):
        root = ablation_root / name / f"seed_{args.seed}"
        training_ablations[name] = {
            condition: _comparison(
                root / "evaluations" / f"{condition}.jsonl",
                v2 / "evaluations" / f"{condition}.jsonl",
                seed=args.seed + 200 + ablation_index * 10 + index,
                iterations=args.bootstrap_iterations,
            )
            for index, condition in enumerate(ABLATION_CONDITIONS)
        }
    payload = {
        "schema_version": 1,
        "protocol": "probabilistic_direction_paired_comparison_v1",
        "v2_vs_v1": v2_vs_v1,
        "decoder_ablations_vs_full": decoder_ablations,
        "training_ablations_vs_full": training_ablations,
        "source_sha256": sha256_file(Path(__file__).resolve()),
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
        "# Probabilistic direction paired comparison",
        "",
        "| Condition | v1 NMAE | v2 NMAE | v1 angle MAE | v2 angle MAE | Δ NMAE v2-v1 (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, value in v2_vs_v1.items():
        interval = value["paired"]["group_bootstrap_95ci"]
        lines.append(
            f"| {labels[condition]} | {value['baseline']['nmae']:.4f} | "
            f"{value['candidate']['nmae']:.4f} | "
            f"{value['baseline']['angle_mae_degrees_success_only']:.3f} | "
            f"{value['candidate']['angle_mae_degrees_success_only']:.3f} | "
            f"{value['paired']['delta_nmae']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Ablations (candidate minus full v2)",
            "",
            "| Ablation | Condition | Ablated / full angle MAE | Ablated NMAE | Full NMAE | Δ NMAE (95% CI) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for family in (decoder_ablations, training_ablations):
        for name, conditions in family.items():
            for condition, value in conditions.items():
                interval = value["paired"]["group_bootstrap_95ci"]
                lines.append(
                    f"| {name} | {labels[condition]} | "
                    f"{value['candidate']['angle_mae_degrees_success_only']:.3f} / "
                    f"{value['baseline']['angle_mae_degrees_success_only']:.3f} | "
                    f"{value['candidate']['nmae']:.4f} | {value['baseline']['nmae']:.4f} | "
                    f"{value['paired']['delta_nmae']:+.4f} "
                    f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |"
                )
    _atomic_write(markdown, "\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
