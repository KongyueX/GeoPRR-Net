"""Scene-cluster paired bootstrap for the completed R2MT ablations."""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np


SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
PROJECTIVE: Final[frozenset[str]] = frozenset(CONDITIONS[3:])
COMPARATORS: Final[tuple[tuple[str, str], ...]] = (
    (
        "Mean-risk RCMT",
        "remst_resnet18_relative_context_transport_fullfit40",
    ),
    (
        "w/o deep representations",
        "remst_resnet18_risk_arbitration_ablation_no_representation",
    ),
    (
        "w/o risk-imitation loss",
        "remst_resnet18_risk_arbitration_ablation_no_route_imitation",
    ),
    (
        "posterior probability mixture",
        "r2mt_ablation_posterior_mixture",
    ),
)


def _records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        (str(row["sample_id"]), str(row["condition"])): {
            "scene": str(row["scene_stem"]),
            "target": float(row["normalized_target"]),
            "prediction": float(row["mett"]["prediction"]),
        }
        for row in payload["per_sample_condition"]
    }


def _load_run(root: Path, run_name: str) -> list[dict[tuple[str, str], dict[str, Any]]]:
    return [
        _records(
            root
            / "artifacts"
            / "runs"
            / run_name
            / f"seed_{seed}"
            / "six_condition_results.json"
        )
        for seed in SEEDS
    ]


def _paired_report(
    full: Sequence[dict[tuple[str, str], dict[str, Any]]],
    comparator: Sequence[dict[tuple[str, str], dict[str, Any]]],
    *,
    selected_conditions: frozenset[str],
    bootstrap_replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    scene_sums: dict[str, float] = {}
    scene_counts: dict[str, int] = {}
    row_differences: list[float] = []
    per_seed: list[float] = []
    for full_seed, comparator_seed in zip(full, comparator):
        if full_seed.keys() != comparator_seed.keys():
            raise ValueError("R2MT paired rows differ")
        seed_differences = []
        for key in sorted(full_seed):
            if key[1] not in selected_conditions:
                continue
            full_row = full_seed[key]
            comparator_row = comparator_seed[key]
            if (
                full_row["scene"] != comparator_row["scene"]
                or abs(full_row["target"] - comparator_row["target"]) > 1.0e-12
            ):
                raise ValueError("R2MT paired metadata differs")
            difference = abs(
                comparator_row["prediction"] - comparator_row["target"]
            ) - abs(full_row["prediction"] - full_row["target"])
            scene = full_row["scene"]
            scene_sums[scene] = scene_sums.get(scene, 0.0) + difference
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            row_differences.append(difference)
            seed_differences.append(difference)
        per_seed.append(float(np.mean(seed_differences)))

    scenes = sorted(scene_sums)
    sums = np.asarray([scene_sums[scene] for scene in scenes], dtype=np.float64)
    counts = np.asarray([scene_counts[scene] for scene in scenes], dtype=np.float64)
    draws = rng.integers(
        0, len(scenes), size=(int(bootstrap_replicates), len(scenes))
    )
    bootstrap = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    differences = np.asarray(row_differences, dtype=np.float64)
    estimate = float(sums.sum() / counts.sum())
    return {
        "nmae_reduction_vs_comparator": estimate,
        "ci95_scene_cluster": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
        "paired_row_improvement_fraction": float(np.mean(differences > 0.0)),
        "paired_scene_improvement_fraction": float(
            np.mean((sums / counts) > 0.0)
        ),
        "per_seed_nmae_reduction": per_seed,
        "scenes": len(scenes),
        "paired_rows_across_seeds": int(differences.size),
    }


def analyze(
    *, root: Path, output_path: Path, bootstrap_replicates: int
) -> dict[str, Any]:
    full = _load_run(root, "r2mt_final_conservative")
    rng = np.random.default_rng(20262020)
    selections = {
        "all_conditions": frozenset(CONDITIONS),
        "projective_pooled": PROJECTIVE,
        **{condition: frozenset((condition,)) for condition in CONDITIONS},
    }
    comparisons = {}
    for display_name, run_name in COMPARATORS:
        comparator = _load_run(root, run_name)
        comparisons[display_name] = {
            name: _paired_report(
                full,
                comparator,
                selected_conditions=conditions,
                bootstrap_replicates=int(bootstrap_replicates),
                rng=rng,
            )
            for name, conditions in selections.items()
        }
    result = {
        "protocol": "r2mt_scene_cluster_paired_bootstrap_v1",
        "status": "complete",
        "formal_holdout_access": False,
        "estimand": (
            "comparator absolute error minus R2MT absolute error; positive "
            "values favor R2MT"
        ),
        "bootstrap_unit": "scene_stem",
        "bootstrap_replicates": int(bootstrap_replicates),
        "seeds": list(SEEDS),
        "comparisons": comparisons,
    }
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/probes/r2mt_paired_statistics.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = analyze(
        root=args.root.resolve(),
        output_path=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    compact = {
        comparator: {
            name: values[name]
            for name in ("all_conditions", "projective_pooled")
        }
        for comparator, values in result["comparisons"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
