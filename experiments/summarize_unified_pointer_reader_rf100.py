"""Summarize three-seed GeoPRR-Net and Raw-CNN results on RF100."""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.evaluate_a15_2_syncg_scene_holdout import CONDITIONS
from experiments.evaluate_paper_syncg_only_lightweight_baselines import (
    _paired_group_bootstrap_fast,
)
from experiments.evaluate_unified_pointer_reader_rf100 import PROTOCOL as EVALUATION_PROTOCOL
from experiments.summarize_unified_pointer_reader_experiments import (
    DEFAULT_SEEDS,
    RAW_EXTERNAL_MODELS,
    _load_raw_external_dataset,
)


PROTOCOL: Final[str] = "unified_pointer_reader_rf100_summary_v1"
DEFAULT_RUN_ROOT: Final[Path] = Path("artifacts/runs/unified_pointer_reader")
DEFAULT_EXTERNAL_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1")
DEFAULT_OUTPUT: Final[Path] = Path(
    "artifacts/reports/unified_pointer_reader/rf100_results.json"
)
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20_260_827


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"missing RF100 result: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("status") == "complete"
        and payload.get("protocol") == EVALUATION_PROTOCOL,
        f"invalid RF100 result: {source}",
    )
    return payload


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    _require(bool(data), "metric distribution is empty")
    return {
        "mean": float(statistics.fmean(data)),
        "sample_sd": float(statistics.stdev(data)) if len(data) >= 2 else 0.0,
        "per_seed": data,
    }


def _metric_summary(
    errors: Mapping[int, np.ndarray],
    passed: Mapping[int, np.ndarray],
    indices: np.ndarray,
) -> dict[str, Any]:
    selected_errors = {seed: values[indices] for seed, values in errors.items()}
    selected_passed = {seed: values[indices] for seed, values in passed.items()}
    return {
        "rows_per_seed": int(indices.sum()),
        "nmae": _distribution(
            [float(selected_errors[seed].mean()) for seed in DEFAULT_SEEDS]
        ),
        "acc_at_2_percent": _distribution(
            [
                float((selected_errors[seed] <= 0.02).mean())
                for seed in DEFAULT_SEEDS
            ]
        ),
        "acc_at_5_percent": _distribution(
            [
                float((selected_errors[seed] <= 0.05).mean())
                for seed in DEFAULT_SEEDS
            ]
        ),
        "coverage": _distribution(
            [float(selected_passed[seed].mean()) for seed in DEFAULT_SEEDS]
        ),
    }


def summarize_rf100(
    *,
    run_root: Path,
    external_root: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Assemble aligned metrics and paired group-bootstrap comparisons."""

    _require(bootstrap_replicates >= 1, "bootstrap replicate count must be positive")
    payloads = [
        _load(Path(run_root) / f"seed_{seed}" / "full" / "rf100.json")
        for seed in DEFAULT_SEEDS
    ]
    reference_rows = payloads[0]["dataset"]["per_sample_condition"]
    keys = [(str(row["sample_id"]), str(row["condition"])) for row in reference_rows]
    targets = {
        key: float(row["normalized_target"])
        for key, row in zip(keys, reference_rows, strict=True)
    }
    groups = [f"rf100:{row['group_id']}" for row in reference_rows]
    conditions = np.asarray([str(row["condition"]) for row in reference_rows])
    _require(len(keys) == 151 * len(CONDITIONS), "RF100 row count differs")
    _require(len(set(groups)) == 35, "RF100 group count differs")

    proposed_errors: dict[int, np.ndarray] = {}
    proposed_passed: dict[int, np.ndarray] = {}
    for seed, payload in zip(DEFAULT_SEEDS, payloads, strict=True):
        rows = payload["dataset"]["per_sample_condition"]
        observed_keys = [
            (str(row["sample_id"]), str(row["condition"])) for row in rows
        ]
        _require(observed_keys == keys, f"GeoPRR-Net seed {seed} row roster differs")
        proposed_errors[seed] = np.asarray(
            [float(row["candidate"]["mett"]["absolute_error"]) for row in rows],
            dtype=np.float64,
        )
        proposed_passed[seed] = np.asarray(
            [row["candidate"]["mett"]["status"] == "pass" for row in rows],
            dtype=np.bool_,
        )

    baseline_errors, baseline_passed, provenance = _load_raw_external_dataset(
        external_root=Path(external_root),
        dataset_key="rf100",
        keys=keys,
        targets=targets,
    )

    all_mask = np.ones(len(keys), dtype=np.bool_)
    masks = {"all_conditions": all_mask}
    masks.update({condition: conditions == condition for condition in CONDITIONS})
    methods: dict[str, Any] = {
        "GeoPRR-Net": {
            name: _metric_summary(proposed_errors, proposed_passed, mask)
            for name, mask in masks.items()
        }
    }
    for display_name in RAW_EXTERNAL_MODELS:
        methods[f"Raw {display_name}"] = {
            name: _metric_summary(
                baseline_errors[display_name], baseline_passed[display_name], mask
            )
            for name, mask in masks.items()
        }

    proposed_mean = np.mean(
        np.stack([proposed_errors[seed] for seed in DEFAULT_SEEDS], axis=0), axis=0
    )
    comparisons: dict[str, Any] = {}
    for index, display_name in enumerate(RAW_EXTERNAL_MODELS, start=1):
        baseline_mean = np.mean(
            np.stack(
                [baseline_errors[display_name][seed] for seed in DEFAULT_SEEDS],
                axis=0,
            ),
            axis=0,
        )
        method_comparisons: dict[str, Any] = {}
        for offset, (name, mask) in enumerate(masks.items()):
            comparison = _paired_group_bootstrap_fast(
                proposed_mean[mask],
                baseline_mean[mask],
                np.asarray(groups)[mask].tolist(),
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + index * 100 + offset,
            )
            comparison.update(
                {
                    "delta_definition": f"GeoPRR-Net minus Raw {display_name}",
                    "rows": int(mask.sum()),
                    "groups": len(set(np.asarray(groups)[mask].tolist())),
                    "seed_handling": "mean aligned row error over three fitting seeds",
                }
            )
            method_comparisons[name] = comparison
        comparisons[f"GeoPRR-Net_minus_Raw_{display_name}"] = method_comparisons

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "dataset": {
            "name": "RF100-VL needle-base-tip-min-max test split",
            "samples": 151,
            "groups": 35,
            "conditions": list(CONDITIONS),
            "rows_per_seed": len(keys),
            "role": "annotation-derived public external transfer cohort",
            "official_scalar_reading_benchmark": False,
            "target_derivation": (
                "clockwise pointer phase between annotated minimum and maximum "
                "endpoints, normalized to [0,1]"
            ),
        },
        "seeds": list(DEFAULT_SEEDS),
        "methods": methods,
        "paired_group_bootstrap": comparisons,
        "baseline_provenance": provenance,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "conservative RF100 source group",
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# GeoPRR-Net RF100 external-transfer results",
        "",
        "RF100 is used as an annotation-derived external transfer cohort, not as an official scalar-reading leaderboard.",
        "",
        "## All six conditions",
        "",
        "| Method | NMAE mean±SD | Acc@2% mean±SD | Acc@5% mean±SD | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, scopes in report["methods"].items():
        row = scopes["all_conditions"]
        lines.append(
            f"| {method} | {row['nmae']['mean']:.6f}±{row['nmae']['sample_sd']:.6f} "
            f"| {row['acc_at_2_percent']['mean']:.4f}±{row['acc_at_2_percent']['sample_sd']:.4f} "
            f"| {row['acc_at_5_percent']['mean']:.4f}±{row['acc_at_5_percent']['sample_sd']:.4f} "
            f"| {row['coverage']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## GeoPRR-Net NMAE by condition",
            "",
            "| Condition | NMAE mean±SD |",
            "|---|---:|",
        ]
    )
    for condition in CONDITIONS:
        row = report["methods"]["GeoPRR-Net"][condition]["nmae"]
        lines.append(f"| {condition} | {row['mean']:.6f}±{row['sample_sd']:.6f} |")
    lines.extend(
        [
            "",
            "## Paired all-condition NMAE differences",
            "",
            "Negative values favor GeoPRR-Net.",
            "",
            "| Comparison | Delta NMAE | 95% group-bootstrap interval |",
            "|---|---:|---:|",
        ]
    )
    for name, scopes in report["paired_group_bootstrap"].items():
        row = scopes["all_conditions"]
        interval = row["paired_group_bootstrap_ci95"]
        lines.append(
            f"| {name.replace('_', ' ')} "
            f"| {row['delta_nmae_candidate_minus_comparator']:.6f} "
            f"| [{interval['low']:.6f}, {interval['high']:.6f}] |"
        )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = summarize_rf100(
        run_root=args.run_root,
        external_root=args.external_root,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    output.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "nmae": report["methods"]["GeoPRR-Net"]["all_conditions"][
                    "nmae"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "render_markdown", "summarize_rf100"]
