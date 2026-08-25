"""Aggregate ReMST-ResNet18 seed evaluations into the paper main table."""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.evaluate_remst_resnet18_syncg import PROTOCOL as EVALUATION_PROTOCOL


PROTOCOL: Final[str] = "remst_resnet18_three_seed_main_table_summary_v1"
CONDITION_ORDER: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
    "all_conditions",
    "projective_pooled",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize_main_table(
    result_paths: Sequence[Path], *, output_path: Path
) -> dict[str, Any]:
    _require(len(result_paths) == 3, "exactly three seed results are required")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"summary output already exists: {output}")

    sources: list[dict[str, Any]] = []
    for result_path in result_paths:
        source = Path(result_path).resolve()
        _require(source.is_file(), f"seed result is missing: {source}")
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        _require(isinstance(payload, Mapping), f"seed result is invalid: {source}")
        _require(
            payload.get("protocol") == EVALUATION_PROTOCOL
            and payload.get("status") == "complete",
            f"seed result protocol/status differs: {source}",
        )
        model = payload.get("model")
        summary = payload.get("summary")
        _require(
            isinstance(model, Mapping) and isinstance(summary, Mapping),
            f"seed result model/summary is missing: {source}",
        )
        source_anchor = model.get("source_anchor")
        _require(
            isinstance(source_anchor, Mapping),
            f"seed result source anchor is missing: {source}",
        )
        sources.append(
            {
                "path": str(source),
                "seed": int(source_anchor["source_seed"]),
                "data": dict(payload["data"]),
                "summary": summary,
            }
        )

    seeds = [int(source["seed"]) for source in sources]
    _require(len(set(seeds)) == len(seeds), "source seeds repeat")
    cohort_identity = sources[0]["data"]
    _require(
        all(source["data"] == cohort_identity for source in sources[1:]),
        "seed evaluation cohorts differ",
    )

    rows: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_ORDER:
        seed_rows = [source["summary"][condition] for source in sources]
        values = [float(row["candidate"]["mett"]["nmae"]) for row in seed_rows]
        external_models = seed_rows[0]["external_models"]
        external_means = {
            str(model): float(metrics["mean_across_seeds"]["nmae"])
            for model, metrics in external_models.items()
        }
        _require(
            all(
                {
                    str(model): float(metrics["mean_across_seeds"]["nmae"])
                    for model, metrics in row["external_models"].items()
                }
                == external_means
                for row in seed_rows[1:]
            ),
            f"external comparison differs across seeds: {condition}",
        )
        external_best_model = min(external_means, key=external_means.__getitem__)
        external_best = external_means[external_best_model]
        candidate_mean = statistics.fmean(values)
        candidate_std = statistics.stdev(values)
        rows[condition] = {
            "per_seed_nmae": dict(zip((str(seed) for seed in seeds), values, strict=True)),
            "mean_nmae": candidate_mean,
            "sample_std_nmae": candidate_std,
            "mean_plus_minus_std": f"{candidate_mean:.6f}±{candidate_std:.6f}",
            "external_seed_mean_nmae": external_means,
            "external_best_model": external_best_model,
            "external_best_mean_nmae": external_best,
            "absolute_margin_below_external_best": external_best - candidate_mean,
            "relative_improvement_percent": 100.0
            * (external_best - candidate_mean)
            / external_best,
            "mean_strictly_better_than_all_external_models": candidate_mean
            < external_best,
            "every_seed_strictly_better_than_external_best_mean": all(
                value < external_best for value in values
            ),
        }

    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "development_cohort": True,
            "historical_holdout_already_used": True,
            "formal_holdout_touched": False,
        },
        "seeds": seeds,
        "source_results": [source["path"] for source in sources],
        "data": cohort_identity,
        "main_table": rows,
        "all_columns_mean_strictly_better_than_all_external_models": all(
            row["mean_strictly_better_than_all_external_models"]
            for row in rows.values()
        ),
        "all_columns_every_seed_strictly_better_than_external_best_mean": all(
            row["every_seed_strictly_better_than_external_best_mean"]
            for row in rows.values()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = summarize_main_table(args.result, output_path=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "seeds": result["seeds"],
                "all_columns_win": result[
                    "all_columns_mean_strictly_better_than_all_external_models"
                ],
                "main_table": {
                    condition: row["mean_plus_minus_std"]
                    for condition, row in result["main_table"].items()
                },
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONDITION_ORDER", "PROTOCOL", "summarize_main_table"]
