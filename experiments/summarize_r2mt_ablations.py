"""Summarize the completed three-seed R2MT component ablations."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np


SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
METRICS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
    "all_conditions",
    "projective_pooled",
)


RUN_ROWS: Final[tuple[tuple[str, str], ...]] = (
    (
        "Mean-risk RCMT",
        "remst_resnet18_relative_context_transport_fullfit40",
    ),
    (
        "Tail-risk specialist",
        "remst_resnet18_relative_context_transport_probe_tail020",
    ),
    (
        "Combined-risk specialist",
        "remst_resnet18_relative_context_transport_probe_tail020_cw_1_1_2",
    ),
    (
        "Adaptive multi-risk routing",
        "remst_resnet18_risk_arbitration_probe",
    ),
    (
        "R2MT-Net (full)",
        "r2mt_final_conservative",
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


def _metric(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_nmae": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "per_seed_nmae": [float(value) for value in array],
    }


def _run_row(root: Path, display_name: str, run_name: str) -> dict[str, Any]:
    payloads = []
    paths = []
    for seed in SEEDS:
        path = (
            root
            / "artifacts"
            / "runs"
            / run_name
            / f"seed_{seed}"
            / "six_condition_results.json"
        )
        paths.append(str(path.resolve()))
        payloads.append(json.loads(path.read_text(encoding="utf-8-sig")))
    return {
        "method": display_name,
        "source_type": "three_seed_evaluation",
        "sources": paths,
        "metrics": {
            name: _metric(
                [
                    float(payload["summary"][name]["candidate"]["mett"]["nmae"])
                    for payload in payloads
                ]
            )
            for name in METRICS
        },
    }


def _probe_row(root: Path) -> dict[str, Any]:
    path = (
        root
        / "artifacts"
        / "probes"
        / "transport_multiseed_three_risk_blend.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    def values(name: str) -> Sequence[float]:
        if name in {"all_conditions", "projective_pooled"}:
            return payload[name]["per_seed_nmae"]
        return payload["conditions"][name]["per_seed_nmae"]

    return {
        "method": "Fixed three-risk prior",
        "source_type": "three_seed_prediction_probe",
        "sources": [str(path.resolve())],
        "weights": dict(payload["weights"]),
        "metrics": {name: _metric(values(name)) for name in METRICS},
    }


def summarize(root: Path, output_path: Path) -> dict[str, Any]:
    rows = [_run_row(root, *row) for row in RUN_ROWS]
    rows.insert(3, _probe_row(root))
    full = next(row for row in rows if row["method"] == "R2MT-Net (full)")
    for row in rows:
        row["delta_vs_full"] = {
            name: (
                row["metrics"][name]["mean_nmae"]
                - full["metrics"][name]["mean_nmae"]
            )
            for name in METRICS
        }
    result = {
        "protocol": "r2mt_component_ablation_summary_v1",
        "status": "complete",
        "formal_holdout_access": False,
        "summary": "arithmetic mean and sample SD across three fitted seeds",
        "seeds": list(SEEDS),
        "metrics": list(METRICS),
        "rows": rows,
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
        default=Path("artifacts/probes/r2mt_ablation_summary.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = summarize(args.root.resolve(), args.output)
    compact = {
        row["method"]: {
            "all_conditions": row["metrics"]["all_conditions"],
            "projective_pooled": row["metrics"]["projective_pooled"],
        }
        for row in result["rows"]
    }
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
