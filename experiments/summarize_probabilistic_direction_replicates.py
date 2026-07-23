"""Summarize three formal probabilistic-direction training/evaluation seeds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.vdn_baseline import sha256_file


DEFAULT_SEEDS = (20260720, 20260721, 20260722)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/runs/probabilistic_pivot_direction_syncg"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/runs/probabilistic_pivot_direction_syncg/replicate_stability.json"
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


def _aggregate(rows: list[dict[str, float]], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "mean": {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        },
        "sample_std": {
            key: float(np.std([row[key] for row in rows], ddof=1)) for key in keys
        },
    }


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
    if len(set(args.seeds)) < 2:
        raise ValueError("replicate stability requires at least two seeds")
    for path in (output, markdown):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    training_rows: list[dict[str, Any]] = []
    evaluations: dict[str, list[dict[str, Any]]] = {"clean": [], "rpm10k": []}
    for seed in args.seeds:
        run = root / f"seed_{seed}"
        verification_path = run / "verification.json"
        verification = _load(verification_path)
        if verification.get("verified") is not True:
            raise ValueError(f"seed {seed} is not formally verified")
        training_rows.append(
            {
                "seed": seed,
                "validation_angle_mae_degrees": float(
                    verification["best_validation_angle_mae_degrees"]
                ),
                "validation_calibration_nll": float(
                    verification["best_validation_angular_calibration_nll"]
                ),
                "validation_pivot_error_fraction": float(
                    verification["best_validation_pivot_error_fraction"]
                ),
                "verification": str(verification_path),
                "verification_sha256": sha256_file(verification_path),
            }
        )
        for condition in evaluations:
            summary_path = run / "evaluations" / f"{condition}.summary.json"
            summary = _load(summary_path)
            evaluations[condition].append(
                {
                    "seed": seed,
                    "nmae": float(summary["metrics"]["nmae"]),
                    "acc_2pct": float(summary["metrics"]["acc_2pct"]),
                    "coverage": float(summary["metrics"]["coverage"]),
                    "summary": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                }
            )
    payload = {
        "schema_version": 1,
        "protocol": "probabilistic_direction_replicate_stability_v1",
        "seeds": list(args.seeds),
        "training": {
            "runs": training_rows,
            **_aggregate(
                training_rows,
                (
                    "validation_angle_mae_degrees",
                    "validation_calibration_nll",
                    "validation_pivot_error_fraction",
                ),
            ),
        },
        "evaluations": {
            condition: {
                "runs": rows,
                **_aggregate(rows, ("nmae", "acc_2pct", "coverage")),
            }
            for condition, rows in evaluations.items()
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_write(
        output,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    lines = [
        "# Probabilistic direction three-seed stability",
        "",
        "| Quantity | Mean ± sample std |",
        "|---|---:|",
        f"| Validation angle MAE (degrees) | "
        f"{payload['training']['mean']['validation_angle_mae_degrees']:.4f} ± "
        f"{payload['training']['sample_std']['validation_angle_mae_degrees']:.4f} |",
    ]
    for condition in ("clean", "rpm10k"):
        value = payload["evaluations"][condition]
        lines.append(
            f"| {condition} NMAE | {value['mean']['nmae']:.4f} ± "
            f"{value['sample_std']['nmae']:.4f} |"
        )
        lines.append(
            f"| {condition} Acc@2% | {value['mean']['acc_2pct']:.4f} ± "
            f"{value['sample_std']['acc_2pct']:.4f} |"
        )
    _atomic_write(markdown, "\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
