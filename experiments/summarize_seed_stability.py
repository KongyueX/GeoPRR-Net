"""Aggregate the frozen three-seed residual/gate stability experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from experiments.vdn_baseline import sha256_file


CONDITIONS = (
    "syncg_clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
    "rpm10k",
)
METHODS = ("Original Transformer", "Residual without Gate", "Ours")
METRICS = ("nmae", "acc_2pct", "coverage", "negative_transfer_rate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/runs/seed_stability"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260720, 20260721, 20260722],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _calibrator_for_seed(root: Path, seed: int) -> Path:
    if seed == 20260720:
        return root.parent / "syncg_full" / "calibrator.joblib"
    return root / f"seed_{seed}" / "calibrator.joblib"


def _format(value: dict[str, Any], digits: int = 4) -> str:
    return f"{value['mean']:.{digits}f} ± {value['sample_std']:.{digits}f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Three-seed stability",
        "",
        "All values are mean ± sample standard deviation over the three frozen ",
        "residual/gate seeds. The image predictions are shared and unchanged.",
        "",
        "| Condition | Transformer NMAE | Residual w/o gate NMAE | "
        "Ours NMAE | Ours Acc@2% | Ours coverage | Ours negative transfer |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        methods = payload["aggregate"][condition]
        transformer = methods["Original Transformer"]["nmae"]
        residual = methods["Residual without Gate"]["nmae"]
        ours = methods["Ours"]
        lines.append(
            "| "
            + " | ".join(
                (
                    condition,
                    _format(transformer),
                    _format(residual),
                    _format(ours["nmae"]),
                    _format(ours["acc_2pct"]),
                    _format(ours["coverage"]),
                    _format(ours["negative_transfer_rate"]),
                )
            )
            + " |"
        )
    training = payload["training_aggregate"]
    lines.extend(
        [
            "",
            "Training-side gate diagnostics:",
            "",
            f"- Threshold: {_format(training['threshold'])}",
            f"- OOF gate AUROC: {_format(training['gate_auroc'])}",
            f"- OOF gate Brier: {_format(training['gate_brier'])}",
            f"- OOF gate ECE: {_format(training['gate_ece_10_bins'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = (args.output or (root / "seed_stability.json")).resolve()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("at least two distinct seeds are required")

    raw: dict[str, Any] = {}
    training_rows: list[dict[str, float]] = []
    shared_cache_audit: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        seed_root = root / f"seed_{seed}"
        calibrator_path = _calibrator_for_seed(root, seed)
        if not calibrator_path.is_file():
            raise FileNotFoundError(calibrator_path)
        package = joblib.load(calibrator_path)
        training = package.get("training") or {}
        if int(training.get("seed", -1)) != seed:
            raise ValueError(f"{calibrator_path} does not belong to seed {seed}")
        seed_value: dict[str, Any] = {
            "calibrator": str(calibrator_path),
            "calibrator_sha256": sha256_file(calibrator_path),
            "prediction_caches": {},
            "conditions": {},
        }
        training_summary_path = (
            root.parent / "syncg_full" / "training_summary.json"
            if seed == 20260720
            else seed_root / "training_summary.json"
        )
        training_summary = _load_json(training_summary_path)
        gate = training_summary.get("gate") or {}
        training_rows.append(
            {
                "threshold": float(training_summary["threshold"]),
                "gate_auroc": float(gate["auroc"]),
                "gate_brier": float(gate["brier"]),
                "gate_ece_10_bins": float(gate["ece_10_bins"]),
            }
        )
        for condition in CONDITIONS:
            metrics_path = seed_root / condition / "metrics.json"
            metrics = _load_json(metrics_path)
            if metrics.get("front_end_signature_verified") is not True:
                raise ValueError(f"front-end signature was not verified: {metrics_path}")
            recorded_calibrator = Path(str(metrics.get("calibrator") or "")).resolve()
            if recorded_calibrator != calibrator_path.resolve():
                raise ValueError(f"wrong calibrator recorded in {metrics_path}")
            prediction_path = Path(
                str(metrics.get("source_predictions") or "")
            ).resolve()
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            cache_identity = {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "signature": metrics.get("prediction_cache_signature"),
            }
            expected_cache = shared_cache_audit.setdefault(condition, cache_identity)
            if cache_identity != expected_cache:
                raise ValueError(
                    f"{condition} does not reuse one identical prediction cache"
                )
            seed_value["prediction_caches"][condition] = cache_identity
            method_metrics = metrics.get("metrics") or {}
            seed_value["conditions"][condition] = {
                method: {
                    metric: method_metrics[method].get(metric)
                    for metric in METRICS
                }
                for method in METHODS
            }
        raw[str(seed)] = seed_value

    aggregate: dict[str, Any] = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for method in METHODS:
            aggregate[condition][method] = {}
            for metric in METRICS:
                values = [
                    raw[str(seed)]["conditions"][condition][method][metric]
                    for seed in args.seeds
                ]
                finite = [float(value) for value in values if value is not None]
                aggregate[condition][method][metric] = (
                    _stats(finite) if len(finite) == len(values) else None
                )

    training_aggregate = {
        name: _stats([row[name] for row in training_rows])
        for name in training_rows[0]
    }
    payload = {
        "protocol": "three_seed_frozen_frontend_stability_v1",
        "seeds": args.seeds,
        "conditions": list(CONDITIONS),
        "raw": raw,
        "shared_cache_audit": shared_cache_audit,
        "aggregate": aggregate,
        "training_aggregate": training_aggregate,
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, payload)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print(output)
    print(markdown_path)


if __name__ == "__main__":
    main()
