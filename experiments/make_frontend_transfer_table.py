"""Compare released and SyncG-fine-tuned segmentation on one frozen external set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.selective_experiment import (
    _method_metrics,
    _method_prediction,
    _paired_comparison,
    _prediction_cache_signature,
    _read_jsonl,
)


METHODS = (
    ("transformer", "Original Transformer"),
    ("weighted_fusion", "Quality-weighted Fusion"),
)


def _key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("dataset"), row.get("split"), row.get("sample_id")


def _aligned_rows(
    released_path: Path,
    finetuned_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    released = _read_jsonl(released_path)
    finetuned = _read_jsonl(finetuned_path)
    released_by_key = {_key(row): row for row in released}
    finetuned_by_key = {_key(row): row for row in finetuned}
    if set(released_by_key) != set(finetuned_by_key):
        raise ValueError(
            "released/fine-tuned caches do not contain identical sample keys"
        )
    keys = sorted(released_by_key, key=lambda value: tuple(map(str, value)))
    released_aligned = [released_by_key[key] for key in keys]
    finetuned_aligned = [finetuned_by_key[key] for key in keys]
    for released_row, finetuned_row in zip(
        released_aligned,
        finetuned_aligned,
    ):
        for field in (
            "ground_truth",
            "scale_start",
            "scale_end",
            "group_id",
            "meter_id",
        ):
            if released_row.get(field) != finetuned_row.get(field):
                raise ValueError(
                    f"{_key(released_row)} differs in {field!r} "
                    "between front-end caches"
                )
    return released_aligned, finetuned_aligned


def _verify_only_segmentation_differs(
    released_path: Path,
    finetuned_path: Path,
) -> dict[str, str]:
    released = _prediction_cache_signature(released_path)
    finetuned = _prediction_cache_signature(finetuned_path)
    if released is None or finetuned is None:
        raise ValueError(
            "front-end transfer comparison requires both cache signatures"
        )

    def without_segmentation(signature: dict[str, Any]) -> dict[str, Any]:
        value = json.loads(json.dumps(signature, sort_keys=True))
        weights = dict(value.get("weights_sha256") or {})
        weights.pop("segmentation", None)
        value["weights_sha256"] = weights
        return value

    if without_segmentation(released) != without_segmentation(finetuned):
        raise ValueError(
            "front-end caches differ in more than segmentation weights"
        )
    released_hash = str(
        (released.get("weights_sha256") or {}).get("segmentation") or ""
    )
    finetuned_hash = str(
        (finetuned.get("weights_sha256") or {}).get("segmentation") or ""
    )
    if not released_hash or not finetuned_hash:
        raise ValueError("cache signature is missing segmentation SHA-256")
    if released_hash == finetuned_hash:
        raise ValueError(
            "released and fine-tuned caches use identical segmentation weights"
        )
    return {
        "released_segmentation_sha256": released_hash,
        "finetuned_segmentation_sha256": finetuned_hash,
    }


def summarize_frontend_transfer(
    released_path: Path,
    finetuned_path: Path,
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    segmentation_hashes = _verify_only_segmentation_differs(
        released_path,
        finetuned_path,
    )
    released, finetuned = _aligned_rows(released_path, finetuned_path)
    variants = {
        "Released segmentation": released,
        "SyncG-finetuned segmentation": finetuned,
    }
    metrics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for method_index, (method, label) in enumerate(METHODS):
        released_predictions = [
            _method_prediction(row, method) for row in released
        ]
        finetuned_predictions = [
            _method_prediction(row, method) for row in finetuned
        ]
        metrics[label] = {
            variant: _method_metrics(
                rows,
                [_method_prediction(row, method) for row in rows],
                seed=seed + method_index * 10 + variant_index,
                bootstrap_iterations=bootstrap_iterations,
            )
            for variant_index, (variant, rows) in enumerate(variants.items())
        }
        comparisons[
            f"{label}: fine-tuned minus released"
        ] = _paired_comparison(
            released,
            finetuned_predictions,
            released_predictions,
            seed=seed + 100 + method_index,
            bootstrap_iterations=bootstrap_iterations,
        )
    return {
        "protocol": "frozen_external_frontend_transfer_diagnostic_v1",
        "selection_uses_external_labels": False,
        "model_selection_permitted_from_this_table": False,
        "only_segmentation_checkpoint_differs": True,
        **segmentation_hashes,
        "samples": len(released),
        "released_predictions": str(released_path.resolve()),
        "finetuned_predictions": str(finetuned_path.resolve()),
        "metrics": metrics,
        "paired_comparisons": comparisons,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "| Segmentation front-end | Reading head | E2E NMAE ↓ | "
        "Accε@1% ↑ | Accθ@5% ↑ | Coverage ↑ | Successful Ref ↓ | "
        "Successful Rel ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, variants in summary["metrics"].items():
        for variant, metrics in variants.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        variant,
                        method,
                        _fmt(metrics.get("nmae")),
                        _fmt(metrics.get("dialbench_acc_epsilon_e2e")),
                        _fmt(metrics.get("dialbench_acc_theta_e2e")),
                        _fmt(metrics.get("coverage")),
                        _fmt(metrics.get("dialbench_ref_successful")),
                        _fmt(metrics.get("dialbench_rel_successful")),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "_This frozen external diagnostic tests synthetic-to-real "
            "segmentation transfer only. External labels must not be used to "
            "select the front-end or retune any threshold._",
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released", type=Path, required=True)
    parser.add_argument("--finetuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()

    summary = summarize_frontend_transfer(
        args.released,
        args.finetuned,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(summary), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
