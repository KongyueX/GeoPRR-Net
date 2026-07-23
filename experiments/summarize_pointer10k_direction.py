"""Aggregate Pointer-10K direction runs and make a paired comparison table."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments.evaluate_pointer10k_direction import (
    EVALUATION_PROTOCOL,
    summarize_direction_rows,
)


LOW_QUALITY_GROUP = "natural_low_quality_2of4"
DISPLAY_METRICS = (
    "mean_angle_error_degrees",
    "acc_5deg",
    "acc_10deg",
    "coverage",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _failure_aware_error(row: dict[str, Any]) -> float:
    value = row.get("angle_error_degrees")
    if row.get("status") is not True or value is None:
        return 180.0
    result = float(value)
    return result if math.isfinite(result) else 180.0


def _index_rows(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in result:
            raise ValueError(f"duplicate sample id in Pointer-10K result: {sample_id}")
        result[sample_id] = row
    return result


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate_runs(
    label: str,
    runs: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    if not runs:
        raise ValueError(f"method {label!r} has no result runs")
    indexes = [_index_rows(run) for run in runs]
    sample_ids = sorted(indexes[0])
    for index in indexes[1:]:
        if sorted(index) != sample_ids:
            raise ValueError(f"method {label!r} runs use different sample sets")
    run_metrics = [
        summarize_direction_rows(run, bootstrap_iterations=0)
        for run in runs
    ]
    metrics = {
        name: _mean_std([float(run[name]) for run in run_metrics])
        for name in DISPLAY_METRICS
    }
    per_sample_errors = {
        sample_id: float(
            np.mean(
                [
                    _failure_aware_error(index[sample_id])
                    for index in indexes
                ]
            )
        )
        for sample_id in sample_ids
    }
    low_quality_ids = [
        sample_id
        for sample_id in sample_ids
        if LOW_QUALITY_GROUP
        in (indexes[0][sample_id].get("quality_groups") or [])
    ]
    low_quality_run_metrics = []
    for run in runs:
        selected = [
            row for row in run if LOW_QUALITY_GROUP in (row.get("quality_groups") or [])
        ]
        if selected:
            low_quality_run_metrics.append(
                summarize_direction_rows(selected, bootstrap_iterations=0)
            )
    low_quality = (
        {
            name: _mean_std([float(run[name]) for run in low_quality_run_metrics])
            for name in DISPLAY_METRICS
        }
        if low_quality_run_metrics
        else None
    )
    return {
        "label": label,
        "runs": len(runs),
        "samples": len(sample_ids),
        "sample_ids": sample_ids,
        "per_sample_errors": per_sample_errors,
        "metrics": metrics,
        "run_metrics": run_metrics,
        "low_quality_group": LOW_QUALITY_GROUP,
        "low_quality_samples": len(low_quality_ids),
        "low_quality_sample_ids": low_quality_ids,
        "low_quality_metrics": low_quality,
    }


def paired_bootstrap_comparison(
    method: dict[str, Any],
    baseline: dict[str, Any],
    *,
    iterations: int,
    seed: int,
    sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    method_errors = method["per_sample_errors"]
    baseline_errors = baseline["per_sample_errors"]
    ids = list(sample_ids) if sample_ids is not None else sorted(method_errors)
    if set(method_errors) != set(baseline_errors):
        raise ValueError("paired methods use different Pointer-10K samples")
    if not ids:
        raise ValueError("paired comparison has no samples")
    delta = np.asarray(
        [float(method_errors[item]) - float(baseline_errors[item]) for item in ids],
        dtype=np.float64,
    )
    interval = None
    if iterations > 0:
        rng = np.random.default_rng(seed)
        estimates = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            sampled = rng.integers(0, delta.size, size=delta.size)
            estimates[index] = float(np.mean(delta[sampled]))
        low, high = np.quantile(estimates, [0.025, 0.975])
        interval = [float(low), float(high)]
    return {
        "samples": len(ids),
        "delta_mean_angle_degrees_method_minus_baseline": float(np.mean(delta)),
        "paired_bootstrap_95ci": interval,
        "method_win_rate": float(np.mean(delta < 0.0)),
        "tie_rate": float(np.mean(delta == 0.0)),
    }


def build_pairwise_comparisons(
    aggregates: Sequence[dict[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Compare every later method with every earlier method using paired images."""
    comparisons: dict[str, dict[str, Any]] = {}
    offset = 0
    for baseline_index, baseline in enumerate(aggregates):
        for method in aggregates[baseline_index + 1 :]:
            offset += 1
            comparison = paired_bootstrap_comparison(
                method,
                baseline,
                iterations=iterations,
                seed=seed + offset,
            )
            comparison["method"] = method["label"]
            comparison["baseline"] = baseline["label"]
            comparison["low_quality"] = paired_bootstrap_comparison(
                method,
                baseline,
                iterations=iterations,
                seed=seed + 10_000 + offset,
                sample_ids=method["low_quality_sample_ids"],
            )
            comparisons[f"{method['label']} vs {baseline['label']}"] = comparison
    return comparisons


def _parse_method(specification: str) -> tuple[str, list[Path]]:
    if "=" not in specification:
        raise ValueError("--method must use LABEL=run1.jsonl[,run2.jsonl]")
    label, raw_paths = specification.split("=", 1)
    label = label.strip()
    paths = [Path(value.strip()).resolve() for value in raw_paths.split(",") if value.strip()]
    if not label or not paths:
        raise ValueError("--method must contain a non-empty label and path list")
    return label, paths


def _load_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata_path = _metadata_path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    signature = metadata.get("signature") or {}
    if signature.get("protocol") != EVALUATION_PROTOCOL:
        raise ValueError(f"{path} is not a Pointer-10K direction evaluation")
    if signature.get("diagnostic_limit") is not None:
        raise ValueError(f"{path} is a diagnostic --limit run, not a formal result")
    if signature.get("zero_shot_external_test") is not True:
        raise ValueError(f"{path} is not signed as zero-shot external testing")
    expected_result_hash = signature.get("result_sha256")
    if not isinstance(expected_result_hash, str) or len(expected_result_hash) != 64:
        raise ValueError(f"{path} metadata lacks a signed result SHA-256")
    actual_result_hash = _sha256(path)
    if actual_result_hash != expected_result_hash:
        raise ValueError(
            f"{path} result SHA-256 is {actual_result_hash}, "
            f"expected {expected_result_hash}"
        )
    return _read_jsonl(path), signature


def _format_mean_std(value: dict[str, float], *, percent: bool = False) -> str:
    factor = 100.0 if percent else 1.0
    mean = factor * float(value["mean"])
    std = factor * float(value["std"])
    if std > 0.0:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def _format_interval(value: Sequence[float] | None) -> str:
    if value is None:
        return "n/a"
    return f"[{float(value[0]):+.3f}, {float(value[1]):+.3f}]"


def _markdown_report(
    aggregates: Sequence[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    *,
    baseline_label: str,
) -> str:
    lines = [
        "# Pointer-10K 零样本单指针方向对比",
        "",
        "固定协议：官方 test 中预先筛出的 438 张单指针图像；使用官方表盘框；"
        "所有模型仅在 SyncG train 训练，Pointer-10K 训练样本使用量为 0。",
        "",
        "| 方法 | 独立训练数 | 角度 MAE↓ | Acc@5°↑ | Acc@10°↑ | 覆盖率↑ | 自然低质量组 MAE↓ | "
        f"相对 {baseline_label} 的 ΔMAE（95% CI） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for aggregate in aggregates:
        label = aggregate["label"]
        metrics = aggregate["metrics"]
        low_quality = aggregate["low_quality_metrics"]
        comparison = comparisons.get(label)
        delta = (
            f"{comparison['delta_mean_angle_degrees_method_minus_baseline']:+.3f} "
            f"{_format_interval(comparison['paired_bootstrap_95ci'])}"
            if comparison is not None
            else "baseline"
        )
        low_quality_mae = (
            _format_mean_std(low_quality["mean_angle_error_degrees"])
            if low_quality is not None
            else "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(aggregate["runs"]),
                    _format_mean_std(metrics["mean_angle_error_degrees"]),
                    _format_mean_std(metrics["acc_5deg"], percent=True) + "%",
                    _format_mean_std(metrics["acc_10deg"], percent=True) + "%",
                    _format_mean_std(metrics["coverage"], percent=True) + "%",
                    low_quality_mae,
                    delta,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"基线：{baseline_label}。自然低质量组定义为模糊、低照度、低对比、"
            "小表盘四个预测无关 Q1 条件中至少命中两个。",
            "",
            "该表是跨域、单指针方向部件实验，不是 Pointer-10K 完整多指针 "
            "OKS/VDS 排名，也不是最终标量读数结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="LABEL=run1.jsonl[,run2.jsonl]; repeat for each method",
    )
    parser.add_argument("--baseline-label", default="VDN")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 0:
        raise ValueError("bootstrap iterations must be non-negative")
    parsed = [_parse_method(value) for value in args.method]
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("duplicate --method label")

    aggregates = []
    signatures: dict[str, list[dict[str, Any]]] = {}
    common_manifest_hash: str | None = None
    for label, paths in parsed:
        runs = []
        signatures[label] = []
        for path in paths:
            rows, signature = _load_run(path)
            manifest_hash = str(signature.get("manifest_sha256"))
            if common_manifest_hash is None:
                common_manifest_hash = manifest_hash
            elif manifest_hash != common_manifest_hash:
                raise ValueError("Pointer-10K methods use different manifests")
            runs.append(rows)
            signatures[label].append(signature)
        aggregates.append(aggregate_runs(label, runs))

    by_label = {item["label"]: item for item in aggregates}
    if args.baseline_label not in by_label:
        raise ValueError(f"baseline label {args.baseline_label!r} was not provided")
    baseline = by_label[args.baseline_label]
    comparisons: dict[str, dict[str, Any]] = {}
    for offset, aggregate in enumerate(aggregates, 1):
        if aggregate["label"] == args.baseline_label:
            continue
        comparison = paired_bootstrap_comparison(
            aggregate,
            baseline,
            iterations=args.bootstrap_iterations,
            seed=args.seed + offset,
        )
        comparison["low_quality"] = paired_bootstrap_comparison(
            aggregate,
            baseline,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 100 + offset,
            sample_ids=aggregate["low_quality_sample_ids"],
        )
        comparisons[aggregate["label"]] = comparison
    pairwise_comparisons = build_pairwise_comparisons(
        aggregates,
        iterations=args.bootstrap_iterations,
        seed=args.seed + 20_000,
    )

    payload = {
        "protocol": "pointer10k_direction_comparison_v1",
        "baseline": args.baseline_label,
        "manifest_sha256": common_manifest_hash,
        "methods": aggregates,
        "paired_comparisons": comparisons,
        "all_pairwise_comparisons": pairwise_comparisons,
        "signatures": signatures,
        "interpretation": (
            "zero-shot single-pointer component comparison; not the official "
            "full-test multi-pointer OKS/VDS leaderboard"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        _markdown_report(
            aggregates,
            comparisons,
            baseline_label=args.baseline_label,
        ),
        encoding="utf-8",
    )
    print(args.output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
