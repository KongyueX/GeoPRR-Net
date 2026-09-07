"""Summarize three-seed SyncG and real-photo ROI geometry comparisons."""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments import run_cagh_v5_plain_paper_batch as primary_batch
from experiments.roi_geometry_comparison import write_json
from experiments.roi_geometry_field import FIELD_DATASETS


SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
SYNCG_SLUG: Final[str] = "syncg_scene_holdout"
INDUSTRIAL_SLUG: Final[str] = "industrial_pooled"
INDUSTRIAL_FIELD_SLUGS: Final[tuple[str, ...]] = tuple(
    slug for slug in FIELD_DATASETS if slug != "rf100"
)
DATASET_NAMES: Final[dict[str, str]] = {
    SYNCG_SLUG: "SyncG Scene-Holdout",
    **{slug: value.paper_name for slug, value in FIELD_DATASETS.items()},
}
METHOD_NAMES: Final[dict[str, str]] = {
    "geoprr": "GeoPRR-Net",
    "deeplabv3plus_roi": "DeepLabV3+-ROI",
    "yolo11s_pose4kp": "YOLO11s-Pose-4KP",
    "vdn": "VDN",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    with Path(path).resolve().open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            values.append(value)
    if not values:
        raise ValueError(f"prediction ledger is empty: {path}")
    return tuple(values)


def _new_prediction_path(
    run_root: Path,
    missing_root: Path,
    vdn_matched_root: Path,
    *,
    rf100_missing_root: Path,
    method: str,
    seed: int,
    dataset_slug: str,
) -> Path:
    base = Path(run_root) / f"seed_{seed}"
    if method == "deeplabv3plus_roi":
        method_dir = base / "deeplabv3plus_roi"
        if dataset_slug == SYNCG_SLUG:
            return method_dir / "evaluation" / "predictions.jsonl"
        if dataset_slug == "rf100":
            return method_dir / "field" / "rf100" / "predictions.jsonl"
        if dataset_slug in INDUSTRIAL_FIELD_SLUGS:
            return (
                Path(missing_root)
                / f"seed_{seed}"
                / "deeplabv3plus_roi_auto_geometry"
                / "field"
                / dataset_slug
                / "predictions.jsonl"
            )
    if method == "yolo11s_pose4kp":
        method_dir = base / "yolo11s_pose4kp"
        if dataset_slug == SYNCG_SLUG:
            return method_dir / "evaluation_square" / "predictions.jsonl"
        if dataset_slug in FIELD_DATASETS:
            return method_dir / "field" / dataset_slug / "predictions.jsonl"
    if method == "vdn":
        if dataset_slug == SYNCG_SLUG:
            return (
                Path(vdn_matched_root)
                / f"seed_{seed}"
                / "syncg_predictions.jsonl"
            )
        if dataset_slug in FIELD_DATASETS:
            return (
                Path(rf100_missing_root if dataset_slug == "rf100" else missing_root)
                / f"seed_{seed}"
                / "vdn"
                / "field"
                / dataset_slug
                / "predictions.jsonl"
            )
    raise ValueError(f"no prediction path for {method}/{dataset_slug}")


def _normalized_new_rows(path: Path) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for row in _jsonl(path):
        passed = row.get("status") == "pass"
        normalized.append(
            {
                "sample_id": str(row["sample_id"]),
                "condition": str(row["condition"]),
                "group_id": str(row.get("group_id") or row.get("scene_stem") or ""),
                "status": "pass" if passed else "fail",
                "absolute_error": float(row["absolute_error"]) if passed else 1.0,
            }
        )
    return tuple(normalized)


def _normalized_vdn_syncg_rows(
    prediction_path: Path, reference_path: Path
) -> tuple[dict[str, Any], ...]:
    payload = _json(reference_path)
    source_rows = payload.get("per_sample_condition")
    if not isinstance(source_rows, list):
        raise ValueError("SyncG VDN pixel reference lacks per-sample rows")
    reference: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise ValueError("SyncG VDN pixel reference row is invalid")
        key = (str(row.get("sample_id") or ""), str(row.get("condition") or ""))
        if not all(key) or key in reference:
            raise ValueError(f"SyncG VDN pixel reference key is invalid: {key}")
        reference[key] = row

    predictions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in _jsonl(prediction_path):
        key = (str(row.get("sample_id") or ""), str(row.get("condition") or ""))
        if not all(key) or key in predictions:
            raise ValueError(f"SyncG VDN prediction key is invalid: {key}")
        predictions[key] = row
    if set(predictions) != set(reference):
        raise ValueError("SyncG VDN prediction and reference rosters differ")

    normalized: list[dict[str, Any]] = []
    for key in sorted(reference):
        row = predictions[key]
        target_row = reference[key]
        if row.get("condition_pixel_sha256") != target_row.get(
            "condition_pixel_sha256"
        ):
            raise ValueError(f"SyncG VDN conditioned pixels differ for {key}")
        passed = row.get("status") == "pass"
        prediction = row.get("normalized_progress")
        if passed:
            if prediction is None:
                raise ValueError(f"SyncG VDN pass lacks a prediction for {key}")
            error = abs(float(prediction) - float(target_row["normalized_target"]))
        else:
            error = 1.0
        normalized.append(
            {
                "sample_id": key[0],
                "condition": key[1],
                # Use the shared reference's scene identity so all methods use
                # the same cluster labels in paired scene bootstrap.
                "group_id": str(target_row.get("scene_stem") or ""),
                "status": "pass" if passed else "fail",
                "absolute_error": float(error),
            }
        )
    return tuple(normalized)


def _geoprr_payload_path(root: Path, *, seed: int, dataset_slug: str) -> Path:
    base = Path(root) / f"seed_{seed}" / "full"
    return base / ("syncg.json" if dataset_slug == SYNCG_SLUG else "rf100.json" if dataset_slug == "rf100" else "industrial.json")


def _geoprr_rows(root: Path, *, seed: int, dataset_slug: str) -> tuple[dict[str, Any], ...]:
    payload = _json(_geoprr_payload_path(root, seed=seed, dataset_slug=dataset_slug))
    if dataset_slug == SYNCG_SLUG:
        source_rows = payload["per_sample_condition"]
    elif dataset_slug == "rf100":
        dataset = payload["dataset"]
        source_rows = dataset["per_sample_condition"]
    else:
        dataset = payload["datasets"][dataset_slug]
        source_rows = dataset["per_sample_condition"]
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        prediction = row["mett"] if dataset_slug == SYNCG_SLUG else row["candidate"]["mett"]
        passed = prediction.get("status", "pass") == "pass"
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "condition": str(row["condition"]),
                "group_id": str(row.get("group_id") or row.get("scene_stem") or ""),
                "status": "pass" if passed else "fail",
                "absolute_error": float(prediction["absolute_error"]) if passed else 1.0,
            }
        )
    return tuple(rows)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["absolute_error"]) for row in rows], dtype=np.float64)
    passed = np.asarray([row.get("status") == "pass" for row in rows], dtype=bool)
    if len(errors) == 0 or not np.isfinite(errors).all():
        raise ValueError("metric rows are empty or non-finite")
    return {
        "rows": int(len(errors)),
        "nmae": float(np.mean(errors)),
        "nmae_percent_fs": float(100.0 * np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "acc_at_1_percent_fs": float(np.mean(errors <= 0.01)),
        "acc_at_2_percent_fs": float(np.mean(errors <= 0.02)),
        "acc_at_5_percent_fs": float(np.mean(errors <= 0.05)),
        "coverage": float(np.mean(passed)),
        "failures": int(np.sum(~passed)),
    }


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "per_seed": [float(value) for value in values],
    }


def _aggregate_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "nmae",
        "nmae_percent_fs",
        "rmse",
        "acc_at_1_percent_fs",
        "acc_at_2_percent_fs",
        "acc_at_5_percent_fs",
        "coverage",
        "failures",
    )
    return {
        key: _mean_sd([float(value[key]) for value in values])
        for key in keys
    } | {"rows_per_seed": int(values[0]["rows"])}


def _method_summary(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    if tuple(rows_by_seed) != SEEDS:
        raise ValueError("method summary requires the three declared seeds in order")
    per_seed = {str(seed): _metrics(rows) for seed, rows in rows_by_seed.items()}
    per_condition: dict[str, Any] = {}
    for condition in primary_batch.CONDITIONS:
        values = [
            _metrics([row for row in rows_by_seed[seed] if row["condition"] == condition])
            for seed in SEEDS
        ]
        per_condition[condition] = _aggregate_metrics(values)
    return {
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "all_conditions": _aggregate_metrics(list(per_seed.values())),
        "per_condition": per_condition,
    }


def _rowwise_seed_mean(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]]
) -> dict[tuple[str, str], dict[str, Any]]:
    mappings: list[dict[tuple[str, str], Mapping[str, Any]]] = []
    for seed in SEEDS:
        mapping: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in rows_by_seed[seed]:
            key = (str(row["sample_id"]), str(row["condition"]))
            if key in mapping:
                raise ValueError(f"duplicate prediction row: {key}")
            mapping[key] = row
        mappings.append(mapping)
    keys = set(mappings[0])
    if any(set(mapping) != keys for mapping in mappings[1:]):
        raise ValueError("three-seed prediction rosters differ")
    averaged: dict[tuple[str, str], dict[str, Any]] = {}
    for key in sorted(keys):
        group_ids = {str(mapping[key]["group_id"]) for mapping in mappings}
        if len(group_ids) != 1:
            raise ValueError(f"group identity differs across seeds for {key}")
        averaged[key] = {
            "absolute_error": statistics.fmean(
                float(mapping[key]["absolute_error"]) for mapping in mappings
            ),
            "group_id": group_ids.pop(),
        }
    return averaged


def _pooled_field_rows(
    rows_registry: Mapping[
        str, Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]]
    ],
    *,
    method: str,
) -> dict[int, tuple[dict[str, Any], ...]]:
    """Pool field cohorts while keeping sample and group identities disjoint."""

    pooled: dict[int, tuple[dict[str, Any], ...]] = {}
    for seed in SEEDS:
        combined: list[dict[str, Any]] = []
        for dataset_slug in INDUSTRIAL_FIELD_SLUGS:
            for row in rows_registry[dataset_slug][method][seed]:
                combined.append(
                    {
                        **dict(row),
                        "sample_id": f"{dataset_slug}::{row['sample_id']}",
                        "group_id": f"{dataset_slug}::{row['group_id']}",
                    }
                )
        pooled[seed] = tuple(combined)
    return pooled


def _paired_group_bootstrap(
    candidate: Mapping[tuple[str, str], Mapping[str, Any]],
    comparator: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if set(candidate) != set(comparator):
        raise ValueError("paired method rosters differ")
    selected = set(conditions)
    effects: list[float] = []
    groups: list[str] = []
    for key in sorted(candidate):
        if key[1] not in selected:
            continue
        candidate_row = candidate[key]
        comparator_row = comparator[key]
        if candidate_row["group_id"] != comparator_row["group_id"]:
            raise ValueError(f"paired group differs for {key}")
        effects.append(
            float(candidate_row["absolute_error"])
            - float(comparator_row["absolute_error"])
        )
        groups.append(str(candidate_row["group_id"]))
    group_ids = sorted(set(groups))
    if len(group_ids) < 2:
        raise ValueError("paired bootstrap requires at least two groups")
    index = {group_id: position for position, group_id in enumerate(group_ids)}
    sums = np.zeros(len(group_ids), dtype=np.float64)
    counts = np.zeros(len(group_ids), dtype=np.int64)
    for group, effect in zip(groups, effects, strict=True):
        position = index[group]
        sums[position] += effect
        counts[position] += 1
    rng = np.random.default_rng(int(seed))
    choices = rng.integers(0, len(group_ids), size=(int(replicates), len(group_ids)))
    draws = np.sum(sums[choices], axis=1) / np.sum(counts[choices], axis=1)
    observed = statistics.fmean(effects)
    low, high = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "delta_definition": "GeoPRR-Net NMAE minus comparator NMAE",
        "negative_favors_geoprr": True,
        "observed": observed,
        "observed_percent_fs": 100.0 * observed,
        "ci95_lower": float(low),
        "ci95_upper": float(high),
        "ci95_lower_percent_fs": 100.0 * float(low),
        "ci95_upper_percent_fs": 100.0 * float(high),
        "groups": len(group_ids),
        "rows": len(effects),
        "replicates": int(replicates),
        "geoprr_advantage_ci_excludes_zero": bool(high < 0.0),
    }


def _available_methods(dataset_slug: str) -> tuple[str, ...]:
    return ("geoprr", "deeplabv3plus_roi", "yolo11s_pose4kp", "vdn")


def _load_method_rows(
    *,
    method: str,
    dataset_slug: str,
    run_root: Path,
    geoprr_root: Path,
    missing_root: Path,
    rf100_missing_root: Path,
    vdn_matched_root: Path,
    vdn_pixel_reference: Path,
) -> dict[int, tuple[dict[str, Any], ...]]:
    rows: dict[int, tuple[dict[str, Any], ...]] = {}
    for seed in SEEDS:
        if method == "geoprr":
            rows[seed] = _geoprr_rows(geoprr_root, seed=seed, dataset_slug=dataset_slug)
        elif method == "vdn" and dataset_slug == SYNCG_SLUG:
            rows[seed] = _normalized_vdn_syncg_rows(
                _new_prediction_path(
                    run_root,
                    missing_root,
                    vdn_matched_root,
                    rf100_missing_root=rf100_missing_root,
                    method=method,
                    seed=seed,
                    dataset_slug=dataset_slug,
                ),
                vdn_pixel_reference,
            )
        else:
            rows[seed] = _normalized_new_rows(
                _new_prediction_path(
                    run_root,
                    missing_root,
                    vdn_matched_root,
                    rf100_missing_root=rf100_missing_root,
                    method=method,
                    seed=seed,
                    dataset_slug=dataset_slug,
                )
            )
    return rows


def _industrial_geometry_provenance(missing_root: Path) -> list[dict[str, Any]]:
    cache_paths: set[Path] = set()
    for seed in SEEDS:
        for method in ("deeplabv3plus_roi_auto_geometry", "vdn"):
            for dataset_slug in INDUSTRIAL_FIELD_SLUGS:
                summary_path = (
                    Path(missing_root) / f"seed_{seed}" / method
                    / "field" / dataset_slug / "summary.json"
                )
                configuration = _json(summary_path)["configuration"]
                cache_paths.add(Path(configuration["geometry_cache"]).resolve())
    return [
        {"summary": str(cache.with_suffix(".summary.json")),
         **_json(cache.with_suffix(".summary.json"))}
        for cache in sorted(cache_paths)
    ]


def _format_mean_sd(value: Mapping[str, Any], *, scale: float = 1.0) -> str:
    mean = scale * float(value["mean"])
    sample_sd = value.get("sample_sd")
    return f"{mean:.4f} ± {scale * float(sample_sd):.4f}" if sample_sd is not None else f"{mean:.4f}"


def _render_markdown(summary: Mapping[str, Any]) -> str:
    syncg = summary["datasets"][SYNCG_SLUG]
    industrial = summary["datasets"][INDUSTRIAL_SLUG]
    rf100 = summary["datasets"]["rf100"]
    lines = [
        "# 三种子 ROI 对比实验简报",
        "",
        "所有失败样本均保留在分母中并记为 normalized error 1.0。",
        "Industrial-1395 将三个受限实图来源按样本-条件行合并，每张图等权；不做来源级宏平均。",
        "",
        "| 测试域 | 方法 | 六条件 NMAE (%FS) | clean NMAE (%FS) | Acc@5 (%) | Coverage (%) | GeoPRR−该方法差值及 95% CI (%FS) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset_name, dataset in (
        ("SyncG", syncg),
        ("Industrial-1395", industrial),
        ("RF100-VL", rf100),
    ):
        for method, value in dataset["methods"].items():
            metrics = value["all_conditions"]
            comparison = dataset["paired_comparisons"].get(method)
            if comparison is None:
                paired = "—"
            else:
                all_scope = comparison["all_conditions"]
                paired = (
                    f"{all_scope['observed_percent_fs']:.4f} "
                    f"[{all_scope['ci95_lower_percent_fs']:.4f}, "
                    f"{all_scope['ci95_upper_percent_fs']:.4f}]"
                )
            clean = value["per_condition"]["clean"]["nmae_percent_fs"]
            lines.append(
                f"| {dataset_name} | {METHOD_NAMES[method]} | "
                f"{_format_mean_sd(metrics['nmae_percent_fs'])} | "
                f"{_format_mean_sd(clean)} | "
                f"{_format_mean_sd(metrics['acc_at_5_percent_fs'], scale=100.0)} | "
                f"{_format_mean_sd(metrics['coverage'], scale=100.0)} | {paired} |"
            )

    lines.extend(
        [
            "",
            "## 简要结论",
            "",
            f"- SyncG：GeoPRR-Net 为 {_format_mean_sd(syncg['methods']['geoprr']['all_conditions']['nmae_percent_fs'])}%FS，"
            f"低于 DeepLabV3+-ROI 的 {_format_mean_sd(syncg['methods']['deeplabv3plus_roi']['all_conditions']['nmae_percent_fs'])}%FS "
            f"、YOLO11s-Pose-4KP 的 {_format_mean_sd(syncg['methods']['yolo11s_pose4kp']['all_conditions']['nmae_percent_fs'])}%FS "
            f"和 VDN 的 {_format_mean_sd(syncg['methods']['vdn']['all_conditions']['nmae_percent_fs'])}%FS。",
            f"- Industrial-1395（1,395 张、52 个独立组）：GeoPRR-Net 为 {_format_mean_sd(industrial['methods']['geoprr']['all_conditions']['nmae_percent_fs'])}%FS，"
            f"DeepLabV3+-ROI、YOLO11s-Pose-4KP 和 VDN 分别为 "
            f"{_format_mean_sd(industrial['methods']['deeplabv3plus_roi']['all_conditions']['nmae_percent_fs'])}%FS、"
            f"{_format_mean_sd(industrial['methods']['yolo11s_pose4kp']['all_conditions']['nmae_percent_fs'])}%FS 和 "
            f"{_format_mean_sd(industrial['methods']['vdn']['all_conditions']['nmae_percent_fs'])}%FS。",
            f"- RF100-VL：GeoPRR-Net、DeepLabV3+-ROI、YOLO11s-Pose-4KP 和 VDN 分别为 "
            f"{_format_mean_sd(rf100['methods']['geoprr']['all_conditions']['nmae_percent_fs'])}%FS、"
            f"{_format_mean_sd(rf100['methods']['deeplabv3plus_roi']['all_conditions']['nmae_percent_fs'])}%FS、"
            f"{_format_mean_sd(rf100['methods']['yolo11s_pose4kp']['all_conditions']['nmae_percent_fs'])}%FS、"
            f"{_format_mean_sd(rf100['methods']['vdn']['all_conditions']['nmae_percent_fs'])}%FS。",
            f"- Industrial 的 DeepLab 与 VDN 的实际参考点来源：{summary['industrial_geometry_scope']}。"
            "SyncG/RF100 的两者均为标注几何辅助组件。Under Pressure 因任务口径不同未并入同表。",
            "",
            f"均值后的 ± 为三种子样本标准差；差值区间按 scene/capture group 做 {summary['bootstrap_replicates']:,} 次 paired bootstrap。",
            "子 cohort 明细仅保留在 `summary.json` 的 `audit_subcohorts` 中用于审计，不作为主数据集展示。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    rows_registry: dict[
        str, dict[str, dict[int, tuple[dict[str, Any], ...]]]
    ] = {}
    for dataset_index, dataset_slug in enumerate(DATASET_NAMES):
        methods: dict[str, Any] = {}
        rows_by_method: dict[str, dict[int, tuple[dict[str, Any], ...]]] = {}
        for method in _available_methods(dataset_slug):
            rows = _load_method_rows(
                method=method,
                dataset_slug=dataset_slug,
                run_root=args.run_root,
                geoprr_root=args.geoprr_root,
                missing_root=args.missing_root,
                rf100_missing_root=args.rf100_missing_root,
                vdn_matched_root=args.vdn_matched_root,
                vdn_pixel_reference=args.vdn_pixel_reference,
            )
            rows_by_method[method] = rows
            methods[method] = _method_summary(rows)
        rows_registry[dataset_slug] = rows_by_method
        paired: dict[str, Any] = {}
        geoprr_mean = _rowwise_seed_mean(rows_by_method["geoprr"])
        for method_index, method in enumerate(methods):
            if method == "geoprr":
                continue
            comparator_mean = _rowwise_seed_mean(rows_by_method[method])
            scopes = {
                "all_conditions": tuple(primary_batch.CONDITIONS),
                **{condition: (condition,) for condition in primary_batch.CONDITIONS},
            }
            paired[method] = {
                scope: _paired_group_bootstrap(
                    geoprr_mean,
                    comparator_mean,
                    conditions=conditions,
                    replicates=int(args.bootstrap_replicates),
                    seed=int(args.bootstrap_seed + 100 * dataset_index + 10 * method_index + scope_index),
                )
                for scope_index, (scope, conditions) in enumerate(scopes.items())
            }
        datasets[dataset_slug] = {
            "paper_name": DATASET_NAMES[dataset_slug],
            "samples": len({row["sample_id"] for row in rows_by_method["geoprr"][SEEDS[0]]}),
            "groups": len({row["group_id"] for row in rows_by_method["geoprr"][SEEDS[0]]}),
            "rows_per_seed": len(rows_by_method["geoprr"][SEEDS[0]]),
            "conditions": list(primary_batch.CONDITIONS),
            "evaluation_role": (
                "scene-disjoint synthetic holdout"
                if dataset_slug == SYNCG_SLUG
                else "retrospective real-photo test cohort"
            ),
            "methods": methods,
            "paired_comparisons": paired,
        }

    pooled_rows = {
        method: _pooled_field_rows(rows_registry, method=method)
        for method in ("geoprr", "deeplabv3plus_roi", "yolo11s_pose4kp", "vdn")
    }
    pooled_methods = {
        method: _method_summary(rows)
        for method, rows in pooled_rows.items()
    }
    geoprr_mean = _rowwise_seed_mean(pooled_rows["geoprr"])
    scopes = {
        "all_conditions": tuple(primary_batch.CONDITIONS),
        **{condition: (condition,) for condition in primary_batch.CONDITIONS},
    }
    pooled_bootstrap_offsets = {
        "yolo11s_pose4kp": 0,
        "deeplabv3plus_roi": 100,
        "vdn": 200,
    }
    pooled_paired = {}
    for method in ("deeplabv3plus_roi", "yolo11s_pose4kp", "vdn"):
        comparator_mean = _rowwise_seed_mean(pooled_rows[method])
        pooled_paired[method] = {
            scope: _paired_group_bootstrap(
                geoprr_mean,
                comparator_mean,
                conditions=conditions,
                replicates=int(args.bootstrap_replicates),
                seed=int(
                    args.bootstrap_seed
                    + 10_000
                    + pooled_bootstrap_offsets[method]
                    + scope_index
                ),
            )
            for scope_index, (scope, conditions) in enumerate(scopes.items())
        }
    pooled_reference = pooled_rows["geoprr"][SEEDS[0]]
    datasets[INDUSTRIAL_SLUG] = {
        "paper_name": "Industrial (pooled)",
        "samples": len({row["sample_id"] for row in pooled_reference}),
        "groups": len({row["group_id"] for row in pooled_reference}),
        "rows_per_seed": len(pooled_reference),
        "conditions": list(primary_batch.CONDITIONS),
        "evaluation_role": "pooled retrospective real-photo test cohort",
        "pooling": "sample-condition micro-average; dataset-prefixed group bootstrap",
        "constituent_datasets": list(INDUSTRIAL_FIELD_SLUGS),
        "methods": pooled_methods,
        "paired_comparisons": pooled_paired,
    }
    reported_datasets = {
        SYNCG_SLUG: datasets[SYNCG_SLUG],
        INDUSTRIAL_SLUG: datasets[INDUSTRIAL_SLUG],
        "rf100": datasets["rf100"],
    }
    audit_subcohorts = {
        dataset_slug: datasets[dataset_slug]
        for dataset_slug in INDUSTRIAL_FIELD_SLUGS
    }
    industrial_geometry = _industrial_geometry_provenance(args.missing_root)
    industrial_geometry_scope = "; ".join(
        sorted({str(item["geometry_role"]) for item in industrial_geometry})
    )
    field_scope = (
        "annotation geometry on SyncG/RF100-VL; Industrial-1395: "
        + industrial_geometry_scope
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "roi_geometry_three_seed_real_photo_summary_v2",
        "seeds": list(SEEDS),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "run_root": str(Path(args.run_root).resolve()),
        "geoprr_root": str(Path(args.geoprr_root).resolve()),
        "missing_run_root": str(Path(args.missing_root).resolve()),
        "rf100_missing_run_root": str(Path(args.rf100_missing_root).resolve()),
        "vdn_matched_root": str(Path(args.vdn_matched_root).resolve()),
        "vdn_pixel_reference": str(Path(args.vdn_pixel_reference).resolve()),
        "failure_policy": "failed row receives normalized absolute error 1.0",
        "field_role": "retrospective test only; no field image or label used for training or checkpoint selection",
        "deeplab_field_scope": field_scope,
        "vdn_field_scope": field_scope,
        "industrial_geometry_scope": industrial_geometry_scope,
        "industrial_geometry_provenance": industrial_geometry,
        "datasets": reported_datasets,
        "audit_subcohorts": audit_subcohorts,
    }
    output = Path(args.output).resolve()
    report = Path(args.report).resolve() if args.report else output.with_suffix(".md")
    result["report"] = str(report)
    write_json(output, result)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_render_markdown(result), encoding="utf-8", newline="\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("artifacts/runs/roi_comparison_pilot"))
    parser.add_argument("--geoprr-root", type=Path, default=Path("artifacts/runs/unified_pointer_reader"))
    parser.add_argument(
        "--missing-root",
        type=Path,
        default=Path("artifacts/runs/roi_missing_baselines_three_seed"),
    )
    parser.add_argument(
        "--rf100-missing-root",
        type=Path,
        default=Path("artifacts/runs/roi_missing_baselines_three_seed"),
        help="RF100 VDN predictions; independent of the Industrial --missing-root",
    )
    parser.add_argument(
        "--vdn-matched-root",
        type=Path,
        default=Path("artifacts/runs/geoprr_vdn_matched"),
    )
    parser.add_argument(
        "--vdn-pixel-reference",
        type=Path,
        default=Path(
            "C:/pointer_read/sgca_multiview_pilot_v1/a15_2_syncg_scene_holdout_external.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap-replicates must be positive")
    result = summarize(args)
    print(json.dumps({"status": result["status"], "datasets": list(result["datasets"]), "report": result["report"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "summarize"]
