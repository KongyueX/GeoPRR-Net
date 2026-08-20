"""Summarize the four-arm Support-Normalized CBAM pilot with paired CIs.

The four arms are the frozen DB-GAR18 parent, the integrated SN-CBAM pilot,
and each model behind the same frozen SARN-v2 frontend.  This post-hoc utility
joins label-free prediction JSONL with an explicit target roster, verifies that
all arms used identical degraded pixels, and performs paired group bootstrap on
candidate-minus-comparator absolute normalized-progress error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final


CONDITIONS: Final[tuple[str, ...]] = (
    "clean",
    "blur_moderate",
    "blur_severe",
    "perspective_moderate",
    "perspective_severe",
    "combined_severe",
)
FAILURE_ERROR: Final[float] = 1.0
BOOTSTRAP_REPLICATES: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 20260812
_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {"ground_truth", "normalized_target", "target", "label", "labels", "group_id"}
)


class PilotSummaryError(ValueError):
    """Raised when a pilot artifact violates the frozen comparison contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotSummaryError(message)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    source = Path(path).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            _require(isinstance(row, dict), f"{source}:{line_number} is not an object")
            rows.append(row)
    _require(bool(rows), f"{source} is empty")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, *, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    result = float(value)
    _require(math.isfinite(result), label)
    return result


def load_real_targets(labels_path: Path, sample_ids_path: Path) -> dict[str, tuple[float, str]]:
    requested = _read_json(sample_ids_path)
    _require(isinstance(requested, list) and requested, "real-photo sample IDs are invalid")
    requested_ids = [str(value) for value in requested]
    requested_set = set(requested_ids)
    _require(len(requested_ids) == len(requested_set), "duplicate real-photo sample ID")
    targets: dict[str, tuple[float, str]] = {}
    for row in _read_jsonl(labels_path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in requested_set:
            continue
        _require(sample_id not in targets, f"duplicate real-photo label: {sample_id}")
        target = _finite_float(
            row.get("normalized_progress"), label="invalid real-photo target"
        )
        group_id = str(row.get("group_id", ""))
        _require(bool(group_id), f"empty real-photo group: {sample_id}")
        targets[sample_id] = (target, group_id)
    _require(
        set(targets) == requested_set,
        "real-photo labels do not match sample roster",
    )
    return {sample_id: targets[sample_id] for sample_id in requested_ids}


def load_scene_targets(manifest_path: Path, split_path: Path) -> dict[str, tuple[float, str]]:
    split = _read_json(split_path)
    _require(isinstance(split, Mapping), "scene split is invalid")
    requested = split.get("validation_sample_ids")
    _require(isinstance(requested, list) and requested, "scene validation roster is invalid")
    requested_ids = [str(value) for value in requested]
    requested_set = set(requested_ids)
    _require(len(requested_ids) == len(requested_set), "duplicate scene sample ID")
    targets: dict[str, tuple[float, str]] = {}
    for row in _read_jsonl(manifest_path):
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in requested_set:
            continue
        ground_truth = _finite_float(row.get("ground_truth"), label="invalid scene GT")
        scale_start = _finite_float(row.get("scale_start"), label="invalid scene scale start")
        scale_end = _finite_float(row.get("scale_end"), label="invalid scene scale end")
        _require(scale_end > scale_start, f"invalid scene scale: {sample_id}")
        target = (ground_truth - scale_start) / (scale_end - scale_start)
        _require(0.0 <= target <= 1.0, f"scene target outside [0,1]: {sample_id}")
        metadata = row.get("metadata")
        _require(isinstance(metadata, Mapping), f"missing scene metadata: {sample_id}")
        scene_name = str(metadata.get("scene_name", ""))
        _require(bool(scene_name), f"empty scene name: {sample_id}")
        scene_stem = Path(scene_name).stem
        _require(bool(scene_stem), f"empty scene stem: {sample_id}")
        targets[sample_id] = (target, scene_stem)
    _require(set(targets) == requested_set, "scene manifest does not match validation roster")
    return {sample_id: targets[sample_id] for sample_id in requested_ids}


def load_predictions(
    paths: Sequence[Path],
    *,
    sample_ids: set[str],
) -> tuple[dict[str, dict[tuple[str, str], tuple[float, bool]]], dict[str, str]]:
    methods: dict[str, dict[tuple[str, str], tuple[float, bool]]] = {}
    pixel_hashes: dict[tuple[str, str], str] = {}
    bindings: dict[str, str] = {}
    for path in paths:
        source = Path(path).resolve(strict=True)
        bindings[str(source)] = _sha256_file(source)
        file_methods: set[str] = set()
        for row_index, row in enumerate(_read_jsonl(source), start=1):
            leaked = _FORBIDDEN_KEYS & set(row)
            _require(not leaked, f"label field(s) in prediction {source}:{row_index}: {leaked}")
            sample_id = str(row.get("sample_id", ""))
            method = str(row.get("method", ""))
            condition = str(row.get("condition", ""))
            status = str(row.get("status", ""))
            _require(sample_id in sample_ids, f"unexpected sample: {sample_id}")
            _require(condition in CONDITIONS, f"unexpected condition: {condition}")
            _require(status in {"pass", "fail"}, f"invalid status: {status}")
            file_methods.add(method)
            passed = status == "pass"
            if passed:
                progress = _finite_float(
                    row.get("normalized_progress"), label="invalid normalized progress"
                )
                _require(0.0 <= progress <= 1.0, "normalized progress outside [0,1]")
            else:
                _require(row.get("normalized_progress") is None, "failed row has progress")
                progress = FAILURE_ERROR
            key = (sample_id, condition)
            method_rows = methods.setdefault(method, {})
            _require(key not in method_rows, f"duplicate prediction: {method}/{key}")
            method_rows[key] = (progress, passed)
            pixel_hash = str(row.get("condition_pixel_sha256", ""))
            _require(len(pixel_hash) == 64, f"invalid condition hash: {method}/{key}")
            if key in pixel_hashes:
                _require(pixel_hashes[key] == pixel_hash, f"pixel mismatch across arms: {key}")
            else:
                pixel_hashes[key] = pixel_hash
        _require(len(file_methods) == 1, f"prediction file mixes methods: {source}")
    expected_keys = {(sample_id, condition) for sample_id in sample_ids for condition in CONDITIONS}
    for method, rows in methods.items():
        _require(set(rows) == expected_keys, f"incomplete Cartesian predictions: {method}")
    _require(len(methods) == len(paths), "method identity collision across files")
    return methods, bindings


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_group_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    groups: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    _require(
        len(candidate_errors) == len(comparator_errors) == len(groups) and bool(groups),
        "paired bootstrap arrays are misaligned",
    )
    group_rows: dict[str, list[int]] = {}
    for index, group_id in enumerate(groups):
        group_rows.setdefault(group_id, []).append(index)
    group_ids = sorted(group_rows)
    _require(len(group_ids) >= 2, "paired bootstrap needs at least two groups")
    deltas = [candidate - comparator for candidate, comparator in zip(
        candidate_errors, comparator_errors, strict=True
    )]
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        sampled: list[int] = []
        for _group in group_ids:
            sampled.extend(group_rows[rng.choice(group_ids)])
        draws.append(sum(deltas[index] for index in sampled) / len(sampled))
    return {
        "delta_nmae_candidate_minus_comparator": sum(deltas) / len(deltas),
        "paired_group_bootstrap_ci95": {
            "low": _quantile(draws, 0.025),
            "high": _quantile(draws, 0.975),
        },
        "groups": len(group_ids),
        "replicates": replicates,
        "seed": seed,
    }


def summarize(
    *,
    dataset: str,
    targets: Mapping[str, tuple[float, str]],
    prediction_paths: Sequence[Path],
    comparisons: Sequence[tuple[str, str]],
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    sample_ids = list(targets)
    methods, bindings = load_predictions(prediction_paths, sample_ids=set(sample_ids))
    errors: dict[tuple[str, str], list[float]] = {}
    metrics: list[dict[str, Any]] = []
    groups = [targets[sample_id][1] for sample_id in sample_ids]
    for method in sorted(methods):
        for condition in CONDITIONS:
            values: list[float] = []
            passes = 0
            for sample_id in sample_ids:
                prediction, passed = methods[method][(sample_id, condition)]
                target = targets[sample_id][0]
                values.append(abs(prediction - target) if passed else FAILURE_ERROR)
                passes += int(passed)
            errors[(method, condition)] = values
            metrics.append(
                {
                    "method": method,
                    "condition": condition,
                    "samples": len(values),
                    "groups": len(set(groups)),
                    "nmae": sum(values) / len(values),
                    "coverage": passes / len(values),
                }
            )
    pairwise: list[dict[str, Any]] = []
    for pair_index, (candidate, comparator) in enumerate(comparisons):
        _require(candidate in methods and comparator in methods, f"unknown comparison: {candidate}/{comparator}")
        for condition_index, condition in enumerate(CONDITIONS):
            value = paired_group_bootstrap(
                errors[(candidate, condition)],
                errors[(comparator, condition)],
                groups,
                replicates=replicates,
                seed=seed + pair_index * len(CONDITIONS) + condition_index,
            )
            pairwise.append(
                {
                    "candidate": candidate,
                    "comparator": comparator,
                    "condition": condition,
                    **value,
                }
            )
    return {
        "schema_version": 1,
        "protocol": "support_normalized_cbam_four_arm_pilot_summary_v1",
        "status": "complete",
        "dataset": dataset,
        "samples": len(sample_ids),
        "groups": len(set(groups)),
        "conditions": list(CONDITIONS),
        "pixel_identity_across_arms": True,
        "prediction_bindings": bindings,
        "method_condition_metrics": metrics,
        "paired_comparisons": pairwise,
        "interpretation_boundary": (
            "single-seed diagnostic; candidate selection and formal paper claims require "
            "a frozen multi-seed protocol"
        ),
    }


def _comparison(value: str) -> tuple[str, str]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("comparison must be CANDIDATE:COMPARATOR")
    return parts[0], parts[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("xm2", "rf100", "scene"), required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--comparisons", type=_comparison, nargs="+", required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(args.bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    if args.dataset in {"xm2", "rf100"}:
        _require(
            args.labels is not None and args.sample_ids is not None,
            "real-photo scoring needs labels and sample IDs",
        )
        targets = load_real_targets(args.labels, args.sample_ids)
    else:
        _require(args.manifest is not None and args.split is not None, "scene needs manifest and split")
        targets = load_scene_targets(args.manifest, args.split)
    value = summarize(
        dataset=args.dataset,
        targets=targets,
        prediction_paths=args.predictions,
        comparisons=args.comparisons,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    output = Path(args.output).resolve()
    _require(output not in {Path(path).resolve() for path in args.predictions}, "output would overwrite input")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    output.write_text(payload, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "complete", "output": str(output), "sha256": _sha256_file(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PilotSummaryError",
    "load_predictions",
    "load_scene_targets",
    "load_real_targets",
    "paired_group_bootstrap",
    "summarize",
]
