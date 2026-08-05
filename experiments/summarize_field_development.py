"""Independently summarize a verified, frozen field-development evaluation.

The command is statistics-only: it never imports or runs a model, never opens
the sealed confirmatory manifest, and never changes a threshold or selects a
method.  Every accepted prediction file must be hash-bound by the independent
field-development verification report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SUMMARY_PROTOCOL = "field_development_independent_summary_v1"
FREEZE_PROTOCOL = "field_publication_evaluation_freeze_v1"
VERIFICATION_PROTOCOL = "field_development_reference_conditioned_verification_v1"
SPLIT_PROTOCOL = "group_disjoint_field_development_confirmatory_split_v1"
DEVELOPMENT_SPLIT = "field_development"
CONFIRMATORY_SPLIT = "field_confirmatory"
EXPECTED_BOOTSTRAP_ITERATIONS = 5000

METHOD_ORDER = (
    "final",
    "base",
    "vector",
    "reference_conditioned_vector",
    "vdn",
    "original_transformer",
)
METHOD_LABELS_ZH = {
    "final": "最终路由",
    "base": "掩膜基线",
    "vector": "原始概率向量",
    "reference_conditioned_vector": "参考点条件校准向量",
    "vdn": "VDN",
    "original_transformer": "原始 Transformer",
}
VERIFICATION_FILE_KEYS = {
    "raw": "raw",
    "base": "base",
    "vector": "probabilistic",
    "vdn": "vdn",
    "final": "final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--vdn", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_ids_sha256(sample_ids: Iterable[str]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in sample_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _protocol_path(manifest: Path) -> Path:
    return manifest.with_name(manifest.name + ".protocol.json")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resolved_claim(value: Any) -> Path:
    return Path(str(value or "")).resolve()


def _rows_by_id(
    rows: Sequence[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label} row {index} lacks sample_id")
        if sample_id in result:
            raise ValueError(f"{label} contains duplicate sample_id {sample_id!r}")
        result[sample_id] = row
    return result


def _same_number(left: Any, right: Any) -> bool:
    left_number = _finite(left)
    right_number = _finite(right)
    return (
        left_number is not None
        and right_number is not None
        and math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_manifest(
    manifest: Path,
    freeze: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if "confirmatory" in manifest.name.lower() or "sealed" in manifest.name.lower():
        raise PermissionError(
            "REFUSED: a confirmatory/sealed manifest path was supplied"
        )
    protocol_path = _protocol_path(manifest)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = _read_json(protocol_path)
    if (
        protocol.get("split") == CONFIRMATORY_SPLIT
        or protocol.get("confirmatory_sealed") is True
    ):
        raise PermissionError(
            "REFUSED: field_confirmatory/sealed manifests cannot be summarized "
            "by the development command"
        )
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol") != SPLIT_PROTOCOL
        or protocol.get("split") != DEVELOPMENT_SPLIT
        or protocol.get("confirmatory_sealed") is not False
        or protocol.get("assignment_uses_model_predictions") is not False
        or protocol.get("assignment_uses_ground_truth_reading") is not False
        or protocol.get("group_disjoint") is not True
    ):
        raise ValueError("manifest protocol is not a valid field_development split")
    if protocol.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("development manifest hash differs from its protocol")

    manifests = freeze.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("freeze lacks manifest identities")
    development = manifests.get("development")
    confirmatory = manifests.get("confirmatory")
    if not isinstance(development, Mapping) or not isinstance(confirmatory, Mapping):
        raise ValueError("freeze must record development and confirmatory identities")
    confirmatory_path_claim = str(confirmatory.get("path") or "").lower()
    if manifest == _resolved_claim(confirmatory.get("path")) or (
        confirmatory.get("sha256") == sha256_file(manifest)
    ):
        raise PermissionError("REFUSED: supplied manifest is the sealed confirmatory set")
    if "confirmatory" not in confirmatory_path_claim:
        raise ValueError("freeze confirmatory identity is malformed")
    if (
        manifest != _resolved_claim(development.get("path"))
        or development.get("sha256") != sha256_file(manifest)
        or development.get("protocol_sha256") != sha256_file(protocol_path)
    ):
        raise ValueError("development manifest/protocol does not match the freeze")

    rows = _read_jsonl(manifest)
    mapping = _rows_by_id(rows, label="manifest")
    groups = Counter()
    dataset: str | None = None
    quality_memberships: Counter[str] = Counter()
    samples_without_quality_group = 0
    samples_with_multiple_quality_groups = 0
    for line_number, row in enumerate(rows, start=1):
        if row.get("split") != DEVELOPMENT_SPLIT:
            raise PermissionError(
                f"REFUSED: manifest row {line_number} is not field_development"
            )
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        if metadata.get("field_partition") not in (None, DEVELOPMENT_SPLIT):
            raise PermissionError(
                f"REFUSED: manifest row {line_number} has a non-development partition"
            )
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"manifest row {line_number} lacks physical group_id")
        groups[group_id] += 1
        row_dataset = row.get("dataset")
        if not isinstance(row_dataset, str) or not row_dataset:
            raise ValueError(f"manifest row {line_number} lacks dataset")
        if dataset is None:
            dataset = row_dataset
        elif dataset != row_dataset:
            raise ValueError("manifest mixes datasets")
        ground_truth = _finite(row.get("ground_truth"))
        scale_start = _finite(row.get("scale_start"))
        scale_end = _finite(row.get("scale_end"))
        if (
            ground_truth is None
            or scale_start is None
            or scale_end is None
            or abs(scale_end - scale_start) <= 1e-12
        ):
            raise ValueError(f"manifest row {line_number} has invalid label/scale")
        quality = metadata.get("quality")
        quality = quality if isinstance(quality, Mapping) else {}
        quality_groups = quality.get("groups") or []
        if isinstance(quality_groups, str):
            quality_groups = [quality_groups]
        if not isinstance(quality_groups, Sequence):
            raise ValueError(
                f"manifest row {line_number} quality.groups is not an array"
            )
        normalized_groups = sorted(
            {
                str(value)
                for value in quality_groups
                if isinstance(value, str) and value
            }
        )
        if len(normalized_groups) != len(
            [value for value in quality_groups if isinstance(value, str) and value]
        ):
            raise ValueError(
                f"manifest row {line_number} quality.groups has duplicates"
            )
        if not normalized_groups:
            samples_without_quality_group += 1
        if len(normalized_groups) > 1:
            samples_with_multiple_quality_groups += 1
        for quality_group in normalized_groups:
            quality_memberships[quality_group] += 1

    if not rows:
        raise ValueError("development manifest is empty")
    identifiers_hash = sample_ids_sha256(mapping)
    expected_counts = dict(sorted(groups.items()))
    if (
        int(protocol.get("rows", -1)) != len(rows)
        or int(protocol.get("groups", -1)) != len(groups)
        or protocol.get("group_counts") != expected_counts
        or protocol.get("sample_ids_sha256") != identifiers_hash
        or int(development.get("rows", -1)) != len(rows)
        or int(development.get("groups", -1)) != len(groups)
        or development.get("sample_ids_sha256") != identifiers_hash
        or sorted(development.get("group_ids") or []) != sorted(groups)
    ):
        raise ValueError("manifest identity/counts differ from protocol or freeze")
    identity = {
        "dataset": dataset,
        "split": DEVELOPMENT_SPLIT,
        "samples": len(rows),
        "physical_groups": len(groups),
        "sample_ids_sha256": identifiers_hash,
        "per_physical_group_samples": expected_counts,
        "natural_quality_group_memberships": dict(
            sorted(quality_memberships.items())
        ),
        "samples_without_natural_quality_group": samples_without_quality_group,
        "samples_with_multiple_natural_quality_groups": (
            samples_with_multiple_quality_groups
        ),
        "quality_groups_are_overlapping": (
            samples_with_multiple_quality_groups > 0
        ),
    }
    return rows, protocol, identity


def _validate_freeze(freeze: Mapping[str, Any]) -> tuple[float, int, int]:
    if (
        freeze.get("schema_version") != 1
        or freeze.get("protocol") != FREEZE_PROTOCOL
    ):
        raise ValueError("unsupported field evaluation freeze")
    evaluation = freeze.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("freeze lacks evaluation settings")
    bootstrap = evaluation.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("freeze lacks bootstrap settings")
    failure_penalty = _finite(evaluation.get("failure_penalty_nmae"))
    iterations = int(bootstrap.get("iterations", -1))
    seed = int(bootstrap.get("seed", -1))
    if failure_penalty != 1.0:
        raise ValueError("frozen full-denominator failure penalty must be 1.0")
    if (
        iterations != EXPECTED_BOOTSTRAP_ITERATIONS
        or bootstrap.get("paired") is not True
        or bootstrap.get("unit") != "physical meter group_id"
        or seed < 0
    ):
        raise ValueError("frozen paired physical-group bootstrap protocol changed")
    if evaluation.get("primary_method") != "reference_conditioned_router":
        raise ValueError("frozen primary method changed")
    if evaluation.get("test_labels_used_for_selection") != 0:
        raise ValueError("freeze reports test-label use for selection")
    expected_primary = {"NMAE", "Acc@1%", "Acc@2%", "Acc@5%", "coverage"}
    if set(evaluation.get("primary_metrics") or []) != expected_primary:
        raise ValueError("frozen primary metric set changed")
    required_secondary = {
        "median normalized absolute error",
        "P90 normalized absolute error",
        "P95 normalized absolute error",
        "catastrophic error rate >10%",
        "catastrophic error rate >20%",
        "macro NMAE by physical meter",
        "quality-stratum metrics",
    }
    if not required_secondary.issubset(set(evaluation.get("secondary_metrics") or [])):
        raise ValueError("freeze lacks one or more predeclared secondary metrics")
    return failure_penalty, iterations, seed


def _validate_verification(
    verification: Mapping[str, Any],
    *,
    verification_path: Path,
    freeze_path: Path,
    manifest: Path,
    inputs: Mapping[str, Path],
    expected_samples: int,
    expected_groups: int,
    identifiers_hash: str,
) -> None:
    if (
        verification.get("schema_version") != 1
        or verification.get("protocol") != VERIFICATION_PROTOCOL
        or verification.get("status") != "verified"
        or verification.get("scope") != DEVELOPMENT_SPLIT
        or verification.get("confirmatory_evaluated") is not False
        or verification.get("row_level_recomputation") is not True
        or verification.get("base_recomputed_from_raw") is not True
    ):
        raise ValueError("independent field-development verification did not pass")
    if (
        _resolved_claim(verification.get("freeze")) != freeze_path
        or verification.get("freeze_sha256") != sha256_file(freeze_path)
        or _resolved_claim(verification.get("manifest")) != manifest
        or verification.get("manifest_sha256") != sha256_file(manifest)
        or verification.get("sample_ids_sha256") != identifiers_hash
        or int(verification.get("samples", -1)) != expected_samples
        or int(verification.get("groups", -1)) != expected_groups
    ):
        raise ValueError("verification scope/manifest/freeze identity is inconsistent")
    files = verification.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("verification report lacks file hashes")
    for input_name, verification_name in VERIFICATION_FILE_KEYS.items():
        entry = files.get(verification_name)
        path = inputs[input_name]
        if (
            not isinstance(entry, Mapping)
            or _resolved_claim(entry.get("path")) != path
            or entry.get("sha256") != sha256_file(path)
            or int(entry.get("bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(
                f"{input_name} prediction is not bound by verification "
                f"{verification_path}"
            )


def _validate_prediction_rows(
    manifest_rows: Sequence[dict[str, Any]],
    prediction_rows: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    manifest_by_id = _rows_by_id(manifest_rows, label="manifest")
    expected_ids = set(manifest_by_id)
    mappings: dict[str, dict[str, dict[str, Any]]] = {}
    for label, rows in prediction_rows.items():
        mapping = _rows_by_id(rows, label=label)
        if set(mapping) != expected_ids:
            raise ValueError(
                f"{label} IDs differ from manifest: "
                f"missing={len(expected_ids-set(mapping))}, "
                f"extra={len(set(mapping)-expected_ids)}"
            )
        for sample_id, row in mapping.items():
            manifest_row = manifest_by_id[sample_id]
            if row.get("split") != DEVELOPMENT_SPLIT:
                raise PermissionError(
                    f"REFUSED: {label}[{sample_id}] is not field_development"
                )
            if (
                row.get("dataset") != manifest_row.get("dataset")
                or row.get("group_id") != manifest_row.get("group_id")
                or not _same_number(
                    row.get("ground_truth"), manifest_row.get("ground_truth")
                )
                or not _same_number(
                    row.get("scale_start"), manifest_row.get("scale_start")
                )
                or not _same_number(
                    row.get("scale_end"), manifest_row.get("scale_end")
                )
            ):
                raise ValueError(f"{label}[{sample_id}] identity/label differs")
            condition = row.get("condition")
            if condition not in (None, DEVELOPMENT_SPLIT):
                raise PermissionError(
                    f"REFUSED: {label}[{sample_id}] has condition={condition!r}"
                )
        mappings[label] = mapping
    return mappings


def load_verified_development_inputs(
    *,
    manifest: Path,
    raw: Path,
    base: Path,
    vector: Path,
    vdn: Path,
    final: Path,
    freeze: Path,
    verification: Path,
) -> dict[str, Any]:
    paths = {
        "manifest": Path(manifest).resolve(),
        "raw": Path(raw).resolve(),
        "base": Path(base).resolve(),
        "vector": Path(vector).resolve(),
        "vdn": Path(vdn).resolve(),
        "final": Path(final).resolve(),
        "freeze": Path(freeze).resolve(),
        "verification": Path(verification).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze_data = _read_json(paths["freeze"])
    failure_penalty, bootstrap_iterations, bootstrap_seed = _validate_freeze(
        freeze_data
    )
    manifest_rows, manifest_protocol, dataset_identity = _validate_manifest(
        paths["manifest"],
        freeze_data,
    )
    prediction_rows = {
        label: _read_jsonl(paths[label])
        for label in ("raw", "base", "vector", "vdn", "final")
    }
    mappings = _validate_prediction_rows(manifest_rows, prediction_rows)
    verification_data = _read_json(paths["verification"])
    _validate_verification(
        verification_data,
        verification_path=paths["verification"],
        freeze_path=paths["freeze"],
        manifest=paths["manifest"],
        inputs=paths,
        expected_samples=len(manifest_rows),
        expected_groups=dataset_identity["physical_groups"],
        identifiers_hash=dataset_identity["sample_ids_sha256"],
    )
    provenance = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(paths.items())
    }
    protocol_path = _protocol_path(paths["manifest"])
    provenance["manifest_protocol"] = {
        "path": str(protocol_path),
        "sha256": sha256_file(protocol_path),
        "bytes": protocol_path.stat().st_size,
    }
    return {
        "paths": paths,
        "freeze": freeze_data,
        "verification": verification_data,
        "manifest_protocol": manifest_protocol,
        "manifest_rows": manifest_rows,
        "mappings": mappings,
        "dataset_identity": dataset_identity,
        "failure_penalty": failure_penalty,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "provenance": provenance,
    }


def _flat_prediction(row: Mapping[str, Any]) -> float | None:
    if row.get("status", True) is False:
        return None
    return _finite(row.get("prediction"))


def _base_prediction(row: Mapping[str, Any]) -> float | None:
    predictions = row.get("predictions")
    if not isinstance(predictions, Mapping):
        return None
    return _finite(predictions.get("ours"))


def _original_transformer_prediction(row: Mapping[str, Any]) -> float | None:
    methods = row.get("methods")
    transformer = methods.get("transformer") if isinstance(methods, Mapping) else None
    if not isinstance(transformer, Mapping) or transformer.get("status") is False:
        return None
    return _finite(transformer.get("prediction"))


def extract_method_predictions(
    manifest_rows: Sequence[dict[str, Any]],
    mappings: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, list[float | None]]:
    identifiers = [str(row["sample_id"]) for row in manifest_rows]
    return {
        "final": [_flat_prediction(mappings["final"][key]) for key in identifiers],
        "base": [_base_prediction(mappings["base"][key]) for key in identifiers],
        "vector": [_flat_prediction(mappings["vector"][key]) for key in identifiers],
        "reference_conditioned_vector": [
            _finite(mappings["final"][key].get("reference_conditioned_prediction"))
            for key in identifiers
        ],
        "vdn": [_flat_prediction(mappings["vdn"][key]) for key in identifiers],
        "original_transformer": [
            _original_transformer_prediction(mappings["raw"][key])
            for key in identifiers
        ],
    }


def normalized_errors(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float | None],
    *,
    failure_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts differ")
    errors: list[float] = []
    success: list[bool] = []
    for row, prediction in zip(rows, predictions):
        ground_truth = float(row["ground_truth"])
        span = abs(float(row["scale_end"]) - float(row["scale_start"]))
        if span <= 1e-12:
            raise ValueError("scale span must be positive")
        valid = prediction is not None and math.isfinite(float(prediction))
        success.append(valid)
        errors.append(
            abs(float(prediction) - ground_truth) / span
            if valid
            else float(failure_penalty)
        )
    return np.asarray(errors, dtype=np.float64), np.asarray(success, dtype=bool)


def compute_metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[float | None],
    *,
    failure_penalty: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    errors, success = normalized_errors(
        rows,
        predictions,
        failure_penalty=failure_penalty,
    )
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=object)
    successful_errors = errors[success]
    group_nmae = {
        str(group): float(np.mean(errors[groups == group]))
        for group in sorted(set(groups.tolist()))
    }
    metrics = {
        "denominator": len(rows),
        "successful": int(np.sum(success)),
        "failures": int(np.sum(~success)),
        "coverage": float(np.mean(success)),
        "full_denominator_nmae": float(np.mean(errors)),
        "failure_penalty_nmae": float(failure_penalty),
        "acc_at_1pct": float(np.mean(success & (errors <= 0.01))),
        "acc_at_2pct": float(np.mean(success & (errors <= 0.02))),
        "acc_at_5pct": float(np.mean(success & (errors <= 0.05))),
        "success_subset_median_nae": (
            float(np.median(successful_errors)) if successful_errors.size else None
        ),
        "success_subset_p90_nae": (
            float(np.quantile(successful_errors, 0.90))
            if successful_errors.size
            else None
        ),
        "success_subset_p95_nae": (
            float(np.quantile(successful_errors, 0.95))
            if successful_errors.size
            else None
        ),
        "full_denominator_catastrophic_gt_10pct_count": int(
            np.sum(errors > 0.10)
        ),
        "full_denominator_catastrophic_gt_10pct_rate": float(
            np.mean(errors > 0.10)
        ),
        "full_denominator_catastrophic_gt_20pct_count": int(
            np.sum(errors > 0.20)
        ),
        "full_denominator_catastrophic_gt_20pct_rate": float(
            np.mean(errors > 0.20)
        ),
        "macro_physical_group_nmae": float(np.mean(list(group_nmae.values()))),
        "physical_groups": len(group_nmae),
    }
    return metrics, errors, success


def paired_physical_group_bootstrap(
    candidate_errors: Sequence[float] | np.ndarray,
    baseline_errors: Sequence[float] | np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    if (
        candidate.ndim != 1
        or baseline.shape != candidate.shape
        or group_array.shape != candidate.shape
        or candidate.size == 0
        or not bool(np.isfinite(candidate).all())
        or not bool(np.isfinite(baseline).all())
    ):
        raise ValueError("paired bootstrap inputs are invalid")
    if iterations <= 0:
        raise ValueError("paired bootstrap iterations must be positive")
    unique_groups = np.asarray(sorted(set(group_array.tolist())), dtype=object)
    indices = {
        group: np.flatnonzero(group_array == group) for group in unique_groups
    }
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        deltas[iteration] = float(
            np.mean(candidate[selected]) - np.mean(baseline[selected])
        )
    delta = float(np.mean(candidate) - np.mean(baseline))
    return {
        "delta_full_denominator_nmae_final_minus_comparator": delta,
        "paired_physical_group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "bootstrap_probability_final_better": float(np.mean(deltas < 0.0)),
        "iterations": int(iterations),
        "seed": int(seed),
        "physical_groups": int(len(unique_groups)),
        "unit": "physical meter group_id",
        "paired": True,
        "lower_is_better": True,
    }


def _subset_method_metrics(
    rows: Sequence[dict[str, Any]],
    predictions: Mapping[str, Sequence[float | None]],
    indices: Sequence[int],
    *,
    failure_penalty: float,
) -> dict[str, dict[str, Any]]:
    subset_rows = [rows[index] for index in indices]
    result: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        subset_predictions = [predictions[method][index] for index in indices]
        result[method] = compute_metrics(
            subset_rows,
            subset_predictions,
            failure_penalty=failure_penalty,
        )[0]
    return result


def _quality_groups(row: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = row.get("metadata")
    quality = metadata.get("quality") if isinstance(metadata, Mapping) else None
    groups = quality.get("groups") if isinstance(quality, Mapping) else None
    if isinstance(groups, str):
        groups = [groups]
    if not isinstance(groups, Sequence):
        return ()
    return tuple(sorted({str(value) for value in groups if isinstance(value, str) and value}))


def _failure_reason(row: Mapping[str, Any], fallback: str) -> str:
    reason = row.get("error_code")
    if isinstance(reason, str) and reason:
        return reason
    return fallback


def failure_analysis(
    manifest_rows: Sequence[dict[str, Any]],
    mappings: Mapping[str, Mapping[str, dict[str, Any]]],
    predictions: Mapping[str, Sequence[float | None]],
) -> dict[str, Any]:
    identifiers = [str(row["sample_id"]) for row in manifest_rows]
    stage_reasons: dict[str, Counter[str]] = {
        "raw_front_end": Counter(),
        "original_transformer": Counter(),
        "base": Counter(),
        "vector": Counter(),
        "reference_conditioned_vector": Counter(),
        "vdn": Counter(),
        "final": Counter(),
    }
    final_attribution = Counter()
    for index, sample_id in enumerate(identifiers):
        raw = mappings["raw"][sample_id]
        base = mappings["base"][sample_id]
        vector = mappings["vector"][sample_id]
        vdn = mappings["vdn"][sample_id]
        final = mappings["final"][sample_id]
        if raw.get("status") is not True:
            stage_reasons["raw_front_end"][
                _failure_reason(raw, "raw_front_end_failed")
            ] += 1
        if predictions["original_transformer"][index] is None:
            methods = raw.get("methods")
            transformer = (
                methods.get("transformer")
                if isinstance(methods, Mapping)
                else None
            )
            message = (
                transformer.get("message")
                if isinstance(transformer, Mapping)
                else None
            )
            reason = (
                str(message)
                if isinstance(message, str) and message
                else _failure_reason(raw, "original_transformer_unavailable")
            )
            stage_reasons["original_transformer"][reason] += 1
        if predictions["base"][index] is None:
            upstream = _failure_reason(raw, "base_prediction_unavailable")
            stage_reasons["base"][upstream] += 1
        if predictions["vector"][index] is None:
            stage_reasons["vector"][
                _failure_reason(vector, "vector_prediction_unavailable")
            ] += 1
        if predictions["reference_conditioned_vector"][index] is None:
            stage_reasons["reference_conditioned_vector"][
                _failure_reason(
                    vector,
                    "reference_conditioned_vector_unavailable",
                )
            ] += 1
        if predictions["vdn"][index] is None:
            stage_reasons["vdn"][
                _failure_reason(vdn, "vdn_prediction_unavailable")
            ] += 1
        if predictions["final"][index] is None:
            stage_reasons["final"][
                _failure_reason(final, str(final.get("route") or "final_failure"))
            ] += 1
            if raw.get("status") is not True:
                final_attribution["raw_front_end"] += 1
            elif (
                predictions["base"][index] is None
                and predictions["reference_conditioned_vector"][index] is None
            ):
                final_attribution["base_and_reference_conditioned_vector"] += 1
            elif predictions["base"][index] is None:
                final_attribution["base"] += 1
            elif predictions["reference_conditioned_vector"][index] is None:
                final_attribution["reference_conditioned_vector"] += 1
            else:
                final_attribution["final_routing"] += 1
    return {
        "per_stage": {
            stage: {
                "failures": int(sum(reasons.values())),
                "reasons": dict(sorted(reasons.items())),
            }
            for stage, reasons in stage_reasons.items()
        },
        "final_failure_earliest_attribution": dict(sorted(final_attribution.items())),
        "reason_values_are_diagnostic_not_model_selection_inputs": True,
    }


def _migration_counts(
    first_errors: np.ndarray,
    second_errors: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    tolerance = 1e-12
    delta = first_errors - second_errors
    positive = eligible & (delta < -tolerance)
    negative = eligible & (delta > tolerance)
    neutral = eligible & ~(positive | negative)
    eligible_count = int(np.sum(eligible))
    return {
        "eligible_paired_samples": eligible_count,
        "positive_transfers": int(np.sum(positive)),
        "negative_transfers": int(np.sum(negative)),
        "neutral_transfers": int(np.sum(neutral)),
        "positive_transfer_rate": (
            float(np.sum(positive) / eligible_count) if eligible_count else None
        ),
        "negative_transfer_rate": (
            float(np.sum(negative) / eligible_count) if eligible_count else None
        ),
        "mean_delta_nae_first_minus_second": (
            float(np.mean(delta[eligible])) if eligible_count else None
        ),
        "lower_delta_is_better": True,
    }


def migration_analysis(
    manifest_rows: Sequence[dict[str, Any]],
    mappings: Mapping[str, Mapping[str, dict[str, Any]]],
    errors: Mapping[str, np.ndarray],
    success: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    identifiers = [str(row["sample_id"]) for row in manifest_rows]
    final_rows = [mappings["final"][sample_id] for sample_id in identifiers]
    calibration_paired = success["reference_conditioned_vector"] & success["vector"]
    calibration_applied = np.asarray(
        [row.get("calibration_applied") is True for row in final_rows],
        dtype=bool,
    )
    routes = [str(row.get("route") or "missing_route") for row in final_rows]
    quality_switch = np.asarray(
        [route == "reference_conditioned_quality_switch" for route in routes],
        dtype=bool,
    )
    routing_paired = success["final"] & success["base"] & quality_switch
    return {
        "delta_definition": "first method NAE minus second method NAE; negative is beneficial",
        "calibration_all_paired": _migration_counts(
            errors["reference_conditioned_vector"],
            errors["vector"],
            calibration_paired,
        ),
        "calibration_applied_only": {
            **_migration_counts(
                errors["reference_conditioned_vector"],
                errors["vector"],
                calibration_paired & calibration_applied,
            ),
            "calibration_applied_rows": int(np.sum(calibration_applied)),
        },
        "routing_quality_switch_only": {
            **_migration_counts(
                errors["final"],
                errors["base"],
                routing_paired,
            ),
            "quality_switch_rows": int(np.sum(quality_switch)),
        },
        "route_counts": dict(sorted(Counter(routes).items())),
    }


def build_summary(loaded: Mapping[str, Any]) -> dict[str, Any]:
    rows = loaded["manifest_rows"]
    mappings = loaded["mappings"]
    predictions = extract_method_predictions(rows, mappings)
    failure_penalty = float(loaded["failure_penalty"])
    methods: dict[str, dict[str, Any]] = {}
    errors: dict[str, np.ndarray] = {}
    success: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        methods[method], errors[method], success[method] = compute_metrics(
            rows,
            predictions[method],
            failure_penalty=failure_penalty,
        )

    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=object)
    per_group: list[dict[str, Any]] = []
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group).tolist()
        per_group.append(
            {
                "physical_group_id": group,
                "samples": len(indices),
                "methods": _subset_method_metrics(
                    rows,
                    predictions,
                    indices,
                    failure_penalty=failure_penalty,
                ),
            }
        )

    quality_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for quality_group in _quality_groups(row):
            quality_to_indices[quality_group].append(index)
    quality_strata = {
        quality_group: {
            "samples": len(indices),
            "sample_fraction": len(indices) / len(rows),
            "physical_groups": len({str(rows[index]["group_id"]) for index in indices}),
            "methods": _subset_method_metrics(
                rows,
                predictions,
                indices,
                failure_penalty=failure_penalty,
            ),
        }
        for quality_group, indices in sorted(quality_to_indices.items())
    }

    iterations = int(loaded["bootstrap_iterations"])
    seed = int(loaded["bootstrap_seed"])
    paired = {}
    for comparator in ("base", "vector", "vdn"):
        paired[f"final_vs_{comparator}"] = {
            "candidate": "final",
            "comparator": comparator,
            **paired_physical_group_bootstrap(
                errors["final"],
                errors[comparator],
                groups,
                iterations=iterations,
                seed=seed,
            ),
        }

    return {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "status": "complete",
        "scope": {
            "dataset_partition": DEVELOPMENT_SPLIT,
            "confirmatory_opened_or_read": False,
            "confirmatory_evaluated": False,
            "runs_models": False,
            "automatic_threshold_tuning": False,
            "automatic_model_or_method_selection": False,
            "development_results_are_descriptive_only": True,
        },
        "predeclared_statistics": {
            "failure_penalty_nmae": failure_penalty,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "bootstrap_unit": "physical meter group_id",
            "paired": True,
            "catastrophic_thresholds": [0.10, 0.20],
        },
        "inputs": loaded["provenance"],
        "verification_binding": {
            "verification_protocol": loaded["verification"]["protocol"],
            "verification_status": loaded["verification"]["status"],
            "verification_sha256": loaded["provenance"]["verification"]["sha256"],
            "row_level_recomputation": loaded["verification"][
                "row_level_recomputation"
            ],
        },
        "manifest_identity": loaded["dataset_identity"],
        "methods": methods,
        "paired_physical_group_bootstrap": paired,
        "per_physical_group": per_group,
        "natural_quality_strata": quality_strata,
        "failure_analysis": failure_analysis(rows, mappings, predictions),
        "migration_analysis": migration_analysis(
            rows,
            mappings,
            errors,
            success,
        ),
        "interpretation": [
            "NMAE、Acc 与 catastrophic 指标均使用完整 manifest 分母；失败样本 NMAE 记为 1.0。",
            "median/P90/P95 NAE 只在成功预测子集计算，并与 coverage 同时报告。",
            "自然质量层来自 manifest metadata.quality.groups；层之间允许重叠。",
            "bootstrap 成对重采样完整 physical group_id，不拆分同一物理仪表。",
            "本汇总器不会根据 field_development 结果自动选择模型、路由阈值或校准阈值。",
            "field_confirmatory 保持封存且未被本工具打开、读取或评估。",
        ],
    }


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def render_chinese_markdown(summary: Mapping[str, Any]) -> str:
    identity = summary["manifest_identity"]
    methods = summary["methods"]
    lines = [
        "# 现场开发集独立统计汇总",
        "",
        f"- 数据分区：`{identity['split']}`（confirmatory 未打开、未读取、未评估）",
        (
            f"- 样本 / 物理仪表组：{identity['samples']} / "
            f"{identity['physical_groups']}"
        ),
        "- 全分母失败惩罚：NMAE = 1.0",
        (
            "- Bootstrap：按完整 physical group_id 成对重采样，"
            f"{summary['predeclared_statistics']['bootstrap_iterations']} 次，"
            f"seed={summary['predeclared_statistics']['bootstrap_seed']}"
        ),
        "- 本报告不执行自动阈值调整或模型选择。",
        "",
        "## 预声明主次指标",
        "",
        (
            "| 方法 | NMAE（全分母） | Acc@1% | Acc@2% | Acc@5% | Coverage | "
            "成功子集 Median/P90/P95 NAE | Cat. >10% / >20% | Macro-group NMAE |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        metrics = methods[method]
        lines.append(
            "| {label} | {nmae} | {a1} | {a2} | {a5} | {coverage} | "
            "{median}/{p90}/{p95} | {cat10}/{cat20} | {macro} |".format(
                label=METHOD_LABELS_ZH[method],
                nmae=_number(metrics["full_denominator_nmae"]),
                a1=_pct(metrics["acc_at_1pct"]),
                a2=_pct(metrics["acc_at_2pct"]),
                a5=_pct(metrics["acc_at_5pct"]),
                coverage=_pct(metrics["coverage"]),
                median=_number(metrics["success_subset_median_nae"]),
                p90=_number(metrics["success_subset_p90_nae"]),
                p95=_number(metrics["success_subset_p95_nae"]),
                cat10=_pct(
                    metrics["full_denominator_catastrophic_gt_10pct_rate"]
                ),
                cat20=_pct(
                    metrics["full_denominator_catastrophic_gt_20pct_rate"]
                ),
                macro=_number(metrics["macro_physical_group_nmae"]),
            )
        )

    lines.extend(
        [
            "",
            "## Final 与基线的物理组配对 Bootstrap",
            "",
            "| 比较 | ΔNMAE（Final - 对照） | 95% CI | P(Final 更优) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for result in summary["paired_physical_group_bootstrap"].values():
        interval = result["paired_physical_group_bootstrap_95ci"]
        lines.append(
            "| Final vs {label} | {delta} | [{low}, {high}] | {probability} |".format(
                label=METHOD_LABELS_ZH[result["comparator"]],
                delta=_number(
                    result[
                        "delta_full_denominator_nmae_final_minus_comparator"
                    ]
                ),
                low=_number(interval[0]),
                high=_number(interval[1]),
                probability=_pct(
                    result["bootstrap_probability_final_better"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## 每个物理仪表组",
            "",
            "| group_id | 方法 | 样本 | NMAE | Coverage | Cat. >10% | Cat. >20% |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in summary["per_physical_group"]:
        for method in METHOD_ORDER:
            metrics = group["methods"][method]
            lines.append(
                "| {group} | {method} | {samples} | {nmae} | {coverage} | "
                "{cat10} | {cat20} |".format(
                    group=group["physical_group_id"],
                    method=METHOD_LABELS_ZH[method],
                    samples=group["samples"],
                    nmae=_number(metrics["full_denominator_nmae"]),
                    coverage=_pct(metrics["coverage"]),
                    cat10=_pct(
                        metrics[
                            "full_denominator_catastrophic_gt_10pct_rate"
                        ]
                    ),
                    cat20=_pct(
                        metrics[
                            "full_denominator_catastrophic_gt_20pct_rate"
                        ]
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Manifest 自然质量层",
            "",
            (
                "| 质量层 | 方法 | 样本 | 物理组 | NMAE | Acc@5% | "
                "Coverage | Cat. >10% |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for quality, payload in summary["natural_quality_strata"].items():
        for method in METHOD_ORDER:
            metrics = payload["methods"][method]
            lines.append(
                "| {quality} | {method} | {samples} | {groups} | {nmae} | "
                "{a5} | {coverage} | {cat10} |".format(
                    quality=quality,
                    method=METHOD_LABELS_ZH[method],
                    samples=payload["samples"],
                    groups=payload["physical_groups"],
                    nmae=_number(metrics["full_denominator_nmae"]),
                    a5=_pct(metrics["acc_at_5pct"]),
                    coverage=_pct(metrics["coverage"]),
                    cat10=_pct(
                        metrics[
                            "full_denominator_catastrophic_gt_10pct_rate"
                        ]
                    ),
                )
            )

    lines.extend(["", "## 失败原因与阶段", ""])
    for stage, payload in summary["failure_analysis"]["per_stage"].items():
        reasons = "、".join(
            f"{reason}={count}" for reason, count in payload["reasons"].items()
        )
        lines.append(
            f"- `{stage}`：{payload['failures']} 个失败"
            + (f"（{reasons}）" if reasons else "。")
        )

    migration = summary["migration_analysis"]
    calibration = migration["calibration_applied_only"]
    routing = migration["routing_quality_switch_only"]
    lines.extend(
        [
            "",
            "## 校准与路由迁移",
            "",
            (
                f"- 校准实际应用：{calibration['calibration_applied_rows']} 行；"
                f"正/负/中性迁移 = {calibration['positive_transfers']}/"
                f"{calibration['negative_transfers']}/"
                f"{calibration['neutral_transfers']}。"
            ),
            (
                f"- 质量切换：{routing['quality_switch_rows']} 行；"
                f"正/负/中性迁移 = {routing['positive_transfers']}/"
                f"{routing['negative_transfers']}/"
                f"{routing['neutral_transfers']}。"
            ),
            "",
            "## 解释边界",
            "",
        ]
    )
    for note in summary["interpretation"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_outputs(
    summary: Mapping[str, Any],
    output: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    output = output.resolve()
    markdown = output.with_suffix(".md")
    if output == markdown:
        raise ValueError("JSON and Markdown output paths must differ")
    if not overwrite:
        existing = [path for path in (output, markdown) if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to overwrite: " + ", ".join(map(str, existing))
            )
    json_text = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_text(output, json_text)
    _atomic_text(markdown, render_chinese_markdown(summary))
    return output, markdown


def main() -> None:
    args = parse_args()
    loaded = load_verified_development_inputs(
        manifest=args.manifest,
        raw=args.raw,
        base=args.base,
        vector=args.vector,
        vdn=args.vdn,
        final=args.final,
        freeze=args.freeze,
        verification=args.verification,
    )
    summary = build_summary(loaded)
    output, markdown = write_outputs(
        summary,
        args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "json": str(output),
                "markdown": str(markdown),
                "samples": summary["manifest_identity"]["samples"],
                "physical_groups": summary["manifest_identity"][
                    "physical_groups"
                ],
                "confirmatory_evaluated": False,
                "automatic_selection_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
