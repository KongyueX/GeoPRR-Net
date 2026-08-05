"""Freeze the train-only, same-sample Original Transformer vs FADR-v2 comparison.

The script only reads an explicit SyncG/train allowlist.  It consumes existing
CPU-readable OOF artifacts and never runs model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FADR_SEEDS = (20260722, 20260723, 20260724)
EXPECTED_SAMPLES = 4380
EXPECTED_GROUPS = 197

ORIGINAL_OOF = (
    PROJECT_ROOT / "artifacts/runs/quality_router_syncg/oof_clean.jsonl"
)
ORIGINAL_OOF_SUMMARY = ORIGINAL_OOF.with_suffix(".summary.json")
ORIGINAL_OOF_METADATA = ORIGINAL_OOF.with_name(ORIGINAL_OOF.name + ".meta.json")
ORIGINAL_TRAIN_PREDICTIONS = (
    PROJECT_ROOT / "artifacts/predictions/syncg_train.jsonl"
)
ORIGINAL_TRAIN_METADATA = ORIGINAL_TRAIN_PREDICTIONS.with_name(
    ORIGINAL_TRAIN_PREDICTIONS.name + ".meta.json"
)

FADR_INPUT_ROOT = (
    PROJECT_ROOT / "artifacts/runs/fadr_multiseed_v2_inputs_authoritative_v2"
)
FADR_INPUT = FADR_INPUT_ROOT / "pepd_mixed_authoritative_oof.jsonl"
FADR_INPUT_METADATA = FADR_INPUT.with_name(FADR_INPUT.name + ".meta.json")
FADR_INPUT_SUMMARY = FADR_INPUT.with_suffix(".summary.json")
FADR_INPUT_AUTHORIZATION = FADR_INPUT_ROOT / "fadr_input_authorization_v2.json"

FADR_COHORT_ROOT = (
    PROJECT_ROOT / "artifacts/runs/fadr_multiseed_v2_joint_authoritative_v2"
)
FADR_INPUT_PREFLIGHT = FADR_COHORT_ROOT / "input_preflight.json"
FADR_COHORT = FADR_COHORT_ROOT / "cohort.json"

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts/runs/original_transformer_vs_fadr_v2_train_oof_v1/comparison.json"
)
DEFAULT_VERIFICATION = DEFAULT_OUTPUT.with_name("verification.json")

_FORBIDDEN_EXACT_PATH_PARTS = frozenset(
    {
        "public",
        "test",
        "tests",
        "field",
        "sealed",
        "confirmatory",
        "holdout",
    }
)
_FORBIDDEN_PATH_PREFIXES = (
    "public_",
    "public-",
    "test_",
    "test-",
    "field_",
    "field-",
    "sealed_",
    "sealed-",
    "confirmatory_",
    "confirmatory-",
    "holdout_",
    "holdout-",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _assert_train_only_path(path: Path, *, must_exist: bool) -> Path:
    resolved = path.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path is outside the project root: {resolved}") from exc
    for raw_part in relative.parts:
        part = raw_part.lower()
        if part in _FORBIDDEN_EXACT_PATH_PARTS or part.startswith(
            _FORBIDDEN_PATH_PREFIXES
        ):
            raise ValueError(f"forbidden non-train path: {resolved}")
    return resolved


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_nonfinite(nested, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, label=f"{label}[{index}]")


def _decode_json(text: str, *, label: str) -> Any:
    value = json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    _reject_nonfinite(value, label=label)
    return value


def _load_json(path: Path) -> dict[str, Any]:
    path = _assert_train_only_path(path, must_exist=True)
    value = _decode_json(path.read_text(encoding="utf-8"), label=str(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = _assert_train_only_path(path, must_exist=True)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = _decode_json(line, label=f"{path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _sha256_file(path: Path) -> str:
    path = _assert_train_only_path(path, must_exist=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = _assert_train_only_path(path, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    return _assert_train_only_path(path, must_exist=True).relative_to(
        PROJECT_ROOT
    ).as_posix()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _method_prediction(method: Any) -> float | None:
    if not isinstance(method, Mapping) or method.get("status") is not True:
        return None
    return _finite(method.get("prediction"))


def _normalized_error(row: Mapping[str, Any], prediction: Any) -> float:
    prediction_value = _finite(prediction)
    ground_truth = _finite(row.get("ground_truth"))
    scale_start = _finite(row.get("scale_start"))
    scale_end = _finite(row.get("scale_end"))
    if (
        prediction_value is None
        or ground_truth is None
        or scale_start is None
        or scale_end is None
    ):
        return 1.0
    span = abs(scale_end - scale_start)
    if span <= 1e-12:
        return 1.0
    return abs(prediction_value - ground_truth) / span


def _metrics(errors: np.ndarray, successful: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(errors.size),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "acc_5pct": float(np.mean(errors <= 0.05)),
    }


def _mean_metrics(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_keys = ("successful", "coverage", "nmae", "acc_1pct", "acc_2pct", "acc_5pct")
    return {
        "samples": EXPECTED_SAMPLES,
        **{
            key: float(np.mean([float(metrics[key]) for metrics in per_seed]))
            for key in scalar_keys
        },
    }


def _sample_std_metrics(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = ("coverage", "nmae", "acc_1pct", "acc_2pct", "acc_5pct")
    return {
        key: float(np.std([float(metrics[key]) for metrics in per_seed], ddof=1))
        for key in keys
    }


def _assert_metric_close(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in (
        "samples",
        "successful",
        "coverage",
        "nmae",
        "acc_1pct",
        "acc_2pct",
        "acc_5pct",
    ):
        _require(key in expected, f"signed metric is missing {key}")
        if key in {"samples", "successful"}:
            _require(int(actual[key]) == int(expected[key]), f"metric mismatch: {key}")
        else:
            _require(
                math.isclose(
                    float(actual[key]),
                    float(expected[key]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"metric mismatch: {key}",
            )


def _rows_by_id(rows: Sequence[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        _require(bool(sample_id), f"{label} contains an empty sample_id")
        _require(sample_id not in result, f"{label} contains duplicate sample_id {sample_id}")
        result[sample_id] = row
    return result


def _source_identity(path: Path, *, role: str) -> dict[str, str]:
    return {
        "path": _relative(path),
        "role": role,
        "sha256": _sha256_file(path),
    }


def _paired_group_bootstrap(
    fadr_errors: np.ndarray,
    transformer_errors: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    _require(iterations > 0, "bootstrap iterations must be positive")
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    fadr_values = np.empty(iterations, dtype=np.float64)
    transformer_values = np.empty(iterations, dtype=np.float64)
    relative_reductions = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in selected])
        fadr_nmae = float(np.mean(fadr_errors[sample_indices]))
        transformer_nmae = float(np.mean(transformer_errors[sample_indices]))
        fadr_values[iteration] = fadr_nmae
        transformer_values[iteration] = transformer_nmae
        deltas[iteration] = fadr_nmae - transformer_nmae
        relative_reductions[iteration] = 1.0 - fadr_nmae / transformer_nmae

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    point_fadr = float(np.mean(fadr_errors))
    point_transformer = float(np.mean(transformer_errors))
    return {
        "unit": "physical_group",
        "physical_groups": int(len(unique)),
        "iterations": int(iterations),
        "seed": int(seed),
        "confidence_level": 0.95,
        "replicate_handling": (
            "average the exact three FADR-seed normalized errors within each "
            "sample, then resample physical groups; FADR seeds are not bootstrap units"
        ),
        "fadr_v2_three_seed_mean_nmae": point_fadr,
        "fadr_v2_three_seed_mean_nmae_95ci": interval(fadr_values),
        "original_transformer_nmae": point_transformer,
        "original_transformer_nmae_95ci": interval(transformer_values),
        "delta_nmae_fadr_minus_transformer": point_fadr - point_transformer,
        "delta_nmae_95ci": interval(deltas),
        "relative_nmae_reduction": 1.0 - point_fadr / point_transformer,
        "relative_nmae_reduction_95ci": interval(relative_reductions),
    }


def _build_artifact(
    *, bootstrap_seed: int, bootstrap_iterations: int
) -> tuple[dict[str, Any], dict[str, bool]]:
    read_paths = [
        ORIGINAL_OOF,
        ORIGINAL_OOF_SUMMARY,
        ORIGINAL_OOF_METADATA,
        ORIGINAL_TRAIN_PREDICTIONS,
        ORIGINAL_TRAIN_METADATA,
        FADR_INPUT,
        FADR_INPUT_METADATA,
        FADR_INPUT_SUMMARY,
        FADR_INPUT_AUTHORIZATION,
        FADR_INPUT_PREFLIGHT,
        FADR_COHORT,
        *[
            FADR_COHORT_ROOT
            / f"seed_{seed}/joint/joint_oof_predictions.jsonl"
            for seed in FADR_SEEDS
        ],
    ]
    for path in read_paths:
        _assert_train_only_path(path, must_exist=True)

    original_summary = _load_json(ORIGINAL_OOF_SUMMARY)
    original_metadata = _load_json(ORIGINAL_OOF_METADATA)
    original_train_metadata = _load_json(ORIGINAL_TRAIN_METADATA)
    fadr_input_summary = _load_json(FADR_INPUT_SUMMARY)
    fadr_input_metadata = _load_json(FADR_INPUT_METADATA)
    authorization = _load_json(FADR_INPUT_AUTHORIZATION)
    preflight = _load_json(FADR_INPUT_PREFLIGHT)
    cohort = _load_json(FADR_COHORT)

    original_oof_sha256 = _sha256_file(ORIGINAL_OOF)
    original_train_sha256 = _sha256_file(ORIGINAL_TRAIN_PREDICTIONS)
    fadr_input_sha256 = _sha256_file(FADR_INPUT)

    _require(original_summary.get("status") == "complete", "original OOF is incomplete")
    _require(
        original_summary.get("protocol") == "syncg_quality_router_cross_model_oof_v1",
        "unexpected original OOF protocol",
    )
    _require(original_summary.get("samples") == EXPECTED_SAMPLES, "original OOF row mismatch")
    _require(original_summary.get("groups") == EXPECTED_GROUPS, "original OOF group mismatch")
    _require(original_summary.get("group_leakage_count") == 0, "original OOF group leakage")
    _require(original_summary.get("test_samples_used") == 0, "original OOF used test rows")
    _require(original_summary.get("output_sha256") == original_oof_sha256, "original OOF hash mismatch")
    original_signature = original_summary.get("signature") or {}
    _require(original_signature.get("split") == "SyncG/train only", "original OOF is not train-only")
    _require(original_signature.get("test_sets_used") == [], "original OOF used a non-train set")
    _require(
        original_signature.get("raw_predictions_sha256") == original_train_sha256,
        "original Transformer source hash mismatch",
    )
    _require(
        (original_metadata.get("manifest_protocol") or {}).get("split") == "train",
        "original OOF metadata is not train-only",
    )
    train_signature = original_train_metadata.get("signature") or {}
    _require(
        (original_train_metadata.get("manifest_protocol") or {}).get("split") == "train",
        "original prediction source is not SyncG/train",
    )
    _require(train_signature.get("include_transformer") is True, "Transformer was not collected")
    _require(train_signature.get("reading_backend") == "compare", "unexpected Transformer source mode")

    _require(fadr_input_summary.get("status") == "complete", "FADR input is incomplete")
    _require(fadr_input_summary.get("samples") == EXPECTED_SAMPLES, "FADR input row mismatch")
    _require(fadr_input_summary.get("groups") == EXPECTED_GROUPS, "FADR input group mismatch")
    for key in ("public_samples_used", "test_samples_used", "field_samples_used", "group_leakage_count"):
        _require(fadr_input_summary.get(key) == 0, f"FADR input failed train-only check: {key}")
    _require(fadr_input_summary.get("output_sha256") == fadr_input_sha256, "FADR input hash mismatch")
    fadr_input_signature = fadr_input_summary.get("signature") or {}
    _require(fadr_input_signature.get("split") == "SyncG/train only", "FADR input is not train-only")
    _require(fadr_input_signature.get("test_sets_used") == [], "FADR input used a non-train set")
    _require(
        fadr_input_signature.get("source_oof_sha256") == original_oof_sha256,
        "FADR input is not bound to the exact original OOF source",
    )
    _require(
        (fadr_input_metadata.get("signature") or {}).get("source_oof_sha256")
        == original_oof_sha256,
        "FADR input metadata source binding mismatch",
    )

    _require(authorization.get("status") == "authorized", "FADR input is not authorized")
    _require(authorization.get("train_only_certified") is True, "authorization is not train-only")
    _require(preflight.get("status") == "verified", "FADR input preflight is not verified")
    _require(preflight.get("train_only_certified") is True, "preflight is not train-only")
    _require(cohort.get("status") == "verified", "FADR cohort is not verified")
    for document, label in ((authorization, "authorization"), (preflight, "preflight"), (cohort, "cohort")):
        for key in ("public_samples_used", "test_samples_used", "field_samples_used", "group_leakage_count"):
            _require(document.get(key) == 0, f"{label} failed train-only check: {key}")
    _require(cohort.get("public_test_field_evaluation_authorized") is False, "evaluation data was authorized")
    _require(cohort.get("samples") == EXPECTED_SAMPLES, "cohort row mismatch")
    _require(cohort.get("groups") == EXPECTED_GROUPS, "cohort group mismatch")
    _require(tuple(cohort.get("seeds") or ()) == FADR_SEEDS, "unexpected FADR seed cohort")

    authorization_inputs = authorization.get("inputs") or {}
    preflight_inputs = preflight.get("inputs") or {}
    for document_inputs, label in ((authorization_inputs, "authorization"), (preflight_inputs, "preflight")):
        _require(
            (document_inputs.get("oof_pairs") or {}).get("sha256") == fadr_input_sha256,
            f"{label} FADR input hash mismatch",
        )
        _require(
            (document_inputs.get("oof_summary") or {}).get("sha256")
            == _sha256_file(FADR_INPUT_SUMMARY),
            f"{label} FADR input summary hash mismatch",
        )

    original_rows = _load_jsonl(ORIGINAL_OOF)
    fadr_input_rows = _load_jsonl(FADR_INPUT)
    original_by_id = _rows_by_id(original_rows, label="original OOF")
    input_by_id = _rows_by_id(fadr_input_rows, label="FADR input")
    _require(len(original_rows) == EXPECTED_SAMPLES, "original OOF does not contain 4380 rows")
    _require(len(fadr_input_rows) == EXPECTED_SAMPLES, "FADR input does not contain 4380 rows")
    _require(list(original_by_id) == list(input_by_id), "OOF row order changed")
    _require(set(original_by_id) == set(input_by_id), "OOF sample sets differ")

    sample_order = list(original_by_id)
    for sample_id in sample_order:
        original = original_by_id[sample_id]
        fadr_input_row = input_by_id[sample_id]
        _require(original.get("dataset") == "SyncG", f"{sample_id}: unexpected dataset")
        _require(original.get("split") == "train", f"{sample_id}: original row is not train")
        _require(fadr_input_row.get("dataset") == "SyncG", f"{sample_id}: unexpected FADR dataset")
        _require(fadr_input_row.get("split") == "train", f"{sample_id}: FADR row is not train")
        for key in ("group_id", "held_out_seed", "ground_truth", "scale_start", "scale_end"):
            _require(original.get(key) == fadr_input_row.get(key), f"{sample_id}: binding mismatch for {key}")
        _require(
            _canonical_sha256(original.get("raw"))
            == _canonical_sha256(fadr_input_row.get("raw")),
            f"{sample_id}: raw Original Transformer payload changed",
        )

    groups = np.asarray([str(original_by_id[sample_id]["group_id"]) for sample_id in sample_order], dtype=object)
    unique_groups = sorted(set(groups.tolist()))
    _require(len(unique_groups) == EXPECTED_GROUPS, "physical group count is not 197")
    sorted_sample_ids = sorted(sample_order)
    signed_rows = preflight.get("rows") or {}
    _require(
        _sha256_strings(sorted_sample_ids) == signed_rows.get("sample_ids_sha256"),
        "sample-id binding does not match signed preflight",
    )
    _require(
        _sha256_strings(unique_groups) == signed_rows.get("group_ids_sha256"),
        "group-id binding does not match signed preflight",
    )

    transformer_predictions: list[float | None] = []
    transformer_success: list[bool] = []
    transformer_errors: list[float] = []
    for sample_id in sample_order:
        row = original_by_id[sample_id]
        method = ((row.get("raw") or {}).get("methods") or {}).get("transformer")
        prediction = _method_prediction(method)
        transformer_predictions.append(prediction)
        transformer_success.append(prediction is not None)
        transformer_errors.append(_normalized_error(row, prediction))
    transformer_error_array = np.asarray(transformer_errors, dtype=np.float64)
    transformer_success_array = np.asarray(transformer_success, dtype=bool)
    transformer_metrics = _metrics(transformer_error_array, transformer_success_array)

    cohort_per_seed = {int(item["seed"]): item for item in cohort.get("per_seed") or []}
    per_seed_metrics: list[dict[str, Any]] = []
    per_seed_artifact: dict[str, Any] = {}
    fadr_error_arrays: list[np.ndarray] = []
    fadr_prediction_binding: dict[str, str] = {}
    fadr_sources: dict[str, Any] = {}
    for seed in FADR_SEEDS:
        path = FADR_COHORT_ROOT / f"seed_{seed}/joint/joint_oof_predictions.jsonl"
        source_hash = _sha256_file(path)
        signed_seed = cohort_per_seed.get(seed) or {}
        _require(
            signed_seed.get("joint_diagnostics_sha256") == source_hash,
            f"seed {seed}: joint OOF hash mismatch",
        )
        rows = _load_jsonl(path)
        rows_by_id = _rows_by_id(rows, label=f"FADR seed {seed}")
        _require(len(rows) == EXPECTED_SAMPLES, f"seed {seed}: row count mismatch")
        _require(set(rows_by_id) == set(sample_order), f"seed {seed}: sample set mismatch")
        predictions: list[float | None] = []
        successful: list[bool] = []
        errors: list[float] = []
        binding_rows: list[dict[str, Any]] = []
        for sample_id in sample_order:
            source_row = original_by_id[sample_id]
            row = rows_by_id[sample_id]
            _require(row.get("group_id") == source_row.get("group_id"), f"seed {seed}: group binding mismatch")
            _require(row.get("held_out_seed") == source_row.get("held_out_seed"), f"seed {seed}: held-out binding mismatch")
            variant = (row.get("variants") or {}).get("full") or {}
            prediction = _finite(variant.get("prediction"))
            error = _normalized_error(source_row, prediction)
            _require(
                math.isclose(error, float(variant.get("normalized_error")), rel_tol=0.0, abs_tol=1e-12),
                f"seed {seed}: normalized error mismatch for {sample_id}",
            )
            predictions.append(prediction)
            successful.append(prediction is not None)
            errors.append(error)
            binding_rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": row.get("group_id"),
                    "prediction": prediction,
                    "normalized_error": error,
                }
            )
        error_array = np.asarray(errors, dtype=np.float64)
        metrics = _metrics(error_array, np.asarray(successful, dtype=bool))
        _assert_metric_close(metrics, (signed_seed.get("metrics") or {}).get("full") or {})
        per_seed_metrics.append(metrics)
        fadr_error_arrays.append(error_array)
        fadr_prediction_binding[str(seed)] = _canonical_sha256(binding_rows)
        fadr_sources[str(seed)] = {
            "path": _relative(path),
            "sha256": source_hash,
        }
        per_seed_artifact[str(seed)] = metrics

    fadr_three_seed_mean = _mean_metrics(per_seed_metrics)
    fadr_three_seed_sample_std = _sample_std_metrics(per_seed_metrics)
    signed_distribution = (
        ((cohort.get("feature_ablation_aggregate") or {}).get("full") or {}).get(
            "per_seed_metric_distribution"
        )
        or {}
    )
    for key in ("coverage", "nmae", "acc_1pct", "acc_2pct", "acc_5pct"):
        _require(
            math.isclose(
                float(fadr_three_seed_mean[key]),
                float((signed_distribution.get(key) or {}).get("mean")),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"FADR three-seed mean mismatch: {key}",
        )

    fadr_seed_averaged_errors = np.mean(np.stack(fadr_error_arrays, axis=0), axis=0)
    bootstrap = _paired_group_bootstrap(
        fadr_seed_averaged_errors,
        transformer_error_array,
        groups,
        seed=bootstrap_seed,
        iterations=bootstrap_iterations,
    )

    sample_binding_rows = [
        {
            "sample_id": sample_id,
            "group_id": original_by_id[sample_id]["group_id"],
            "ground_truth": original_by_id[sample_id]["ground_truth"],
            "scale_start": original_by_id[sample_id]["scale_start"],
            "scale_end": original_by_id[sample_id]["scale_end"],
        }
        for sample_id in sorted_sample_ids
    ]
    transformer_binding_rows = [
        {
            "sample_id": sample_id,
            "prediction": transformer_predictions[index],
            "successful": transformer_success[index],
            "normalized_error": transformer_errors[index],
        }
        for index, sample_id in enumerate(sample_order)
    ]

    sources = {
        "original_transformer_oof": _source_identity(
            ORIGINAL_OOF,
            role="exact 4380-row Original Transformer same-sample OOF source",
        ),
        "original_transformer_oof_summary": _source_identity(
            ORIGINAL_OOF_SUMMARY,
            role="signed Original Transformer OOF summary",
        ),
        "original_transformer_oof_metadata": _source_identity(
            ORIGINAL_OOF_METADATA,
            role="SyncG/train manifest and OOF provenance",
        ),
        "original_transformer_full_train_predictions": _source_identity(
            ORIGINAL_TRAIN_PREDICTIONS,
            role="signed raw prediction source bound by Original OOF summary",
        ),
        "original_transformer_full_train_metadata": _source_identity(
            ORIGINAL_TRAIN_METADATA,
            role="Original Transformer code/weight/manifest identity",
        ),
        "fadr_v2_oof_input": _source_identity(
            FADR_INPUT,
            role="exact 4380-row train-only FADR-v2 OOF input",
        ),
        "fadr_v2_oof_input_metadata": _source_identity(
            FADR_INPUT_METADATA,
            role="FADR-v2 input provenance",
        ),
        "fadr_v2_oof_input_summary": _source_identity(
            FADR_INPUT_SUMMARY,
            role="FADR-v2 input train-only audit",
        ),
        "fadr_v2_input_authorization": _source_identity(
            FADR_INPUT_AUTHORIZATION,
            role="FADR-v2 input authorization",
        ),
        "fadr_v2_input_preflight": _source_identity(
            FADR_INPUT_PREFLIGHT,
            role="FADR-v2 signed sample/group binding",
        ),
        "fadr_v2_cohort": _source_identity(
            FADR_COHORT,
            role="verified three-seed FADR-v2 cohort",
        ),
        "fadr_v2_joint_oof_by_seed": fadr_sources,
    }

    model_source_hashes = train_signature.get("source_sha256") or {}
    model_weight_hashes = train_signature.get("weights_sha256") or {}
    artifact = {
        "schema_version": 1,
        "protocol": "original_transformer_vs_fadr_v2_same_sample_train_oof_v1",
        "status": "frozen",
        "scope": "SyncG/train grouped OOF only",
        "claim_boundary": {
            "complete_manifest_oof": False,
            "description": (
                "union of three authoritative grouped validation splits; "
                "4380 of 16000 SyncG/train rows and 197 of 725 physical groups"
            ),
            "gpu_inference_performed": False,
            "public_samples_used": 0,
            "test_samples_used": 0,
            "field_samples_used": 0,
            "sealed_samples_used": 0,
            "confirmatory_samples_used": 0,
        },
        "cohort": {
            "samples": EXPECTED_SAMPLES,
            "physical_groups": EXPECTED_GROUPS,
            "fadr_seeds": list(FADR_SEEDS),
            "sample_fraction_of_syncg_train": float(EXPECTED_SAMPLES / 16000),
        },
        "scoring": {
            "normalized_error": "abs(prediction-ground_truth)/abs(scale_end-scale_start)",
            "failure_penalty": 1.0,
            "denominator": "all 4380 bound rows",
            "acc_thresholds": [0.01, 0.02, 0.05],
            "coverage": "finite successful predictions / 4380",
        },
        "metrics": {
            "original_transformer": transformer_metrics,
            "fadr_v2_full": {
                "per_seed": per_seed_artifact,
                "three_seed_mean": fadr_three_seed_mean,
                "three_seed_sample_std": fadr_three_seed_sample_std,
            },
        },
        "comparison": {
            "delta_nmae_fadr_minus_transformer": float(
                fadr_three_seed_mean["nmae"] - transformer_metrics["nmae"]
            ),
            "relative_nmae_reduction": float(
                1.0 - fadr_three_seed_mean["nmae"] / transformer_metrics["nmae"]
            ),
            "delta_acc_1pct_fadr_minus_transformer": float(
                fadr_three_seed_mean["acc_1pct"] - transformer_metrics["acc_1pct"]
            ),
            "delta_acc_2pct_fadr_minus_transformer": float(
                fadr_three_seed_mean["acc_2pct"] - transformer_metrics["acc_2pct"]
            ),
            "delta_acc_5pct_fadr_minus_transformer": float(
                fadr_three_seed_mean["acc_5pct"] - transformer_metrics["acc_5pct"]
            ),
            "delta_coverage_fadr_minus_transformer": float(
                fadr_three_seed_mean["coverage"] - transformer_metrics["coverage"]
            ),
            "paired_group_bootstrap": bootstrap,
        },
        "sample_binding": {
            "same_sample_set_and_order_across_original_oof_fadr_input": True,
            "same_sample_set_across_all_three_fadr_seeds": True,
            "sample_ids_sorted_sha256": _sha256_strings(sorted_sample_ids),
            "sample_ids_ordered_sha256": _sha256_strings(sample_order),
            "physical_group_ids_sorted_sha256": _sha256_strings(unique_groups),
            "sample_ground_truth_scale_binding_sha256": _canonical_sha256(
                sample_binding_rows
            ),
            "original_transformer_prediction_binding_sha256": _canonical_sha256(
                transformer_binding_rows
            ),
            "fadr_full_prediction_binding_sha256_by_seed": fadr_prediction_binding,
        },
        "original_transformer_identity": {
            "meter_transformer_source_sha256": model_source_hashes.get(
                "meter_transformer"
            ),
            "meter_transformer_adapter_sha256": model_source_hashes.get(
                "meter_transformer_adapter"
            ),
            "meter_transformer_weight_sha256": model_weight_hashes.get(
                "meter_transformer"
            ),
            "manifest_sha256": train_signature.get("manifest_sha256"),
            "manifest_protocol_sha256": train_signature.get(
                "manifest_protocol_sha256"
            ),
        },
        "sources": sources,
        "generator": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_iterations": int(bootstrap_iterations),
        },
    }
    checks = {
        "explicit_train_only_read_allowlist": True,
        "all_source_paths_reject_forbidden_domains": True,
        "signed_source_hashes_match": True,
        "original_oof_is_syncg_train_only": True,
        "fadr_input_is_syncg_train_only": True,
        "public_test_field_usage_is_zero": True,
        "group_leakage_is_zero": True,
        "row_count_is_4380": True,
        "physical_group_count_is_197": True,
        "sample_and_group_hashes_match_signed_preflight": True,
        "embedded_original_transformer_payload_is_exact": True,
        "all_three_fadr_seed_sample_sets_match": True,
        "recomputed_fadr_metrics_match_verified_cohort": True,
        "gpu_inference_was_not_run": True,
    }
    return artifact, checks


def _verify_exact(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _require(
        _canonical_sha256(actual) == _canonical_sha256(expected),
        "comparison artifact differs from a fresh source recomputation",
    )


def main() -> int:
    args = _parse_args()
    output = _assert_train_only_path(args.output, must_exist=args.verify_only)
    verification_path = _assert_train_only_path(
        args.verification, must_exist=False
    )
    expected, checks = _build_artifact(
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    if args.verify_only:
        actual = _load_json(output)
        _verify_exact(actual, expected)
        result = {
            "status": "verified",
            "comparison": _relative(output),
            "comparison_sha256": _sha256_file(output),
            "checks": checks,
        }
        print(json.dumps(result, allow_nan=False, ensure_ascii=False, sort_keys=True))
        return 0

    _write_json(output, expected)
    fresh_expected, fresh_checks = _build_artifact(
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    actual = _load_json(output)
    _verify_exact(actual, fresh_expected)
    verification = {
        "schema_version": 1,
        "protocol": "original_transformer_vs_fadr_v2_same_sample_train_oof_verification_v1",
        "status": "verified",
        "comparison": {
            "path": _relative(output),
            "sha256": _sha256_file(output),
            "canonical_sha256": _canonical_sha256(actual),
        },
        "checks": fresh_checks,
        "evidence": {
            "samples": EXPECTED_SAMPLES,
            "physical_groups": EXPECTED_GROUPS,
            "fadr_seeds": list(FADR_SEEDS),
            "source_recomputation_exact": True,
            "gpu_inference_performed": False,
        },
        "verifier": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
        },
    }
    _write_json(verification_path, verification)
    result = {
        "status": "verified",
        "comparison": _relative(output),
        "verification": _relative(verification_path),
        "comparison_sha256": _sha256_file(output),
        "verification_sha256": _sha256_file(verification_path),
        "metrics": expected["metrics"],
        "paired_group_bootstrap": expected["comparison"][
            "paired_group_bootstrap"
        ],
    }
    print(json.dumps(result, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
