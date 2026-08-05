"""Run the predeclared official200 VDN field comparator exactly once.

The evaluator is fail-closed around a separate adapter authorization.  It
reuses the already frozen meter box and reference angles from the one-shot
reference-support cache, and only replaces its legacy direction network with
the predeclared official200 seed-20260720 checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from experiments.datasets import syncg_sample_ids_sha256
from experiments.vdn_baseline import (
    build_vdn_model,
    image_angle_from_direction,
    predict_directions,
    reading_from_pointer_angle,
    sha256_file,
    vdn_tensor_from_bbox,
    verify_vdn_source,
)
from experiments.vdn_official200_protocol import OFFICIAL200_PROTOCOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PROTOCOL = "official200_vdn_field_confirmatory_adapter_authorization_v2"
AUTHORIZATION_V1_PROTOCOL = "official200_vdn_field_confirmatory_adapter_authorization_v1"
DEVIATION_PROTOCOL = "official200_vdn_field_confirmatory_adapter_authorization_deviation_v1"
EVALUATION_PROTOCOL = "official200_vdn_field_confirmatory_frozen_geometry_v1"
VERIFICATION_PROTOCOL = "official200_vdn_field_confirmatory_verification_v1"
EXPECTED_ROWS = 814
EXPECTED_GROUPS = 20
EXPECTED_SEED = 20260720


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _resolve_binding(binding: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(binding.get("path") or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    _require(sha256_file(path) == binding.get("sha256"), f"{label} hash drift")
    if "bytes" in binding:
        _require(path.stat().st_size == int(binding["bytes"]), f"{label} size drift")
    return path


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _error(row: Mapping[str, Any], prediction: Any) -> float:
    value = _finite(prediction)
    if value is None:
        return 1.0
    span = abs(float(row["scale_end"]) - float(row["scale_start"]))
    _require(span > 0.0, "scale span must be positive")
    return abs(value - float(row["ground_truth"])) / span


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([_error(row, row.get("prediction")) for row in rows], dtype=np.float64)
    success = np.asarray([_finite(row.get("prediction")) is not None for row in rows], dtype=bool)
    groups = np.asarray([str(row["group_id"]) for row in rows], dtype=object)
    successful_errors = errors[success]
    group_values = [float(np.mean(errors[groups == group])) for group in np.unique(groups)]
    return {
        "denominator": int(len(rows)),
        "successful": int(np.sum(success)),
        "failures": int(len(rows) - np.sum(success)),
        "coverage": float(np.mean(success)),
        "full_denominator_nmae": float(np.mean(errors)),
        "failure_penalty_nmae": 1.0,
        "acc_at_1pct": float(np.mean(success & (errors <= 0.01))),
        "acc_at_2pct": float(np.mean(success & (errors <= 0.02))),
        "acc_at_5pct": float(np.mean(success & (errors <= 0.05))),
        "success_subset_median_nae": float(np.median(successful_errors)) if successful_errors.size else None,
        "success_subset_p90_nae": float(np.quantile(successful_errors, 0.90)) if successful_errors.size else None,
        "success_subset_p95_nae": float(np.quantile(successful_errors, 0.95)) if successful_errors.size else None,
        "full_denominator_catastrophic_gt_10pct_rate": float(np.mean(errors > 0.10)),
        "full_denominator_catastrophic_gt_20pct_rate": float(np.mean(errors > 0.20)),
        "macro_physical_group_nmae": float(np.mean(group_values)),
        "physical_groups": int(len(group_values)),
    }


def _paired_bootstrap(
    vdn_rows: Sequence[Mapping[str, Any]],
    pepd_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    pepd_by_id = {str(row["sample_id"]): row for row in pepd_rows}
    vdn_errors: list[float] = []
    pepd_errors: list[float] = []
    groups: list[str] = []
    for row in vdn_rows:
        sample_id = str(row["sample_id"])
        pepd = pepd_by_id[sample_id]
        vdn_errors.append(_error(row, row.get("prediction")))
        pepd_errors.append(_error(pepd, pepd.get("prediction")))
        groups.append(str(row["group_id"]))
    vdn_array = np.asarray(vdn_errors, dtype=np.float64)
    pepd_array = np.asarray(pepd_errors, dtype=np.float64)
    group_array = np.asarray(groups, dtype=object)
    unique = np.unique(group_array)
    indices = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    relative = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample_indices = np.concatenate([indices[group] for group in selected])
        vdn_nmae = float(np.mean(vdn_array[sample_indices]))
        pepd_nmae = float(np.mean(pepd_array[sample_indices]))
        deltas[index] = vdn_nmae - pepd_nmae
        relative[index] = 1.0 - vdn_nmae / pepd_nmae
    vdn_point = float(np.mean(vdn_array))
    pepd_point = float(np.mean(pepd_array))
    return {
        "candidate": "official200_vdn_seed20260720",
        "comparator": "pepd_only_seed20260722",
        "paired": True,
        "unit": "physical meter group_id",
        "physical_groups": int(len(unique)),
        "iterations": int(iterations),
        "seed": int(seed),
        "lower_is_better": True,
        "delta_full_denominator_nmae_vdn_minus_pepd": vdn_point - pepd_point,
        "paired_physical_group_bootstrap_95ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
        "bootstrap_probability_vdn_better": float(np.mean(deltas < 0.0)),
        "relative_nmae_reduction_vdn_vs_pepd": 1.0 - vdn_point / pepd_point,
        "relative_nmae_reduction_95ci": [
            float(np.quantile(relative, 0.025)),
            float(np.quantile(relative, 0.975)),
        ],
        "sample_level_vdn_better_rate": float(np.mean(vdn_array < pepd_array)),
        "sample_level_tie_rate": float(np.mean(vdn_array == pepd_array)),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    authorization_path = args.authorization.resolve()
    authorization = _load_json(authorization_path)
    _require(authorization.get("protocol") == AUTHORIZATION_PROTOCOL, "wrong adapter authorization protocol")
    _require(authorization.get("status") == "authorized_not_started", "adapter authorization is not executable")
    _require((authorization.get("authorization") or {}).get("user_explicit_field_blind_test_authorization") is True, "user authorization missing")
    _require((authorization.get("authorization") or {}).get("predeclared_independent_comparator") is True, "comparator was not predeclared")
    _require((authorization.get("authorization") or {}).get("no_tuning_or_adapter_selection") is True, "no-tuning lock missing")

    expected_output = Path(str((authorization.get("output") or {}).get("root") or "")).resolve()
    output_root = args.output_root.resolve()
    _require(output_root == expected_output, "output root differs from adapter authorization")
    if not args.verify_only:
        _require(not output_root.exists(), "official200 field comparator output already exists")
    else:
        _require(output_root.is_dir(), "official200 field comparator output is missing")

    sources = authorization.get("sources") or {}
    evaluator_source = Path(__file__).resolve()
    vdn_adapter_source = PROJECT_ROOT / "experiments/vdn_baseline.py"
    for name, path in (("evaluator", evaluator_source), ("vdn_adapter", vdn_adapter_source)):
        binding = sources.get(name) or {}
        _require(Path(str(binding.get("path") or "")).resolve() == path, f"{name} source path drift")
        _require(binding.get("sha256") == sha256_file(path), f"{name} source byte hash drift")
        _require(binding.get("canonical_sha256") == _sha256_source(path), f"{name} source canonical hash drift")

    bindings = authorization.get("bindings") or {}
    paths = {name: _resolve_binding(binding, label=name) for name, binding in bindings.items()}
    authorization_v1 = _load_json(paths["authorization_v1"])
    _require(authorization_v1.get("protocol") == AUTHORIZATION_V1_PROTOCOL, "wrong predecessor authorization protocol")
    _require(authorization_v1.get("status") == "authorized_not_started", "predecessor authorization status drift")
    _require(
        Path(str((authorization_v1.get("output") or {}).get("root") or "")).resolve() == output_root,
        "predecessor authorization output drift",
    )
    deviation = _load_json(paths["adapter_deviation"])
    _require(deviation.get("protocol") == DEVIATION_PROTOCOL, "wrong adapter deviation protocol")
    _require(deviation.get("status") == "recorded_before_official200_field_inference", "adapter deviation status drift")
    deviation_bindings = deviation.get("bindings") or {}
    _require(
        (deviation_bindings.get("authorization_v1") or {}).get("sha256") == sha256_file(paths["authorization_v1"]),
        "adapter deviation predecessor binding drift",
    )
    deviation_scope = deviation.get("scope") or {}
    _require(deviation_scope.get("official200_field_predictions_existed_when_recorded") is False, "adapter deviation was recorded after inference")
    _require(deviation_scope.get("checkpoint_or_adapter_selection_performed") is False, "adapter selection occurred")
    resolution = deviation.get("authorized_resolution") or {}
    _require(resolution.get("join_key") == "sample_id", "adapter deviation join key drift")
    _require(resolution.get("canonical_output_order") == "manifest sample_id order", "adapter deviation order policy drift")
    _require(resolution.get("require_exact_sample_id_set") is True, "exact sample-ID set lock missing")
    _require(resolution.get("reject_duplicate_sample_ids") is True, "duplicate-ID rejection lock missing")
    _require(resolution.get("prediction_values_may_be_modified") is False, "prediction immutability lock missing")
    freeze = _load_json(paths["freeze"])
    _require(freeze.get("protocol") == "field_confirmatory_one_shot_freeze_v1", "field freeze protocol drift")
    _require(freeze.get("status") == "frozen_authorized_not_started", "field freeze status drift")
    comparator = (freeze.get("comparators") or {}).get("official200_vdn") or {}
    _require(comparator.get("seed") == EXPECTED_SEED, "field freeze VDN seed drift")
    _require((comparator.get("checkpoint") or {}).get("sha256") == sha256_file(paths["checkpoint"]), "field freeze checkpoint drift")
    _require((comparator.get("cohort") or {}).get("sha256") == sha256_file(paths["cohort"]), "field freeze cohort drift")

    manifest_protocol = _load_json(paths["manifest_protocol"])
    _require(manifest_protocol.get("split") == "field_confirmatory", "manifest is not confirmatory")
    _require(manifest_protocol.get("confirmatory_sealed") is True, "manifest sidecar seal flag drift")
    _require(manifest_protocol.get("manifest_sha256") == sha256_file(paths["manifest"]), "manifest sidecar hash drift")
    _require(int(manifest_protocol.get("rows") or manifest_protocol.get("emitted_rows") or 0) == EXPECTED_ROWS, "manifest row count drift")

    raw_metadata = _load_json(paths["raw_metadata"])
    raw_signature = raw_metadata.get("signature") or {}
    _require(raw_signature.get("manifest_sha256") == sha256_file(paths["manifest"]), "raw manifest binding drift")
    _require(raw_signature.get("manifest_protocol_sha256") == sha256_file(paths["manifest_protocol"]), "raw manifest protocol drift")
    _require(raw_signature.get("correction_mode") == "off", "raw correction mode drift")
    _require((raw_signature.get("input_degradation") or {}).get("condition") == "clean", "raw degradation drift")
    _require((raw_signature.get("input_degradation") or {}).get("seed") == EXPECTED_SEED, "raw degradation seed drift")

    reference_summary = _load_json(paths["reference_summary"])
    reference_signature = reference_summary.get("signature") or {}
    _require(reference_summary.get("status") == "complete", "reference-support cache is incomplete")
    _require(reference_summary.get("predictions_sha256") == sha256_file(paths["reference_support"]), "reference-support prediction hash drift")
    _require(reference_summary.get("metadata_sha256") == sha256_file(paths["reference_metadata"]), "reference-support metadata hash drift")
    _require(reference_signature.get("shared_predictions_sha256") == sha256_file(paths["raw"]), "reference/raw binding drift")
    _require(reference_signature.get("shared_predictions_metadata_sha256") == sha256_file(paths["raw_metadata"]), "reference/raw metadata binding drift")
    _require(reference_signature.get("manifest_sha256") == sha256_file(paths["manifest"]), "reference manifest binding drift")

    pepd_summary = _load_json(paths["pepd_summary"])
    pepd_signature = pepd_summary.get("signature") or {}
    _require(pepd_summary.get("status") == "complete", "PEPD cache is incomplete")
    _require(pepd_summary.get("output_sha256") == sha256_file(paths["pepd"]), "PEPD prediction hash drift")
    _require(pepd_signature.get("reference_predictions_sha256") == sha256_file(paths["reference_support"]), "PEPD/reference binding drift")
    _require(pepd_signature.get("manifest_sha256") == sha256_file(paths["manifest"]), "PEPD manifest binding drift")
    pepd_repair = pepd_summary.get("serialization_repair") or {}
    _require(
        pepd_repair.get("protocol")
        == "pepd_confirmatory_summary_serialization_repair_v1",
        "PEPD serialization repair protocol drift",
    )
    _require(
        pepd_repair.get("sha256")
        == sha256_file(paths["pepd_serialization_repair"]),
        "PEPD serialization repair hash drift",
    )
    repair_record = _load_json(paths["pepd_serialization_repair"])
    _require(repair_record.get("protocol") == "pepd_confirmatory_summary_serialization_repair_v1", "PEPD repair record protocol drift")
    _require(repair_record.get("predictions_sha256") == sha256_file(paths["pepd"]), "PEPD repair prediction binding drift")
    repair = repair_record.get("repair") or {}
    _require(repair.get("prediction_rows_modified") is False, "PEPD rows were modified by repair")
    _require(repair.get("predictions_recomputed") is False, "PEPD rows were recomputed by repair")
    _require(repair.get("weights_or_options_modified") is False, "PEPD settings were modified by repair")
    _require(repair.get("metrics_or_thresholds_selected") is False, "PEPD metrics were selected by repair")

    primary_summary = _load_json(paths["primary_summary"])
    _require(primary_summary.get("protocol") == "field_confirmatory_one_shot_summary_v1", "primary summary protocol drift")
    _require(primary_summary.get("status") == "complete", "primary one-shot summary incomplete")
    _require((primary_summary.get("identity") or {}).get("samples") == EXPECTED_ROWS, "primary summary row drift")
    _require((primary_summary.get("identity") or {}).get("physical_groups") == EXPECTED_GROUPS, "primary summary group drift")
    _require((primary_summary.get("scope") or {}).get("official200_vdn_included") is False, "official200 comparator already reported")

    manifest_rows = _load_jsonl(paths["manifest"])
    raw_rows = _load_jsonl(paths["raw"])
    reference_rows = _load_jsonl(paths["reference_support"])
    pepd_rows = _load_jsonl(paths["pepd"])
    _require(len(manifest_rows) == len(raw_rows) == len(reference_rows) == len(pepd_rows) == EXPECTED_ROWS, "field input row counts drift")
    lists = [[str(row.get("sample_id")) for row in rows] for rows in (manifest_rows, raw_rows, reference_rows, pepd_rows)]
    for label, values in zip(("manifest", "raw", "reference", "PEPD"), lists, strict=True):
        _require(len(set(values)) == EXPECTED_ROWS, f"{label} sample IDs are duplicated")
        _require(set(values) == set(lists[0]), f"{label} sample-ID set differs from manifest")
    _require(lists[1] == lists[0], "raw row order differs from manifest")
    _require(syncg_sample_ids_sha256(lists[0]) == (freeze.get("dataset") or {}).get("sample_ids_sha256"), "field sample identity hash drift")

    raw_by_id = {str(row["sample_id"]): row for row in raw_rows}
    reference_by_id = {str(row["sample_id"]): row for row in reference_rows}
    pepd_by_id = {str(row["sample_id"]): row for row in pepd_rows}
    raw_rows = [raw_by_id[sample_id] for sample_id in lists[0]]
    reference_rows = [reference_by_id[sample_id] for sample_id in lists[0]]
    pepd_rows = [pepd_by_id[sample_id] for sample_id in lists[0]]

    groups: set[str] = set()
    raw_geometry_comparable_rows = 0
    for manifest, raw, reference, pepd in zip(manifest_rows, raw_rows, reference_rows, pepd_rows, strict=True):
        sample_id = str(manifest["sample_id"])
        _require(manifest.get("split") == raw.get("split") == reference.get("split") == pepd.get("split") == "field_confirmatory", f"{sample_id}: split drift")
        _require(manifest.get("group_id") == raw.get("group_id") == reference.get("group_id") == pepd.get("group_id"), f"{sample_id}: group drift")
        _require(manifest.get("dataset") == raw.get("dataset") == reference.get("dataset") == pepd.get("dataset"), f"{sample_id}: dataset drift")
        _require(manifest.get("meter_id") == raw.get("meter_id") == reference.get("meter_id") == pepd.get("meter_id"), f"{sample_id}: meter identity drift")
        _require(manifest.get("metadata") == raw.get("metadata") == reference.get("metadata") == pepd.get("metadata"), f"{sample_id}: metadata identity drift")
        groups.add(str(manifest["group_id"]))
        for field in ("ground_truth", "scale_start", "scale_end"):
            value = float(manifest[field])
            for row, label in ((raw, "raw"), (reference, "reference"), (pepd, "PEPD")):
                _require(math.isclose(value, float(row[field]), rel_tol=0.0, abs_tol=1e-12), f"{sample_id}: {label} {field} drift")
        manifest_image = Path(str(manifest["image_path"])).resolve()
        _require(manifest_image.is_file(), f"{sample_id}: image missing")
        for row, label in ((raw, "raw"), (reference, "reference"), (pepd, "PEPD")):
            _require(Path(str(row["image_path"])).resolve() == manifest_image, f"{sample_id}: {label} image path drift")
            degradation = row.get("degradation") or {}
            _require(degradation.get("condition") == "clean", f"{sample_id}: {label} degradation condition drift")
            _require(degradation.get("protocol") == "controlled_blur_perspective_v1", f"{sample_id}: {label} degradation protocol drift")
            _require(degradation.get("seed") == EXPECTED_SEED, f"{sample_id}: {label} degradation seed drift")
        _require(reference.get("status") is pepd.get("status"), f"{sample_id}: reference/PEPD status drift")
        _require(reference.get("error_code") == pepd.get("error_code"), f"{sample_id}: reference/PEPD error code drift")
        _require(reference.get("input_image_sha256") == pepd.get("input_image_sha256"), f"{sample_id}: reference/PEPD image hash drift")
        _require(reference.get("reference_branch") == pepd.get("reference_branch"), f"{sample_id}: reference/PEPD branch drift")
        if reference.get("status") is True:
            bbox = reference.get("meter_bbox")
            _require(isinstance(bbox, list) and len(bbox) == 4, f"{sample_id}: invalid frozen meter box")
            _require(_finite(reference.get("start_angle")) is not None, f"{sample_id}: missing frozen start angle")
            _require(_finite(reference.get("range_angle")) is not None, f"{sample_id}: missing frozen range angle")
            pepd_bbox = pepd.get("meter_bbox")
            _require(isinstance(pepd_bbox, list) and len(pepd_bbox) == 4, f"{sample_id}: invalid PEPD frozen meter box")
            _require(
                all(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12) for left, right in zip(bbox, pepd_bbox, strict=True)),
                f"{sample_id}: reference/PEPD meter box drift",
            )
            for reference_field, pepd_field in (("start_angle", "start_angle"), ("range_angle", "range_angle")):
                _require(
                    math.isclose(float(reference[reference_field]), float(pepd[pepd_field]), rel_tol=0.0, abs_tol=1e-12),
                    f"{sample_id}: reference/PEPD {reference_field} drift",
                )
            features = raw.get("features") or {}
            raw_start = _finite(features.get("startAngle"))
            raw_range = _finite(features.get("disAngle"))
            if raw_start is not None or raw_range is not None:
                _require(raw_start is not None and raw_range is not None, f"{sample_id}: partial raw shared geometry")
                raw_geometry_comparable_rows += 1
                _require(math.isclose(raw_start, float(reference["start_angle"]), rel_tol=0.0, abs_tol=1e-12), f"{sample_id}: raw/reference start angle drift")
                _require(math.isclose(raw_range, float(reference["range_angle"]), rel_tol=0.0, abs_tol=1e-12), f"{sample_id}: raw/reference range angle drift")
                _require(raw.get("branch") == reference.get("reference_branch"), f"{sample_id}: raw/reference branch drift")
        else:
            _require(reference.get("prediction") is None, f"{sample_id}: failed reference has a prediction")
            _require(pepd.get("prediction") is None, f"{sample_id}: failed PEPD has a prediction")
            _require(reference.get("meter_bbox") == pepd.get("meter_bbox"), f"{sample_id}: failed reference/PEPD meter box drift")
            _require(reference.get("start_angle") == pepd.get("start_angle"), f"{sample_id}: failed reference/PEPD start angle drift")
            _require(reference.get("range_angle") == pepd.get("range_angle"), f"{sample_id}: failed reference/PEPD range angle drift")
    _require(len(groups) == EXPECTED_GROUPS, "field physical group count drift")
    deviation_audit = deviation.get("row_identity_audit") or {}
    _require(raw_geometry_comparable_rows == int(deviation_audit.get("raw_reference_geometry_comparable_rows") or -1), "raw/reference geometry audit count drift")

    cohort = _load_json(paths["cohort"])
    _require(cohort.get("status") == "passed" and cohort.get("verified") is True, "official200 cohort unverified")
    cohort_run = next((row for row in cohort.get("runs") or [] if row.get("seed") == EXPECTED_SEED), None)
    _require(isinstance(cohort_run, dict), "official200 seed20260720 absent")
    _require(cohort_run.get("best_checkpoint_sha256") == sha256_file(paths["checkpoint"]), "official200 cohort checkpoint drift")
    _require(cohort_run.get("verification_sha256") == sha256_file(paths["checkpoint_verification"]), "official200 verification binding drift")

    return {
        "authorization": authorization,
        "authorization_path": authorization_path,
        "paths": paths,
        "manifest_rows": manifest_rows,
        "raw_rows": raw_rows,
        "reference_rows": reference_rows,
        "pepd_rows": pepd_rows,
        "primary_summary": primary_summary,
        "cohort_run": cohort_run,
        "groups": groups,
        "output_root": output_root,
    }


def _evaluate(prepared: Mapping[str, Any], *, device: torch.device) -> list[dict[str, Any]]:
    authorization = prepared["authorization"]
    paths = prepared["paths"]
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    signature = checkpoint.get("signature") or {}
    _require(signature.get("protocol") == OFFICIAL200_PROTOCOL, "official200 checkpoint protocol drift")
    _require(signature.get("seed") == EXPECTED_SEED, "official200 checkpoint seed drift")
    image_size = int(signature.get("image_size", 384))
    settings = authorization.get("adapter") or {}
    _require(image_size == int(settings.get("image_size")), "authorized image size drift")
    batch_size = int(settings.get("batch_size"))
    amp = bool(settings.get("amp")) and device.type == "cuda"
    vdn_source = Path(str(settings.get("vdn_source"))).resolve()
    _require(verify_vdn_source(vdn_source) == signature.get("vdn_source_commit"), "VDN source commit drift")
    model = build_vdn_model(vdn_source, image_size=image_size, imagenet_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    outputs: dict[str, dict[str, Any]] = {}
    references = prepared["reference_rows"]
    manifests = {str(row["sample_id"]): row for row in prepared["manifest_rows"]}
    successful_references = [row for row in references if row.get("status") is True]
    for row in references:
        if row.get("status") is True:
            continue
        sample_id = str(row["sample_id"])
        outputs[sample_id] = {
            "sample_id": sample_id,
            "group_id": row["group_id"],
            "dataset": row.get("dataset"),
            "split": "field_confirmatory",
            "ground_truth": float(row["ground_truth"]),
            "scale_start": float(row["scale_start"]),
            "scale_end": float(row["scale_end"]),
            "status": False,
            "prediction": None,
            "progress": None,
            "pointer_angle": None,
            "direction": None,
            "heatmap_peak": None,
            "meter_bbox": row.get("meter_bbox"),
            "start_angle": row.get("start_angle"),
            "range_angle": row.get("range_angle"),
            "reference_branch": row.get("reference_branch"),
            "reference_source": "frozen_reference_support_failure",
            "error_code": row.get("error_code") or "frozen_front_end_failure",
            "runtime_seconds": 0.0,
        }

    for offset in tqdm(
        range(0, len(successful_references), batch_size),
        desc="official200 VDN field comparator",
        dynamic_ncols=True,
    ):
        batch = successful_references[offset : offset + batch_size]
        tensors: list[torch.Tensor] = []
        started: list[float] = []
        for row in batch:
            start = time.perf_counter()
            manifest = manifests[str(row["sample_id"])]
            image = cv2.imread(
                str(Path(str(manifest["image_path"])).resolve()),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                raise ValueError(f"cannot read frozen field image: {row['sample_id']}")
            tensors.append(
                vdn_tensor_from_bbox(
                    image,
                    [float(value) for value in row["meter_bbox"]],
                    image_size=image_size,
                )
            )
            started.append(start)
        inputs = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(device.type, enabled=amp):
            heatmaps, vector_maps = model(inputs)
        directions, peaks, valid = predict_directions(heatmaps.float(), vector_maps.float())
        directions_np = directions.detach().cpu().numpy()
        peaks_np = peaks.detach().cpu().numpy()
        valid_np = valid.detach().cpu().numpy()
        for index, row in enumerate(batch):
            sample_id = str(row["sample_id"])
            result: dict[str, Any] = {
                "sample_id": sample_id,
                "group_id": row["group_id"],
                "dataset": row.get("dataset"),
                "split": "field_confirmatory",
                "ground_truth": float(row["ground_truth"]),
                "scale_start": float(row["scale_start"]),
                "scale_end": float(row["scale_end"]),
                "status": False,
                "prediction": None,
                "progress": None,
                "pointer_angle": None,
                "direction": None,
                "heatmap_peak": float(peaks_np[index]),
                "meter_bbox": [float(value) for value in row["meter_bbox"]],
                "start_angle": float(row["start_angle"]),
                "range_angle": float(row["range_angle"]),
                "reference_branch": row.get("reference_branch"),
                "reference_source": "frozen_reference_support_geometry",
                "error_code": None,
                "runtime_seconds": float(time.perf_counter() - started[index]),
            }
            if not bool(valid_np[index]):
                result["error_code"] = "invalid_direction"
                outputs[sample_id] = result
                continue
            direction = directions_np[index].astype(np.float64)
            try:
                pointer_angle = image_angle_from_direction(direction)
                reading, progress = reading_from_pointer_angle(
                    pointer_angle,
                    start_angle=float(row["start_angle"]),
                    range_angle=float(row["range_angle"]),
                    scale_start=float(row["scale_start"]),
                    scale_end=float(row["scale_end"]),
                )
            except ValueError as exc:
                result["error_code"] = "reading_conversion_failed"
                result["error_message"] = str(exc)
                outputs[sample_id] = result
                continue
            result.update(
                {
                    "status": True,
                    "prediction": float(reading),
                    "progress": float(progress),
                    "pointer_angle": float(pointer_angle),
                    "direction": direction.tolist(),
                }
            )
            outputs[sample_id] = result
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ordered = [outputs[str(row["sample_id"])] for row in prepared["manifest_rows"]]
    _require(len(ordered) == EXPECTED_ROWS, "official200 field output is incomplete")
    return ordered


def _build_summary(
    prepared: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    *,
    predictions_sha256: str,
) -> dict[str, Any]:
    authorization = prepared["authorization"]
    statistics = authorization.get("statistics") or {}
    vdn_metrics = _metrics(predictions)
    pepd_metrics = _metrics(prepared["pepd_rows"])
    frozen_pepd = ((prepared["primary_summary"].get("methods") or {}).get("pepd_vector") or {})
    for name in ("coverage", "full_denominator_nmae", "acc_at_1pct", "acc_at_2pct", "acc_at_5pct"):
        _require(math.isclose(float(pepd_metrics[name]), float(frozen_pepd[name]), rel_tol=0.0, abs_tol=1e-12), f"frozen PEPD metric drift: {name}")
    paired = _paired_bootstrap(
        predictions,
        prepared["pepd_rows"],
        iterations=int(statistics["bootstrap_iterations"]),
        seed=int(statistics["bootstrap_seed"]),
    )
    return {
        "schema_version": 1,
        "protocol": EVALUATION_PROTOCOL,
        "status": "complete",
        "role": "predeclared independent official200 VDN field comparator",
        "scope": {
            "split": "field_confirmatory",
            "one_shot": True,
            "predeclared_in_original_freeze": True,
            "post_result_checkpoint_or_adapter_selection": False,
            "training_or_tuning_performed": False,
            "frozen_geometry_reused": True,
            "legacy_reference_direction_ignored": True,
        },
        "identity": {
            "samples": EXPECTED_ROWS,
            "physical_groups": EXPECTED_GROUPS,
            "sample_ids_sha256": (prepared["authorization"].get("dataset") or {}).get("sample_ids_sha256"),
            "group_ids_sorted_sha256": _sha256_strings(sorted(prepared["groups"])),
        },
        "method": {
            "name": "official200 VDN",
            "seed": EXPECTED_SEED,
            "checkpoint_sha256": sha256_file(prepared["paths"]["checkpoint"]),
            "direction_model_only_replaced": True,
            "meter_bbox_start_angle_range_angle_source": "frozen reference_support.jsonl",
        },
        "methods": {
            "official200_vdn": vdn_metrics,
            "pepd_only": pepd_metrics,
        },
        "paired_physical_group_bootstrap": {
            "official200_vdn_vs_pepd_only": paired,
        },
        "predeclared_statistics": {
            "bootstrap_iterations": int(statistics["bootstrap_iterations"]),
            "bootstrap_seed": int(statistics["bootstrap_seed"]),
            "bootstrap_unit": "physical meter group_id",
            "failure_penalty_nmae": 1.0,
        },
        "inputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in prepared["paths"].items()
        },
        "predictions": {
            "path": "predictions.jsonl",
            "sha256": predictions_sha256,
            "rows": EXPECTED_ROWS,
        },
        "authorization": {
            "path": str(prepared["authorization_path"]),
            "sha256": sha256_file(prepared["authorization_path"]),
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "canonical_sha256": _sha256_source(Path(__file__).resolve()),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "device": str((prepared["authorization"].get("adapter") or {}).get("device")),
            "batch_size": int((prepared["authorization"].get("adapter") or {}).get("batch_size")),
            "amp": bool((prepared["authorization"].get("adapter") or {}).get("amp")),
        },
    }


def main() -> int:
    args = _parse_args()
    _require(not (args.preflight_only and args.verify_only), "preflight-only and verify-only are mutually exclusive")
    prepared = _prepare(args)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "samples": EXPECTED_ROWS,
                    "physical_groups": EXPECTED_GROUPS,
                    "seed": EXPECTED_SEED,
                    "output_absent": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    output_root: Path = prepared["output_root"]
    if args.verify_only:
        predictions_path = output_root / "predictions.jsonl"
        summary_path = output_root / "summary.json"
        verification_path = output_root / "verification.json"
        for path in (predictions_path, summary_path, verification_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        predictions = _load_jsonl(predictions_path)
        _require(len(predictions) == EXPECTED_ROWS, "stored comparator rows drift")
        expected = _build_summary(
            prepared,
            predictions,
            predictions_sha256=sha256_file(predictions_path),
        )
        actual = _load_json(summary_path)
        _require(_canonical_sha256(actual) == _canonical_sha256(expected), "stored summary differs from fresh recomputation")
        verification = _load_json(verification_path)
        _require(verification.get("status") == "verified", "stored verification failed")
        _require(verification.get("summary_sha256") == sha256_file(summary_path), "verification summary hash drift")
        _require(verification.get("predictions_sha256") == sha256_file(predictions_path), "verification prediction hash drift")
        print(json.dumps({"status": "verified", "summary": str(summary_path), "summary_sha256": sha256_file(summary_path)}, ensure_ascii=False, sort_keys=True))
        return 0

    device = torch.device(args.device)
    _require(str(device) == str((prepared["authorization"].get("adapter") or {}).get("device")), "runtime device differs from authorization")
    _require(device.type != "cuda" or torch.cuda.is_available(), "CUDA is unavailable")
    predictions = _evaluate(prepared, device=device)
    temporary_root = output_root.with_name(f".{output_root.name}.tmp.{os.getpid()}")
    _require(not temporary_root.exists(), "temporary output already exists")
    temporary_root.mkdir(parents=True)
    try:
        predictions_path = temporary_root / "predictions.jsonl"
        summary_path = temporary_root / "summary.json"
        verification_path = temporary_root / "verification.json"
        _atomic_jsonl(predictions_path, predictions)
        summary = _build_summary(
            prepared,
            predictions,
            predictions_sha256=sha256_file(predictions_path),
        )
        _atomic_json(summary_path, summary)
        verification = {
            "schema_version": 1,
            "protocol": VERIFICATION_PROTOCOL,
            "status": "verified",
            "predictions_sha256": sha256_file(predictions_path),
            "summary_sha256": sha256_file(summary_path),
            "summary_canonical_sha256": _canonical_sha256(summary),
            "rows": EXPECTED_ROWS,
            "physical_groups": EXPECTED_GROUPS,
            "source_recomputation_exact": True,
            "authorization_sha256": sha256_file(prepared["authorization_path"]),
            "evaluator_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        _atomic_json(verification_path, verification)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_root, output_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    print(output_root / "summary.json")
    print(output_root / "verification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
