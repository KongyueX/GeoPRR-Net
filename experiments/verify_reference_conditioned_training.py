"""Recompute and audit reference-conditioned train-only OOF artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.evaluate_reference_conditioned_pipeline import (
    _validate_calibrator,
    _validate_router,
)
from experiments.fadr_multiseed_protocol import (
    FADR_INPUT_PREFLIGHT_PROTOCOL,
    FADR_SEEDS,
    sha256_file as raw_sha256_file,
    strict_json_load,
    strict_jsonl_load,
)
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.strict_json import strict_json_source_sha256
from experiments.train_reference_conditioned_progress_calibrator import (
    TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL,
)
from experiments.train_reference_conditioned_router import (
    TRAINING_PROTOCOL as ROUTER_TRAINING_PROTOCOL,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_file,
    sha256_source_file,
)


VERIFICATION_PROTOCOL = "reference_conditioned_train_only_verification_v2"


def _declared_oof_protocol_matches(value: Any, *, expected: str) -> bool:
    """Accept missing declarations only for artifacts made by the legacy v1 code."""

    if value is None:
        return expected == UNCERTAINTY_FUSION_OOF_PROTOCOL
    return value == expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oof-pairs",
        type=Path,
        default=Path(
            "artifacts/runs/uncertainty_fusion_syncg/probabilistic_oof_clean.jsonl"
        ),
    )
    parser.add_argument(
        "--calibrator-root",
        type=Path,
        default=Path("artifacts/runs/reference_conditioned_progress_calibrator_syncg"),
    )
    parser.add_argument(
        "--router-root",
        type=Path,
        default=Path("artifacts/runs/reference_conditioned_router_syncg"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-seed", type=int, choices=FADR_SEEDS)
    parser.add_argument("--input-preflight", type=Path)
    parser.add_argument(
        "--expected-oof-protocol",
        default=UNCERTAINTY_FUSION_OOF_PROTOCOL,
        help=(
            "Exact input OOF protocol to verify. Defaults to legacy v1; "
            "formal FADR v2 launchers pass the frozen v2 value explicitly."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{path} already exists; refusing to overwrite"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _nested_prediction(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if not isinstance(value, Mapping) or value.get("status") is not True:
        return None
    return finite_float(value.get("prediction"))


def _validated_training_seed(
    summary: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    label: str,
) -> int:
    summary_seed = summary.get("seed")
    artifact_seed = artifact.get("seed")
    if (
        type(summary_seed) is not int
        or type(artifact_seed) is not int
        or summary_seed != artifact_seed
    ):
        raise ValueError(f"{label} summary/artifact seed audit failed")
    return summary_seed


def _declared_path_is(
    value: Any,
    expected: Path,
    *,
    label: str,
) -> None:
    if not isinstance(value, str) or Path(value).resolve() != expected.resolve():
        raise ValueError(f"{label} path binding mismatch")


def _validate_source_identity(
    summary: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    expected: Mapping[str, Path],
    label: str,
) -> None:
    if (
        summary.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL
        or artifact.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL
    ):
        raise ValueError(f"{label} source-hash protocol mismatch")
    summary_sources = summary.get("source_sha256")
    artifact_sources = artifact.get("source_sha256")
    if not isinstance(summary_sources, Mapping) or summary_sources != artifact_sources:
        raise ValueError(f"{label} summary/artifact source binding mismatch")
    if set(summary_sources) != set(expected):
        raise ValueError(f"{label} source identity key set drifted")
    for name, path in expected.items():
        if summary_sources.get(name) != sha256_source_file(path):
            raise ValueError(f"{label} source changed after training: {name}")


def _validate_diagnostic_identities(
    rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    identifiers = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in identifiers):
        raise ValueError(f"{label} input contains invalid sample identifiers")
    by_id = {row.get("sample_id"): row for row in diagnostics}
    if (
        len(identifiers) != len(set(identifiers))
        or len(by_id) != len(diagnostics)
        or set(by_id) != set(identifiers)
    ):
        raise ValueError(f"{label} diagnostic identifiers differ from input")
    for row in rows:
        diagnostic = by_id[row["sample_id"]]
        if diagnostic.get("group_id") != row.get("group_id"):
            raise ValueError(f"{row['sample_id']}: {label} group identity mismatch")
    return [str(value) for value in identifiers], by_id


def _metric_values(
    rows: list[dict[str, Any]],
    values: list[float | None],
) -> dict[str, float | int]:
    errors = np.asarray(
        [normalized_error(row, value) for row, value in zip(rows, values)],
        dtype=np.float64,
    )
    successful = np.asarray([value is not None for value in values])
    return {
        "samples": len(rows),
        "successful": int(np.sum(successful)),
        "coverage": float(np.mean(successful)),
        "nmae": float(np.mean(errors)),
        "acc_1pct": float(np.mean(errors <= 0.01)),
        "acc_2pct": float(np.mean(errors <= 0.02)),
        "acc_5pct": float(np.mean(errors <= 0.05)),
    }


def _assert_metrics(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name, value in actual.items():
        reference = expected.get(name)
        if isinstance(value, int):
            if int(reference) != value:
                raise ValueError(
                    f"{label}.{name}: recomputed {value}, summary {reference}"
                )
        elif reference is None or not math.isclose(
            float(reference),
            float(value),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{label}.{name}: recomputed {value}, summary {reference}")


def _assert_group_folds(
    diagnostics: list[dict[str, Any]],
    *,
    fold_name: str,
    label: str,
) -> dict[str, int]:
    group_folds: dict[str, set[int]] = defaultdict(set)
    fold_counts: dict[str, int] = defaultdict(int)
    for row in diagnostics:
        fold = row.get(fold_name)
        if fold is None:
            continue
        if type(fold) is not int or fold < 1:
            raise ValueError(f"{label} has an invalid fold identifier: {fold!r}")
        fold_value = fold
        group_value = row.get("group_id")
        if not isinstance(group_value, str) or not group_value:
            raise ValueError(f"{label} has an invalid group identifier")
        group = group_value
        group_folds[group].add(fold_value)
        fold_counts[str(fold_value)] += 1
    leaking = {
        group: sorted(folds) for group, folds in group_folds.items() if len(folds) != 1
    }
    if leaking:
        first = next(iter(leaking.items()))
        raise ValueError(f"{label} group assigned to multiple folds: {first}")
    return dict(sorted(fold_counts.items(), key=lambda item: int(item[0])))


def _audit_calibrator(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    model_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    expected_oof_protocol: str,
) -> dict[str, Any]:
    artifact = _validate_calibrator(model_path)
    summary = strict_json_load(summary_path)
    seed = _validated_training_seed(summary, artifact, label="calibrator")
    if (
        summary.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("model_sha256") != sha256_file(model_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("strict_nested_oof") is not True
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
        or artifact.get("training_protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or not _declared_oof_protocol_matches(
            summary.get("input_oof_protocol"),
            expected=expected_oof_protocol,
        )
        or not _declared_oof_protocol_matches(
            artifact.get("input_oof_protocol"),
            expected=expected_oof_protocol,
        )
        or artifact.get("training_oof_pairs_sha256") != sha256_file(input_path)
        or not isinstance(summary.get("training_parameters"), Mapping)
        or summary.get("training_parameters") != artifact.get("training_parameters")
    ):
        raise ValueError("calibrator summary audit failed")
    _declared_path_is(summary.get("input"), input_path, label="calibrator input")
    _declared_path_is(summary.get("model"), model_path, label="calibrator model")
    _declared_path_is(
        summary.get("diagnostics"),
        diagnostics_path,
        label="calibrator diagnostics",
    )
    _validate_source_identity(
        summary,
        artifact,
        expected={
            "features": PROJECT_DIR / "experiments" / "progress_calibrator.py",
            "reference_policy": (
                PROJECT_DIR
                / "experiments"
                / "reference_conditioned_progress_calibrator.py"
            ),
            "trainer": (
                PROJECT_DIR
                / "experiments"
                / "train_reference_conditioned_progress_calibrator.py"
            ),
        },
        label="calibrator",
    )
    diagnostics = strict_jsonl_load(diagnostics_path)
    identifiers, by_id = _validate_diagnostic_identities(
        rows,
        diagnostics,
        label="calibrator",
    )

    raw_values: list[float | None] = []
    corrected_values: list[float | None] = []
    for row, sample_id in zip(rows, identifiers):
        diagnostic = by_id[sample_id]
        raw = _nested_prediction(row, "vector")
        corrected = finite_float(diagnostic.get("corrected_prediction_oof"))
        raw_progress = finite_float(diagnostic.get("raw_progress"))
        residual = finite_float(diagnostic.get("model_residual_oof"))
        applied = finite_float(diagnostic.get("applied_residual_oof"))
        clip = finite_float(diagnostic.get("nested_correction_clip"))
        deadband = finite_float(diagnostic.get("nested_deadband"))
        corrected_progress = finite_float(diagnostic.get("corrected_progress_oof"))
        if raw is None:
            if any(
                value is not None
                for value in (
                    corrected,
                    residual,
                    applied,
                    clip,
                    deadband,
                    corrected_progress,
                )
            ):
                raise ValueError(f"{sample_id}: failed vector has calibration output")
        else:
            if any(
                value is None
                for value in (
                    raw_progress,
                    residual,
                    applied,
                    clip,
                    deadband,
                    corrected_progress,
                )
            ):
                raise ValueError(
                    f"{sample_id}: successful vector lacks calibration fields"
                )
            assert residual is not None
            assert applied is not None
            assert clip is not None
            assert deadband is not None
            assert raw_progress is not None
            assert corrected_progress is not None
            expected_applied = (
                0.0
                if abs(residual) < deadband
                else float(np.clip(residual, -clip, clip))
            )
            expected_progress = float(
                np.clip(raw_progress + expected_applied, 0.0, 1.0)
            )
            if not math.isclose(
                applied,
                expected_applied,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{sample_id}: applied residual mismatch")
            if not math.isclose(
                corrected_progress,
                expected_progress,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{sample_id}: corrected progress mismatch")
        raw_values.append(raw)
        corrected_values.append(corrected)

    metrics = {
        "raw_probabilistic_vector": _metric_values(rows, raw_values),
        "reference_conditioned_nested_oof": _metric_values(
            rows,
            corrected_values,
        ),
    }
    for name, values in metrics.items():
        _assert_metrics(values, summary["metrics"][name], label=name)
    fold_counts = _assert_group_folds(
        diagnostics,
        fold_name="calibrator_fold",
        label="calibrator",
    )
    expected_folds = {str(index) for index in range(1, int(summary["folds"]) + 1)}
    if set(fold_counts) != expected_folds:
        raise ValueError("calibrator diagnostics do not cover every declared fold")
    return {
        "protocol": artifact.get("protocol"),
        "training_protocol": summary.get("protocol"),
        "seed": seed,
        "model_sha256": sha256_file(model_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary_sha256": sha256_file(summary_path),
        "source_sha256": dict(summary["source_sha256"]),
        "source_hash_protocol": summary["source_hash_protocol"],
        "training_parameters": dict(summary["training_parameters"]),
        "fold_counts": fold_counts,
        "metrics": metrics,
    }


def _audit_router(
    rows: list[dict[str, Any]],
    *,
    input_path: Path,
    calibrator_path: Path,
    model_path: Path,
    diagnostics_path: Path,
    summary_path: Path,
    expected_oof_protocol: str,
) -> dict[str, Any]:
    artifact = _validate_router(model_path, calibrator_path)
    summary = strict_json_load(summary_path)
    seed = _validated_training_seed(summary, artifact, label="router")
    if (
        summary.get("protocol") != ROUTER_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("model_sha256") != sha256_file(model_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("nested_threshold_selection") is not True
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
        or artifact.get("training_protocol") != ROUTER_TRAINING_PROTOCOL
        or not _declared_oof_protocol_matches(
            summary.get("input_oof_protocol"),
            expected=expected_oof_protocol,
        )
        or not _declared_oof_protocol_matches(
            artifact.get("input_oof_protocol"),
            expected=expected_oof_protocol,
        )
        or artifact.get("training_oof_pairs_sha256") != sha256_file(input_path)
        or artifact.get("calibration_diagnostics_sha256")
        != summary.get("calibration_diagnostics_sha256")
        or artifact.get("calibrator_sha256") != sha256_file(calibrator_path)
        or not isinstance(summary.get("training_parameters"), Mapping)
        or summary.get("training_parameters") != artifact.get("training_parameters")
    ):
        raise ValueError("router summary audit failed")
    _declared_path_is(summary.get("input"), input_path, label="router input")
    _declared_path_is(summary.get("model"), model_path, label="router model")
    _declared_path_is(
        summary.get("diagnostics"),
        diagnostics_path,
        label="router diagnostics",
    )
    _declared_path_is(
        summary.get("calibrator"),
        calibrator_path,
        label="router calibrator",
    )
    _validate_source_identity(
        summary,
        artifact,
        expected={
            "features": (
                PROJECT_DIR / "experiments" / "calibrated_progress_router.py"
            ),
            "quality_features": (
                PROJECT_DIR / "experiments" / "uncertainty_fusion.py"
            ),
            "policy": PROJECT_DIR / "experiments" / "quality_router.py",
            "router_protocol": (
                PROJECT_DIR / "experiments" / "reference_conditioned_router.py"
            ),
            "trainer": (
                PROJECT_DIR
                / "experiments"
                / "train_reference_conditioned_router.py"
            ),
            "shared_training_helpers": (
                PROJECT_DIR / "experiments" / "train_quality_router.py"
            ),
        },
        label="router",
    )
    diagnostics = strict_jsonl_load(diagnostics_path)
    identifiers, by_id = _validate_diagnostic_identities(
        rows,
        diagnostics,
        label="router",
    )

    routed_values: list[float | None] = []
    base_values: list[float | None] = []
    calibrated_values: list[float | None] = []
    for sample_id in identifiers:
        diagnostic = by_id[sample_id]
        base = finite_float(diagnostic.get("base_prediction"))
        calibrated = finite_float(diagnostic.get("reference_conditioned_prediction"))
        routed = finite_float(diagnostic.get("prediction"))
        score = finite_float(diagnostic.get("router_score_oof"))
        threshold = finite_float(diagnostic.get("nested_threshold"))
        route = str(diagnostic.get("route"))
        if base is None:
            expected = calibrated
            expected_route = (
                "calibrated_hard_fallback" if calibrated is not None else "failure"
            )
        elif (
            calibrated is not None
            and score is not None
            and threshold is not None
            and score > threshold
        ):
            expected = calibrated
            expected_route = "calibrated_quality_switch"
        else:
            expected = base
            expected_route = "base"
        if route != expected_route:
            raise ValueError(
                f"{sample_id}: route {route!r}, expected {expected_route!r}"
            )
        if expected is None:
            if routed is not None:
                raise ValueError(f"{sample_id}: expected a failed route")
        elif routed is None or not math.isclose(
            routed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{sample_id}: routed prediction mismatch")
        routed_values.append(routed)
        base_values.append(base)
        calibrated_values.append(calibrated)

    metrics = {
        "base_mask": _metric_values(rows, base_values),
        "reference_conditioned_vector": _metric_values(
            rows,
            calibrated_values,
        ),
        "reference_conditioned_router_nested_oof": _metric_values(
            rows,
            routed_values,
        ),
    }
    for name, values in metrics.items():
        _assert_metrics(values, summary["metrics"][name], label=name)
    fold_counts = _assert_group_folds(
        diagnostics,
        fold_name="router_fold",
        label="router",
    )
    expected_folds = {str(index) for index in range(1, int(summary["folds"]) + 1)}
    if set(fold_counts) != expected_folds:
        raise ValueError("router diagnostics do not cover every declared fold")
    return {
        "protocol": artifact.get("protocol"),
        "training_protocol": summary.get("protocol"),
        "seed": seed,
        "model_sha256": sha256_file(model_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary_sha256": sha256_file(summary_path),
        "calibrator_sha256": artifact.get("calibrator_sha256"),
        "calibration_diagnostics_sha256": artifact.get(
            "calibration_diagnostics_sha256"
        ),
        "source_sha256": dict(summary["source_sha256"]),
        "source_hash_protocol": summary["source_hash_protocol"],
        "training_parameters": dict(summary["training_parameters"]),
        "fold_counts": fold_counts,
        "metrics": metrics,
    }


def _validate_input_preflight(
    path: Path,
    *,
    input_path: Path,
    expected_seed: int,
    expected_oof_protocol: str,
) -> dict[str, Any]:
    preflight = strict_json_load(path)
    oof_identity = (preflight.get("inputs") or {}).get("oof_pairs") or {}
    if (
        preflight.get("schema_version") != 1
        or preflight.get("protocol") != FADR_INPUT_PREFLIGHT_PROTOCOL
        or preflight.get("status") != "verified"
        or preflight.get("train_only_certified") is not True
        or preflight.get("group_leakage_count") != 0
        or preflight.get("test_samples_used") != 0
        or preflight.get("public_samples_used") != 0
        or preflight.get("field_samples_used") != 0
        or preflight.get("public_test_field_evaluation_authorized") is not False
        or preflight.get("fadr_seeds") != list(FADR_SEEDS)
        or preflight.get("oof_protocol") != expected_oof_protocol
        or expected_seed not in preflight.get("fadr_seeds", [])
        or oof_identity.get("path") != str(input_path)
        or oof_identity.get("sha256") != sha256_file(input_path)
    ):
        raise ValueError("FADR input preflight audit failed")
    expected_sources = {
        "preflight": raw_sha256_file(
            PROJECT_DIR / "experiments" / "preflight_fadr_multiseed.py"
        ),
        "protocol": raw_sha256_file(
            PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
        ),
        "strict_json": strict_json_source_sha256(),
    }
    if preflight.get("source_identity") != expected_sources:
        raise ValueError("FADR input preflight source identity drifted")
    return preflight


def verify_training_pair(
    *,
    oof_pairs: Path,
    calibrator_root: Path,
    router_root: Path,
    expected_seed: int | None = None,
    input_preflight: Path | None = None,
    expected_oof_protocol: str = UNCERTAINTY_FUSION_OOF_PROTOCOL,
) -> dict[str, Any]:
    """Audit one calibrator/router seed without writing a verification file."""

    oof_pairs = oof_pairs.resolve()
    calibrator_root = calibrator_root.resolve()
    router_root = router_root.resolve()
    if not isinstance(expected_oof_protocol, str) or not expected_oof_protocol.strip():
        raise ValueError("expected OOF protocol must be a non-empty string")
    calibrator_model_dir = calibrator_root / "model"
    router_model_dir = router_root / "model"
    paths = {
        "input": oof_pairs,
        "input_metadata": oof_pairs.with_name(oof_pairs.name + ".meta.json"),
        "input_summary": oof_pairs.with_name(oof_pairs.stem + ".summary.json"),
        "calibrator_model": (
            calibrator_model_dir / "reference_conditioned_calibrator.joblib"
        ),
        "calibrator_diagnostics": (
            calibrator_model_dir / "strict_nested_oof_predictions.jsonl"
        ),
        "calibrator_summary": calibrator_model_dir / "training_summary.json",
        "router_model": router_model_dir / "reference_conditioned_router.joblib",
        "router_diagnostics": (router_model_dir / "strict_nested_oof_routing.jsonl"),
        "router_summary": router_model_dir / "training_summary.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = strict_jsonl_load(oof_pairs)
    input_metadata = strict_json_load(paths["input_metadata"])
    input_summary = strict_json_load(paths["input_summary"])
    if (
        ((input_metadata.get("signature") or {}).get("protocol"))
        != expected_oof_protocol
        or input_summary.get("protocol") != expected_oof_protocol
        or input_summary.get("status") != "complete"
        or input_summary.get("output_sha256") != sha256_file(oof_pairs)
        or input_summary.get("group_leakage_count") != 0
        or input_summary.get("test_samples_used") != 0
    ):
        raise ValueError("reference-conditioned verifier OOF protocol audit failed")
    if any(
        row.get("dataset") != "SyncG" or row.get("split") != "train" for row in rows
    ):
        raise ValueError("reference-conditioned verifier accepts only SyncG/train")
    calibrator = _audit_calibrator(
        rows,
        input_path=oof_pairs,
        model_path=paths["calibrator_model"],
        diagnostics_path=paths["calibrator_diagnostics"],
        summary_path=paths["calibrator_summary"],
        expected_oof_protocol=expected_oof_protocol,
    )
    router = _audit_router(
        rows,
        input_path=oof_pairs,
        calibrator_path=paths["calibrator_model"],
        model_path=paths["router_model"],
        diagnostics_path=paths["router_diagnostics"],
        summary_path=paths["router_summary"],
        expected_oof_protocol=expected_oof_protocol,
    )
    if router["calibrator_sha256"] != calibrator["model_sha256"]:
        raise ValueError("router is not bound to the audited calibrator")
    if (
        router["calibration_diagnostics_sha256"]
        != calibrator["diagnostics_sha256"]
    ):
        raise ValueError("router is not bound to the audited calibration diagnostics")
    if router["seed"] != calibrator["seed"]:
        raise ValueError("router and calibrator use different FADR seeds")
    if expected_seed is not None:
        if expected_seed not in FADR_SEEDS or router["seed"] != expected_seed:
            raise ValueError("audited FADR seed differs from the expected seed")
        if calibrator_root.parent != router_root.parent:
            raise ValueError("FADR calibrator/router roots do not share one seed directory")
        if calibrator_root.parent.name != f"seed_{expected_seed}":
            raise ValueError("FADR seed directory name does not bind the expected seed")
        if input_preflight is None:
            raise ValueError("formal FADR seed verification requires --input-preflight")
    elif input_preflight is not None:
        raise ValueError("--input-preflight requires --expected-seed")

    preflight = None
    preflight_path = None
    if input_preflight is not None:
        preflight_path = input_preflight.resolve()
        preflight = _validate_input_preflight(
            preflight_path,
            input_path=oof_pairs,
            expected_seed=router["seed"],
            expected_oof_protocol=expected_oof_protocol,
        )
    result = {
        "schema_version": 2,
        "protocol": VERIFICATION_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "seed": router["seed"],
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_oof_protocol": expected_oof_protocol,
        "samples": len(rows),
        "groups": len(set(str(row.get("group_id")) for row in rows)),
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "evidence_role": (
            "component evidence and final full-data fitting only; the "
            "standalone calibrator OOF and independently folded router OOF "
            "are not valid combined FADR OOF evidence"
        ),
        "combined_fadr_oof_authorized": False,
        "joint_outer_group_lineage_present": False,
        "calibrator_root": str(calibrator_root),
        "router_root": str(router_root),
        "calibrator": calibrator,
        "router": router,
        "source_identity": {
            "verifier": sha256_source_file(Path(__file__).resolve()),
        },
    }
    if preflight is not None and preflight_path is not None:
        result.update(
            {
                "input_preflight": str(preflight_path),
                "input_preflight_sha256": sha256_file(preflight_path),
                "input_authorization_sha256": preflight[
                    "input_authorization_sha256"
                ],
            }
        )
    return result


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.calibrator_root = args.calibrator_root.resolve()
    args.router_root = args.router_root.resolve()
    if (args.expected_seed is None) != (args.input_preflight is None):
        raise ValueError(
            "--expected-seed and --input-preflight must be supplied together"
        )
    input_preflight = (
        args.input_preflight.resolve()
        if args.input_preflight is not None
        else None
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else args.router_root / "verification.json"
    )
    result = verify_training_pair(
        oof_pairs=args.oof_pairs,
        calibrator_root=args.calibrator_root,
        router_root=args.router_root,
        expected_seed=args.expected_seed,
        input_preflight=input_preflight,
        expected_oof_protocol=args.expected_oof_protocol,
    )
    if args.overwrite:
        _atomic_json(output, result)
    else:
        _json_no_clobber(output, result)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
