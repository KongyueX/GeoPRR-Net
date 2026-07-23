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
from experiments.quality_router import finite_float, normalized_error, read_jsonl
from experiments.train_reference_conditioned_progress_calibrator import (
    TRAINING_PROTOCOL as CALIBRATOR_TRAINING_PROTOCOL,
)
from experiments.train_reference_conditioned_router import (
    TRAINING_PROTOCOL as ROUTER_TRAINING_PROTOCOL,
)
from experiments.vdn_baseline import sha256_file


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
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nested_prediction(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if not isinstance(value, Mapping) or value.get("status") is not True:
        return None
    return finite_float(value.get("prediction"))


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
        fold_value = int(fold)
        group = str(row.get("group_id"))
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
) -> dict[str, Any]:
    artifact = _validate_calibrator(model_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("protocol") != CALIBRATOR_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("model_sha256") != sha256_file(model_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("strict_nested_oof") is not True
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("calibrator summary audit failed")
    diagnostics = read_jsonl(diagnostics_path)
    by_id = {str(row.get("sample_id")): row for row in diagnostics}
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(by_id) != len(diagnostics) or set(by_id) != set(identifiers):
        raise ValueError("calibrator diagnostic identifiers differ from input")

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
    return {
        "protocol": artifact.get("protocol"),
        "model_sha256": sha256_file(model_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary_sha256": sha256_file(summary_path),
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
) -> dict[str, Any]:
    artifact = _validate_router(model_path, calibrator_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("protocol") != ROUTER_TRAINING_PROTOCOL
        or summary.get("status") != "complete"
        or summary.get("input_sha256") != sha256_file(input_path)
        or summary.get("model_sha256") != sha256_file(model_path)
        or summary.get("diagnostics_sha256") != sha256_file(diagnostics_path)
        or summary.get("nested_threshold_selection") is not True
        or int(summary.get("group_leakage_count", -1)) != 0
        or int(summary.get("test_samples_used", -1)) != 0
    ):
        raise ValueError("router summary audit failed")
    diagnostics = read_jsonl(diagnostics_path)
    by_id = {str(row.get("sample_id")): row for row in diagnostics}
    identifiers = [str(row.get("sample_id")) for row in rows]
    if len(by_id) != len(diagnostics) or set(by_id) != set(identifiers):
        raise ValueError("router diagnostic identifiers differ from input")

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
    return {
        "protocol": artifact.get("protocol"),
        "model_sha256": sha256_file(model_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "summary_sha256": sha256_file(summary_path),
        "calibrator_sha256": artifact.get("calibrator_sha256"),
        "fold_counts": fold_counts,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    args.oof_pairs = args.oof_pairs.resolve()
    args.calibrator_root = args.calibrator_root.resolve()
    args.router_root = args.router_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else args.router_root / "verification.json"
    )
    calibrator_model_dir = args.calibrator_root / "model"
    router_model_dir = args.router_root / "model"
    paths = {
        "input": args.oof_pairs,
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
    rows = read_jsonl(args.oof_pairs)
    calibrator = _audit_calibrator(
        rows,
        input_path=args.oof_pairs,
        model_path=paths["calibrator_model"],
        diagnostics_path=paths["calibrator_diagnostics"],
        summary_path=paths["calibrator_summary"],
    )
    router = _audit_router(
        rows,
        input_path=args.oof_pairs,
        calibrator_path=paths["calibrator_model"],
        model_path=paths["router_model"],
        diagnostics_path=paths["router_diagnostics"],
        summary_path=paths["router_summary"],
    )
    if router["calibrator_sha256"] != calibrator["model_sha256"]:
        raise ValueError("router is not bound to the audited calibrator")
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "input": str(args.oof_pairs),
        "input_sha256": sha256_file(args.oof_pairs),
        "samples": len(rows),
        "groups": len(set(str(row.get("group_id")) for row in rows)),
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "calibrator": calibrator,
        "router": router,
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
