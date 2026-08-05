"""Verify one seed of joint outer-group FADR stacking evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ablate_fadr_router_features import (
    _grouped_splits,
    _metrics,
    _route,
    _validate_preflight,
)
from experiments.fadr_feature_sets import FADR_ROUTER_FEATURE_SETS
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    FADR_JOINT_LINEAGE_PROTOCOL,
    FADR_JOINT_TRAINING_PROTOCOL,
    FADR_JOINT_VERIFICATION_PROTOCOL,
    FADR_SEEDS,
    assert_train_only_path,
    sha256_file,
    strict_json_load,
    strict_jsonl_load,
)
from experiments.quality_router import finite_float, normalized_error
from experiments.strict_json import STRICT_JSON_PROTOCOL
from experiments.train_joint_nested_fadr import (
    DIAGNOSTICS_FILENAME,
    LINEAGE_FILENAME,
    SUMMARY_FILENAME,
    _baseline_prediction,
    _canonical_sha256,
    _identity,
    _write_no_clobber,
    joint_source_identity,
)
from experiments.train_quality_router import _paired_group_bootstrap
from experiments.train_reference_conditioned_progress_calibrator import (
    _target_progress,
    _vector_progress,
)
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    sha256_source_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-pairs", type=Path, required=True)
    parser.add_argument("--input-preflight", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, choices=FADR_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _equal_optional(first: Any, second: Any) -> bool:
    left = finite_float(first)
    right = finite_float(second)
    if left is None or right is None:
        return left is None and right is None
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def _assert_metrics(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{label}: metric schema mismatch")
    for name, value in actual.items():
        recorded = expected.get(name)
        if isinstance(value, int):
            if type(recorded) is not int or recorded != value:
                raise ValueError(f"{label}: {name} mismatch")
        elif not _equal_optional(value, recorded):
            raise ValueError(f"{label}: {name} mismatch")


def _expected_source_identity() -> dict[str, str]:
    return joint_source_identity()


def verify_joint_run(
    *,
    oof_pairs: Path,
    input_preflight: Path,
    joint_root: Path,
    expected_seed: int,
) -> dict[str, Any]:
    oof_pairs = oof_pairs.resolve()
    input_preflight = input_preflight.resolve()
    joint_root = joint_root.resolve()
    for label, path in {
        "joint FADR input": oof_pairs,
        "joint FADR preflight": input_preflight,
        "joint FADR root": joint_root,
    }.items():
        assert_train_only_path(path, label=label)
    preflight = _validate_preflight(
        input_preflight,
        oof_pairs=oof_pairs,
        seed=expected_seed,
    )
    diagnostics_path = joint_root / DIAGNOSTICS_FILENAME
    lineage_path = joint_root / LINEAGE_FILENAME
    summary_path = joint_root / SUMMARY_FILENAME
    diagnostics = strict_jsonl_load(diagnostics_path)
    lineage = strict_json_load(lineage_path)
    summary = strict_json_load(summary_path)
    expected_sources = _expected_source_identity()
    common_audit = (
        summary.get("protocol") == FADR_JOINT_TRAINING_PROTOCOL
        and summary.get("status") == "complete"
        and summary.get("seed") == expected_seed
        and summary.get("joint_outer_group_nested") is True
        and summary.get("standalone_calibrator_oof_used") is False
        and summary.get("standalone_router_oof_used") is False
        and summary.get("combined_fadr_oof_authorized") is True
        and summary.get("full_is_preregistered_primary") is True
        and summary.get("no_reference_variant_interpretation")
        == (
            "router evidence ablation only; the fold-specific calibrated "
            "candidate remains present"
        )
        and summary.get("bootstrap_role")
        == (
            "post-hoc paired physical-group inference only; never used to "
            "tune an outer prediction"
        )
        and summary.get("input") == str(oof_pairs)
        and summary.get("input_sha256") == sha256_file(oof_pairs)
        and summary.get("input_oof_protocol") == EXPECTED_OOF_PROTOCOL
        and summary.get("input_preflight") == str(input_preflight)
        and summary.get("input_preflight_sha256")
        == sha256_file(input_preflight)
        and summary.get("input_authorization_sha256")
        == preflight.get("input_authorization_sha256")
        and summary.get("diagnostics") == str(diagnostics_path)
        and summary.get("diagnostics_sha256") == sha256_file(diagnostics_path)
        and summary.get("lineage") == str(lineage_path)
        and summary.get("lineage_sha256") == sha256_file(lineage_path)
        and summary.get("source_identity") == expected_sources
        and summary.get("source_hash_protocol") == SOURCE_TEXT_SHA256_PROTOCOL
        and summary.get("strict_json_protocol") == STRICT_JSON_PROTOCOL
        and summary.get("group_leakage_count") == 0
        and summary.get("test_samples_used") == 0
        and summary.get("public_samples_used") == 0
        and summary.get("field_samples_used") == 0
        and summary.get("public_test_field_evaluation_authorized") is False
    )
    if not common_audit:
        raise ValueError("joint FADR summary audit failed")
    if (
        lineage.get("protocol") != FADR_JOINT_LINEAGE_PROTOCOL
        or lineage.get("status") != "complete"
        or lineage.get("seed") != expected_seed
        or lineage.get("input") != str(oof_pairs)
        or lineage.get("input_sha256") != sha256_file(oof_pairs)
        or lineage.get("input_oof_protocol") != EXPECTED_OOF_PROTOCOL
        or lineage.get("input_preflight") != str(input_preflight)
        or lineage.get("input_preflight_sha256")
        != sha256_file(input_preflight)
        or lineage.get("input_authorization_sha256")
        != preflight.get("input_authorization_sha256")
        or lineage.get("standalone_calibrator_oof_used") is not False
        or lineage.get("standalone_router_oof_used") is not False
        or lineage.get("combined_fadr_oof_authorized") is not True
        or lineage.get("source_identity") != expected_sources
        or lineage.get("source_hash_protocol") != SOURCE_TEXT_SHA256_PROTOCOL
        or lineage.get("strict_json_protocol") != STRICT_JSON_PROTOCOL
        or lineage.get("group_leakage_count") != 0
    ):
        raise ValueError("joint FADR lineage top-level audit failed")
    if summary.get("parameters") != lineage.get("parameters"):
        raise ValueError("joint FADR summary/lineage parameter mismatch")
    if summary.get("feature_sets") != {
        name: list(features)
        for name, features in FADR_ROUTER_FEATURE_SETS.items()
    } or lineage.get("feature_sets") != summary.get("feature_sets"):
        raise ValueError("joint FADR feature-set identity drifted")

    rows = strict_jsonl_load(oof_pairs)
    if any(
        row.get("dataset") != "SyncG" or row.get("split") != "train"
        for row in rows
    ):
        raise ValueError("joint FADR verifier accepts only SyncG/train")
    identifiers = [str(row.get("sample_id")) for row in rows]
    groups = np.asarray(
        [str(row.get("group_id")) for row in rows],
        dtype=object,
    )
    diagnostic_by_id = {
        str(row.get("sample_id")): row for row in diagnostics
    }
    if (
        len(identifiers) != len(set(identifiers))
        or len(diagnostic_by_id) != len(diagnostics)
        or set(diagnostic_by_id) != set(identifiers)
        or summary.get("samples") != len(rows)
        or summary.get("groups") != len(set(groups.tolist()))
        or lineage.get("samples") != len(rows)
        or lineage.get("groups") != len(set(groups.tolist()))
    ):
        raise ValueError("joint FADR row identity audit failed")
    parameters = summary["parameters"]
    outer_splits = _grouped_splits(
        groups,
        np.zeros(len(rows), dtype=np.float64),
        folds=int(parameters["outer_folds"]),
        seed=expected_seed,
        label="joint verifier outer",
    )
    outer_records = lineage.get("outer_folds")
    if not isinstance(outer_records, list) or len(outer_records) != len(
        outer_splits
    ):
        raise ValueError("joint FADR outer lineage count mismatch")
    raw_progress = np.asarray(
        [
            value if (value := _vector_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    target_progress = np.asarray(
        [
            value if (value := _target_progress(row)) is not None else math.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    vector_available = np.isfinite(raw_progress)
    calibrator_trainable = vector_available & np.isfinite(target_progress)
    base_values = [_baseline_prediction(row) for row in rows]
    base_success = np.asarray([value is not None for value in base_values])

    for fold, ((outer_train, outer_validation), record) in enumerate(
        zip(outer_splits, outer_records),
        start=1,
    ):
        if not isinstance(record, Mapping) or record.get("fold") != fold:
            raise ValueError(f"joint FADR outer fold {fold} identity mismatch")
        without_id = {
            key: value for key, value in record.items() if key != "fold_lineage_id"
        }
        if record.get("fold_lineage_id") != _canonical_sha256(without_id):
            raise ValueError(f"joint FADR outer fold {fold} lineage hash mismatch")
        exclusion = record.get("outer_validation_exclusion")
        expected_exclusion_keys = {
            "calibrator_model_fits",
            "calibrator_clip_deadband_selection",
            "router_model_fits",
            "router_threshold_selection",
            "router_feature_construction_for_training",
            "router_target_construction",
            "bootstrap_prediction_tuning",
        }
        if (
            record.get("outer_train")
            != _identity(outer_train, identifiers, groups)
            or record.get("outer_validation")
            != _identity(outer_validation, identifiers, groups)
            or record.get("group_overlap") != 0
            or not isinstance(exclusion, Mapping)
            or set(exclusion) != expected_exclusion_keys
            or set(exclusion.values()) != {True}
            or (record.get("bootstrap_prediction_tuning") or {}).get(
                "performed"
            )
            is not False
        ):
            raise ValueError(f"joint FADR outer fold {fold} exclusion audit failed")
        for index in outer_validation.tolist():
            diagnostic = diagnostic_by_id[identifiers[index]]
            if (
                diagnostic.get("joint_outer_fold") != fold
                or diagnostic.get("fold_lineage_id")
                != record.get("fold_lineage_id")
                or diagnostic.get("group_id") != str(groups[index])
                or diagnostic.get("fadr_seed") != expected_seed
                or diagnostic.get("held_out_seed")
                != rows[index].get("held_out_seed")
            ):
                raise ValueError(
                    f"joint FADR outer fold {fold} diagnostic binding failed"
                )
        train_calibrator = outer_train[calibrator_trainable[outer_train]]
        validation_calibrator = outer_validation[
            vector_available[outer_validation]
        ]
        calibrator = record.get("calibrator")
        if not isinstance(calibrator, Mapping):
            raise ValueError(f"joint FADR outer fold {fold} calibrator missing")
        if (
            calibrator.get("trainable_outer_training")
            != _identity(train_calibrator, identifiers, groups)
            or calibrator.get("external_outer_validation")
            != _identity(validation_calibrator, identifiers, groups)
            or calibrator.get("cross_fit_seed")
            != expected_seed + fold * 100_000 + 10_000
            or (calibrator.get("policy_selection") or {}).get("scope")
            != _identity(train_calibrator, identifiers, groups)
            or (calibrator.get("final_fit") or {}).get("scope")
            != _identity(train_calibrator, identifiers, groups)
            or (calibrator.get("final_fit") or {}).get("seed")
            != expected_seed + fold * 100_000 + 50_000
        ):
            raise ValueError(
                f"joint FADR outer fold {fold} calibrator scope drifted"
            )
        calibrator_splits = _grouped_splits(
            groups[train_calibrator],
            np.zeros(len(train_calibrator), dtype=np.float64),
            folds=int(parameters["calibrator_folds"]),
            seed=expected_seed + fold * 100_000 + 10_000,
            label=f"joint verifier calibrator fold {fold}",
        )
        stored_calibrator_folds = calibrator.get(
            "cross_fitted_training_candidates"
        )
        if not isinstance(stored_calibrator_folds, list) or len(
            stored_calibrator_folds
        ) != len(calibrator_splits):
            raise ValueError(
                f"joint FADR outer fold {fold} calibrator folds missing"
            )
        for inner_fold, ((fit_local, held_local), stored) in enumerate(
            zip(calibrator_splits, stored_calibrator_folds),
            start=1,
        ):
            if (
                stored.get("fold") != inner_fold
                or stored.get("seed")
                != expected_seed
                + fold * 100_000
                + inner_fold * 100
                or stored.get("train")
                != _identity(
                    train_calibrator[fit_local],
                    identifiers,
                    groups,
                )
                or stored.get("validation")
                != _identity(
                    train_calibrator[held_local],
                    identifiers,
                    groups,
                )
                or stored.get("group_overlap") != 0
                or stored.get("outer_validation_group_overlap") != 0
            ):
                raise ValueError(
                    f"joint FADR outer {fold} calibrator inner {inner_fold} drifted"
                )
        train_joint = outer_train[
            base_success[outer_train] & calibrator_trainable[outer_train]
        ]
        validation_joint = outer_validation[
            base_success[outer_validation] & vector_available[outer_validation]
        ]
        router = record.get("router")
        if (
            not isinstance(router, Mapping)
            or router.get("inner_split_seed")
            != expected_seed + fold * 100_000 + 60_000
            or router.get("training_joint")
            != _identity(train_joint, identifiers, groups)
            or router.get("validation_joint")
            != _identity(validation_joint, identifiers, groups)
        ):
            raise ValueError(f"joint FADR outer fold {fold} router scope drifted")
        router_splits = _grouped_splits(
            groups[train_joint],
            np.zeros(len(train_joint), dtype=np.float64),
            folds=int(parameters["router_inner_folds"]),
            seed=expected_seed + fold * 100_000 + 60_000,
            label=f"joint verifier router fold {fold}",
        )
        stored_router_folds = router.get("inner_folds")
        if not isinstance(stored_router_folds, list) or len(
            stored_router_folds
        ) != len(router_splits):
            raise ValueError(f"joint FADR outer fold {fold} router folds missing")
        for inner_fold, ((fit_local, held_local), stored) in enumerate(
            zip(router_splits, stored_router_folds),
            start=1,
        ):
            if (
                stored.get("fold") != inner_fold
                or stored.get("train")
                != _identity(train_joint[fit_local], identifiers, groups)
                or stored.get("validation")
                != _identity(train_joint[held_local], identifiers, groups)
                or stored.get("group_overlap") != 0
                or stored.get("outer_validation_group_overlap") != 0
            ):
                raise ValueError(
                    f"joint FADR outer {fold} router inner {inner_fold} drifted"
                )
        variants = router.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(
            FADR_ROUTER_FEATURE_SETS
        ):
            raise ValueError(f"joint FADR outer fold {fold} variants drifted")
        validation_joint_set = set(validation_joint.tolist())
        for variant, names in FADR_ROUTER_FEATURE_SETS.items():
            variant_lineage = variants[variant]
            threshold = finite_float(variant_lineage.get("threshold"))
            if (
                threshold is None
                or variant_lineage.get("feature_names") != list(names)
                or variant_lineage.get("feature_count") != len(names)
                or variant_lineage.get("inner_score_seed_rule")
                != (
                    "fadr_seed + outer_fold*100000 + 70000 + inner_fold"
                )
                or variant_lineage.get("final_model_seed")
                != expected_seed + fold * 100_000 + 80_000
                or variant_lineage.get("training")
                != _identity(train_joint, identifiers, groups)
                or variant_lineage.get("validation")
                != _identity(validation_joint, identifiers, groups)
                or variant_lineage.get("outer_validation_group_overlap") != 0
            ):
                raise ValueError(
                    f"joint FADR outer fold {fold}, {variant} lineage drifted"
                )
            for index in outer_validation.tolist():
                diagnostic_variant = diagnostic_by_id[identifiers[index]][
                    "variants"
                ][variant]
                score = finite_float(
                    diagnostic_variant.get("router_score_oof")
                )
                recorded_threshold = finite_float(
                    diagnostic_variant.get("nested_threshold")
                )
                if index in validation_joint_set:
                    if score is None or not _equal_optional(
                        recorded_threshold,
                        threshold,
                    ):
                        raise ValueError(
                            f"joint FADR {fold}, {variant}: score/threshold missing"
                        )
                elif score is not None or recorded_threshold is not None:
                    raise ValueError(
                        f"joint FADR {fold}, {variant}: non-joint row has score"
                    )

    calibrated_values: list[float | None] = []
    variant_values: dict[str, list[float | None]] = {
        name: [] for name in FADR_ROUTER_FEATURE_SETS
    }
    variant_routes: dict[str, list[str]] = {
        name: [] for name in FADR_ROUTER_FEATURE_SETS
    }
    for index, row in enumerate(rows):
        diagnostic = diagnostic_by_id[identifiers[index]]
        calibrated = finite_float(
            diagnostic.get("reference_conditioned_prediction")
        )
        if not _equal_optional(
            diagnostic.get("base_prediction"),
            base_values[index],
        ):
            raise ValueError(f"{identifiers[index]}: base prediction drifted")
        calibrator = diagnostic.get("calibrator")
        if vector_available[index]:
            if (
                not isinstance(calibrator, Mapping)
                or calibrator.get("sample_id") != identifiers[index]
                or calibrator.get("group_id") != str(groups[index])
                or calibrator.get("joint_role")
                != "external_outer_validation_candidate"
                or not _equal_optional(
                    calibrator.get("corrected_prediction_oof"),
                    calibrated,
                )
            ):
                raise ValueError(
                    f"{identifiers[index]}: fold-specific calibrator drifted"
                )
        elif calibrator is not None or calibrated is not None:
            raise ValueError(
                f"{identifiers[index]}: unavailable vector has calibrator output"
            )
        calibrated_values.append(calibrated)
        variants = diagnostic.get("variants")
        if not isinstance(variants, Mapping) or set(variants) != set(
            FADR_ROUTER_FEATURE_SETS
        ):
            raise ValueError(f"{identifiers[index]}: variants missing")
        for variant in FADR_ROUTER_FEATURE_SETS:
            variant_row = variants[variant]
            prediction, route = _route(
                base_values[index],
                calibrated,
                finite_float(variant_row.get("router_score_oof")),
                finite_float(variant_row.get("nested_threshold")),
            )
            error = normalized_error(row, prediction)
            if (
                variant_row.get("route") != route
                or not _equal_optional(
                    variant_row.get("prediction"),
                    prediction,
                )
                or not _equal_optional(
                    variant_row.get("normalized_error"),
                    error,
                )
            ):
                raise ValueError(
                    f"{identifiers[index]}, {variant}: route/error drifted"
                )
            variant_values[variant].append(prediction)
            variant_routes[variant].append(route)

    calibrated_success = np.asarray(
        [value is not None for value in calibrated_values]
    )
    base_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, base_values)
        ],
        dtype=np.float64,
    )
    calibrated_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, calibrated_values)
        ],
        dtype=np.float64,
    )
    hard_values = [
        base if base is not None else calibrated
        for base, calibrated in zip(base_values, calibrated_values)
    ]
    hard_success = base_success | calibrated_success
    hard_error = np.asarray(
        [
            normalized_error(row, value)
            for row, value in zip(rows, hard_values)
        ],
        dtype=np.float64,
    )
    baselines = {
        "base_mask": _metrics(base_error, base_success),
        "reference_conditioned_vector": _metrics(
            calibrated_error,
            calibrated_success,
        ),
        "hard_fallback": _metrics(hard_error, hard_success),
    }
    for name, metrics in baselines.items():
        _assert_metrics(
            metrics,
            summary["baselines"][name],
            label=f"baseline.{name}",
        )
    errors_by_variant: dict[str, np.ndarray] = {}
    for variant in FADR_ROUTER_FEATURE_SETS:
        values = variant_values[variant]
        routes = variant_routes[variant]
        errors = np.asarray(
            [
                normalized_error(row, value)
                for row, value in zip(rows, values)
            ],
            dtype=np.float64,
        )
        successful = np.asarray([value is not None for value in values])
        _assert_metrics(
            _metrics(errors, successful),
            summary["variants"][variant]["metrics"],
            label=f"variant.{variant}",
        )
        switched = np.asarray(
            [route == "calibrated_quality_switch" for route in routes]
        )
        routing = {
            "counts": dict(sorted(Counter(routes).items())),
            "quality_switches": int(np.sum(switched)),
            "positive_transfers": int(
                np.sum(switched & (calibrated_error < base_error))
            ),
            "negative_transfers": int(
                np.sum(switched & (calibrated_error > base_error))
            ),
        }
        if routing != summary["variants"][variant]["routing"]:
            raise ValueError(f"variant.{variant}: routing summary drifted")
        expected_comparisons = {
            "router_vs_base_mask": _paired_group_bootstrap(
                errors,
                base_error,
                groups,
                seed=expected_seed,
                iterations=int(parameters["bootstrap_iterations"]),
            ),
            "router_vs_reference_conditioned_vector": _paired_group_bootstrap(
                errors,
                calibrated_error,
                groups,
                seed=expected_seed + 1,
                iterations=int(parameters["bootstrap_iterations"]),
            ),
            "router_vs_hard_fallback": _paired_group_bootstrap(
                errors,
                hard_error,
                groups,
                seed=expected_seed + 2,
                iterations=int(parameters["bootstrap_iterations"]),
            ),
        }
        if variant != "full":
            expected_comparisons["router_vs_full"] = _paired_group_bootstrap(
                errors,
                np.asarray(
                    [
                        normalized_error(row, value)
                        for row, value in zip(
                            rows,
                            variant_values["full"],
                        )
                    ],
                    dtype=np.float64,
                ),
                groups,
                seed=expected_seed + 3,
                iterations=int(parameters["bootstrap_iterations"]),
            )
        if expected_comparisons != summary["variants"][variant][
            "paired_comparisons"
        ]:
            raise ValueError(f"variant.{variant}: paired comparison drifted")
        errors_by_variant[variant] = errors
    return {
        "schema_version": 1,
        "protocol": FADR_JOINT_VERIFICATION_PROTOCOL,
        "status": "verified",
        "seed": expected_seed,
        "scope": "SyncG/train joint outer-group stacking only",
        "joint_outer_group_nested": True,
        "combined_fadr_oof_authorized": True,
        "standalone_calibrator_oof_used": False,
        "standalone_router_oof_used": False,
        "input": str(oof_pairs),
        "input_sha256": sha256_file(oof_pairs),
        "input_oof_protocol": EXPECTED_OOF_PROTOCOL,
        "input_preflight": str(input_preflight),
        "input_preflight_sha256": sha256_file(input_preflight),
        "input_authorization_sha256": preflight[
            "input_authorization_sha256"
        ],
        "joint_root": str(joint_root),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "lineage": str(lineage_path),
        "lineage_sha256": sha256_file(lineage_path),
        "diagnostics": str(diagnostics_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "samples": len(rows),
        "groups": len(set(groups.tolist())),
        "outer_folds": int(parameters["outer_folds"]),
        "feature_sets": list(FADR_ROUTER_FEATURE_SETS),
        "metrics": {
            variant: summary["variants"][variant]["metrics"]
            for variant in FADR_ROUTER_FEATURE_SETS
        },
        "group_leakage_count": 0,
        "test_samples_used": 0,
        "public_samples_used": 0,
        "field_samples_used": 0,
        "public_test_field_evaluation_authorized": False,
        "source_identity": {
            "verifier": sha256_source_file(Path(__file__).resolve()),
            "joint_trainer": expected_sources["joint_trainer"],
            "protocol": sha256_source_file(
                PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
            ),
        },
        "source_hash_protocol": SOURCE_TEXT_SHA256_PROTOCOL,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    assert_train_only_path(output, label="joint FADR verification output")
    result = verify_joint_run(
        oof_pairs=args.oof_pairs,
        input_preflight=args.input_preflight,
        joint_root=args.joint_root,
        expected_seed=args.expected_seed,
    )
    _write_no_clobber(
        output,
        (
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
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
