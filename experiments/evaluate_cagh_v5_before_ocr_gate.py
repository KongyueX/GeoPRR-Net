"""Evaluate the frozen public-only V5 performance gate before OCR training.

This evaluator reads only JSON summaries named by the frozen protocol.  It
does not load images, predictions, checkpoints, public test data, field data,
or sealed/confirmatory data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "experiments/cagh_v5_before_ocr_gate_protocol.json"
DEFAULT_OUTPUT_ROOT = Path(r"C:\pointer_read\cagh_v5_before_ocr_gate_v1")
PROTOCOL_NAME = "cagh_v5_before_ocr_public_gate_v1"
V5_PROTOCOL_NAME = "cagh_v5_enhanced_authoritative_pepd_oof_v1"
BASELINE_PROTOCOL_NAME = "strict_current_experiment_common_holdout_v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def exact_integer(value: Any, label: str) -> int:
    result = finite_number(value, label)
    require(result.is_integer(), f"{label} is not an integer")
    return int(result)


def _absolute(path_text: Any, label: str) -> Path:
    require(isinstance(path_text, str) and path_text, f"{label} path is missing")
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    require(path.is_file(), f"{label} is missing: {path}")
    return path


def _check(checks: list[dict[str, Any]], name: str, passed: bool, **evidence: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **evidence})


def _validate_frozen_input(
    record: Mapping[str, Any], label: str, *, protocol_path: Path
) -> tuple[Path, str]:
    path = _absolute(record.get("path"), label)
    expected = str(record.get("sha256") or "").lower()
    require(len(expected) == 64 and all(char in "0123456789abcdef" for char in expected), f"{label} SHA256 is invalid")
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA256 drift: expected {expected}, got {actual}")
    if label == "evaluator_source":
        require(path == Path(__file__).resolve(), "evaluator source path drift")
    require(path != protocol_path.resolve(), f"{label} cannot self-bind the gate protocol")
    return path, actual


def load_protocol(protocol_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    protocol_path = protocol_path.resolve()
    protocol = strict_json(protocol_path)
    require(protocol.get("schema_version") == 1, "gate protocol schema drift")
    require(protocol.get("protocol") == PROTOCOL_NAME, "gate protocol identity drift")
    require(protocol.get("status") == "frozen_public_only", "gate protocol is not frozen")
    scope = protocol.get("scope")
    require(isinstance(scope, Mapping), "gate scope is missing")
    for key in (
        "public_test_data_allowed",
        "field_data_allowed",
        "sealed_data_allowed",
        "confirmatory_data_allowed",
        "ocr_training_authorized_by_protocol",
    ):
        require(scope.get(key) is False, f"gate scope unexpectedly authorizes {key}")

    frozen_inputs = protocol.get("frozen_inputs")
    require(isinstance(frozen_inputs, Mapping), "frozen inputs are missing")
    required_inputs = ("strict_common_holdout_v2", "v5_oof_protocol", "v5_oof_runner", "evaluator_source")
    identities: dict[str, dict[str, str]] = {}
    for label in required_inputs:
        record = frozen_inputs.get(label)
        require(isinstance(record, Mapping), f"missing frozen input: {label}")
        path, digest = _validate_frozen_input(record, label, protocol_path=protocol_path)
        identities[label] = {"path": str(path), "sha256": digest}
    v5_artifacts = protocol.get("v5_artifacts")
    require(isinstance(v5_artifacts, Mapping), "V5 artifact protocol is missing")
    require(v5_artifacts.get("formal_root_override_allowed") is False, "formal V5 root override must be disabled")
    return protocol, identities


def _fold_summary_path(output_root: Path, seed: int) -> Path:
    return output_root / "folds" / f"pepd_seed_{seed}" / "summary.json"


def _relative_artifact_path(output_root: Path, relative_path: Any, label: str) -> Path:
    require(isinstance(relative_path, str) and relative_path, f"{label} relative path is missing")
    relative = Path(relative_path)
    require(not relative.is_absolute(), f"{label} must be relative to the frozen V5 root")
    path = (output_root / relative).resolve()
    require(path.is_relative_to(output_root), f"{label} escapes the frozen V5 root")
    require(path.is_file(), f"{label} is missing: {path}")
    return path


def _recorded_artifact_path(value: Any, expected: Path, label: str) -> None:
    require(isinstance(value, str) and value, f"{label} recorded path is missing")
    require(Path(value).resolve() == expected.resolve(), f"{label} recorded path drift")


def _zero_access_counts(record: Mapping[str, Any], label: str) -> None:
    for key in (
        "field_samples_read",
        "public_test_samples_read",
        "confirmatory_samples_read",
        "sealed_samples_read",
    ):
        if key in record:
            require(exact_integer(record.get(key), f"{label} {key}") == 0, f"{label} {key} is non-zero")


def _validate_aggregate_chain(
    *,
    protocol: Mapping[str, Any],
    frozen_identities: Mapping[str, Mapping[str, str]],
    v5_root: Path,
    fold_results: list[dict[str, Any]],
) -> dict[str, Any]:
    artifacts = protocol.get("v5_artifacts")
    require(isinstance(artifacts, Mapping), "V5 artifact protocol is missing")
    root_summary_path = _relative_artifact_path(
        v5_root, artifacts.get("aggregate_summary_relative_path"), "V5 aggregate summary"
    )
    strict_summary_path = _relative_artifact_path(
        v5_root, artifacts.get("strict_oof_summary_relative_path"), "V5 strict OOF summary"
    )
    root_summary = strict_json(root_summary_path)
    strict_summary = strict_json(strict_summary_path)
    for label, value in (("V5 aggregate summary", root_summary), ("V5 strict OOF summary", strict_summary)):
        require(value.get("schema_version") == 1, f"{label} schema drift")
        require(value.get("protocol") == V5_PROTOCOL_NAME, f"{label} protocol drift")
        require(value.get("status") == "complete", f"{label} is incomplete")

    root_artifacts = root_summary.get("artifacts")
    require(isinstance(root_artifacts, Mapping), "V5 aggregate artifact block is missing")
    _recorded_artifact_path(
        root_artifacts.get("strict_oof_summary"), strict_summary_path, "V5 strict OOF summary"
    )
    strict_sha = sha256_file(strict_summary_path)
    require(
        root_artifacts.get("strict_oof_summary_sha256") == strict_sha,
        "V5 aggregate -> strict OOF summary SHA256 chain drift",
    )
    require(root_summary.get("strict_oof") == strict_summary, "V5 embedded strict OOF summary drift")
    _recorded_artifact_path(root_summary.get("output_root"), v5_root, "V5 aggregate output root")

    final_input_sha = root_summary.get("input_sha256")
    require(isinstance(final_input_sha, Mapping), "V5 aggregate input SHA256 block is missing")
    require(
        final_input_sha.get("protocol") == frozen_identities["v5_oof_protocol"]["sha256"],
        "V5 aggregate protocol SHA256 binding drift",
    )
    require(
        final_input_sha.get("runner_source") == frozen_identities["v5_oof_runner"]["sha256"],
        "V5 aggregate runner SHA256 binding drift",
    )

    root_scope = root_summary.get("scope")
    require(isinstance(root_scope, Mapping), "V5 aggregate scope is missing")
    _zero_access_counts(root_scope, "V5 aggregate scope")
    _zero_access_counts(strict_summary, "V5 strict OOF summary")

    strict_metrics = strict_summary.get("metrics")
    strict_audit = strict_summary.get("overlap_and_assignment_audit")
    strict_folds = strict_summary.get("folds")
    require(isinstance(strict_metrics, Mapping), "V5 strict OOF metrics are missing")
    require(isinstance(strict_audit, Mapping), "V5 strict OOF assignment audit is missing")
    require(isinstance(strict_folds, list) and len(strict_folds) == 3, "V5 strict OOF fold chain must contain three folds")
    require(root_summary.get("overlap_and_assignment_audit") == strict_audit, "V5 aggregate assignment audit drift")
    _zero_access_counts(strict_audit, "V5 strict OOF assignment audit")
    require(strict_audit.get("all_rows_jointly_unseen_by_pepd_and_head") is True, "V5 jointly-unseen union contract is missing")

    v5_protocol_path = Path(frozen_identities["v5_oof_protocol"]["path"])
    v5_protocol = strict_json(v5_protocol_path)
    require(v5_protocol.get("protocol") == V5_PROTOCOL_NAME, "frozen V5 protocol identity drift")
    require(v5_protocol.get("status") == "frozen_public_only", "frozen V5 protocol status drift")
    v5_assignment = v5_protocol.get("assignment")
    require(isinstance(v5_assignment, Mapping), "frozen V5 assignment block is missing")
    expected_union = artifacts.get("expected_union")
    require(isinstance(expected_union, Mapping), "gate expected V5 union is missing")
    for key in (
        "eligible_union_samples",
        "eligible_union_groups",
        "union_sample_ids_sha256",
        "union_group_ids_sha256",
        "sample_assignment_sha256",
        "group_assignment_sha256",
    ):
        require(expected_union.get(key) == v5_assignment.get(key), f"gate/V5 protocol union {key} drift")
        require(strict_audit.get(key) == expected_union.get(key), f"V5 aggregate union {key} drift")
    require(
        exact_integer(strict_metrics.get("samples"), "V5 strict OOF union samples")
        == exact_integer(expected_union.get("eligible_union_samples"), "expected V5 union samples"),
        "V5 strict OOF union sample count drift",
    )
    require(
        exact_integer(strict_metrics.get("groups"), "V5 strict OOF union groups")
        == exact_integer(expected_union.get("eligible_union_groups"), "expected V5 union groups"),
        "V5 strict OOF union group count drift",
    )
    require(finite_number(strict_metrics.get("failed_rows_penalty"), "V5 strict OOF failure penalty") == 1.0, "V5 strict OOF failure penalty drift")

    fold_by_seed: dict[int, Mapping[str, Any]] = {}
    for entry in strict_folds:
        require(isinstance(entry, Mapping), "V5 strict OOF fold entry is malformed")
        seed = exact_integer(entry.get("pepd_seed"), "V5 strict OOF fold seed")
        require(seed not in fold_by_seed, f"duplicate V5 aggregate fold seed: {seed}")
        fold_by_seed[seed] = entry
    expected_seeds = {int(row["pepd_seed"]) for row in fold_results}
    require(set(fold_by_seed) == expected_seeds, "V5 aggregate fold roster drift")
    for fold in fold_results:
        seed = int(fold["pepd_seed"])
        aggregate_fold = fold_by_seed[seed]
        fold_path = Path(fold["summary"]).resolve()
        _recorded_artifact_path(aggregate_fold.get("summary"), fold_path, f"seed {seed} aggregate fold summary")
        require(
            aggregate_fold.get("summary_sha256") == fold["summary_sha256"],
            f"seed {seed}: aggregate fold summary SHA256 drift",
        )
        require(
            exact_integer(aggregate_fold.get("head_seed"), f"seed {seed}: aggregate head seed")
            == exact_integer(fold["head_seed"], f"seed {seed}: fold head seed"),
            f"seed {seed}: aggregate head-seed drift",
        )
        require(
            aggregate_fold.get("strict_assigned_metrics")
            == strict_json(fold_path).get("strict_assigned_metrics"),
            f"seed {seed}: aggregate strict metrics drift",
        )

    return {
        "aggregate_summary": str(root_summary_path),
        "aggregate_summary_sha256": sha256_file(root_summary_path),
        "strict_oof_summary": str(strict_summary_path),
        "strict_oof_summary_sha256": strict_sha,
        "union_samples": exact_integer(strict_metrics.get("samples"), "V5 union samples"),
        "union_groups": exact_integer(strict_metrics.get("groups"), "V5 union groups"),
        "union_sample_ids_sha256": str(strict_audit.get("union_sample_ids_sha256")),
        "union_group_ids_sha256": str(strict_audit.get("union_group_ids_sha256")),
    }


def evaluate(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    v5_output_root: Path | None = None,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    protocol, frozen_identities = load_protocol(protocol_path)
    inputs = protocol["frozen_inputs"]
    baseline_path = _absolute(inputs["strict_common_holdout_v2"]["path"], "strict_common_holdout_v2")
    baseline = strict_json(baseline_path)
    require(baseline.get("schema_version") == 2, "strict-common schema drift")
    require(baseline.get("protocol") == BASELINE_PROTOCOL_NAME, "strict-common protocol drift")
    require(baseline.get("status") == "complete", "strict-common summary is incomplete")

    selection = baseline.get("selection")
    methods = baseline.get("metrics")
    scoring = baseline.get("scoring")
    require(isinstance(selection, Mapping), "strict-common selection is missing")
    require(isinstance(methods, Mapping), "strict-common metrics are missing")
    require(isinstance(scoring, Mapping), "strict-common scoring is missing")
    require(finite_number(scoring.get("failure_penalty"), "baseline failure penalty") == 1.0, "baseline failure penalty drift")

    comparison = protocol.get("comparison")
    thresholds = protocol.get("thresholds")
    folds_protocol = protocol.get("folds")
    require(isinstance(comparison, Mapping), "comparison protocol is missing")
    require(isinstance(thresholds, Mapping), "thresholds are missing")
    require(isinstance(folds_protocol, list) and len(folds_protocol) == 3, "exactly three folds are required")

    baseline_key = str(comparison.get("baseline_method_key") or "")
    baseline_metrics = methods.get(baseline_key)
    require(isinstance(baseline_metrics, Mapping), f"baseline method is missing: {baseline_key}")
    baseline_nmae = finite_number(baseline_metrics.get("nmae"), "baseline NMAE")
    baseline_coverage = finite_number(baseline_metrics.get("coverage"), "baseline coverage")
    require(0.0 < baseline_nmae <= 1.0, "baseline NMAE is outside (0, 1]")
    require(0.0 <= baseline_coverage <= 1.0, "baseline coverage is outside [0, 1]")

    cohort_seed = exact_integer(comparison.get("same_cohort_seed"), "same-cohort seed")
    baseline_seed = exact_integer(selection.get("held_out_seed"), "baseline held-out seed")
    require(cohort_seed == baseline_seed, "same-cohort seed differs from baseline held-out seed")
    cohort_hash = str(selection.get("sample_ids_sha256") or "")
    require(len(cohort_hash) == 64, "baseline sample cohort hash is invalid")
    cohort_samples = exact_integer(selection.get("samples"), "baseline samples")
    cohort_groups = exact_integer(selection.get("physical_groups"), "baseline groups")

    v5_artifacts = protocol.get("v5_artifacts")
    require(isinstance(v5_artifacts, Mapping), "V5 artifact protocol is missing")
    frozen_root_text = v5_artifacts.get("output_root")
    require(isinstance(frozen_root_text, str) and frozen_root_text, "frozen V5 output root is missing")
    frozen_root = Path(frozen_root_text).resolve()
    requested_root = Path(v5_output_root).resolve() if v5_output_root is not None else frozen_root
    require(requested_root == frozen_root, "V5 output root differs from the frozen protocol root")
    v5_root = frozen_root
    fold_results: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    observed_seeds: set[int] = set()
    same_cohort_metrics: Mapping[str, Any] | None = None
    same_cohort_split: Mapping[str, Any] | None = None

    max_nmae = finite_number(thresholds.get("per_fold_max_nmae"), "max fold NMAE")
    min_coverage = finite_number(thresholds.get("per_fold_min_coverage"), "min fold coverage")
    max_group_macro = finite_number(thresholds.get("per_fold_max_group_macro_nmae"), "max fold group macro NMAE")
    max_p95 = finite_number(thresholds.get("per_fold_max_p95_nmae"), "max fold P95 NMAE")

    for fold_spec in folds_protocol:
        require(isinstance(fold_spec, Mapping), "fold protocol entry is malformed")
        seed = exact_integer(fold_spec.get("pepd_seed"), "fold PEPD seed")
        require(seed not in observed_seeds, f"duplicate fold seed: {seed}")
        observed_seeds.add(seed)
        summary_path = _fold_summary_path(v5_root, seed)
        require(summary_path.is_file(), f"V5 fold summary is missing: {summary_path}")
        summary = strict_json(summary_path)
        require(summary.get("schema_version") == 1, f"seed {seed}: summary schema drift")
        require(summary.get("protocol") == V5_PROTOCOL_NAME, f"seed {seed}: V5 protocol drift")
        require(summary.get("status") == "complete", f"seed {seed}: fold is incomplete")
        require(summary.get("jointly_unseen_contract") is True, f"seed {seed}: jointly-unseen contract missing")
        require(exact_integer(summary.get("pepd_seed"), f"seed {seed}: recorded PEPD seed") == seed, f"seed {seed}: identity drift")
        require(exact_integer(summary.get("head_seed"), f"seed {seed}: head seed") == exact_integer(fold_spec.get("head_seed"), f"seed {seed}: expected head seed"), f"seed {seed}: head-seed drift")
        split = summary.get("split_identity")
        metrics = summary.get("strict_assigned_metrics")
        require(isinstance(split, Mapping), f"seed {seed}: split identity is missing")
        require(isinstance(metrics, Mapping), f"seed {seed}: strict metrics are missing")
        require(split.get("validation_sample_ids_sha256") == fold_spec.get("validation_sample_ids_sha256"), f"seed {seed}: validation cohort hash drift")
        require(exact_integer(split.get("group_overlap"), f"seed {seed}: split group overlap") == 0, f"seed {seed}: train/validation group overlap")
        samples = exact_integer(metrics.get("samples"), f"seed {seed}: samples")
        groups = exact_integer(metrics.get("groups"), f"seed {seed}: groups")
        require(samples == exact_integer(fold_spec.get("strict_assigned_samples"), f"seed {seed}: expected samples"), f"seed {seed}: strict sample count drift")
        require(groups == exact_integer(fold_spec.get("strict_assigned_groups"), f"seed {seed}: expected groups"), f"seed {seed}: strict group count drift")
        require(finite_number(metrics.get("failed_rows_penalty"), f"seed {seed}: failure penalty") == 1.0, f"seed {seed}: failure penalty drift")

        nmae = finite_number(metrics.get("full_denominator_nmae"), f"seed {seed}: NMAE")
        coverage = finite_number(metrics.get("coverage"), f"seed {seed}: coverage")
        group_macro = finite_number(metrics.get("group_macro_full_denominator_nmae"), f"seed {seed}: group macro NMAE")
        p95 = finite_number(metrics.get("p95_absolute_progress_error"), f"seed {seed}: P95")
        for label, value in (("NMAE", nmae), ("coverage", coverage), ("group macro NMAE", group_macro), ("P95", p95)):
            require(0.0 <= value <= 1.0, f"seed {seed}: {label} outside [0, 1]")
        seed_checks = {
            "nmae": nmae <= max_nmae,
            "coverage": coverage >= min_coverage,
            "group_macro_nmae": group_macro <= max_group_macro,
            "p95_nmae": p95 <= max_p95,
        }
        for metric_name, passed in seed_checks.items():
            _check(checks, f"fold_{seed}_{metric_name}", passed)
        fold_results.append(
            {
                "pepd_seed": seed,
                "head_seed": exact_integer(summary.get("head_seed"), f"seed {seed}: head seed"),
                "summary": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path),
                "samples": samples,
                "groups": groups,
                "metrics": {
                    "nmae": nmae,
                    "coverage": coverage,
                    "group_macro_nmae": group_macro,
                    "p95_nmae": p95,
                },
                "checks": seed_checks,
            }
        )
        if seed == cohort_seed:
            same_cohort_metrics = metrics
            same_cohort_split = split

    require(same_cohort_metrics is not None and same_cohort_split is not None, "same-cohort V5 fold is missing")
    exact_hash_match = same_cohort_split.get("validation_sample_ids_sha256") == cohort_hash
    exact_count_match = (
        exact_integer(same_cohort_metrics.get("samples"), "same-cohort V5 samples") == cohort_samples
        and exact_integer(same_cohort_metrics.get("groups"), "same-cohort V5 groups") == cohort_groups
    )
    _check(checks, "same_cohort_sample_hash_exact_match", exact_hash_match, sample_ids_sha256=cohort_hash)
    _check(checks, "same_cohort_counts_exact_match", exact_count_match, samples=cohort_samples, groups=cohort_groups)

    v5_nmae = finite_number(same_cohort_metrics.get("full_denominator_nmae"), "same-cohort V5 NMAE")
    v5_coverage = finite_number(same_cohort_metrics.get("coverage"), "same-cohort V5 coverage")
    relative_improvement = (baseline_nmae - v5_nmae) / baseline_nmae
    min_improvement = finite_number(thresholds.get("same_cohort_min_relative_nmae_improvement"), "minimum relative NMAE improvement")
    max_coverage_drop = finite_number(thresholds.get("same_cohort_max_coverage_drop_absolute"), "maximum coverage drop")
    improvement_pass = relative_improvement >= min_improvement
    coverage_pass = v5_coverage >= baseline_coverage - max_coverage_drop
    _check(checks, "same_cohort_relative_nmae_improvement", improvement_pass)
    _check(checks, "same_cohort_coverage_noninferiority", coverage_pass)

    aggregate_chain = _validate_aggregate_chain(
        protocol=protocol,
        frozen_identities=frozen_identities,
        v5_root=v5_root,
        fold_results=fold_results,
    )

    decision = "pass" if all(check["passed"] for check in checks) else "fail"
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_NAME,
        "status": "complete",
        "decision": decision,
        "decision_meaning": "OCR may proceed only when decision is pass; this gate itself does not launch OCR.",
        "protocol_file": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
        "frozen_inputs": frozen_identities,
        "runtime_inputs": {
            "v5_output_root": str(v5_root),
            "fold_summaries": fold_results,
            "aggregate_chain": aggregate_chain,
        },
        "thresholds": dict(thresholds),
        "same_cohort_comparison": {
            "pepd_seed": cohort_seed,
            "sample_ids_sha256": cohort_hash,
            "samples": cohort_samples,
            "groups": cohort_groups,
            "v5_nmae": v5_nmae,
            "v5_coverage": v5_coverage,
            "scalemark_v4_three_seed_mean_nmae": baseline_nmae,
            "scalemark_v4_three_seed_mean_coverage": baseline_coverage,
            "relative_nmae_improvement": relative_improvement,
            "coverage_difference_signed": v5_coverage - baseline_coverage,
        },
        "checks": checks,
        "data_access_audit": {
            "images_read": 0,
            "annotations_read": 0,
            "prediction_rows_read": 0,
            "public_test_samples_read": 0,
            "field_samples_read": 0,
            "sealed_samples_read": 0,
            "confirmatory_samples_read": 0,
        },
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--v5-output-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(protocol_path=args.protocol, v5_output_root=args.v5_output_root)
    output_path = args.output_root.resolve() / "summary.json"
    atomic_json(output_path, result)
    print(output_path)
    print(f"decision={result['decision']}")
    return 0 if result["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
