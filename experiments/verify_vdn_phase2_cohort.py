"""Verify all three VDN Phase-2 runs before any supporting test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.freeze_vdn_phase2_syncg_train_inventory import (
    DEFAULT_OUTPUT as _DEFAULT_CONTENT_INVENTORY,
)
from experiments.probe_vdn_phase2_determinism import (
    DEFAULT_DETERMINISM_REPORT as _DEFAULT_DETERMINISM_REPORT,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file, sha256_source_file
from experiments.vdn_phase2_evaluation_plan import (
    evaluation_execution_source_identity,
    evaluation_plan_source_sha256,
)
from experiments.vdn_phase2_protocol import (
    FORMAL_PHASE2_SEEDS,
    PHASE2_END_EPOCH,
    PHASE2_PROTOCOL,
    PHASE2_VERIFICATION_PROTOCOL,
)
from experiments.verify_vdn_phase2 import verify_phase2_run


COHORT_PROTOCOL = "vdn_phase2_three_seed_cohort_verification_v1"
COHORT_SCHEMA_VERSION = 1
COHORT_AUTHORIZATION_SCOPE = (
    "frozen VDN supporting test evaluation only; this artifact does "
    "not authorize field confirmatory evaluation"
)
FORMAL_PARENT_SEEDS = tuple(sorted(FORMAL_PHASE2_SEEDS))
DEFAULT_RUN_ROOT = PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg_phase2"
DEFAULT_PARENT_ROOT = PROJECT_DIR / "artifacts" / "runs" / "vdn_syncg"
DEFAULT_MANIFEST = PROJECT_DIR / "artifacts" / "manifests" / "syncg_train.jsonl"
DEFAULT_VDN_SOURCE = (
    PROJECT_DIR / "artifacts" / "vendor" / "VectorDetectionNetwork"
)
DEFAULT_CONTENT_INVENTORY = (
    _DEFAULT_CONTENT_INVENTORY
    if _DEFAULT_CONTENT_INVENTORY.is_absolute()
    else PROJECT_DIR / _DEFAULT_CONTENT_INVENTORY
)
DEFAULT_DETERMINISM_REPORT = (
    _DEFAULT_DETERMINISM_REPORT
    if _DEFAULT_DETERMINISM_REPORT.is_absolute()
    else PROJECT_DIR / _DEFAULT_DETERMINISM_REPORT
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "vdn_phase2_three_seed_cohort_v1.json"
)
INDIVIDUAL_REPORT_NAME = "verification_v2.json"

_INDIVIDUAL_KEYS = frozenset(
    {
        "protocol",
        "verified",
        "training_artifacts_verified",
        "eligible_for_test_evaluation",
        "run_dir",
        "parent_run_dir",
        "parent_summary_sha256",
        "parent_last_checkpoint_sha256",
        "parent_best_checkpoint_sha256",
        "parent_verification_sha256",
        "phase_seed",
        "epochs",
        "best_epoch",
        "best_validation_angle_mae_degrees",
        "convergence",
        "optimizer_attempts_per_epoch",
        "optimizer_attempts",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "skipped_optimizer_step_rate",
        "max_skipped_optimizer_step_rate",
        "full_phase_max_skipped_optimizer_steps",
        "content_inventory",
        "determinism_authorization",
        "determinism_policy",
        "authorized_runtime_environment",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "summary_sha256",
        "source_hash_protocol",
        "verifier_source_sha256",
        "parent_state_health",
        "authoritative_checkpoint_health",
        "derived_best_model_health",
    }
)

_COHORT_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "status",
        "verified",
        "three_seed_cohort_complete",
        "vdn_supporting_test_evaluation_authorized",
        "field_confirmatory_evaluation_authorized",
        "authorization_scope",
        "parent_seeds",
        "phase_seeds",
        "runs",
        "shared_identity",
        "best_validation_angle_mae_degrees",
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "cohort_verifier_source_sha256",
        "evaluation_plan_source_sha256",
        "evaluation_execution_source_identity",
        "canonical_cohort_payload_sha256",
    }
)

_COHORT_RUN_KEYS = frozenset(
    {
        "parent_seed",
        "phase_seed",
        "best_epoch",
        "best_validation_angle_mae_degrees",
        "optimizer_attempts",
        "optimizer_steps",
        "skipped_optimizer_steps",
        "skipped_optimizer_step_rate",
        "phase2_improved_over_parent",
        "run_dir",
        "parent_run_dir",
        "verification_path",
        "verification_sha256",
        "verification_canonical_sha256",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "summary_sha256",
    }
)

_SHARED_IDENTITY_KEYS = frozenset(
    {
        "content_inventory",
        "determinism_authorization",
        "determinism_policy",
        "authorized_runtime_environment",
        "source_hash_protocol",
        "verifier_source_sha256",
    }
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: verification report is not UTF-8") from exc

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: verification report is not an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_individual(
    value: Mapping[str, Any],
    *,
    seed: int,
    run_dir: Path,
    parent_dir: Path,
) -> dict[str, Any]:
    if set(value) != _INDIVIDUAL_KEYS:
        raise ValueError(f"seed {seed} verification schema drifted")
    if (
        value.get("protocol") != PHASE2_VERIFICATION_PROTOCOL
        or value.get("verified") is not True
        or value.get("training_artifacts_verified") is not True
        or value.get("eligible_for_test_evaluation") is not True
    ):
        raise ValueError(f"seed {seed} is not an eligible Phase-2 verification")
    if (
        value.get("run_dir") != str(run_dir.resolve())
        or value.get("parent_run_dir") != str(parent_dir.resolve())
        or int(value.get("phase_seed", -1)) != FORMAL_PHASE2_SEEDS[seed]
        or int(value.get("epochs", -1)) != PHASE2_END_EPOCH
    ):
        raise ValueError(f"seed {seed} run/parent/phase identity drifted")
    convergence = value.get("convergence")
    if (
        not isinstance(convergence, Mapping)
        or convergence.get("passed") is not True
    ):
        raise ValueError(f"seed {seed} convergence gate did not pass")

    attempts = int(value.get("optimizer_attempts", -1))
    successful = int(value.get("optimizer_steps", -1))
    skipped = int(value.get("skipped_optimizer_steps", -1))
    if attempts <= 0 or successful < 0 or skipped < 0:
        raise ValueError(f"seed {seed} optimizer accounting is invalid")
    if successful + skipped != attempts:
        raise ValueError(f"seed {seed} optimizer accounting does not close")
    skip_rate = _require_finite(
        value.get("skipped_optimizer_step_rate"),
        f"seed {seed} skipped optimizer step rate",
    )
    maximum = _require_finite(
        value.get("max_skipped_optimizer_step_rate"),
        f"seed {seed} maximum skipped optimizer step rate",
    )
    if (
        not math.isclose(
            skip_rate,
            skipped / attempts,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or skip_rate > maximum
        or skipped > int(value.get("full_phase_max_skipped_optimizer_steps", -1))
    ):
        raise ValueError(f"seed {seed} AMP skip accounting is invalid")

    for field in (
        "parent_summary_sha256",
        "parent_last_checkpoint_sha256",
        "parent_best_checkpoint_sha256",
        "parent_verification_sha256",
        "best_checkpoint_sha256",
        "last_checkpoint_sha256",
        "summary_sha256",
        "verifier_source_sha256",
    ):
        _require_sha256(value.get(field), f"seed {seed} {field}")
    for field in (
        "content_inventory",
        "determinism_authorization",
        "determinism_policy",
        "authorized_runtime_environment",
        "parent_state_health",
        "authoritative_checkpoint_health",
        "derived_best_model_health",
    ):
        if not isinstance(value.get(field), Mapping):
            raise ValueError(f"seed {seed} {field} is not an object")
    best_angle = _require_finite(
        value.get("best_validation_angle_mae_degrees"),
        f"seed {seed} best validation angle MAE",
    )
    if best_angle < 0.0:
        raise ValueError(f"seed {seed} best validation angle MAE is negative")
    return {
        "parent_seed": int(seed),
        "phase_seed": int(value["phase_seed"]),
        "best_epoch": int(value["best_epoch"]),
        "best_validation_angle_mae_degrees": best_angle,
        "optimizer_attempts": attempts,
        "optimizer_steps": successful,
        "skipped_optimizer_steps": skipped,
        "skipped_optimizer_step_rate": skip_rate,
        "phase2_improved_over_parent": bool(
            convergence.get("phase2_improved_over_parent")
        ),
    }


def build_cohort_report(
    entries: Sequence[tuple[int, Path, Mapping[str, Any], Mapping[str, Any]]],
    *,
    run_root: Path,
    parent_root: Path,
) -> dict[str, Any]:
    """Build a passed cohort report from stored and fresh-equal verifications."""

    run_root = Path(run_root).resolve()
    parent_root = Path(parent_root).resolve()
    observed_seeds = [int(entry[0]) for entry in entries]
    if observed_seeds != list(FORMAL_PARENT_SEEDS):
        raise ValueError(
            f"cohort seeds must be exactly {list(FORMAL_PARENT_SEEDS)}"
        )

    records: list[dict[str, Any]] = []
    shared: dict[str, Any] | None = None
    for seed, report_path, stored, fresh in entries:
        seed = int(seed)
        expected_run = run_root / f"seed_{seed}"
        expected_parent = parent_root / f"seed_{seed}"
        expected_report = expected_run / INDIVIDUAL_REPORT_NAME
        if Path(report_path).resolve() != expected_report.resolve():
            raise ValueError(f"seed {seed} verification path drifted")
        if dict(stored) != dict(fresh):
            raise ValueError(f"seed {seed} stored verification is stale")
        summary = _validate_individual(
            stored,
            seed=seed,
            run_dir=expected_run,
            parent_dir=expected_parent,
        )
        shared_value = {
            "content_inventory": stored["content_inventory"],
            "determinism_authorization": stored["determinism_authorization"],
            "determinism_policy": stored["determinism_policy"],
            "authorized_runtime_environment": stored[
                "authorized_runtime_environment"
            ],
            "source_hash_protocol": stored["source_hash_protocol"],
            "verifier_source_sha256": stored["verifier_source_sha256"],
        }
        if shared is None:
            shared = shared_value
        elif shared_value != shared:
            raise ValueError(f"seed {seed} shared cohort identity drifted")
        records.append(
            {
                **summary,
                "run_dir": str(expected_run.resolve()),
                "parent_run_dir": str(expected_parent.resolve()),
                "verification_path": str(expected_report.resolve()),
                "verification_sha256": sha256_file(expected_report),
                "verification_canonical_sha256": canonical_json_sha256(stored),
                "best_checkpoint_sha256": stored["best_checkpoint_sha256"],
                "last_checkpoint_sha256": stored["last_checkpoint_sha256"],
                "summary_sha256": stored["summary_sha256"],
            }
        )
    assert shared is not None
    angles = np.asarray(
        [record["best_validation_angle_mae_degrees"] for record in records],
        dtype=np.float64,
    )
    report = {
        "protocol": COHORT_PROTOCOL,
        "schema_version": COHORT_SCHEMA_VERSION,
        "status": "passed",
        "verified": True,
        "three_seed_cohort_complete": True,
        "vdn_supporting_test_evaluation_authorized": True,
        "field_confirmatory_evaluation_authorized": False,
        "authorization_scope": COHORT_AUTHORIZATION_SCOPE,
        "parent_seeds": list(FORMAL_PARENT_SEEDS),
        "phase_seeds": [FORMAL_PHASE2_SEEDS[seed] for seed in FORMAL_PARENT_SEEDS],
        "runs": records,
        "shared_identity": shared,
        "best_validation_angle_mae_degrees": {
            "mean": float(np.mean(angles)),
            "sample_standard_deviation": float(np.std(angles, ddof=1)),
            "minimum": float(np.min(angles)),
            "maximum": float(np.max(angles)),
        },
        "test_data_opened_or_read": False,
        "public_data_opened_or_read": False,
        "field_data_opened_or_read": False,
        "sealed_data_opened_or_read": False,
        "cohort_verifier_source_sha256": sha256_source_file(
            Path(__file__).resolve()
        ),
        "evaluation_plan_source_sha256": evaluation_plan_source_sha256(),
        "evaluation_execution_source_identity": (
            evaluation_execution_source_identity()
        ),
    }
    report["canonical_cohort_payload_sha256"] = canonical_json_sha256(report)
    return report


def validate_cohort_evaluation_authorization(
    report_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Fail closed unless a frozen Phase-2 best checkpoint is cohort-authorized.

    This validator intentionally reads only the authorization report and the
    requested checkpoint.  Evaluation callers must invoke it before opening a
    test/public manifest, prediction cache, or image.
    """

    report_path = Path(report_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    expected_report_path = Path(DEFAULT_OUTPUT).resolve()
    expected_run_root = Path(DEFAULT_RUN_ROOT).resolve()
    expected_parent_root = Path(DEFAULT_PARENT_ROOT).resolve()
    if report_path != expected_report_path:
        raise ValueError(
            "Phase-2 cohort authorization is not the formal report path"
        )
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    report = _strict_json(report_path)
    if set(report) != _COHORT_KEYS:
        raise ValueError("Phase-2 cohort authorization schema drifted")
    if (
        report.get("protocol") != COHORT_PROTOCOL
        or report.get("schema_version") != COHORT_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("verified") is not True
        or report.get("three_seed_cohort_complete") is not True
        or report.get("vdn_supporting_test_evaluation_authorized") is not True
        or report.get("field_confirmatory_evaluation_authorized") is not False
        or report.get("authorization_scope") != COHORT_AUTHORIZATION_SCOPE
    ):
        raise ValueError("Phase-2 cohort does not authorize supporting evaluation")
    for field in (
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
    ):
        if report.get(field) is not False:
            raise ValueError(f"Phase-2 cohort provenance flag changed: {field}")

    expected_parent_seeds = list(FORMAL_PARENT_SEEDS)
    expected_phase_seeds = [
        FORMAL_PHASE2_SEEDS[seed] for seed in FORMAL_PARENT_SEEDS
    ]
    if (
        report.get("parent_seeds") != expected_parent_seeds
        or report.get("phase_seeds") != expected_phase_seeds
    ):
        raise ValueError("Phase-2 cohort seed identity drifted")

    source_digest = _require_sha256(
        report.get("cohort_verifier_source_sha256"),
        "cohort verifier source SHA-256",
    )
    if source_digest != sha256_source_file(Path(__file__).resolve()):
        raise ValueError("Phase-2 cohort verifier source changed after authorization")
    plan_source_digest = _require_sha256(
        report.get("evaluation_plan_source_sha256"),
        "Phase-2 evaluation-plan source SHA-256",
    )
    if plan_source_digest != evaluation_plan_source_sha256():
        raise ValueError(
            "Phase-2 evaluation-plan source changed after authorization"
        )
    execution_source_identity = report.get(
        "evaluation_execution_source_identity"
    )
    if (
        not isinstance(execution_source_identity, Mapping)
        or execution_source_identity
        != evaluation_execution_source_identity()
    ):
        raise ValueError(
            "Phase-2 evaluation execution sources changed after authorization"
        )
    stored_canonical = _require_sha256(
        report.get("canonical_cohort_payload_sha256"),
        "canonical cohort payload SHA-256",
    )
    unsigned_report = dict(report)
    unsigned_report.pop("canonical_cohort_payload_sha256")
    if canonical_json_sha256(unsigned_report) != stored_canonical:
        raise ValueError("Phase-2 cohort canonical digest mismatch")

    shared = report.get("shared_identity")
    if not isinstance(shared, Mapping) or set(shared) != _SHARED_IDENTITY_KEYS:
        raise ValueError("Phase-2 cohort shared identity schema drifted")
    _require_sha256(
        shared.get("verifier_source_sha256"),
        "individual verifier source SHA-256",
    )
    for field in (
        "content_inventory",
        "determinism_authorization",
        "determinism_policy",
        "authorized_runtime_environment",
    ):
        if not isinstance(shared.get(field), Mapping):
            raise ValueError(f"Phase-2 cohort shared {field} is not an object")
    if not isinstance(shared.get("source_hash_protocol"), str):
        raise ValueError("Phase-2 cohort source-hash protocol is invalid")
    for identity, expected_path, label in (
        (
            shared["content_inventory"],
            DEFAULT_CONTENT_INVENTORY.resolve(),
            "content inventory",
        ),
        (
            shared["determinism_authorization"],
            DEFAULT_DETERMINISM_REPORT.resolve(),
            "determinism authorization",
        ),
    ):
        report_value = identity.get("report_path")
        if not isinstance(report_value, str) or not report_value:
            raise ValueError(f"Phase-2 {label} report path is absent")
        bound_path = Path(report_value)
        if not bound_path.is_absolute():
            bound_path = PROJECT_DIR / bound_path
        bound_path = bound_path.resolve()
        if bound_path != expected_path:
            raise ValueError(f"Phase-2 {label} report path drifted")
        if not bound_path.is_file():
            raise FileNotFoundError(bound_path)
        expected_digest = _require_sha256(
            identity.get("report_sha256"),
            f"Phase-2 {label} report SHA-256",
        )
        if sha256_file(bound_path) != expected_digest:
            raise ValueError(f"Phase-2 {label} report digest drifted")

    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != len(FORMAL_PARENT_SEEDS):
        raise ValueError("Phase-2 cohort must contain exactly three runs")
    checkpoint_digest = sha256_file(checkpoint_path)
    authorized_run: Mapping[str, Any] | None = None
    angles: list[float] = []
    for index, (run, parent_seed) in enumerate(
        zip(runs, FORMAL_PARENT_SEEDS, strict=True)
    ):
        if not isinstance(run, Mapping) or set(run) != _COHORT_RUN_KEYS:
            raise ValueError(f"Phase-2 cohort run {index} schema drifted")
        if (
            type(run.get("parent_seed")) is not int
            or int(run["parent_seed"]) != parent_seed
            or type(run.get("phase_seed")) is not int
            or int(run["phase_seed"]) != FORMAL_PHASE2_SEEDS[parent_seed]
        ):
            raise ValueError(f"Phase-2 cohort run {index} seed drifted")
        run_dir = Path(str(run.get("run_dir"))).resolve()
        parent_run_dir = Path(str(run.get("parent_run_dir"))).resolve()
        verification_path = Path(str(run.get("verification_path"))).resolve()
        expected_run_dir = expected_run_root / f"seed_{parent_seed}"
        expected_parent_dir = expected_parent_root / f"seed_{parent_seed}"
        expected_verification_path = (
            expected_run_dir / INDIVIDUAL_REPORT_NAME
        )
        if (
            run_dir != expected_run_dir
            or parent_run_dir != expected_parent_dir
            or verification_path != expected_verification_path
        ):
            raise ValueError(f"Phase-2 cohort run {index} path identity drifted")
        for field in (
            "verification_sha256",
            "verification_canonical_sha256",
            "best_checkpoint_sha256",
            "last_checkpoint_sha256",
            "summary_sha256",
        ):
            _require_sha256(run.get(field), f"Phase-2 run {index} {field}")

        summary_path = run_dir / "summary.json"
        best_path = run_dir / "best.pt"
        last_path = run_dir / "last.pt"
        parent_summary_path = parent_run_dir / "summary.json"
        parent_best_path = parent_run_dir / "best.pt"
        parent_last_path = parent_run_dir / "last.pt"
        parent_verification_path = parent_run_dir / "verification.json"
        for artifact in (
            verification_path,
            summary_path,
            best_path,
            last_path,
            parent_summary_path,
            parent_best_path,
            parent_last_path,
            parent_verification_path,
        ):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        if (
            sha256_file(verification_path) != run["verification_sha256"]
            or sha256_file(summary_path) != run["summary_sha256"]
            or sha256_file(best_path) != run["best_checkpoint_sha256"]
            or sha256_file(last_path) != run["last_checkpoint_sha256"]
        ):
            raise ValueError(
                f"Phase-2 cohort run {index} artifact digest drifted"
            )
        stored_verification = _strict_json(verification_path)
        if (
            canonical_json_sha256(stored_verification)
            != run["verification_canonical_sha256"]
        ):
            raise ValueError(
                f"Phase-2 cohort run {index} verification canonical digest drifted"
            )
        stored_summary = _validate_individual(
            stored_verification,
            seed=parent_seed,
            run_dir=run_dir,
            parent_dir=parent_run_dir,
        )
        if (
            sha256_file(parent_summary_path)
            != stored_verification["parent_summary_sha256"]
            or sha256_file(parent_best_path)
            != stored_verification["parent_best_checkpoint_sha256"]
            or sha256_file(parent_last_path)
            != stored_verification["parent_last_checkpoint_sha256"]
            or sha256_file(parent_verification_path)
            != stored_verification["parent_verification_sha256"]
        ):
            raise ValueError(
                f"Phase-2 cohort run {index} parent artifact digest drifted"
            )
        for field in (
            "phase_seed",
            "best_epoch",
            "optimizer_attempts",
            "optimizer_steps",
            "skipped_optimizer_steps",
        ):
            if int(stored_summary[field]) != int(run[field]):
                raise ValueError(
                    f"Phase-2 cohort run {index} {field} drifted"
                )
        for field in (
            "best_validation_angle_mae_degrees",
            "skipped_optimizer_step_rate",
        ):
            if not math.isclose(
                float(stored_summary[field]),
                float(run[field]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError(
                    f"Phase-2 cohort run {index} {field} drifted"
                )
        if (
            bool(stored_summary["phase2_improved_over_parent"])
            is not bool(run["phase2_improved_over_parent"])
        ):
            raise ValueError(
                f"Phase-2 cohort run {index} convergence summary drifted"
            )
        expected_individual_verifier_sha = sha256_source_file(
            PROJECT_DIR / "experiments" / "verify_vdn_phase2.py"
        )
        if (
            stored_verification["verifier_source_sha256"]
            != expected_individual_verifier_sha
            or shared["verifier_source_sha256"]
            != expected_individual_verifier_sha
        ):
            raise ValueError(
                "Phase-2 individual verifier source changed after authorization"
            )
        attempts = int(run.get("optimizer_attempts", -1))
        successful = int(run.get("optimizer_steps", -1))
        skipped = int(run.get("skipped_optimizer_steps", -1))
        skip_rate = _require_finite(
            run.get("skipped_optimizer_step_rate"),
            f"Phase-2 run {index} skipped optimizer step rate",
        )
        if (
            attempts <= 0
            or successful < 0
            or skipped < 0
            or successful + skipped != attempts
            or not math.isclose(
                skip_rate,
                skipped / attempts,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(f"Phase-2 cohort run {index} optimizer accounting drifted")
        angle = _require_finite(
            run.get("best_validation_angle_mae_degrees"),
            f"Phase-2 run {index} best validation angle MAE",
        )
        if angle < 0.0:
            raise ValueError(f"Phase-2 cohort run {index} angle MAE is negative")
        angles.append(angle)
        expected_checkpoint = best_path.resolve()
        if checkpoint_path == expected_checkpoint:
            if run["best_checkpoint_sha256"] != checkpoint_digest:
                raise ValueError("requested Phase-2 checkpoint digest drifted")
            authorized_run = run

    if authorized_run is None:
        raise ValueError(
            "requested checkpoint is not an authorized Phase-2 cohort best.pt"
        )

    aggregate = report.get("best_validation_angle_mae_degrees")
    if not isinstance(aggregate, Mapping) or set(aggregate) != {
        "mean",
        "sample_standard_deviation",
        "minimum",
        "maximum",
    }:
        raise ValueError("Phase-2 cohort aggregate schema drifted")
    angle_array = np.asarray(angles, dtype=np.float64)
    expected_aggregate = {
        "mean": float(np.mean(angle_array)),
        "sample_standard_deviation": float(np.std(angle_array, ddof=1)),
        "minimum": float(np.min(angle_array)),
        "maximum": float(np.max(angle_array)),
    }
    for field, expected in expected_aggregate.items():
        actual = _require_finite(
            aggregate.get(field),
            f"Phase-2 cohort aggregate {field}",
        )
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"Phase-2 cohort aggregate {field} drifted")

    return {
        "protocol": COHORT_PROTOCOL,
        "training_protocol": PHASE2_PROTOCOL,
        "authorization_scope": COHORT_AUTHORIZATION_SCOPE,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "canonical_cohort_payload_sha256": stored_canonical,
        "cohort_verifier_source_sha256": source_digest,
        "evaluation_plan_source_sha256": plan_source_digest,
        "evaluation_execution_source_identity": dict(
            execution_source_identity
        ),
        "parent_seed": int(authorized_run["parent_seed"]),
        "phase_seed": int(authorized_run["phase_seed"]),
        "run_dir": str(authorized_run["run_dir"]),
        "verification_path": str(authorized_run["verification_path"]),
        "verification_sha256": str(
            authorized_run["verification_sha256"]
        ),
        "summary_sha256": str(authorized_run["summary_sha256"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
    }


def write_json_no_clobber(value: Mapping[str, Any], output: Path) -> Path:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=f".{uuid.uuid4().hex}.tmp",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite cohort report: {output}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return output


def verify_cohort(
    *,
    run_root: Path,
    parent_root: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    determinism_report: Path,
    device: str,
    verifier: Callable[..., dict[str, Any]] = verify_phase2_run,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    parent_root = Path(parent_root).resolve()
    entries: list[tuple[int, Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for seed in FORMAL_PARENT_SEEDS:
        run_dir = run_root / f"seed_{seed}"
        parent_dir = parent_root / f"seed_{seed}"
        report_path = run_dir / INDIVIDUAL_REPORT_NAME
        stored = _strict_json(report_path)
        fresh = verifier(
            run_dir,
            parent_run=parent_dir,
            manifest=Path(manifest).resolve(),
            vdn_source=Path(vdn_source).resolve(),
            content_inventory=Path(content_inventory).resolve(),
            determinism_report=Path(determinism_report).resolve(),
            device=device,
        )
        entries.append((seed, report_path, stored, fresh))
    return build_cohort_report(
        entries,
        run_root=run_root,
        parent_root=parent_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--vdn-source", type=Path, default=DEFAULT_VDN_SOURCE)
    parser.add_argument(
        "--content-inventory",
        type=Path,
        default=DEFAULT_CONTENT_INVENTORY,
    )
    parser.add_argument(
        "--determinism-report",
        type=Path,
        default=DEFAULT_DETERMINISM_REPORT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort report: {output}")
    report = verify_cohort(
        run_root=args.run_root,
        parent_root=args.parent_root,
        manifest=args.manifest,
        vdn_source=args.vdn_source,
        content_inventory=args.content_inventory,
        determinism_report=args.determinism_report,
        device=args.device,
    )
    write_json_no_clobber(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
