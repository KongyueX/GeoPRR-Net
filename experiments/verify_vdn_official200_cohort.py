"""Fail-closed three-seed cohort gate for completed official-200 VDN runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.preflight_vdn_official200 import (
    DEFAULT_OUTPUT as DEFAULT_PREFLIGHT,
    DEFAULT_RUN_ROOT,
    _strict_json,
    validate_preflight_report,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_file, sha256_source_file
from experiments.vdn_official200_protocol import (
    OFFICIAL200_COHORT_PROTOCOL,
    OFFICIAL200_EPOCHS,
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_SCHEMA_VERSION,
    OFFICIAL200_STOPPING_POLICY,
    OFFICIAL200_VERIFICATION_PROTOCOL,
    assert_syncg_train_manifest_path,
    assert_train_only_path,
    canonical_json_sha256,
)
from experiments.verify_vdn_official200 import (
    VERIFICATION_SCHEMA_KEYS,
    write_json_no_clobber,
)


INDIVIDUAL_REPORT_NAME = "verification_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "protocols"
    / "vdn_official200_three_seed_cohort_v1.json"
)
COHORT_SCHEMA_KEYS = frozenset(
    {
        "protocol",
        "schema_version",
        "status",
        "verified",
        "three_seed_cohort_complete",
        "three_seed_equal_epoch_budget",
        "vdn_supporting_test_evaluation_authorized",
        "field_confirmatory_evaluation_authorized",
        "additional_training_authorized",
        "phase4_authorized",
        "authorization_scope",
        "formal_seeds",
        "epochs_per_seed",
        "runs",
        "shared_identity",
        "best_validation_angle_mae_degrees",
        "tail_diagnostic",
        "stopping_policy",
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
        "cohort_verifier_source_sha256",
        "individual_verifier_source_sha256",
        "canonical_cohort_payload_sha256",
    }
)


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_individual(
    report: Mapping[str, Any],
    *,
    seed: int,
    run_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    if set(report) != VERIFICATION_SCHEMA_KEYS:
        raise ValueError(f"official-200 seed {seed} report schema drifted")
    canonical = report.get("canonical_verification_payload_sha256")
    payload = dict(report)
    payload.pop("canonical_verification_payload_sha256", None)
    if canonical != canonical_json_sha256(payload):
        raise ValueError(
            f"official-200 seed {seed} report canonical digest drifted"
        )
    if (
        report.get("protocol") != OFFICIAL200_VERIFICATION_PROTOCOL
        or int(report.get("schema_version", -1))
        != OFFICIAL200_SCHEMA_VERSION
        or report.get("verified") is not True
        or report.get("training_artifacts_verified") is not True
        or report.get("eligible_for_three_seed_cohort") is not True
        or report.get("supporting_test_evaluation_authorized") is not False
        or report.get("field_confirmatory_evaluation_authorized") is not False
        or report.get("run_dir") != str(run_dir)
        or int(report.get("seed", -1)) != seed
        or int(report.get("epochs", -1)) != OFFICIAL200_EPOCHS
        or report.get("official_stopping_boundary_reached") is not True
        or report.get("additional_training_authorized") is not False
        or report.get("phase4_authorized") is not False
        or report.get("stopping_policy") != OFFICIAL200_STOPPING_POLICY
    ):
        raise ValueError(
            f"official-200 seed {seed} is not cohort eligible"
        )
    for field in (
        "test_data_opened_or_read",
        "public_data_opened_or_read",
        "field_data_opened_or_read",
        "sealed_data_opened_or_read",
        "confirmatory_data_opened_or_read",
    ):
        if report.get(field) is not False:
            raise ValueError(
                f"official-200 seed {seed} provenance flag drifted: {field}"
            )
    for field, filename in (
        ("best_checkpoint_sha256", "best.pt"),
        ("last_checkpoint_sha256", "last.pt"),
        ("summary_sha256", "summary.json"),
    ):
        digest = _validate_sha256(
            report.get(field),
            label=f"official-200 seed {seed} {field}",
        )
        artifact = run_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(
                f"official-200 seed {seed} {filename} changed "
                "after verification"
            )
    _validate_sha256(
        report.get("verifier_source_sha256"),
        label=f"official-200 seed {seed} verifier source",
    )
    attempts = int(report.get("optimizer_attempts", -1))
    successful = int(report.get("optimizer_steps", -1))
    skipped = int(report.get("skipped_optimizer_steps", -1))
    if (
        attempts <= 0
        or successful <= 0
        or skipped < 0
        or successful + skipped != attempts
        or skipped
        > int(report.get("full_run_max_skipped_optimizer_steps", -1))
        or not math.isclose(
            float(report.get("skipped_optimizer_step_rate", math.nan)),
            skipped / attempts,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            f"official-200 seed {seed} optimizer accounting drifted"
        )
    diagnostic = report.get("tail_diagnostic")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("diagnostic_only") is not True
        or diagnostic.get("authorization_gate") is not False
        or diagnostic.get("additional_training_authorized") is not False
        or diagnostic.get("phase4_authorized") is not False
    ):
        raise ValueError(
            f"official-200 seed {seed} tail diagnostic changed policy"
        )
    angle = float(report["best_validation_angle_mae_degrees"])
    if not math.isfinite(angle) or angle < 0.0:
        raise ValueError(
            f"official-200 seed {seed} best angle is invalid"
        )
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "verification_path": str(report_path),
        "verification_sha256": sha256_file(report_path),
        "verification_canonical_sha256": canonical,
        "best_epoch": int(report["best_epoch"]),
        "best_validation_angle_mae_degrees": angle,
        "optimizer_attempts": attempts,
        "optimizer_steps": successful,
        "skipped_optimizer_steps": skipped,
        "skipped_optimizer_step_rate": skipped / attempts,
        "tail_plateau_observed": bool(
            diagnostic.get("plateau_observed")
        ),
        "manuscript_limitation_required": bool(
            diagnostic.get("manuscript_limitation_required")
        ),
        "best_checkpoint_sha256": report["best_checkpoint_sha256"],
        "last_checkpoint_sha256": report["last_checkpoint_sha256"],
        "summary_sha256": report["summary_sha256"],
    }


def build_cohort_report(
    reports: Sequence[tuple[int, Path, Mapping[str, Any]]],
    *,
    run_root: Path,
    preflight_report: Mapping[str, Any],
    preflight_binding: Mapping[str, Any],
) -> dict[str, Any]:
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    if [int(item[0]) for item in reports] != list(
        OFFICIAL200_FORMAL_SEEDS
    ):
        raise ValueError(
            "official-200 cohort seeds are incomplete or out of order"
        )
    expected_verifier_hash = sha256_source_file(
        PROJECT_DIR / "experiments" / "verify_vdn_official200.py"
    )
    records = []
    shared: dict[str, Any] | None = None
    for seed, report_path, report in reports:
        seed = int(seed)
        run_dir = run_root / f"seed_{seed}"
        report_path = assert_train_only_path(
            report_path,
            label=f"official-200 seed {seed} verification report",
        )
        if report_path != run_dir / INDIVIDUAL_REPORT_NAME:
            raise ValueError(
                f"official-200 seed {seed} verification path drifted"
            )
        if report.get("verifier_source_sha256") != expected_verifier_hash:
            raise ValueError(
                f"official-200 seed {seed} verifier source is stale"
            )
        records.append(
            _validate_individual(
                report,
                seed=seed,
                run_dir=run_dir,
                report_path=report_path,
            )
        )
        runtime = dict(report["runtime_environment"])
        if runtime.get("pythonhashseed") != str(seed):
            raise ValueError(
                f"official-200 seed {seed} PYTHONHASHSEED drifted"
            )
        runtime.pop("pythonhashseed")
        shared_value = {
            "preflight": report["preflight"],
            "content_inventory": report["content_inventory"],
            "determinism_policy": report["determinism_policy"],
            "determinism_authorization": report[
                "determinism_authorization"
            ],
            "runtime_environment_except_pythonhashseed": runtime,
            "source_hash_protocol": report["source_hash_protocol"],
            "training_source_sha256": report["training_source_sha256"],
        }
        if shared is None:
            shared = shared_value
        elif shared != shared_value:
            raise ValueError(
                f"official-200 seed {seed} shared identity drifted"
            )
    assert shared is not None
    if shared["preflight"] != dict(preflight_binding):
        raise ValueError("official-200 cohort preflight binding drifted")
    if preflight_report.get("formal_seeds") != list(
        OFFICIAL200_FORMAL_SEEDS
    ):
        raise ValueError("official-200 preflight cohort seeds drifted")

    angles = np.asarray(
        [record["best_validation_angle_mae_degrees"] for record in records],
        dtype=np.float64,
    )
    limitation_seeds = [
        record["seed"]
        for record in records
        if record["manuscript_limitation_required"]
    ]
    report: dict[str, Any] = {
        "protocol": OFFICIAL200_COHORT_PROTOCOL,
        "schema_version": OFFICIAL200_SCHEMA_VERSION,
        "status": "passed",
        "verified": True,
        "three_seed_cohort_complete": True,
        "three_seed_equal_epoch_budget": True,
        "vdn_supporting_test_evaluation_authorized": True,
        "field_confirmatory_evaluation_authorized": False,
        "additional_training_authorized": False,
        "phase4_authorized": False,
        "authorization_scope": (
            "frozen VDN supporting test evaluation only; field/sealed/"
            "confirmatory evaluation remains unauthorized"
        ),
        "formal_seeds": list(OFFICIAL200_FORMAL_SEEDS),
        "epochs_per_seed": {
            str(seed): OFFICIAL200_EPOCHS
            for seed in OFFICIAL200_FORMAL_SEEDS
        },
        "runs": records,
        "shared_identity": shared,
        "best_validation_angle_mae_degrees": {
            "mean": float(np.mean(angles)),
            "sample_standard_deviation": float(np.std(angles, ddof=1)),
            "minimum": float(np.min(angles)),
            "maximum": float(np.max(angles)),
        },
        "tail_diagnostic": {
            "diagnostic_only": True,
            "authorization_gate": False,
            "plateau_observed_for_all_seeds": not limitation_seeds,
            "manuscript_limitation_required_seeds": limitation_seeds,
            "additional_training_authorized": False,
            "phase4_authorized": False,
        },
        "stopping_policy": OFFICIAL200_STOPPING_POLICY,
        "test_data_opened_or_read": False,
        "public_data_opened_or_read": False,
        "field_data_opened_or_read": False,
        "sealed_data_opened_or_read": False,
        "confirmatory_data_opened_or_read": False,
        "cohort_verifier_source_sha256": sha256_source_file(
            Path(__file__).resolve()
        ),
        "individual_verifier_source_sha256": expected_verifier_hash,
    }
    report["canonical_cohort_payload_sha256"] = canonical_json_sha256(
        report
    )
    if set(report) != COHORT_SCHEMA_KEYS:
        raise RuntimeError("official-200 cohort schema drifted internally")
    return report


def verify_cohort(
    *,
    run_root: Path,
    manifest: Path,
    vdn_source: Path,
    content_inventory: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    run_root = assert_train_only_path(
        run_root,
        label="official-200 run root",
    )
    assert_syncg_train_manifest_path(manifest)
    assert_train_only_path(
        content_inventory,
        label="official-200 content inventory",
    )
    assert_train_only_path(
        preflight_path,
        label="official-200 preflight report",
    )
    preflight, binding = validate_preflight_report(
        preflight_path,
        manifest=manifest,
        vdn_source=vdn_source,
        content_inventory=content_inventory,
        run_root=run_root,
        require_output_absent=False,
    )
    reports = []
    for seed in OFFICIAL200_FORMAL_SEEDS:
        report_path = (
            run_root / f"seed_{seed}" / INDIVIDUAL_REPORT_NAME
        )
        reports.append((seed, report_path, _strict_json(report_path)))
    return build_cohort_report(
        reports,
        run_root=run_root,
        preflight_report=preflight,
        preflight_binding=binding,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifests/syncg_train.jsonl"),
    )
    parser.add_argument(
        "--vdn-source",
        type=Path,
        default=Path("artifacts/vendor/VectorDetectionNetwork"),
    )
    parser.add_argument(
        "--content-inventory",
        type=Path,
        default=Path(
            "artifacts/protocols/"
            "vdn_phase2_syncg_train_content_inventory_v1.json"
        ),
    )
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = assert_train_only_path(
        args.output,
        label="official-200 cohort output",
    )
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite official-200 cohort: {output}"
        )
    report = verify_cohort(
        run_root=args.run_root,
        manifest=args.manifest,
        vdn_source=args.vdn_source,
        content_inventory=args.content_inventory,
        preflight_path=args.preflight,
    )
    write_json_no_clobber(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
