"""Build the mixed PEPD authority: v1 seeds 20/22 plus v2 seed 21."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

from experiments.pepd_convergence_extension_v2_protocol import (
    EXTENSION_SEED,
    PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
    PEPD_EXTENSION_PROTOCOL,
    PEPD_EXTENSION_VERIFICATION_PROTOCOL,
    V1_RUN_PINS,
    audit_v1_artifacts,
    authoritative_cohort_path,
    extension_output_dir,
    v1_run_dir,
)
from experiments.pepd_convergence_protocol import (
    FORMAL_SEEDS,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_RUN_VERIFICATION_PROTOCOL,
    PROJECT_ROOT,
    sha256_file,
)
from experiments.strict_json import strict_json_load, strict_json_source_sha256
from experiments.vdn_baseline import sha256_source_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=authoritative_cohort_path())
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _load_verified_run(seed: int) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if seed == EXTENSION_SEED:
        run_dir = extension_output_dir()
        summary_protocol = PEPD_EXTENSION_PROTOCOL
        verification_protocol = PEPD_EXTENSION_VERIFICATION_PROTOCOL
    else:
        run_dir = v1_run_dir(seed)
        summary_protocol = PEPD_CONTINUATION_PROTOCOL
        verification_protocol = PEPD_RUN_VERIFICATION_PROTOCOL
    summary_path = run_dir / "summary.json"
    verification_path = run_dir / "verification.json"
    summary = strict_json_load(summary_path)
    verification = strict_json_load(verification_path)
    if (
        summary.get("protocol") != summary_protocol
        or summary.get("status") != "complete"
        or summary.get("seed") != seed
        or verification.get("protocol") != verification_protocol
        or verification.get("verified") is not True
        or verification.get("seed") != seed
        or verification.get("summary_sha256") != sha256_file(summary_path)
        or verification.get("best_checkpoint_sha256")
        != summary.get("best_checkpoint_sha256")
        or verification.get("best_epoch") != summary.get("best_epoch")
        or summary.get("public_or_field_evaluation_authorized") is not False
        or verification.get("public_or_field_evaluation_authorized") is not False
    ):
        raise ValueError(f"seed {seed}: authoritative source audit failed")
    checkpoint_path = run_dir / "best.pt"
    if sha256_file(checkpoint_path) != verification["best_checkpoint_sha256"]:
        raise ValueError(f"seed {seed}: best checkpoint changed after verification")
    return run_dir, summary, verification


def build_cohort() -> dict[str, Any]:
    v1 = audit_v1_artifacts()
    rows: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        run_dir, summary, verification = _load_verified_run(seed)
        summary_path = run_dir / "summary.json"
        verification_path = run_dir / "verification.json"
        environment = dict(summary.get("environment") or {})
        environment.pop("pythonhashseed", None)
        environments.append(environment)
        if seed == EXTENSION_SEED:
            legacy_parent_best = float(
                v1["runs"][seed]["summary"]["parent"][
                    "best_validation_angle_mae_degrees"
                ]
            )
            legacy_parent_epoch = int(
                v1["runs"][seed]["summary"]["parent"]["best_epoch"]
            )
            source_phase = "bounded_extension_v2"
            extension_applied = True
        else:
            legacy_parent_best = float(
                summary["parent"]["best_validation_angle_mae_degrees"]
            )
            legacy_parent_epoch = int(summary["parent"]["best_epoch"])
            source_phase = "convergence_v1"
            extension_applied = False
        final_best = float(summary["best_validation_angle_mae_degrees"])
        rows.append(
            {
                "seed": seed,
                "source_phase": source_phase,
                "source_run_dir": str(run_dir),
                "authoritative_run_protocol": summary["protocol"],
                "verification_protocol": verification["protocol"],
                "verified": True,
                "converged": bool(verification["converged"]),
                "extension_applied": extension_applied,
                "legacy_parent_best_epoch": legacy_parent_epoch,
                "final_best_epoch": int(summary["best_epoch"]),
                "legacy_parent_best_validation_angle_mae_degrees": (
                    legacy_parent_best
                ),
                "final_best_validation_angle_mae_degrees": final_best,
                "legacy_parent_minus_final_best_angle_mae_degrees": (
                    legacy_parent_best - final_best
                ),
                "selected_checkpoint_origin": verification.get(
                    "selected_checkpoint_origin"
                ),
                "checkpoint_changed_from_frozen_parent": bool(
                    verification["checkpoint_changed_from_frozen_parent"]
                ),
                "best_checkpoint_sha256": verification[
                    "best_checkpoint_sha256"
                ],
                "summary_sha256": verification["summary_sha256"],
                "verification_sha256": sha256_file(verification_path),
                "convergence_audit": verification["convergence_audit"],
            }
        )
    if not environments[0] or any(
        environment != environments[0] for environment in environments[1:]
    ):
        raise ValueError("mixed PEPD runs used different runtime environments")
    all_converged = all(row["converged"] for row in rows)
    values = [
        row["final_best_validation_angle_mae_degrees"] for row in rows
    ]
    improvements = [
        row["legacy_parent_minus_final_best_angle_mae_degrees"] for row in rows
    ]
    return {
        "schema_version": 2,
        "protocol": PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
        "status": "converged" if all_converged else "not_converged",
        "scope": "SyncG official train grouped validation only",
        "role": (
            "mixed authoritative PEPD convergence cohort; v1 seeds 20260720/"
            "20260722 and one-shot bounded-v2 seed 20260721"
        ),
        "seeds": list(FORMAL_SEEDS),
        "runs": rows,
        "all_runs_verified": True,
        "all_runs_converged": all_converged,
        "mixed_authority": {
            "20260720": "convergence_v1",
            "20260721": "bounded_extension_v2",
            "20260722": "convergence_v1",
        },
        "v1_cohort": str(v1["cohort_path"]),
        "v1_cohort_sha256": sha256_file(v1["cohort_path"]),
        "v1_artifacts_preserved": {
            str(seed): {
                "summary_sha256": V1_RUN_PINS[seed].summary_sha256,
                "verification_sha256": V1_RUN_PINS[seed].verification_sha256,
                "best_checkpoint_sha256": V1_RUN_PINS[seed].best_sha256,
                "last_checkpoint_sha256": V1_RUN_PINS[seed].last_sha256,
            }
            for seed in FORMAL_SEEDS
        },
        "best_validation_angle_mae_degrees_across_seeds": _mean_std(values),
        "legacy_parent_minus_final_improvement_degrees_across_seeds": (
            _mean_std(improvements)
        ),
        "downstream_authorization": {
            "mixed_authoritative_oof_rebuild_required": all_converged,
            "fadr_rebuild_authorized": all_converged,
            "udsf_rebuild_authorized": False,
            "reason": (
                "UDSF remains gated on a separately verified joint FADR cohort"
            ),
        },
        "grouped_validation_controlled_robustness_authorized": all_converged,
        "grouped_validation_controlled_perspective_authorized": all_converged,
        "further_epoch_extension_authorized": False,
        "public_test_field_evaluation_authorized": False,
        "failure_action": (
            "If seed 20260721 still fails at epoch 80, stop permanently; do "
            "not add epochs and do not inspect restricted evaluation data."
        ),
        "runtime_environment_without_seed": environments[0],
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "extension_protocol": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "pepd_convergence_extension_v2_protocol.py"
            ),
            "extension_trainer": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "train_pepd_convergence_extension_v2.py"
            ),
            "extension_verifier": sha256_source_file(
                PROJECT_ROOT
                / "experiments"
                / "verify_pepd_convergence_extension_v2.py"
            ),
            "summarizer": sha256_source_file(Path(__file__).resolve()),
        },
    }


def _write_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"{path} exists with different cohort content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    expected = authoritative_cohort_path()
    if output != expected:
        raise ValueError(f"formal authoritative cohort output must be {expected}")
    cohort = build_cohort()
    _write_or_validate(output, cohort)
    print(
        json.dumps(
            cohort,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
