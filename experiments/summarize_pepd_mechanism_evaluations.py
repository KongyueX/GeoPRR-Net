"""Summarize paired PEPD mechanism effects on grouped-validation perspective."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from experiments.strict_json import (
    strict_json_load,
    strict_jsonl_load,
    strict_json_source_sha256,
)

from experiments.pepd_convergence_protocol import (
    DECODER_VIEWS,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    GROUPED_VAL_BOOTSTRAP_ITERATIONS,
    GROUPED_VAL_BOOTSTRAP_SEED,
    GROUPED_VAL_CONDITIONS,
    PEPD_DECODER_VIEW_PROTOCOL,
    PEPD_GROUPED_VAL_EVALUATION_PROTOCOL,
    PEPD_MECHANISM_COHORT_PROTOCOL,
    PRIMARY_MECHANISM_ARMS,
    PROJECT_ROOT,
    formal_output_dir,
    mechanism_output_dir,
    sha256_file,
)
from experiments.pepd_convergence_extension_v2_protocol import (
    EXTENSION_SEED,
    PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
    authoritative_cohort_path,
    extension_output_dir,
)
from experiments.vdn_baseline import sha256_source_file


MECHANISM_COHORT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_mechanism_phase2"
    / "cohort.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "artifacts"
    / "runs"
    / "pepd_mechanism_phase2"
    / "grouped_validation_mechanism_report.json"
)
AUTHORITATIVE_COHORT = authoritative_cohort_path()
EVALUATOR_SOURCE = (
    PROJECT_ROOT / "experiments" / "evaluate_pepd_grouped_validation.py"
)
PROTOCOL_SOURCE = PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
DEGRADATION_SOURCE = PROJECT_ROOT / "experiments" / "robustness_degradations.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return strict_json_load(path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return strict_jsonl_load(path)


def _run_dir(arm: str, seed: int) -> Path:
    if arm != "full":
        return mechanism_output_dir(arm, seed)
    return extension_output_dir() if seed == EXTENSION_SEED else formal_output_dir(seed)


def _cohort_run(
    cohort: Mapping[str, Any],
    *,
    seed: int,
    arm: str | None = None,
) -> Mapping[str, Any]:
    runs = cohort.get("runs")
    if not isinstance(runs, list):
        raise ValueError("cohort run membership is missing")
    matches = [
        row
        for row in runs
        if isinstance(row, Mapping)
        and int(row.get("seed", -1)) == seed
        and (arm is None or row.get("arm") == arm)
    ]
    if len(matches) != 1:
        raise ValueError(f"cohort has no unique run binding for {arm}:{seed}")
    return matches[0]


def _audit_cohorts() -> tuple[dict[str, Any], dict[str, Any], str]:
    mechanism = _load(MECHANISM_COHORT)
    authoritative = _load(AUTHORITATIVE_COHORT)
    if (
        mechanism.get("protocol") != PEPD_MECHANISM_COHORT_PROTOCOL
        or mechanism.get("status") != "complete"
        or mechanism.get("all_runs_verified_and_converged") is not True
        or mechanism.get("controlled_grouped_validation_authorized") is not True
        or mechanism.get("public_test_field_evaluation_authorized") is not False
        or tuple(mechanism.get("seeds") or ()) != FORMAL_SEEDS
        or tuple(mechanism.get("arms") or ()) != PRIMARY_MECHANISM_ARMS
    ):
        raise ValueError("mechanism cohort is not complete/converged")
    if (
        authoritative.get("protocol") != PEPD_AUTHORITATIVE_COHORT_PROTOCOL
        or authoritative.get("status") != "converged"
        or authoritative.get("all_runs_verified") is not True
        or authoritative.get("all_runs_converged") is not True
        or authoritative.get("public_test_field_evaluation_authorized") is not False
        or tuple(authoritative.get("seeds") or ()) != FORMAL_SEEDS
        or authoritative.get("mixed_authority")
        != {
            "20260720": "convergence_v1",
            "20260721": "bounded_extension_v2",
            "20260722": "convergence_v1",
        }
    ):
        raise ValueError("mixed authoritative PEPD v2 cohort gate failed")
    horizon_note = mechanism.get("horizon_note")
    if not isinstance(horizon_note, str) or "bounded extension" not in horizon_note:
        raise ValueError("mechanism cohort horizon note is missing")
    source_identity = mechanism.get("source_identity")
    if (
        not isinstance(source_identity, Mapping)
        or source_identity.get("strict_json") != strict_json_source_sha256()
        or source_identity.get("protocol") != sha256_source_file(PROTOCOL_SOURCE)
        or source_identity.get("authoritative_cohort")
        != sha256_file(AUTHORITATIVE_COHORT)
    ):
        raise ValueError("mechanism cohort source identity drifted")
    if (authoritative.get("source_identity") or {}).get(
        "strict_json"
    ) != strict_json_source_sha256():
        raise ValueError("authoritative cohort strict-JSON identity drifted")
    for seed in FORMAL_SEEDS:
        authoritative_row = _cohort_run(authoritative, seed=seed)
        mechanism_row = _cohort_run(mechanism, seed=seed, arm="full")
        for name in (
            "best_checkpoint_sha256",
            "summary_sha256",
            "verification_sha256",
        ):
            if mechanism_row.get(name) != authoritative_row.get(name):
                raise ValueError(
                    f"full seed {seed} mechanism/authoritative {name} mismatch"
                )
    return mechanism, authoritative, horizon_note


def _project_decoder_view(
    rows: Mapping[tuple[int, str], Mapping[str, Any]],
    view: str,
) -> dict[tuple[int, str], Mapping[str, Any]]:
    if view not in DECODER_VIEWS:
        raise ValueError(f"unsupported decoder view: {view}")
    projected: dict[tuple[int, str], Mapping[str, Any]] = {}
    for key, row in rows.items():
        views = row.get("decoder_views")
        if not isinstance(views, Mapping) or set(views) != set(DECODER_VIEWS):
            raise ValueError("decoder-view row membership drifted")
        values = views.get(view)
        if not isinstance(values, Mapping):
            raise ValueError(f"decoder-view row has no {view} output")
        item = dict(row)
        item["valid"] = values.get("valid")
        item["angle_error_degrees"] = values.get("angle_error_degrees")
        item["signed_angle_error_degrees"] = values.get(
            "signed_angle_error_degrees"
        )
        projected[key] = item
    return projected


def _paired_effect(
    left: Mapping[tuple[int, str], Mapping[str, Any]],
    right: Mapping[tuple[int, str], Mapping[str, Any]],
    *,
    interpretation: str,
    seed: int,
    metric: str = "angle_error_degrees",
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("mechanism contrast sample identities differ")
    by_group: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[int, list[float]] = defaultdict(list)
    for key in sorted(left):
        left_row = left[key]
        right_row = right[key]
        if str(left_row["group_id"]) != str(right_row["group_id"]):
            raise ValueError("mechanism contrast group identity differs")
        if metric == "angle_error_degrees":
            valid_key = "valid"
            failure_penalty = 180.0
            units = "degrees"
        elif metric == "equivariance_direction_residual_degrees":
            valid_key = "equivariance_valid"
            failure_penalty = 180.0
            units = "degrees"
        elif metric == "equivariance_pivot_residual_fraction":
            valid_key = "equivariance_valid"
            failure_penalty = math.sqrt(2.0)
            units = "image_fraction"
        else:
            raise ValueError(f"unsupported paired mechanism metric: {metric}")
        # Positive means the right-hand mechanism lowers error.
        left_value = (
            float(left_row[metric])
            if left_row.get(valid_key) is True
            else failure_penalty
        )
        right_value = (
            float(right_row[metric])
            if right_row.get(valid_key) is True
            else failure_penalty
        )
        effect = left_value - right_value
        by_group[str(left_row["group_id"])].append(effect)
        per_seed[int(key[0])].append(effect)
    group_effects = np.asarray(
        [float(np.mean(values)) for _, values in sorted(by_group.items())],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(GROUPED_VAL_BOOTSTRAP_ITERATIONS, dtype=np.float64)
    for index in range(GROUPED_VAL_BOOTSTRAP_ITERATIONS):
        selected = generator.integers(
            0,
            len(group_effects),
            size=len(group_effects),
        )
        bootstrap[index] = float(np.mean(group_effects[selected]))
    interval = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    return {
        "interpretation": interpretation,
        "metric": metric,
        "units": units,
        "invalid_pair_penalty": failure_penalty,
        "positive_means_right_hand_arm_has_lower_error": True,
        "paired_samples": len(left),
        "unique_groups_across_seed_splits": len(group_effects),
        "group_macro_effect": float(np.mean(group_effects)),
        "group_bootstrap_95ci": interval,
        "group_bootstrap_iterations": GROUPED_VAL_BOOTSTRAP_ITERATIONS,
        "group_bootstrap_seed": seed,
        "per_seed_sample_mean_effect": {
            str(run_seed): float(np.mean(values))
            for run_seed, values in sorted(per_seed.items())
        },
        "mechanism_supported": interval[0] > 0.0,
    }


def build_report() -> dict[str, Any]:
    cohort, authoritative_cohort, horizon_note = _audit_cohorts()
    mechanism_cohort_sha256 = sha256_file(MECHANISM_COHORT)
    authoritative_cohort_sha256 = sha256_file(AUTHORITATIVE_COHORT)
    evaluator_source_sha256 = sha256_source_file(EVALUATOR_SOURCE)
    protocol_source_sha256 = sha256_source_file(PROTOCOL_SOURCE)
    degradation_source_sha256 = sha256_source_file(DEGRADATION_SOURCE)
    summaries: dict[str, dict[str, dict[str, Any]]] = {
        condition: {arm: {} for arm in PRIMARY_MECHANISM_ARMS}
        for condition in GROUPED_VAL_CONDITIONS
    }
    indexed: dict[
        str,
        dict[str, dict[tuple[int, str], Mapping[str, Any]]],
    ] = {
        condition: {arm: {} for arm in PRIMARY_MECHANISM_ARMS}
        for condition in GROUPED_VAL_CONDITIONS
    }
    decoder_seed_metrics: dict[str, dict[str, list[float]]] = {
        condition: {view: [] for view in DECODER_VIEWS}
        for condition in GROUPED_VAL_CONDITIONS
    }
    artifact_hashes: dict[str, str] = {}
    for condition in GROUPED_VAL_CONDITIONS:
        for arm in PRIMARY_MECHANISM_ARMS:
            arm_metrics: list[float] = []
            equivariance_direction: list[float] = []
            equivariance_pivot: list[float] = []
            equivariance_validity: list[float] = []
            for seed in FORMAL_SEEDS:
                path = _run_dir(arm, seed) / "grouped_validation" / f"{condition}.jsonl"
                summary_path = path.with_suffix(".summary.json")
                summary = _load(summary_path)
                if summary.get("protocol") != PEPD_GROUPED_VAL_EVALUATION_PROTOCOL:
                    raise ValueError(f"{arm}:{seed}:{condition}: protocol mismatch")
                if (
                    summary.get("arm") != arm
                    or int(summary.get("seed", -1)) != seed
                    or summary.get("condition") != condition
                ):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: evaluation identity mismatch"
                    )
                if summary.get("output_sha256") != sha256_file(path):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: prediction hash mismatch"
                    )
                if summary.get("public_test_field_evaluation") is not False:
                    raise ValueError("mechanism evaluation scope drifted")
                if summary.get("horizon_note") != horizon_note:
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: horizon note drifted"
                    )
                if (
                    summary.get("mechanism_cohort_sha256")
                    != mechanism_cohort_sha256
                ):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: mechanism cohort binding drifted"
                    )
                source_identity = summary.get("source_identity")
                if (
                    not isinstance(source_identity, Mapping)
                    or source_identity.get("strict_json")
                    != strict_json_source_sha256()
                    or source_identity.get("evaluator")
                    != evaluator_source_sha256
                    or source_identity.get("protocol")
                    != protocol_source_sha256
                    or source_identity.get("model")
                    != FORMAL_MODEL_SOURCE_SHA256
                    or source_identity.get("degradation")
                    != degradation_source_sha256
                ):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: evaluator source identity drifted"
                    )
                if arm == "full":
                    cohort_row = _cohort_run(authoritative_cohort, seed=seed)
                    expected_cohort_sha256 = authoritative_cohort_sha256
                    expected_authority = (
                        "bounded_extension_v2"
                        if seed == EXTENSION_SEED
                        else "convergence_v1"
                    )
                else:
                    cohort_row = _cohort_run(cohort, seed=seed, arm=arm)
                    expected_cohort_sha256 = mechanism_cohort_sha256
                    expected_authority = "mechanism_phase2"
                if (
                    summary.get("cohort_sha256") != expected_cohort_sha256
                    or summary.get("run_authority") != expected_authority
                    or summary.get("checkpoint_sha256")
                    != cohort_row.get("best_checkpoint_sha256")
                    or summary.get("run_verification_sha256")
                    != cohort_row.get("verification_sha256")
                ):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: training authority binding drifted"
                    )
                rows = _jsonl(path)
                for row in rows:
                    key = (seed, str(row["sample_id"]))
                    if key in indexed[condition][arm]:
                        raise ValueError("duplicate mechanism evaluation identity")
                    indexed[condition][arm][key] = row
                if arm == "full":
                    decoder = summary.get("decoder_view_ablation")
                    if not isinstance(decoder, Mapping):
                        raise ValueError(
                            f"{arm}:{seed}:{condition}: decoder evidence missing"
                        )
                    if (
                        decoder.get("protocol") != PEPD_DECODER_VIEW_PROTOCOL
                        or tuple(decoder.get("views") or ()) != DECODER_VIEWS
                        or decoder.get("primary_view") != "fused"
                        or decoder.get("training_or_finetuning") is not False
                        or decoder.get("checkpoint_selection") is not False
                        or decoder.get("same_model_forward_and_logits") is not True
                    ):
                        raise ValueError(
                            f"{arm}:{seed}:{condition}: decoder protocol drifted"
                        )
                    per_view = decoder.get("per_view")
                    if not isinstance(per_view, Mapping):
                        raise ValueError(
                            f"{arm}:{seed}:{condition}: decoder metrics missing"
                        )
                    for view in DECODER_VIEWS:
                        grouped_view = (
                            (per_view.get(view) or {}).get("grouped_metrics")
                            if isinstance(per_view.get(view), Mapping)
                            else None
                        )
                        if not isinstance(grouped_view, Mapping):
                            raise ValueError(
                                f"{arm}:{seed}:{condition}:{view}: "
                                "grouped decoder metrics missing"
                            )
                        decoder_seed_metrics[condition][view].append(
                            float(grouped_view["macro_angle_mae_degrees"])
                        )
                arm_metrics.append(float(summary["metrics"]["angle_mae_degrees"]))
                residual = summary.get("paired_equivariance_residual") or {}
                residual_metrics = residual.get("metrics") or {}
                residual_bootstrap = residual.get("group_bootstrap") or {}
                if (
                    residual.get("claim_boundary")
                    != (
                        "projective-equivariance-regularized/trained; residual "
                        "is empirical and does not establish an exact "
                        "equivariant architecture"
                    )
                ):
                    raise ValueError(
                        f"{arm}:{seed}:{condition}: unsafe equivariance claim"
                    )
                equivariance_direction.append(
                    float(
                        residual_bootstrap[
                            "macro_direction_residual_degrees_failure_penalized"
                        ]
                    )
                )
                equivariance_pivot.append(
                    float(
                        residual_bootstrap[
                            "macro_pivot_residual_fraction_failure_penalized"
                        ]
                    )
                )
                equivariance_validity.append(
                    float(residual_metrics["valid_pair_fraction"])
                )
                artifact_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
                artifact_hashes[
                    str(summary_path.relative_to(PROJECT_ROOT))
                ] = sha256_file(summary_path)
            summaries[condition][arm] = {
                "per_seed_sample_angle_mae_degrees": arm_metrics,
                "mean_of_seed_sample_angle_mae_degrees": float(
                    np.mean(arm_metrics)
                ),
                "sample_std_across_seeds": float(
                    np.std(arm_metrics, ddof=1)
                ),
                "per_seed_equivariance_direction_residual_degrees": (
                    equivariance_direction
                ),
                "mean_equivariance_direction_residual_degrees": float(
                    np.mean(equivariance_direction)
                ),
                "per_seed_equivariance_pivot_residual_fraction": (
                    equivariance_pivot
                ),
                "mean_equivariance_pivot_residual_fraction": float(
                    np.mean(equivariance_pivot)
                ),
                "per_seed_equivariance_valid_pair_fraction": (
                    equivariance_validity
                ),
            }
    effects: dict[str, Any] = {}
    decoder_effects: dict[str, Any] = {}
    for condition_index, condition in enumerate(GROUPED_VAL_CONDITIONS):
        effects[condition] = {
            "explicit_equivariance": _paired_effect(
                indexed[condition]["paired_supervision_only"],
                indexed[condition]["full"],
                interpretation=(
                    "paired_supervision_only error minus full error; isolates "
                    "the explicit equivariance term"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 100 + condition_index,
            ),
            "projective_pairing": _paired_effect(
                indexed[condition]["no_projective_pair"],
                indexed[condition]["paired_supervision_only"],
                interpretation=(
                    "no_projective_pair error minus paired_supervision_only "
                    "error; isolates transformed paired supervision"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 200 + condition_index,
            ),
            "explicit_regularization_direction_residual": _paired_effect(
                indexed[condition]["paired_supervision_only"],
                indexed[condition]["full"],
                metric="equivariance_direction_residual_degrees",
                interpretation=(
                    "paired-supervision-only transported-direction residual "
                    "minus full residual; empirical effect of explicit "
                    "projective-equivariance regularization"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 700 + condition_index,
            ),
            "explicit_regularization_pivot_residual": _paired_effect(
                indexed[condition]["paired_supervision_only"],
                indexed[condition]["full"],
                metric="equivariance_pivot_residual_fraction",
                interpretation=(
                    "paired-supervision-only transported-pivot residual minus "
                    "full residual; empirical regularization effect"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 800 + condition_index,
            ),
        }
        full_views = {
            view: _project_decoder_view(
                indexed[condition]["full"],
                view,
            )
            for view in DECODER_VIEWS
        }
        decoder_effects[condition] = {
            "per_view": {
                view: {
                    "per_seed_group_macro_angle_mae_degrees": (
                        decoder_seed_metrics[condition][view]
                    ),
                    "mean_across_seeds_group_macro_angle_mae_degrees": float(
                        np.mean(decoder_seed_metrics[condition][view])
                    ),
                }
                for view in DECODER_VIEWS
            },
            "direct_minus_fused": _paired_effect(
                full_views["direct"],
                full_views["fused"],
                interpretation=(
                    "direct-only error minus fused error on the same retained "
                    "checkpoint outputs; positive favors fused decoding"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 900 + condition_index,
            ),
            "circular_minus_fused": _paired_effect(
                full_views["circular"],
                full_views["fused"],
                interpretation=(
                    "circular-only error minus fused error on the same "
                    "retained checkpoint outputs; positive favors fused"
                ),
                seed=GROUPED_VAL_BOOTSTRAP_SEED + 1000 + condition_index,
            ),
        }
    return {
        "schema_version": 1,
        "protocol": "pepd_grouped_val_mechanism_effects_v1",
        "status": "complete",
        "scope": "SyncG official train grouped validation only",
        "role": "mechanism evidence; not algorithm selection",
        "horizon_note": horizon_note,
        "mixed_authoritative_full_arm": {
            "cohort_protocol": PEPD_AUTHORITATIVE_COHORT_PROTOCOL,
            "cohort_sha256": authoritative_cohort_sha256,
            "run_authority": {
                "20260720": "convergence_v1",
                "20260721": "bounded_extension_v2",
                "20260722": "convergence_v1",
            },
            "equal_horizon_model_selection_permitted": False,
        },
        "arms": list(PRIMARY_MECHANISM_ARMS),
        "conditions": list(GROUPED_VAL_CONDITIONS),
        "metrics": summaries,
        "paired_effects": effects,
        "decoder_view_ablation": {
            "protocol": PEPD_DECODER_VIEW_PROTOCOL,
            "retained_arm": "full",
            "views": list(DECODER_VIEWS),
            "same_model_forward_and_logits": True,
            "training_or_finetuning": False,
            "checkpoint_or_decoder_selection": False,
            "all_denominator": True,
            "conditions": decoder_effects,
        },
        "primary_interpretive_conditions": [
            "perspective_severe",
            "combined_severe",
        ],
        "claim_policy": (
            "A mechanism is supported on a condition only when its pre-declared "
            "group-bootstrap 95% CI is strictly above zero. Unsupported effects "
            "narrow the claim and never trigger a replacement-model search. "
            "PEPD is described as projective-equivariance-regularized/trained; "
            "these residuals do not prove exact architectural equivariance."
        ),
        "equivariance_claim_boundary": (
            "empirical transported pivot/direction residual with group "
            "bootstrap; no exact-equivariant architecture claim"
        ),
        "eligible_for_model_selection": False,
        "public_test_field_evaluation_authorized": False,
        "mechanism_cohort_sha256": sha256_file(MECHANISM_COHORT),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "source_identity": {
            "strict_json": strict_json_source_sha256(),
            "summarizer": sha256_source_file(Path(__file__).resolve()),
            "protocol": sha256_source_file(
                PROJECT_ROOT / "experiments" / "pepd_convergence_protocol.py"
            ),
            "evaluator": evaluator_source_sha256,
            "authoritative_cohort": authoritative_cohort_sha256,
        },
    }


def _write_or_validate(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"{path} exists with different content")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError(f"formal mechanism report output must be {DEFAULT_OUTPUT}")
    report = build_report()
    payload = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _write_or_validate(output, payload)
    print(payload, end="")
    print(output)


if __name__ == "__main__":
    main()
