"""Recompute the frozen field table after image-only physical-entity exclusions.

This is a post-hoc, prediction-independent leakage sensitivity analysis.  It
never runs inference, trains a model, changes a threshold, or selects individual
samples based on labels or errors.  The exclusion rule and all prediction-cache
hashes are bound by a protocol frozen before Reviewer B was opened.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.summarize_field_confirmatory import (
    base_prediction,
    flat_prediction,
    read_json,
    read_jsonl,
    rows_by_id,
    transformer_prediction,
)
from experiments.summarize_field_development import (
    compute_metrics,
    paired_physical_group_bootstrap,
    sample_ids_sha256,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "artifacts/protocols/field_confirmatory_physical_entity_leakage_sensitivity_v1.json"
)
DEFAULT_REVIEW_A = (
    PROJECT_ROOT
    / "artifacts/runs/field_holdout_xiangmu2_2026/phash_manual_review_v1/reviewer_a.json"
)
DEFAULT_REVIEW_B = (
    PROJECT_ROOT
    / "artifacts/runs/field_holdout_xiangmu2_2026/phash_manual_review_v1/reviewer_b.json"
)

SUMMARY_PROTOCOL = "field_confirmatory_physical_entity_leakage_sensitivity_summary_v1"
VERIFICATION_PROTOCOL = (
    "field_confirmatory_physical_entity_leakage_sensitivity_verification_v1"
)
TRIGGER_DECISIONS = {
    "same_source_frame/derived_duplicate",
    "same_physical_meter_cross_split",
}
DECISION_ALIASES = {
    "likely_derived_duplicate_cross_split": "same_source_frame/derived_duplicate",
    "false_positive": "visually_similar_distinct_meter",
}
METHOD_ORDER = (
    "pepd_only",
    "pepd_fadr",
    "vdn_official200",
    "base_mask",
    "original_transformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--reviewer-a", type=Path, default=DEFAULT_REVIEW_A)
    parser.add_argument("--reviewer-b", type=Path, default=DEFAULT_REVIEW_B)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def resolve_bound_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = resolve_bound_path(str(binding["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    expected_bytes = binding.get("bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"{label} byte size differs from the frozen protocol")
    if sha256_file(path) != str(binding["sha256"]):
        raise ValueError(f"{label} SHA-256 differs from the frozen protocol")
    return path


def bind(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def extract_reviews(value: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    for key in ("reviews", "pairs", "decisions"):
        rows = value.get(key)
        if isinstance(rows, list):
            if not all(isinstance(row, dict) for row in rows):
                raise ValueError(f"{label}.{key} contains a non-object")
            return list(rows)
    raise ValueError(f"{label} lacks a review-record array")


def reviewer_id(value: Mapping[str, Any]) -> str:
    reviewer = value.get("reviewer")
    if isinstance(reviewer, Mapping):
        return str(reviewer.get("id") or "")
    return str(reviewer or value.get("reviewer_id") or "")


def candidate_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    container = audit.get("development_confirmatory_phash")
    rows = container.get("candidates_le_4") if isinstance(container, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("bound leakage audit does not contain exactly 30 <=4 candidates")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("bound leakage audit candidate is not an object")
    return list(rows)


def validate_review(
    review: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    expected_id: str,
) -> dict[str, dict[str, Any]]:
    if reviewer_id(review) != expected_id:
        raise ValueError(f"expected {expected_id} output")
    records = extract_reviews(review, expected_id)
    if len(records) != len(candidates):
        raise ValueError(f"{expected_id} did not review all candidates")
    result: dict[str, dict[str, Any]] = {}
    allowed = TRIGGER_DECISIONS | {"visually_similar_distinct_meter", "uncertain"}
    for index, (record, candidate) in enumerate(zip(records, candidates), 1):
        pair_id = f"pair_{index:03d}"
        development = candidate.get("development")
        confirmatory = candidate.get("confirmatory")
        if not isinstance(development, Mapping) or not isinstance(confirmatory, Mapping):
            raise ValueError(f"{pair_id} candidate identity is malformed")
        record_development = record.get("development")
        record_confirmatory = record.get("confirmatory")
        development_sample_id = record.get("development_sample_id")
        confirmatory_sample_id = record.get("confirmatory_sample_id")
        if isinstance(record_development, Mapping):
            development_sample_id = record_development.get("sample_id")
        if isinstance(record_confirmatory, Mapping):
            confirmatory_sample_id = record_confirmatory.get("sample_id")
        submitted_pair_id = str(record.get("pair_id") or "")
        if (
            submitted_pair_id not in {pair_id, f"phash_{pair_id}"}
            or development_sample_id != development.get("sample_id")
            or confirmatory_sample_id != confirmatory.get("sample_id")
            or int(record.get("phash_hamming_distance", -1))
            != int(candidate.get("phash_hamming_distance", -2))
        ):
            raise ValueError(f"{expected_id} identity/order differs at {pair_id}")
        decision = DECISION_ALIASES.get(
            str(record.get("decision") or ""), str(record.get("decision") or "")
        )
        if decision not in allowed:
            raise ValueError(f"{expected_id} has invalid decision at {pair_id}")
        if pair_id in result:
            raise ValueError(f"{expected_id} duplicated {pair_id}")
        result[pair_id] = {**dict(record), "decision": decision}
    return result


def same_number(left: Any, right: Any) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=0.0, abs_tol=1e-12
    )


def validate_prediction_rows(
    manifest_by_id: Mapping[str, Mapping[str, Any]],
    rows: Sequence[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    mapping = rows_by_id(rows, label)
    if set(mapping) != set(manifest_by_id):
        raise ValueError(f"{label} sample IDs differ from the frozen manifest")
    for sample_id, row in mapping.items():
        source = manifest_by_id[sample_id]
        if (
            row.get("group_id") != source.get("group_id")
            or not same_number(row.get("ground_truth"), source.get("ground_truth"))
            or not same_number(row.get("scale_start"), source.get("scale_start"))
            or not same_number(row.get("scale_end"), source.get("scale_end"))
        ):
            raise ValueError(f"{label}[{sample_id}] identity/label differs")
    return mapping


def build_summary(
    protocol_path: Path,
    review_a_path: Path,
    review_b_path: Path,
) -> tuple[dict[str, Any], Path, Path]:
    protocol_path = protocol_path.resolve()
    review_a_path = review_a_path.resolve()
    review_b_path = review_b_path.resolve()
    protocol = read_json(protocol_path)
    if (
        protocol.get("protocol")
        != "field_confirmatory_physical_entity_leakage_sensitivity_v1"
        or protocol.get("analysis_status")
        != "post_hoc_prediction_independent_leakage_sensitivity"
    ):
        raise ValueError("unexpected sensitivity protocol")

    bound = protocol.get("bound_inputs")
    if not isinstance(bound, Mapping):
        raise ValueError("sensitivity protocol lacks bound inputs")
    paths = {name: verify_binding(value, name) for name, value in bound.items()}
    review_a = read_json(review_a_path)
    review_b = read_json(review_b_path)
    candidates = candidate_rows(read_json(paths["leakage_audit"]))
    a_records = validate_review(review_a, candidates, expected_id="reviewer_a")
    b_records = validate_review(review_b, candidates, expected_id="reviewer_b")

    disagreements: list[str] = []
    triggered_pairs: list[str] = []
    leaked_relations: set[tuple[str, str]] = set()
    pair_audit: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, 1):
        pair_id = f"pair_{index:03d}"
        decision_a = str(a_records[pair_id]["decision"])
        decision_b = str(b_records[pair_id]["decision"])
        if decision_a != decision_b:
            disagreements.append(pair_id)
        if decision_a in TRIGGER_DECISIONS and decision_b in TRIGGER_DECISIONS:
            triggered_pairs.append(pair_id)
            development = candidate["development"]
            confirmatory = candidate["confirmatory"]
            leaked_relations.add(
                (str(development["group_id"]), str(confirmatory["group_id"]))
            )
        pair_audit.append(
            {
                "pair_id": pair_id,
                "reviewer_a_decision": decision_a,
                "reviewer_b_decision": decision_b,
            }
        )
    if disagreements:
        raise RuntimeError(
            "reviewer disagreement requires a third image-only adjudicator before "
            f"metrics: {disagreements}"
        )
    if not leaked_relations:
        raise RuntimeError("dual review did not identify an exclusion relation")
    excluded_groups = sorted({confirmatory for _, confirmatory in leaked_relations})

    original_freeze = read_json(paths["original_confirmatory_protocol"])
    dataset = original_freeze.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("original freeze lacks dataset identity")
    manifest_path = resolve_bound_path(str(dataset["path"]))
    if sha256_file(manifest_path) != str(dataset["sha256"]):
        raise ValueError("confirmatory manifest differs from original freeze")
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = rows_by_id(manifest_rows, "manifest")
    if len(manifest_rows) != int(dataset["rows"]):
        raise ValueError("confirmatory manifest row count differs from original freeze")

    mappings = {
        "raw": validate_prediction_rows(
            manifest_by_id,
            read_jsonl(paths["original_transformer_and_geometry_cache"]),
            "raw",
        ),
        "pepd": validate_prediction_rows(
            manifest_by_id, read_jsonl(paths["pepd_cache"]), "pepd"
        ),
        "base": validate_prediction_rows(
            manifest_by_id, read_jsonl(paths["base_cache"]), "base"
        ),
        "fadr": validate_prediction_rows(
            manifest_by_id, read_jsonl(paths["fadr_cache"]), "fadr"
        ),
        "vdn": validate_prediction_rows(
            manifest_by_id, read_jsonl(paths["vdn_cache"]), "vdn"
        ),
    }
    ordered_ids = [str(row["sample_id"]) for row in manifest_rows]
    predictions: dict[str, list[float | None]] = {
        "pepd_only": [flat_prediction(mappings["pepd"][key]) for key in ordered_ids],
        "pepd_fadr": [flat_prediction(mappings["fadr"][key]) for key in ordered_ids],
        "vdn_official200": [flat_prediction(mappings["vdn"][key]) for key in ordered_ids],
        "base_mask": [base_prediction(mappings["base"][key]) for key in ordered_ids],
        "original_transformer": [
            transformer_prediction(mappings["raw"][key]) for key in ordered_ids
        ],
    }

    retained_indices = [
        index
        for index, row in enumerate(manifest_rows)
        if str(row["group_id"]) not in excluded_groups
    ]
    excluded_indices = [
        index
        for index, row in enumerate(manifest_rows)
        if str(row["group_id"]) in excluded_groups
    ]
    retained_rows = [manifest_rows[index] for index in retained_indices]
    retained_groups = np.asarray(
        [str(row["group_id"]) for row in retained_rows], dtype=object
    )
    if len(set(retained_groups.tolist())) < 2:
        raise ValueError("fewer than two confirmatory groups remain")

    all_metrics: dict[str, Any] = {}
    retained_metrics: dict[str, Any] = {}
    retained_errors: dict[str, np.ndarray] = {}
    for method in METHOD_ORDER:
        all_metrics[method], _, _ = compute_metrics(
            manifest_rows, predictions[method], failure_penalty=1.0
        )
        method_predictions = [predictions[method][index] for index in retained_indices]
        retained_metrics[method], retained_errors[method], _ = compute_metrics(
            retained_rows, method_predictions, failure_penalty=1.0
        )

    bootstrap = protocol["analysis"]["bootstrap"]
    paired_effects: dict[str, Any] = {}
    for comparator in (
        "vdn_official200",
        "base_mask",
        "original_transformer",
        "pepd_fadr",
    ):
        effect = paired_physical_group_bootstrap(
            retained_errors[comparator],
            retained_errors["pepd_only"],
            retained_groups,
            iterations=int(bootstrap["iterations"]),
            seed=int(bootstrap["seed"]),
        )
        paired_effects[f"{comparator}_minus_pepd_only"] = {
            "comparator": comparator,
            "reference": "pepd_only",
            "delta_nmae_comparator_minus_pepd": effect[
                "delta_full_denominator_nmae_final_minus_comparator"
            ],
            "paired_physical_group_bootstrap_95ci": effect[
                "paired_physical_group_bootstrap_95ci"
            ],
            "bootstrap_probability_comparator_better_than_pepd": effect[
                "bootstrap_probability_final_better"
            ],
            "iterations": effect["iterations"],
            "seed": effect["seed"],
            "physical_groups": effect["physical_groups"],
            "unit": effect["unit"],
            "analysis_status": "post_hoc_prediction_independent_leakage_sensitivity",
        }

    retained_ids = {str(row["sample_id"]): {} for row in retained_rows}
    excluded_ids = {
        str(manifest_rows[index]["sample_id"]): {} for index in excluded_indices
    }
    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("sensitivity protocol lacks output paths")
    output_path = resolve_bound_path(str(outputs["summary"]))
    verification_path = resolve_bound_path(str(outputs["verification"]))
    summary = {
        "schema_version": 1,
        "protocol": SUMMARY_PROTOCOL,
        "status": "complete",
        "analysis_status": "post_hoc_prediction_independent_leakage_sensitivity",
        "scope": {
            "training_performed": False,
            "inference_performed": False,
            "threshold_or_method_selection_performed": False,
            "individual_samples_selected_by_error": False,
            "original_one_shot_result_retained_for_audit": True,
            "original_split_may_be_called_physical_entity_disjoint": False,
        },
        "bindings": {
            "protocol": bind(protocol_path),
            "reviewer_a": bind(review_a_path),
            "reviewer_b": bind(review_b_path),
            "manifest": bind(manifest_path),
            "prediction_caches": {
                name: bind(path)
                for name, path in paths.items()
                if name.endswith("_cache")
                or name == "original_transformer_and_geometry_cache"
            },
        },
        "dual_review": {
            "candidate_pairs": len(candidates),
            "agreement_pairs": len(candidates) - len(disagreements),
            "disagreement_pair_ids": disagreements,
            "triggered_pair_ids": triggered_pairs,
            "pair_decisions": pair_audit,
            "leaked_group_relations": [
                {
                    "development_group_id": development,
                    "confirmatory_group_id": confirmatory,
                }
                for development, confirmatory in sorted(leaked_relations)
            ],
        },
        "denominators": {
            "original_samples": len(manifest_rows),
            "original_groups": len({str(row["group_id"]) for row in manifest_rows}),
            "excluded_confirmatory_group_ids": excluded_groups,
            "excluded_samples": len(excluded_indices),
            "excluded_sample_ids_sha256": sample_ids_sha256(excluded_ids),
            "retained_samples": len(retained_rows),
            "retained_groups": len(set(retained_groups.tolist())),
            "retained_sample_ids_sha256": sample_ids_sha256(retained_ids),
        },
        "original_cache_recomputed_metrics": all_metrics,
        "sensitivity_metrics": retained_metrics,
        "paired_effects": paired_effects,
        "interpretation_boundary": [
            "The original one-shot table remains an audit record but is not physically entity-disjoint.",
            "The sensitivity denominator excludes whole confirmatory group IDs selected only by dual image review.",
            "No model was retrained or rerun; all values come from immutable prediction caches.",
            "Because the leak review occurred after the one-shot evaluation, sensitivity intervals are descriptive post-hoc evidence, not a replacement confirmatory test.",
        ],
    }
    return summary, output_path, verification_path


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    expected, output_path, verification_path = build_summary(
        args.protocol, args.reviewer_a, args.reviewer_b
    )
    if args.verify_only:
        actual = read_json(output_path)
        if actual != expected:
            raise ValueError("existing sensitivity summary differs from recomputation")
        print(
            json.dumps(
                {
                    "verified": True,
                    "summary": str(output_path),
                    "sha256": sha256_file(output_path),
                },
                ensure_ascii=False,
            )
        )
        return

    write_immutable_json(output_path, expected)
    verification = {
        "schema_version": 1,
        "protocol": VERIFICATION_PROTOCOL,
        "verified": True,
        "summary": bind(output_path),
        "checks": {
            "dual_review_agreement_complete": True,
            "whole_confirmatory_groups_excluded": True,
            "bound_prediction_cache_hashes_match": True,
            "no_training_or_inference": True,
            "all_methods_share_retained_denominator": True,
        },
    }
    write_immutable_json(verification_path, verification)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_path),
                "verification": str(verification_path),
                "excluded_groups": expected["denominators"][
                    "excluded_confirmatory_group_ids"
                ],
                "retained_samples": expected["denominators"]["retained_samples"],
                "retained_groups": expected["denominators"]["retained_groups"],
                "pepd_nmae": expected["sensitivity_metrics"]["pepd_only"][
                    "full_denominator_nmae"
                ],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
