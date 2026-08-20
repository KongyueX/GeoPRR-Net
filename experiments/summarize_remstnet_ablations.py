"""Summarize the publication ReMSTNet mechanism ablations on one paired seed."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.evaluate_a15_2_mett_syncg import CONDITIONS
from experiments.evaluate_remst_block_syncg import PROTOCOL as EVALUATION_PROTOCOL
from experiments.remst_block_net import (
    ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
    ADAPTIVE_REMST_NET_ARCHITECTURE,
    COORDINATED_REMST_NET_ARCHITECTURE,
    PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
    REMST_BLOCK_NET_ARCHITECTURE,
)
from experiments.summarize_remstnet_multiseed import _candidate_record, _metrics
from experiments.train_remst_block_pilot import (
    ADAPTIVE_BUDGET_ABLATION_PROTOCOL,
    ADAPTIVE_PROTOCOL,
    COORDINATED_PROTOCOL,
    PROGRESS_MIXING_ABLATION_PROTOCOL,
    PROTOCOL as HIERARCHICAL_PROTOCOL,
)


PROTOCOL: Final[str] = "remstnet_mechanism_factorial_ablation_summary_v1"
SOURCE_SEED: Final[int] = 20262022
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260818
ENDPOINT_TOLERANCE: Final[float] = 5.0e-4
FACTORIAL_LABELS: Final[tuple[str, ...]] = (
    "coordinated_fixed_budget_no_mixing",
    "progress_mixing_fixed_budget",
    "adaptive_budget_no_mixing",
    "full_v3",
)
ARM_IDENTITIES: Final[dict[str, tuple[str, str, str, bool, bool]]] = {
    "coordinated_fixed_budget_no_mixing": (
        COORDINATED_REMST_NET_ARCHITECTURE,
        COORDINATED_PROTOCOL,
        "cross_scale_coordinated_v2",
        False,
        False,
    ),
    "progress_mixing_fixed_budget": (
        PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
        PROGRESS_MIXING_ABLATION_PROTOCOL,
        "progress_mixing_fixed_budget_ablation",
        True,
        False,
    ),
    "adaptive_budget_no_mixing": (
        ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
        ADAPTIVE_BUDGET_ABLATION_PROTOCOL,
        "adaptive_budget_no_progress_mixing_ablation",
        False,
        True,
    ),
    "full_v3": (
        ADAPTIVE_REMST_NET_ARCHITECTURE,
        ADAPTIVE_PROTOCOL,
        "adaptive_budget_progress_mixing_v3",
        True,
        True,
    ),
}
PROJECTIVE_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"perspective_moderate", "perspective_severe", "combined_severe"}
)


class ReMSTNetAblationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetAblationError(message)


def _parse_spec(spec: str) -> tuple[str, Path]:
    label, separator, path = spec.partition("=")
    _require(bool(separator) and bool(label) and bool(path), "evaluation must be label=path")
    return label, Path(path)


def _load_arm(label: str, path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    # Formal table assembly is where a correctly shaped but mislabeled arm
    # would become a false mechanism claim, so identity is checked here.
    _require(isinstance(payload, Mapping), f"{label}: evaluation malformed")
    _require(payload.get("protocol") == EVALUATION_PROTOCOL, f"{label}: evaluation protocol differs")
    _require(payload.get("status") == "pilot_complete", f"{label}: evaluation incomplete")
    model = payload.get("model")
    _require(isinstance(model, Mapping), f"{label}: model metadata missing")
    construction = model.get("construction")
    _require(isinstance(construction, Mapping), f"{label}: construction missing")
    if label == "hierarchical_local_v1":
        expected = (
            REMST_BLOCK_NET_ARCHITECTURE,
            HIERARCHICAL_PROTOCOL,
            "hierarchical_v1",
        )
        _require(
            model.get("architecture") == expected[0]
            and model.get("protocol") == expected[1]
            and construction.get("architecture_variant") == expected[2],
            f"{label}: hierarchical identity differs",
        )
    else:
        _require(label in ARM_IDENTITIES, f"{label}: unknown factorial arm")
        architecture, training_protocol, variant, mixing, adaptive = ARM_IDENTITIES[label]
        _require(model.get("architecture") == architecture, f"{label}: architecture differs")
        _require(model.get("protocol") == training_protocol, f"{label}: training protocol differs")
        _require(construction.get("architecture_variant") == variant, f"{label}: variant differs")
        _require(
            bool(construction.get("use_progress_mixing", False)) is mixing
            and bool(construction.get("learnable_budget_gain", False)) is adaptive,
            f"{label}: mechanism flags differ",
        )
    _require(int(model.get("epochs", 0)) == 5, f"{label}: epochs differ")
    source_foundation = model.get("source_foundation")
    _require(isinstance(source_foundation, Mapping), f"{label}: source foundation missing")
    _require(int(source_foundation.get("source_seed", -1)) == SOURCE_SEED, f"{label}: source seed differs")
    rows = payload.get("per_sample_condition")
    _require(isinstance(rows, list) and bool(rows), f"{label}: rows missing")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), f"{label}: row malformed")
        key = (str(row["sample_id"]), str(row["condition"]))
        _require(key not in indexed and key[1] in CONDITIONS, f"{label}: row key differs")
        _candidate_record(row)
        indexed[key] = row
    return {
        "label": label,
        "path": str(source),
        "model": dict(model),
        "rows": indexed,
    }


def _scope_keys(keys: Sequence[tuple[str, str]], label: str) -> tuple[tuple[str, str], ...]:
    if label == "all_conditions":
        return tuple(keys)
    if label == "projective_pooled":
        return tuple(key for key in keys if key[1] in PROJECTIVE_CONDITIONS)
    return tuple(key for key in keys if key[1] == label)


def _bootstrap_effect(
    effects: Sequence[float],
    scenes: Sequence[str],
    *,
    definition: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(effects, dtype=np.float64)
    _require(len(values) == len(scenes) and bool(scenes), "effect vectors misaligned")
    grouped: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        grouped.setdefault(str(scene), []).append(index)
    scene_ids = tuple(sorted(grouped))
    _require(len(scene_ids) >= 2, "effect bootstrap needs two scenes")
    sums = np.asarray([values[grouped[scene]].sum() for scene in scene_ids])
    counts = np.asarray([len(grouped[scene]) for scene in scene_ids])
    rng = random.Random(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = [rng.randrange(len(scene_ids)) for _ in scene_ids]
        draws[replicate] = float(sums[selected].sum() / counts[selected].sum())
    low, high = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "delta_definition": definition,
        "delta_nmae": float(values.mean()),
        "scene_grouped_bootstrap_ci95": {"low": low, "high": high},
        "different_from_zero_ci95": high < 0.0 or low > 0.0,
        "improvement_fraction_if_negative": float(np.mean(values < 0.0)),
        "negative_fraction_if_positive": float(np.mean(values > 0.0)),
        "scene_clusters": len(scene_ids),
        "rows": len(values),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def summarize_ablations(
    evaluation_specs: Mapping[str, Path],
    *,
    hierarchical_evaluation: Path | None = None,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(tuple(evaluation_specs) == FACTORIAL_LABELS, "factorial arm order/roster differs")
    arms = {label: _load_arm(label, evaluation_specs[label]) for label in FACTORIAL_LABELS}
    hierarchical = (
        _load_arm("hierarchical_local_v1", hierarchical_evaluation)
        if hierarchical_evaluation is not None
        else None
    )
    reference = arms["full_v3"]
    keys = tuple(reference["rows"])
    expected_seeds = reference["model"].get("seeds")
    endpoint_audit: dict[str, Any] = {}
    for label, arm in ({**arms, **({"hierarchical_local_v1": hierarchical} if hierarchical else {})}).items():
        _require(tuple(arm["rows"]) == keys, f"{label}: roster/order differs")
        _require(arm["model"].get("seeds") == expected_seeds, f"{label}: init/order seeds differ")
        raw_deltas: list[float] = []
        sarn_deltas: list[float] = []
        for key in keys:
            left = reference["rows"][key]
            right = arm["rows"][key]
            _require(
                str(left["scene_stem"]) == str(right["scene_stem"])
                and abs(float(left["normalized_target"]) - float(right["normalized_target"])) <= 1.0e-8,
                f"{label}: target/scene differs",
            )
            _require(left["external"] == right["external"], f"{label}: external rows differ")
            raw_deltas.append(abs(float(left["raw_anchor"]["prediction"]) - float(right["raw_anchor"]["prediction"])))
            sarn_deltas.append(abs(float(left["sarn_endpoint"]["prediction"]) - float(right["sarn_endpoint"]["prediction"])))
        maximum = max((*raw_deltas, *sarn_deltas))
        _require(maximum <= ENDPOINT_TOLERANCE, f"{label}: endpoint replay differs")
        endpoint_audit[label] = {
            "rows": len(keys),
            "maximum_absolute_prediction_delta": maximum,
            "tolerance": ENDPOINT_TOLERANCE,
        }

    scopes = ("all_conditions", "projective_pooled", *CONDITIONS)
    summary: dict[str, Any] = {}
    for scope_index, scope in enumerate(scopes):
        selected = _scope_keys(keys, scope)
        scenes = [str(reference["rows"][key]["scene_stem"]) for key in selected]
        errors = {
            label: [float(_candidate_record(arm["rows"][key])["absolute_error"]) for key in selected]
            for label, arm in arms.items()
        }
        arm_metrics = {label: _metrics(values) for label, values in errors.items()}
        full = np.asarray(errors["full_v3"])
        comparisons = {
            label: _bootstrap_effect(
                full - np.asarray(errors[label]),
                scenes,
                definition=f"full_v3 minus {label} NMAE",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + scope_index * 100 + index,
            )
            for index, label in enumerate(FACTORIAL_LABELS[:-1])
        }
        base = np.asarray(errors["coordinated_fixed_budget_no_mixing"])
        mixing = np.asarray(errors["progress_mixing_fixed_budget"])
        adaptive = np.asarray(errors["adaptive_budget_no_mixing"])
        factorial_effects = {
            "progress_mixing_at_fixed_budget": mixing - base,
            "progress_mixing_at_adaptive_budget": full - adaptive,
            "adaptive_budget_without_progress_mixing": adaptive - base,
            "adaptive_budget_with_progress_mixing": full - mixing,
            "interaction": full - mixing - adaptive + base,
        }
        effects = {
            label: _bootstrap_effect(
                values,
                scenes,
                definition=f"{label}; negative favors enabled mechanism",
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + scope_index * 100 + 20 + index,
            )
            for index, (label, values) in enumerate(factorial_effects.items())
        }
        block: dict[str, Any] = {
            "rows": len(selected),
            "scenes": len(set(scenes)),
            "metrics": arm_metrics,
            "full_vs_ablation": comparisons,
            "factorial_effects": effects,
        }
        if hierarchical is not None:
            hierarchical_errors = [
                float(_candidate_record(hierarchical["rows"][key])["absolute_error"])
                for key in selected
            ]
            block["architecture_evolution_supplement"] = {
                "hierarchical_local_v1": _metrics(hierarchical_errors),
                "coordinated_v2_minus_hierarchical_v1": _bootstrap_effect(
                    base - np.asarray(hierarchical_errors),
                    scenes,
                    definition="coordinated_v2 minus hierarchical_local_v1 NMAE",
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 100 + 40,
                ),
            }
        summary[scope] = block

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "main_table_role": "mechanism attribution only",
            "external_model_comparison_is_reported_separately": True,
            "factorial_design": "progress mixing x adaptive moment budget",
            "single_paired_source_seed": SOURCE_SEED,
            "same_initialization_and_sample_order_seeds": True,
        },
        "arms": {
            label: {"evaluation": arm["path"], "model": arm["model"]}
            for label, arm in arms.items()
        },
        "hierarchical_supplement": (
            {"evaluation": hierarchical["path"], "model": hierarchical["model"]}
            if hierarchical is not None
            else None
        ),
        "endpoint_replay_audit": endpoint_audit,
        "bootstrap": {
            "method": "paired complete-scene resampling",
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
        },
        "summary": summary,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--hierarchical-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    parsed = dict(_parse_spec(spec) for spec in args.evaluation)
    result = summarize_ablations(
        parsed,
        hierarchical_evaluation=args.hierarchical_evaluation,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output).resolve()
    _require(not output.exists(), f"ablation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": result["status"], "output": str(output), "all_conditions": result["summary"]["all_conditions"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FACTORIAL_LABELS",
    "PROTOCOL",
    "ReMSTNetAblationError",
    "summarize_ablations",
]
