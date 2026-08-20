"""Summarize three ReMSTNet-v3 runs against paired external architectures."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.evaluate_a15_2_mett_syncg import CONDITIONS
from experiments.evaluate_remst_block_syncg import PROTOCOL as EVALUATION_PROTOCOL
from experiments.remst_block_net import ADAPTIVE_REMST_NET_ARCHITECTURE
from experiments.train_remst_block_pilot import ADAPTIVE_PROTOCOL


PROTOCOL: Final[str] = "remstnet_v3_multiseed_external_comparison_v1"
EXPECTED_SOURCE_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
EXPECTED_EXTERNAL_MODELS: Final[tuple[str, ...]] = (
    "Direct-ResNet18",
    "EfficientNet-B0",
    "MobileNetV3-Large",
)
PROJECTIVE_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"perspective_moderate", "perspective_severe", "combined_severe"}
)
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260818
ENDPOINT_REPLAY_TOLERANCE: Final[float] = 5.0e-4


class ReMSTNetMultiseedError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetMultiseedError(message)


def _metrics(errors: Sequence[float]) -> dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    _require(bool(values.size) and bool(np.isfinite(values).all()), "metric errors invalid")
    return {
        "nmae": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_absolute_error": float(np.quantile(values, 0.95)),
        "p99_absolute_error": float(np.quantile(values, 0.99)),
        "maximum_absolute_error": float(values.max()),
        "acc_at_1_percent": float(np.mean(values <= 0.01)),
        "acc_at_2_percent": float(np.mean(values <= 0.02)),
        "acc_at_5_percent": float(np.mean(values <= 0.05)),
    }


def _mean_sd(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "seed statistic is empty")
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["sample_id"]), str(row["condition"])


def _candidate_record(row: Mapping[str, Any]) -> Mapping[str, Any]:
    record = row.get("remstnet", row.get("mett"))
    _require(isinstance(record, Mapping), "ReMSTNet row candidate missing")
    return record


def _validate_external(row: Mapping[str, Any]) -> None:
    external = row.get("external")
    _require(isinstance(external, Mapping), "external comparison block missing")
    _require(
        tuple(external) == EXPECTED_EXTERNAL_MODELS,
        "external model roster differs",
    )
    target = float(row["normalized_target"])
    for model_name in EXPECTED_EXTERNAL_MODELS:
        block = external[model_name]
        _require(isinstance(block, Mapping), f"{model_name} block malformed")
        seeds = tuple(int(value) for value in block.get("seeds", ()))
        predictions = tuple(float(value) for value in block.get("predictions", ()))
        errors = tuple(float(value) for value in block.get("absolute_errors", ()))
        passed = tuple(bool(value) for value in block.get("passed", ()))
        _require(seeds == EXPECTED_SOURCE_SEEDS, f"{model_name} seed roster differs")
        _require(
            len(predictions) == len(errors) == len(passed) == len(seeds)
            and all(passed),
            f"{model_name} external vector/status differs",
        )
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in predictions)
            and all(math.isfinite(value) and value >= 0.0 for value in errors),
            f"{model_name} external values invalid",
        )
        _require(
            all(abs(abs(prediction - target) - error) <= 1.0e-8 for prediction, error in zip(predictions, errors, strict=True)),
            f"{model_name} external error is inconsistent",
        )


def _load_evaluation(path: Path) -> dict[str, Any]:
    """Load one formal run and prove its same-seed B0 endpoint pairing.

    The concrete failure is supplying another terminal checkpoint with the
    same architecture, protocol, seed, row keys, and value types.  Git,
    versions, primary/unique constraints, transactions, and typing cannot
    establish that its numerical endpoint equals the external ledger, while
    ordinary tests do not execute this concrete artifact pair.  At the formal
    publication boundary we therefore retain a full-row numerical replay;
    the model evaluations themselves have already executed normally.
    """

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, Mapping), "evaluation payload malformed")
    _require(payload.get("protocol") == EVALUATION_PROTOCOL, "evaluation protocol differs")
    _require(payload.get("status") == "pilot_complete", "evaluation is incomplete")
    scope = payload.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("training_or_adaptation_during_evaluation") is False
        and scope.get("prediction_dependent_routing") is False
        and scope.get("inference_precision") == "float32",
        "evaluation scope differs",
    )
    model = payload.get("model")
    _require(isinstance(model, Mapping), "evaluation model metadata missing")
    construction = model.get("construction")
    _require(isinstance(construction, Mapping), "construction metadata missing")
    _require(model.get("architecture") == ADAPTIVE_REMST_NET_ARCHITECTURE, "not full ReMSTNet-v3")
    _require(model.get("protocol") == ADAPTIVE_PROTOCOL, "training protocol differs")
    _require(int(model.get("epochs", 0)) == 5, "full ReMSTNet must use five epochs")
    _require(
        construction.get("architecture_variant") == "adaptive_budget_progress_mixing_v3"
        and construction.get("use_progress_mixing") is True
        and construction.get("learnable_budget_gain") is True,
        "full ReMSTNet mechanism identity differs",
    )
    source_foundation = model.get("source_foundation")
    _require(isinstance(source_foundation, Mapping), "source foundation metadata missing")
    source_seed = int(source_foundation.get("source_seed", -1))
    counts = model.get("parameter_counts")
    training_seeds = model.get("seeds")
    _require(
        source_seed in EXPECTED_SOURCE_SEEDS
        and source_foundation.get("source_protocol")
        == "syncg_lightweight_regression_baselines_v1"
        and source_foundation.get("source_architecture") == "efficientnet_b0"
        and int(source_foundation.get("source_epochs", -1)) == 30
        and source_foundation.get("source_checkpoint_selection")
        == "terminal_fixed_epoch"
        and isinstance(counts, Mapping)
        and isinstance(training_seeds, Mapping),
        "source or matched model identity differs",
    )
    rows = payload.get("per_sample_condition")
    _require(isinstance(rows, list) and bool(rows), "evaluation rows missing")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    endpoint_deltas: list[float] = []
    endpoint_seed_index = EXPECTED_SOURCE_SEEDS.index(source_seed)
    for row in rows:
        _require(isinstance(row, Mapping), "evaluation row malformed")
        key = _row_key(row)
        _require(key not in indexed, "evaluation row keys repeat")
        _require(key[1] in CONDITIONS, "evaluation condition differs")
        candidate = _candidate_record(row)
        prediction = float(candidate["prediction"])
        error = float(candidate["absolute_error"])
        target = float(row["normalized_target"])
        _require(
            math.isfinite(prediction)
            and 0.0 <= prediction <= 1.0
            and abs(abs(prediction - target) - error) <= 1.0e-8,
            "candidate prediction/error is inconsistent",
        )
        _validate_external(row)
        endpoint = row.get("sarn_endpoint")
        external_b0 = row["external"]["EfficientNet-B0"]
        _require(
            isinstance(endpoint, Mapping)
            and bool(external_b0["passed"][endpoint_seed_index]),
            "same-seed endpoint replay status differs",
        )
        internal_endpoint = float(endpoint.get("prediction"))
        external_endpoint = float(external_b0["predictions"][endpoint_seed_index])
        _require(
            math.isfinite(internal_endpoint)
            and math.isfinite(external_endpoint),
            "same-seed endpoint replay is non-finite",
        )
        endpoint_deltas.append(abs(internal_endpoint - external_endpoint))
        indexed[key] = row
    sample_conditions: dict[str, set[str]] = {}
    for sample_id, condition in indexed:
        sample_conditions.setdefault(sample_id, set()).add(condition)
    _require(
        all(values == set(CONDITIONS) for values in sample_conditions.values())
        and len(indexed) == len(sample_conditions) * len(CONDITIONS),
        "evaluation Cartesian roster differs",
    )
    maximum_endpoint_delta = max(endpoint_deltas)
    _require(
        len(endpoint_deltas) == len(indexed)
        and maximum_endpoint_delta <= ENDPOINT_REPLAY_TOLERANCE,
        "same-seed SARN-v2 EfficientNet-B0 endpoint replay differs",
    )
    return {
        "path": str(source),
        "source_seed": source_seed,
        "model": dict(model),
        "scope": dict(scope),
        "elapsed_seconds": float(payload.get("evaluation_elapsed_seconds", 0.0)),
        "rows": indexed,
        "model_identity": {
            "construction": dict(construction),
            "parameter_counts": dict(counts),
            "source_protocol": source_foundation.get("source_protocol"),
            "source_architecture": source_foundation.get("source_architecture"),
            "source_epochs": int(source_foundation.get("source_epochs")),
            "source_checkpoint_selection": source_foundation.get(
                "source_checkpoint_selection"
            ),
        },
        "training_seeds": {
            "initialization": int(training_seeds.get("initialization", -1)),
            "sample_order": int(training_seeds.get("sample_order", -1)),
        },
        "endpoint_replay_evidence": {
            "comparator": "same-source-seed SARN-v2 + EfficientNet-B0",
            "source_seed": source_seed,
            "compared_rows": len(endpoint_deltas),
            "maximum_absolute_prediction_delta": maximum_endpoint_delta,
            "absolute_prediction_tolerance": ENDPOINT_REPLAY_TOLERANCE,
            "within_tolerance": True,
        },
    }


def _paired_scene_bootstrap(
    candidate_errors: Sequence[float],
    comparator_errors: Sequence[float],
    scenes: Sequence[str],
    *,
    replicates: int,
    seed: int,
    comparator: str,
) -> dict[str, Any]:
    _require(replicates >= 1, "bootstrap replicates must be positive")
    _require(
        len(candidate_errors) == len(comparator_errors) == len(scenes) and bool(scenes),
        "paired vectors are misaligned",
    )
    effects = np.asarray(candidate_errors) - np.asarray(comparator_errors)
    grouped: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        grouped.setdefault(str(scene), []).append(index)
    scene_ids = tuple(sorted(grouped))
    _require(len(scene_ids) >= 2, "scene bootstrap needs at least two scenes")
    sums = np.asarray([effects[grouped[scene]].sum() for scene in scene_ids])
    counts = np.asarray([len(grouped[scene]) for scene in scene_ids])
    rng = random.Random(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = [rng.randrange(len(scene_ids)) for _ in scene_ids]
        draws[replicate] = float(sums[sampled].sum() / counts[sampled].sum())
    low, high = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "comparator": comparator,
        "delta_definition": "ReMSTNet-v3 minus comparator NMAE",
        "delta_nmae": float(effects.mean()),
        "scene_grouped_bootstrap_ci95": {"low": low, "high": high},
        "superiority_ci95": high < 0.0,
        "improvement_fraction": float(np.mean(effects < 0.0)),
        "negative_transfer_fraction": float(np.mean(effects > 0.0)),
        "tie_fraction": float(np.mean(effects == 0.0)),
        "scene_clusters": len(scene_ids),
        "rows": len(effects),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def _scope_keys(keys: Sequence[tuple[str, str]], label: str) -> tuple[tuple[str, str], ...]:
    if label == "all_conditions":
        return tuple(keys)
    if label == "projective_pooled":
        return tuple(key for key in keys if key[1] in PROJECTIVE_CONDITIONS)
    return tuple(key for key in keys if key[1] == label)


def summarize_evaluations(
    evaluation_paths: Sequence[Path],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(len(evaluation_paths) == 3, "exactly three ReMSTNet evaluations required")
    loaded = [_load_evaluation(path) for path in evaluation_paths]
    loaded.sort(key=lambda item: int(item["source_seed"]))
    _require(
        tuple(int(item["source_seed"]) for item in loaded) == EXPECTED_SOURCE_SEEDS,
        "ReMSTNet source seed roster differs",
    )
    _require(
        len(
            {
                (
                    int(item["training_seeds"]["initialization"]),
                    int(item["training_seeds"]["sample_order"]),
                )
                for item in loaded
            }
        )
        == 3,
        "independent ReMSTNet training seed pairs must differ",
    )
    reference_keys = tuple(loaded[0]["rows"])
    _require(bool(reference_keys), "evaluation roster is empty")
    reference_rows = loaded[0]["rows"]
    for evaluation in loaded[1:]:
        _require(
            evaluation["model_identity"] == loaded[0]["model_identity"],
            "ReMSTNet construction or parameter inventory differs",
        )
        _require(tuple(evaluation["rows"]) == reference_keys, "evaluation row order differs")
        for key in reference_keys:
            left = reference_rows[key]
            right = evaluation["rows"][key]
            _require(
                str(left["scene_stem"]) == str(right["scene_stem"])
                and abs(float(left["normalized_target"]) - float(right["normalized_target"])) <= 1.0e-8,
                "evaluation target/scene alignment differs",
            )
            _require(left["external"] == right["external"], "external paired rows differ")

    scopes = ("all_conditions", "projective_pooled", *CONDITIONS)
    summary: dict[str, Any] = {}
    for scope_index, label in enumerate(scopes):
        selected = _scope_keys(reference_keys, label)
        _require(bool(selected), f"{label} scope is empty")
        scenes = [str(reference_rows[key]["scene_stem"]) for key in selected]
        candidate_by_seed = {
            int(evaluation["source_seed"]): [
                float(_candidate_record(evaluation["rows"][key])["absolute_error"])
                for key in selected
            ]
            for evaluation in loaded
        }
        per_seed_metrics = {
            str(seed): _metrics(errors) for seed, errors in candidate_by_seed.items()
        }
        metric_mean_sd = {
            metric: _mean_sd([per_seed_metrics[str(seed)][metric] for seed in EXPECTED_SOURCE_SEEDS])
            for metric in next(iter(per_seed_metrics.values()))
        }
        candidate_row_mean = [
            statistics.fmean(candidate_by_seed[seed][index] for seed in EXPECTED_SOURCE_SEEDS)
            for index in range(len(selected))
        ]
        external: dict[str, Any] = {}
        for model_index, model_name in enumerate(EXPECTED_EXTERNAL_MODELS):
            errors_by_seed = {
                seed: [
                    float(reference_rows[key]["external"][model_name]["absolute_errors"][seed_index])
                    for key in selected
                ]
                for seed_index, seed in enumerate(EXPECTED_SOURCE_SEEDS)
            }
            external_per_seed = {
                str(seed): _metrics(errors) for seed, errors in errors_by_seed.items()
            }
            external_row_mean = [
                statistics.fmean(errors_by_seed[seed][index] for seed in EXPECTED_SOURCE_SEEDS)
                for index in range(len(selected))
            ]
            same_seed = {
                str(seed): _paired_scene_bootstrap(
                    candidate_by_seed[seed],
                    errors_by_seed[seed],
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 100 + model_index * 10 + seed_index,
                    comparator=f"same-seed SARN-v2+{model_name}",
                )
                for seed_index, seed in enumerate(EXPECTED_SOURCE_SEEDS)
            }
            external[model_name] = {
                "per_seed": external_per_seed,
                "metric_across_seed_mean_sd": {
                    metric: _mean_sd([external_per_seed[str(seed)][metric] for seed in EXPECTED_SOURCE_SEEDS])
                    for metric in next(iter(external_per_seed.values()))
                },
                "paired_rowwise_three_seed_mean": _paired_scene_bootstrap(
                    candidate_row_mean,
                    external_row_mean,
                    scenes,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + scope_index * 100 + model_index,
                    comparator=f"three-seed-mean SARN-v2+{model_name}",
                ),
                "same_seed_paired": same_seed,
            }
        summary[label] = {
            "conditions": list(CONDITIONS) if label == "all_conditions" else (
                sorted(PROJECTIVE_CONDITIONS) if label == "projective_pooled" else [label]
            ),
            "rows_per_seed": len(selected),
            "scenes": len(set(scenes)),
            "remstnet_v3": {
                "per_seed": per_seed_metrics,
                "metric_across_seed_mean_sd": metric_mean_sd,
                "rowwise_seed_mean_error_metrics": _metrics(candidate_row_mean),
            },
            "external_models": external,
        }

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "three_independent_training_runs": True,
            "same_roster_and_conditions": True,
            "external_comparator": "SARN-v2 plus external architecture",
            "prediction_ensemble_used_for_claim": False,
            "rowwise_seed_mean_is_error_aggregation_not_prediction_fusion": True,
        },
        "seeds": list(EXPECTED_SOURCE_SEEDS),
        "inputs": [
            {
                "source_seed": int(evaluation["source_seed"]),
                "evaluation": evaluation["path"],
                "model": evaluation["model"],
                "evaluation_elapsed_seconds": evaluation["elapsed_seconds"],
            }
            for evaluation in loaded
        ],
        "endpoint_replay_evidence": {
            str(evaluation["source_seed"]): evaluation[
                "endpoint_replay_evidence"
            ]
            for evaluation in loaded
        },
        "bootstrap": {
            "method": "paired complete-scene resampling",
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
            "interval": "two-sided percentile 95%",
        },
        "summary": summary,
    }


def write_summary(
    evaluation_paths: Sequence[Path],
    output_path: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"summary output already exists: {output}")
    result = summarize_evaluations(
        evaluation_paths,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = write_summary(
        args.evaluation,
        args.output,
        bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    all_conditions = result["summary"]["all_conditions"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "remstnet_v3_nmae": all_conditions["remstnet_v3"]["metric_across_seed_mean_sd"]["nmae"],
                "external": {
                    name: block["paired_rowwise_three_seed_mean"]
                    for name, block in all_conditions["external_models"].items()
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "EXPECTED_EXTERNAL_MODELS",
    "EXPECTED_SOURCE_SEEDS",
    "PROTOCOL",
    "ReMSTNetMultiseedError",
    "summarize_evaluations",
    "write_summary",
]
