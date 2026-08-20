"""Summarize three frozen ReMSTNet-v3 fits on the four paired real domains."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments import evaluate_a15_2_mett_real_domains as shared
from experiments.evaluate_remstnet_real_domains import PROTOCOL as EVALUATION_PROTOCOL
from experiments.train_remstnet import ADAPTIVE_PROTOCOL
from remstnet.model import ADAPTIVE_REMST_NET_ARCHITECTURE


PROTOCOL: Final[str] = "remstnet_v3_real_domain_three_seed_summary_v1"
EXPECTED_SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20260818


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{label} is non-finite")
    return result


def _mean_sample_sd(values: Sequence[float]) -> dict[str, float]:
    numbers = tuple(float(value) for value in values)
    _require(bool(numbers), "seed metric vector is empty")
    return {
        "mean": statistics.fmean(numbers),
        "sample_sd": statistics.stdev(numbers) if len(numbers) >= 2 else 0.0,
    }


def _full_identity(model: Mapping[str, Any], *, source: Path) -> tuple[int, dict[str, Any], dict[str, int]]:
    construction = model.get("construction")
    foundation = model.get("source_foundation")
    counts = model.get("parameter_counts")
    training_seeds = model.get("seeds")
    _require(
        isinstance(construction, Mapping)
        and isinstance(foundation, Mapping)
        and isinstance(counts, Mapping)
        and isinstance(training_seeds, Mapping),
        f"ReMSTNet model identity is incomplete: {source}",
    )
    _require(
        model.get("architecture") == ADAPTIVE_REMST_NET_ARCHITECTURE
        and model.get("protocol") == ADAPTIVE_PROTOCOL
        and int(model.get("epochs", 0)) == 5
        and construction.get("architecture_variant") == "adaptive_budget_progress_mixing_v3"
        and construction.get("use_progress_mixing") is True
        and construction.get("learnable_budget_gain") is True,
        f"evaluation is not a terminal full ReMSTNet-v3 fit: {source}",
    )
    seed = int(foundation.get("source_seed", -1))
    _require(seed in EXPECTED_SEEDS, f"source seed is outside the paired roster: {source}")
    identity = {
        "construction": dict(construction),
        "parameter_counts": dict(counts),
        "source_protocol": foundation.get("source_protocol"),
        "source_architecture": foundation.get("source_architecture"),
        "source_epochs": int(foundation.get("source_epochs", -1)),
        "source_checkpoint_selection": foundation.get("source_checkpoint_selection"),
    }
    seeds = {
        "initialization": int(training_seeds.get("initialization", -1)),
        "sample_order": int(training_seeds.get("sample_order", -1)),
    }
    return seed, identity, seeds


def _load_evaluation(path: Path) -> dict[str, Any]:
    """Load one formal artifact and retain its measured replay evidence.

    The concrete failure is pairing a valid ReMSTNet checkpoint with a valid
    but different same-seed external ledger, which leaves paths, types, row
    keys, uniqueness, Git state, versions, and atomic writes all apparently
    correct while changing the numerical comparison.  Ordinary unit tests do
    not execute this particular artifact pair, so the publication boundary
    requires the evaluator's full-roster numerical endpoint replay instead.
    """

    source = Path(path).resolve()
    _require(source.is_file(), f"real-domain evaluation is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, Mapping), f"evaluation is not an object: {source}")
    scope = payload.get("scope")
    _require(
        payload.get("protocol") == EVALUATION_PROTOCOL
        and payload.get("status") == "complete"
        and isinstance(scope, Mapping)
        and scope.get("inference_precision") == "float32"
        and scope.get("training_or_adaptation_during_evaluation") is False
        and scope.get("sample_router_or_prediction_fusion") is False,
        f"real-domain evaluation protocol or scope differs: {source}",
    )
    model = payload.get("model")
    _require(isinstance(model, Mapping), f"model metadata is missing: {source}")
    seed, identity, training_seeds = _full_identity(model, source=source)
    _require(int(payload.get("source_seed", -1)) == seed, f"top-level source seed differs: {source}")
    documents = payload.get("datasets")
    _require(
        isinstance(documents, Mapping)
        and set(documents) == set(shared.REAL_DATASET_KEYS),
        f"four-domain roster differs: {source}",
    )

    datasets: dict[str, Any] = {}
    for dataset_name in shared.REAL_DATASET_KEYS:
        document = documents[dataset_name]
        _require(isinstance(document, Mapping), f"dataset document is invalid: {dataset_name}")
        rows = document.get("per_sample_condition")
        audit = document.get("endpoint_replay_audit")
        _require(isinstance(rows, list) and bool(rows), f"rows are missing: {dataset_name}")
        _require(
            isinstance(audit, Mapping)
            and audit.get("within_tolerance") is True
            and int(audit.get("source_seed", -1)) == seed
            and int(audit.get("rows", -1)) == len(rows)
            and int(audit.get("status_mismatch_rows", -1)) == 0
            and _finite(audit.get("maximum_absolute_prediction_delta"), label="endpoint replay delta")
            <= _finite(audit.get("tolerance"), label="endpoint replay tolerance"),
            f"complete endpoint replay evidence differs: {source}/{dataset_name}",
        )
        indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
        identities: dict[tuple[str, str], tuple[float, str]] = {}
        for row_index, row in enumerate(rows):
            _require(isinstance(row, Mapping), f"row is invalid: {dataset_name}/{row_index}")
            key = (str(row.get("sample_id", "")), str(row.get("condition", "")))
            _require(
                bool(key[0]) and key[1] in shared.CONDITIONS and key not in indexed,
                f"sample-condition key differs: {dataset_name}/{key}",
            )
            target = _finite(row.get("normalized_target"), label=f"target {key}")
            group = str(row.get("group_id", ""))
            candidate = row.get("candidate")
            remstnet = candidate.get("mett") if isinstance(candidate, Mapping) else None
            _require(
                bool(group) and isinstance(remstnet, Mapping) and remstnet.get("status") == "pass",
                f"ReMSTNet record is incomplete: {dataset_name}/{key}",
            )
            prediction = _finite(remstnet.get("prediction"), label=f"ReMSTNet prediction {key}")
            error = _finite(remstnet.get("absolute_error"), label=f"ReMSTNet error {key}")
            _require(abs(error - abs(prediction - target)) <= 2.0e-6, f"ReMSTNet error differs: {key}")
            external = row.get("efficientnet_b0")
            sarn = external.get("sarn_v2") if isinstance(external, Mapping) else None
            _require(
                isinstance(sarn, Mapping)
                and set(sarn) == {str(value) for value in EXPECTED_SEEDS},
                f"external seed roster differs: {dataset_name}/{key}",
            )
            for external_seed in EXPECTED_SEEDS:
                record = sarn[str(external_seed)]
                _require(isinstance(record, Mapping) and record.get("status") == "pass", f"external row failed: {key}/{external_seed}")
                external_prediction = _finite(record.get("prediction"), label=f"external prediction {key}")
                external_error = _finite(record.get("absolute_error"), label=f"external error {key}")
                _require(abs(external_error - abs(external_prediction - target)) <= 2.0e-6, f"external error differs: {key}")
            indexed[key] = row
            identities[key] = (target, group)
        datasets[dataset_name] = {
            "rows": indexed,
            "identities": identities,
            "dataset": dict(document.get("dataset", {})),
            "baseline_inputs": document.get("baseline_inputs"),
            "endpoint_replay": dict(audit),
        }
    return {
        "path": source,
        "seed": seed,
        "identity": identity,
        "training_seeds": training_seeds,
        "datasets": datasets,
    }


def _paired(candidate: Sequence[float], comparator: Sequence[float], groups: Sequence[str], *, replicates: int, seed: int) -> dict[str, Any]:
    result = shared.paired_group_bootstrap(
        candidate,
        comparator,
        groups,
        replicates=replicates,
        seed=seed,
        comparator_label="sarn_v2_efficientnet_b0",
    )
    result["delta_definition"] = "remstnet_minus_sarn_v2_efficientnet_b0"
    return result


def _subset_summary(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    keys: Sequence[tuple[str, str]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    reference = evaluations[0]["datasets"][dataset_name]
    groups = [reference["identities"][key][1] for key in keys]
    candidate_by_seed: list[list[float]] = []
    comparator_by_seed: list[list[float]] = []
    per_seed: dict[str, Any] = {}
    for seed_index, evaluation in enumerate(evaluations):
        source_seed = int(evaluation["seed"])
        rows = evaluation["datasets"][dataset_name]["rows"]
        candidate = [float(rows[key]["candidate"]["mett"]["absolute_error"]) for key in keys]
        comparator = [
            float(rows[key]["efficientnet_b0"]["sarn_v2"][str(source_seed)]["absolute_error"])
            for key in keys
        ]
        candidate_by_seed.append(candidate)
        comparator_by_seed.append(comparator)
        per_seed[str(source_seed)] = {
            "remstnet": shared.full_denominator_metrics(candidate),
            "sarn_v2_efficientnet_b0_seed_matched": shared.full_denominator_metrics(comparator),
            "remstnet_group_macro": shared.group_macro_metrics(candidate, [True] * len(candidate), groups)["macro"],
            "paired": _paired(
                candidate,
                comparator,
                groups,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + seed_index,
            ),
        }
    candidate_metrics = [per_seed[str(seed)]["remstnet"] for seed in EXPECTED_SEEDS]
    comparator_metrics = [per_seed[str(seed)]["sarn_v2_efficientnet_b0_seed_matched"] for seed in EXPECTED_SEEDS]
    mean_candidate_error = np.mean(np.asarray(candidate_by_seed, dtype=np.float64), axis=0).tolist()
    mean_comparator_error = np.mean(np.asarray(comparator_by_seed, dtype=np.float64), axis=0).tolist()
    return {
        "rows_per_seed": len(keys),
        "groups": len(set(groups)),
        "per_seed": per_seed,
        "three_seed": {
            "remstnet_metrics_mean_sample_sd": {
                metric: _mean_sample_sd([row[metric] for row in candidate_metrics])
                for metric in candidate_metrics[0]
            },
            "comparator_metrics_mean_sample_sd": {
                metric: _mean_sample_sd([row[metric] for row in comparator_metrics])
                for metric in comparator_metrics[0]
            },
            "mean_row_error_across_independent_seeds": {
                "remstnet": shared.full_denominator_metrics(mean_candidate_error),
                "sarn_v2_efficientnet_b0": shared.full_denominator_metrics(mean_comparator_error),
                "paired": _paired(
                    mean_candidate_error,
                    mean_comparator_error,
                    groups,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + 10_000,
                ),
            },
        },
    }


def summarize_evaluations(
    evaluation_paths: Sequence[Path],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    evaluations = sorted((_load_evaluation(path) for path in evaluation_paths), key=lambda row: int(row["seed"]))
    _require(
        len(evaluations) == 3
        and tuple(int(row["seed"]) for row in evaluations) == EXPECTED_SEEDS,
        "exact three-seed evaluation roster differs",
    )
    _require(
        len({(row["training_seeds"]["initialization"], row["training_seeds"]["sample_order"]) for row in evaluations}) == 3,
        "independent training seed pairs must differ",
    )
    reference = evaluations[0]
    for evaluation in evaluations[1:]:
        _require(evaluation["identity"] == reference["identity"], f"model construction differs: {evaluation['seed']}")
        for dataset_name in shared.REAL_DATASET_KEYS:
            observed = evaluation["datasets"][dataset_name]
            expected = reference["datasets"][dataset_name]
            _require(
                observed["identities"] == expected["identities"]
                and set(observed["rows"]) == set(expected["rows"])
                and observed["dataset"] == expected["dataset"]
                and observed["baseline_inputs"] == expected["baseline_inputs"],
                f"paired dataset identity differs: {evaluation['seed']}/{dataset_name}",
            )
            for key, row in observed["rows"].items():
                _require(row["efficientnet_b0"] == expected["rows"][key]["efficientnet_b0"], f"external rows differ: {evaluation['seed']}/{dataset_name}/{key}")

    datasets: dict[str, Any] = {}
    for dataset_index, dataset_name in enumerate(shared.REAL_DATASET_KEYS):
        keys = tuple(sorted(reference["datasets"][dataset_name]["rows"], key=lambda key: (shared.CONDITIONS.index(key[1]), key[0])))
        subsets: dict[str, Any] = {}
        for condition_index, condition in enumerate((*shared.CONDITIONS, "all_conditions")):
            subset_keys = keys if condition == "all_conditions" else tuple(key for key in keys if key[1] == condition)
            subsets[condition] = _subset_summary(
                evaluations,
                dataset_name=dataset_name,
                keys=subset_keys,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + dataset_index * 100_000 + condition_index * 1_000,
            )
        datasets[dataset_name] = {
            "dataset": reference["datasets"][dataset_name]["dataset"],
            "subsets": subsets,
            "endpoint_replay_by_seed": {
                str(row["seed"]): row["datasets"][dataset_name]["endpoint_replay"]
                for row in evaluations
            },
        }
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "publication_model": {
            "machine_key": "remstnet_v3",
            "display_name": "ReMSTNet-v3",
            "architecture": ADAPTIVE_REMST_NET_ARCHITECTURE,
        },
        "scope": {
            "external_comparator": "same-source-seed SARN-v2 + EfficientNet-B0",
            "main_reporting": "mean and sample SD across three independent fits",
            "mean_row_error_is_not_prediction_ensemble": True,
        },
        "seeds": list(EXPECTED_SEEDS),
        "input_files": [str(row["path"]) for row in evaluations],
        "bootstrap": {
            "unit": "complete physical instrument group",
            "replicates": bootstrap_replicates,
            "base_seed": bootstrap_seed,
            "interval": "two-sided percentile 95%",
        },
        "datasets": datasets,
    }


def write_summary(evaluation_paths: Sequence[Path], *, output_path: Path, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES, bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"real-domain summary output exists: {output}")
    result = summarize_evaluations(
        evaluation_paths,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = write_summary(
        args.evaluation,
        output_path=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "datasets": list(result["datasets"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "summarize_evaluations", "write_summary"]
