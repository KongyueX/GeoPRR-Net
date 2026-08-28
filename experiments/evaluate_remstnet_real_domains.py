"""Evaluate frozen ReMSTNet-v3 checkpoints on the four paired real domains."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments import evaluate_a15_2_mett_real_domains as mett_real
from experiments.evaluate_a15_2_mett_syncg import posterior_batch_diagnostics
from experiments.train_remstnet import ADAPTIVE_PROTOCOL, load_remstnet_checkpoint
from remstnet.model import ADAPTIVE_REMST_NET_ARCHITECTURE, CoordinatedReMSTNet


PROTOCOL: Final[str] = "remstnet_v3_real_domain_frozen_evaluation_v1"


class ReMSTNetRealDomainError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetRealDomainError(message)


def _validate_full_model(model: CoordinatedReMSTNet, metadata: Mapping[str, Any]) -> int:
    """Reject a loadable ablation mislabeled as the publication model.

    A four-arm ablation checkpoint has the same tensor types, row keys,
    storage format, source seed, and much of the same code/version history as
    full v3, so Git, versions, primary/unique keys, transactions, typing, and
    ordinary model tests cannot detect a wrong CLI path.  This check is kept
    at the formal evaluation boundary and the selected model is still run.
    """

    construction = metadata.get("construction")
    source = metadata.get("source_foundation")
    _require(isinstance(construction, Mapping), "ReMSTNet construction metadata missing")
    _require(isinstance(source, Mapping), "ReMSTNet source metadata missing")
    _require(
        metadata.get("architecture") == ADAPTIVE_REMST_NET_ARCHITECTURE
        and metadata.get("protocol") == ADAPTIVE_PROTOCOL
        and int(metadata.get("epochs", 0)) == 5,
        "real-domain evaluation requires full five-epoch ReMSTNet-v3",
    )
    _require(
        construction.get("architecture_variant") == "adaptive_budget_progress_mixing_v3"
        and construction.get("use_progress_mixing") is True
        and construction.get("learnable_budget_gain") is True
        and model.use_progress_mixing
        and model.learnable_budget_gain,
        "real-domain ReMSTNet mechanism identity differs",
    )
    source_seed = int(source.get("source_seed", -1))
    _require(source_seed in mett_real.SEEDS, "ReMSTNet seed lacks paired real baselines")
    return source_seed


def _evaluate_dataset(
    dataset: Any,
    *,
    model: CoordinatedReMSTNet,
    source_seed: int,
    prediction_root: Path,
    device: torch.device,
    workers: int,
    batch_size: int,
    bootstrap_replicates: int,
    include_posterior_diagnostics: bool,
    dataset_index: int,
    enforce_endpoint_replay: bool = True,
) -> dict[str, Any]:
    """Run one domain and numerically confirm its same-seed comparator.

    A different but structurally valid external ledger can share all sample
    IDs, types, versions, and uniqueness properties.  Neither Git nor atomic
    writes establish prediction equivalence, and unit tests do not cover this
    concrete checkpoint/ledger pair; therefore the formal result requires
    full-row endpoint replay after actual inference.
    """

    targets_tuple, targets, runs, baseline_inputs = mett_real._load_baseline_runs(
        dataset,
        prediction_root=prediction_root,
    )
    target_for_dataset = {
        target.sample_id: (float(target.normalized_target), str(target.group_id))
        for target in targets_tuple
    }
    manifest_rows = mett_real.load_manifest(dataset.manifest)
    manifest_by_id = {row.sample_id: row for row in manifest_rows}
    _require(
        tuple(manifest_by_id) == tuple(target.sample_id for target in targets_tuple),
        f"{dataset.slug}: model manifest order differs from roster",
    )
    ordered_manifest = tuple(manifest_by_id[target.sample_id] for target in targets_tuple)

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, condition in enumerate(mett_real.CONDITIONS):
            shared_dataset = mett_real._SharedPixelDataset(
                ordered_manifest,
                target_for_dataset,
                condition=condition,
            )
            loader = mett_real._loader(
                shared_dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=mett_real.BOOTSTRAP_SEED + dataset_index * 100 + condition_index,
                cuda=device.type == "cuda",
            )
            for raw_batch in loader:
                mett_real._validate_batch_pixel_identity(raw_batch, condition=condition, runs=runs)
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                groups = tuple(str(value) for value in raw_batch["scene_stem"])
                target_tensor = raw_batch["target"].float()
                try:
                    original = raw_batch["original_view"].to(device)
                    sarn = raw_batch["sarn_view"].to(device)
                    support = raw_batch["sarn_support_mask"].to(device)
                    active = raw_batch["sarn_active"].to(device).bool()
                    homography = raw_batch["raw_to_sarn_homography"].to(device)
                    effective_active = active & (condition != "clean")
                    prediction = model(
                        original,
                        sarn,
                        support,
                        sarn_active=effective_active,
                        raw_to_sarn_homography=homography,
                    )
                    values = {
                        "raw_anchor": prediction["raw_anchor_mean"].float().cpu(),
                        "sarn_endpoint": prediction["sarn_endpoint_mean"].float().cpu(),
                        "tangent_base": prediction["sarn_endpoint_mean"].float().cpu(),
                        "mett": prediction["mean"].float().cpu(),
                    }
                    diagnostics: dict[str, dict[str, torch.Tensor]] = {}
                    if include_posterior_diagnostics:
                        diagnostics = {
                            method: posterior_batch_diagnostics(value, target_tensor)
                            for method, value in {
                                "raw_anchor": prediction["raw_anchor_posterior"],
                                "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                                "mett": prediction["progress_posterior"],
                            }.items()
                        }
                    relation = prediction["relation_available"].bool().cpu()
                    inference_failure: str | None = None
                except Exception as exc:  # full-denominator failure accounting
                    values = {}
                    diagnostics = {}
                    relation = torch.zeros(len(ids), dtype=torch.bool)
                    inference_failure = f"model_exception:{type(exc).__name__}"

                for row_index, (sample_id, group_id) in enumerate(zip(ids, groups, strict=True)):
                    target = float(target_tensor[row_index])
                    expected = float(targets[sample_id].normalized_target)
                    _require(abs(target - expected) <= 1.0e-6, f"{sample_id}: target differs")
                    candidate: dict[str, Any] = {}
                    for method in mett_real.CANDIDATE_METHODS:
                        value = float(values[method][row_index]) if inference_failure is None else None
                        candidate[method] = mett_real._record(
                            value,
                            expected,
                            passed=inference_failure is None,
                        )
                        if inference_failure is not None:
                            candidate[method]["failure_code"] = inference_failure
                        if method in diagnostics:
                            candidate[method]["posterior"] = {
                                name: bool(tensor[row_index]) if tensor.dtype == torch.bool else float(tensor[row_index])
                                for name, tensor in diagnostics[method].items()
                            }
                    key = (sample_id, condition)
                    external = {
                        variant: {
                            str(seed): mett_real._external_record(runs[variant][seed], key=key, target=expected)
                            for seed in mett_real.SEEDS
                        }
                        for variant in ("raw", "sarn_v2")
                    }
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "group_id": group_id,
                            "condition": condition,
                            "normalized_target": expected,
                            "relation_available": bool(relation[row_index]),
                            "candidate": candidate,
                            "efficientnet_b0": external,
                        }
                    )

    _require(
        len(rows) == dataset.expected_samples * len(mett_real.CONDITIONS),
        f"{dataset.slug}: Cartesian output count differs",
    )
    summary = mett_real.summarize_rows(
        rows,
        source_seed=source_seed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=mett_real.BOOTSTRAP_SEED + dataset_index * 1000,
    )
    raw_deltas: list[float] = []
    sarn_deltas: list[float] = []
    status_mismatches = 0
    for row in rows:
        for method, variant, deltas in (
            ("raw_anchor", "raw", raw_deltas),
            ("sarn_endpoint", "sarn_v2", sarn_deltas),
        ):
            internal = row["candidate"][method]
            external = row["efficientnet_b0"][variant][str(source_seed)]
            if internal["status"] != external["status"]:
                status_mismatches += 1
            if internal["status"] == external["status"] == "pass":
                deltas.append(abs(float(internal["prediction"]) - float(external["prediction"])))
    maximum = max((*raw_deltas, *sarn_deltas), default=float("inf"))
    endpoint_replay_within_tolerance = (
        status_mismatches == 0
        and len(raw_deltas) == len(rows)
        and len(sarn_deltas) == len(rows)
        and maximum <= mett_real.ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE
    )
    if enforce_endpoint_replay:
        _require(
            endpoint_replay_within_tolerance,
            f"{dataset.slug}: paired endpoint replay differs",
        )
    return {
        "dataset": {
            "slug": dataset.slug,
            "paper_name": dataset.paper_name,
            "samples": dataset.expected_samples,
            "groups": dataset.expected_groups,
            "group_unit": dataset.group_unit,
            "labels": str(dataset.labels.resolve()),
            "roster": str(dataset.roster.resolve()),
            "manifest": str(dataset.manifest.resolve()),
        },
        "baseline_inputs": baseline_inputs,
        "endpoint_replay_audit": {
            "source_seed": source_seed,
            "rows": len(rows),
            "status_mismatch_rows": status_mismatches,
            "maximum_absolute_prediction_delta": maximum,
            "tolerance": mett_real.ENDPOINT_REPLAY_ABSOLUTE_TOLERANCE,
            "within_tolerance": endpoint_replay_within_tolerance,
            "enforced": bool(enforce_endpoint_replay),
        },
        "summary": summary,
        "per_sample_condition": rows,
    }


def evaluate_real_domains(
    *,
    checkpoint_path: Path,
    output_path: Path,
    dataset_keys: Sequence[str] = mett_real.REAL_DATASET_KEYS,
    prediction_root: Path = mett_real.DEFAULT_PREDICTION_ROOT,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = mett_real.DEFAULT_REAL_DOMAIN_BATCH_SIZE,
    bootstrap_replicates: int = mett_real.DEFAULT_BOOTSTRAP_REPLICATES,
    include_posterior_diagnostics: bool = False,
) -> dict[str, Any]:
    selected = tuple(str(value) for value in dataset_keys)
    _require(
        bool(selected)
        and len(selected) == len(set(selected))
        and set(selected) <= set(mett_real.REAL_DATASET_KEYS),
        "real-domain dataset selection invalid",
    )
    _require(workers >= 0 and batch_size >= 1 and bootstrap_replicates >= 1, "runtime settings invalid")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"output already exists: {output}")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.manual_seed_all(mett_real.BOOTSTRAP_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(mett_real.BOOTSTRAP_SEED)
    model, metadata = load_remstnet_checkpoint(checkpoint_path, device=device)
    _require(isinstance(model, CoordinatedReMSTNet), "checkpoint is not coordinated ReMSTNet")
    source_seed = _validate_full_model(model, metadata)
    model.eval()

    started = time.perf_counter()
    datasets: dict[str, Any] = {}
    for dataset_index, key in enumerate(selected):
        result = _evaluate_dataset(
            mett_real.factorial.DATASETS[key],
            model=model,
            source_seed=source_seed,
            prediction_root=prediction_root,
            device=device,
            workers=workers,
            batch_size=batch_size,
            bootstrap_replicates=bootstrap_replicates,
            include_posterior_diagnostics=include_posterior_diagnostics,
            dataset_index=dataset_index,
        )
        datasets[key] = result
        all_conditions = result["summary"]["all_conditions"]
        print(
            json.dumps(
                {
                    "dataset": key,
                    "remstnet_nmae": all_conditions["candidate"]["mett"]["full_denominator"]["nmae"],
                    "matched_external_nmae": all_conditions["efficientnet_b0_baselines"]["sarn_v2"]["per_seed"][str(source_seed)]["full_denominator"]["nmae"],
                    "delta": all_conditions["comparisons"]["matched_same_seed_sarn_v2"]["delta_nmae"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "frozen_checkpoint": True,
            "training_or_adaptation_during_evaluation": False,
            "sample_router_or_prediction_fusion": False,
            "inference_precision": "float32",
            "posterior_diagnostics": bool(include_posterior_diagnostics),
            "external_comparator": "same-seed Raw/SARN-v2 EfficientNet-B0",
        },
        "model": metadata,
        "source_seed": source_seed,
        "conditions": list(mett_real.CONDITIONS),
        "bootstrap": {
            "method": "paired complete-group resampling",
            "replicates": bootstrap_replicates,
            "base_seed": mett_real.BOOTSTRAP_SEED,
        },
        "evaluation_elapsed_seconds": float(time.perf_counter() - started),
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", action="append", choices=mett_real.REAL_DATASET_KEYS)
    parser.add_argument("--prediction-root", type=Path, default=mett_real.DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=mett_real.DEFAULT_REAL_DOMAIN_BATCH_SIZE)
    parser.add_argument("--bootstrap", type=int, default=mett_real.DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--posterior", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_real_domains(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        dataset_keys=args.dataset or mett_real.REAL_DATASET_KEYS,
        prediction_root=args.prediction_root,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap,
        include_posterior_diagnostics=args.posterior,
    )
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "source_seed": result["source_seed"], "elapsed_seconds": result["evaluation_elapsed_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "evaluate_real_domains"]
