"""Evaluate three frozen ReMSTNet-v3 seeds on the natural-repeat cohort."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments import evaluate_a15_2_mett_natural_repeat_stability as mett_repeat
from experiments.evaluate_remstnet_real_domains import _validate_full_model
from experiments.remst_block_net import CoordinatedReMSTNet
from experiments.score_natural_repeat_stability import PredictionValue
from experiments.train_remst_block_pilot import load_remst_block_checkpoint


PROTOCOL: Final[str] = "remstnet_v3_natural_repeat_stability_v1"
METHOD_LABELS: Final[dict[str, str]] = {
    "raw_anchor": "Raw anchor",
    "sarn_endpoint": "SARN endpoint",
    "mett": "ReMSTNet-v3",
}


class ReMSTNetNaturalRepeatError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetNaturalRepeatError(message)


def checkpoint_roster(paths: Sequence[Path]) -> dict[int, Path]:
    checkpoints = tuple(Path(path).resolve() for path in paths)
    _require(len(checkpoints) == 3 and len(set(checkpoints)) == 3, "exactly three distinct checkpoints required")
    result: dict[int, Path] = {}
    for checkpoint in checkpoints:
        _require(checkpoint.is_file(), f"checkpoint missing: {checkpoint}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _require(isinstance(payload, Mapping), "checkpoint payload malformed")
        source = payload.get("source_foundation")
        construction = payload.get("construction")
        _require(isinstance(source, Mapping) and isinstance(construction, Mapping), "checkpoint identity missing")
        seed = int(source.get("source_seed", -1))
        _require(
            payload.get("architecture") == "ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3"
            and int(payload.get("epochs", 0)) == 5
            and construction.get("use_progress_mixing") is True
            and construction.get("learnable_budget_gain") is True,
            "natural-repeat requires full ReMSTNet-v3",
        )
        _require(seed in mett_repeat.EXPECTED_SEEDS and seed not in result, "checkpoint seed roster differs")
        result[seed] = checkpoint
    _require(tuple(sorted(result)) == mett_repeat.EXPECTED_SEEDS, "checkpoint seeds incomplete")
    return result


def _run_checkpoint(
    *,
    checkpoint: Path,
    seed: int,
    dataset: Any,
    device: torch.device,
    workers: int,
    batch_size: int,
    use_amp: bool,
) -> tuple[
    dict[str, dict[str, PredictionValue]],
    dict[str, dict[str, bool]],
    dict[str, Any],
]:
    model, metadata = load_remst_block_checkpoint(checkpoint, device=device)
    _require(isinstance(model, CoordinatedReMSTNet), "checkpoint is not coordinated ReMSTNet")
    observed_seed = _validate_full_model(model, metadata)
    _require(observed_seed == seed, "loaded ReMSTNet seed differs")
    model.eval()
    loader = mett_repeat._loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        seed=20260818 + seed,
        cuda=device.type == "cuda",
    )
    predictions: dict[str, dict[str, PredictionValue]] = {
        method: {} for method in mett_repeat.METHODS
    }
    evidence: dict[str, dict[str, bool]] = {}
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in loader:
            original = batch["original_view"].to(device, non_blocking=device.type == "cuda")
            sarn = batch["sarn_view"].to(device, non_blocking=device.type == "cuda")
            support = batch["sarn_support_mask"].to(device, non_blocking=device.type == "cuda")
            active = batch["sarn_active"].to(device, non_blocking=device.type == "cuda").bool()
            homography = batch["raw_to_sarn_homography"].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                output = model(
                    original,
                    sarn,
                    support,
                    sarn_active=active,
                    raw_to_sarn_homography=homography,
                )
            values = {
                "raw_anchor": output["raw_anchor_mean"].detach().float().cpu(),
                "sarn_endpoint": output["sarn_endpoint_mean"].detach().float().cpu(),
                "mett": output["mean"].detach().float().cpu(),
            }
            active_cpu = active.detach().cpu()
            relation_cpu = output["relation_available"].detach().bool().cpu()
            sample_ids = tuple(str(value) for value in batch["sample_id"])
            for row_index, sample_id in enumerate(sample_ids):
                _require(sample_id not in evidence, f"duplicate sample: {sample_id}")
                evidence[sample_id] = {
                    "sarn_applied": bool(active_cpu[row_index]),
                    "relation_available": bool(relation_cpu[row_index]),
                }
                for method, tensor in values.items():
                    value = float(tensor[row_index])
                    _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"invalid prediction: {seed}/{sample_id}")
                    predictions[method][sample_id] = PredictionValue(True, value)
    _require(len(evidence) == len(dataset), f"prediction count differs for seed {seed}")
    return predictions, evidence, metadata


def evaluate_natural_repeat_stability(
    *,
    checkpoints: Sequence[Path],
    output_path: Path,
    manifest_path: Path | None = None,
    assets: Any = mett_repeat.DEFAULT_ASSETS,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
    bootstrap_replicates: int = mett_repeat.DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = mett_repeat.DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require(workers >= 0 and batch_size >= 1, "loader configuration invalid")
    _require(bootstrap_replicates >= 100, "bootstrap needs at least 100 replicates")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"natural-repeat output exists: {output}")
    started = time.perf_counter()
    checkpoints_by_seed = checkpoint_roster(checkpoints)
    cohort, cohort_summary = mett_repeat.load_xm2_repeat_cohort(
        provenance_path=assets.provenance,
        labels_path=assets.labels,
        xm2_manifest_path=assets.physical_capture_manifest,
    )
    manifest = Path(manifest_path or mett_repeat.cohort_manifest_path(mett_repeat.DEFAULT_ROOT)).resolve()
    _require(manifest.is_file(), f"cohort manifest missing: {manifest}")
    manifest_rows = mett_repeat.load_manifest(manifest)
    manifest_by_id = {row.sample_id: row for row in manifest_rows}
    cohort_ids = {row.sample_id for row in cohort}
    _require(len(manifest_by_id) == len(manifest_rows) and set(manifest_by_id) == cohort_ids, "cohort roster differs")
    ordered_manifest = tuple(manifest_by_id[row.sample_id] for row in cohort)
    targets = {row.sample_id: (row.target, row.group_id) for row in cohort}
    dataset = mett_repeat._SharedPixelDataset(ordered_manifest, targets, condition="clean")

    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.manual_seed_all(20260818)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(20260818)
    predictions: dict[str, dict[int, dict[str, PredictionValue]]] = {
        method: {} for method in mett_repeat.METHODS
    }
    runtime_evidence: dict[int, dict[str, dict[str, bool]]] = {}
    model_metadata: dict[str, Any] = {}
    seed_elapsed_seconds: dict[str, float] = {}
    for seed in mett_repeat.EXPECTED_SEEDS:
        seed_started = time.perf_counter()
        seed_predictions, evidence, metadata = _run_checkpoint(
            checkpoint=checkpoints_by_seed[seed],
            seed=seed,
            dataset=dataset,
            device=device,
            workers=workers,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        for method in mett_repeat.METHODS:
            predictions[method][seed] = seed_predictions[method]
        runtime_evidence[seed] = evidence
        model_metadata[str(seed)] = metadata
        seed_elapsed_seconds[str(seed)] = float(time.perf_counter() - seed_started)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary, units = mett_repeat.summarize_predictions(
        cohort,
        predictions,
        seeds=mett_repeat.EXPECTED_SEEDS,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        candidate_label="ReMSTNet-v3",
    )
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "condition": "clean natural repeated captures",
            "training_or_adaptation_during_evaluation": False,
            "prediction_dependent_routing_or_selection": False,
            "cross_seed_ensemble": False,
            "same_source_image_rule": "average predictions before within-unit spread",
            "candidate_machine_key": "remstnet",
            "legacy_candidate_machine_key": "mett",
            "inference_precision": "bfloat16_or_float16_autocast" if device.type == "cuda" and use_amp else "float32",
        },
        "publication_model": {
            "machine_key": "remstnet",
            "display_name": "ReMSTNet-v3",
            "architecture": "ReMSTNet-Adaptive-Budget-Progress-Mixing-Relation-Moment-Backbone-v3",
        },
        "cohort": mett_repeat._canonical_cohort_summary(cohort_summary),
        "models": model_metadata,
        "summary": summary,
        "per_unit": mett_repeat._unit_evidence(cohort, units, seeds=mett_repeat.EXPECTED_SEEDS),
        "per_sample": mett_repeat._prediction_evidence(
            cohort,
            predictions,
            runtime_evidence,
            seeds=mett_repeat.EXPECTED_SEEDS,
        ),
        "inputs": {
            "manifest": str(manifest),
            "provenance": str(Path(assets.provenance).resolve()),
            "labels": str(Path(assets.labels).resolve()),
            "physical_capture_manifest": str(Path(assets.physical_capture_manifest).resolve()),
            "checkpoints": {str(seed): str(checkpoints_by_seed[seed]) for seed in mett_repeat.EXPECTED_SEEDS},
        },
        "timing": {
            "per_seed_seconds": seed_elapsed_seconds,
            "total_seconds": float(time.perf_counter() - started),
        },
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
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=mett_repeat.cohort_manifest_path(mett_repeat.DEFAULT_ROOT))
    parser.add_argument("--provenance", type=Path, default=mett_repeat.DEFAULT_ASSETS.provenance)
    parser.add_argument("--labels", type=Path, default=mett_repeat.DEFAULT_ASSETS.labels)
    parser.add_argument("--physical-capture-manifest", type=Path, default=mett_repeat.DEFAULT_ASSETS.physical_capture_manifest)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=mett_repeat.DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=mett_repeat.DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    assets = mett_repeat.RepeatAssets(
        provenance=args.provenance,
        labels=args.labels,
        physical_capture_manifest=args.physical_capture_manifest,
        source_input_manifest=mett_repeat.DEFAULT_ASSETS.source_input_manifest,
    )
    result = evaluate_natural_repeat_stability(
        checkpoints=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        assets=assets,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "summary": result["summary"], "timing": result["timing"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "checkpoint_roster", "evaluate_natural_repeat_stability"]
