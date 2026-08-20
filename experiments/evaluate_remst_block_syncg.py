"""Evaluate one ReMSTNet pilot on the established six-condition SyncG roster."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.evaluate_a15_2_mett_syncg import (
    DEFAULT_MANIFEST,
    DEFAULT_REFERENCE,
    _metrics,
    _summarize,
    posterior_batch_diagnostics,
    validate_external_reference,
    validate_external_reference_row,
)
from experiments.evaluate_a15_2_syncg_scene_holdout import (
    CONDITIONS,
    PROJECTIVE_CONDITIONS,
    _SharedPixelDataset,
)
from experiments.run_cagh_v5_plain_paper_batch import load_manifest
from experiments.train_remst_block_pilot import load_remst_block_checkpoint
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    _loader,
)


PROTOCOL: Final[str] = "remst_block_syncg_pilot_evaluation_v1"
DEFAULT_OLD_REMST_EVALUATION: Final[Path] = Path(
    "C:/pointer_read/sgca_multiview_pilot_v1/"
    "a15_2_mett_direct_scalar_seed20262022_epoch5_"
    "syncg_verified_posterior_fp32.json"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_reference(
    reference_path: Path,
) -> tuple[
    Mapping[str, Any],
    dict[tuple[str, str], Mapping[str, Any]],
    tuple[str, ...],
    dict[str, tuple[float, str]],
]:
    source = Path(reference_path).resolve()
    reference = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(reference, Mapping), "pilot reference is malformed")
    validate_external_reference(reference)
    rows = reference.get("per_sample_condition")
    _require(isinstance(rows, list) and bool(rows), "pilot reference rows are missing")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    targets: dict[str, tuple[float, str]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "pilot reference row is malformed")
        key = (str(row["sample_id"]), str(row["condition"]))
        _require(key not in indexed, "pilot reference row keys repeat")
        validate_external_reference_row(row, key=key)
        indexed[key] = row
        target = (float(row["normalized_target"]), str(row["scene_stem"]))
        if key[0] in targets:
            _require(targets[key[0]] == target, "pilot reference target differs")
        else:
            targets[key[0]] = target
    ordered_ids = tuple(dict.fromkeys(str(row["sample_id"]) for row in rows))
    return reference, indexed, ordered_ids, targets


def _load_old_remst(
    evaluation_path: Path | None,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]], str | None]:
    if evaluation_path is None:
        return {}, None
    source = Path(evaluation_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(payload, Mapping), "old ReMST evaluation is malformed")
    rows = payload.get("per_sample_condition")
    _require(isinstance(rows, list) and bool(rows), "old ReMST rows are missing")
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "old ReMST row is malformed")
        key = (str(row["sample_id"]), str(row["condition"]))
        _require(key not in indexed, "old ReMST row keys repeat")
        indexed[key] = row
    return indexed, str(source)


def _add_old_remst_comparison(
    summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    condition_sets = {
        **{condition: (condition,) for condition in CONDITIONS},
        "projective_pooled": PROJECTIVE_CONDITIONS,
        "all_conditions": CONDITIONS,
    }
    for label, selected in condition_sets.items():
        subset = [row for row in rows if str(row["condition"]) in selected]
        if not subset or not all("old_remst" in row for row in subset):
            continue
        new_errors = [float(row["mett"]["absolute_error"]) for row in subset]
        old_errors = [float(row["old_remst"]["absolute_error"]) for row in subset]
        old_metrics = _metrics(old_errors)
        summary[label]["candidate"]["old_remst"] = old_metrics
        summary[label]["remst_block_minus_old_remst_nmae"] = (
            summary[label]["candidate"]["mett"]["nmae"] - old_metrics["nmae"]
        )
        summary[label]["remst_block_improvement_fraction_vs_old_remst"] = (
            statistics.fmean(
                float(new < old) for new, old in zip(new_errors, old_errors, strict=True)
            )
        )
        summary[label]["remst_block_negative_fraction_vs_old_remst"] = (
            statistics.fmean(
                float(new > old) for new, old in zip(new_errors, old_errors, strict=True)
            )
        )


def _add_remstnet_publication_aliases(
    summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Expose the paper name while retaining legacy METT compatibility keys."""

    for row in rows:
        if isinstance(row, dict):
            row["remstnet"] = dict(row["mett"])
    for values in summary.values():
        candidate = values.get("candidate")
        if isinstance(candidate, dict) and "mett" in candidate:
            candidate["remstnet"] = dict(candidate["mett"])
        posterior = values.get("posterior")
        if isinstance(posterior, dict) and "mett" in posterior:
            posterior["remstnet"] = dict(posterior["mett"])
        macro = values.get("macro_scene_nmae")
        if isinstance(macro, dict) and "mett" in macro:
            macro["remstnet"] = float(macro["mett"])
        for name in tuple(values):
            if name.startswith("mett_"):
                values[f"remstnet_{name.removeprefix('mett_')}"] = values[name]


def evaluate_remst_block(
    *,
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    reference_path: Path = DEFAULT_REFERENCE,
    old_remst_evaluation_path: Path | None = DEFAULT_OLD_REMST_EVALUATION,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
) -> dict[str, Any]:
    _require(workers >= 0 and batch_size >= 1, "pilot loader sizes are invalid")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"pilot output already exists: {output}")
    _reference, reference_by_key, ordered_ids, targets = _load_reference(
        reference_path
    )
    old_by_key, old_source = _load_old_remst(old_remst_evaluation_path)
    if old_by_key:
        _require(
            set(old_by_key) == set(reference_by_key),
            "old ReMST and pilot reference rosters differ",
        )

    manifest_rows = load_manifest(manifest_path)
    by_id = {row.sample_id: row for row in manifest_rows}
    _require(set(by_id) == set(ordered_ids), "pilot manifest roster differs")
    ordered_manifest = tuple(by_id[sample_id] for sample_id in ordered_ids)
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    model, model_metadata = load_remst_block_checkpoint(
        checkpoint_path, device=device
    )
    model.eval()
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    started = time.perf_counter()
    candidate_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, condition in enumerate(CONDITIONS):
            dataset = _SharedPixelDataset(
                ordered_manifest, targets, condition=condition
            )
            loader = _loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=20260818 + condition_index,
                cuda=device.type == "cuda",
            )
            for raw_batch in loader:
                raw = raw_batch["original_view"].to(device)
                sarn = raw_batch["sarn_view"].to(device)
                support = raw_batch["sarn_support_mask"].to(device)
                active = raw_batch["sarn_active"].to(device).bool()
                homography = raw_batch["raw_to_sarn_homography"].to(device)
                target = raw_batch["target"].float()
                effective_active = active & (condition != "clean")
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    prediction = model(
                        raw,
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
                posterior_values = {
                    "raw_anchor": prediction["raw_anchor_posterior"],
                    "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                    "mett": prediction["progress_posterior"],
                }
                diagnostics = {
                    name: posterior_batch_diagnostics(value, target)
                    for name, value in posterior_values.items()
                }
                relation = prediction["relation_available"].cpu().bool()
                shift8 = prediction["scale8_moment_shift"].float().cpu()
                shift16 = prediction["scale16_moment_shift"].float().cpu()
                shift_context = prediction["context_moment_shift"].float().cpu()
                stage_allocation = prediction.get("stage_moment_allocation")
                if isinstance(stage_allocation, torch.Tensor):
                    stage_allocation = stage_allocation.float().cpu()
                budget_gain = prediction.get("moment_budget_gain")
                if isinstance(budget_gain, torch.Tensor):
                    budget_gain = budget_gain.float().cpu()
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                scenes = tuple(str(value) for value in raw_batch["scene_stem"])
                for row_index, sample_id in enumerate(ids):
                    key = (sample_id, condition)
                    _require(key in reference_by_key, "pilot reference row is missing")
                    reference_row = reference_by_key[key]
                    expected_target = float(reference_row["normalized_target"])
                    _require(
                        abs(float(target[row_index]) - expected_target) <= 1.0e-6,
                        "pilot target differs",
                    )
                    row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "scene_stem": scenes[row_index],
                        "condition": condition,
                        "normalized_target": expected_target,
                        "relation_available": bool(relation[row_index]),
                        "external": reference_row["external"],
                        "stage_moment_shifts": {
                            "scale8": float(shift8[row_index]),
                            "scale16": float(shift16[row_index]),
                            "context": float(shift_context[row_index]),
                        },
                    }
                    if isinstance(stage_allocation, torch.Tensor):
                        row["stage_moment_allocation"] = {
                            "scale8": float(stage_allocation[row_index, 0]),
                            "scale16": float(stage_allocation[row_index, 1]),
                            "context": float(stage_allocation[row_index, 2]),
                        }
                    if isinstance(budget_gain, torch.Tensor):
                        row["moment_budget_gain"] = float(
                            budget_gain[row_index]
                        )
                    for method, tensor in values.items():
                        scalar = float(tensor[row_index])
                        _require(
                            math.isfinite(scalar) and 0.0 <= scalar <= 1.0,
                            "pilot prediction is invalid",
                        )
                        row[method] = {
                            "prediction": scalar,
                            "absolute_error": abs(scalar - expected_target),
                        }
                        if method in diagnostics:
                            row[method]["posterior"] = {
                                name: (
                                    bool(value[row_index])
                                    if value.dtype == torch.bool
                                    else float(value[row_index])
                                )
                                for name, value in diagnostics[method].items()
                            }
                    if old_by_key:
                        old = old_by_key[key]["mett"]
                        row["old_remst"] = {
                            "prediction": float(old["prediction"]),
                            "absolute_error": float(old["absolute_error"]),
                        }
                    candidate_rows.append(row)

    _require(
        len(candidate_rows) == len(ordered_ids) * len(CONDITIONS),
        "pilot Cartesian row count differs",
    )
    summary = _summarize(candidate_rows)
    _add_old_remst_comparison(summary, candidate_rows)
    _add_remstnet_publication_aliases(summary, candidate_rows)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "pilot_complete",
        "scope": {
            "development_cohort": True,
            "single_seed_architecture_pilot": True,
            "training_or_adaptation_during_evaluation": False,
            "prediction_dependent_routing": False,
            "candidate_machine_key": "remstnet",
            "legacy_candidate_machine_key": "mett",
            "candidate_display_name": (
                "ReMSTNet adaptive pilot"
                if "Adaptive" in str(model_metadata.get("architecture", ""))
                else (
                    "ReMSTNet coordinated pilot"
                    if "Coordinated"
                    in str(model_metadata.get("architecture", ""))
                    else "ReMSTNet block pilot"
                )
            ),
            "inference_precision": (
                str(autocast_dtype).removeprefix("torch.")
                if autocast_enabled
                else "float32"
            ),
        },
        "model": model_metadata,
        "data": {
            "manifest": str(Path(manifest_path).resolve()),
            "reference": str(Path(reference_path).resolve()),
            "old_remst_evaluation": old_source,
            "samples": len(ordered_ids),
            "conditions": list(CONDITIONS),
            "rows": len(candidate_rows),
        },
        "summary": summary,
        "evaluation_elapsed_seconds": float(time.perf_counter() - started),
        "per_sample_condition": candidate_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "pilot_complete",
        "output": str(output),
        "all_conditions": summary["all_conditions"],
        "projective_pooled": summary["projective_pooled"],
        "evaluation_elapsed_seconds": result["evaluation_elapsed_seconds"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--old-remst-evaluation",
        type=Path,
        default=DEFAULT_OLD_REMST_EVALUATION,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_remst_block(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        reference_path=args.reference,
        old_remst_evaluation_path=args.old_remst_evaluation,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "evaluate_remst_block"]
