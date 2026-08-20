"""Run the frozen ReMSTNet-v3 over the established 18-condition stress sweep."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

import torch

from experiments import evaluate_a15_2_mett_stress_sweep as mett_stress
from experiments.evaluate_a15_2_mett_syncg import posterior_batch_diagnostics
from experiments.evaluate_remstnet_real_domains import _validate_full_model
from experiments.train_remstnet import load_remstnet_checkpoint
from remstnet.model import CoordinatedReMSTNet


PROTOCOL: Final[str] = "remstnet_v3_syncg_stress_sweep_v1"


class ReMSTNetStressError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReMSTNetStressError(message)


def evaluate_stress_sweep(
    *,
    manifest_path: Path,
    labels_path: Path,
    split_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
) -> dict[str, Any]:
    _require(workers >= 0 and batch_size >= 1, "stress evaluation sizes invalid")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"stress output exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "stress device invalid")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA unavailable")
    torch.manual_seed(mett_stress.EVALUATION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(mett_stress.EVALUATION_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    targets = mett_stress.load_scene_targets(labels_path, split_path)
    ordered_sample_ids = tuple(targets)
    manifest_rows = mett_stress.load_manifest(manifest_path)
    by_id = {row.sample_id: row for row in manifest_rows}
    _require(len(by_id) == len(manifest_rows), "stress manifest IDs repeat")
    _require(set(by_id) == set(ordered_sample_ids), "stress manifest/target rosters differ")
    ordered_manifest = tuple(by_id[sample_id] for sample_id in ordered_sample_ids)

    model, metadata = load_remstnet_checkpoint(checkpoint_path, device=device)
    _require(isinstance(model, CoordinatedReMSTNet), "stress checkpoint is not coordinated ReMSTNet")
    source_seed = _validate_full_model(model, metadata)
    model.eval()
    autocast_enabled = device.type == "cuda" and bool(use_amp)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, spec in enumerate(mett_stress.STRESS_SPECS.values()):
            dataset = mett_stress._StressDataset(ordered_manifest, targets, spec=spec)
            loader = mett_stress._loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=mett_stress.EVALUATION_SEED + condition_index,
                cuda=device.type == "cuda",
            )
            condition_ids: list[str] = []
            for raw_batch in loader:
                original = raw_batch["original_view"].to(device)
                sarn = raw_batch["sarn_view"].to(device)
                support = raw_batch["sarn_support_mask"].to(device)
                active = raw_batch["sarn_active"].to(device).bool()
                homography = raw_batch["raw_to_sarn_homography"].to(device)
                target = raw_batch["target"].float()
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    prediction = model(
                        original,
                        sarn,
                        support,
                        sarn_active=active,
                        raw_to_sarn_homography=homography,
                    )
                values = {
                    "raw_anchor": prediction["raw_anchor_mean"].float().cpu(),
                    "sarn_endpoint": prediction["sarn_endpoint_mean"].float().cpu(),
                    "mett": prediction["mean"].float().cpu(),
                }
                diagnostics = {
                    method: posterior_batch_diagnostics(posterior, target)
                    for method, posterior in {
                        "raw_anchor": prediction["raw_anchor_posterior"],
                        "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                        "mett": prediction["progress_posterior"],
                    }.items()
                }
                relation = prediction["relation_available"].bool().cpu()
                ids = tuple(str(value) for value in raw_batch["sample_id"])
                scenes = tuple(str(value) for value in raw_batch["scene_stem"])
                seeds = raw_batch["stress_sample_seed"].long().cpu()
                active_cpu = active.cpu()
                condition_ids.extend(ids)
                for row_index, sample_id in enumerate(ids):
                    expected_target, expected_scene = targets[sample_id]
                    observed_target = float(target[row_index])
                    _require(abs(expected_target - observed_target) <= 1.0e-6, "stress target differs")
                    _require(scenes[row_index] == expected_scene, "stress scene differs")
                    row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "scene_stem": expected_scene,
                        "condition": spec.name,
                        "family": spec.family,
                        "severity": float(spec.severity),
                        "normalized_target": float(expected_target),
                        "stress_sample_seed": int(seeds[row_index]),
                        "sarn_applied": bool(active_cpu[row_index]),
                        "relation_available": bool(relation[row_index]),
                    }
                    for method, tensor in values.items():
                        scalar = float(tensor[row_index])
                        _require(math.isfinite(scalar) and 0.0 <= scalar <= 1.0, "stress prediction invalid")
                        row[method] = {
                            "prediction": scalar,
                            "absolute_error": abs(scalar - expected_target),
                            "posterior": {
                                name: bool(value[row_index]) if value.dtype == torch.bool else float(value[row_index])
                                for name, value in diagnostics[method].items()
                            },
                        }
                    row["remstnet"] = row["mett"]
                    rows.append(row)
            _require(tuple(condition_ids) == ordered_sample_ids, f"stress roster differs: {spec.name}")

    condition_names = tuple(mett_stress.STRESS_SPECS)
    mett_stress._validate_cartesian_roster(
        rows,
        ordered_sample_ids=ordered_sample_ids,
        condition_names=condition_names,
    )
    condition_summary = mett_stress.summarize_conditions(rows)
    family_summary = mett_stress.summarize_family_curves(condition_summary)
    perspective_summary = mett_stress.summarize_perspective_curve(condition_summary)
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "scope": {
            "development_stress_characterisation": True,
            "training_or_adaptation_during_evaluation": False,
            "sample_router_or_external_prediction_fusion": False,
            "fresh_sarn_after_every_stress": True,
            "paired_clean_and_stress_roster": True,
            "candidate_machine_key": "remstnet",
            "legacy_candidate_machine_key": "mett",
            "inference_precision": str(autocast_dtype).removeprefix("torch.") if autocast_enabled else "float32",
            "posterior_interpretation": (
                "fixed-scale sensitivity diagnostic; not a learned or calibrated "
                "uncertainty claim"
            ),
        },
        "model": metadata,
        "source_seed": source_seed,
        "data": {
            "manifest": str(Path(manifest_path).resolve()),
            "labels": str(Path(labels_path).resolve()),
            "split": str(Path(split_path).resolve()),
            "samples": len(ordered_sample_ids),
            "scenes": len({scene for _target, scene in targets.values()}),
            "conditions": list(condition_names),
            "condition_count": len(condition_names),
            "rows": len(rows),
            "stress_protocol": mett_stress.STRESS_PROTOCOL,
            "stress_seed": mett_stress.STRESS_SEED,
            "stress_specs": {name: asdict(spec) for name, spec in mett_stress.STRESS_SPECS.items()},
        },
        "summary": {
            "conditions": condition_summary,
            "families": family_summary,
            "perspective_continuous_curve": perspective_summary,
        },
        "per_sample_condition": rows,
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
    parser.add_argument("--manifest", type=Path, default=mett_stress.DEFAULT_MANIFEST)
    parser.add_argument("--labels", type=Path, default=mett_stress.DEFAULT_LABELS)
    parser.add_argument("--split", type=Path, default=mett_stress.DEFAULT_SPLIT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_stress_sweep(
        manifest_path=args.manifest,
        labels_path=args.labels,
        split_path=args.split,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "source_seed": result["source_seed"], "conditions": result["data"]["condition_count"], "rows": result["data"]["rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "evaluate_stress_sweep"]
