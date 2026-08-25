"""Run the frozen R²MT-Net downstream evidence replays.

This is a thin compatibility driver around the established real-domain, VDN,
OCR, natural-repeat, and CUDA-efficiency evaluators.  It does not train or
select a model.  Historical evaluator machine keys such as ``mett`` and
``remstnet`` are retained where their scorers require them and are disclosed in
the resulting artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SEEDS: Final[tuple[int, ...]] = (20262020, 20262021, 20262022)
R2MT_ADAPTIVE_STRENGTH: Final[float] = 0.5
R2MT_FUSION_MODE: Final[str] = "single_exact_transport"
BOOTSTRAP_REPLICATES: Final[int] = 20_000
STAGE_TIMEOUT_SECONDS: Final[int] = 30 * 60
REAL_PROTOCOL: Final[str] = "r2mt_real_domain_frozen_evaluation_v1"
REAL_SUMMARY_PROTOCOL: Final[str] = "r2mt_real_domain_three_seed_summary_v1"
VDN_PROTOCOL: Final[str] = "r2mt_vdn_double_holdout_intersection_v1"
OCR_PROTOCOL: Final[str] = "r2mt_ocr_end_to_end_deployment_repro_v1"
REPEAT_PROTOCOL: Final[str] = "r2mt_natural_repeat_stability_v1"
EFFICIENCY_PROTOCOL: Final[str] = "r2mt_matched_cuda_efficiency_v1"
FACTORIAL_ROOT: Final[Path] = Path(
    r"C:\pointer_read\paper_syncg_only_retrain_v1\factorial"
)
VDN_AUTOMATIC: Final[Path] = Path(
    r"C:\pointer_read\cagh_v5_plain_paper_8x6_v1\predictions"
    r"\vdn_official200_terminal_seed20.jsonl"
)
VDN_ORACLE: Final[Path] = Path(
    r"C:\pointer_read\cagh_v5_plain_paper_8x6_v1\predictions"
    r"\vdn_oracle_reference_component.jsonl"
)
REPEAT_COHORT_REFERENCE: Final[Path] = Path(
    r"C:\pointer_read\sgca_multiview_pilot_v1"
    r"\a15_2_mett_direct_scalar_natural_repeat_stability.json"
)


class R2MTReplayError(RuntimeError):
    """A replay input, adapter, subprocess, or result is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise R2MTReplayError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(source.is_file(), f"JSON artifact is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON artifact is not an object: {source}")
    return value


def _checkpoint(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "remst_resnet18_risk_arbitration_probe"
        / f"seed_{seed}"
        / "terminal.pt"
    ).resolve()


def _main_evaluation(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts"
        / "runs"
        / "r2mt_final_conservative"
        / f"seed_{seed}"
        / "six_condition_results.json"
    ).resolve()


def _publication_identity(_expected_variant: str = "direct_scalar") -> dict[str, str]:
    from experiments.r2mt_net import publication_model_identity

    return publication_model_identity()


def _configured_loader(checkpoint: Path, *, device: Any):
    from experiments.train_r2mt_net import (
        load_r2mt_net_checkpoint,
    )

    anchor, model, metadata = load_r2mt_net_checkpoint(
        Path(checkpoint).resolve(), device=device
    )
    model.adaptive_strength = R2MT_ADAPTIVE_STRENGTH
    model.fusion_mode = R2MT_FUSION_MODE
    metadata = dict(metadata)
    construction = dict(metadata["construction"])
    construction.update(
        {
            "evaluation_adaptive_strength": R2MT_ADAPTIVE_STRENGTH,
            "evaluation_fusion_mode": R2MT_FUSION_MODE,
        }
    )
    seed = int(metadata["seed"])
    metadata.update(
        {
            "construction": construction,
            # Compatibility shape consumed by the established METT evaluators.
            "source_anchor": {
                "source_seed": seed,
                "source_protocol": metadata["protocol"],
                "source_architecture": metadata["architecture"],
                "source_checkpoint_selection": metadata["checkpoint_selection"],
                "source_epochs": int(metadata["epochs"]),
            },
            "training_seeds": {"initialization": seed, "sample_order": seed},
        }
    )
    return anchor, model, metadata


def _validate_metadata(
    metadata: Mapping[str, Any], *, expected_variant: str
) -> dict[str, Any]:
    from experiments.evaluate_r2mt_net_syncg import (
        validate_metadata,
    )

    return validate_metadata(metadata, expected_variant=expected_variant)


def _direct_resnet18_baseline_runs(dataset: Any, *, prediction_root: Path):
    """Load the paired Direct-ResNet18 ledgers under legacy baseline keys."""

    from experiments import evaluate_paper_syncg_only_factorial as factorial
    from experiments import score_db_gar18_factorial_3seed as factorial_score
    from experiments import evaluate_a15_2_mett_real_domains as real_eval

    sample_ids = factorial_score.load_validation_ids(dataset.roster)
    _require(
        len(sample_ids) == dataset.expected_samples,
        f"{dataset.slug}: sample count differs",
    )
    targets_tuple = factorial_score._load_targets_with_group_source(
        dataset.labels,
        sample_ids,
        group_source=dataset.group_source,
    )
    _require(
        len({target.group_id for target in targets_tuple}) == dataset.expected_groups,
        f"{dataset.slug}: group count differs",
    )
    targets = {target.sample_id: target for target in targets_tuple}
    manifest_by_id, manifest_audit = factorial_score._load_plain_manifest(
        dataset.manifest,
        sample_ids=sample_ids,
    )
    runs: dict[str, dict[int, Any]] = {"raw": {}, "sarn_v2": {}}
    inputs: dict[str, Any] = {
        "prediction_root": str(Path(prediction_root).resolve()),
        "comparator": "Direct-ResNet18",
        "legacy_machine_key": "efficientnet_b0",
    }
    for variant in runs:
        inputs[variant] = {}
        for seed in SEEDS:
            artifacts = factorial.prediction_artifacts(
                prediction_root,
                cell="00",
                seed=seed,
                dataset=dataset,
                variant=variant,
            )
            binding: dict[str, str] = {
                "path": str(artifacts["predictions"]),
                "method": factorial.method_id(
                    cell="00", seed=seed, variant=variant
                ),
                "protocol": (
                    factorial.PROTOCOL
                    if variant == "raw"
                    else factorial.sarn_v2.PROTOCOL
                ),
            }
            if variant == "sarn_v2":
                binding["sidecar"] = str(artifacts["sidecar"])
            run = factorial_score._load_prediction_run(
                binding,
                label=f"{dataset.slug}.{variant}.{seed}",
                base_dir=Path(prediction_root).resolve(),
                require_sidecar=variant == "sarn_v2",
                robustness_seed=factorial.ROBUSTNESS_SEED,
                targets=targets,
                conditions=real_eval.CONDITIONS,
                manifest_by_id=manifest_by_id,
            )
            runs[variant][seed] = run
            inputs[variant][str(seed)] = {
                "predictions": str(run.path),
                "predictions_sha256": run.sha256,
                "sidecar": str(run.sidecar_path) if run.sidecar_path else None,
                "sidecar_sha256": run.sidecar_sha256,
                "failure_codes": dict(run.failure_codes),
            }
    return targets_tuple, targets, runs, {"manifest": manifest_audit, **inputs}


def _evaluate_r2mt_real_dataset(
    dataset: Any,
    *,
    anchor: Any,
    correction: Any,
    source_seed: int,
    prediction_root: Path,
    device: Any,
    workers: int,
    batch_size: int,
    bootstrap_replicates: int,
    include_posterior_diagnostics: bool,
    dataset_index: int,
) -> dict[str, Any]:
    """Evaluate R²MT while treating Direct-ResNet18 as a comparator, not an anchor replay."""

    import torch
    from experiments import evaluate_a15_2_mett_real_domains as real_eval

    targets_tuple, targets, runs, baseline_inputs = real_eval._load_baseline_runs(
        dataset, prediction_root=prediction_root
    )
    target_for_dataset = {
        target.sample_id: (float(target.normalized_target), str(target.group_id))
        for target in targets_tuple
    }
    manifest_rows = real_eval.load_manifest(dataset.manifest)
    manifest_by_id = {row.sample_id: row for row in manifest_rows}
    _require(
        tuple(manifest_by_id) == tuple(target.sample_id for target in targets_tuple),
        f"{dataset.slug}: model manifest order differs from roster",
    )
    ordered_manifest = tuple(
        manifest_by_id[target.sample_id] for target in targets_tuple
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for condition_index, condition in enumerate(real_eval.CONDITIONS):
            shared_dataset = real_eval._SharedPixelDataset(
                ordered_manifest, target_for_dataset, condition=condition
            )
            loader = real_eval._loader(
                shared_dataset,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=real_eval.BOOTSTRAP_SEED + dataset_index * 100 + condition_index,
                cuda=device.type == "cuda",
            )
            for raw_batch in loader:
                real_eval._validate_batch_pixel_identity(
                    raw_batch, condition=condition, runs=runs
                )
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
                    endpoints = real_eval.frozen_twin_endpoint_forward(
                        anchor, original, sarn
                    )
                    prediction = real_eval.forward_a15_correction(
                        correction,
                        {
                            "raw_posterior": endpoints["raw_posterior"],
                            "sarn_posterior": endpoints["sarn_posterior"],
                            "raw_mean": endpoints["raw_mean"],
                            "sarn_mean": endpoints["sarn_mean"],
                            "raw_features": endpoints["raw_features"],
                            "sarn_features": endpoints["sarn_features"],
                            "sarn_support_mask": support,
                            "sarn_active": effective_active,
                            "raw_to_sarn_homography": homography,
                        },
                        endpoint_null=False,
                    )
                    values = {
                        "raw_anchor": prediction["raw_anchor_mean"].float().cpu(),
                        "sarn_endpoint": prediction["sarn_endpoint_mean"].float().cpu(),
                        "tangent_base": prediction["tangent_base_mean"].float().cpu(),
                        "mett": prediction["mean"].float().cpu(),
                    }
                    diagnostics: dict[str, dict[str, Any]] = {}
                    if include_posterior_diagnostics:
                        posterior_values = {
                            "raw_anchor": prediction["raw_anchor_posterior"],
                            "sarn_endpoint": prediction["sarn_endpoint_posterior"],
                            "mett": prediction["progress_posterior"],
                        }
                        diagnostics = {
                            method: real_eval.posterior_batch_diagnostics(
                                value, target_tensor
                            )
                            for method, value in posterior_values.items()
                        }
                    relation = prediction["relation_available"].bool().cpu()
                    inference_failure: str | None = None
                except Exception as exc:  # full-denominator accounting
                    values = {}
                    diagnostics = {}
                    relation = torch.zeros(len(ids), dtype=torch.bool)
                    inference_failure = f"model_exception:{type(exc).__name__}"

                for row_index, (sample_id, group_id) in enumerate(
                    zip(ids, groups, strict=True)
                ):
                    target = float(target_tensor[row_index])
                    expected = float(targets[sample_id].normalized_target)
                    _require(
                        abs(target - expected) <= 1.0e-6,
                        f"{sample_id}: regenerated target differs",
                    )
                    candidate: dict[str, Any] = {}
                    for method in real_eval.CANDIDATE_METHODS:
                        value = (
                            float(values[method][row_index])
                            if inference_failure is None
                            else None
                        )
                        candidate[method] = real_eval._record(
                            value, expected, passed=inference_failure is None
                        )
                        if inference_failure is not None:
                            candidate[method]["failure_code"] = inference_failure
                        if method in diagnostics:
                            candidate[method]["posterior"] = {
                                name: (
                                    bool(tensor[row_index])
                                    if tensor.dtype == torch.bool
                                    else float(tensor[row_index])
                                )
                                for name, tensor in diagnostics[method].items()
                            }
                    key = (sample_id, condition)
                    external = {
                        variant: {
                            str(seed): real_eval._external_record(
                                runs[variant][seed], key=key, target=expected
                            )
                            for seed in SEEDS
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
                            # Legacy scorer key; this block is Direct-ResNet18 here.
                            "efficientnet_b0": external,
                        }
                    )

    _require(
        len(rows) == dataset.expected_samples * len(real_eval.CONDITIONS),
        f"{dataset.slug}: Cartesian output count differs",
    )
    summary = real_eval.summarize_rows(
        rows,
        source_seed=source_seed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=real_eval.BOOTSTRAP_SEED + dataset_index * 1000,
    )
    raw_deltas: list[float] = []
    sarn_deltas: list[float] = []
    raw_status_mismatches = 0
    sarn_status_mismatches = 0
    for row in rows:
        raw_internal = row["candidate"]["raw_anchor"]
        sarn_internal = row["candidate"]["sarn_endpoint"]
        raw_external = row["efficientnet_b0"]["raw"][str(source_seed)]
        sarn_external = row["efficientnet_b0"]["sarn_v2"][str(source_seed)]
        raw_status_mismatches += int(raw_internal["status"] != raw_external["status"])
        sarn_status_mismatches += int(
            sarn_internal["status"] != sarn_external["status"]
        )
        if raw_internal["status"] == raw_external["status"] == "pass":
            raw_deltas.append(
                abs(float(raw_internal["prediction"]) - float(raw_external["prediction"]))
            )
        if sarn_internal["status"] == sarn_external["status"] == "pass":
            sarn_deltas.append(
                abs(
                    float(sarn_internal["prediction"])
                    - float(sarn_external["prediction"])
                )
            )
    comparison_diagnostic = {
        "audit_type": "different_model_endpoint_comparison_diagnostic",
        "identity_expected": False,
        "reason": (
            "R²MT-Net uses a subsequently refined ResNet18 anchor; Direct-ResNet18 "
            "is a paired external comparator, not the same checkpoint replay."
        ),
        "same_condition_pixels_verified": True,
        "source_seed": source_seed,
        "rows": len(rows),
        "raw_compared_rows": len(raw_deltas),
        "raw_status_mismatch_rows": raw_status_mismatches,
        "raw_mean_absolute_prediction_delta": (
            sum(raw_deltas) / len(raw_deltas) if raw_deltas else None
        ),
        "raw_max_absolute_prediction_delta": max(raw_deltas, default=None),
        "sarn_compared_rows": len(sarn_deltas),
        "sarn_status_mismatch_rows": sarn_status_mismatches,
        "sarn_mean_absolute_prediction_delta": (
            sum(sarn_deltas) / len(sarn_deltas) if sarn_deltas else None
        ),
        "sarn_max_absolute_prediction_delta": max(sarn_deltas, default=None),
    }
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
        "endpoint_replay_audit": comparison_diagnostic,
        "summary": summary,
        "per_sample_condition": rows,
    }


def _postprocess_real(path: Path) -> None:
    value = _read_json(path)
    value["publication_model"] = _publication_identity()
    value["scope"].update(
        {
            "internal_representation_conditioned_risk_weighting": True,
            "external_prediction_fusion": False,
            "sample_router_or_prediction_fusion": True,
        }
    )
    value["compatibility_machine_keys"] = {
        "candidate.mett": "R²MT-Net output",
        "efficientnet_b0": (
            "Direct-ResNet18 Raw/SARN-v2 ledgers loaded by the R²MT adapter"
        ),
    }
    value["external_comparator"] = "same-source-seed SARN-v2 + Direct-ResNet18"
    _write_json(path, value)


def stage_real(output_root: Path, seed: int, device_name: str) -> None:
    from experiments import evaluate_a15_2_mett_real_domains as real_eval

    _require(seed in SEEDS, f"unexpected real-domain seed: {seed}")
    real_eval.load_mett_models = _configured_loader
    real_eval.validate_mett_variant_metadata = _validate_metadata
    real_eval.publication_model_identity = _publication_identity
    real_eval.PROTOCOL = REAL_PROTOCOL
    real_eval._load_baseline_runs = _direct_resnet18_baseline_runs
    real_eval._evaluate_dataset = _evaluate_r2mt_real_dataset
    output = Path(output_root) / "real" / f"seed_{seed}.json"
    result = real_eval.evaluate_real_domains(
        checkpoint_path=_checkpoint(seed),
        output_path=output,
        prediction_root=FACTORIAL_ROOT,
        device_name=device_name,
        workers=4,
        batch_size=len(real_eval.CONDITIONS),
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        include_posterior_diagnostics=False,
        expected_variant="direct_scalar",
    )
    _postprocess_real(output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


def _load_r2mt_real_evaluation(path: Path) -> dict[str, Any]:
    from experiments import evaluate_a15_2_mett_real_domains as shared

    source = Path(path).resolve()
    payload = _read_json(source)
    scope = payload.get("scope")
    _require(
        payload.get("protocol") == REAL_PROTOCOL
        and payload.get("status") == "complete"
        and isinstance(scope, Mapping)
        and scope.get("inference_precision") == "float32"
        and scope.get("training_or_adaptation_during_evaluation") is False
        and scope.get("internal_representation_conditioned_risk_weighting") is True
        and scope.get("external_prediction_fusion") is False,
        f"R²MT real-domain protocol or scope differs: {source}",
    )
    model = payload.get("model")
    _require(isinstance(model, Mapping), f"model metadata is missing: {source}")
    _validate_metadata(model, expected_variant="direct_scalar")
    seed = int(model.get("seed", -1))
    _require(
        seed in SEEDS and int(payload.get("source_seed", -1)) == seed,
        f"R²MT source seed differs: {source}",
    )
    identity = {
        "architecture": model.get("architecture"),
        "protocol": model.get("protocol"),
        "construction": dict(model.get("construction", {})),
        "parameter_counts": dict(model.get("parameter_counts", {})),
    }
    documents = payload.get("datasets")
    _require(
        isinstance(documents, Mapping)
        and set(documents) == set(shared.REAL_DATASET_KEYS),
        f"four-domain roster differs: {source}",
    )
    datasets: dict[str, Any] = {}
    for dataset_name in shared.REAL_DATASET_KEYS:
        document = documents[dataset_name]
        rows = document.get("per_sample_condition")
        audit = document.get("endpoint_replay_audit")
        _require(
            isinstance(rows, list)
            and bool(rows)
            and isinstance(audit, Mapping)
            and audit.get("identity_expected") is False
            and audit.get("same_condition_pixels_verified") is True
            and int(audit.get("source_seed", -1)) == seed
            and int(audit.get("rows", -1)) == len(rows),
            f"R²MT real-domain comparison evidence differs: {source}/{dataset_name}",
        )
        indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
        identities: dict[tuple[str, str], tuple[float, str]] = {}
        for row_index, row in enumerate(rows):
            _require(isinstance(row, Mapping), f"invalid row {dataset_name}/{row_index}")
            key = (str(row.get("sample_id", "")), str(row.get("condition", "")))
            _require(
                bool(key[0]) and key[1] in shared.CONDITIONS and key not in indexed,
                f"sample-condition key differs: {dataset_name}/{key}",
            )
            target = float(row["normalized_target"])
            group = str(row.get("group_id", ""))
            candidate = row.get("candidate")
            r2mt = candidate.get("mett") if isinstance(candidate, Mapping) else None
            _require(
                bool(group)
                and isinstance(r2mt, Mapping)
                and r2mt.get("status") == "pass",
                f"R²MT record is incomplete: {dataset_name}/{key}",
            )
            prediction = float(r2mt["prediction"])
            error = float(r2mt["absolute_error"])
            _require(
                math.isfinite(target)
                and math.isfinite(prediction)
                and abs(error - abs(prediction - target)) <= 2.0e-6,
                f"R²MT error differs: {dataset_name}/{key}",
            )
            external = row.get("efficientnet_b0")
            sarn = external.get("sarn_v2") if isinstance(external, Mapping) else None
            _require(
                isinstance(sarn, Mapping)
                and set(sarn) == {str(value) for value in SEEDS},
                f"Direct-ResNet18 seed roster differs: {dataset_name}/{key}",
            )
            for external_seed in SEEDS:
                record = sarn[str(external_seed)]
                _require(
                    isinstance(record, Mapping) and record.get("status") == "pass",
                    f"Direct-ResNet18 row failed: {dataset_name}/{key}/{external_seed}",
                )
                external_prediction = float(record["prediction"])
                external_error = float(record["absolute_error"])
                _require(
                    abs(external_error - abs(external_prediction - target)) <= 2.0e-6,
                    f"Direct-ResNet18 error differs: {dataset_name}/{key}",
                )
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
        "training_seeds": {"initialization": seed, "sample_order": seed},
        "datasets": datasets,
    }


def _decorate_real_summary(value: dict[str, Any]) -> dict[str, Any]:
    value["protocol"] = REAL_SUMMARY_PROTOCOL
    value["publication_model"] = _publication_identity()
    value["scope"].update(
        {
            "external_comparator": "same-source-seed SARN-v2 + Direct-ResNet18",
            "internal_representation_conditioned_risk_weighting": True,
            "external_prediction_fusion": False,
        }
    )
    value["compatibility_machine_keys"] = {
        "remstnet": "R²MT-Net",
        "sarn_v2_efficientnet_b0": "SARN-v2 + Direct-ResNet18",
    }
    return value


def stage_real_summary(output_root: Path) -> None:
    from experiments import summarize_remstnet_real_domains as summary_module

    summary_module._load_evaluation = _load_r2mt_real_evaluation
    summary_module.PROTOCOL = REAL_SUMMARY_PROTOCOL
    paths = [Path(output_root) / "real" / f"seed_{seed}.json" for seed in SEEDS]
    result = summary_module.summarize_evaluations(
        paths,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=20260818,
    )
    result = _decorate_real_summary(result)
    output = Path(output_root) / "real" / "three_seed_summary.json"
    _require(not output.exists(), f"real summary already exists: {output}")
    _write_json(output, result)
    print(json.dumps({"status": "complete", "output": str(output.resolve())}), flush=True)


def _load_r2mt_vdn_evaluation(path: Path) -> dict[str, Any]:
    from experiments import summarize_remstnet_vdn_intersection as vdn

    source = Path(path).resolve()
    value = _read_json(source)
    _require(
        value.get("protocol") == "syncg_r2mt_net_eval_v1"
        and value.get("status") == "complete",
        f"R²MT SyncG evaluation protocol differs: {source}",
    )
    model = value.get("model")
    _require(isinstance(model, Mapping), f"R²MT model metadata is missing: {source}")
    _validate_metadata(model, expected_variant="direct_scalar")
    source_seed = int(model.get("seed", -1))
    _require(source_seed in SEEDS, f"R²MT source seed differs: {source}")
    raw_rows = value.get("per_sample_condition")
    _require(isinstance(raw_rows, list) and bool(raw_rows), f"R²MT rows are missing: {source}")
    rows = vdn._index_rows(raw_rows, label="R²MT-Net")
    condition_pixels = {
        key: str(row.get("condition_pixel_sha256", "")) for key, row in rows.items()
    }
    _require(all(condition_pixels.values()), f"R²MT condition pixels are absent: {source}")
    for key, row in rows.items():
        target = float(row["normalized_target"])
        record = vdn._candidate(row)
        prediction = float(record["prediction"])
        error = float(record["absolute_error"])
        _require(
            math.isfinite(target)
            and 0.0 <= target <= 1.0
            and math.isfinite(prediction)
            and 0.0 <= prediction <= 1.0
            and abs(abs(prediction - target) - error) <= 1.0e-8,
            f"R²MT prediction/error is inconsistent: {key}",
        )
    return {
        "path": str(source),
        "source_seed": source_seed,
        "rows": rows,
        "condition_pixels": condition_pixels,
    }


def stage_vdn(output_root: Path) -> None:
    from experiments import summarize_remstnet_vdn_intersection as vdn

    vdn._load_remst_evaluation = _load_r2mt_vdn_evaluation
    vdn.PROTOCOL = VDN_PROTOCOL
    result = vdn.summarize_vdn_intersection(
        [_main_evaluation(seed) for seed in SEEDS],
        VDN_AUTOMATIC,
        VDN_ORACLE,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=20260821,
    )
    result["protocol"] = VDN_PROTOCOL
    result["publication_model"] = _publication_identity()
    result["compatibility_machine_keys"] = {
        "remstnet_v3": "R²MT-Net",
        "sarn_v2_efficientnet_b0": "historical comparator block",
    }
    result["reporting_notes"] = [
        note.replace("ReMSTNet-v3", "R²MT-Net").replace("ReMSTNet", "R²MT-Net")
        for note in result["reporting_notes"]
    ]
    output = Path(output_root) / "vdn" / "intersection_summary.json"
    _require(not output.exists(), f"VDN summary already exists: {output}")
    _write_json(output, result)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output.resolve()),
                "intersection_samples": result["cohort"]["intersection_samples"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _predict_r2mt_checkpoint(
    checkpoint_path: Path,
    items: Sequence[Mapping[str, Any]],
    *,
    device: Any,
    batch_size: int,
) -> tuple[int, dict[str, float], dict[str, str], dict[str, Any]]:
    import torch
    from experiments.a15_fteb import frozen_twin_endpoint_forward
    from experiments.evaluate_remstnet_clean_external import IMAGE_SIZE, _load_roi
    from experiments.pivot_direction_fallback import normalized_rgb_tensor
    from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
    from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi

    anchor, model, metadata = _configured_loader(checkpoint_path, device=device)
    _validate_metadata(metadata, expected_variant="direct_scalar")
    source_seed = int(metadata["seed"])
    predictions: dict[str, float] = {}
    failures: dict[str, str] = {}
    detected = [item for item in items if bool(item["detector_passed"])]
    with torch.inference_mode():
        for offset in range(0, len(detected), batch_size):
            batch_items = detected[offset : offset + batch_size]
            try:
                images = torch.stack(
                    [
                        normalized_rgb_tensor(
                            direct_resize_whole_roi(_load_roi(item), size=IMAGE_SIZE)
                        )
                        for item in batch_items
                    ]
                ).to(device)
                count = int(images.shape[0])
                support = torch.zeros(
                    (count, 1, IMAGE_SIZE, IMAGE_SIZE),
                    dtype=images.dtype,
                    device=device,
                )
                active = torch.zeros(count, dtype=torch.bool, device=device)
                homography = torch.eye(
                    3, dtype=images.dtype, device=device
                )[None].repeat(count, 1, 1)
                endpoints = frozen_twin_endpoint_forward(anchor, images, images)
                output = forward_a15_correction(
                    model,
                    {
                        "raw_posterior": endpoints["raw_posterior"],
                        "sarn_posterior": endpoints["sarn_posterior"],
                        "raw_mean": endpoints["raw_mean"],
                        "sarn_mean": endpoints["sarn_mean"],
                        "raw_features": endpoints["raw_features"],
                        "sarn_features": endpoints["sarn_features"],
                        "sarn_support_mask": support,
                        "sarn_active": active,
                        "raw_to_sarn_homography": homography,
                    },
                    endpoint_null=False,
                )
                _require(
                    not bool(output["relation_available"].any().item()),
                    "single-view OCR input unexpectedly activated the relation path",
                )
                values = output["mean"].detach().float().cpu().tolist()
                _require(
                    len(values) == len(batch_items),
                    "R²MT OCR prediction batch length differs",
                )
                for item, value in zip(batch_items, values, strict=True):
                    scalar = float(value)
                    _require(
                        math.isfinite(scalar) and 0.0 <= scalar <= 1.0,
                        f"{item['sample_id']}: R²MT prediction is invalid",
                    )
                    predictions[str(item["sample_id"])] = scalar
            except Exception as exc:  # preserve the established per-batch failure rule
                code = f"model_exception:{type(exc).__name__}"
                for item in batch_items:
                    failures[str(item["sample_id"])] = code
    return source_seed, predictions, failures, metadata


def _postprocess_ocr(output_root: Path) -> None:
    prediction_summary_path = Path(output_root) / "prediction_summary.json"
    prediction_summary = _read_json(prediction_summary_path)
    prediction_summary["task"] = (
        "full frame -> cached best-confidence meter box -> R²MT-Net normalized "
        "progress -> PP-OCRv4/GARC automatic range -> physical reading"
    )
    prediction_summary["publication_model"] = _publication_identity()
    prediction_summary["models"]["r2mt"] = prediction_summary["models"].pop(
        "remstnet"
    )
    timing = prediction_summary["timing"]
    timing["r2mt_three_seed_seconds"] = timing.pop("remstnet_three_seed_seconds")
    prediction_summary["compatibility_machine_keys"] = {
        "remst_progress_by_seed": "R²MT-Net predictions retained for scorer compatibility"
    }
    _write_json(prediction_summary_path, prediction_summary)

    score_path = Path(output_root) / "score.json"
    score = _read_json(score_path)
    score["publication_model"] = _publication_identity()
    score["compatibility_machine_keys"] = {
        "remst_progress_by_seed": "R²MT-Net predictions retained for scorer compatibility"
    }
    score["timing"] = timing
    _write_json(score_path, score)


def stage_ocr(output_root: Path, device_name: str) -> None:
    from experiments import evaluate_remstnet_ocr_end_to_end as ocr

    ocr.PROTOCOL = OCR_PROTOCOL
    ocr.PREDICTION_PROTOCOL = f"{OCR_PROTOCOL}_predictions"
    ocr.SCORE_PROTOCOL = f"{OCR_PROTOCOL}_score"
    ocr._predict_checkpoint = _predict_r2mt_checkpoint
    output = Path(output_root) / "ocr"
    ocr.run_label_free_prediction(
        field_root=ocr.DEFAULT_FIELD_ROOT,
        detector_labeled_path=(
            ocr.DEFAULT_DETECTOR_ROOT
            / "field_gauge_full_frame_labeled_predictions.jsonl"
        ),
        detector_unlabeled_path=(
            ocr.DEFAULT_DETECTOR_ROOT
            / "field_gauge_full_frame_unlabeled_predictions.jsonl"
        ),
        checkpoint_paths=[_checkpoint(seed) for seed in SEEDS],
        bundle_path=ocr.DEFAULT_BUNDLE,
        point_detector_path=ocr.DEFAULT_POINT_DETECTOR,
        v4_score_path=ocr.DEFAULT_V4_SCORE,
        v5_score_path=ocr.DEFAULT_V5_SCORE,
        v5_server_diagnostic_path=ocr.DEFAULT_V5_SERVER_DIAGNOSTIC,
        output_root=output,
        device_name=device_name,
        batch_size=64,
        cpu_threads=2,
        ocr_view_mode="original",
        limit=None,
    )
    ocr.score_predictions(
        prediction_root=output,
        labels_path=ocr.DEFAULT_FIELD_ROOT / "full_frame_labels.jsonl",
        output_path=output / "score.json",
        bootstrap_seed=20260821,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
    _postprocess_ocr(output)
    print(
        json.dumps({"status": "complete", "output": str(output.resolve())}),
        flush=True,
    )


def _r2mt_checkpoint_roster(
    paths: Sequence[Path], *, expected_variant: str = "direct_scalar"
) -> dict[int, Path]:
    import torch
    from experiments.train_r2mt_net import (
        PROTOCOL as TRAINING_PROTOCOL,
    )
    from experiments.r2mt_net import ARCHITECTURE

    _require(
        expected_variant == "direct_scalar" and len(paths) == 3,
        "R²MT natural-repeat checkpoint roster differs",
    )
    result: dict[int, Path] = {}
    stable_identity: dict[str, Any] | None = None
    for path in paths:
        source = Path(path).resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        _require(
            isinstance(payload, Mapping)
            and payload.get("protocol") == TRAINING_PROTOCOL
            and payload.get("architecture") == ARCHITECTURE,
            f"R²MT checkpoint identity differs: {source}",
        )
        seed = int(payload["seed"])
        identity = {
            "construction": dict(payload["construction"]),
            "parameter_counts": dict(payload["parameter_counts"]),
        }
        _require(seed not in result, f"duplicate R²MT seed: {seed}")
        if stable_identity is None:
            stable_identity = identity
        else:
            _require(identity == stable_identity, f"R²MT construction differs: {source}")
        result[seed] = source
    _require(set(result) == set(SEEDS), f"R²MT seed roster differs: {sorted(result)}")
    return result


def _load_frozen_repeat_cohort(
    *,
    provenance_path: Path,
    labels_path: Path,
    xm2_manifest_path: Path,
):
    """Reuse the user's previously curated retained cohort without reselection."""

    from experiments.score_natural_repeat_stability import (
        CohortRow,
        _exact_number_key,
    )

    reference = _read_json(Path(xm2_manifest_path))
    _require(
        reference.get("status") == "complete"
        and isinstance(reference.get("cohort"), Mapping)
        and isinstance(reference.get("per_sample"), list),
        "frozen natural-repeat cohort reference is incomplete",
    )

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        source = Path(path).resolve()
        _require(source.is_file(), f"natural-repeat JSONL is missing: {source}")
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]

    provenance = {
        str(row["sample_id"]): row for row in read_jsonl(provenance_path)
    }
    labels = {str(row["sample_id"]): row for row in read_jsonl(labels_path)}
    reference_rows = reference["per_sample"]
    _require(
        len(reference_rows) == 1203,
        "frozen natural-repeat cohort must contain 1,203 rows",
    )
    cohort: list[Any] = []
    seen: set[str] = set()
    for row in reference_rows:
        sample_id = str(row.get("sample_id") or "")
        _require(
            sample_id
            and sample_id not in seen
            and sample_id in provenance
            and sample_id in labels,
            f"frozen cohort row cannot be joined: {sample_id}",
        )
        seen.add(sample_id)
        source = provenance[sample_id]
        label = labels[sample_id]
        original_sample_id = str(row.get("original_sample_id") or "")
        group_id = str(row.get("physical_group_id") or "")
        ground_truth_key = str(row.get("ground_truth_key") or "")
        source_image = str(row.get("source_image") or "").replace("\\", "/")
        target = float(row["normalized_target"])
        _require(
            original_sample_id == str(source.get("original_sample_id") or "")
            and group_id == str(source.get("original_group_id") or "")
            and ground_truth_key
            == _exact_number_key(
                label.get("ground_truth"), label=f"ground_truth[{sample_id}]"
            )
            and abs(target - float(label["normalized_progress"])) <= 1.0e-12
            and bool(source_image),
            f"frozen cohort metadata differs from current inputs: {sample_id}",
        )
        cohort.append(
            CohortRow(
                sample_id=sample_id,
                original_sample_id=original_sample_id,
                group_id=group_id,
                ground_truth_key=ground_truth_key,
                source_image=source_image,
                target=target,
            )
        )
    frozen = reference["cohort"]
    summary = {
        "joined_xm2_samples": int(frozen["source_joined_samples"]),
        "joined_physical_groups": int(frozen["source_joined_physical_groups"]),
        "joined_exact_reading_units": int(
            frozen["source_joined_exact_reading_units"]
        ),
        "retained_samples": len(cohort),
        "retained_physical_groups": len({row.group_id for row in cohort}),
        "retained_exact_reading_units": len(
            {(row.group_id, row.ground_truth_key) for row in cohort}
        ),
        "retained_distinct_source_images": len(
            {row.source_image for row in cohort}
        ),
        "unit_rule": str(frozen["unit_rule"]),
        "same_source_image_rule": str(frozen["same_source_image_rule"]),
    }
    _require(
        summary["retained_samples"] == int(frozen["retained_samples"])
        and summary["retained_physical_groups"]
        == int(frozen["retained_physical_groups"])
        and summary["retained_exact_reading_units"]
        == int(frozen["retained_exact_reading_units"])
        and summary["retained_distinct_source_images"]
        == int(frozen["retained_distinct_source_images"]),
        "frozen cohort aggregate identity differs",
    )
    return tuple(cohort), summary


def stage_repeat(output_root: Path, device_name: str) -> None:
    from experiments import evaluate_a15_2_mett_natural_repeat_stability as repeat

    repeat.load_mett_models = _configured_loader
    repeat.checkpoint_roster = _r2mt_checkpoint_roster
    repeat.load_xm2_repeat_cohort = _load_frozen_repeat_cohort
    repeat.publication_model_identity = _publication_identity
    repeat.PROTOCOL = REPEAT_PROTOCOL
    repeat.METHOD_LABELS["mett"] = "R²MT-Net"
    output = Path(output_root) / "repeat" / "natural_repeat.json"
    frozen_assets = repeat.RepeatAssets(
        provenance=repeat.DEFAULT_ASSETS.provenance,
        labels=repeat.DEFAULT_ASSETS.labels,
        physical_capture_manifest=REPEAT_COHORT_REFERENCE,
        source_input_manifest=repeat.DEFAULT_ASSETS.source_input_manifest,
    )
    result = repeat.evaluate_natural_repeat_stability(
        checkpoints=[_checkpoint(seed) for seed in SEEDS],
        output_path=output,
        assets=frozen_assets,
        device_name=device_name,
        workers=4,
        batch_size=64,
        use_amp=False,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=20260818,
        expected_variant="direct_scalar",
    )
    value = _read_json(output)
    value["publication_model"] = _publication_identity()
    value["scope"].update(
        {
            "internal_representation_conditioned_risk_weighting": True,
            "prediction_dependent_routing_or_selection": True,
            "external_model_selection": False,
            "cross_seed_ensemble": False,
            "frozen_user_curated_cohort_reused": True,
            "cohort_selection_recomputed": False,
        }
    )
    value["compatibility_machine_keys"] = {"mett": "R²MT-Net"}
    value["inputs"]["cohort_definition_reference"] = value["inputs"].pop(
        "physical_capture_manifest"
    )
    _write_json(output, value)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


def stage_efficiency(output_root: Path, device_name: str, arm: str) -> None:
    from experiments import benchmark_a15_2_mett_efficiency as benchmark

    _require(arm in benchmark.ARMS, f"unknown efficiency arm: {arm}")
    benchmark.load_mett_models = _configured_loader
    benchmark.validate_mett_variant_metadata = _validate_metadata
    benchmark.publication_model_identity = _publication_identity
    benchmark.PROTOCOL = EFFICIENCY_PROTOCOL
    benchmark.DISPLAY_NAMES.update(
        {
            "raw_only": "R²MT-Net frozen Raw anchor",
            "twin_endpoint": "R²MT-Net frozen Raw + SARN twin endpoint",
            "full_mett": "R²MT-Net full risk-conditioned transport",
        }
    )
    output_dir = Path(output_root) / "efficiency"
    output_json = output_dir / f"{arm}.json"
    output_markdown = output_dir / f"{arm}.md"
    report = benchmark.run_benchmark(
        arm=arm,
        checkpoint_path=_checkpoint(SEEDS[0]),
        correction_train_manifest_path=benchmark.DEFAULT_CORRECTION_TRAIN_MANIFEST,
        output_json=output_json,
        output_markdown=output_markdown,
        device_name=device_name,
        expected_variant="direct_scalar",
    )
    report["compatibility_machine_keys"] = {
        "mett_correction": "R²MT-Net risk-conditioned correction module",
        "full_mett": "R²MT-Net full inference arm",
    }
    _write_json(output_json, report)
    output_markdown.write_text(
        benchmark.render_markdown(report), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"status": "complete", "arm": arm, "output": str(output_json.resolve())}
        ),
        flush=True,
    )


def _metric(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        current = current[key]
    return current


def _build_experiment_summary(output_root: Path, run_manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_root).resolve()
    real = _read_json(root / "real" / "three_seed_summary.json")
    vdn = _read_json(root / "vdn" / "intersection_summary.json")
    ocr = _read_json(root / "ocr" / "score.json")
    repeat = _read_json(root / "repeat" / "natural_repeat.json")
    efficiency = {
        arm: _read_json(root / "efficiency" / f"{arm}.json")
        for arm in ("raw_only", "twin_endpoint", "full_mett")
    }
    industrial = real["industrial_real_photo_baseline"]["subsets"]
    vdn_all = vdn["summary"]["all_conditions"]
    repeat_r2mt = repeat["summary"]["methods"]["mett"]["three_seed"]
    efficiency_summary: dict[str, Any] = {}
    for arm, report in efficiency.items():
        profiles = report["measurement"]["profiles"]
        efficiency_summary[arm] = {
            "display_name": report["display_name"],
            "parameters": report["parameters"],
            "batch_1": profiles["batch_1"],
            "batch_8": profiles["batch_8"],
        }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": _utc_now(),
        "publication_model": _publication_identity(),
        "exact_command": run_manifest["exact_command"],
        "environment": run_manifest["environment"],
        "stages": run_manifest["stages"],
        "boundaries": {
            "training_or_adaptation": False,
            "vdn_scope": (
                "re-score the frozen official-200 automatic/oracle ledgers on the "
                "fixed R²MT/VDN intersection; VDN was not retrained"
            ),
            "ocr_scope": (
                "replay cached detector boxes and the frozen OCR/range stack with "
                "fresh three-seed R²MT progress inference"
            ),
            "legacy_machine_keys_retained": True,
        },
        "results": {
            "real": {
                "industrial_all_conditions_r2mt_nmae": _metric(
                    industrial,
                    "all_conditions",
                    "three_seed",
                    "remstnet_metrics_mean_sample_sd",
                    "nmae",
                ),
                "industrial_all_conditions_direct_resnet18_nmae": _metric(
                    industrial,
                    "all_conditions",
                    "three_seed",
                    "comparator_metrics_mean_sample_sd",
                    "nmae",
                ),
                "industrial_projective_r2mt_nmae": _metric(
                    industrial,
                    "projective_pooled",
                    "three_seed",
                    "remstnet_metrics_mean_sample_sd",
                    "nmae",
                ),
                "industrial_projective_direct_resnet18_nmae": _metric(
                    industrial,
                    "projective_pooled",
                    "three_seed",
                    "comparator_metrics_mean_sample_sd",
                    "nmae",
                ),
                "all_conditions_paired": _metric(
                    industrial,
                    "all_conditions",
                    "three_seed",
                    "mean_row_error_across_independent_seeds",
                    "paired",
                ),
                "projective_paired": _metric(
                    industrial,
                    "projective_pooled",
                    "three_seed",
                    "mean_row_error_across_independent_seeds",
                    "paired",
                ),
            },
            "vdn": {
                "cohort": vdn["cohort"],
                "r2mt_nmae": vdn_all["remstnet_v3"][
                    "metric_across_seed_mean_sd"
                ]["nmae"],
                "vdn_automatic_coverage": vdn_all[
                    "vdn_official200_automatic_reference"
                ]["coverage"],
                "vdn_automatic_failure_inclusive_nmae": vdn_all[
                    "vdn_official200_automatic_reference"
                ]["failure_inclusive_metrics"]["nmae"],
                "vdn_annotation_reference_nmae": vdn_all[
                    "vdn_official200_annotation_reference"
                ]["metrics"]["nmae"],
                "paired": vdn_all["paired_scene_bootstrap"],
            },
            "ocr": {
                "samples": ocr["samples"],
                "coverage_all_frames": ocr["coverage_all_frames"],
                "rowwise_three_seed_mean": ocr["rowwise_three_seed_mean"],
                "operating_point_metrics": ocr["operating_point_metrics"],
                "range_metrics_on_labeled_frames": ocr[
                    "range_metrics_on_labeled_frames"
                ],
            },
            "repeat": {
                "cohort": repeat["cohort"],
                "r2mt": repeat_r2mt,
                "paired_comparisons": repeat["summary"]["paired_comparisons"],
                "timing": repeat["timing"],
            },
            "efficiency": efficiency_summary,
        },
        "artifacts": {
            "real_summary": str((root / "real" / "three_seed_summary.json").resolve()),
            "vdn_summary": str((root / "vdn" / "intersection_summary.json").resolve()),
            "ocr_score": str((root / "ocr" / "score.json").resolve()),
            "repeat": str((root / "repeat" / "natural_repeat.json").resolve()),
            "efficiency": {
                arm: str((root / "efficiency" / f"{arm}.json").resolve())
                for arm in efficiency
            },
        },
    }
    return summary


def _render_result_markdown(summary: Mapping[str, Any]) -> str:
    real = summary["results"]["real"]
    vdn = summary["results"]["vdn"]
    ocr = summary["results"]["ocr"]
    repeat = summary["results"]["repeat"]
    efficiency = summary["results"]["efficiency"]
    lines = [
        "# R²MT-Net downstream replay result",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {summary['created_utc']}",
        "- Verification Status: UNVERIFIED",
        "- Version Label: exp_result_v1",
        "",
        "## Execution",
        "",
        f"- Exact command: `{summary['exact_command']}`",
        "- Training/adaptation: none",
        "- VDN: frozen official-200 ledgers re-scored; no VDN retraining",
        "- OCR: cached detector boxes and frozen OCR/range stack; fresh R²MT inference",
        "",
        "## Real photographs",
        "",
        "| Scope | R²MT-Net NMAE mean ± sample SD | Direct-ResNet18 NMAE mean ± sample SD |",
        "|---|---:|---:|",
        (
            "| Industrial / all six | "
            f"{real['industrial_all_conditions_r2mt_nmae']['mean']:.6f} ± "
            f"{real['industrial_all_conditions_r2mt_nmae']['sample_sd']:.6f} | "
            f"{real['industrial_all_conditions_direct_resnet18_nmae']['mean']:.6f} ± "
            f"{real['industrial_all_conditions_direct_resnet18_nmae']['sample_sd']:.6f} |"
        ),
        (
            "| Industrial / projective pooled | "
            f"{real['industrial_projective_r2mt_nmae']['mean']:.6f} ± "
            f"{real['industrial_projective_r2mt_nmae']['sample_sd']:.6f} | "
            f"{real['industrial_projective_direct_resnet18_nmae']['mean']:.6f} ± "
            f"{real['industrial_projective_direct_resnet18_nmae']['sample_sd']:.6f} |"
        ),
        "",
        "The real-domain adapter verifies identical condition pixels against the frozen "
        "ledgers. R²MT-Net's refined anchor is intentionally not asserted to be the same "
        "checkpoint as Direct-ResNet18.",
        "",
        "## VDN intersection",
        "",
        f"- Intersection: {vdn['cohort']['intersection_samples']} samples, "
        f"{vdn['cohort']['intersection_rows']} rows, "
        f"{vdn['cohort']['intersection_scene_groups']} scene groups.",
        f"- R²MT-Net NMAE: {vdn['r2mt_nmae']['mean']:.6f} ± "
        f"{vdn['r2mt_nmae']['sample_sd']:.6f}.",
        f"- VDN automatic coverage: {vdn['vdn_automatic_coverage']:.4f}; "
        f"failure-inclusive NMAE: {vdn['vdn_automatic_failure_inclusive_nmae']:.6f}.",
        f"- VDN annotation-reference component NMAE: "
        f"{vdn['vdn_annotation_reference_nmae']:.6f}.",
        "",
        "## OCR end to end",
        "",
        f"- Coverage denominator: {ocr['coverage_all_frames']['denominator']} frames; "
        f"labeled accuracy denominator: {ocr['samples']['labeled_full_frames_for_accuracy']}.",
        f"- Default end-to-end coverage: "
        f"{ocr['coverage_all_frames']['end_to_end_decoder_default']}/"
        f"{ocr['coverage_all_frames']['denominator']}.",
        f"- Default rowwise three-seed mean NMAE: "
        f"{ocr['rowwise_three_seed_mean']['decoder_default_nmae']:.6f}.",
        f"- Oracle-range progress NMAE: "
        f"{ocr['rowwise_three_seed_mean']['oracle_range_nmae']:.6f}.",
        "",
        "## Natural repeatability",
        "",
        f"- Retained samples: {repeat['cohort']['retained_samples']}; "
        f"exact-reading units: {repeat['cohort']['retained_exact_reading_units']}.",
        f"- Full-denominator NMAE: "
        f"{repeat['r2mt']['full_denominator_nmae']['mean']:.6f} ± "
        f"{repeat['r2mt']['full_denominator_nmae']['sample_sd']:.6f}.",
        f"- Mean within-unit population SD: "
        f"{repeat['r2mt']['mean_within_unit_prediction_sd_population']['mean']:.6f} ± "
        f"{repeat['r2mt']['mean_within_unit_prediction_sd_population']['sample_sd']:.6f}.",
        "",
        "## CUDA/BF16 efficiency",
        "",
        "| Arm | B1 mean / p95 ms | B8 mean / p95 ms | B8 samples/s |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("raw_only", "twin_endpoint", "full_mett"):
        value = efficiency[arm]
        b1 = value["batch_1"]
        b8 = value["batch_8"]
        lines.append(
            f"| {value['display_name']} | "
            f"{b1['latency_per_sample_ms']['mean_ms']:.3f} / "
            f"{b1['latency_per_sample_ms']['p95_ms']:.3f} | "
            f"{b8['latency_per_sample_ms']['mean_ms']:.3f} / "
            f"{b8['latency_per_sample_ms']['p95_ms']:.3f} | "
            f"{b8['throughput_samples_per_second']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Historical evaluator keys (`mett`, `remstnet`, and in one real-domain "
            "scorer `efficientnet_b0`) remain only for scorer compatibility; each "
            "artifact records its actual R²MT-Net or Direct-ResNet18 meaning.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_final_summary(output_root: Path) -> None:
    root = Path(output_root).resolve()
    manifest = _read_json(root / "run_manifest.json")
    summary = _build_experiment_summary(root, manifest)
    output_json = root / "experiment_summary.json"
    output_md = root / "experiment_result.md"
    _require(
        not output_json.exists() and not output_md.exists(),
        "final experiment summary already exists",
    )
    _write_json(output_json, summary)
    output_md.write_text(_render_result_markdown(summary), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(output_json),
                "report": str(output_md),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _preflight(output_root: Path, device_name: str) -> dict[str, Any]:
    import torch
    from experiments import benchmark_a15_2_mett_efficiency as benchmark
    from experiments import evaluate_remstnet_ocr_end_to_end as ocr
    from experiments.evaluate_paper_syncg_only_natural_repeat_stability import (
        DEFAULT_ASSETS,
        DEFAULT_ROOT,
        cohort_manifest_path,
    )

    root = Path(output_root).resolve()
    _require(not root.exists(), f"refusing to overwrite output root: {root}")
    required_files = [
        *[_checkpoint(seed) for seed in SEEDS],
        *[_main_evaluation(seed) for seed in SEEDS],
        VDN_AUTOMATIC,
        VDN_ORACLE,
        ocr.DEFAULT_FIELD_ROOT / "raw_inventory.jsonl",
        ocr.DEFAULT_FIELD_ROOT / "full_frame_labels.jsonl",
        ocr.DEFAULT_DETECTOR_ROOT
        / "field_gauge_full_frame_labeled_predictions.jsonl",
        ocr.DEFAULT_DETECTOR_ROOT
        / "field_gauge_full_frame_unlabeled_predictions.jsonl",
        ocr.DEFAULT_BUNDLE,
        ocr.DEFAULT_POINT_DETECTOR,
        ocr.DEFAULT_V4_SCORE,
        ocr.DEFAULT_V5_SCORE,
        ocr.DEFAULT_V5_SERVER_DIAGNOSTIC,
        DEFAULT_ASSETS.provenance,
        DEFAULT_ASSETS.labels,
        REPEAT_COHORT_REFERENCE,
        cohort_manifest_path(DEFAULT_ROOT),
        benchmark.DEFAULT_CORRECTION_TRAIN_MANIFEST,
    ]
    missing = [str(Path(path).resolve()) for path in required_files if not Path(path).is_file()]
    _require(not missing, f"required replay files are missing: {missing}")
    _require(FACTORIAL_ROOT.is_dir(), f"factorial prediction root is missing: {FACTORIAL_ROOT}")
    device = torch.device(device_name)
    _require(device.type == "cuda", "formal replay command requires a CUDA device")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    disk = shutil.disk_usage(root.parent)
    _require(disk.free >= 2 * 1024**3, "less than 2 GiB remains on the output volume")
    git = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_bf16": bool(torch.cuda.is_bf16_supported()),
        "device": device_name,
        "cpu_count": os.cpu_count(),
        "free_bytes_at_start": disk.free,
        "git_status_short": git.stdout.splitlines(),
    }


def _reader(stream: Any, messages: queue.Queue[str | None]) -> None:
    try:
        for line in iter(stream.readline, ""):
            messages.put(line)
    finally:
        messages.put(None)


def _run_child(
    *,
    label: str,
    args: Sequence[str],
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "experiments.run_r2mt_downstream_replays", *args]
    started_utc = _utc_now()
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _require(process.stdout is not None, "child process stdout pipe is absent")
        messages: queue.Queue[str | None] = queue.Queue()
        thread = threading.Thread(
            target=_reader, args=(process.stdout, messages), daemon=True
        )
        thread.start()
        reader_done = False
        last_heartbeat = time.perf_counter()
        timed_out = False
        while process.poll() is None or not reader_done:
            try:
                line = messages.get(timeout=1.0)
                if line is None:
                    reader_done = True
                else:
                    print(f"[{label}] {line}", end="", flush=True)
                    log.write(line)
                    log.flush()
            except queue.Empty:
                pass
            elapsed = time.perf_counter() - started
            if process.poll() is None and elapsed > timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            if process.poll() is None and time.perf_counter() - last_heartbeat >= 30:
                print(f"[{label}] still running ({elapsed:.1f}s)", flush=True)
                last_heartbeat = time.perf_counter()
        return_code = process.wait()
        thread.join(timeout=5)
    elapsed_seconds = float(time.perf_counter() - started)
    record = {
        "label": label,
        "command": subprocess.list2cmdline(command),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "timed_out": timed_out,
        "log": str(log_path.resolve()),
    }
    _require(not timed_out, f"stage timed out after {timeout_seconds}s: {label}")
    _require(return_code == 0, f"stage failed with exit code {return_code}: {label}")
    return record


def run_all(output_root: Path, device_name: str) -> int:
    root = Path(output_root).resolve()
    environment = _preflight(root, device_name)
    root.mkdir(parents=True, exist_ok=False)
    exact_command = (
        ".\\.venv\\Scripts\\python.exe -m "
        "experiments.run_r2mt_downstream_replays --output-root "
        "artifacts\\runs\\r2mt_downstream_repro_20260825 --device cuda:0"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_utc": _utc_now(),
        "exact_command": exact_command,
        "invocation_argv": [sys.executable, *sys.argv],
        "environment": environment,
        "timeout_seconds_per_stage": STAGE_TIMEOUT_SECONDS,
        "automatic_retry": False,
        "stages": [],
    }
    _write_json(root / "run_manifest.json", manifest)
    specifications: list[tuple[str, list[str]]] = [
        *[
            (
                f"real_seed_{seed}",
                [
                    "--output-root",
                    str(root),
                    "--device",
                    device_name,
                    "--stage",
                    "real",
                    "--seed",
                    str(seed),
                ],
            )
            for seed in SEEDS
        ],
        (
            "real_summary",
            ["--output-root", str(root), "--device", device_name, "--stage", "real_summary"],
        ),
        (
            "vdn",
            ["--output-root", str(root), "--device", device_name, "--stage", "vdn"],
        ),
        (
            "ocr",
            ["--output-root", str(root), "--device", device_name, "--stage", "ocr"],
        ),
        (
            "repeat",
            ["--output-root", str(root), "--device", device_name, "--stage", "repeat"],
        ),
        *[
            (
                f"efficiency_{arm}",
                [
                    "--output-root",
                    str(root),
                    "--device",
                    device_name,
                    "--stage",
                    "efficiency",
                    "--arm",
                    arm,
                ],
            )
            for arm in ("raw_only", "twin_endpoint", "full_mett")
        ],
    ]
    try:
        for label, args in specifications:
            print(f"[orchestrator] starting {label}", flush=True)
            record = _run_child(
                label=label,
                args=args,
                log_path=root / "logs" / f"{label}.log",
                timeout_seconds=STAGE_TIMEOUT_SECONDS,
            )
            manifest["stages"].append(record)
            _write_json(root / "run_manifest.json", manifest)
            print(
                f"[orchestrator] completed {label} in {record['elapsed_seconds']:.1f}s",
                flush=True,
            )

        manifest["status"] = "complete_pending_summary"
        manifest["finished_utc"] = _utc_now()
        manifest["total_elapsed_seconds"] = float(
            sum(float(stage["elapsed_seconds"]) for stage in manifest["stages"])
        )
        _write_json(root / "run_manifest.json", manifest)
        record = _run_child(
            label="final_summary",
            args=[
                "--output-root",
                str(root),
                "--device",
                device_name,
                "--stage",
                "final_summary",
            ],
            log_path=root / "logs" / "final_summary.log",
            timeout_seconds=STAGE_TIMEOUT_SECONDS,
        )
        manifest["stages"].append(record)
        manifest["status"] = "complete"
        manifest["finished_utc"] = _utc_now()
        manifest["total_elapsed_seconds"] = float(
            sum(float(stage["elapsed_seconds"]) for stage in manifest["stages"])
        )
        _write_json(root / "run_manifest.json", manifest)
        # Refresh final summary with the terminal manifest status and duration.
        summary = _build_experiment_summary(root, manifest)
        _write_json(root / "experiment_summary.json", summary)
        (root / "experiment_result.md").write_text(
            _render_result_markdown(summary), encoding="utf-8", newline="\n"
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_root": str(root),
                    "summary": str(root / "experiment_summary.json"),
                    "report": str(root / "experiment_result.md"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_utc"] = _utc_now()
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(root / "run_manifest.json", manifest)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stage",
        choices=(
            "real",
            "real_summary",
            "vdn",
            "ocr",
            "repeat",
            "efficiency",
            "final_summary",
        ),
    )
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=("raw_only", "twin_endpoint", "full_mett"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.stage is None:
        return run_all(args.output_root, args.device)
    if args.stage == "real":
        _require(args.seed is not None, "--seed is required for the real stage")
        stage_real(args.output_root, args.seed, args.device)
    elif args.stage == "real_summary":
        stage_real_summary(args.output_root)
    elif args.stage == "vdn":
        stage_vdn(args.output_root)
    elif args.stage == "ocr":
        stage_ocr(args.output_root, args.device)
    elif args.stage == "repeat":
        stage_repeat(args.output_root, args.device)
    elif args.stage == "efficiency":
        _require(args.arm is not None, "--arm is required for the efficiency stage")
        stage_efficiency(args.output_root, args.device, args.arm)
    else:
        stage_final_summary(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
