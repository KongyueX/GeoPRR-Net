"""One descriptive untouched-Fold-B confirmation for the fixed A15.2 model.

The only data input is the existing 1,508-row/14-scene physical Fold-B
manifest produced by :mod:`experiments.prepare_a10_fold_b_manifest`.  Before
that manifest is opened, the terminal A11 checkpoint and terminal five-epoch
A15.2 correction checkpoint are independently strict-validated, their states
are compared bit-for-bit, and their train-only access metadata is checked.

Each condition batch supplies the same Raw/SARN/support/homography tensors to
the endpoint comparison and to terminal A11 Full-SCORT.  q0, fail-closed qS,
fixed geometric pooling, and final A15.2 share one frozen twin-endpoint call;
the Full-SCORT reference must reproduce that call's q0 exactly.  The output is
one create-once JSON report with per-row recomputation evidence.  It performs
no threshold gate, result-based model selection, retry, or automatic follow-up.
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

import experiments.evaluate_a15_2_fteb_correction_dev as dev_evaluation
from experiments.a10_pccot_data import (
    build_fold_b_evaluation_dataset,
    load_fold_b_manifest,
)
from experiments.a10_pccot_protocol import (
    DEFAULT_SEED as EVALUATION_SEED,
    EVALUATION_CONDITIONS,
    FOLD_B_SAMPLES,
    FOLD_B_SCENES,
    PROJECTIVE_CONDITIONS,
    condition_evaluation_specs,
)
from experiments.a13_correction_dev_protocol import CORRECTION_TRAIN_SCENES
from experiments.a11_scort import TRANSPORT_STAGE_COUNT
from experiments.a15_2_fteb import A15_2_ARCHITECTURE, A152FTEBCorrection
from experiments.a15_2_fteb_untouched_protocol import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    FINAL_CONFIRMATION_SPEC,
    FINAL_METHOD,
    METHOD_A11_FULL,
    METHOD_A15_2,
    METHOD_FIXED_GEOMETRIC,
    METHOD_ORDER,
    METHOD_Q0,
    METHOD_QS,
    PROTOCOL,
    scene_block_bootstrap,
    summarize_prediction_rows,
)
from experiments.a15_fteb import (
    BRIDGE_LAYERS,
    fixed_geometric_natural_parameter_base,
    frozen_twin_endpoint_forward,
)
from experiments.evaluate_a11_scort_core_audit import (
    EXPECTED_CHECKPOINT_ACCESS_FLAGS as A11_EXPECTED_ACCESS_FLAGS,
    load_terminal_checkpoint as load_fully_validated_terminal_a11,
)
from experiments.prepare_a10_fold_b_manifest import PROTOCOL as PREPARE_PROTOCOL
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.train_support_geometry_multiview_efficientnet_pilot import _loader


EVALUATION_BATCH_SIZE: Final[int] = 24
A15_POSTERIOR_TOLERANCE: Final[float] = 1.0e-6
A11_POSTERIOR_TOLERANCE: Final[float] = 1.0e-5


class A152FoldBEvaluationError(ValueError):
    """The fixed checkpoints, Fold-B rows, model outputs, or output differ."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152FoldBEvaluationError(message)


def _exact_model_state(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return tuple(left_state) == tuple(right_state) and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def load_fixed_final_models_before_fold_b(
    *,
    terminal_a11_checkpoint_path: Path,
    terminal_a15_2_correction_checkpoint_path: Path,
    device: torch.device,
) -> tuple[Any, A152FTEBCorrection, dict[str, Any]]:
    """Strict-validate both fixed terminal artifacts without opening Fold-B."""

    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    a15_path = Path(terminal_a15_2_correction_checkpoint_path).resolve()
    _require(a11_path != a15_path, "A11 and A15.2 checkpoint paths must differ")

    # This validator checks the full A11 training history, two-optimizer
    # evidence, train-only access flags, state schema, parameter counts, and
    # fresh strict load.  It performs no manifest access.
    fully_validated_a11, a11_metadata = load_fully_validated_terminal_a11(
        a11_path, device=device
    )
    # The A15.2 validator checks all 4,135 correction steps, complete history,
    # correction-only optimizer state, fixed-quarter architecture, access
    # flags, final-method metadata, and its exact nested A11 source.
    anchor, correction, a15_metadata = (
        dev_evaluation.load_terminal_a15_2_correction_and_anchor(
            a15_path, a11_path, device=device
        )
    )
    _require(
        _exact_model_state(fully_validated_a11, anchor),
        "independently validated A11 states are not bit-exact",
    )
    _require(
        a15_metadata.get("architecture") == A15_2_ARCHITECTURE
        and a15_metadata.get("anchor_unchanged") is True
        and a15_metadata.get("terminal_fallback_exact_q0") is True
        and a15_metadata.get("fresh_correction_strict_load") is True
        and a15_metadata.get("training", {}).get("validation_manifest") is None
        and a15_metadata.get("training", {}).get(
            "intermediate_checkpoint_selection"
        )
        is None,
        "A15.2 final-method metadata differs",
    )
    _require(
        a11_metadata.get("fresh_strict_load") is True
        and a11_metadata.get("pre_evaluation_access_flags")
        == A11_EXPECTED_ACCESS_FLAGS,
        "terminal A11 strict-load or train-only access metadata differs",
    )
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    correction.eval()
    _require(
        not anchor.training
        and not correction.training
        and all(not parameter.requires_grad for parameter in anchor.parameters()),
        "fixed final A11/A15.2 modules are not frozen in eval mode",
    )
    del fully_validated_a11
    return anchor, correction, {
        "validation_order": [
            "terminal_a11_complete_training_access_and_strict_state",
            "terminal_a15_2_complete_training_access_final_method_and_strict_state",
            "a11_states_bit_exact_across_independent_loads",
            "fold_b_manifest_open",
        ],
        "both_checkpoints_validated_before_fold_b_manifest_open": True,
        "terminal_a11": a11_metadata,
        "terminal_a15_2": a15_metadata,
        "final_method": FINAL_METHOD,
        "final_method_fixed_before_fold_b_manifest_open": True,
        "result_based_checkpoint_or_method_selection": False,
    }


def _tensor(
    output: Mapping[str, Any], field: str, shape: tuple[int, ...]
) -> torch.Tensor:
    value = output.get(field)
    _require(isinstance(value, torch.Tensor), f"model output is missing: {field}")
    _require(value.shape == shape, f"model output shape differs: {field}")
    return value


def _exact_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    _require(left.shape == right.shape, "exact-row comparison shapes differ")
    return (left == right).reshape(left.shape[0], -1).all(dim=1)


def _posterior_diagnostics(posterior: torch.Tensor) -> list[dict[str, Any]]:
    value = posterior.detach().float().cpu()
    _require(
        value.ndim == 2
        and value.shape[1] == 128
        and bool(torch.isfinite(value).all()),
        "posterior diagnostic input differs",
    )
    cdf = value.cumsum(dim=1)
    return [
        {
            "absolute_mass_error": float(abs(value[index].sum().item() - 1.0)),
            "negative_probability_count": int(
                (value[index] < -A15_POSTERIOR_TOLERANCE).sum()
            ),
            "cdf_monotonic_violation_count": int(
                (
                    (cdf[index, 1:] - cdf[index, :-1])
                    < -A15_POSTERIOR_TOLERANCE
                ).sum()
            ),
            "absolute_cdf_terminal_error": float(
                abs(cdf[index, -1].item() - 1.0)
            ),
        }
        for index in range(value.shape[0])
    ]


def _layer_diagnostics(
    posterior: torch.Tensor,
    *,
    serialized_cdf: torch.Tensor | None,
    label: str,
) -> list[dict[str, Any]]:
    layers_device = posterior.detach().float()
    _require(
        layers_device.ndim == 3
        and layers_device.shape[1:] == (BRIDGE_LAYERS, 128)
        and bool(torch.isfinite(layers_device).all()),
        f"{label} layer posterior shape or finiteness differs",
    )
    calculated_device = torch.stack(
        [
            layers_device[:, layer].cumsum(dim=1)
            for layer in range(BRIDGE_LAYERS)
        ],
        dim=1,
    )
    serialized_exact = None
    if serialized_cdf is not None:
        observed = serialized_cdf.detach().float()
        _require(
            observed.shape == layers_device.shape
            and observed.device == layers_device.device,
            f"{label} serialized layer CDF differs",
        )
        serialized_exact = (
            (observed == calculated_device)
            .reshape(layers_device.shape[0], -1)
            .all(dim=1)
            .cpu()
        )
    layers = layers_device.cpu()
    calculated = calculated_device.cpu()
    result: list[dict[str, Any]] = []
    for index in range(layers.shape[0]):
        mass_error = torch.abs(layers[index].sum(dim=1) - 1.0)
        cdf_steps = calculated[index, :, 1:] - calculated[index, :, :-1]
        row = {
            "layers": BRIDGE_LAYERS,
            "maximum_absolute_mass_error": float(mass_error.max()),
            "negative_probability_count": int(
                (layers[index] < -A15_POSTERIOR_TOLERANCE).sum()
            ),
            "cdf_monotonic_violation_count": int(
                (cdf_steps < -A15_POSTERIOR_TOLERANCE).sum()
            ),
            "maximum_absolute_cdf_terminal_error": float(
                torch.abs(calculated[index, :, -1] - 1.0).max()
            ),
            "cdf_recomputed_from_posterior": True,
        }
        if serialized_exact is not None:
            row["serialized_cdfs_exact_from_posteriors"] = bool(
                serialized_exact[index]
            )
        result.append(row)
    return result


def _diagnostics_valid(value: Mapping[str, Any], *, tolerance: float) -> bool:
    return (
        float(value["absolute_mass_error"]) <= tolerance
        and int(value["negative_probability_count"]) == 0
        and int(value["cdf_monotonic_violation_count"]) == 0
        and float(value["absolute_cdf_terminal_error"]) <= tolerance
    )


def _layer_diagnostics_valid(
    value: Mapping[str, Any], *, tolerance: float
) -> bool:
    return (
        int(value["layers"]) == BRIDGE_LAYERS
        and float(value["maximum_absolute_mass_error"]) <= tolerance
        and int(value["negative_probability_count"]) == 0
        and int(value["cdf_monotonic_violation_count"]) == 0
        and float(value["maximum_absolute_cdf_terminal_error"])
        <= tolerance
        and value.get("cdf_recomputed_from_posterior") is True
        and value.get("serialized_cdfs_exact_from_posteriors", True) is True
    )


def _a11_fallback_exact_rows(output: Mapping[str, Any]) -> torch.Tensor:
    q0_value = output.get("raw_anchor_posterior")
    _require(
        isinstance(q0_value, torch.Tensor)
        and q0_value.ndim == 2
        and q0_value.shape[1] == 128,
        "A11 q0 shape differs",
    )
    q0 = q0_value
    batch = q0.shape[0]
    q0_mean = _tensor(output, "raw_anchor_mean", (batch,))
    q0_variance = _tensor(output, "raw_anchor_variance", (batch,))
    layers = _tensor(output, "layer_posteriors", (batch, BRIDGE_LAYERS, 128))
    layer_means = _tensor(output, "layer_means", (batch, BRIDGE_LAYERS))
    layer_variances = _tensor(output, "layer_variances", (batch, BRIDGE_LAYERS))
    correction = _tensor(output, "correction_active", (batch, BRIDGE_LAYERS))
    relation = _tensor(output, "relation_available", (batch,))
    _require(
        correction.dtype == relation.dtype == torch.bool,
        "A11 fallback activity tensors must be boolean",
    )
    columns = (
        _exact_rows(_tensor(output, "progress_posterior", (batch, 128)), q0),
        _exact_rows(_tensor(output, "mean", (batch,)), q0_mean),
        _exact_rows(_tensor(output, "variance", (batch,)), q0_variance),
        _exact_rows(layers, q0[:, None].expand_as(layers)),
        _exact_rows(layer_means, q0_mean[:, None].expand_as(layer_means)),
        _exact_rows(
            layer_variances, q0_variance[:, None].expand_as(layer_variances)
        ),
        ~correction.any(dim=1),
        ~relation,
    )
    return torch.stack(columns, dim=1).all(dim=1)


def _configure_reproducibility(device: torch.device) -> None:
    random.seed(EVALUATION_SEED)
    np.random.seed(EVALUATION_SEED % (2**32))
    torch.manual_seed(EVALUATION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(EVALUATION_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def evaluate_same_batch_loader(
    anchor: Any,
    correction: A152FTEBCorrection,
    loader: Any,
    *,
    device: torch.device,
    expected_sample_ids: Sequence[str] | None = None,
    expected_condition: str | None = None,
) -> dict[str, Any]:
    """Evaluate exactly the five fixed methods on each shared input batch."""

    anchor.eval()
    correction.eval()
    rows: list[dict[str, Any]] = []
    batches = 0
    autocast_enabled = device.type == "cuda"
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            _require(isinstance(raw_batch, Mapping), "Fold-B batch is not a mapping")
            original = raw_batch["original_view"].to(device)
            sarn = raw_batch["sarn_view"].to(device)
            support = raw_batch["sarn_support_mask"].to(device)
            dataset_active = raw_batch["sarn_active"].to(device).bool()
            homography = raw_batch["raw_to_sarn_homography"].to(device)
            target = raw_batch["target"].to(device).float()
            batch = int(target.shape[0])
            ids = raw_batch.get("sample_id")
            scenes = raw_batch.get("scene_stem")
            names = raw_batch.get("condition_name")
            _require(
                isinstance(ids, Sequence)
                and not isinstance(ids, (str, bytes))
                and len(ids) == batch
                and isinstance(scenes, Sequence)
                and not isinstance(scenes, (str, bytes))
                and len(scenes) == batch
                and isinstance(names, Sequence)
                and not isinstance(names, (str, bytes))
                and len(names) == batch,
                "Fold-B row identities are batch-misaligned",
            )
            if expected_condition is not None:
                _require(
                    all(str(name) == expected_condition for name in names),
                    "Fold-B condition pixels differ from the requested arm",
                )
            clean_rows = torch.tensor(
                [str(name) == "clean" for name in names],
                device=device,
                dtype=torch.bool,
            )
            effective_active = dataset_active & ~clean_rows

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
                a15_output = forward_a15_correction(
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
                full_output = anchor(
                    original,
                    sarn,
                    support,
                    sarn_active=effective_active,
                    raw_to_sarn_homography=homography,
                )
            batches += 1

            q0 = _tensor(a15_output, "raw_anchor_posterior", (batch, 128)).float()
            proposed_qs = endpoints["sarn_posterior"].float()
            relation = _tensor(a15_output, "relation_available", (batch,))
            full_relation = _tensor(full_output, "relation_available", (batch,))
            _require(
                relation.dtype == full_relation.dtype == torch.bool
                and torch.equal(relation, full_relation),
                "A15.2 and terminal A11 relation availability differ",
            )
            _require(
                torch.equal(q0, endpoints["raw_posterior"].float())
                and torch.equal(
                    _tensor(full_output, "raw_anchor_posterior", (batch, 128)).float(),
                    q0,
                ),
                "terminal A11 Full-SCORT does not reproduce the shared q0 endpoint",
            )
            _require(
                torch.equal(
                    _tensor(
                        a15_output,
                        "proposed_sarn_endpoint_posterior",
                        (batch, 128),
                    ).float(),
                    proposed_qs,
                ),
                "A15.2 does not preserve the shared qS endpoint",
            )

            qs = _tensor(a15_output, "sarn_endpoint_posterior", (batch, 128)).float()
            fixed = _tensor(a15_output, "geometric_base", (batch, 128)).float()
            final = _tensor(a15_output, "progress_posterior", (batch, 128)).float()
            full = _tensor(full_output, "progress_posterior", (batch, 128)).float()
            proposed_fixed = fixed_geometric_natural_parameter_base(q0, proposed_qs)[
                "geometric_base"
            ]
            _require(
                torch.equal(fixed, torch.where(relation[:, None], proposed_fixed, q0)),
                "fixed geometric posterior is not from the shared endpoints",
            )
            q0_mean = _tensor(a15_output, "raw_anchor_mean", (batch,)).float()
            means = {
                METHOD_Q0: q0_mean,
                METHOD_QS: _tensor(
                    a15_output, "sarn_endpoint_mean", (batch,)
                ).float(),
                METHOD_FIXED_GEOMETRIC: _tensor(
                    a15_output, "geometric_base_mean", (batch,)
                ).float(),
                METHOD_A15_2: _tensor(a15_output, "mean", (batch,)).float(),
                METHOD_A11_FULL: _tensor(full_output, "mean", (batch,)).float(),
            }
            posteriors = {
                METHOD_Q0: q0,
                METHOD_QS: qs,
                METHOD_FIXED_GEOMETRIC: fixed,
                METHOD_A15_2: final,
                METHOD_A11_FULL: full,
            }
            explicit_cdfs = {
                METHOD_Q0: _tensor(a15_output, "raw_anchor_cdf", (batch, 128)),
                METHOD_QS: _tensor(a15_output, "sarn_endpoint_cdf", (batch, 128)),
                METHOD_FIXED_GEOMETRIC: _tensor(
                    a15_output, "geometric_base_cdf", (batch, 128)
                ),
                METHOD_A15_2: _tensor(a15_output, "progress_cdf", (batch, 128)),
            }
            _require(
                all(
                    torch.equal(
                        explicit_cdfs[method].float(),
                        posteriors[method].cumsum(dim=1),
                    )
                    for method in explicit_cdfs
                ),
                "serialized endpoint/A15.2 CDF differs from its posterior",
            )

            diagnostics = {
                method: _posterior_diagnostics(posterior)
                for method, posterior in posteriors.items()
            }
            a15_layers = _layer_diagnostics(
                _tensor(
                    a15_output,
                    "layer_posteriors",
                    (batch, BRIDGE_LAYERS, 128),
                ),
                serialized_cdf=_tensor(
                    a15_output, "layer_cdfs", (batch, BRIDGE_LAYERS, 128)
                ),
                label="A15.2",
            )
            full_layers = _layer_diagnostics(
                _tensor(
                    full_output,
                    "layer_posteriors",
                    (batch, TRANSPORT_STAGE_COUNT, 128),
                ),
                serialized_cdf=None,
                label="terminal A11 Full-SCORT",
            )
            a15_fallback = dev_evaluation._fallback_exact_rows(a15_output)
            full_fallback = _a11_fallback_exact_rows(full_output)
            unavailable = ~relation
            _require(
                bool(a15_fallback[unavailable].all())
                and bool(full_fallback[unavailable].all()),
                "a relation-unavailable final path is not exact q0",
            )
            _require(
                not bool(relation[clean_rows].any())
                and bool(a15_fallback[clean_rows].all())
                and bool(full_fallback[clean_rows].all()),
                "a clean Fold-B row is not exact q0",
            )

            for index in range(batch):
                row_diagnostics = {
                    method: diagnostics[method][index] for method in METHOD_ORDER
                }
                invalid_methods = [
                    method
                    for method, value in row_diagnostics.items()
                    if not _diagnostics_valid(
                        value,
                        tolerance=(
                            A11_POSTERIOR_TOLERANCE
                            if method == METHOD_A11_FULL
                            else A15_POSTERIOR_TOLERANCE
                        ),
                    )
                ]
                a15_path_valid = _layer_diagnostics_valid(
                    a15_layers[index], tolerance=A15_POSTERIOR_TOLERANCE
                )
                a11_path_valid = _layer_diagnostics_valid(
                    full_layers[index], tolerance=A11_POSTERIOR_TOLERANCE
                )
                _require(
                    not invalid_methods and a15_path_valid and a11_path_valid,
                    "per-row posterior mass/CDF integrity differs: "
                    + json.dumps(
                        {
                            "batch_index": batch_index,
                            "batch_row_index": index,
                            "sample_id": str(ids[index]),
                            "scene_stem": str(scenes[index]),
                            "condition": str(names[index]),
                            "invalid_methods": invalid_methods,
                            "a15_2_path_valid": a15_path_valid,
                            "terminal_a11_full_path_valid": a11_path_valid,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                )
                row_target = float(target[index])
                row_means = {
                    method: float(means[method][index]) for method in METHOD_ORDER
                }
                rows.append(
                    {
                        "row_index": len(rows),
                        "sample_id": str(ids[index]),
                        "scene_stem": str(scenes[index]),
                        "condition": str(names[index]),
                        "normalized_target": row_target,
                        "mean": row_means,
                        "absolute_error": {
                            method: abs(value - row_target)
                            for method, value in row_means.items()
                        },
                        "dataset_sarn_active": bool(dataset_active[index]),
                        "effective_sarn_active": bool(effective_active[index]),
                        "relation_available": bool(relation[index]),
                        "fallback": {
                            "clean_forced_off": bool(clean_rows[index]),
                            "relation_unavailable": bool(unavailable[index]),
                            "q_sarn_fixed_geometric_and_a15_2_exact_q0": bool(
                                a15_fallback[index]
                            ),
                            "terminal_a11_full_path_exact_q0": bool(
                                full_fallback[index]
                            ),
                        },
                        "posterior_mass_cdf": row_diagnostics,
                        "path_mass_cdf": {
                            METHOD_A15_2: a15_layers[index],
                            METHOD_A11_FULL: full_layers[index],
                        },
                    }
                )

    _require(bool(rows), "Fold-B evaluation produced no rows")
    ids = tuple(str(row["sample_id"]) for row in rows)
    _require(len(ids) == len(set(ids)), "Fold-B condition sample IDs repeat")
    if expected_sample_ids is not None:
        _require(
            ids == tuple(str(value) for value in expected_sample_ids),
            "Fold-B condition roster/order differs from its manifest",
        )
    summary = summarize_prediction_rows(rows)
    relation_unavailable = sum(not bool(row["relation_available"]) for row in rows)
    return {
        "rows": len(rows),
        "same_input_batch_for_all_methods": True,
        "one_shared_twin_endpoint_forward_per_batch": True,
        "twin_endpoint_forward_batches": batches,
        "terminal_a11_full_scort_raw_q0_exact_shared_endpoint": True,
        "q0_qs_fixed_and_a15_2_share_exact_endpoints": True,
        "relation_available_rows": len(rows) - relation_unavailable,
        "relation_unavailable_rows": relation_unavailable,
        "clean_forced_off_rows": sum(
            bool(row["fallback"]["clean_forced_off"]) for row in rows
        ),
        "relation_unavailable_exact_q0_rows": sum(
            bool(row["fallback"]["relation_unavailable"])
            and bool(
                row["fallback"]["q_sarn_fixed_geometric_and_a15_2_exact_q0"]
            )
            and bool(row["fallback"]["terminal_a11_full_path_exact_q0"])
            for row in rows
        ),
        "posterior_mass_cdf_all_valid": True,
        "metrics": summary,
        "per_sample": rows,
        "per_sample_evidence_recomputes_nmae_cvar25_wtl_availability_and_fallback": True,
    }


def pool_condition_results(
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool full denominators after exact cross-condition roster alignment."""

    _require(
        set(conditions) == set(EVALUATION_CONDITIONS),
        "Fold-B condition result set differs",
    )
    rows_by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for condition in EVALUATION_CONDITIONS:
        rows = conditions[condition].get("per_sample")
        _require(
            isinstance(rows, Sequence)
            and not isinstance(rows, (str, bytes))
            and bool(rows),
            f"Fold-B per-row evidence is missing: {condition}",
        )
        rows_by_condition[condition] = list(rows)
    reference = tuple(
        (
            str(row["sample_id"]),
            str(row["scene_stem"]),
            float(row["normalized_target"]),
        )
        for row in rows_by_condition["clean"]
    )
    for condition, rows in rows_by_condition.items():
        observed = tuple(
            (
                str(row["sample_id"]),
                str(row["scene_stem"]),
                float(row["normalized_target"]),
            )
            for row in rows
        )
        _require(
            observed == reference,
            f"Fold-B roster/scene/target alignment differs: {condition}",
        )

    def pool(names: Sequence[str]) -> dict[str, Any]:
        rows = [row for condition in names for row in rows_by_condition[condition]]
        return {
            "conditions": list(names),
            "physical_fold_b_samples": len(reference),
            "rows": len(rows),
            "full_denominator_no_row_filtering": True,
            "metrics": summarize_prediction_rows(rows),
            "per_sample": rows,
        }

    return {
        "all_conditions": pool(EVALUATION_CONDITIONS),
        "projective_conditions": pool(PROJECTIVE_CONDITIONS),
    }


def evaluate_a15_2_fold_b_once(
    *,
    fold_b_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    terminal_a15_2_correction_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
    batch_size: int = EVALUATION_BATCH_SIZE,
) -> dict[str, Any]:
    """Run the sole create-once final confirmation, with no decision logic."""

    _require(type(workers) is int and workers >= 0, "Fold-B workers differ")
    _require(
        type(batch_size) is int and 1 <= batch_size <= EVALUATION_BATCH_SIZE,
        "Fold-B batch size differs",
    )
    manifest = Path(fold_b_manifest_path).resolve()
    a11_path = Path(terminal_a11_checkpoint_path).resolve()
    a15_path = Path(terminal_a15_2_correction_checkpoint_path).resolve()
    output = Path(output_path).resolve()
    _require(not output.exists(), f"Fold-B output already exists: {output}")
    _require(
        len({manifest, a11_path, a15_path, output}) == 4,
        "Fold-B manifest/checkpoints/output paths must all differ",
    )
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "Fold-B device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(device)

    # The call order here is part of the evidence: no Fold-B bytes are opened
    # until both fixed final artifacts and their access metadata are valid.
    anchor, correction, checkpoint_metadata = load_fixed_final_models_before_fold_b(
        terminal_a11_checkpoint_path=a11_path,
        terminal_a15_2_correction_checkpoint_path=a15_path,
        device=device,
    )
    samples = tuple(load_fold_b_manifest(manifest))
    _require(len(samples) == FOLD_B_SAMPLES, "physical Fold-B sample count differs")
    scenes = sorted({str(sample.scene_stem) for sample in samples})
    _require(len(scenes) == FOLD_B_SCENES, "physical Fold-B scene count differs")
    expected_ids = tuple(str(sample.sample_id) for sample in samples)
    terminal_a15_metadata = checkpoint_metadata.get("terminal_a15_2")
    _require(
        isinstance(terminal_a15_metadata, dict),
        "validated A15.2 metadata is missing before Fold-B overlap checks",
    )
    correction_train_ids_value = terminal_a15_metadata.pop(
        "physical_correction_train_sample_ids", None
    )
    _require(
        isinstance(correction_train_ids_value, list)
        and correction_train_ids_value
        and all(
            isinstance(value, str) and bool(value)
            for value in correction_train_ids_value
        ),
        "validated correction-train sample IDs are missing",
    )
    _require(
        set(expected_ids).isdisjoint(correction_train_ids_value)
        and set(scenes).isdisjoint(
            Path(scene).stem for scene in CORRECTION_TRAIN_SCENES
        ),
        "Fold-B overlaps the physical A15.2 correction-training roster",
    )

    specs = condition_evaluation_specs()
    _require(
        tuple(spec.condition for spec in specs) == EVALUATION_CONDITIONS,
        "fixed four-condition Fold-B specification differs",
    )
    conditions: dict[str, dict[str, Any]] = {}
    for spec in specs:
        dataset = build_fold_b_evaluation_dataset(
            samples,
            seed=spec.dataset_seed,
            total_epochs=spec.dataset_total_epochs,
            condition=spec.condition,
        )
        dataset.set_epoch(spec.transform_epoch)
        loader = _loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            workers=workers,
            seed=spec.loader_seed,
            cuda=device.type == "cuda",
        )
        conditions[spec.condition] = evaluate_same_batch_loader(
            anchor,
            correction,
            loader,
            device=device,
            expected_sample_ids=expected_ids,
            expected_condition=spec.condition,
        )
    _require(
        all(int(result["rows"]) == FOLD_B_SAMPLES for result in conditions.values()),
        "a Fold-B condition did not retain the full denominator",
    )
    clean = conditions["clean"]
    _require(
        int(clean["relation_available_rows"]) == 0
        and int(clean["clean_forced_off_rows"]) == FOLD_B_SAMPLES
        and int(clean["relation_unavailable_exact_q0_rows"]) == FOLD_B_SAMPLES,
        "clean Fold-B is not all-method exact q0",
    )
    pooled_internal = pool_condition_results(conditions)
    projective_rows = pooled_internal["projective_conditions"]["per_sample"]
    _require(
        len(projective_rows) == FOLD_B_SAMPLES * len(PROJECTIVE_CONDITIONS),
        "projective Fold-B denominator differs",
    )
    bootstrap = scene_block_bootstrap(
        projective_rows,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_SEED,
    )
    _require(
        int(bootstrap["clusters"]) == FOLD_B_SCENES
        and int(bootstrap["rows"])
        == FOLD_B_SAMPLES * len(PROJECTIVE_CONDITIONS)
        and int(bootstrap["seed"]) == BOOTSTRAP_SEED
        and int(bootstrap["iterations"]) == BOOTSTRAP_ITERATIONS,
        "fixed Fold-B scene-block bootstrap coverage differs",
    )

    # Per-condition rows remain the authoritative row evidence.  Avoid a
    # duplicate serialized copy under pooled while preserving all aggregates.
    pooled = {
        name: {key: value for key, value in result.items() if key != "per_sample"}
        for name, result in pooled_internal.items()
    }
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "prepare_protocol": PREPARE_PROTOCOL,
        "fixed_protocol": FINAL_CONFIRMATION_SPEC,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "scope": {
            "physical_fold_b_only": True,
            "terminal_a15_2_was_fixed_before_fold_b_open": True,
            "terminal_a11_full_scort_reference_only": True,
            "core_content_access": False,
            "fold_a_content_access": False,
            "formal_holdout_content_access": False,
            "field_photo_content_access": False,
            "result_based_model_or_checkpoint_selection": False,
            "threshold_gate": False,
            "retry": False,
            "automatic_execution_or_advancement": False,
        },
        "comparison": {
            "method_order": list(METHOD_ORDER),
            "same_batch_and_input_tensor_objects_for_all_methods": True,
            "one_shared_twin_endpoint_forward_for_q0_qs_fixed_and_a15_2": True,
            "terminal_a11_full_scort_q0_bit_exact_shared_q0": True,
            "q_sarn_fail_closed": True,
            "fixed_geometric_and_a15_2_fail_closed": True,
            "all_five_methods_use_full_condition_denominators": True,
        },
        "checkpoints": checkpoint_metadata,
        "data": {
            "fold_b_manifest": str(manifest),
            "physical_fold_b_samples": len(samples),
            "physical_fold_b_scenes": scenes,
            "conditions": list(EVALUATION_CONDITIONS),
            "projective_conditions": list(PROJECTIVE_CONDITIONS),
            "condition_sample_rosters_exact_manifest_order": True,
            "fold_b_manifest_loaded_after_both_checkpoint_validations": True,
            "correction_train_sample_and_scene_overlap": False,
            "other_partition_content_access": False,
        },
        "conditions": conditions,
        "pooled": pooled,
        "projective_scene_block_bootstrap": bootstrap,
        "evidence": {
            "per_row_target_means_absolute_errors_scenes_conditions_saved": True,
            "aggregate_nmae_cvar25_wtl_recomputable_from_per_row": True,
            "availability_and_exact_fallback_saved_per_row": True,
            "posterior_mass_and_cdf_diagnostics_saved_per_row": True,
            "a15_2_serialized_layer_cdfs_checked_exact": True,
            "terminal_a11_full_layer_cdfs_recomputed_from_posteriors": True,
            "no_prediction_row_was_used_for_selection_or_control_flow": True,
        },
    }
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise A152FoldBEvaluationError(
            f"Fold-B output already exists: {output}"
        ) from exc
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-b-manifest", type=Path, required=True)
    parser.add_argument("--terminal-a11-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--terminal-a15-2-correction-checkpoint", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_a15_2_fold_b_once(
        fold_b_manifest_path=args.fold_b_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        terminal_a15_2_correction_checkpoint_path=(
            args.terminal_a15_2_correction_checkpoint
        ),
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "protocol": result["protocol"],
                "pooled": result["pooled"],
                "projective_scene_block_bootstrap": result[
                    "projective_scene_block_bootstrap"
                ],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A152FoldBEvaluationError",
    "EVALUATION_BATCH_SIZE",
    "PROTOCOL",
    "build_argument_parser",
    "evaluate_a15_2_fold_b_once",
    "evaluate_same_batch_loader",
    "load_fixed_final_models_before_fold_b",
    "pool_condition_results",
]
