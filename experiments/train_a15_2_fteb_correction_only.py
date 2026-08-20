"""Train the final A15.2 correction on all physical correction-train rows.

The only accepted data input is the 6,616-sample, 60-scene A13 physical
correction-train manifest.  Every physical sample is materialized under the
same three fixed projective conditions in one optimizer step, so a physical
batch of eight becomes 24 aligned rows.  The terminal A11 model is evaluated
separately on Raw and SARN pixels under ``eval`` and ``no_grad`` and remains
bit-exact; only a fresh fixed-quarter :class:`A152FTEBCorrection` is optimized.

There is no validation input, intermediate selection, clipping, EMA, or
automatic advancement decision in this runner.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from experiments.a15_1_fteb_targets import (
    LOSS_DESIGN_METADATA,
    LOSS_WEIGHTS,
    a15_1_fteb_loss,
    build_a15_1_targets,
)
from experiments.a15_2_fteb import (
    A15_2_ARCHITECTURE,
    A15_2_LEARNED_RESIDUAL_SCALE,
    A152FTEBCorrection,
)
from experiments.a15_2_fteb_oof_protocol import (
    PROTOCOL as SELECTION_PROTOCOL,
)
from experiments.a15_fteb import (
    frozen_twin_endpoint_forward,
    fteb_parameter_counts,
)
from experiments.a15_fteb_inner_scene_probe_protocol import (
    PIXEL_CONDITION_SEED,
    PROJECTIVE_CONDITIONS,
    SHARED_INITIALIZATION_SEED,
)
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    PROTOCOL as DATA_PROTOCOL,
    load_a13_correction_train_manifest,
)
from experiments.a13_correction_dev_protocol import (
    CORRECTION_TRAIN_SCENE_COUNT,
    CORRECTION_TRAIN_SCENES,
    CORRECTION_TRAIN_SAMPLES,
)
from experiments.resnet18_direct_progress import DirectSample
from experiments.run_a12_fixed_q0_causal_probe import (
    DEFAULT_TERMINAL_A11_CHECKPOINT,
    load_frozen_terminal_a11,
    module_states_bit_exact,
    posterior_integrity_evidence,
)
from experiments.run_a15_fteb_inner_scene_probe import (
    _posterior_mass_cdf_evidence,
    forward_a15_correction,
)
from experiments.train_a11_scort_syncg import (
    _autocast_settings,
    optimizer_and_scaler_state_finite_evidence,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _configure_reproducibility,
)


PROTOCOL: Final[str] = "syncg_a15_2_fteb_correction_only_terminal5_v1"
SYSTEM: Final[str] = "a15_2_fteb_fixed_quarter_correction_only"
TERMINAL_EPOCHS: Final[int] = 5
DEFAULT_EPOCHS: Final[int] = TERMINAL_EPOCHS
PHYSICAL_BATCH_SIZE: Final[int] = 8
CONDITIONS_PER_PHYSICAL: Final[int] = len(PROJECTIVE_CONDITIONS)
ROW_BATCH_SIZE: Final[int] = PHYSICAL_BATCH_SIZE * CONDITIONS_PER_PHYSICAL
LEARNING_RATE: Final[float] = 3.0e-4
WEIGHT_DECAY: Final[float] = 1.0e-4
INITIALIZATION_SEED: Final[int] = SHARED_INITIALIZATION_SEED
PIXEL_AUGMENTATION_SEED: Final[int] = PIXEL_CONDITION_SEED
SAMPLE_ORDER_SEED: Final[int] = 20_262_219
EXPECTED_STEPS_PER_EPOCH: Final[int] = math.ceil(
    CORRECTION_TRAIN_SAMPLES / PHYSICAL_BATCH_SIZE
)
EXPECTED_TOTAL_OPTIMIZER_STEPS: Final[int] = (
    TERMINAL_EPOCHS * EXPECTED_STEPS_PER_EPOCH
)
SEMANTIC_GROUP_NAMES: Final[tuple[str, ...]] = (
    "shared_relation_encoder",
    "geometry_token_encoder",
    "bin_endpoint_encoder",
    "bin_relation_decoder",
    "natural_parameter_bridge",
)
SCALAR_LOSS_COMPONENTS: Final[tuple[str, ...]] = (
    "total",
    "final_read",
    "dense_cdf_path",
    "geometric_excess_mae",
    "absolute_final_cvar25",
    "cross_condition_w1",
    "learned_delta_energy",
    "sarn_endpoint_ce_metric",
    "sarn_endpoint_cdf_metric",
)
ACCESS_FLAGS: Final[dict[str, bool]] = {
    "correction_train_manifest_access": True,
    "terminal_a11_checkpoint_access": True,
    "correction_dev_manifest_access": False,
    "core_audit_manifest_access": False,
    "fold_a_content_access": False,
    "fold_b_content_access": False,
    "formal_holdout_content_access": False,
    "field_photo_content_access": False,
}


class A152TrainingError(ValueError):
    """An A15.2 full-training input, runtime contract, or state is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152TrainingError(message)


def configure_a15_2_reproducibility(
    device: torch.device,
    *,
    seed: int = SAMPLE_ORDER_SEED,
) -> None:
    _configure_reproducibility(int(seed), device)
    torch.use_deterministic_algorithms(True, warn_only=False)


def epoch_sample_order_seed(
    epoch: int,
    *,
    base_seed: int = SAMPLE_ORDER_SEED,
) -> int:
    value = int(epoch)
    _require(0 <= value < TERMINAL_EPOCHS, "A15.2 epoch is out of range")
    return int(base_seed) + value


class A152ThreeConditionTrainDataset(
    Dataset[tuple[dict[str, Any], ...]]
):
    """Return one physical sample as a synchronized three-condition tuple.

    All three backing datasets use identical ``seed/epoch/index`` inputs for
    the base photo augmentation.  Their only intentional difference is the
    explicitly fixed projective condition.
    """

    def __init__(
        self,
        samples: Sequence[DirectSample],
        *,
        seed: int = PIXEL_AUGMENTATION_SEED,
        total_epochs: int = TERMINAL_EPOCHS,
        dataset_factory: Callable[..., Dataset[dict[str, Any]]] = (
            SyncGSupportGeometryMultiViewDataset
        ),
    ) -> None:
        self.samples = tuple(samples)
        self.seed = int(seed)
        self.total_epochs = int(total_epochs)
        self.epoch = 0
        _require(bool(self.samples), "A15.2 correction-train samples are empty")
        _require(
            self.seed == PIXEL_AUGMENTATION_SEED,
            "A15.2 pixel/augmentation seed differs",
        )
        _require(
            self.total_epochs == TERMINAL_EPOCHS,
            "A15.2 terminal epoch count differs",
        )
        _require(
            tuple(PROJECTIVE_CONDITIONS)
            == (
                "perspective_moderate",
                "perspective_severe",
                "combined_severe",
            ),
            "A15.2 projective condition roster differs",
        )
        self._datasets = {
            condition: dataset_factory(
                self.samples,
                training=True,
                seed=self.seed,
                total_epochs=self.total_epochs,
                condition=condition,
            )
            for condition in PROJECTIVE_CONDITIONS
        }

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        _require(0 <= value < self.total_epochs, "A15.2 dataset epoch differs")
        self.epoch = value
        for dataset in self._datasets.values():
            setter = getattr(dataset, "set_epoch", None)
            _require(callable(setter), "A15.2 backing dataset has no set_epoch")
            setter(value)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], ...]:
        position = int(index)
        _require(0 <= position < len(self.samples), "A15.2 sample index differs")
        rows = tuple(
            dict(self._datasets[condition][position])
            for condition in PROJECTIVE_CONDITIONS
        )
        sample_ids = {str(row["sample_id"]) for row in rows}
        scene_stems = {str(row["scene_stem"]) for row in rows}
        conditions = tuple(str(row["condition_name"]) for row in rows)
        targets = tuple(row["target"] for row in rows)
        _require(
            len(sample_ids) == len(scene_stems) == 1
            and conditions == tuple(PROJECTIVE_CONDITIONS),
            "A15.2 synchronized physical tuple metadata differs",
        )
        _require(
            all(
                isinstance(value, torch.Tensor)
                and torch.equal(value, targets[0])
                for value in targets
            ),
            "A15.2 synchronized physical tuple target differs",
        )
        for row in rows:
            row["source_index"] = position
            row["pixel_epoch"] = self.epoch
        return rows


def build_a15_2_train_dataset(
    samples: Sequence[DirectSample],
) -> A152ThreeConditionTrainDataset:
    values = tuple(samples)
    if CORRECTION_TRAIN_SAMPLES > 0:
        _require(
            len(values) == CORRECTION_TRAIN_SAMPLES,
            "A15.2 correction-train sample count differs",
        )
    if CORRECTION_TRAIN_SCENES:
        _require(
            {sample.scene_stem for sample in values}
            == {Path(scene).stem for scene in CORRECTION_TRAIN_SCENES},
            "A15.2 correction-train scene roster differs",
        )
    return A152ThreeConditionTrainDataset(values)


def collate_a15_2_physical_triplets(
    physical_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Flatten physical-major triplets and attach local group identifiers."""

    groups = tuple(tuple(rows) for rows in physical_rows)
    _require(bool(groups), "A15.2 physical batch is empty")
    _require(
        all(len(rows) == CONDITIONS_PER_PHYSICAL for rows in groups),
        "A15.2 physical batch does not contain condition triplets",
    )
    flattened: list[Mapping[str, Any]] = []
    for rows in groups:
        _require(
            tuple(str(row["condition_name"]) for row in rows)
            == tuple(PROJECTIVE_CONDITIONS),
            "A15.2 physical condition order differs",
        )
        _require(
            len({str(row["sample_id"]) for row in rows}) == 1
            and len({str(row["scene_stem"]) for row in rows}) == 1
            and len({int(row["source_index"]) for row in rows}) == 1,
            "A15.2 physical grouping metadata differs",
        )
        flattened.extend(rows)
    batch = dict(default_collate(flattened))
    batch["physical_group_ids"] = torch.arange(
        len(groups), dtype=torch.long
    ).repeat_interleave(CONDITIONS_PER_PHYSICAL)
    return batch


def build_a15_2_epoch_loader(
    dataset: A152ThreeConditionTrainDataset,
    *,
    epoch: int,
    workers: int,
    cuda: bool,
    sample_order_seed: int = SAMPLE_ORDER_SEED,
) -> DataLoader[dict[str, Any]]:
    _require(workers >= 0, "A15.2 workers must be non-negative")
    dataset.set_epoch(epoch)
    return DataLoader(
        dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=int(workers),
        pin_memory=bool(cuda),
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(
            epoch_sample_order_seed(epoch, base_seed=sample_order_seed)
        ),
        collate_fn=collate_a15_2_physical_triplets,
    )


def _device_value(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {name: _device_value(item, device) for name, item in value.items()}
    return value


def _cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {name: _cpu_value(item) for name, item in value.items()}
    return value


def _state_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _module_state_finite(module: nn.Module) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in module.state_dict().values()
    )


def _anchor_runtime_evidence(anchor: nn.Module) -> dict[str, bool]:
    return {
        "eval_mode": not anchor.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in anchor.parameters()
        ),
        "all_gradients_absent": all(
            parameter.grad is None for parameter in anchor.parameters()
        ),
        "state_finite": _module_state_finite(anchor),
    }


def semantic_parameter_groups(
    model: A152FTEBCorrection,
) -> dict[str, tuple[nn.Parameter, ...]]:
    """Partition every trainable tensor into five diagnostic groups."""

    decoder = model.bin_decoder
    groups = {
        "shared_relation_encoder": tuple(
            model.shared_relation_encoder.parameters()
        ),
        "geometry_token_encoder": tuple(
            model.geometry_token_encoder.parameters()
        ),
        "bin_endpoint_encoder": (
            decoder.progress_position_embedding,
            *tuple(decoder.endpoint_embedding.parameters()),
        ),
        "bin_relation_decoder": (
            *tuple(decoder.decoder.parameters()),
            *tuple(decoder.output_norm.parameters()),
        ),
        "natural_parameter_bridge": tuple(
            model.natural_parameter_bridge.parameters()
        ),
    }
    _require(
        tuple(groups) == SEMANTIC_GROUP_NAMES and all(groups.values()),
        "A15.2 semantic group roster differs",
    )
    grouped_ids = {
        id(parameter) for values in groups.values() for parameter in values
    }
    _require(
        len(grouped_ids) == sum(len(values) for values in groups.values())
        and grouped_ids == {id(parameter) for parameter in model.parameters()},
        "A15.2 semantic groups overlap or do not cover the correction",
    )
    return groups


def _gradient_summary(parameters: Sequence[nn.Parameter]) -> dict[str, Any]:
    gradients = tuple(
        parameter.grad for parameter in parameters if parameter.grad is not None
    )
    finite = all(bool(value is not None and torch.isfinite(value).all()) for value in gradients)
    absolute_sum = (
        sum(float(value.detach().abs().double().sum().cpu()) for value in gradients)
        if finite
        else float("nan")
    )
    return {
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero": bool(finite and absolute_sum > 0.0),
        "absolute_sum": absolute_sum,
    }


def correction_optimizer_partition_evidence(
    anchor: nn.Module,
    correction: A152FTEBCorrection,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    correction_ids = {id(parameter) for parameter in correction.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    anchor_ids = {id(parameter) for parameter in anchor.parameters()}
    return {
        "optimizer_exactly_correction_parameters": optimizer_ids == correction_ids,
        "optimizer_excludes_terminal_a11": optimizer_ids.isdisjoint(anchor_ids),
        "all_correction_parameters_trainable": all(
            parameter.requires_grad for parameter in correction.parameters()
        ),
        "all_terminal_a11_parameters_frozen": all(
            not parameter.requires_grad for parameter in anchor.parameters()
        ),
        "correction_parameter_count": sum(
            parameter.numel() for parameter in correction.parameters()
        ),
        "optimizer_parameter_count": sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ),
    }


def _exact_q0_rows(output: Mapping[str, Any], rows: torch.Tensor) -> bool:
    selected = rows.detach().bool()
    if not bool(selected.any()):
        return True
    q0 = output["raw_anchor_posterior"][selected]
    q0_cdf = output["raw_anchor_cdf"][selected]
    q0_mean = output["raw_anchor_mean"][selected]
    q0_variance = output["raw_anchor_variance"][selected]
    layers = output["layer_posteriors"][selected]
    layer_cdfs = output["layer_cdfs"][selected]
    layer_means = output["layer_means"][selected]
    layer_variances = output["layer_variances"][selected]
    return (
        torch.equal(output["progress_posterior"][selected], q0)
        and torch.equal(output["progress_cdf"][selected], q0_cdf)
        and torch.equal(output["mean"][selected], q0_mean)
        and torch.equal(output["variance"][selected], q0_variance)
        and torch.equal(
            layers, q0[:, None].expand_as(layers)
        )
        and torch.equal(
            layer_cdfs, q0_cdf[:, None].expand_as(layer_cdfs)
        )
        and torch.equal(
            layer_means, q0_mean[:, None].expand_as(layer_means)
        )
        and torch.equal(
            layer_variances, q0_variance[:, None].expand_as(layer_variances)
        )
        and torch.equal(output["sarn_endpoint_posterior"][selected], q0)
        and torch.equal(output["sarn_endpoint_cdf"][selected], q0_cdf)
        and torch.equal(output["sarn_endpoint_mean"][selected], q0_mean)
        and torch.equal(output["sarn_endpoint_variance"][selected], q0_variance)
    )


def _output_subset(output: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "progress_posterior",
        "progress_cdf",
        "mean",
        "variance",
        "raw_anchor_posterior",
        "raw_anchor_cdf",
        "raw_anchor_mean",
        "raw_anchor_variance",
        "sarn_endpoint_posterior",
        "geometric_base",
        "layer_posteriors",
        "layer_cdfs",
        "relation_available",
    )
    return {name: output[name].detach().cpu() for name in names}


def _outputs_bit_exact(
    expected: Mapping[str, torch.Tensor], observed: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(expected) == tuple(observed) and all(
        torch.equal(expected[name], observed[name]) for name in expected
    )


def _fresh_strict_load_evidence(
    correction: A152FTEBCorrection,
    construction: Mapping[str, int],
    replay_batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    correction.eval()
    batch = _device_value(replay_batch, device)
    with torch.no_grad():
        expected = forward_a15_correction(correction, batch, endpoint_null=False)
    terminal_state = _state_cpu(correction)
    fresh = A152FTEBCorrection(
        **{name: int(value) for name, value in construction.items()}
    ).to(device)
    incompatibility = fresh.load_state_dict(terminal_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        observed = forward_a15_correction(fresh, batch, endpoint_null=False)
    integrity = _posterior_mass_cdf_evidence(observed)
    return {
        "missing_keys": list(incompatibility.missing_keys),
        "unexpected_keys": list(incompatibility.unexpected_keys),
        "strict_load_clean": (
            not incompatibility.missing_keys and not incompatibility.unexpected_keys
        ),
        "state_bit_exact": module_states_bit_exact(
            terminal_state, _state_cpu(fresh)
        ),
        "same_input_reported_outputs_bit_exact": _outputs_bit_exact(
            _output_subset(expected), _output_subset(observed)
        ),
        "fresh_state_finite": _module_state_finite(fresh),
        "fresh_posterior_mass_cdf": integrity,
    }


def run_a15_2_correction_epoch(
    anchor: nn.Module,
    correction: A152FTEBCorrection,
    loader: Any,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    anchor_state_reference: Mapping[str, torch.Tensor],
    max_steps: int | None = None,
    force_first_row_inactive: bool = False,
    record_parameter_updates: bool = False,
) -> dict[str, Any]:
    """Run one full or smoke-prefix epoch through the real dynamic train path."""

    anchor.eval()
    correction.train()
    runtime = _anchor_runtime_evidence(anchor)
    _require(all(runtime.values()), "A15.2 frozen anchor runtime contract differs")
    semantic = semantic_parameter_groups(correction)
    autocast_enabled, autocast_dtype, precision = _autocast_settings(device)
    total_rows = 0
    total_physical = 0
    active_rows_total = 0
    active_groups_total = 0
    steps = 0
    condition_counts: Counter[str] = Counter()
    source_index_order: list[int] = []
    gradient_finite_every_step = {name: True for name in semantic}
    gradient_nonzero_seen = {name: False for name in semantic}
    optimizer_state_finite_every_step = True
    correction_state_finite_every_step = True
    anchor_runtime_contract_every_step = True
    q0_output_identity_every_step = True
    inactive_exact_q0_every_step = True
    posterior_mass_cdf_every_step = True
    mixed_active_inactive_seen = False
    scalar_weighted_sums = {name: 0.0 for name in SCALAR_LOSS_COMPONENTS}
    trace: list[dict[str, Any]] = []
    last_replay_batch: dict[str, Any] | None = None
    initial_active_exact_fixed_geometric: bool | None = None

    batches = loader if max_steps is None else itertools.islice(loader, int(max_steps))
    for raw_batch in batches:
        batch = {
            name: _device_value(raw_batch[name], device)
            for name in (
                "target",
                "physical_group_ids",
                "original_view",
                "sarn_view",
                "sarn_support_mask",
                "sarn_active",
                "raw_to_sarn_homography",
            )
        }
        row_count = int(batch["target"].shape[0])
        _require(
            row_count % CONDITIONS_PER_PHYSICAL == 0,
            "A15.2 row batch is not physical-triplet aligned",
        )
        physical_count = row_count // CONDITIONS_PER_PHYSICAL
        group_ids = batch["physical_group_ids"].long()
        _require(
            tuple(str(value) for value in raw_batch["condition_name"])
            == tuple(PROJECTIVE_CONDITIONS) * physical_count
            and all(
                int((group_ids == group).sum()) == CONDITIONS_PER_PHYSICAL
                for group in group_ids.unique().tolist()
            ),
            "A15.2 synchronized condition/group batch differs",
        )
        if force_first_row_inactive:
            batch["sarn_active"] = batch["sarn_active"].clone()
            batch["sarn_active"][0] = False

        optimizer.zero_grad(set_to_none=True)
        anchor.eval()
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            endpoints = frozen_twin_endpoint_forward(
                anchor, batch["original_view"], batch["sarn_view"]
            )
            correction_batch = {
                "raw_posterior": endpoints["raw_posterior"],
                "sarn_posterior": endpoints["sarn_posterior"],
                "raw_mean": endpoints["raw_mean"],
                "sarn_mean": endpoints["sarn_mean"],
                "raw_features": endpoints["raw_features"],
                "sarn_features": endpoints["sarn_features"],
                "sarn_support_mask": batch["sarn_support_mask"],
                "sarn_active": batch["sarn_active"],
                "raw_to_sarn_homography": batch["raw_to_sarn_homography"],
            }
            output = forward_a15_correction(
                correction, correction_batch, endpoint_null=False
            )
            targets = build_a15_1_targets({"target": batch["target"]})
            loss, components = a15_1_fteb_loss(output, targets, group_ids)

        _require(bool(torch.isfinite(loss)), "A15.2 correction loss is non-finite")
        coverage = components["coverage"]
        active = coverage["active"].detach().bool()
        inactive = ~active
        fallback_rows = ~output["relation_available"].detach().bool()
        active_count = int(active.sum().cpu())
        inactive_count = row_count - active_count
        active_groups = int(components["cross_condition_active_group_count"])
        if initial_active_exact_fixed_geometric is None:
            initial_active_exact_fixed_geometric = (
                active_count > 0
                and bool(output["delta_zero"][active].all())
                and torch.equal(
                    output["progress_posterior"][active],
                    output["proposed_geometric_base"][active],
                )
            )
        q0_identity = (
            torch.equal(
                output["raw_anchor_posterior"], endpoints["raw_posterior"]
            )
            and torch.equal(output["raw_anchor_mean"], endpoints["raw_mean"])
            and torch.equal(
                output["raw_anchor_variance"], endpoints["raw_variance"]
            )
        )
        inactive_exact = _exact_q0_rows(output, fallback_rows)
        integrity = _posterior_mass_cdf_evidence(output)
        _require(q0_identity, "A15.2 q0 output identity differs")
        _require(inactive_exact, "A15.2 inactive fallback differs from exact q0")
        _require(integrity["valid"], "A15.2 posterior mass/CDF differs")

        parameters_before = (
            {
                name: tuple(parameter.detach().clone() for parameter in values)
                for name, values in semantic.items()
            }
            if record_parameter_updates
            else {}
        )
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        step_gradient = {
            name: _gradient_summary(parameters)
            for name, parameters in semantic.items()
        }
        _require(
            all(summary["finite"] for summary in step_gradient.values()),
            "A15.2 semantic gradient is non-finite",
        )
        for name, summary in step_gradient.items():
            gradient_finite_every_step[name] &= bool(summary["finite"])
            gradient_nonzero_seen[name] |= bool(summary["nonzero"])
        _require(
            all(parameter.grad is None for parameter in anchor.parameters()),
            "A15.2 correction loss reached the terminal A11 anchor",
        )
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        parameter_updated = (
            {
                name: any(
                    not torch.equal(previous, current.detach())
                    for previous, current in zip(
                        parameters_before[name], semantic[name], strict=True
                    )
                )
                for name in semantic
            }
            if record_parameter_updates
            else {}
        )
        optimizer_evidence = optimizer_and_scaler_state_finite_evidence(
            optimizer, scaler
        )
        correction_finite = _module_state_finite(correction)
        anchor_step = _anchor_runtime_evidence(anchor)
        _require(
            bool(optimizer_evidence["finite"])
            and correction_finite
            and all(anchor_step.values()),
            "A15.2 optimizer/correction/anchor runtime state differs",
        )

        optimizer_state_finite_every_step &= bool(optimizer_evidence["finite"])
        correction_state_finite_every_step &= correction_finite
        anchor_runtime_contract_every_step &= all(anchor_step.values())
        q0_output_identity_every_step &= q0_identity
        inactive_exact_q0_every_step &= inactive_exact
        posterior_mass_cdf_every_step &= bool(integrity["valid"])
        mixed_active_inactive_seen |= active_count > 0 and inactive_count > 0
        condition_counts.update(str(value) for value in raw_batch["condition_name"])
        physical_sources = [
            int(raw_batch["source_index"][offset])
            for offset in range(0, row_count, CONDITIONS_PER_PHYSICAL)
        ]
        _require(
            len(set(physical_sources)) == len(physical_sources),
            "A15.2 physical batch repeats a source sample",
        )
        source_index_order.extend(physical_sources)
        for name in SCALAR_LOSS_COMPONENTS:
            scalar_weighted_sums[name] += (
                float(components[name].detach().cpu()) * max(active_count, 1)
            )
        steps += 1
        total_rows += row_count
        total_physical += physical_count
        active_rows_total += active_count
        active_groups_total += active_groups
        trace.append(
            {
                "step": steps,
                "physical_samples": physical_count,
                "rows": row_count,
                "source_indices": physical_sources,
                "active_rows": active_count,
                "inactive_rows": inactive_count,
                "fallback_rows": int(fallback_rows.sum().cpu()),
                "active_fraction": active_count / float(row_count),
                "cross_condition_active_groups": active_groups,
                "loss": float(loss.detach().cpu()),
                "components": {
                    name: float(components[name].detach().cpu())
                    for name in SCALAR_LOSS_COMPONENTS
                },
                "gradient": step_gradient,
                "parameter_updated": parameter_updated,
                "optimizer_state_finite": bool(optimizer_evidence["finite"]),
                "correction_state_finite": correction_finite,
                "anchor_runtime_contract": anchor_step,
                "q0_output_identity": q0_identity,
                "inactive_exact_q0": inactive_exact,
                "posterior_mass_cdf": integrity,
            }
        )
        last_replay_batch = _cpu_value(correction_batch)

    _require(steps > 0 and total_physical > 0, "A15.2 epoch produced no samples")
    _require(
        len(source_index_order) == len(set(source_index_order)),
        "A15.2 epoch repeated a physical sample",
    )
    anchor_state_exact = module_states_bit_exact(
        dict(anchor_state_reference), _state_cpu(anchor)
    )
    _require(anchor_state_exact, "A15.2 terminal A11 state changed")
    denominator = float(max(active_rows_total, 1))
    return {
        "physical_samples": total_physical,
        "condition_rows": total_rows,
        "steps": steps,
        "optimizer_steps": steps,
        "source_index_order": source_index_order,
        "condition_counts": dict(sorted(condition_counts.items())),
        "active_rows": active_rows_total,
        "inactive_rows": total_rows - active_rows_total,
        "active_fraction": active_rows_total / float(total_rows),
        "cross_condition_active_groups": active_groups_total,
        "loss_components_active_row_weighted": {
            name: value / denominator
            for name, value in scalar_weighted_sums.items()
        },
        "loss_denominator": "read_valid_and_relation_available_rows_only",
        "cross_condition_denominator": "complete_three_row_all_active_physical_groups_only",
        "gradient_finite_every_step": gradient_finite_every_step,
        "gradient_nonzero_seen": gradient_nonzero_seen,
        "optimizer_state_finite_every_step": optimizer_state_finite_every_step,
        "correction_state_finite_every_step": correction_state_finite_every_step,
        "anchor_runtime_contract_every_step": anchor_runtime_contract_every_step,
        "anchor_state_bit_exact_at_epoch_end": anchor_state_exact,
        "q0_output_identity_every_step": q0_output_identity_every_step,
        "inactive_exact_q0_every_step": inactive_exact_q0_every_step,
        "posterior_mass_cdf_every_step": posterior_mass_cdf_every_step,
        "mixed_active_inactive_seen": mixed_active_inactive_seen,
        "initial_active_exact_fixed_geometric": bool(
            initial_active_exact_fixed_geometric
        ),
        "autocast_precision": precision,
        "gradient_clipping": None,
        "step_trace": trace,
        "_last_replay_batch": last_replay_batch,
    }


def _terminal_fallback_evidence(
    correction: A152FTEBCorrection,
    replay_batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    batch = _device_value(replay_batch, device)
    batch["sarn_active"] = torch.zeros_like(
        batch["sarn_active"], dtype=torch.bool
    )
    correction.eval()
    with torch.no_grad():
        output = forward_a15_correction(correction, batch, endpoint_null=False)
    rows = torch.ones_like(output["relation_available"], dtype=torch.bool)
    integrity = _posterior_mass_cdf_evidence(output)
    evidence = {
        "relation_available_all_false": not bool(
            output["relation_available"].any()
        ),
        "posterior_cdf_layers_and_moments_exact_q0": _exact_q0_rows(
            output, rows
        ),
        "posterior_mass_cdf": integrity,
    }
    _require(
        evidence["relation_available_all_false"]
        and evidence["posterior_cdf_layers_and_moments_exact_q0"]
        and integrity["valid"],
        "A15.2 terminal all-row fallback differs",
    )
    return evidence


def train_a15_2_correction_only(
    *,
    correction_train_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
    workers: int = 6,
) -> dict[str, Any]:
    """Train a fresh A15.2 correction for exactly five terminal epochs."""

    _require(workers >= 0, "A15.2 workers must be non-negative")
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A15.2 terminal output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "A15.2 device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_a15_2_reproducibility(device)

    samples = tuple(
        load_a13_correction_train_manifest(
            Path(correction_train_manifest_path).resolve()
        )
    )
    dataset = build_a15_2_train_dataset(samples)
    anchor, terminal_evidence, construction = load_frozen_terminal_a11(
        terminal_a11_checkpoint_path, device=device
    )
    anchor.eval()
    anchor_initial_state = _state_cpu(anchor)
    torch.manual_seed(INITIALIZATION_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    correction = A152FTEBCorrection(
        **{name: int(value) for name, value in construction.items()}
    ).to(device)
    _require(
        correction.learned_residual_scale == A15_2_LEARNED_RESIDUAL_SCALE,
        "A15.2 learned residual scale differs",
    )
    optimizer = torch.optim.AdamW(
        correction.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    partition = correction_optimizer_partition_evidence(
        anchor, correction, optimizer
    )
    _require(
        all(
            bool(partition[name])
            for name in (
                "optimizer_exactly_correction_parameters",
                "optimizer_excludes_terminal_a11",
                "all_correction_parameters_trainable",
                "all_terminal_a11_parameters_frozen",
            )
        ),
        "A15.2 correction-only optimizer partition differs",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TERMINAL_EPOCHS
    )
    use_fp16_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    history: list[dict[str, Any]] = []
    total_steps = 0
    replay_batch: Mapping[str, Any] | None = None
    condition_totals: Counter[str] = Counter()
    for epoch in range(TERMINAL_EPOCHS):
        loader = build_a15_2_epoch_loader(
            dataset, epoch=epoch, workers=workers, cuda=device.type == "cuda"
        )
        metrics = run_a15_2_correction_epoch(
            anchor,
            correction,
            loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            anchor_state_reference=anchor_initial_state,
        )
        replay_batch = metrics.pop("_last_replay_batch")
        expected_conditions = {
            condition: len(samples) for condition in PROJECTIVE_CONDITIONS
        }
        _require(
            metrics["physical_samples"] == len(samples)
            and metrics["condition_rows"]
            == len(samples) * CONDITIONS_PER_PHYSICAL
            and metrics["steps"] == math.ceil(len(samples) / PHYSICAL_BATCH_SIZE)
            and sorted(metrics["source_index_order"])
            == list(range(len(samples))),
            "A15.2 epoch does not cover every physical sample exactly once",
        )
        _require(
            metrics["condition_counts"] == expected_conditions,
            "A15.2 epoch condition coverage differs",
        )
        _require(
            all(metrics["gradient_finite_every_step"].values())
            and all(metrics["gradient_nonzero_seen"].values()),
            "A15.2 epoch semantic gradient evidence differs",
        )
        _require(
            metrics["optimizer_state_finite_every_step"]
            and metrics["correction_state_finite_every_step"]
            and metrics["anchor_runtime_contract_every_step"]
            and metrics["anchor_state_bit_exact_at_epoch_end"]
            and metrics["q0_output_identity_every_step"]
            and metrics["inactive_exact_q0_every_step"]
            and metrics["posterior_mass_cdf_every_step"],
            "A15.2 epoch runtime integrity differs",
        )
        total_steps += int(metrics["optimizer_steps"])
        condition_totals.update(metrics["condition_counts"])
        history.append(
            {
                "epoch": epoch + 1,
                "sample_order_seed": epoch_sample_order_seed(epoch),
                "pixel_augmentation_seed": PIXEL_AUGMENTATION_SEED,
                "pixel_epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "metrics": metrics,
            }
        )
        scheduler.step()

    expected_total_steps = TERMINAL_EPOCHS * math.ceil(
        len(samples) / PHYSICAL_BATCH_SIZE
    )
    _require(
        total_steps == expected_total_steps == EXPECTED_TOTAL_OPTIMIZER_STEPS,
        "A15.2 total optimizer step count differs",
    )
    _require(replay_batch is not None, "A15.2 has no terminal replay batch")
    _require(_module_state_finite(correction), "A15.2 terminal state is non-finite")
    anchor_exact = module_states_bit_exact(anchor_initial_state, _state_cpu(anchor))
    _require(anchor_exact, "A15.2 terminal A11 is not bit-exact")
    terminal_optimizer = optimizer_and_scaler_state_finite_evidence(
        optimizer, scaler
    )
    _require(
        bool(terminal_optimizer["finite"]),
        "A15.2 terminal AdamW/scaler state is non-finite",
    )
    fresh = _fresh_strict_load_evidence(
        correction, construction, replay_batch, device=device
    )
    _require(
        fresh["strict_load_clean"]
        and fresh["state_bit_exact"]
        and fresh["same_input_reported_outputs_bit_exact"]
        and fresh["fresh_state_finite"]
        and fresh["fresh_posterior_mass_cdf"]["valid"],
        "A15.2 fresh strict-load replay differs",
    )
    fallback = _terminal_fallback_evidence(
        correction, replay_batch, device=device
    )
    scene_roster = tuple(sorted({sample.scene_stem for sample in samples}))
    _require(
        len(scene_roster) == CORRECTION_TRAIN_SCENE_COUNT,
        "A15.2 correction-train scene count differs",
    )

    training = {
        "initialization_seed": INITIALIZATION_SEED,
        "sample_order_seed": SAMPLE_ORDER_SEED,
        "sample_order_seed_rule": "base_plus_zero_based_epoch",
        "pixel_augmentation_seed": PIXEL_AUGMENTATION_SEED,
        "pixel_augmentation_epoch_rule": "same_seed_epoch_index_across_three_fixed_conditions",
        "epochs": TERMINAL_EPOCHS,
        "terminal_checkpoint_selection": "epoch_5_no_validation_selection",
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "conditions_per_physical": CONDITIONS_PER_PHYSICAL,
        "row_batch_size_full": ROW_BATCH_SIZE,
        "last_row_batch_size": (
            (CORRECTION_TRAIN_SAMPLES % PHYSICAL_BATCH_SIZE)
            or PHYSICAL_BATCH_SIZE
        ) * CONDITIONS_PER_PHYSICAL,
        "physical_samples_per_epoch": len(samples),
        "condition_rows_per_epoch": len(samples) * CONDITIONS_PER_PHYSICAL,
        "steps_per_epoch": EXPECTED_STEPS_PER_EPOCH,
        "optimizer_steps": total_steps,
        "optimizer": "AdamW_A15.2_correction_only",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR_Tmax5_epoch_end_step",
        "scheduler_steps": TERMINAL_EPOCHS,
        "condition_counts_terminal_total": dict(sorted(condition_totals.items())),
        "projective_conditions": list(PROJECTIVE_CONDITIONS),
        "clean_presentations": 0,
        "gradient_clipping": None,
        "ema": None,
        "amp": "bfloat16_if_supported_else_float16_cuda",
        "validation_manifest": None,
        "intermediate_checkpoint_selection": None,
    }
    checkpoint = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "data_protocol": DATA_PROTOCOL,
        "system": SYSTEM,
        "architecture": A15_2_ARCHITECTURE,
        "learned_residual_scale": A15_2_LEARNED_RESIDUAL_SCALE,
        "source_terminal_a11": {
            **terminal_evidence,
            "source": str(Path(terminal_a11_checkpoint_path).resolve()),
            "state_bit_exact_terminal": anchor_exact,
            "eval_frozen_gradient_free_every_step": all(
                row["metrics"]["anchor_runtime_contract_every_step"]
                for row in history
            ),
        },
        "construction": dict(construction),
        "parameter_counts": fteb_parameter_counts(correction),
        "optimizer_partition_evidence": partition,
        "terminal_optimizer_and_scaler_state": terminal_optimizer,
        "training": training,
        "physical_correction_train": {
            "manifest": str(Path(correction_train_manifest_path).resolve()),
            "samples": len(samples),
            "scenes": len(scene_roster),
            "scene_roster": list(scene_roster),
            "sample_ids": [sample.sample_id for sample in samples],
        },
        "loss_design": {
            **dict(LOSS_DESIGN_METADATA),
            "loss_function": "a15_1_fteb_loss",
            "inactive_rows_excluded_from_every_correction_denominator": True,
            "cross_condition_only_complete_all_active_three_row_groups": True,
        },
        "loss_weights": LOSS_WEIGHTS.as_dict(),
        "semantic_gradient_groups": list(SEMANTIC_GROUP_NAMES),
        "history": history,
        "access_flags": dict(ACCESS_FLAGS),
        "anchor_unchanged_evidence": {
            "state_bit_exact_terminal": anchor_exact,
            "state_bit_exact_each_epoch": [
                row["metrics"]["anchor_state_bit_exact_at_epoch_end"]
                for row in history
            ],
            "runtime_contract_every_step": [
                row["metrics"]["anchor_runtime_contract_every_step"]
                for row in history
            ],
            "q0_output_identity_every_step": [
                row["metrics"]["q0_output_identity_every_step"]
                for row in history
            ],
        },
        "fresh_strict_load": fresh,
        "terminal_fallback_evidence": fallback,
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "automatic_execution_or_advancement_control": False,
        "model_state": _state_cpu(correction),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {
        "status": "complete",
        "terminal_checkpoint": str(output),
        "optimizer_steps": total_steps,
        "physical_samples": len(samples),
        "scenes": len(scene_roster),
        "condition_counts": training["condition_counts_terminal_total"],
        "anchor_state_bit_exact": anchor_exact,
        "fresh_strict_load": fresh,
        "terminal_fallback_evidence": fallback,
        "access_flags": dict(ACCESS_FLAGS),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correction-train-manifest",
        type=Path,
        default=DEFAULT_CORRECTION_TRAIN_MANIFEST,
    )
    parser.add_argument(
        "--terminal-a11-checkpoint",
        type=Path,
        default=DEFAULT_TERMINAL_A11_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = train_a15_2_correction_only(
        correction_train_manifest_path=args.correction_train_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCESS_FLAGS",
    "A152ThreeConditionTrainDataset",
    "A152TrainingError",
    "CONDITIONS_PER_PHYSICAL",
    "DEFAULT_EPOCHS",
    "EXPECTED_STEPS_PER_EPOCH",
    "EXPECTED_TOTAL_OPTIMIZER_STEPS",
    "INITIALIZATION_SEED",
    "LEARNING_RATE",
    "PHYSICAL_BATCH_SIZE",
    "PIXEL_AUGMENTATION_SEED",
    "PROTOCOL",
    "ROW_BATCH_SIZE",
    "SAMPLE_ORDER_SEED",
    "SEMANTIC_GROUP_NAMES",
    "SYSTEM",
    "TERMINAL_EPOCHS",
    "WEIGHT_DECAY",
    "build_a15_2_epoch_loader",
    "build_a15_2_train_dataset",
    "build_argument_parser",
    "collate_a15_2_physical_triplets",
    "configure_a15_2_reproducibility",
    "correction_optimizer_partition_evidence",
    "epoch_sample_order_seed",
    "run_a15_2_correction_epoch",
    "semantic_parameter_groups",
    "train_a15_2_correction_only",
]
