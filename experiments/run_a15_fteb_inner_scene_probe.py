"""Run the A15 FTEB matched inner-scene probe on correction-train only.

The only data population accepted by this command is the 6,616-row A13
correction-train manifest.  A frozen terminal A11 anchor is called separately
on Raw and SARN pixels to materialize one canonical twin-endpoint cache.  The
target-blind 48/12 scene partition supplies 384 training and 96 unseen-scene
physical samples, each expanded to the same three projective conditions.

Two A15 correction arms start from bit-identical, storage-disjoint states and
consume the same 200x8 physical-group schedule.  The full arm receives the
frozen q_sarn posterior endpoint.  The endpoint-null arm receives q0 in that
one slot while retaining the exact same SARN features, support, homography,
topology, optimizer, loss, and row order.  Inner-eval is read once after both
arms finish; no correction-dev, Core-audit, Fold, formal, or field input is
accepted by this module.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from experiments.a15_fteb import (
    A15FTEBCorrection,
    fixed_geometric_natural_parameter_base,
    frozen_twin_endpoint_forward,
    fteb_parameter_counts,
)
from experiments.a15_fteb_inner_scene_probe_protocol import (
    ACCESS_EXPECTATION,
    BATCH_SCHEDULE_SEED,
    DESCRIPTIVE_REFERENCE,
    DEVELOPMENT_SCOPE,
    ENDPOINT_CACHE_SEMANTICS,
    EVAL_CACHE_ROWS,
    EVAL_PHYSICAL_SAMPLES,
    FIXED_GEOMETRIC_SEMANTICS,
    INNER_EVAL_SCENES,
    INNER_TRAIN_SCENES,
    MATCHED_ARM_SEMANTICS,
    METHOD_A15_LEARNED,
    METHOD_DIRECT_QS,
    METHOD_ENDPOINT_NULL,
    METHOD_FIXED_GEOMETRIC,
    METHOD_ORDER,
    METHOD_Q0,
    PHYSICAL_BATCH_SIZE,
    PHYSICAL_SAMPLES_PER_SCENE,
    PIXEL_CONDITION_SEED,
    PROJECTIVE_CONDITIONS,
    PROTOCOL as SCIENTIFIC_PROTOCOL,
    ROW_BATCH_SIZE,
    SHARED_INITIALIZATION_SEED,
    SOURCE_SAMPLES,
    TRAIN_CACHE_ROWS,
    TRAIN_PHYSICAL_SAMPLES,
    TRAIN_STEPS,
    expand_physical_indices_to_cache_rows,
    fixed_physical_batch_schedule,
    inner_eval_metric_report,
    per_scene_candidate_order,
    schedule_presentation_counts,
    select_relation_complete_physical_samples,
)
from experiments.a15_fteb_targets import build_a15_targets, a15_fteb_loss
from experiments.prepare_a13_correction_scene_split import (
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    PROTOCOL as CORRECTION_MANIFEST_PROTOCOL,
    load_a13_correction_train_manifest,
)
from experiments.run_a12_fixed_q0_causal_probe import (
    DEFAULT_TERMINAL_A11_CHECKPOINT,
    load_frozen_terminal_a11,
    module_states_bit_exact,
    posterior_integrity_evidence,
)
from experiments.resnet18_direct_progress import IMAGE_SIZE
from experiments.train_a11_scort_syncg import (
    _autocast_settings,
    optimizer_and_scaler_state_finite_evidence,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (
    SyncGSupportGeometryMultiViewDataset,
    _configure_reproducibility,
)


PROTOCOL: Final[str] = "syncg_a15_fteb_inner_scene_matched_runner_v1"
LEARNING_RATE: Final[float] = 3.0e-4
WEIGHT_DECAY: Final[float] = 1.0e-4
OPTIMIZER: Final[str] = "AdamW"
SCHEDULER: Final[None] = None
CANDIDATE_PHYSICAL_BATCH_SIZE: Final[int] = 8
EVAL_ROW_BATCH_SIZE: Final[int] = ROW_BATCH_SIZE
STEP_NOISE_SEED_OFFSET: Final[int] = 75_000_023
SCALAR_LOSS_COMPONENTS: Final[tuple[str, ...]] = (
    "total",
    "final_read",
    "dense_cdf_path",
    "geometric_softplus_regret",
    "absolute_final_cvar25",
    "cross_condition_w1",
    "learned_delta_energy",
    "sarn_endpoint_ce_metric",
    "sarn_endpoint_cdf_metric",
)
SEMANTIC_GROUPS: Final[tuple[str, ...]] = (
    "shared_relation_encoder",
    "geometry_token_encoder",
    "bin_decoder",
    "natural_parameter_bridge",
)


class A15ProbeRunError(ValueError):
    """An A15 cache, matched arm, objective, or artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A15ProbeRunError(message)


def _device_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=device.type == "cuda")


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


def _posterior_moments(
    posterior: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = posterior.detach().float()
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    grid = torch.linspace(
        0.0,
        1.0,
        probability.shape[1],
        device=probability.device,
        dtype=torch.float32,
    )
    mean = (probability * grid[None]).sum(dim=1)
    variance = (
        probability * (grid[None] - mean[:, None]).square()
    ).sum(dim=1)
    return mean, variance


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").clone()


def _seed_step(
    step: int,
    device: torch.device,
    *,
    seed_base: int = BATCH_SCHEDULE_SEED,
) -> None:
    seed = int(seed_base) + STEP_NOISE_SEED_OFFSET + int(step)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class A15EndpointCache:
    """One CPU twin-endpoint cache in physical-major condition order."""

    partition: str
    sample_ids: tuple[str, ...]
    scene_stems: tuple[str, ...]
    condition_names: tuple[str, ...]
    source_indices: tuple[int, ...]
    physical_group_ids: torch.Tensor
    target: torch.Tensor
    sarn_support_mask: torch.Tensor
    sarn_active: torch.Tensor
    raw_to_sarn_homography: torch.Tensor
    relation_available: torch.Tensor
    raw_posterior: torch.Tensor
    sarn_posterior: torch.Tensor
    raw_mean: torch.Tensor
    sarn_mean: torch.Tensor
    raw_stride8: torch.Tensor
    raw_stride16: torch.Tensor
    sarn_stride8: torch.Tensor
    sarn_stride16: torch.Tensor
    expected_scene_stems: tuple[str, ...] | None = None

    @property
    def rows(self) -> int:
        return len(self.sample_ids)

    @property
    def physical_samples(self) -> int:
        return self.rows // len(PROJECTIVE_CONDITIONS)

    def validate(self) -> None:
        default_scenes = {
            "inner_train": INNER_TRAIN_SCENES,
            "inner_eval": INNER_EVAL_SCENES,
        }
        _require(self.partition in default_scenes, "A15 cache partition differs")
        scenes = (
            tuple(self.expected_scene_stems)
            if self.expected_scene_stems is not None
            else default_scenes[self.partition]
        )
        physical = len(scenes) * PHYSICAL_SAMPLES_PER_SCENE
        rows = physical * len(PROJECTIVE_CONDITIONS)
        _require(
            self.rows == rows and self.physical_samples == physical,
            f"A15 {self.partition} cache size differs",
        )
        _require(
            len(self.scene_stems)
            == len(self.condition_names)
            == len(self.source_indices)
            == rows,
            "A15 cache metadata lengths differ",
        )
        _require(
            set(self.scene_stems) == set(scenes)
            and set(self.condition_names) == set(PROJECTIVE_CONDITIONS),
            "A15 cache scene/condition roster differs",
        )
        vector_shapes = (
            self.physical_group_ids.shape,
            self.target.shape,
            self.sarn_active.shape,
            self.relation_available.shape,
            self.raw_mean.shape,
            self.sarn_mean.shape,
        )
        _require(
            all(shape == (rows,) for shape in vector_shapes),
            "A15 cache vector shapes differ",
        )
        _require(
            self.physical_group_ids.dtype == torch.long
            and self.sarn_active.dtype == torch.bool
            and self.relation_available.dtype == torch.bool
            and bool(self.sarn_active.all())
            and bool(self.relation_available.all()),
            "A15 selected cache is not relation-active",
        )
        _require(
            self.raw_posterior.shape == self.sarn_posterior.shape == (rows, 128),
            "A15 endpoint posterior shapes differ",
        )
        _require(
            self.sarn_support_mask.ndim == 4
            and self.sarn_support_mask.shape[:2] == (rows, 1)
            and self.raw_to_sarn_homography.shape == (rows, 3, 3),
            "A15 support/homography shapes differ",
        )
        _require(
            self.raw_stride8.shape == self.sarn_stride8.shape
            and self.raw_stride16.shape == self.sarn_stride16.shape
            and self.raw_stride8.ndim == self.raw_stride16.ndim == 4
            and self.raw_stride8.shape[0] == self.raw_stride16.shape[0] == rows,
            "A15 twin feature shapes differ",
        )
        tensors = (
            self.physical_group_ids,
            self.target,
            self.sarn_support_mask,
            self.sarn_active,
            self.raw_to_sarn_homography,
            self.relation_available,
            self.raw_posterior,
            self.sarn_posterior,
            self.raw_mean,
            self.sarn_mean,
            self.raw_stride8,
            self.raw_stride16,
            self.sarn_stride8,
            self.sarn_stride16,
        )
        _require(
            all(value.device.type == "cpu" for value in tensors),
            "A15 endpoint cache must reside on CPU",
        )
        _require(
            all(
                not value.is_floating_point() or bool(torch.isfinite(value).all())
                for value in tensors
            ),
            "A15 endpoint cache contains a non-finite tensor",
        )
        for name, posterior in (
            ("q0", self.raw_posterior),
            ("q_sarn", self.sarn_posterior),
        ):
            _require(
                bool((posterior >= 0.0).all())
                and bool(
                    torch.allclose(
                        posterior.sum(dim=1),
                        torch.ones(rows),
                        rtol=0.0,
                        atol=1.0e-6,
                    )
                ),
                f"A15 cached {name} is not a probability distribution",
            )
        observed_raw_mean, _ = _posterior_moments(self.raw_posterior)
        observed_sarn_mean, _ = _posterior_moments(self.sarn_posterior)
        _require(
            torch.allclose(observed_raw_mean, self.raw_mean, rtol=1.0e-6, atol=1.0e-7)
            and torch.allclose(
                observed_sarn_mean, self.sarn_mean, rtol=1.0e-6, atol=1.0e-7
            ),
            "A15 cached endpoint means differ from their posteriors",
        )
        for physical_index in range(physical):
            start = physical_index * len(PROJECTIVE_CONDITIONS)
            stop = start + len(PROJECTIVE_CONDITIONS)
            _require(
                tuple(self.condition_names[start:stop]) == PROJECTIVE_CONDITIONS
                and len(set(self.sample_ids[start:stop])) == 1
                and len(set(self.scene_stems[start:stop])) == 1
                and len(set(self.source_indices[start:stop])) == 1
                and bool((self.physical_group_ids[start:stop] == physical_index).all())
                and bool(
                    torch.equal(
                        self.target[start:stop],
                        self.target[start].expand(len(PROJECTIVE_CONDITIONS)),
                    )
                ),
                "A15 physical-major three-condition grouping differs",
            )

    def batch(
        self,
        indices: Sequence[int],
        *,
        device: torch.device,
    ) -> dict[str, Any]:
        row_indices = tuple(int(value) for value in indices)
        _require(bool(row_indices), "A15 cache batch is empty")
        _require(
            min(row_indices) >= 0 and max(row_indices) < self.rows,
            "A15 cache row index is out of range",
        )
        index = torch.tensor(row_indices, dtype=torch.long)

        def take(value: torch.Tensor) -> torch.Tensor:
            return _device_tensor(value.index_select(0, index), device)

        return {
            "target": take(self.target),
            "physical_group_ids": take(self.physical_group_ids),
            "sarn_support_mask": take(self.sarn_support_mask),
            "sarn_active": take(self.sarn_active),
            "raw_to_sarn_homography": take(self.raw_to_sarn_homography),
            "raw_posterior": take(self.raw_posterior),
            "sarn_posterior": take(self.sarn_posterior),
            "raw_mean": take(self.raw_mean),
            "sarn_mean": take(self.sarn_mean),
            "raw_features": {
                "stride8": take(self.raw_stride8),
                "stride16": take(self.raw_stride16),
            },
            "sarn_features": {
                "stride8": take(self.sarn_stride8),
                "stride16": take(self.sarn_stride16),
            },
        }


def forward_a15_correction(
    model: A15FTEBCorrection,
    batch: Mapping[str, Any],
    *,
    endpoint_null: bool,
) -> dict[str, Any]:
    """Run FTEB; endpoint-null changes only q_sarn := q0."""

    q_sarn = batch["raw_posterior"] if endpoint_null else batch["sarn_posterior"]
    return model(
        batch["raw_posterior"],
        batch["raw_features"],
        q_sarn,
        batch["sarn_features"],
        batch["sarn_support_mask"],
        sarn_active=batch["sarn_active"],
        raw_to_sarn_homography=batch["raw_to_sarn_homography"],
    )


def _candidate_rows(
    datasets: Mapping[str, SyncGSupportGeometryMultiViewDataset],
    refs: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[tuple[int, str]]]:
    items: list[dict[str, Any]] = []
    keys: list[tuple[int, str]] = []
    for ref in refs:
        for condition in PROJECTIVE_CONDITIONS:
            item = datasets[condition][ref.source_index]
            _require(
                str(item["sample_id"]) == ref.sample_id
                and str(item["scene_stem"]) == ref.scene_stem
                and str(item["condition_name"]) == condition,
                "A15 candidate pixel row metadata differs",
            )
            items.append(item)
            keys.append((int(ref.source_index), condition))
    return items, keys


def _cache_from_payloads(
    *,
    partition: str,
    selected: Sequence[Any],
    payloads: Mapping[tuple[int, str], Mapping[str, Any]],
    expected_scene_stems: Sequence[str] | None = None,
) -> A15EndpointCache:
    metadata: dict[str, list[Any]] = {
        "sample_ids": [],
        "scene_stems": [],
        "condition_names": [],
        "source_indices": [],
    }
    tensor_names = (
        "target",
        "sarn_support_mask",
        "sarn_active",
        "raw_to_sarn_homography",
        "relation_available",
        "raw_posterior",
        "sarn_posterior",
        "raw_mean",
        "sarn_mean",
        "raw_stride8",
        "raw_stride16",
        "sarn_stride8",
        "sarn_stride16",
    )
    tensors: dict[str, list[torch.Tensor]] = {name: [] for name in tensor_names}
    group_ids: list[int] = []
    for physical_index, ref in enumerate(selected):
        for condition in PROJECTIVE_CONDITIONS:
            key = (int(ref.source_index), condition)
            _require(key in payloads, "A15 selected endpoint payload is missing")
            payload = payloads[key]
            metadata["sample_ids"].append(str(payload["sample_id"]))
            metadata["scene_stems"].append(str(payload["scene_stem"]))
            metadata["condition_names"].append(str(payload["condition_name"]))
            metadata["source_indices"].append(int(ref.source_index))
            group_ids.append(physical_index)
            for name in tensor_names:
                tensors[name].append(payload[name])
    cache = A15EndpointCache(
        partition=partition,
        physical_group_ids=torch.tensor(group_ids, dtype=torch.long),
        expected_scene_stems=(
            tuple(str(scene) for scene in expected_scene_stems)
            if expected_scene_stems is not None
            else None
        ),
        **{name: tuple(values) for name, values in metadata.items()},
        **{name: torch.stack(values) for name, values in tensors.items()},
    )
    cache.validate()
    return cache


def materialize_a15_endpoint_caches(
    *,
    correction_train_manifest_path: Path,
    frozen_terminal_model: nn.Module,
    construction: Mapping[str, int],
    device: torch.device,
    partition_scene_rosters: Mapping[str, Sequence[str]] | None = None,
    relation_complete_start_by_scene: Mapping[str, int] | None = None,
    pixel_condition_seed: int = PIXEL_CONDITION_SEED,
) -> tuple[dict[str, A15EndpointCache], dict[str, Any]]:
    """Scan target-blind candidates and cache 480 physical twin endpoints.

    The optional roster/rank arguments support correction-train-only role
    rotation.  Defaults reproduce the original A15 48/12 split and select the
    first eight relation-complete candidates in every scene.
    """

    source = Path(correction_train_manifest_path).resolve()
    samples = load_a13_correction_train_manifest(source)
    _require(len(samples) == SOURCE_SAMPLES, "A15 source sample count differs")
    candidate_order = per_scene_candidate_order(
        sample_ids=tuple(sample.sample_id for sample in samples),
        scene_stems=tuple(sample.scene_stem for sample in samples),
    )
    rosters = (
        {
            "inner_train": tuple(INNER_TRAIN_SCENES),
            "inner_eval": tuple(INNER_EVAL_SCENES),
        }
        if partition_scene_rosters is None
        else {
            partition: tuple(Path(str(scene)).stem for scene in scenes)
            for partition, scenes in partition_scene_rosters.items()
        }
    )
    _require(
        set(rosters) == {"inner_train", "inner_eval"}
        and len(rosters["inner_train"]) == 48
        and len(rosters["inner_eval"]) == 12
        and not (set(rosters["inner_train"]) & set(rosters["inner_eval"]))
        and set(rosters["inner_train"]) | set(rosters["inner_eval"])
        == set(candidate_order),
        "A15 cache scene rosters differ",
    )
    starts = (
        {scene: 0 for scene in candidate_order}
        if relation_complete_start_by_scene is None
        else {
            Path(str(scene)).stem: int(value)
            for scene, value in relation_complete_start_by_scene.items()
        }
    )
    _require(
        set(starts) == set(candidate_order) and all(value >= 0 for value in starts.values()),
        "A15 relation-complete selection offsets differ",
    )
    required_by_scene = {
        scene: starts[scene] + PHYSICAL_SAMPLES_PER_SCENE
        for scene in candidate_order
    }
    datasets = {
        condition: SyncGSupportGeometryMultiViewDataset(
            samples,
            training=False,
            seed=int(pixel_condition_seed),
            total_epochs=1,
            image_size=IMAGE_SIZE,
            condition=condition,
        )
        for condition in PROJECTIVE_CONDITIONS
    }
    selection_model = A15FTEBCorrection(
        **{name: int(value) for name, value in construction.items()}
    ).eval().to(device)
    relation_available: dict[tuple[int, str], bool] = {}
    payloads: dict[tuple[int, str], dict[str, Any]] = {}
    accepted_by_scene: dict[str, int] = {
        scene: 0 for scene in candidate_order
    }
    relation_complete_by_scene: dict[str, list[Any]] = {
        scene: [] for scene in candidate_order
    }
    frontier = {scene: 0 for scene in candidate_order}
    scanned_physical = 0
    endpoint_calls = 0
    autocast_enabled, autocast_dtype, autocast_name = _autocast_settings(device)
    frozen_terminal_model.eval()
    selection_model.eval()
    while any(
        accepted_by_scene[scene] < required_by_scene[scene]
        for scene in accepted_by_scene
    ):
        wave: list[Any] = []
        for scene in sorted(candidate_order):
            if accepted_by_scene[scene] >= required_by_scene[scene]:
                continue
            position = frontier[scene]
            _require(
                position < len(candidate_order[scene]),
                f"A15 exhausted target-blind candidates for scene {scene}",
            )
            wave.append(candidate_order[scene][position])
            frontier[scene] = position + 1
        for offset in range(0, len(wave), CANDIDATE_PHYSICAL_BATCH_SIZE):
            refs = wave[offset : offset + CANDIDATE_PHYSICAL_BATCH_SIZE]
            items, keys = _candidate_rows(datasets, refs)
            raw_batch = default_collate(items)
            original = _device_tensor(raw_batch["original_view"], device)
            sarn = _device_tensor(raw_batch["sarn_view"], device)
            support = _device_tensor(raw_batch["sarn_support_mask"], device)
            active = _device_tensor(raw_batch["sarn_active"], device)
            homography = _device_tensor(
                raw_batch["raw_to_sarn_homography"], device
            )
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
                dtype=autocast_dtype,
            ):
                endpoints = frozen_twin_endpoint_forward(
                    frozen_terminal_model, original, sarn
                )
                output = selection_model(
                    endpoints["raw_posterior"],
                    endpoints["raw_features"],
                    endpoints["sarn_posterior"],
                    endpoints["sarn_features"],
                    support,
                    sarn_active=active,
                    raw_to_sarn_homography=homography,
                )
            endpoint_calls += 1
            availability = output["relation_available"].detach().cpu().bool()
            _require(
                availability.shape == (len(keys),),
                "A15 candidate availability shape differs",
            )
            for row, key in enumerate(keys):
                relation_available[key] = bool(availability[row])
            for physical_offset, ref in enumerate(refs):
                row_start = physical_offset * len(PROJECTIVE_CONDITIONS)
                row_stop = row_start + len(PROJECTIVE_CONDITIONS)
                physical_active = bool(availability[row_start:row_stop].all())
                scanned_physical += 1
                if not physical_active:
                    continue
                accepted_rank = accepted_by_scene[ref.scene_stem]
                accepted_by_scene[ref.scene_stem] += 1
                relation_complete_by_scene[ref.scene_stem].append(ref)
                if accepted_rank < starts[ref.scene_stem]:
                    continue
                for local_row, condition in enumerate(PROJECTIVE_CONDITIONS):
                    row = row_start + local_row
                    key = (int(ref.source_index), condition)
                    payloads[key] = {
                        "sample_id": str(raw_batch["sample_id"][row]),
                        "scene_stem": str(raw_batch["scene_stem"][row]),
                        "condition_name": condition,
                        # Target is copied only after all three availability
                        # booleans accepted this physical candidate.
                        "target": _cpu_clone(raw_batch["target"][row]),
                        "sarn_support_mask": _cpu_clone(
                            raw_batch["sarn_support_mask"][row]
                        ),
                        "sarn_active": _cpu_clone(raw_batch["sarn_active"][row]),
                        "raw_to_sarn_homography": _cpu_clone(
                            raw_batch["raw_to_sarn_homography"][row]
                        ),
                        "relation_available": _cpu_clone(availability[row]),
                        "raw_posterior": _cpu_clone(
                            endpoints["raw_posterior"][row]
                        ),
                        "sarn_posterior": _cpu_clone(
                            endpoints["sarn_posterior"][row]
                        ),
                        "raw_mean": _cpu_clone(endpoints["raw_mean"][row]),
                        "sarn_mean": _cpu_clone(endpoints["sarn_mean"][row]),
                        "raw_stride8": _cpu_clone(
                            endpoints["raw_features"]["stride8"][row]
                        ),
                        "raw_stride16": _cpu_clone(
                            endpoints["raw_features"]["stride16"][row]
                        ),
                        "sarn_stride8": _cpu_clone(
                            endpoints["sarn_features"]["stride8"][row]
                        ),
                        "sarn_stride16": _cpu_clone(
                            endpoints["sarn_features"]["stride16"][row]
                        ),
                    }
            del original, sarn, support, active, homography, endpoints, output
    selected_by_scene = {
        scene: tuple(
            relation_complete_by_scene[scene][
                starts[scene] : starts[scene] + PHYSICAL_SAMPLES_PER_SCENE
            ]
        )
        for scene in candidate_order
    }
    _require(
        all(len(refs) == PHYSICAL_SAMPLES_PER_SCENE for refs in selected_by_scene.values()),
        "A15 relation-complete selection count differs",
    )
    selected = {
        partition: tuple(
            ref for scene in rosters[partition] for ref in selected_by_scene[scene]
        )
        for partition in ("inner_train", "inner_eval")
    }
    caches = {
        partition: _cache_from_payloads(
            partition=partition,
            selected=selected[partition],
            payloads=payloads,
            expected_scene_stems=rosters[partition],
        )
        for partition in ("inner_train", "inner_eval")
    }
    _require(
        len(payloads) == TRAIN_CACHE_ROWS + EVAL_CACHE_ROWS,
        "A15 accepted endpoint payload count differs",
    )
    selected_indices = {
        partition: [int(ref.source_index) for ref in refs]
        for partition, refs in selected.items()
    }
    evidence = {
        "correction_train_manifest": str(source),
        "correction_train_manifest_protocol": CORRECTION_MANIFEST_PROTOCOL,
        "source_rows": len(samples),
        "source_scenes": len(candidate_order),
        "selection_inputs": [
            "sample_id",
            "scene_stem",
            "fixed_seeded_candidate_order",
            "relation_available_all_three_conditions",
        ],
        "selection_excluded_inputs": [
            "target",
            "q0_error",
            "q_sarn_error",
            "geometric_error",
            "learned_error",
        ],
        "scanned_physical_candidates": scanned_physical,
        "endpoint_forward_batches": endpoint_calls,
        "selected_physical": TRAIN_PHYSICAL_SAMPLES + EVAL_PHYSICAL_SAMPLES,
        "selected_rows": TRAIN_CACHE_ROWS + EVAL_CACHE_ROWS,
        "selected_source_indices": selected_indices,
        "selected_sample_ids": {
            partition: [ref.sample_id for ref in refs]
            for partition, refs in selected.items()
        },
        "selected_scene_stems": {
            partition: [ref.scene_stem for ref in refs]
            for partition, refs in selected.items()
        },
        "partition_scene_rosters": {
            partition: list(scenes) for partition, scenes in rosters.items()
        },
        "relation_complete_start_by_scene": dict(sorted(starts.items())),
        "condition_counts": {
            partition: dict(sorted(Counter(cache.condition_names).items()))
            for partition, cache in caches.items()
        },
        "raw_and_sarn_anchor_calls_separate": True,
        "canonical_q0": "raw_only_forward_bits",
        "terminal_a11_eval_no_grad": True,
        "endpoint_autocast_enabled": autocast_enabled,
        "endpoint_autocast_dtype": autocast_name,
        "cached_feature_dtypes": {
            partition: {
                "raw_stride8": str(cache.raw_stride8.dtype),
                "sarn_stride8": str(cache.sarn_stride8.dtype),
            }
            for partition, cache in caches.items()
        },
        "target_blind_selection": True,
        "pixel_materialization": {
            "image_size": IMAGE_SIZE,
            "training": False,
            "total_epochs": 1,
            "effective_epoch": 0,
            "seed": int(pixel_condition_seed),
            "same_seed_and_epoch_for_all_three_conditions": True,
        },
    }
    return caches, evidence


def _state_equality_and_storage(
    left: nn.Module, right: nn.Module
) -> dict[str, Any]:
    left_state = left.state_dict()
    right_state = right.state_dict()
    _require(tuple(left_state) == tuple(right_state), "A15 matched state keys differ")
    equal = {
        name: torch.equal(left_state[name], right_state[name])
        for name in left_state
    }
    disjoint = {
        name: left_state[name].untyped_storage().data_ptr()
        != right_state[name].untyped_storage().data_ptr()
        for name in left_state
    }
    return {
        "tensor_equal": equal,
        "storage_disjoint": disjoint,
        "all_tensor_equal": all(equal.values()),
        "all_storage_disjoint": all(disjoint.values()),
    }


def build_matched_a15_models(
    construction: Mapping[str, int],
) -> tuple[A15FTEBCorrection, A15FTEBCorrection, dict[str, Any]]:
    """Build two state-identical A15 arms with independent tensor storage."""

    kwargs = {name: int(value) for name, value in construction.items()}
    torch.manual_seed(SHARED_INITIALIZATION_SEED)
    full = A15FTEBCorrection(**kwargs)
    torch.manual_seed(SHARED_INITIALIZATION_SEED)
    endpoint_null = A15FTEBCorrection(**kwargs)
    load = endpoint_null.load_state_dict(full.state_dict(), strict=True)
    _require(
        not load.missing_keys and not load.unexpected_keys,
        "A15 matched initialization strict copy differs",
    )
    comparison = _state_equality_and_storage(full, endpoint_null)
    _require(
        comparison["all_tensor_equal"] and comparison["all_storage_disjoint"],
        "A15 matched arms are not state-identical and storage-disjoint",
    )
    return full, endpoint_null, {
        "seed": SHARED_INITIALIZATION_SEED,
        "same_initial_state": True,
        "storage_disjoint": True,
        "strict_state_copy": True,
        "state_comparison": comparison,
        "only_treatment_difference": "q_sarn_endpoint_versus_q0_endpoint",
    }


def _model_factory(construction: Mapping[str, int]) -> Callable[[], A15FTEBCorrection]:
    kwargs = {name: int(value) for name, value in construction.items()}
    return lambda: A15FTEBCorrection(**kwargs)


def semantic_parameter_groups(
    model: A15FTEBCorrection,
) -> dict[str, tuple[nn.Parameter, ...]]:
    groups = {
        name: tuple(getattr(model, name).parameters())
        for name in SEMANTIC_GROUPS
    }
    _require(
        all(groups.values())
        and len({id(parameter) for values in groups.values() for parameter in values})
        == sum(len(values) for values in groups.values()),
        "A15 semantic parameter groups are empty or overlap",
    )
    _require(
        {id(parameter) for values in groups.values() for parameter in values}
        == {id(parameter) for parameter in model.parameters()},
        "A15 semantic parameter groups do not cover the correction",
    )
    return groups


def _gradient_summary(parameters: Sequence[nn.Parameter]) -> dict[str, Any]:
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    finite = all(bool(torch.isfinite(value).all()) for value in gradients)
    absolute_sum = (
        sum(float(value.detach().abs().double().sum().cpu()) for value in gradients)
        if finite
        else 0.0
    )
    return {
        "gradient_tensors": len(gradients),
        "finite": finite,
        "nonzero": absolute_sum > 0.0,
        "absolute_sum": absolute_sum,
    }


def _posterior_mass_cdf_evidence(output: Mapping[str, Any]) -> dict[str, Any]:
    final = output["progress_posterior"].detach().float()
    final_cdf = output["progress_cdf"].detach().float()
    layers = output["layer_posteriors"].detach().float()
    layer_cdfs = output["layer_cdfs"].detach().float()
    _require(
        layers.ndim == 3
        and layer_cdfs.shape == layers.shape
        and final.shape == final_cdf.shape == layers[:, -1].shape,
        "A15 posterior/CDF evidence shapes differ",
    )
    layer_cdf_exact = all(
        torch.equal(layer_cdfs[:, index], layers[:, index].cumsum(dim=1))
        for index in range(layers.shape[1])
    )
    maximum_mass_error = float(
        torch.abs(layers.sum(dim=2) - 1.0).max().detach().cpu()
    )
    minimum_probability = float(layers.min().detach().cpu())
    evidence = {
        "final_cdf_exact_from_posterior": torch.equal(
            final_cdf, final.cumsum(dim=1)
        ),
        "layer_cdf_exact_from_posterior": layer_cdf_exact,
        "maximum_absolute_layer_mass_error": maximum_mass_error,
        "minimum_layer_probability": minimum_probability,
        "finite": bool(torch.isfinite(layers).all())
        and bool(torch.isfinite(layer_cdfs).all()),
    }
    evidence["valid"] = (
        evidence["final_cdf_exact_from_posterior"]
        and evidence["layer_cdf_exact_from_posterior"]
        and evidence["maximum_absolute_layer_mass_error"] <= 1.0e-6
        and evidence["minimum_layer_probability"] >= 0.0
        and evidence["finite"]
    )
    return evidence


def _anchor_inputs_gradient_free(batch: Mapping[str, Any]) -> bool:
    tensors = (
        batch["raw_posterior"],
        batch["sarn_posterior"],
        batch["raw_features"]["stride8"],
        batch["raw_features"]["stride16"],
        batch["sarn_features"]["stride8"],
        batch["sarn_features"]["stride16"],
    )
    return all(not value.requires_grad and value.grad is None for value in tensors)


def _output_subset(output: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    names = (
        "progress_posterior",
        "progress_cdf",
        "mean",
        "variance",
        "raw_anchor_posterior",
        "raw_anchor_cdf",
        "raw_anchor_mean",
        "sarn_endpoint_posterior",
        "geometric_base",
        "relation_available",
    )
    return {name: output[name].detach().cpu() for name in names}


def _outputs_bit_exact(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> bool:
    return tuple(left) == tuple(right) and all(
        torch.equal(left[name], right[name]) for name in left
    )


def train_a15_probe_arm(
    *,
    arm: str,
    model: A15FTEBCorrection,
    model_factory: Callable[[], A15FTEBCorrection],
    cache: A15EndpointCache,
    schedule: Sequence[Sequence[int]],
    device: torch.device,
    record_step_parameter_updates: bool = True,
    target_builder: Callable[[Mapping[str, Any]], Mapping[str, Any]] = build_a15_targets,
    loss_function: Callable[
        [Mapping[str, Any], Mapping[str, Any], Sequence[Any] | torch.Tensor],
        tuple[torch.Tensor, dict[str, Any]],
    ] = a15_fteb_loss,
    scalar_loss_components: Sequence[str] = SCALAR_LOSS_COMPONENTS,
    objective_name: str = "a15_fteb_loss",
    step_seed_base: int = BATCH_SCHEDULE_SEED,
    expected_nonzero_semantic_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train one full or endpoint-null arm on a shared physical schedule."""

    _require(
        arm in (METHOD_A15_LEARNED, METHOD_ENDPOINT_NULL),
        "A15 training arm differs",
    )
    endpoint_null = arm == METHOD_ENDPOINT_NULL
    batches = tuple(tuple(int(index) for index in batch) for batch in schedule)
    _require(bool(batches), "A15 arm schedule is empty")
    _require(
        all(
            len(batch) == PHYSICAL_BATCH_SIZE
            and len(set(batch)) == PHYSICAL_BATCH_SIZE
            and min(batch) >= 0
            and max(batch) < TRAIN_PHYSICAL_SAMPLES
            for batch in batches
        ),
        "A15 arm physical batch differs",
    )
    cache.validate()
    _require(cache.partition == "inner_train", "A15 training cache partition differs")
    model.to(device)
    semantic = semantic_parameter_groups(model)
    expected_nonzero = (
        set(semantic)
        if expected_nonzero_semantic_groups is None
        else {str(name) for name in expected_nonzero_semantic_groups}
    )
    _require(
        bool(expected_nonzero) and expected_nonzero.issubset(semantic),
        f"{arm}: expected nonzero semantic groups differ",
    )
    expected_zero = set(semantic) - expected_nonzero
    _require(
        all(
            any(parameter.requires_grad for parameter in semantic[name])
            for name in expected_nonzero
        )
        and all(
            not any(parameter.requires_grad for parameter in semantic[name])
            for name in expected_zero
        ),
        f"{arm}: semantic trainability does not match gradient expectations",
    )
    initial_state = _state_cpu(model)
    first_rows = expand_physical_indices_to_cache_rows(batches[0])
    first_batch = cache.batch(first_rows, device=device)
    model.eval()
    with torch.no_grad():
        initial_output = forward_a15_correction(
            model, first_batch, endpoint_null=endpoint_null
        )
    initial_active_exact_fixed_geometric = (
        bool(initial_output["relation_available"].all())
        and bool(initial_output["delta_zero"].all())
        and torch.equal(
            initial_output["progress_posterior"],
            initial_output["proposed_geometric_base"],
        )
    )
    _require(
        initial_active_exact_fixed_geometric,
        f"{arm}: zero learned residual is not the exact fixed geometric endpoint",
    )
    optimizer_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    _require(bool(optimizer_parameters), f"{arm}: no trainable correction parameter")
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    autocast_enabled, autocast_dtype, autocast_name = _autocast_settings(device)
    gradient_nonzero_seen = {name: False for name in semantic}
    gradient_finite_every_step = {name: True for name in semantic}
    optimizer_state_finite_every_step = True
    anchor_inputs_gradient_free_every_step = True
    correction_state_finite_every_step = True
    posterior_mass_cdf_every_step = True
    step_trace: list[dict[str, Any]] = []
    schedule_trace: list[list[int]] = []
    for step, physical_indices in enumerate(batches, start=1):
        _seed_step(step, device, seed_base=step_seed_base)
        rows = expand_physical_indices_to_cache_rows(physical_indices)
        batch = cache.batch(rows, device=device)
        _require(
            all(
                int((batch["physical_group_ids"] == group).sum()) == 3
                for group in batch["physical_group_ids"].unique().tolist()
            ),
            f"{arm}: physical group does not contain exactly three rows",
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parameters_before = {
            name: tuple(parameter.detach().clone() for parameter in parameters)
            for name, parameters in semantic.items()
        }
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=autocast_dtype,
        ):
            output = forward_a15_correction(
                model, batch, endpoint_null=endpoint_null
            )
        expected_sarn = (
            batch["raw_posterior"]
            if endpoint_null
            else batch["sarn_posterior"]
        )
        _require(
            bool(output["relation_available"].all())
            and torch.equal(output["raw_anchor_posterior"], batch["raw_posterior"])
            and torch.equal(output["proposed_sarn_endpoint_posterior"], expected_sarn),
            f"{arm}: canonical endpoint input changed in correction forward",
        )
        targets = target_builder({"target": batch["target"]})
        loss, components = loss_function(
            output, targets, batch["physical_group_ids"]
        )
        _require(bool(torch.isfinite(loss)), f"{arm}: A15 loss is non-finite")
        _require(
            bool(components["coverage"]["active"].all())
            and int(components["cross_condition_active_group_count"])
            == PHYSICAL_BATCH_SIZE,
            f"{arm}: objective coverage differs from 8x3 active rows",
        )
        integrity = _posterior_mass_cdf_evidence(output)
        _require(integrity["valid"], f"{arm}: posterior mass/CDF differs")
        anchor_free = _anchor_inputs_gradient_free(batch)
        _require(anchor_free, f"{arm}: frozen endpoint input acquired gradients")
        loss.backward()
        step_gradient = {
            name: _gradient_summary(parameters)
            for name, parameters in semantic.items()
        }
        _require(
            all(summary["finite"] for summary in step_gradient.values()),
            f"{arm}: a semantic gradient is non-finite",
        )
        for name, summary in step_gradient.items():
            gradient_nonzero_seen[name] = gradient_nonzero_seen[name] or bool(
                summary["nonzero"]
            )
            gradient_finite_every_step[name] = (
                gradient_finite_every_step[name] and bool(summary["finite"])
            )
        optimizer.step()
        parameter_updated = {
            name: any(
                not torch.equal(previous, current.detach())
                for previous, current in zip(
                    parameters_before[name], semantic[name], strict=True
                )
            )
            for name in semantic
        }
        optimizer_evidence = optimizer_and_scaler_state_finite_evidence(
            optimizer, None
        )
        state_finite = _module_state_finite(model)
        _require(
            bool(optimizer_evidence["finite"]) and state_finite,
            f"{arm}: AdamW or correction state is non-finite",
        )
        optimizer_state_finite_every_step &= bool(optimizer_evidence["finite"])
        anchor_inputs_gradient_free_every_step &= anchor_free
        correction_state_finite_every_step &= state_finite
        posterior_mass_cdf_every_step &= bool(integrity["valid"])
        scalar_components = {
            name: float(components[name].detach().cpu())
            for name in scalar_loss_components
        }
        step_trace.append(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "components": scalar_components,
                "cvar_tail_count": int(components["cvar_tail_count"]),
                "cross_condition_active_group_count": int(
                    components["cross_condition_active_group_count"]
                ),
                "gradient": step_gradient,
                "parameter_updated": parameter_updated
                if record_step_parameter_updates
                else {},
                "posterior_mass_cdf": integrity,
            }
        )
        schedule_trace.append(list(physical_indices))
    _require(
        all(gradient_finite_every_step.values())
        and all(gradient_nonzero_seen[name] for name in expected_nonzero)
        and not any(gradient_nonzero_seen[name] for name in expected_zero),
        f"{arm}: semantic task gradients differ from the declared treatment",
    )
    terminal_state = _state_cpu(model)
    semantic_updated_state_tensor_count = {
        name: sum(
            not torch.equal(initial_state[key], terminal_state[key])
            for key in initial_state
            if key.startswith(f"{name}.")
        )
        for name in semantic
    }
    _require(
        all(semantic_updated_state_tensor_count[name] > 0 for name in expected_nonzero)
        and all(
            semantic_updated_state_tensor_count[name] == 0
            for name in expected_zero
        ),
        f"{arm}: semantic parameter updates differ from the declared treatment",
    )
    updated_state_tensor_count = sum(
        not torch.equal(initial_state[name], terminal_state[name])
        for name in initial_state
    )
    _require(updated_state_tensor_count > 0, f"{arm}: no correction state changed")
    model.eval()
    with torch.no_grad():
        terminal_output = forward_a15_correction(
            model, first_batch, endpoint_null=endpoint_null
        )
    terminal_subset = _output_subset(terminal_output)
    terminal_integrity = posterior_integrity_evidence(
        terminal_output["progress_posterior"]
    )
    fallback_batch = dict(first_batch)
    fallback_batch["sarn_active"] = torch.zeros_like(first_batch["sarn_active"])
    with torch.no_grad():
        fallback_output = forward_a15_correction(
            model, fallback_batch, endpoint_null=endpoint_null
        )
    q0_layers = first_batch["raw_posterior"][:, None].expand_as(
        fallback_output["layer_posteriors"]
    )
    sarn_off_fallback_exact_q0 = (
        not bool(fallback_output["relation_available"].any())
        and torch.equal(
            fallback_output["progress_posterior"], first_batch["raw_posterior"]
        )
        and torch.equal(fallback_output["raw_anchor_posterior"], first_batch["raw_posterior"])
        and torch.equal(fallback_output["layer_posteriors"], q0_layers)
        and torch.equal(
            fallback_output["progress_cdf"],
            fallback_output["raw_anchor_cdf"],
        )
        and torch.equal(fallback_output["mean"], fallback_output["raw_anchor_mean"])
    )
    _require(sarn_off_fallback_exact_q0, f"{arm}: fallback is not exact q0")
    fresh = model_factory().to(device)
    load = fresh.load_state_dict(terminal_state, strict=True)
    fresh.eval()
    with torch.no_grad():
        fresh_output = forward_a15_correction(
            fresh, first_batch, endpoint_null=endpoint_null
        )
    fresh_exact = (
        not load.missing_keys
        and not load.unexpected_keys
        and _outputs_bit_exact(terminal_subset, _output_subset(fresh_output))
    )
    _require(fresh_exact, f"{arm}: fresh strict-load output differs")
    terminal_optimizer_state = optimizer_and_scaler_state_finite_evidence(
        optimizer, None
    )
    return {
        "arm": arm,
        "steps": len(batches),
        "physical_samples_per_step": PHYSICAL_BATCH_SIZE,
        "samples_per_step": ROW_BATCH_SIZE,
        "schedule_trace": schedule_trace,
        "optimizer": {
            "class": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer_steps": len(batches),
            "state_finite_every_step": optimizer_state_finite_every_step,
        },
        "scheduler": SCHEDULER,
        "autocast_precision": autocast_name,
        "objective": {
            "function": str(objective_name),
            "active_only": True,
            "physical_groups_per_step": PHYSICAL_BATCH_SIZE,
            "rows_per_physical_group": len(PROJECTIVE_CONDITIONS),
        },
        "initial_active_exact_fixed_geometric": initial_active_exact_fixed_geometric,
        "anchor_inputs_gradient_free_every_step": anchor_inputs_gradient_free_every_step,
        "correction_state_finite_every_step": correction_state_finite_every_step,
        "posterior_mass_cdf_every_step": posterior_mass_cdf_every_step,
        "gradient_nonzero_seen": gradient_nonzero_seen,
        "gradient_finite_every_step": gradient_finite_every_step,
        "semantic_gradient_expectation": {
            name: (
                "finite_nonzero_task_gradient"
                if name in expected_nonzero
                else "no_task_gradient_frozen_unused_path"
            )
            for name in semantic
        },
        "semantic_group_trainable": {
            name: any(parameter.requires_grad for parameter in parameters)
            for name, parameters in semantic.items()
        },
        "semantic_updated_state_tensor_count": (
            semantic_updated_state_tensor_count
        ),
        "updated_state_tensor_count": updated_state_tensor_count,
        "sarn_off_fallback_exact_q0": sarn_off_fallback_exact_q0,
        "fresh_strict_load": {
            "strict": True,
            "output_bit_exact": fresh_exact,
        },
        "terminal_posterior_integrity": terminal_integrity,
        "terminal_optimizer_state": terminal_optimizer_state,
        "step_trace": step_trace,
        "history": step_trace,
        "model_state": terminal_state,
    }


def _evaluate_a15_arm(
    model: A15FTEBCorrection,
    cache: A15EndpointCache,
    *,
    endpoint_null: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    cache.validate()
    _require(cache.partition == "inner_eval", "A15 evaluation cache differs")
    model.eval().to(device)
    names = (
        "progress_posterior",
        "progress_cdf",
        "mean",
        "variance",
        "raw_anchor_posterior",
        "raw_anchor_cdf",
        "raw_anchor_mean",
        "sarn_endpoint_posterior",
        "sarn_endpoint_cdf",
        "sarn_endpoint_mean",
        "geometric_base",
        "geometric_base_cdf",
        "geometric_base_mean",
        "relation_available",
    )
    values: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    autocast_enabled, autocast_dtype, _ = _autocast_settings(device)
    with torch.no_grad():
        for offset in range(0, cache.rows, EVAL_ROW_BATCH_SIZE):
            indices = tuple(
                range(offset, min(offset + EVAL_ROW_BATCH_SIZE, cache.rows))
            )
            batch = cache.batch(indices, device=device)
            with torch.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
                dtype=autocast_dtype,
            ):
                output = forward_a15_correction(
                    model, batch, endpoint_null=endpoint_null
                )
            for name in names:
                values[name].append(output[name].detach().cpu())
    result = {name: torch.cat(parts, dim=0) for name, parts in values.items()}
    _require(
        bool(result["relation_available"].all())
        and torch.equal(result["raw_anchor_posterior"], cache.raw_posterior)
        and torch.equal(result["raw_anchor_mean"], cache.raw_mean),
        "A15 inner-eval canonical q0 or availability differs",
    )
    return result


def _method_integrity(
    posteriors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    _require(set(posteriors) == set(METHOD_ORDER), "A15 integrity methods differ")
    result = {
        method: posterior_integrity_evidence(posterior)
        for method, posterior in posteriors.items()
    }
    _require(
        all(
            evidence["negative_probability_row_count"] == 0
            and evidence["mass_violation_row_count"] == 0
            and evidence["cdf_nonmonotone_row_count"] == 0
            and evidence["cdf_terminal_violation_row_count"] == 0
            for evidence in result.values()
        ),
        "A15 inner-eval posterior integrity differs",
    )
    return result


def evaluate_a15_inner_scene_once(
    *,
    full_model: A15FTEBCorrection,
    endpoint_null_model: A15FTEBCorrection,
    cache: A15EndpointCache,
    device: torch.device,
) -> dict[str, Any]:
    """Read the unseen-scene cache once and report all five methods."""

    full = _evaluate_a15_arm(
        full_model, cache, endpoint_null=False, device=device
    )
    endpoint_null = _evaluate_a15_arm(
        endpoint_null_model, cache, endpoint_null=True, device=device
    )
    _require(
        torch.equal(full["raw_anchor_posterior"], endpoint_null["raw_anchor_posterior"])
        and torch.equal(full["raw_anchor_mean"], endpoint_null["raw_anchor_mean"])
        and torch.equal(full["relation_available"], endpoint_null["relation_available"]),
        "A15 matched inner-eval q0/availability differs",
    )
    q0 = cache.raw_posterior.float()
    q_sarn = torch.where(
        full["relation_available"][:, None],
        cache.sarn_posterior.float(),
        q0,
    )
    geometric_proposed = fixed_geometric_natural_parameter_base(
        q0, cache.sarn_posterior.float()
    )["geometric_base"]
    fixed_geometric = torch.where(
        full["relation_available"][:, None], geometric_proposed, q0
    )
    q0_mean, _ = _posterior_moments(q0)
    q_sarn_mean, _ = _posterior_moments(q_sarn)
    fixed_geometric_mean, _ = _posterior_moments(fixed_geometric)
    posteriors = {
        METHOD_Q0: q0,
        METHOD_DIRECT_QS: q_sarn,
        METHOD_FIXED_GEOMETRIC: fixed_geometric,
        METHOD_A15_LEARNED: full["progress_posterior"].float(),
        METHOD_ENDPOINT_NULL: endpoint_null["progress_posterior"].float(),
    }
    means = {
        METHOD_Q0: q0_mean,
        METHOD_DIRECT_QS: q_sarn_mean,
        METHOD_FIXED_GEOMETRIC: fixed_geometric_mean,
        METHOD_A15_LEARNED: full["mean"].float(),
        METHOD_ENDPOINT_NULL: endpoint_null["mean"].float(),
    }
    metrics = inner_eval_metric_report(
        predictions={name: value.tolist() for name, value in means.items()},
        target=cache.target.tolist(),
        condition_names=cache.condition_names,
    )
    integrity = _method_integrity(posteriors)
    per_row = []
    for index in range(cache.rows):
        target = float(cache.target[index])
        row_means = {name: float(value[index]) for name, value in means.items()}
        per_row.append(
            {
                "row_index": index,
                "physical_group_id": int(cache.physical_group_ids[index]),
                "source_index": int(cache.source_indices[index]),
                "sample_id": cache.sample_ids[index],
                "scene_stem": cache.scene_stems[index],
                "condition_name": cache.condition_names[index],
                "target": target,
                "mean": row_means,
                "absolute_error": {
                    name: abs(value - target) for name, value in row_means.items()
                },
                "relation_available": bool(full["relation_available"][index]),
            }
        )
    return {
        "evaluation_scope": "inner_eval_once_after_both_arms",
        "physical_samples": cache.physical_samples,
        "condition_rows": cache.rows,
        "metrics": metrics,
        "posterior_integrity": integrity,
        "predictions": {
            "method_order": list(METHOD_ORDER),
            "sample_ids": list(cache.sample_ids),
            "scene_stems": list(cache.scene_stems),
            "condition_names": list(cache.condition_names),
            "source_indices": list(cache.source_indices),
            "physical_group_ids": cache.physical_group_ids.tolist(),
            "target": cache.target.tolist(),
            "mean": {name: value.tolist() for name, value in means.items()},
            "relation_available": full["relation_available"].tolist(),
        },
        "per_row_evidence": per_row,
        "fixed_geometric_recomputed_from_canonical_inner_eval_endpoints": True,
        "prior_endpoint_analysis_reused": False,
        "automatic_gate_used": False,
    }


def run_a15_fteb_inner_scene_probe(
    *,
    correction_train_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    output_path: Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"A15 output already exists: {output}")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "A15 device must be CPU or CUDA")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    _configure_reproducibility(BATCH_SCHEDULE_SEED, device)
    torch.use_deterministic_algorithms(True, warn_only=False)

    terminal_model, terminal_evidence, construction = load_frozen_terminal_a11(
        terminal_a11_checkpoint_path, device=device
    )
    anchor_before = _state_cpu(terminal_model)
    caches, cache_evidence = materialize_a15_endpoint_caches(
        correction_train_manifest_path=correction_train_manifest_path,
        frozen_terminal_model=terminal_model,
        construction=construction,
        device=device,
    )
    anchor_after = _state_cpu(terminal_model)
    anchor_unchanged = module_states_bit_exact(anchor_before, anchor_after)
    _require(anchor_unchanged, "terminal A11 state changed during A15 cache creation")
    del terminal_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    full_model, endpoint_null_model, initialization = build_matched_a15_models(
        construction
    )
    schedule = fixed_physical_batch_schedule()
    schedule_counts = schedule_presentation_counts(schedule)
    factory = _model_factory(construction)
    full_result = train_a15_probe_arm(
        arm=METHOD_A15_LEARNED,
        model=full_model,
        model_factory=factory,
        cache=caches["inner_train"],
        schedule=schedule,
        device=device,
    )
    endpoint_null_result = train_a15_probe_arm(
        arm=METHOD_ENDPOINT_NULL,
        model=endpoint_null_model,
        model_factory=factory,
        cache=caches["inner_train"],
        schedule=schedule,
        device=device,
    )
    expected_trace = [list(batch) for batch in schedule]
    _require(
        full_result["schedule_trace"]
        == endpoint_null_result["schedule_trace"]
        == expected_trace,
        "A15 matched arms did not consume the same physical order",
    )
    inner_eval = evaluate_a15_inner_scene_once(
        full_model=full_model,
        endpoint_null_model=endpoint_null_model,
        cache=caches["inner_eval"],
        device=device,
    )
    access_flags = dict(ACCESS_EXPECTATION)
    artifact = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "scientific_protocol": SCIENTIFIC_PROTOCOL,
        "development_scope": DEVELOPMENT_SCOPE,
        "automatic_execution_selection_or_advancement_control": False,
        "access_flags": access_flags,
        "terminal_a11": terminal_evidence,
        "terminal_a11_state_unchanged_during_cache": anchor_unchanged,
        "endpoint_cache_semantics": dict(ENDPOINT_CACHE_SEMANTICS),
        "fixed_geometric_semantics": dict(FIXED_GEOMETRIC_SEMANTICS),
        "canonical_cache": cache_evidence,
        "matched_arm_semantics": dict(MATCHED_ARM_SEMANTICS),
        "matched_initialization": initialization,
        "construction": dict(construction),
        "parameter_counts": fteb_parameter_counts(full_model),
        "training": {
            "steps_per_arm": TRAIN_STEPS,
            "physical_batch_size": PHYSICAL_BATCH_SIZE,
            "row_batch_size": ROW_BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": SCHEDULER,
            "objective": "a15_fteb_loss",
            "schedule": expected_trace,
            "presentation_counts": schedule_counts,
            "same_canonical_cache": True,
            "same_sample_condition_order": True,
            "clean_rows": 0,
            "sarn_off_rows": 0,
        },
        "arms": {
            METHOD_A15_LEARNED: full_result,
            METHOD_ENDPOINT_NULL: endpoint_null_result,
        },
        "inner_eval": inner_eval,
        "descriptive_reference": dict(DESCRIPTIVE_REFERENCE),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(artifact, stream)
    pooled = inner_eval["metrics"]["pooled"]
    return {
        "status": "complete",
        "output": str(output),
        "protocol": PROTOCOL,
        "training_physical_samples": TRAIN_PHYSICAL_SAMPLES,
        "inner_eval_physical_samples": EVAL_PHYSICAL_SAMPLES,
        "steps_per_arm": TRAIN_STEPS,
        "learned_vs_q0": pooled["versus_q0"][METHOD_A15_LEARNED],
        "fixed_geometric_vs_q0": pooled["versus_q0"][METHOD_FIXED_GEOMETRIC],
        "learned_vs_fixed_geometric": pooled["learned_comparisons"][
            "learned_minus_fixed_geometric"
        ],
        "learned_vs_endpoint_null": pooled["learned_comparisons"][
            "learned_minus_endpoint_null"
        ],
        "automatic_gate_used": False,
        "access_flags": access_flags,
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_a15_fteb_inner_scene_probe(
        correction_train_manifest_path=args.correction_train_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        output_path=args.output,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A15EndpointCache",
    "A15ProbeRunError",
    "CANDIDATE_PHYSICAL_BATCH_SIZE",
    "LEARNING_RATE",
    "OPTIMIZER",
    "PROTOCOL",
    "SCALAR_LOSS_COMPONENTS",
    "SEMANTIC_GROUPS",
    "WEIGHT_DECAY",
    "build_matched_a15_models",
    "evaluate_a15_inner_scene_once",
    "forward_a15_correction",
    "materialize_a15_endpoint_caches",
    "run_a15_fteb_inner_scene_probe",
    "semantic_parameter_groups",
    "train_a15_probe_arm",
]
