"""Matched CUDA/BF16 efficiency benchmark for the A15.2 comparison arms.

The benchmark consumes only the physical 6,616-row correction-train manifest,
the terminal A11 checkpoint, and (for the learned A15.2 arm) the terminal
A15.2 checkpoint.  The first eight physical samples are materialized once
under one fixed ``perspective_severe`` condition at 256 x 256.  Image decoding
and projective/SARN construction are outside the timed region; host-to-device
transfer and every executed model/analytic stage are inside it.  The optional
terminal-A11 arm executes the complete checkpointed SCORT ``forward`` rather
than a reconstructed or correction-only approximation.

One arm is measured per process.  Each invocation reports both batch-one
latency (cycling through the same eight samples) and batch-eight throughput.
Warmup and timed iteration counts are fixed descriptive settings, not model
selection or execution-advancement criteria.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.a15_2_fteb import (  # noqa: E402
    A15_2_ARCHITECTURE,
    A15_2_LEARNED_RESIDUAL_SCALE,
    A152FTEBCorrection,
)
from experiments.a11_scort import (  # noqa: E402
    TRANSPORT_STAGE_COUNT,
    A11SCORTImageModel,
)
from experiments.a15_fteb import (  # noqa: E402
    _endpoint_from_anchor,
    _posterior_moments,
    fixed_geometric_natural_parameter_base,
)
from experiments.benchmark_cagh_v5_paper_efficiency import (  # noqa: E402
    latency_statistics,
    parameter_inventory,
)
from experiments.prepare_a13_correction_scene_split import (  # noqa: E402
    DEFAULT_CORRECTION_TRAIN_MANIFEST,
    load_a13_correction_train_manifest,
)
from experiments.run_a12_fixed_q0_causal_probe import (  # noqa: E402
    DEFAULT_TERMINAL_A11_CHECKPOINT,
    load_frozen_terminal_a11,
)
from experiments.train_a15_2_fteb_correction_only import (  # noqa: E402
    PIXEL_AUGMENTATION_SEED,
    PROTOCOL as A15_2_TRAINING_PROTOCOL,
    SYSTEM as A15_2_TRAINING_SYSTEM,
    TERMINAL_EPOCHS,
)
from experiments.train_support_geometry_multiview_efficientnet_pilot import (  # noqa: E402
    SyncGSupportGeometryMultiViewDataset,
)


PROTOCOL: Final[str] = "syncg_a15_2_fteb_matched_cuda_efficiency_v1"
A11_FULL_SCORT_ARM: Final[str] = "a11_full_scort"
ARMS: Final[tuple[str, ...]] = (
    "q0_raw_only",
    "fixed_geometric_twin_endpoint",
    "a15_2_full",
    A11_FULL_SCORT_ARM,
)
DISPLAY_NAMES: Final[dict[str, str]] = {
    "q0_raw_only": "Frozen A11 q0 (Raw only)",
    "fixed_geometric_twin_endpoint": "Frozen twin endpoint + fixed geometric",
    "a15_2_full": "A15.2 FTEB (fixed-quarter residual)",
    A11_FULL_SCORT_ARM: "Terminal A11 full SCORT",
}
FINAL_STAGE_OPERATIONS: Final[dict[str, str]] = {
    "q0_raw_only": "q0_posterior_and_moments_validation",
    "fixed_geometric_twin_endpoint": (
        "fixed_geometric_fusion_moments_and_validation"
    ),
    "a15_2_full": "a15_2_correction_moments_and_validation",
    A11_FULL_SCORT_ARM: "a11_full_scort_output_and_fallback_validation",
}
FIXED_CONDITION: Final[str] = "perspective_severe"
IMAGE_SIZE: Final[int] = 256
PHYSICAL_BATCH_SIZE: Final[int] = 8
FIXED_SOURCE_INDICES: Final[tuple[int, ...]] = tuple(range(PHYSICAL_BATCH_SIZE))
PROFILE_BATCH_SIZES: Final[tuple[int, ...]] = (1, PHYSICAL_BATCH_SIZE)
WARMUP_ITERATIONS: Final[int] = 20
TIMED_ITERATIONS: Final[int] = 100
DEVICE_NAME: Final[str] = "cuda:0"
AUTOCAST_DTYPE_NAME: Final[str] = "bfloat16"


class A152EfficiencyError(ValueError):
    """An A15.2 efficiency input or runtime measurement is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A152EfficiencyError(message)


@dataclass(slots=True)
class A152EfficiencyRuntime:
    """Loaded executable modules for one isolated benchmark arm."""

    arm: str
    anchor: nn.Module
    correction: A152FTEBCorrection | None
    device: torch.device
    loading_evidence: dict[str, Any]
    last_output_validation: dict[str, Any] | None = None

    @property
    def parameter_roots(self) -> tuple[nn.Module, ...]:
        roots: tuple[nn.Module, ...] = (self.anchor,)
        if self.correction is not None:
            roots += (self.correction,)
        return roots


class _FrozenA11RawEndpoint(nn.Module):
    """Only the shared A11 modules executed by q0 and the twin endpoints."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        encoder = getattr(source, "raw_encoder", None)
        head = getattr(source, "raw_posterior_head", None)
        _require(
            isinstance(encoder, nn.Module) and isinstance(head, nn.Module),
            "terminal A11 does not expose the frozen Raw endpoint",
        )
        self.raw_encoder = encoder
        self.raw_posterior_head = head


class _FrozenA11FullSCORT(nn.Module):
    """Execute the real terminal A11 forward and expose one stage boundary.

    The permanent pre-hook records a caller-supplied CUDA event immediately
    before the SARN encoder, after the Raw posterior and moments are complete.
    It changes neither tensors nor operator order and lets the benchmark split
    the one real model forward into Raw-anchor and SARN/SCORT intervals without
    executing any module twice.
    """

    def __init__(self, source: A11SCORTImageModel) -> None:
        super().__init__()
        _require(
            isinstance(source, A11SCORTImageModel),
            "terminal A11 full-SCORT source has the wrong type",
        )
        self.model = source
        self._raw_stage_event: torch.cuda.Event | None = None
        self._raw_stage_event_recorded = False
        self._raw_stage_hook = self.model.sarn_encoder.register_forward_pre_hook(
            self._record_raw_stage_event
        )

    def _record_raw_stage_event(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
    ) -> None:
        if self._raw_stage_event is not None:
            self._raw_stage_event.record()
            self._raw_stage_event_recorded = True

    def set_raw_stage_event(self, event: torch.cuda.Event | None) -> None:
        self._raw_stage_event = event
        self._raw_stage_event_recorded = False

    def consume_raw_stage_event_evidence(self) -> bool:
        recorded = self._raw_stage_event_recorded
        self._raw_stage_event = None
        self._raw_stage_event_recorded = False
        return recorded

    def forward(
        self,
        original_view: torch.Tensor,
        sarn_view: torch.Tensor,
        sarn_support_mask: torch.Tensor,
        *,
        sarn_active: torch.Tensor,
        raw_to_sarn_homography: torch.Tensor,
    ) -> dict[str, Any]:
        return self.model(
            original_view,
            sarn_view,
            sarn_support_mask,
            sarn_active=sarn_active,
            raw_to_sarn_homography=raw_to_sarn_homography,
        )


def _create_report_outputs(
    json_path: Path,
    markdown_path: Path,
    report: Mapping[str, Any],
) -> None:
    """Create both report files once, cleaning only files made by this call."""

    output = Path(json_path).resolve()
    markdown = Path(markdown_path).resolve()
    json_text = (
        json.dumps(
            dict(report),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_text = render_markdown(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    json_created = False
    markdown_created = False
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json_created = True
            stream.write(json_text)
        with markdown.open("x", encoding="utf-8", newline="\n") as stream:
            markdown_created = True
            stream.write(markdown_text)
    except BaseException as exc:
        if markdown_created:
            try:
                markdown.unlink(missing_ok=True)
            except OSError:
                pass
        if json_created:
            try:
                output.unlink(missing_ok=True)
            except OSError:
                pass
        if isinstance(exc, FileExistsError):
            raise A152EfficiencyError(
                f"efficiency output already exists: {exc.filename}"
            ) from exc
        raise


def load_fixed_physical_batch(
    manifest_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Materialize the fixed first-eight severe-projective physical batch."""

    source = Path(manifest_path).resolve()
    samples = tuple(load_a13_correction_train_manifest(source))
    _require(
        len(samples) >= PHYSICAL_BATCH_SIZE,
        "correction-train manifest has fewer than eight physical samples",
    )
    selected = tuple(samples[index] for index in FIXED_SOURCE_INDICES)
    dataset = SyncGSupportGeometryMultiViewDataset(
        selected,
        training=False,
        seed=PIXEL_AUGMENTATION_SEED,
        total_epochs=1,
        image_size=IMAGE_SIZE,
        condition=FIXED_CONDITION,
    )
    rows = tuple(dataset[index] for index in range(PHYSICAL_BATCH_SIZE))
    batch = default_collate(rows)
    tensor_names = (
        "original_view",
        "sarn_view",
        "sarn_support_mask",
        "sarn_active",
        "raw_to_sarn_homography",
    )
    tensors = {name: batch[name].contiguous() for name in tensor_names}
    _require(
        tensors["original_view"].shape
        == tensors["sarn_view"].shape
        == (PHYSICAL_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE),
        "fixed efficiency image batch shape differs",
    )
    _require(
        tensors["sarn_support_mask"].shape
        == (PHYSICAL_BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE)
        and tensors["sarn_active"].shape == (PHYSICAL_BATCH_SIZE,)
        and tensors["raw_to_sarn_homography"].shape
        == (PHYSICAL_BATCH_SIZE, 3, 3),
        "fixed efficiency geometry batch shape differs",
    )
    condition_names = tuple(str(value) for value in batch["condition_name"])
    _require(
        condition_names == (FIXED_CONDITION,) * PHYSICAL_BATCH_SIZE,
        "fixed efficiency projective condition differs",
    )
    sample_ids = tuple(str(value) for value in batch["sample_id"])
    scene_stems = tuple(str(value) for value in batch["scene_stem"])
    return tensors, {
        "manifest_path": str(source),
        "manifest_rows": len(samples),
        "selection": "zero_based_source_indices_0_through_7_in_manifest_order",
        "source_indices": list(FIXED_SOURCE_INDICES),
        "physical_samples": PHYSICAL_BATCH_SIZE,
        "sample_ids": list(sample_ids),
        "scene_stems": list(scene_stems),
        "condition": FIXED_CONDITION,
        "image_size": IMAGE_SIZE,
        "pixel_condition_seed": PIXEL_AUGMENTATION_SEED,
        "dataset_training_mode": False,
        "ground_truth_consumed_by_benchmark": False,
        "sarn_active_rows_before_alignment": int(
            tensors["sarn_active"].sum().item()
        ),
        "materialized_once_before_timing": True,
    }


def _a15_2_construction(checkpoint: Mapping[str, Any]) -> dict[str, int]:
    construction = checkpoint.get("construction")
    _require(
        isinstance(construction, Mapping),
        "terminal A15.2 construction metadata is missing",
    )
    names = (
        "relation_channels",
        "token_dim",
        "attention_heads",
        "decoder_layers",
        "memory_grid_size",
        "progress_bins",
    )
    return {name: int(construction[name]) for name in names}


def load_terminal_a15_2_correction(
    checkpoint_path: Path,
    *,
    expected_construction: Mapping[str, int],
    device: torch.device,
) -> tuple[A152FTEBCorrection, dict[str, Any]]:
    """Strict-load the final correction without loading any evaluation data."""

    source = Path(checkpoint_path).resolve()
    _require(source.is_file(), f"terminal A15.2 checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, Mapping), "terminal A15.2 is not a mapping")
    _require(
        checkpoint.get("protocol") == A15_2_TRAINING_PROTOCOL
        and checkpoint.get("system") == A15_2_TRAINING_SYSTEM
        and checkpoint.get("architecture") == A15_2_ARCHITECTURE
        and float(checkpoint.get("learned_residual_scale", float("nan")))
        == A15_2_LEARNED_RESIDUAL_SCALE,
        "checkpoint is not the final fixed-quarter A15.2 correction",
    )
    training = checkpoint.get("training")
    state = checkpoint.get("model_state")
    _require(
        isinstance(training, Mapping)
        and int(training.get("epochs", -1)) == TERMINAL_EPOCHS
        and training.get("validation_manifest") is None,
        "terminal A15.2 is not the five-epoch no-validation state",
    )
    _require(isinstance(state, Mapping), "terminal A15.2 model state is missing")
    construction = _a15_2_construction(checkpoint)
    _require(
        construction
        == {name: int(value) for name, value in expected_construction.items()},
        "terminal A15.2 construction differs from terminal A11",
    )
    correction = A152FTEBCorrection(**construction)
    incompatibility = correction.load_state_dict(state, strict=True)
    _require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        "terminal A15.2 strict load differs",
    )
    correction.eval().to(device)
    return correction, {
        "path": str(source),
        "protocol": str(checkpoint["protocol"]),
        "system": str(checkpoint["system"]),
        "training_epochs": int(training["epochs"]),
        "strict_load": True,
        "learned_residual_scale": A15_2_LEARNED_RESIDUAL_SCALE,
    }


def build_runtime(
    arm: str,
    *,
    terminal_a11_checkpoint_path: Path,
    terminal_a15_2_checkpoint_path: Path | None,
    device_name: str = DEVICE_NAME,
) -> A152EfficiencyRuntime:
    """Load exactly the modules executed by one isolated benchmark arm."""

    _require(arm in ARMS, f"unknown A15.2 efficiency arm: {arm}")
    _require(
        device_name == DEVICE_NAME,
        f"paper efficiency device is fixed to {DEVICE_NAME}",
    )
    device = torch.device(device_name)
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(device)
    _require(torch.cuda.is_bf16_supported(), "CUDA BF16 is unavailable")
    loaded_anchor, a11_evidence, construction = load_frozen_terminal_a11(
        Path(terminal_a11_checkpoint_path), device=device
    )
    full_parameter_count: int | None = None
    if arm == A11_FULL_SCORT_ARM:
        anchor: nn.Module = _FrozenA11FullSCORT(loaded_anchor).eval()
        full_parameter_count = sum(
            int(parameter.numel()) for parameter in anchor.parameters()
        )
    else:
        anchor = _FrozenA11RawEndpoint(loaded_anchor).eval()
        del loaded_anchor
    _require(
        all(not parameter.requires_grad for parameter in anchor.parameters()),
        "terminal A11 executable modules are not frozen",
    )
    correction: A152FTEBCorrection | None = None
    a15_evidence: dict[str, Any] | None = None
    if arm == "a15_2_full":
        _require(
            terminal_a15_2_checkpoint_path is not None,
            "--a15-2-checkpoint is required for the full A15.2 arm",
        )
        correction, a15_evidence = load_terminal_a15_2_correction(
            terminal_a15_2_checkpoint_path,
            expected_construction=construction,
            device=device,
        )
    return A152EfficiencyRuntime(
        arm=arm,
        anchor=anchor,
        correction=correction,
        device=device,
        loading_evidence={
            "terminal_a11": a11_evidence,
            "terminal_a15_2": a15_evidence,
            "unused_A11_correction_modules_retained": False,
            "terminal_A11_full_forward_executed": arm == A11_FULL_SCORT_ARM,
            "terminal_A11_full_parameter_count_measured": full_parameter_count,
            "shared_raw_endpoint_module_loaded_once": True,
        },
    )


def arm_parameter_inventory(runtime: A152EfficiencyRuntime) -> dict[str, Any]:
    """Count unique executable parameters and expose the arm decomposition."""

    inventory = parameter_inventory(runtime.parameter_roots)
    anchor_ids = {id(parameter) for parameter in runtime.anchor.parameters()}
    correction_parameters = (
        tuple(runtime.correction.parameters())
        if runtime.correction is not None
        else ()
    )
    correction_ids = {id(parameter) for parameter in correction_parameters}
    _require(anchor_ids.isdisjoint(correction_ids), "anchor/correction parameters overlap")
    anchor_parameters = sum(
        int(parameter.numel()) for parameter in runtime.anchor.parameters()
    )
    correction_parameter_count = sum(
        int(parameter.numel()) for parameter in correction_parameters
    )
    runtime_requires_grad_parameters = int(inventory["trainable_parameters"])
    if runtime.arm == A11_FULL_SCORT_ARM:
        _require(
            runtime.correction is None,
            "terminal A11 full-SCORT arm has an external correction",
        )
        inventory["components"] = {
            "terminal_a11_full_scort": anchor_parameters,
        }
        # The paper column describes parameters optimized by the source-model
        # training protocol.  They are all frozen only after strict-loading the
        # terminal checkpoint for this inference benchmark.
        inventory["trainable_parameters"] = anchor_parameters
        inventory["runtime_requires_grad_parameters"] = (
            runtime_requires_grad_parameters
        )
        inventory["trainable_parameter_semantics"] = (
            "source-training trainable parameters; runtime checkpoint is frozen"
        )
        inventory["shared_anchor_counting"] = (
            "one complete terminal A11 model; no duplicated Raw anchor"
        )
    else:
        inventory["components"] = {
            "shared_frozen_raw_endpoint": anchor_parameters,
            "a15_2_correction": correction_parameter_count,
        }
        inventory["runtime_requires_grad_parameters"] = (
            runtime_requires_grad_parameters
        )
        inventory["trainable_parameter_semantics"] = (
            "parameters marked requires_grad in the loaded inference modules"
        )
        inventory["shared_anchor_counting"] = (
            "one unique frozen Raw endpoint reused sequentially for Raw and SARN"
        )
    return inventory


def _slice_cpu_batch(
    batch: Mapping[str, torch.Tensor], *, batch_size: int, iteration: int
) -> dict[str, torch.Tensor]:
    _require(
        batch_size in PROFILE_BATCH_SIZES,
        "A15.2 efficiency profile batch size differs",
    )
    if batch_size == PHYSICAL_BATCH_SIZE:
        indices = slice(None)
    else:
        index = int(iteration) % PHYSICAL_BATCH_SIZE
        indices = slice(index, index + 1)
    return {name: value[indices] for name, value in batch.items()}


def _pin_cpu_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.pin_memory() for name, value in batch.items()}


def _to_device(
    batch: Mapping[str, torch.Tensor],
    *,
    arm: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    names = ["original_view"]
    if arm != "q0_raw_only":
        names.extend(("sarn_view", "sarn_active"))
    if arm in ("a15_2_full", A11_FULL_SCORT_ARM):
        names.extend(
            ("sarn_support_mask", "raw_to_sarn_homography")
        )
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in names
    }


def _availability_safe_fixed_geometric(
    raw_posterior: torch.Tensor,
    sarn_posterior: torch.Tensor,
    sarn_active: torch.Tensor,
) -> torch.Tensor:
    """Use the fixed twin endpoint only where SARN is available."""

    _require(
        raw_posterior.shape == sarn_posterior.shape
        and raw_posterior.ndim == 2
        and sarn_active.shape == (raw_posterior.shape[0],)
        and sarn_active.dtype == torch.bool,
        "fixed geometric endpoint availability shapes differ",
    )
    proposed = fixed_geometric_natural_parameter_base(
        raw_posterior, sarn_posterior
    )["geometric_base"]
    return torch.where(sarn_active[:, None], proposed, raw_posterior.detach().float())


def _validate_final_posterior_moments(
    posterior: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    expected_rows: int,
) -> None:
    probability = posterior.float()
    mass = probability.sum(dim=1)
    _require(
        posterior.ndim == 2
        and posterior.shape[0] == expected_rows
        and mean.shape == variance.shape == (expected_rows,)
        and bool(torch.isfinite(probability).all())
        and bool((probability >= 0.0).all())
        and bool(
            torch.allclose(
                mass,
                torch.ones_like(mass),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )
        and bool(torch.isfinite(mean).all())
        and bool(torch.isfinite(variance).all())
        and bool((mean >= 0.0).all())
        and bool((mean <= 1.0).all())
        and bool((variance >= 0.0).all()),
        "A15.2 efficiency final posterior/moments are malformed",
    )


def _probability_mass_cdf_evidence(
    posterior: torch.Tensor,
    *,
    expected_prefix: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    """Validate one posterior path and return measured mass/CDF evidence."""

    probability = posterior.float()
    _require(
        probability.shape[:-1] == expected_prefix
        and probability.shape[-1] == 128
        and bool(torch.isfinite(probability).all())
        and bool((probability >= 0.0).all()),
        f"terminal A11 {label} posterior shape/value differs",
    )
    mass = probability.sum(dim=-1)
    cdf = probability.cumsum(dim=-1)
    monotonicity = _cdf_monotonicity_evidence(cdf)
    ones = torch.ones_like(mass)
    _require(
        bool(monotonicity["within_float32_tolerance"])
        and bool(torch.isfinite(cdf).all())
        and bool(torch.allclose(mass, ones, rtol=1.0e-5, atol=1.0e-6))
        and bool(torch.allclose(cdf[..., -1], ones, rtol=1.0e-5, atol=1.0e-6)),
        f"terminal A11 {label} mass/CDF validation failed",
    )
    return {
        "shape": list(probability.shape),
        "mass_max_abs_error": float((mass - 1.0).abs().max().item()),
        "cdf_terminal_max_abs_error": float(
            (cdf[..., -1] - 1.0).abs().max().item()
        ),
        "cdf_monotone": bool(monotonicity["within_float32_tolerance"]),
        "cdf_strictly_nondecreasing": bool(
            monotonicity["strictly_nondecreasing"]
        ),
        "cdf_minimum_step": float(monotonicity["minimum_step"]),
        "cdf_monotonicity_tolerance": float(monotonicity["tolerance"]),
    }


def _cdf_monotonicity_evidence(cdf: torch.Tensor) -> dict[str, Any]:
    """Measure CUDA-prefix monotonicity at the native float32 precision.

    CUDA's parallel prefix implementation can make adjacent prefixes differ
    by one rounding unit even when every posterior entry is non-negative.
    Keep the exact result as evidence while accepting only a bounded float32
    rounding residual; posterior non-negativity, unit mass, and terminal CDF
    checks remain independent requirements.
    """

    _require(
        cdf.dtype == torch.float32 and cdf.shape[-1] == 128,
        "terminal A11 CDF evidence has the wrong dtype/shape",
    )
    steps = cdf[..., 1:] - cdf[..., :-1]
    tolerance = 8.0 * torch.finfo(torch.float32).eps
    minimum_step = float(steps.min().item())
    return {
        "strictly_nondecreasing": bool((steps >= 0.0).all()),
        "within_float32_tolerance": bool((steps >= -tolerance).all()),
        "minimum_step": minimum_step,
        "tolerance": tolerance,
    }


def _validate_a11_full_scort_output(
    output: Mapping[str, Any],
    *,
    expected_rows: int,
) -> dict[str, Any]:
    """Validate real A11 outputs, including path CDFs and exact fallback."""

    names = (
        "progress_posterior",
        "mean",
        "variance",
        "raw_anchor_posterior",
        "raw_anchor_mean",
        "raw_anchor_variance",
        "layer_posteriors",
        "layer_means",
        "layer_variances",
        "correction_active",
        "relation_available",
        "sarn_active",
    )
    _require(
        all(isinstance(output.get(name), torch.Tensor) for name in names),
        "terminal A11 full-SCORT output roster differs",
    )
    posterior = output["progress_posterior"]
    mean = output["mean"]
    variance = output["variance"]
    raw = output["raw_anchor_posterior"]
    raw_mean = output["raw_anchor_mean"]
    raw_variance = output["raw_anchor_variance"]
    layers = output["layer_posteriors"]
    layer_means = output["layer_means"]
    layer_variances = output["layer_variances"]
    correction_active = output["correction_active"]
    relation_available = output["relation_available"]
    sarn_active = output["sarn_active"]
    _validate_final_posterior_moments(
        posterior,
        mean,
        variance,
        expected_rows=expected_rows,
    )
    _validate_final_posterior_moments(
        raw,
        raw_mean,
        raw_variance,
        expected_rows=expected_rows,
    )
    _require(
        layers.shape == (expected_rows, TRANSPORT_STAGE_COUNT, 128)
        and layer_means.shape
        == layer_variances.shape
        == (expected_rows, TRANSPORT_STAGE_COUNT)
        and correction_active.shape
        == (expected_rows, TRANSPORT_STAGE_COUNT)
        and correction_active.dtype == torch.bool
        and relation_available.shape == sarn_active.shape == (expected_rows,)
        and relation_available.dtype == sarn_active.dtype == torch.bool
        and bool((~relation_available | sarn_active).all())
        and torch.equal(
            correction_active,
            relation_available[:, None].expand_as(correction_active),
        )
        and bool(torch.isfinite(layer_means).all())
        and bool(torch.isfinite(layer_variances).all())
        and bool((layer_means >= 0.0).all())
        and bool((layer_means <= 1.0).all())
        and bool((layer_variances >= 0.0).all()),
        "terminal A11 layer/availability outputs differ",
    )
    raw_evidence = _probability_mass_cdf_evidence(
        raw,
        expected_prefix=(expected_rows,),
        label="raw",
    )
    final_evidence = _probability_mass_cdf_evidence(
        posterior,
        expected_prefix=(expected_rows,),
        label="final",
    )
    layer_evidence = _probability_mass_cdf_evidence(
        layers,
        expected_prefix=(expected_rows, TRANSPORT_STAGE_COUNT),
        label="eight-layer path",
    )
    unavailable = ~relation_available
    fallback_rows = int(unavailable.sum().item())
    expected_layers = raw[:, None, :].expand_as(layers)
    expected_layer_means = raw_mean[:, None].expand_as(layer_means)
    expected_layer_variances = raw_variance[:, None].expand_as(layer_variances)
    fallback_exact = (
        torch.equal(posterior[unavailable], raw[unavailable])
        and torch.equal(mean[unavailable], raw_mean[unavailable])
        and torch.equal(variance[unavailable], raw_variance[unavailable])
        and torch.equal(layers[unavailable], expected_layers[unavailable])
        and torch.equal(
            layer_means[unavailable], expected_layer_means[unavailable]
        )
        and torch.equal(
            layer_variances[unavailable], expected_layer_variances[unavailable]
        )
        and not bool(correction_active[unavailable].any())
    )
    _require(fallback_exact, "terminal A11 unavailable-row fallback differs from q0")
    return {
        "real_checkpoint_forward": True,
        "rows": expected_rows,
        "sarn_active_rows": int(sarn_active.sum().item()),
        "relation_available_rows": int(relation_available.sum().item()),
        "fallback_rows": fallback_rows,
        "fallback_exact_q0_posterior_moments_and_all_layers": fallback_exact,
        "raw": raw_evidence,
        "final": final_evidence,
        "eight_layer_path": layer_evidence,
    }


def _cuda_stage_durations(
    arm: str,
    events: Sequence[Any],
) -> dict[str, float]:
    """Return four consecutive event intervals shared by every arm."""

    _require(arm in ARMS, f"unknown A15.2 efficiency arm: {arm}")
    _require(len(events) == 5, "A15.2 efficiency event roster differs")
    stages = {
        "host_to_device": float(events[0].elapsed_time(events[1])),
        "raw_anchor": float(events[1].elapsed_time(events[2])),
        # q0 deliberately executes no SARN endpoint between these events; the
        # measured near-zero interval remains present so every arm partitions
        # the same event-0 through event-4 span.
        "sarn_endpoint": float(events[2].elapsed_time(events[3])),
        "final_posterior_moment_validation": float(
            events[3].elapsed_time(events[4])
        ),
    }
    stage_sum = sum(stages.values())
    event_total = float(events[0].elapsed_time(events[4]))
    _require(
        math.isclose(stage_sum, event_total, rel_tol=1.0e-5, abs_tol=1.0e-3),
        "A15.2 consecutive CUDA stage sum differs from its event total",
    )
    return stages


def _run_iteration(
    runtime: A152EfficiencyRuntime,
    cpu_batch: Mapping[str, torch.Tensor],
    *,
    events: Sequence[torch.cuda.Event] | None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[float | None, dict[str, float]]:
    """Execute one arm; optional events measure one synchronized timed call."""

    _require(len(cpu_batch) >= 1, "A15.2 efficiency CPU batch is empty")
    rows = int(cpu_batch["original_view"].shape[0])
    if events is not None:
        _require(len(events) == 5, "A15.2 efficiency event roster differs")
        torch.cuda.synchronize(runtime.device)
        started = clock_ns()
        events[0].record()
    else:
        started = 0

    device_batch = _to_device(cpu_batch, arm=runtime.arm, device=runtime.device)
    if events is not None:
        events[1].record()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=True
    ):
        if runtime.arm == A11_FULL_SCORT_ARM:
            _require(
                isinstance(runtime.anchor, _FrozenA11FullSCORT)
                and runtime.correction is None,
                "terminal A11 full-SCORT runtime is incomplete",
            )
            runtime.anchor.set_raw_stage_event(
                None if events is None else events[2]
            )
            output = runtime.anchor(
                device_batch["original_view"],
                device_batch["sarn_view"],
                device_batch["sarn_support_mask"],
                sarn_active=device_batch["sarn_active"],
                raw_to_sarn_homography=device_batch[
                    "raw_to_sarn_homography"
                ],
            )
            raw_stage_recorded = (
                runtime.anchor.consume_raw_stage_event_evidence()
            )
            _require(
                events is None or raw_stage_recorded,
                "terminal A11 Raw stage event was not recorded",
            )
            if events is not None:
                events[3].record()
            runtime.last_output_validation = _validate_a11_full_scort_output(
                output,
                expected_rows=rows,
            )
        else:
            raw = _endpoint_from_anchor(
                runtime.anchor, device_batch["original_view"]
            )
            if events is not None:
                events[2].record()

            sarn: Mapping[str, Any] | None = None
            if runtime.arm != "q0_raw_only":
                sarn = _endpoint_from_anchor(
                    runtime.anchor, device_batch["sarn_view"]
                )
            if events is not None:
                events[3].record()

            if runtime.arm == "q0_raw_only":
                posterior = raw["posterior"]
                mean = raw["mean"]
                variance = raw["variance"]
            elif runtime.arm == "fixed_geometric_twin_endpoint":
                _require(sarn is not None, "fixed geometric SARN endpoint is missing")
                posterior = _availability_safe_fixed_geometric(
                    raw["posterior"],
                    sarn["posterior"],
                    device_batch["sarn_active"],
                )
                mean, variance = _posterior_moments(posterior)
            else:
                _require(
                    sarn is not None and runtime.correction is not None,
                    "full A15.2 runtime is incomplete",
                )
                output = runtime.correction(
                    raw["posterior"],
                    raw["features"],
                    sarn["posterior"],
                    sarn["features"],
                    device_batch["sarn_support_mask"],
                    sarn_active=device_batch["sarn_active"],
                    raw_to_sarn_homography=device_batch[
                        "raw_to_sarn_homography"
                    ],
                )
                posterior = output["progress_posterior"]
                mean = output["mean"]
                variance = output["variance"]
            _validate_final_posterior_moments(
                posterior,
                mean,
                variance,
                expected_rows=rows,
            )

    if events is None:
        return None, {}
    events[4].record()
    events[4].synchronize()
    total_ms = (clock_ns() - started) / 1_000_000.0
    _require(total_ms > 0.0, "A15.2 efficiency clock is non-positive")
    stages = _cuda_stage_durations(runtime.arm, events)
    return total_ms, stages


def _positive_latency_statistics(values: Sequence[float]) -> dict[str, float]:
    # CUDA events can quantize a very short analytic stage to zero.  Preserve
    # the measured value while using a representable positive number only for
    # the shared descriptive-statistics helper.
    epsilon = 1.0e-9
    measured = tuple(max(float(value), epsilon) for value in values)
    return latency_statistics(measured)


def _measure_profile(
    runtime: A152EfficiencyRuntime,
    pinned_batch: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, Any]:
    for iteration in range(WARMUP_ITERATIONS):
        selected = _slice_cpu_batch(
            pinned_batch, batch_size=batch_size, iteration=iteration
        )
        _run_iteration(runtime, selected, events=None)
    torch.cuda.synchronize(runtime.device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    baseline_allocated = int(torch.cuda.memory_allocated(runtime.device))
    baseline_reserved = int(torch.cuda.memory_reserved(runtime.device))

    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(5))
    batch_latencies: list[float] = []
    cuda_stage_totals: list[float] = []
    stage_latencies: dict[str, list[float]] = {}
    for iteration in range(TIMED_ITERATIONS):
        selected = _slice_cpu_batch(
            pinned_batch, batch_size=batch_size, iteration=iteration
        )
        elapsed, stages = _run_iteration(runtime, selected, events=events)
        _require(elapsed is not None, "A15.2 timed iteration has no latency")
        batch_latencies.append(float(elapsed))
        cuda_stage_totals.append(sum(float(value) for value in stages.values()))
        for name, value in stages.items():
            stage_latencies.setdefault(name, []).append(float(value))

    peak_allocated = int(torch.cuda.max_memory_allocated(runtime.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(runtime.device))
    per_sample = [value / float(batch_size) for value in batch_latencies]
    stage_total_per_sample = [
        value / float(batch_size) for value in cuda_stage_totals
    ]
    stage_per_sample = {
        name: [value / float(batch_size) for value in values]
        for name, values in stage_latencies.items()
    }
    return {
        "batch_size": batch_size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "timed_samples": TIMED_ITERATIONS * batch_size,
        "latency_batch_ms": latency_statistics(batch_latencies),
        "latency_per_sample_ms": latency_statistics(per_sample),
        "cuda_stage_total_per_sample_ms": latency_statistics(
            stage_total_per_sample
        ),
        "throughput_samples_per_second": (
            1000.0 * TIMED_ITERATIONS * batch_size / sum(batch_latencies)
        ),
        "stage_latency_per_sample_ms": {
            name: _positive_latency_statistics(values)
            for name, values in stage_per_sample.items()
        },
        "cuda_stage_total_is_sum_of_reported_stages": True,
        "cuda_memory": {
            "baseline_allocated_mib": baseline_allocated / (1024.0**2),
            "baseline_reserved_mib": baseline_reserved / (1024.0**2),
            "peak_allocated_mib": peak_allocated / (1024.0**2),
            "peak_reserved_mib": peak_reserved / (1024.0**2),
            "incremental_peak_allocated_mib": max(
                0, peak_allocated - baseline_allocated
            )
            / (1024.0**2),
            "incremental_peak_reserved_mib": max(
                0, peak_reserved - baseline_reserved
            )
            / (1024.0**2),
            "peak_reset_independently_for_profile": True,
        },
    }


def measure_runtime(
    runtime: A152EfficiencyRuntime,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Measure fixed B1 latency and B8 throughput on CUDA BF16."""

    _require(runtime.device.type == "cuda", "paper efficiency requires CUDA")
    pinned = _pin_cpu_batch(batch)
    profiles = {
        f"batch_{batch_size}": _measure_profile(
            runtime, pinned, batch_size=batch_size
        )
        for batch_size in PROFILE_BATCH_SIZES
    }
    result: dict[str, Any] = {
        "profiles": profiles,
        "timing_scope": {
            "included": [
                "pinned host-to-device transfer for arm-required tensors",
                "frozen Raw anchor forward",
                "frozen SARN endpoint forward when required",
                "fixed-geometric analytic fusion or A15.2 correction when required",
                "complete checkpointed A11 SCORT forward for its selected arm",
                "output validation and one final CUDA synchronization",
            ],
            "excluded": [
                "checkpoint/model construction",
                "source image decode and canonical ROI extraction",
                "fixed projective condition and SARN view construction",
                "ground-truth access and accuracy scoring",
            ],
        },
        "stage_timing_api": "torch.cuda.Event_elapsed_time_on_current_stream",
        "total_timing_api": "perf_counter_ns_with_pre_and_post_cuda_synchronize",
        "stage_semantics": {
            "uniform_stage_order": [
                "host_to_device",
                "raw_anchor",
                "sarn_endpoint",
                "final_posterior_moment_validation",
            ],
            "q0_sarn_endpoint_is_intentional_no_op": runtime.arm == "q0_raw_only",
            "final_stage_operation": FINAL_STAGE_OPERATIONS[runtime.arm],
            "fixed_geometric_inactive_fallback": (
                "exact_q0_on_sarn_active_false"
                if runtime.arm == "fixed_geometric_twin_endpoint"
                else None
            ),
            "a11_full_scort_stage_partition": (
                "pre_SARN_hook_after_complete_raw_anchor_then_SARN_alignment_"
                "relation_decoder_transport_in_the_same_real_forward"
                if runtime.arm == A11_FULL_SCORT_ARM
                else None
            ),
            "cuda_stage_total": (
                "sum_of_four_consecutive_nonoverlapping_event_intervals"
            ),
            "end_to_end_latency": (
                "wall_clock_H2D_forward_final_validation_and_final_sync"
            ),
        },
        "autocast": AUTOCAST_DTYPE_NAME,
    }
    if runtime.arm == A11_FULL_SCORT_ARM:
        _require(
            isinstance(runtime.last_output_validation, Mapping)
            and int(runtime.last_output_validation.get("rows", -1))
            == PHYSICAL_BATCH_SIZE,
            "terminal A11 B8 output-validation evidence is missing",
        )
        result["output_validation"] = dict(runtime.last_output_validation)
    return result


def _environment(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    cuda = device.type == "cuda" and torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "synthetic",
        "cuda_runtime": torch.version.cuda if cuda else None,
        "cudnn": int(torch.backends.cudnn.version() or 0) if cuda else None,
        "autocast_dtype": AUTOCAST_DTYPE_NAME,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    profiles = report["measurement"]["profiles"]
    b1 = profiles["batch_1"]
    b8 = profiles[f"batch_{PHYSICAL_BATCH_SIZE}"]
    b1_latency = b1["latency_per_sample_ms"]
    b8_latency = b8["latency_per_sample_ms"]
    b1_memory = b1["cuda_memory"]
    b8_memory = b8["cuda_memory"]
    parameters = report["parameters"]
    lines = [
        "# A15.2 matched CUDA/BF16 efficiency",
        "",
        (
            "The fixed first-eight correction-train samples use perspective_severe "
            "at 256 x 256. Decode/projective materialization is excluded; H2D and "
            "the complete selected inference path are included."
        ),
        "",
        "| Arm | Unique params (M) | Trainable params (M) | B1 mean (ms) | B1 P50 (ms) | B1 P95 (ms) | B8 per-sample mean (ms) | B8 throughput (sample/s) | B1 peak alloc/reserved (MiB) | B8 peak alloc/reserved (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {report['display_name']} | "
            f"{int(parameters['total_parameters']) / 1_000_000.0:.3f} | "
            f"{int(parameters['trainable_parameters']) / 1_000_000.0:.3f} | "
            f"{float(b1_latency['mean_ms']):.3f} | "
            f"{float(b1_latency['p50_ms']):.3f} | "
            f"{float(b1_latency['p95_ms']):.3f} | "
            f"{float(b8_latency['mean_ms']):.3f} | "
            f"{float(b8['throughput_samples_per_second']):.3f} | "
            f"{float(b1_memory['peak_allocated_mib']):.1f}/"
            f"{float(b1_memory['peak_reserved_mib']):.1f} | "
            f"{float(b8_memory['peak_allocated_mib']):.1f}/"
            f"{float(b8_memory['peak_reserved_mib']):.1f} |"
        ),
        "",
        (
            f"Each profile uses {WARMUP_ITERATIONS} warmup and "
            f"{TIMED_ITERATIONS} timed iterations in one isolated arm process."
        ),
        "",
        "## B1 stage latency (ms/sample)",
        "",
        "| Stage | Mean | P50 | P95 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in b1["stage_latency_per_sample_ms"].items():
        lines.append(
            f"| {name} | {float(values['mean_ms']):.3f} | "
            f"{float(values['p50_ms']):.3f} | "
            f"{float(values['p95_ms']):.3f} |"
        )
    output_validation = report["measurement"].get("output_validation")
    if isinstance(output_validation, Mapping):
        lines.extend(
            [
                "",
                "## Terminal A11 availability and output validation",
                "",
                (
                    f"SARN active: {int(output_validation['sarn_active_rows'])}/"
                    f"{int(output_validation['rows'])}; relation available: "
                    f"{int(output_validation['relation_available_rows'])}/"
                    f"{int(output_validation['rows'])}; exact q0 fallback: "
                    f"{int(output_validation['fallback_rows'])} rows."
                ),
                "",
                "| Posterior path | Max mass error | Max terminal-CDF error | CDF monotone |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("raw", "Raw q0"),
            ("final", "Final SCORT"),
            ("eight_layer_path", "All eight SCORT layers"),
        ):
            evidence = output_validation[key]
            lines.append(
                f"| {label} | {float(evidence['mass_max_abs_error']):.3e} | "
                f"{float(evidence['cdf_terminal_max_abs_error']):.3e} | "
                f"{bool(evidence['cdf_monotone'])} |"
            )
        lines.extend(
            [
                "",
                (
                    "Unavailable-row posterior, moments, and all eight layer "
                    "posteriors matched q0 exactly: "
                    f"{bool(output_validation['fallback_exact_q0_posterior_moments_and_all_layers'])}."
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)


BatchLoader = Callable[[Path], tuple[dict[str, torch.Tensor], dict[str, Any]]]
RuntimeFactory = Callable[..., A152EfficiencyRuntime]
MeasurementFunction = Callable[
    [A152EfficiencyRuntime, Mapping[str, torch.Tensor]], dict[str, Any]
]


def run_benchmark(
    *,
    arm: str,
    correction_train_manifest_path: Path,
    terminal_a11_checkpoint_path: Path,
    terminal_a15_2_checkpoint_path: Path | None,
    output_json: Path,
    output_markdown: Path | None,
    batch_loader: BatchLoader = load_fixed_physical_batch,
    runtime_factory: RuntimeFactory = build_runtime,
    measurement_function: MeasurementFunction = measure_runtime,
    device_name: str = DEVICE_NAME,
) -> dict[str, Any]:
    """Run one isolated arm and write its paper-ready JSON/Markdown record."""

    _require(arm in ARMS, f"unknown A15.2 efficiency arm: {arm}")
    if arm == "a15_2_full":
        _require(
            terminal_a15_2_checkpoint_path is not None,
            "--a15-2-checkpoint is required for the full A15.2 arm",
        )
    manifest = Path(correction_train_manifest_path).resolve()
    terminal_a11 = Path(terminal_a11_checkpoint_path).resolve()
    terminal_a15_2 = (
        None
        if terminal_a15_2_checkpoint_path is None
        else Path(terminal_a15_2_checkpoint_path).resolve()
    )
    output = Path(output_json).resolve()
    markdown = (
        output.with_suffix(".md")
        if output_markdown is None
        else Path(output_markdown).resolve()
    )
    protected = {manifest, terminal_a11}
    if terminal_a15_2 is not None:
        protected.add(terminal_a15_2)
    _require(output not in protected, "output cannot overwrite an input artifact")
    _require(
        markdown != output and markdown not in protected,
        "Markdown output path cannot overwrite an input or JSON artifact",
    )
    existing_outputs = tuple(
        path for path in (output, markdown) if path.exists()
    )
    _require(
        not existing_outputs,
        "efficiency output already exists: "
        + ", ".join(str(path) for path in existing_outputs),
    )

    batch, batch_identity = batch_loader(manifest)
    runtime = runtime_factory(
        arm,
        terminal_a11_checkpoint_path=terminal_a11,
        terminal_a15_2_checkpoint_path=terminal_a15_2,
        device_name=device_name,
    )
    _require(runtime.arm == arm, "runtime factory returned the wrong arm")
    parameters = arm_parameter_inventory(runtime)
    measurement = measurement_function(runtime, batch)
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "display_name": DISPLAY_NAMES[arm],
        "paper_comparison_arms": list(ARMS),
        "input": batch_identity,
        "checkpoints": {
            "terminal_a11": str(terminal_a11),
            "terminal_a15_2": (
                str(terminal_a15_2) if terminal_a15_2 is not None else None
            ),
        },
        "execution": {
            "one_arm_per_process": True,
            "image_size": IMAGE_SIZE,
            "fixed_condition": FIXED_CONDITION,
            "profile_batch_sizes": list(PROFILE_BATCH_SIZES),
            "batch_1_sampling": (
                "timed_iteration_index_modulo_8_over_the_fixed_roster"
            ),
            "batch_8_sampling": "all_eight_fixed_physical_samples_each_iteration",
            "warmup_iterations_per_profile": WARMUP_ITERATIONS,
            "timed_iterations_per_profile": TIMED_ITERATIONS,
            "warmup_or_timing_used_for_model_selection": False,
            "warmup_or_timing_used_for_execution_advancement": False,
        },
        "environment": _environment(device_name),
        "parameters": parameters,
        "loading_evidence": runtime.loading_evidence,
        "measurement": measurement,
        "data_access": {
            "correction_train_manifest": True,
            "terminal_a11_checkpoint": True,
            "terminal_a15_2_checkpoint": arm == "a15_2_full",
            "correction_dev": False,
            "core_audit": False,
            "fold_a_or_b": False,
            "formal_holdout": False,
            "field_photos": False,
        },
    }

    _create_report_outputs(output, markdown, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
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
    parser.add_argument("--a15-2-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_benchmark(
        arm=args.arm,
        correction_train_manifest_path=args.correction_train_manifest,
        terminal_a11_checkpoint_path=args.terminal_a11_checkpoint,
        terminal_a15_2_checkpoint_path=args.a15_2_checkpoint,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "output": str(Path(args.output_json).resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A11_FULL_SCORT_ARM",
    "ARMS",
    "A152EfficiencyError",
    "A152EfficiencyRuntime",
    "DEVICE_NAME",
    "FIXED_CONDITION",
    "IMAGE_SIZE",
    "PHYSICAL_BATCH_SIZE",
    "PROFILE_BATCH_SIZES",
    "PROTOCOL",
    "TIMED_ITERATIONS",
    "WARMUP_ITERATIONS",
    "arm_parameter_inventory",
    "build_argument_parser",
    "build_runtime",
    "load_fixed_physical_batch",
    "load_terminal_a15_2_correction",
    "measure_runtime",
    "render_markdown",
    "run_benchmark",
]
