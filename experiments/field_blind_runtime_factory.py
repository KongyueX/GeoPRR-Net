"""Frozen provider factory used by the full-scene multi-method blind runner.

The factory consumes only hash-bound public/model metadata.  It never accepts a
field path, label, crop, ScaleMark, or physical range.  For VDN/Transformer
controls the exact GARC numeric-range pipeline is reused.  If that range uses
PEPD structural geometry, a private copy of GARC's frozen PEPD provider is
invoked on the same ROI immediately before range inference; the externally
reported progress still comes exclusively from the named control backbone.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range import AutomaticNumericRangePipeline
from experiments.garc_full_auto_public import build_runtime, load_plan
from experiments.v5_unified_full_auto_adapter import (
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    sha256_file,
)
from experiments.v5_unified_full_auto_progress_providers import (
    TransformerFullAutoProgressProvider,
)
from experiments.v5_unified_two_stage_retest import assert_label_free, strict_json_load


PROTOCOL: Final[str] = "field_blind_runtime_factory_v1"
MODES: Final[frozenset[str]] = frozenset(
    {"garc_plan", "v5_plan", "shared_garc_range"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _binding(value: Any, *, label: str) -> Path:
    _require(isinstance(value, Mapping), f"{label} binding absent")
    path = Path(str(value.get("path") or "")).resolve(strict=True)
    digest = str(value.get("sha256") or "").casefold()
    _require(len(digest) == 64 and sha256_file(path) == digest, f"{label} hash drift")
    return path


def _plan(value: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _binding(value, label=label)
    checked, plan = load_plan(path)
    _require(checked == path, f"{label} path drift")
    return checked, plan


def _load_component(path_value: Any, *, label: str) -> FrozenComponentBinding:
    path = _binding(path_value, label=label)
    value = strict_json_load(path)
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return FrozenComponentBinding.from_record(value)


def _load_callable(value: Any, *, label: str):
    _require(isinstance(value, Mapping), f"{label} factory binding absent")
    path = _binding(value, label=label)
    function_name = str(value.get("function") or "")
    _require(bool(function_name), f"{label} function absent")
    module_name = f"field_blind_runtime_{label.replace(' ', '_')}_{sha256_file(path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {label}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    function = getattr(module, function_name, None)
    _require(callable(function), f"{label} lacks callable {function_name}")
    return function


class _PreparedSharedGARCRange(AutomaticNumericRangePipeline):
    """Prime GARC's private structural PEPD state, then run its range head."""

    def __init__(self, source_progress: Any, range_pipeline: AutomaticNumericRangePipeline):
        self.source_progress = source_progress
        self.range_pipeline = range_pipeline

    @property
    def identity(self) -> Mapping[str, Any]:
        # This is the exact GARC range provider; priming recreates the call that
        # GARC's own UnifiedFullAutoAdapter performs before range inference.
        return self.range_pipeline.identity

    def predict(self, canonical_meter_roi_bgr: np.ndarray):
        self.source_progress.predict(
            canonical_meter_roi_bgr.copy(), input_is_canonical_meter_roi=True
        )
        return self.range_pipeline.predict(canonical_meter_roi_bgr)


def _external_progress(
    config: Mapping[str, Any], garc_plan: Mapping[str, Any]
) -> Any:
    kind = str(config.get("progress_kind") or "")
    if kind == "external_progress_factory":
        binding = _load_component(
            config.get("progress_binding"), label="external progress binding"
        )
        function = _load_callable(
            config.get("progress_factory"), label="external progress"
        )
        plan = copy.deepcopy(dict(garc_plan))
        plan["progress_component"] = {"binding": binding.as_record()}
        factory_binding = config["progress_factory"]
        plan["progress_factory"] = {
            "path": str(Path(factory_binding["path"]).resolve(strict=True)),
            "sha256": str(factory_binding["sha256"]).casefold(),
            "function": str(factory_binding["function"]),
        }
        provider = function(plan)
        binding.verify_provider(provider)
        return provider
    if kind == "original_transformer":
        pointer = _binding(
            config.get("pointer_segmentation"), label="Transformer pointer segmentation"
        )
        transformer = _binding(
            config.get("original_transformer"), label="original Transformer"
        )
        return TransformerFullAutoProgressProvider.from_frozen_files(
            pointer_segmentation_path=pointer,
            original_transformer_path=transformer,
            device=str(config.get("device") or "cuda:0"),
            task_config=config.get("task_config"),
        )
    raise ValueError(f"unsupported shared-range progress_kind: {kind}")


def build_field_blind_full_auto_providers(
    bundle_descriptor: Mapping[str, Any],
    factory_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build exactly the providers described by a frozen blind-test bundle."""

    assert_label_free(factory_config, location="field_blind_factory_config")
    _require(factory_config.get("protocol") == PROTOCOL, "runtime factory protocol drift")
    mode = str(factory_config.get("mode") or "")
    _require(mode in MODES, "unsupported runtime factory mode")
    bundle = FrozenFullAutoBundle(bundle_descriptor)
    _, source_plan = _plan(factory_config.get("source_plan"), label="source plan")
    source_adapter, source_range = build_runtime(source_plan)
    if mode in {"garc_plan", "v5_plan"}:
        progress = source_adapter.progress_provider
        numeric_range = source_range
    else:
        progress = _external_progress(factory_config, source_plan)
        numeric_range = _PreparedSharedGARCRange(
            source_adapter.progress_provider, source_range
        )
    bundle.progress_binding.verify_provider(progress)
    bundle.range_binding.verify_provider(numeric_range)
    return {
        "progress_provider": progress,
        "automatic_numeric_range_pipeline": numeric_range,
    }


__all__ = ["PROTOCOL", "build_field_blind_full_auto_providers"]
