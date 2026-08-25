"""Build one fixed midpoint-shrunk, single-head ResNet18 endpoint."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.refresh_resnet18_endpoint import (
    DEFAULT_REFRESH_EPOCHS,
    PROTOCOL as REFRESH_PROTOCOL,
    RAW_CONDITIONS,
    SHRUNK_ENDPOINT_PROTOCOL,
)
from experiments.resnet18_direct_progress import (
    PROTOCOL as DIRECT_PROTOCOL,
)


PROTOCOL: Final[str] = SHRUNK_ENDPOINT_PROTOCOL
DEFAULT_SOURCE_WEIGHT: Final[float] = 0.5
FC_STATE_NAMES: Final[tuple[str, ...]] = (
    "backbone.fc.weight",
    "backbone.fc.bias",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_shrunk_endpoint(
    endpoint_refresh_path: Path,
    *,
    output_path: Path,
    source_weight: float = DEFAULT_SOURCE_WEIGHT,
) -> dict[str, Any]:
    _require(
        0.0 < source_weight < 1.0,
        "source endpoint weight must be strictly between zero and one",
    )
    parent_source = Path(endpoint_refresh_path).resolve()
    output = Path(output_path).resolve()
    _require(parent_source.is_file(), f"endpoint refresh is missing: {parent_source}")
    _require(not output.exists(), f"shrunk endpoint output exists: {output}")
    parent = torch.load(parent_source, map_location="cpu", weights_only=False)
    _require(
        isinstance(parent, Mapping)
        and parent.get("protocol") == REFRESH_PROTOCOL
        and int(parent.get("epochs", -1)) == DEFAULT_REFRESH_EPOCHS
        and parent.get("encoder_state_unchanged") is True,
        "parent endpoint refresh metadata differs",
    )
    refreshed_state = parent.get("model_state")
    direct_source = Path(str(parent["source_direct_checkpoint"])).resolve()
    direct = torch.load(direct_source, map_location="cpu", weights_only=False)
    direct_state = direct.get("model_state") if isinstance(direct, Mapping) else None
    _require(
        isinstance(direct, Mapping)
        and direct.get("protocol") == DIRECT_PROTOCOL
        and isinstance(direct_state, Mapping)
        and isinstance(refreshed_state, Mapping),
        "source Direct-ResNet18 state differs",
    )
    derived_state = {
        str(name): value.detach().cpu().clone()
        for name, value in refreshed_state.items()
    }
    refreshed_weight = 1.0 - float(source_weight)
    for name in FC_STATE_NAMES:
        _require(
            name in direct_state
            and name in refreshed_state
            and direct_state[name].shape == refreshed_state[name].shape,
            f"endpoint state is missing or misaligned: {name}",
        )
        derived_state[name] = (
            direct_state[name].detach().cpu().float() * float(source_weight)
            + refreshed_state[name].detach().cpu().float() * refreshed_weight
        ).to(refreshed_state[name].dtype)

    checkpoint = {
        **dict(parent),
        "protocol": PROTOCOL,
        "parent_endpoint_refresh_checkpoint": str(parent_source),
        "endpoint_shrinkage": {
            "mode": "single_linear_head_parameter_interpolation",
            "source_endpoint_weight": float(source_weight),
            "refreshed_endpoint_weight": refreshed_weight,
            "selection_scope": "historical_development_ceiling_probe",
            "inference_linear_heads": 1,
            "additional_parameters": 0,
        },
        "model_state": derived_state,
        "calibration": {
            "fit_population": "identity_after_anchored_midpoint_endpoint",
            "fit_rows": 0,
            "conditions": list(RAW_CONDITIONS),
            "knots_x": [0.0, 0.5, 1.0],
            "knots_y": [0.0, 0.5, 1.0],
            "identity_mapping": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {
        "status": "complete",
        "protocol": PROTOCOL,
        "output": str(output),
        "parent_endpoint_refresh_checkpoint": str(parent_source),
        "source_direct_checkpoint": str(direct_source),
        "source_seed": int(parent["source"]["seed"]),
        "source_endpoint_weight": float(source_weight),
        "refreshed_endpoint_weight": refreshed_weight,
        "inference_linear_heads": 1,
        "additional_parameters": 0,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-refresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-weight", type=float, default=DEFAULT_SOURCE_WEIGHT
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = build_shrunk_endpoint(
        args.endpoint_refresh,
        output_path=args.output,
        source_weight=args.source_weight,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_SOURCE_WEIGHT", "PROTOCOL", "build_shrunk_endpoint"]
