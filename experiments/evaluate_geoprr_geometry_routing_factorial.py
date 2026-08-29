"""Evaluate the missing geometry-off plus fixed-routing factorial cell.

The joint cell is an inference-only intervention.  It loads the existing
``fixed_routing`` checkpoint for one seed, verifies that the router was not
trained, disables geometry-aware base fusion, and continues to use the same
fixed prior weights.  No joint-specific parameters are loaded or fitted.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

from experiments.evaluate_remstnet_syncg import evaluate_remstnet_syncg
from experiments.evaluate_unified_pointer_reader_syncg import annotate_unified_output
from experiments.unified_pointer_reader import (
    FIXED_ROUTING,
    NO_GEOMETRY_FIXED_ROUTING,
    PROTOCOL as TRAINING_PROTOCOL,
    PUBLICATION_NAME,
    load_unified_pointer_reader_checkpoint,
    parameter_inventory,
)


PROTOCOL: Final[str] = "geoprr_geometry_routing_joint_factorial_evaluation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_joint_factorial_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a fixed-routing checkpoint and stack geometry removal at inference."""

    source = Path(checkpoint_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    _require(isinstance(payload, Mapping), "joint source checkpoint is malformed")
    _require(payload.get("protocol") == TRAINING_PROTOCOL, "training protocol differs")
    _require(payload.get("variant") == FIXED_ROUTING, "joint source must be fixed routing")
    training = payload.get("training")
    _require(isinstance(training, Mapping), "fixed-routing training metadata is missing")
    _require(training.get("router_training") is False, "fixed router unexpectedly has training")

    model, metadata = load_unified_pointer_reader_checkpoint(source, device=device)
    _require(model.variant == FIXED_ROUTING, "loaded source variant differs")
    model.variant = NO_GEOMETRY_FIXED_ROUTING
    result = dict(metadata)
    result.update(
        {
            "source_checkpoint_variant": FIXED_ROUTING,
            "variant": NO_GEOMETRY_FIXED_ROUTING,
            "joint_factorial_cell": {
                "geometry_on": False,
                "adaptive_routing_on": False,
                "training_or_adaptation": False,
                "joint_specific_parameters_loaded": False,
                "router_parameters_executed": False,
                "source": "existing fixed-routing checkpoint",
            },
            "parameter_inventory": parameter_inventory(model),
        }
    )
    return model, result


def evaluate_joint_factorial(
    *,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
    workers: int,
    batch_size: int,
    use_amp: bool,
) -> dict[str, Any]:
    """Run the complete six-condition SyncG evaluation for one seed."""

    evaluate_remstnet_syncg(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        device_name=device_name,
        workers=workers,
        batch_size=batch_size,
        use_amp=use_amp,
        checkpoint_loader=load_joint_factorial_checkpoint,
        evaluation_protocol=PROTOCOL,
        candidate_display_name=f"{PUBLICATION_NAME} geometry-off fixed-routing",
    )
    payload = annotate_unified_output(output_path, expected_protocol=PROTOCOL)
    scope = payload.setdefault("scope", {})
    scope.update(
        {
            "geometry_on": False,
            "adaptive_routing_on": False,
            "joint_factorial_inference_only": True,
            "joint_specific_training_or_adaptation": False,
            "joint_specific_parameters_loaded": False,
        }
    )
    Path(output_path).resolve().write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_joint_factorial(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )
    candidate = result["summary"]["all_conditions"]["candidate"]["mett"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "variant": result["architecture_variant"],
                "rows": result["data"]["rows"],
                "nmae": candidate["nmae"],
                "acc_at_2_percent": candidate["acc_at_2_percent"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "evaluate_joint_factorial",
    "load_joint_factorial_checkpoint",
]
