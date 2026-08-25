"""Evaluate terminal ReMST-ResNet18 on the matched six-condition cohort."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.evaluate_a15_2_mett_syncg import (
    DEFAULT_REFERENCE,
    evaluate_mett,
)
from experiments.evaluate_a15_2_syncg_scene_holdout import DEFAULT_MANIFEST
from experiments.remst_resnet18 import remst_resnet18_publication_identity
from experiments.train_remst_resnet18_probe import (
    TERMINAL_EPOCHS,
    load_remst_resnet18_probe,
)


PROTOCOL: Final[str] = "syncg_remst_resnet18_development_evaluation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_remst_resnet18_metadata(
    metadata: Mapping[str, Any], *, expected_variant: str
) -> dict[str, Any]:
    """Validate the terminal, frozen, single-backbone experiment identity."""

    _require(expected_variant == "direct_scalar", "unexpected ReMST experiment variant")
    _require(
        int(metadata.get("epochs", -1)) == TERMINAL_EPOCHS,
        "ReMST-ResNet18 checkpoint is not terminal",
    )
    evidence = metadata.get("single_backbone_evidence")
    _require(isinstance(evidence, Mapping), "single-backbone evidence is missing")
    _require(
        int(evidence.get("image_encoder_modules", -1)) == 1
        and evidence.get("raw_and_sarn_share_parameter_objects") is True
        and evidence.get("anchor_frozen") is True
        and evidence.get("anchor_state_unchanged") is True,
        "ReMST-ResNet18 single-backbone evidence differs",
    )
    refresh_source = metadata.get("source_endpoint_refresh_checkpoint")
    _require(
        refresh_source is not None and bool(str(refresh_source).strip()),
        "final ReMST-ResNet18 checkpoint lacks the endpoint refresh",
    )
    return {
        **remst_resnet18_publication_identity(),
        "experiment_variant": expected_variant,
        "terminal_epochs": TERMINAL_EPOCHS,
        "endpoint_refreshed": True,
    }


def _publication_identity(_expected_variant: str) -> dict[str, str]:
    return remst_resnet18_publication_identity()


def evaluate_remst_resnet18(
    *,
    manifest_path: Path,
    reference_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
) -> dict[str, Any]:
    return evaluate_mett(
        manifest_path=manifest_path,
        reference_path=reference_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        device_name=device_name,
        workers=workers,
        batch_size=batch_size,
        use_amp=use_amp,
        expected_variant="direct_scalar",
        trained_checkpoint_loader=load_remst_resnet18_probe,
        trained_checkpoint_validator=validate_remst_resnet18_metadata,
        publication_identity_resolver=_publication_identity,
        evaluation_protocol=PROTOCOL,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_remst_resnet18(
        manifest_path=args.manifest,
        reference_path=args.reference,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROTOCOL",
    "evaluate_remst_resnet18",
    "validate_remst_resnet18_metadata",
]
