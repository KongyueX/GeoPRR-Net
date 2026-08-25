"""Evaluate the publication R²MT-Net checkpoint on the SyncG protocol."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments.evaluate_a15_2_mett_syncg import DEFAULT_REFERENCE, evaluate_mett
from experiments.evaluate_a15_2_syncg_scene_holdout import DEFAULT_MANIFEST
from experiments.r2mt_net import ARCHITECTURE, publication_model_identity
from experiments.train_r2mt_net import (
    PROTOCOL as TRAINING_PROTOCOL,
    load_r2mt_net_checkpoint,
)


PROTOCOL: Final[str] = "syncg_r2mt_net_eval_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_metadata(
    metadata: Mapping[str, Any], *, expected_variant: str
) -> dict[str, Any]:
    scope = metadata.get("training_scope")
    construction = metadata.get("construction")
    counts = metadata.get("parameter_counts")
    _require(
        expected_variant == "direct_scalar"
        and metadata.get("protocol") == TRAINING_PROTOCOL
        and metadata.get("architecture") == ARCHITECTURE
        and isinstance(scope, Mapping)
        and isinstance(construction, Mapping)
        and isinstance(counts, Mapping)
        and scope.get("complete_original_fit") is True
        and scope.get("formal_holdout_access") is False
        and scope.get("main_table_development_access") is False
        and int(construction.get("additional_image_encoders", -1)) == 0
        and int(counts.get("additional_image_encoders", -1)) == 0,
        "risk arbitration evaluation metadata differs",
    )
    return {
        "training_protocol": TRAINING_PROTOCOL,
        "experiment_variant": expected_variant,
        "additional_image_encoders": 0,
        "risk_gate_parameters": int(counts["risk_gate"]),
        "risk_heads": 3,
    }


def _publication_identity(_expected_variant: str) -> dict[str, str]:
    return publication_model_identity()


def evaluate_r2mt_net(
    *,
    checkpoint_path: Path,
    output_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    reference_path: Path = DEFAULT_REFERENCE,
    device_name: str = "cuda:0",
    workers: int = 4,
    batch_size: int = 64,
    use_amp: bool = False,
    adaptive_strength: float = 1.0,
    fusion_mode: str = "single_exact_transport",
) -> dict[str, Any]:
    _require(
        0.0 <= float(adaptive_strength) <= 1.0
        and fusion_mode in {"single_exact_transport", "posterior_mixture"},
        "risk arbitration evaluation fusion differs",
    )

    def configured_loader(checkpoint: Path, *, device: Any):
        anchor, model, metadata = load_r2mt_net_checkpoint(
            checkpoint, device=device
        )
        model.adaptive_strength = float(adaptive_strength)
        model.fusion_mode = str(fusion_mode)
        metadata = dict(metadata)
        construction = dict(metadata["construction"])
        construction["evaluation_adaptive_strength"] = float(
            adaptive_strength
        )
        construction["evaluation_fusion_mode"] = str(fusion_mode)
        metadata["construction"] = construction
        return anchor, model, metadata

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
        trained_checkpoint_loader=configured_loader,
        trained_checkpoint_validator=validate_metadata,
        publication_identity_resolver=_publication_identity,
        evaluation_protocol=PROTOCOL,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--adaptive-strength", type=float, default=1.0)
    parser.add_argument(
        "--fusion-mode",
        choices=("single_exact_transport", "posterior_mixture"),
        default="single_exact_transport",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_r2mt_net(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        manifest_path=args.manifest,
        reference_path=args.reference,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        use_amp=args.amp,
        adaptive_strength=args.adaptive_strength,
        fusion_mode=args.fusion_mode,
    )
    persisted = json.loads(Path(args.output).resolve().read_text(encoding="utf-8-sig"))
    summary = persisted["summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                **{
                    f"{condition}_nmae": summary[condition]["candidate"]["mett"][
                        "nmae"
                    ]
                    for condition in (
                        "clean",
                        "blur_moderate",
                        "blur_severe",
                        "perspective_moderate",
                        "perspective_severe",
                        "combined_severe",
                        "projective_pooled",
                        "all_conditions",
                    )
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Internal compatibility alias; public documentation uses evaluate_r2mt_net.
evaluate_risk_arbitration = evaluate_r2mt_net


__all__ = [
    "PROTOCOL",
    "evaluate_r2mt_net",
    "evaluate_risk_arbitration",
    "validate_metadata",
]
