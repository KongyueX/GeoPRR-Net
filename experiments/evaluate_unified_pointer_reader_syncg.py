"""Evaluate one frozen unified-reader variant on the six-condition SyncG holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final, Sequence

from experiments.evaluate_remstnet_syncg import evaluate_remstnet_syncg
from experiments.unified_pointer_reader import (
    PROTOCOL as TRAINING_PROTOCOL,
    PUBLICATION_NAME,
    adaptive_routing_enabled,
    load_unified_pointer_reader_checkpoint,
)


PROTOCOL: Final[str] = "unified_pointer_reader_syncg_six_condition_evaluation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def annotate_unified_output(
    output_path: Path,
    *,
    expected_protocol: str = PROTOCOL,
) -> dict[str, Any]:
    """Replace compatibility labels with the publication-facing model identity."""

    output = Path(output_path).resolve()
    payload = json.loads(output.read_text(encoding="utf-8"))
    _require(
        payload.get("protocol") == str(expected_protocol),
        "evaluation protocol differs",
    )
    model = payload.get("model")
    _require(isinstance(model, dict), "model metadata is missing")
    _require(model.get("protocol") == TRAINING_PROTOCOL, "training protocol differs")
    variant = str(model.get("variant"))
    scope = payload.setdefault("scope", {})
    scope.update(
        {
            "development_cohort": False,
            "formal_syncg_scene_holdout": True,
            "single_seed_architecture_pilot": False,
            "candidate_machine_key": "unified_pointer_reader",
            "legacy_candidate_machine_key": "mett",
            "compatibility_candidate_aliases": ["mett", "remstnet"],
            "compatibility_alias_note": (
                "The mett/remstnet fields contain the unified-reader prediction "
                "only because the shared evaluator retains historical machine keys."
            ),
            "prediction_dependent_routing": adaptive_routing_enabled(variant),
            "frozen_checkpoint": True,
            "training_or_adaptation_during_evaluation": False,
        }
    )
    payload["status"] = "complete"
    payload["publication_model"] = PUBLICATION_NAME
    payload["architecture_variant"] = variant
    output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def evaluate_unified_syncg(
    *,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
    workers: int,
    batch_size: int,
    use_amp: bool,
) -> dict[str, Any]:
    evaluate_remstnet_syncg(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        device_name=device_name,
        workers=workers,
        batch_size=batch_size,
        use_amp=use_amp,
        checkpoint_loader=load_unified_pointer_reader_checkpoint,
        evaluation_protocol=PROTOCOL,
        candidate_display_name=PUBLICATION_NAME,
    )
    return annotate_unified_output(output_path)


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
    result = evaluate_unified_syncg(
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
                "nmae": candidate["nmae"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PROTOCOL", "annotate_unified_output", "evaluate_unified_syncg"]
