"""Evaluate a frozen GeoPRR-Net checkpoint on the RF100 gauge subset.

The evaluation reuses the fixed ``needle-base-tip-min-max`` test roster: 151
meter ROIs grouped into 35 conservative source groups.  Normalized progress is
derived before inference from the annotated center, minimum endpoint, maximum
endpoint, and pointer tip.  RF100 is therefore reported as an annotation-
derived external transfer cohort, not as an official scalar-reading benchmark.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Final, Sequence

import torch

from experiments import evaluate_a15_2_mett_real_domains as mett_real
from experiments import evaluate_remstnet_real_domains as remst_real
from experiments.unified_pointer_reader import (
    PROTOCOL as TRAINING_PROTOCOL,
    PUBLICATION_NAME,
    load_unified_pointer_reader_checkpoint,
)


PROTOCOL: Final[str] = "unified_pointer_reader_rf100_external_evaluation_v1"
DATASET_KEY: Final[str] = "rf100"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evaluate_unified_rf100(
    *,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
    workers: int,
    batch_size: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Run one frozen publication checkpoint on the complete RF100 roster."""

    output = Path(output_path).resolve()
    _require(not output.exists(), f"RF100 evaluation output exists: {output}")
    _require(DATASET_KEY in mett_real.REAL_DATASET_KEYS, "RF100 dataset is unavailable")
    _require(bootstrap_replicates >= 1, "bootstrap replicate count must be positive")

    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        torch.cuda.manual_seed_all(mett_real.BOOTSTRAP_SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(mett_real.BOOTSTRAP_SEED)

    model, metadata = load_unified_pointer_reader_checkpoint(
        checkpoint_path, device=device
    )
    _require(metadata.get("protocol") == TRAINING_PROTOCOL, "training protocol differs")
    source_seed = int(metadata["source_seed"])
    model.eval()

    started = time.perf_counter()
    dataset_result = remst_real._evaluate_dataset(
        mett_real.factorial.DATASETS[DATASET_KEY],
        model=model,
        source_seed=source_seed,
        prediction_root=mett_real.DEFAULT_PREDICTION_ROOT,
        device=device,
        workers=workers,
        batch_size=batch_size,
        bootstrap_replicates=bootstrap_replicates,
        include_posterior_diagnostics=False,
        dataset_index=3,
        enforce_endpoint_replay=False,
    )
    _require(dataset_result["dataset"]["samples"] == 151, "RF100 roster differs")
    _require(dataset_result["dataset"]["groups"] == 35, "RF100 groups differ")

    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "publication_model": PUBLICATION_NAME,
        "architecture_variant": metadata["variant"],
        "scope": {
            "cohort": "RF100-VL needle-base-tip-min-max test split",
            "samples": 151,
            "groups": 35,
            "conditions_per_sample": len(mett_real.CONDITIONS),
            "rows": 151 * len(mett_real.CONDITIONS),
            "frozen_checkpoint": True,
            "training_or_adaptation_during_evaluation": False,
            "normalized_progress_derived_from_annotations_before_inference": True,
            "official_rf100_scalar_reading_benchmark": False,
            "newly_collected_physical_meter_cohort": False,
            "candidate_machine_key": "mett",
            "candidate_machine_key_note": (
                "The shared evaluator retains the mett field name; every candidate "
                "value is the GeoPRR-Net output."
            ),
        },
        "model": metadata,
        "source_seed": source_seed,
        "conditions": list(mett_real.CONDITIONS),
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "dataset": dataset_result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--batch-size", type=int, default=mett_real.DEFAULT_REAL_DOMAIN_BATCH_SIZE
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_unified_rf100(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap,
    )
    metric = result["dataset"]["summary"]["all_conditions"]["candidate"]["mett"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "seed": result["source_seed"],
                "nmae": metric["full_denominator"]["nmae"],
                "elapsed_seconds": result["evaluation_elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DATASET_KEY", "PROTOCOL", "evaluate_unified_rf100"]
