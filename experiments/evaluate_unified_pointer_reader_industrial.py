"""Evaluate a frozen unified-reader checkpoint on the 1,395 Industrial images."""
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


PROTOCOL: Final[str] = "unified_pointer_reader_industrial_1395_frozen_evaluation_v1"
INDUSTRIAL_KEYS: Final[tuple[str, ...]] = (
    "field_gauge_roi_test_a",
    "field_gauge_roi_test_b",
    "field_gauge_external_roi",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def evaluate_unified_industrial(
    *,
    checkpoint_path: Path,
    output_path: Path,
    device_name: str,
    workers: int,
    batch_size: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    _require(not output.exists(), f"Industrial evaluation output exists: {output}")
    _require(
        set(INDUSTRIAL_KEYS) <= set(mett_real.REAL_DATASET_KEYS),
        "Industrial cohort keys differ",
    )
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
    datasets: dict[str, Any] = {}
    for dataset_index, key in enumerate(INDUSTRIAL_KEYS):
        result = remst_real._evaluate_dataset(
            mett_real.factorial.DATASETS[key],
            model=model,
            source_seed=source_seed,
            prediction_root=mett_real.DEFAULT_PREDICTION_ROOT,
            device=device,
            workers=workers,
            batch_size=batch_size,
            bootstrap_replicates=bootstrap_replicates,
            include_posterior_diagnostics=False,
            dataset_index=dataset_index,
            enforce_endpoint_replay=False,
        )
        datasets[key] = result
        metric = result["summary"]["all_conditions"]["candidate"]["mett"]
        print(
            json.dumps(
                {
                    "phase": "industrial_evaluation",
                    "dataset": key,
                    "samples": result["dataset"]["samples"],
                    "nmae": metric["full_denominator"]["nmae"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    samples = sum(int(value["dataset"]["samples"]) for value in datasets.values())
    _require(samples == 1395, "Industrial cohort is not the fixed 1,395 images")
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "complete",
        "publication_model": PUBLICATION_NAME,
        "architecture_variant": metadata["variant"],
        "scope": {
            "cohort": "Test-A + Test-B + External-ROI",
            "samples": samples,
            "conditions_per_sample": len(mett_real.CONDITIONS),
            "rows": samples * len(mett_real.CONDITIONS),
            "industrial_images_test_only": True,
            "training_or_adaptation_during_evaluation": False,
            "checkpoint_frozen_before_access": True,
            "checkpoint_selection_using_industrial": False,
            "rf100_excluded": True,
            "candidate_machine_key": "mett",
            "candidate_machine_key_note": (
                "mett is retained only by the shared evaluator; every candidate "
                "value is the unified-reader output."
            ),
            "legacy_endpoint_replay": (
                "audited but not enforced because it compares internal auxiliary "
                "endpoints with a historical external EfficientNet-B0 ledger, "
                "not the unified reader's final prediction"
            ),
        },
        "model": metadata,
        "source_seed": source_seed,
        "conditions": list(mett_real.CONDITIONS),
        "evaluation_elapsed_seconds": time.perf_counter() - started,
        "datasets": datasets,
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
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = evaluate_unified_industrial(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        bootstrap_replicates=args.bootstrap,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(Path(args.output).resolve()),
                "variant": result["architecture_variant"],
                "samples": result["scope"]["samples"],
                "elapsed_seconds": result["evaluation_elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["INDUSTRIAL_KEYS", "PROTOCOL", "evaluate_unified_industrial"]
