"""Evaluate SyncG-only PG-SIAM against its same-seed GeoAttn parent.

The PG-SIAM checkpoint loader accepts only checkpoints produced by the
synthetic-only trainer.  Field images enter only the label-free prediction
stage; labels are opened later by the ``score`` command.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from experiments import evaluate_projective_geometry_guided_siam as pg_evaluation
from experiments import train_projective_geometry_guided_siam_syncg as pg_training
from experiments.evaluate_paper_syncg_only_factorial import (
    DATASETS,
    SEEDS,
    Dataset,
    prediction_artifacts as parent_prediction_artifacts,
)
from experiments.resnet18_direct_progress import _canonical_json_bytes
from experiments.score_cagh_v5_plain_paper_batch import (
    load_targets,
    load_validation_ids,
)
from experiments.summarize_pgsiam_v3_multiseed import summarize_multiseed
from experiments.summarize_support_normalized_cbam_pilot import load_scene_targets


DEFAULT_ROOT: Final[Path] = Path("C:/pointer_read/paper_syncg_only_retrain_v1")
MODEL_DIRECTORY: Final[str] = "pgsiam"
PAPER_LABEL: Final[str] = "PG-SIAM"


class PaperPGSIAMEvaluationError(RuntimeError):
    """Invalid PG-SIAM paper-evaluation configuration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperPGSIAMEvaluationError(message)


def checkpoint_path(root: Path, *, seed: int) -> Path:
    return Path(root) / MODEL_DIRECTORY / f"seed_{seed}" / "terminal.pt"


def method_id(seed: int) -> str:
    return f"{PAPER_LABEL}_seed_{seed}"


def prediction_artifacts(
    root: Path,
    *,
    seed: int,
    dataset: Dataset,
) -> dict[str, Path]:
    base = (
        Path(root)
        / MODEL_DIRECTORY
        / "evaluation"
        / dataset.slug
        / f"seed_{seed}"
    )
    return {
        "predictions": base / "predictions.jsonl",
        "diagnostics": base / "diagnostics.jsonl",
        "summary": base / "summary.json",
    }


def _canonical_checkpoint_loader(
    checkpoint: Path,
    *,
    device_name: str,
    expected_seed: int,
):
    _method, model, metadata = pg_training.load_checkpoint_model(
        checkpoint,
        device_name=device_name,
    )
    _require(int(metadata.get("seed", -1)) == expected_seed, "PG-SIAM seed mismatch")
    return method_id(expected_seed), model, metadata


def run_prediction_job(
    *,
    root: Path,
    dataset: Dataset,
    seed: int,
    device_name: str,
) -> Mapping[str, Any]:
    artifacts = prediction_artifacts(root, seed=seed, dataset=dataset)

    def checkpoint_loader(checkpoint: Path, *, device_name: str):
        return _canonical_checkpoint_loader(
            checkpoint,
            device_name=device_name,
            expected_seed=seed,
        )

    return pg_evaluation.run_prediction(
        checkpoint_path=checkpoint_path(root, seed=seed),
        manifest_path=dataset.manifest,
        output_path=artifacts["predictions"],
        diagnostics_path=artifacts["diagnostics"],
        summary_path=artifacts["summary"],
        device_name=device_name,
        checkpoint_loader=checkpoint_loader,
    )


def _targets(dataset: Dataset) -> dict[str, tuple[float, str]]:
    if dataset.slug == "syncg_scene_holdout":
        return load_scene_targets(dataset.labels, dataset.roster)
    roster = load_validation_ids(dataset.roster)
    rows = load_targets(dataset.labels, roster)
    return {
        row.sample_id: (row.normalized_target, row.group_id)
        for row in rows
    }


def score_dataset(
    *,
    root: Path,
    dataset: Dataset,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    _require(bootstrap_replicates >= 1, "bootstrap replicates must be positive")
    factorial_root = Path(root) / "factorial"
    candidate_paths = [
        prediction_artifacts(root, seed=seed, dataset=dataset)["predictions"]
        for seed in SEEDS
    ]
    parent_paths = [
        parent_prediction_artifacts(
            factorial_root,
            cell="11",
            seed=seed,
            dataset=dataset,
            variant="sarn_v2",
        )["predictions"]
        for seed in SEEDS
    ]
    return summarize_multiseed(
        dataset=dataset.paper_name,
        targets=_targets(dataset),
        candidate_paths=candidate_paths,
        parent_paths=parent_paths,
        seeds=SEEDS,
        replicates=bootstrap_replicates,
        bootstrap_seed=20260814,
        group_unit=dataset.group_unit,
    )


def score_all(
    *,
    root: Path,
    datasets: Sequence[Dataset],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    results = {
        dataset.paper_name: score_dataset(
            root=root,
            dataset=dataset,
            bootstrap_replicates=bootstrap_replicates,
        )
        for dataset in datasets
    }
    output = Path(root) / MODEL_DIRECTORY / "evaluation" / "results" / "all_datasets.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "paper_syncg_only_pgsiam_multidataset_summary_v1",
        "status": "complete",
        "candidate": PAPER_LABEL,
        "comparator": "SARN-v2+GeoAttn-ResNet18",
        "training_data": "SyncG fit only",
        "field_data_role": "test only",
        "datasets": results,
    }
    output.write_bytes(_canonical_json_bytes(payload))
    return {
        "status": "complete",
        "output": str(output.resolve()),
        "datasets": list(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prediction = commands.add_parser("predict")
    prediction.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prediction.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    prediction.add_argument("--seed", action="append", type=int, choices=SEEDS)
    prediction.add_argument("--device", default="cuda:0")

    score = commands.add_parser("score")
    score.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    score.add_argument("--dataset", action="append", choices=tuple(DATASETS))
    score.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    datasets = (
        tuple(DATASETS[name] for name in args.dataset)
        if args.dataset
        else tuple(DATASETS.values())
    )
    if args.command == "score":
        print(
            json.dumps(
                score_all(
                    root=args.root,
                    datasets=datasets,
                    bootstrap_replicates=args.bootstrap_replicates,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    seeds = tuple(args.seed) if args.seed else SEEDS
    for seed in seeds:
        for dataset in datasets:
            print(
                json.dumps(
                    {
                        "status": "starting",
                        "dataset": dataset.paper_name,
                        "method": method_id(seed),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            result = run_prediction_job(
                root=args.root,
                dataset=dataset,
                seed=seed,
                device_name=args.device,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ROOT",
    "MODEL_DIRECTORY",
    "PAPER_LABEL",
    "checkpoint_path",
    "main",
    "method_id",
    "prediction_artifacts",
    "run_prediction_job",
    "score_all",
    "score_dataset",
]
