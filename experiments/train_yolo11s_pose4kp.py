"""Train a matched four-keypoint YOLO11s-Pose model on SyncG dial ROIs."""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from experiments.roi_geometry_comparison import write_json
from experiments.train_vdn_syncg import _matched_geoprr_partition
from experiments.vdn_baseline import load_syncg_manifest, set_random_seed
from experiments.yolo11s_pose4kp import METHOD, materialize_pose_dataset


def _subset(values: Sequence[Any], limit: int | None, *, seed: int) -> list[Any]:
    selected = list(values)
    if limit is None or limit >= len(selected):
        return selected
    if limit < 1:
        raise ValueError("subset limit must be positive")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(selected)), int(limit)))
    return [selected[index] for index in indices]


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_random_seed(int(args.seed))
    samples, _protocol = load_syncg_manifest(
        Path(args.manifest), expected_split="train"
    )
    train_samples, validation_samples = _matched_geoprr_partition(
        samples, Path(args.outer_split).resolve()
    )
    train_samples = _subset(train_samples, args.train_limit, seed=args.seed)
    validation_samples = _subset(
        validation_samples, args.validation_limit, seed=args.seed + 10_000
    )

    run_dir = Path(args.output_dir).resolve()
    if (run_dir / "training_summary.json").exists() or (run_dir / "ultralytics" / "weights").exists():
        raise FileExistsError(f"YOLO11s-Pose run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml, materialization = materialize_pose_dataset(
        train_samples,
        validation_samples,
        output_root=run_dir / "dataset",
        image_size=int(args.image_size),
        jpeg_quality=int(args.jpeg_quality),
    )

    model = YOLO(str(args.weights))
    results = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.image_size),
        batch=int(args.batch_size),
        workers=int(args.workers),
        device=str(args.device),
        seed=int(args.seed),
        deterministic=True,
        project=str(run_dir),
        name="ultralytics",
        exist_ok=False,
        pretrained=True,
        optimizer="AdamW",
        lr0=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        cos_lr=True,
        patience=0,
        amp=not args.no_amp,
        cache=False,
        plots=False,
        save=True,
        val=True,
        verbose=False,
        single_cls=True,
        degrees=20.0,
        translate=0.04,
        scale=0.08,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,
    )
    save_dir = Path(results.save_dir).resolve()
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError("Ultralytics did not emit best.pt and last.pt")
    metrics = {
        str(key): float(value)
        for key, value in (getattr(results, "results_dict", {}) or {}).items()
        if isinstance(value, (int, float))
    }
    summary = {
        "schema_version": 1,
        "method": METHOD,
        "status": "complete",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "subset_run": bool(args.train_limit is not None or args.validation_limit is not None),
        "checkpoint_selection": "Ultralytics best fixed-inner-validation fitness",
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "weights_initialization": str(args.weights),
        "materialization": materialization,
        "validation_metrics": metrics,
    }
    write_json(run_dir / "training_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/syncg_train.jsonl"))
    parser.add_argument("--outer-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11s-pose.pt")
    parser.add_argument("--seed", type=int, default=20262020)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--device", default="0")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.image_size < 64 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("invalid epoch, image-size, batch-size, or workers")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be between 1 and 100")
    summary = train(args)
    print(json.dumps({key: summary[key] for key in ("status", "method", "seed", "best_checkpoint")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "train"]
