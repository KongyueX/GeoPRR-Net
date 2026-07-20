"""Evaluate frozen pointer-segmentation checkpoints on the SyncG test split.

This is a component-level, ground-truth-dial-crop diagnostic.  Thresholds are
loaded from packaged checkpoints (selected on SyncG train validation); legacy
raw state dictionaries use their historical effective threshold of 1/255.
The test split is never searched for a better threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from experiments.train_syncg_segmentation import (
    SyncGPointerSegDataset,
    _seed_worker,
    discover_syncg_seg_samples,
    retain_largest_components,
)
from utils.angleDetect.pointerSeg.detectSeg import load_u2net_state_dict
from utils.angleDetect.pointerSeg.u2netp import U2NETP


LEGACY_EFFECTIVE_THRESHOLD = 1.0 / 255.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "checkpoint must use NAME=PATH, for example released=weights.pt"
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint name and path cannot be empty")
    return name, path


@torch.inference_mode()
def _evaluate_checkpoint(
    name: str,
    path: Path,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state_dict, metadata = load_u2net_state_dict(path, device)
    model = U2NETP().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    embedded_threshold = metadata.get("probability_threshold")
    if embedded_threshold is None:
        threshold = LEGACY_EFFECTIVE_THRESHOLD
        threshold_source = "legacy_effective_1_over_255"
    else:
        threshold = float(embedded_threshold)
        threshold_source = "checkpoint_train_validation"
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{path}: invalid frozen threshold {threshold}")

    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    sample_dice: list[float] = []
    sample_iou: list[float] = []
    sample_count = 0
    started = time.perf_counter()
    for images, targets, letterbox_content in tqdm(
        loader,
        desc=f"evaluate {name}",
        leave=False,
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).bool()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images, return_logits=True)[0]
        probability = torch.sigmoid(logits.float())
        probability_np = probability.cpu().numpy()[:, 0]
        target_np = targets.cpu().numpy()[:, 0]
        for sample_index, geometry in enumerate(letterbox_content.tolist()):
            left, top, resized_width, resized_height = map(int, geometry)
            sample_probability = probability_np[
                sample_index,
                top : top + resized_height,
                left : left + resized_width,
            ]
            target = target_np[
                sample_index,
                top : top + resized_height,
                left : left + resized_width,
            ]
            prediction = retain_largest_components(
                sample_probability >= threshold
            )[0]
            tp = float(np.sum(prediction & target))
            fp = float(np.sum(prediction & ~target))
            fn = float(np.sum(~prediction & target))
            true_positive += tp
            false_positive += fp
            false_negative += fn
            sample_dice.append(
                (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
            )
            sample_iou.append(
                (tp + 1.0) / (tp + fp + fn + 1.0)
            )
            sample_count += 1

    micro_dice = (2.0 * true_positive + 1.0) / (
        2.0 * true_positive + false_positive + false_negative + 1.0
    )
    micro_iou = (true_positive + 1.0) / (
        true_positive + false_positive + false_negative + 1.0
    )
    precision = (true_positive + 1.0) / (
        true_positive + false_positive + 1.0
    )
    recall = (true_positive + 1.0) / (
        true_positive + false_negative + 1.0
    )
    return {
        "name": name,
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_format": metadata.get("format_version", "legacy_raw_state_dict"),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selected_on_test": False,
        "postprocess": (
            "remove_letterbox_padding_then_largest_connected_component"
        ),
        "samples": sample_count,
        "micro_dice": micro_dice,
        "micro_iou": micro_iou,
        "precision": precision,
        "recall": recall,
        "macro_dice": float(np.mean(sample_dice)),
        "macro_iou": float(np.mean(sample_iou)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=_checkpoint_argument,
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument("--limit", type=int, help="diagnostic subset only")
    parser.add_argument("--allow-dataset-drift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers non-negative")
    names = [name for name, _ in args.checkpoint]
    if len(set(names)) != len(names):
        raise ValueError("checkpoint names must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_enabled = bool(args.amp and device.type == "cuda")

    root, samples = discover_syncg_seg_samples(
        args.root,
        split="test",
        limit=args.limit,
        strict_release=not args.allow_dataset_drift,
    )
    dataset = SyncGPointerSegDataset(
        samples,
        train=False,
        crop_padding=args.crop_padding,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        persistent_workers=args.workers > 0,
    )

    results = [
        _evaluate_checkpoint(
            name,
            path,
            loader,
            device=device,
            amp_enabled=amp_enabled,
        )
        for name, path in args.checkpoint
    ]
    payload = {
        "protocol": "syncg_test_gt_dial_crop_segmentation_v1",
        "component_diagnostic": True,
        "end_to_end_result": False,
        "test_used_for_threshold_selection": False,
        "syncg_root": str(root),
        "split": "test",
        "samples": len(samples),
        "crop_padding": args.crop_padding,
        "device": str(device),
        "amp": amp_enabled,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
