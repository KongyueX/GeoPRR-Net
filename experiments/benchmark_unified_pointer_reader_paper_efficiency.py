"""Measure the full reader with the same batch-1 FP32 pipeline as external CNNs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Final, Sequence

import cv2
import numpy as np
import torch

from experiments.benchmark_cagh_v5_paper_efficiency import BenchmarkTarget
from experiments.benchmark_paper_syncg_only_efficiency import (
    DEFAULT_LIMIT,
    DEFAULT_MANIFEST,
    DEFAULT_WARMUP,
    run_benchmark,
)
from experiments.evaluate_a15_2_syncg_scene_holdout import (
    _normalized_raw_to_sarn_homography,
)
from experiments.pivot_direction_fallback import normalized_rgb_tensor
from experiments.resnet18_direct_progress import IMAGE_SIZE
from experiments.support_aware_roi_normalization_v2 import (
    normalize_support_aware_roi_v2,
)
from experiments.unified_pointer_reader import (
    PUBLICATION_NAME,
    load_unified_pointer_reader_checkpoint,
)
from experiments.v5_shared_roi_comparison_input import direct_resize_whole_roi


ARM: Final[str] = "unified_pointer_reader"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_target(
    arm: str,
    *,
    checkpoint_path: Path,
    device: str,
) -> BenchmarkTarget:
    _require(arm == ARM, "unexpected efficiency arm")
    runtime_device = torch.device(device)
    model, metadata = load_unified_pointer_reader_checkpoint(
        checkpoint_path, device=runtime_device
    )
    model.eval()

    def predict(image_bgr: np.ndarray) -> float:
        degraded = np.ascontiguousarray(image_bgr)
        decision = normalize_support_aware_roi_v2(degraded)
        height, width = degraded.shape[:2]
        active = bool(decision.applied)
        raw_mask = decision.valid_support_mask
        if raw_mask is None:
            active = False
            support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        else:
            support = cv2.resize(
                np.asarray(raw_mask, dtype=np.float32),
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )
            support = np.clip(support, 0.0, 1.0).astype(np.float32)
            if not np.isfinite(support).all() or float(support.sum()) <= 0.0:
                active = False
                support = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        homography = (
            _normalized_raw_to_sarn_homography(
                decision,
                height=height,
                width=width,
            )
            if active
            else np.eye(3, dtype=np.float32)
        )
        raw = normalized_rgb_tensor(
            direct_resize_whole_roi(degraded, size=IMAGE_SIZE)
        )[None].to(runtime_device)
        normalized = normalized_rgb_tensor(
            direct_resize_whole_roi(decision.image, size=IMAGE_SIZE)
        )[None].to(runtime_device)
        support_tensor = torch.from_numpy(
            np.ascontiguousarray(support[None, None], dtype=np.float32)
        ).to(runtime_device)
        active_tensor = torch.tensor([active], dtype=torch.bool, device=runtime_device)
        homography_tensor = torch.from_numpy(
            np.ascontiguousarray(homography[None], dtype=np.float32)
        ).to(runtime_device)
        with torch.inference_mode():
            result = model(
                raw,
                normalized,
                support_tensor,
                sarn_active=active_tensor,
                raw_to_sarn_homography=homography_tensor,
            )
        value = float(result["mean"][0])
        _require(math.isfinite(value) and 0.0 <= value <= 1.0, "invalid prediction")
        return value

    return BenchmarkTarget(
        method=(
            f"unified_pointer_reader_{metadata['variant']}_seed_{metadata['seed']}"
        ),
        display_name=PUBLICATION_NAME,
        predict=predict,
        parameter_roots=(model,),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = run_benchmark(
        arm=ARM,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        condition=args.condition,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        device_name=args.device,
        limit=args.limit,
        warmup=args.warmup,
        target_factory=build_target,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(Path(args.output_json).resolve()),
                "method": report["method"],
                "latency": report["measurement"]["latency"],
                "parameters": report["parameters"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARM", "build_target"]
