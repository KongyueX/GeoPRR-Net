"""Check both lightweight readers with one forward and backward pass."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import timm
import torch

from experiments.geoprr_v2_model import BACKBONES, GeoPRRV2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable the instance GPU before this check")
    device = torch.device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(4)
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pretrained": args.pretrained,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "checks": [],
    }
    for backbone in BACKBONES:
        torch.manual_seed(20262020)
        model = GeoPRRV2(backbone, pretrained=args.pretrained).to(device).train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        output = model(images)
        distribution = output["distribution"]
        torch.testing.assert_close(distribution.sum(1), torch.ones(args.batch_size, device=device))
        loss = output["reading"].square().mean() + output["appearance_reading"].square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"{backbone}: non-finite or missing backward gradients")
        if not bool(torch.isfinite(output["reading"]).all()):
            raise RuntimeError(f"{backbone}: non-finite reading")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        config = timm.data.resolve_model_data_config(model.backbone)
        report["checks"].append({
            "backbone": backbone,
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "backbone_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
            "field_shape": list(output["field_logits"].shape),
            "normalization_mean": config["mean"],
            "normalization_std": config["std"],
            "forward_backward_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "finite_gradients": True,
        })
        del model, images, output, gradients, loss, distribution
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
