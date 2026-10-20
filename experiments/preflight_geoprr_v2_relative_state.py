"""Verify the declared reader on two training images without updating weights."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.geoprr_v2_relative_state import RelativeStateReader, relative_state_nll
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, distribution_loss, seed_everything
from experiments.train_geoprr_v2_relative_state import RelativeStateDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    home = args.home
    code = home / "code"
    config = json.loads((code / "experiments/configs/geoprr_v2_relative_state_joint.json").read_text())
    checkpoint = home / "runs/geoprr_v2_tail/iteration_v3_compositional/readers/supervised_s20262020/best.pt"
    data = RelativeStateDataset(home / "rendering/compositional_source_v1/manifests/train.jsonl",
                               home / "diagnostics/relative_state_labels_v1_20261021/train.jsonl",
                               data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=True,
                               pair_manifest=home / "rendering/compositional_source_v1/manifests/train_edges.jsonl")
    if len(data.rows) != 18590 or len(data) != 22686 or len(data.edges) != 8192:
        raise ValueError("preflight exposure contract differs")
    seed_everything(20262020)
    models = {}
    initialization = {}
    for coupling in ("joint", "independent"):
        seed_everything(20262020)
        model = RelativeStateReader(**{**config["model"], "coupling": coupling})
        initialization[coupling] = model.initialize_shared(checkpoint)
        if initialization[coupling]["source_epoch"] != 63:
            raise ValueError("initial checkpoint is not ordinary-supervised best63")
        models[coupling] = model
    for name, tensor in models["joint"].state_dict().items():
        if not torch.equal(tensor, models["independent"].state_dict()[name]):
            raise ValueError(f"control initialization differs: {name}")
    batch = device_batch(default_collate([data[0], data[1]]), torch.device("cuda"))
    if batch["pair_valid"].any():
        raise ValueError("preflight examples must be original training rows")
    results = {}
    for coupling in ("joint", "independent"):
        model = models.pop(coupling).cuda()
        model.set_shared_trainable(False)
        model.train()
        if model.backbone.training or any(p.requires_grad for p in model.shared_parameters()):
            raise ValueError("warmup does not freeze shared state")
        model.set_shared_trainable(True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(batch["image"])
            loss = relative_state_nll(output, batch["delta_radians"], batch["span_radians"]) / (2 * math.log(256))
            loss += distribution_loss(output["distribution"], batch["y"])
        if output["joint"].dtype != torch.float32 or output["distribution"].dtype != torch.float32:
            raise ValueError("probability computation lost FP32")
        for probability in (output["joint"].flatten(1), output["distribution"]):
            if not torch.isfinite(probability).all() or (probability < 0).any() or (probability.sum(1) - 1).abs().max() > 1e-5:
                raise ValueError("invalid readout probability")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
        if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
            raise ValueError("nonfinite backward gradient")
        for prefix in ("state_head.attention", "state_head.roles.0", "state_head.roles.1", "state_head.mixture", "backbone", "refine"):
            if sum(float(g.abs().sum()) for name, g in gradients.items() if name.startswith(prefix)) <= 0:
                raise ValueError(f"no learning signal in {prefix}")
        results[coupling] = {"parameters": sum(p.numel() for p in model.parameters()), "loss": float(loss.detach()),
                             "gradient_parameter_tensors": len(gradients), "forward_images": 2,
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "maximum_mass_error": float((output["distribution"].sum(1) - 1).abs().max())}
        del gradients, output, loss, scaler, optimizer, model
        torch.cuda.empty_cache()
    result = {"status": "passed", "seed": 20262020, "initialization_equal": True,
              "initialization": initialization, "model_input": "RGB only", "train_unique": len(data.rows),
              "donor_items": len(data), "recipient_images": len(data.edges), "arms": results,
              "unique_training_images_read": 2, "network_forward_images": 4, "optimizer_steps": 0,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
