"""Check exact best48 initialization and full-batch RGB triangle resources."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import default_collate

from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset, reference_relation_losses
from experiments.geoprr_v2_role_triangle import RoleTriangleReader, role_triangle_losses
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, seed_everything
from experiments.train_geoprr_v2_reference_relation import validation_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    started, home = time.perf_counter(), args.home
    output = home / "diagnostics/role_triangle_preflight_v1_20261021"
    if (output / "runtime_preflight.json").exists():
        raise FileExistsError("the triangle runtime preflight is already complete")
    if compute_processes():
        raise RuntimeError("GPU is occupied; do not overlap this preflight")
    repo = Path(__file__).resolve().parents[1]
    configs = {arm: json.loads((repo / f"experiments/configs/geoprr_v2_role_triangle_{arm}.json").read_text())
               for arm in ("full", "separable")}
    normalized = []
    for arm, config in configs.items():
        if config["model"]["association"] != arm or config["initialization"] != "full_best48":
            raise ValueError("triangle association or initialization differs")
        normalized.append({key: value for key, value in config.items() if key not in ("name", "model")}
                          | {"model": {key: value for key, value in config["model"].items() if key != "association"}})
    if normalized[0] != normalized[1]:
        raise ValueError("matched configurations differ beyond P-B association")
    initial = home / "runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt"
    checkpoint = torch.load(initial, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 48 or checkpoint["config"]["training"]["seed"] != 20262020:
        raise ValueError("requires the retained best48 seed20262020")
    config = configs["full"]
    if config["data"] != checkpoint["config"]["data"] or config["training"] != checkpoint["config"]["training"]:
        raise ValueError("data or training arrangements differ from best48")
    teacher_path = home / "diagnostics/reference_relation_preflight_v1_20261021/teacher_gate.json"
    if not json.loads(teacher_path.read_text())["numerical_gate_passed"]:
        raise ValueError("the existing unchanged role-label gate did not pass")
    validation = ReferenceRelationDataset(home / "runs/geoprr_v2/manifests/validation.jsonl",
        data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=False)
    by_id = {row["sample_id"]: index for index, row in enumerate(validation.rows)}
    samples = [validation[by_id[key]] for key in ("sync_2810", "sync_3663")]
    if any(sample["input_failure"] for sample in samples):
        raise ValueError("fixed RGB input failed")
    batch = device_batch(default_collate(samples), torch.device("cuda"))
    saved = {row["sample_id"]: row for row in read_predictions(initial.parent / "best_predictions.csv.gz")}
    report = {"status": "running", "pid": os.getpid(), "configs": configs,
              "code_revision": os.environ.get("GEOPRR_CODE_REVISION"), "optimizer_steps": 0,
              "unique_RGB_images_forwarded": 2, "CNN_image_forwards": 0, "backward_calls": 0,
              "teacher_numerical_gate_passed": True, "reused_teacher_gate": str(teacher_path),
              "max_effective_batch_size": 64, "same_initial_parameters": False, "stages": {},
              "scope": "new reader initialization and RGB resource preflight; no accuracy evaluation"}
    write_json(output / "runtime_progress.json", report)
    states, initialization = None, None
    try:
        for arm in ("full", "separable"):
            seed_everything(20262020)
            model = RoleTriangleReader(**configs[arm]["model"])
            initialization = model.initialize_full(initial)
            current = model.state_dict()
            if states is None:
                states = {key: value.clone() for key, value in current.items()}
            elif any(not torch.equal(value, states[key]) for key, value in current.items()):
                raise ValueError("matched initial tensors differ")
            if sum(parameter.numel() for parameter in model.parameters()) != 7461316:
                raise ValueError("unexpected triangle parameter count")
            model = model.cuda()
            for shared, size, stage in ((False, 2, "head_warmup"), (True, 64, "joint_finetune_maxbatch")):
                stage_started = time.perf_counter()
                model.zero_grad(set_to_none=True)
                model.set_shared_trainable(shared)
                model.train()
                repeat = size // 2
                supervision = {key: (value.repeat((repeat,) + (1,)*(value.ndim-1))
                                     if isinstance(value, torch.Tensor) else value*repeat)
                               for key, value in batch.items()}
                torch.cuda.reset_peak_memory_stats()
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model(supervision["image"])
                report["CNN_image_forwards"] += size
                terms = role_triangle_losses(result, supervision)
                if any(not torch.isfinite(value).all() for value in terms.values()):
                    raise FloatingPointError("triangle loss is nonfinite")
                mass_error = float((result["distribution"].sum(-1)-1).abs().max().detach())
                if mass_error > 1e-4 or not bool(torch.isfinite(result["distribution"]).all()):
                    raise ValueError("triangle readout probability is invalid")
                replay, loss_replay = None, None
                if not shared:
                    rows = validation_rows(result, batch)
                    if not all(row["valid"] for row in rows):
                        raise ValueError("validation CSV interface rejected triangle predictions")
                    replay = max(abs(row["prediction"]-float(saved[row["sample_id"]]["prediction"])) for row in rows)
                    if replay > 1e-5:
                        raise ValueError("zero W failed to reproduce best48")
                    star_losses = reference_relation_losses({**result, "log_A": result["star_log_A"]}, supervision)
                    loss_replay = max(abs(float((terms[key]-star_losses[key]).detach())) for key in terms)
                    if loss_replay > 1e-5:
                        raise ValueError("zero W failed to reproduce the star objective")
                    del star_losses
                forward_seconds = time.perf_counter()-stage_started
                terms["total"].backward()
                report["backward_calls"] += 1
                gradients = [value.grad for value in model.parameters() if value.requires_grad]
                if any(value is None or not bool(torch.isfinite(value).all()) for value in gradients):
                    raise FloatingPointError("triangle gradient is absent or nonfinite")
                if not shared and any(value.grad is not None for value in model.shared_parameters()):
                    raise ValueError("frozen shared module received gradients")
                if not shared and any(not torch.equal(value.cpu(), checkpoint["model"][key])
                    for key, value in model.state_dict().items() if key.startswith(("backbone.", "lateral.", "refine."))):
                    raise ValueError("frozen shared parameters or statistics changed")
                torch.cuda.synchronize()
                report["stages"][f"{arm}_{stage}"] = {
                    "effective_batch": size, "repeated_unique_images": 2,
                    "losses": {key: float(value.detach()) for key, value in terms.items()},
                    "probability_mass_max_error": mass_error, "best48_reading_replay_max": replay,
                    "star_objective_replay_max": loss_replay,
                    "forward_and_loss_seconds": forward_seconds,
                    "forward_backward_seconds": time.perf_counter()-stage_started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "gradients_finite": True}
                write_json(output / "runtime_progress.json", report)
                del gradients, result, terms, supervision
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            del model, current
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status="passed", same_initial_parameters=True, initialization=initialization,
                      shared_finetune_backward_passed=True, parameters_each_arm=7461316,
                      wall_seconds=time.perf_counter()-started,
                      memory_scope="RGB zerooptimizer forward/backward; Adam moments not allocated",
                      full1506_model_evaluation=False)
        write_json(output / "runtime_preflight.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", wall_seconds=time.perf_counter()-started)
        write_json(output / "runtime_progress.json", report)
        raise


if __name__ == "__main__":
    main()
