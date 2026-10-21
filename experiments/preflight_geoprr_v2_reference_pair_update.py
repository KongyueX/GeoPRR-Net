"""Run the bounded RGB interface/resource contract without optimizer updates."""
import time
PROCESS_STARTED = time.perf_counter()

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from torch.utils.data import default_collate

from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset
from experiments.geoprr_v2_reference_pair_reader import ReferencePairReader, reference_pair_losses
from experiments.geoprr_v2_role_triangle import role_triangle_losses
from experiments.run_geoprr_v2_optimization import write_json
from experiments.run_geoprr_v2_render_probe import compute_processes
from experiments.summarize_geoprr_v2_relative_state import read_predictions
from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, seed_everything
from experiments.train_geoprr_v2_reference_relation import validation_rows


FOCUS = ("sync_3172", "sync_9323")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    args = parser.parse_args()
    home = args.home
    output = home / "diagnostics/reference_pair_update_preflight_20261022_v1"
    if any((output / key).exists() for key in ("runtime_progress.json", "runtime_preflight.json")):
        raise FileExistsError("resource artifacts exist; preserve them and inspect the actual job")
    if compute_processes():
        raise RuntimeError("GPU is occupied; do not overlap the resource check")
    initial = home / "runs/geoprr_v2_tail/iteration_v7_role_triangle/readers/full_s20262020/best.pt"
    checkpoint = torch.load(initial, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    manifest = home / "runs/geoprr_v2/manifests/validation.jsonl"
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = [next(row for row in manifest_rows if row["sample_id"] == key) for key in FOCUS]
    dataset = ReferenceRelationDataset(selected, data_root=DEFAULT_DATA_ROOT, data_config=config["data"], augment=False)
    dataset.base.manifest_root = manifest.parent
    expected = {row["sample_id"]: row for row in checkpoint["data_contract"]["validation"]["rows"] if row["sample_id"] in FOCUS}
    if dataset.supervision_contract()["rows"] != [expected[key] for key in FOCUS]:
        raise ValueError("the two original RGB supervision records differ")
    samples = [dataset[index] for index in range(2)]
    if any(sample["input_failure"] for sample in samples):
        raise ValueError("fixed RGB input failed")
    batch = device_batch(default_collate(samples), torch.device("cuda"))
    saved = {row["sample_id"]: row for row in read_predictions(initial.parent / "best_predictions.csv.gz") if row["sample_id"] in FOCUS}
    report = {"status": "running", "pid": os.getpid(), "code_revision": os.environ.get("GEOPRR_CODE_REVISION"),
              "scope": "new RGB interface and repeated-image resource fixture; not model accuracy or training",
              "initial_epoch": 64, "model_seed": 20262020, "focus_ids": list(FOCUS),
              "unique_RGB_images": 2, "CNN_image_attempts": 0, "CNN_image_forwards": 0, "backbone_calls": 0,
              "native_state_head_image_calls": 0, "new_verifier_image_calls": 0,
              "new_update_calls": 0, "readout_module_calls": 0, "readout_image_equivalents": 0,
              "backward_calls": 0, "backward_completed": 0, "optimizer_updates": 0, "same_initial_tensors": False, "stages": {}}
    write_json(output / "runtime_progress.json", report)
    (output / "preflight.pid").write_text(str(os.getpid()) + "\n")

    def count_backbone(_module, inputs, _result):
        report["backbone_calls"] += 1
        report["CNN_image_forwards"] += len(inputs[0])

    def count_head(_module, inputs, _result):
        report["native_state_head_image_calls"] += len(inputs[0])

    def count_verifier(_module, inputs, _result):
        report["new_verifier_image_calls"] += len(inputs[0])
        report["new_update_calls"] += 1

    def count_readout(_module, inputs, _result):
        report["readout_module_calls"] += 1
        report["readout_image_equivalents"] += len(inputs[0])

    states = None
    try:
        for mode in ("pair_preserving", "global"):
            seed_everything(20262020)
            model_args = {key: value for key, value in config["model"].items() if key != "association"}
            model = ReferencePairReader(**model_args, reference_update=mode)
            initialization = model.initialize_full(initial)
            current = model.state_dict()
            if states is None:
                states = {key: value.clone() for key, value in current.items()}
            elif set(current) != set(states) or any(not torch.equal(value, states[key]) for key, value in current.items()):
                raise ValueError("matched initial tensors differ")
            if len(current) != 500 or sum(parameter.numel() for parameter in model.parameters()) != 7465988:
                raise ValueError("new reader state or parameter count differs")
            model = model.cuda()
            model.backbone.register_forward_hook(count_backbone)
            model.state_head.register_forward_hook(count_head)
            model.state_head.reference_verifier.register_forward_hook(count_verifier)
            model.readout.register_forward_hook(count_readout)
            for shared, size, stage in ((False, 2, "frozen_v0_RGB"), (True, 64, "shared_nonzero_probe_maxbatch")):
                model.zero_grad(set_to_none=True)
                model.set_shared_trainable(shared)
                model.train()
                if shared:
                    with torch.no_grad():
                        probe = (torch.arange(32, device="cuda", dtype=torch.float32) - 15.5) * (1e-3 / 32)
                        model.state_head.reference_verifier.last.weight.copy_(probe[None])
                repeat = size // 2
                supervision = {key: (value.repeat((repeat,) + (1,) * (value.ndim - 1))
                                    if isinstance(value, torch.Tensor) else value * repeat) for key, value in batch.items()}
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                stage_started = time.perf_counter()
                report["CNN_image_attempts"] += size
                with torch.autocast("cuda", dtype=torch.float16):
                    result = model(supervision["image"])
                terms = reference_pair_losses(result, supervision)
                if any(not bool(torch.isfinite(value).all()) for value in terms.values()):
                    raise FloatingPointError("new reader loss is nonfinite")
                mass_error = float((result["distribution"].sum(-1) - 1).abs().max().detach())
                if mass_error > 1e-4 or not bool(torch.isfinite(result["distribution"]).all()):
                    raise ValueError("new reader probability mass is invalid")
                replay = loss_replay = None
                evidence_max = float(result["reference_evidence"].abs().max().detach())
                if not shared:
                    rows = validation_rows(result, supervision)
                    if not all(row["valid"] and row["input_valid"] for row in rows):
                        raise ValueError("validation CSV interface rejected the new reader")
                    for row in rows:
                        old = saved[row["sample_id"]]
                        if (row["group_id"], row["target"]) != (old["group_id"], float(old["target"])):
                            raise ValueError("original RGB group or target differs")
                    replay = max(abs(row["prediction"] - float(saved[row["sample_id"]]["prediction"])) for row in rows)
                    base_terms = role_triangle_losses(result, supervision)
                    loss_replay = max(abs(float((terms[key] - base_terms[key]).detach())) for key in base_terms)
                    del base_terms
                    if replay > 1e-5 or loss_replay > 1e-5 or evidence_max != 0:
                        raise ValueError("zero evidence failed the best64 or original objective replay")
                elif evidence_max == 0:
                    raise ValueError("fixed resource probe failed to exercise nonzero reference evidence")
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - stage_started
                report["backward_calls"] += 1
                terms["total"].backward()
                report["backward_completed"] += 1
                gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
                if any(gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise FloatingPointError("new reader gradient is absent or nonfinite")
                verifier_grad = model.state_head.reference_verifier.last.weight.grad
                if not shared and float(verifier_grad.abs().max()) == 0:
                    raise ValueError("zero output verifier has no first-step gradient")
                if not shared and any(parameter.grad is not None for parameter in model.shared_parameters()):
                    raise ValueError("frozen shared modules received gradients")
                if not shared and any(not torch.equal(value.cpu(), checkpoint["model"][key]) for key, value in model.state_dict().items()
                    if key.startswith(("backbone.", "lateral.", "refine."))):
                    raise ValueError("frozen shared weights or statistics changed")
                torch.cuda.synchronize()
                report["stages"][f"{mode}_{stage}"] = {
                    "effective_batch": size, "repeated_unique_RGB_images": 2,
                    "losses": {key: float(value.detach()) for key, value in terms.items()},
                    "evidence_abs_max": evidence_max, "probability_mass_max_error": mass_error,
                    "best64_reading_replay_max": replay, "original_objective_replay_max": loss_replay,
                    "new_last_layer_gradient_abs_max": float(verifier_grad.abs().max()),
                    "forward_and_loss_seconds": forward_seconds, "forward_backward_seconds": time.perf_counter() - stage_started,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "gradients_finite": True,
                    "new_last_layer_probe": "(index-15.5)*1e-3/32, ephemeral, not saved" if shared else "zero initialization"}
                write_json(output / "runtime_progress.json", report)
                del gradients, verifier_grad, result, terms, supervision
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            del model, current
            gc.collect()
            torch.cuda.empty_cache()
        if (report["CNN_image_forwards"], report["backbone_calls"], report["backward_calls"]) != (132, 4, 4):
            raise ValueError("actual RGB/backward counts differ from the bounded contract")
        report.update(status="passed", same_initial_tensors=True, initialization=initialization,
                      parameters_each_arm=7465988, states_each_arm=500, max_effective_batch=64,
                      process_wall_seconds=time.perf_counter() - PROCESS_STARTED,
                      timing_scope="from module imports; includes data, initialization and all GPU stages",
                      memory_scope="zero optimizer; Adam states not allocated", full1506_model_evaluation=False)
        write_json(output / "runtime_preflight.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      process_wall_seconds=time.perf_counter() - PROCESS_STARTED)
        write_json(output / "runtime_progress.json", report)
        raise


if __name__ == "__main__":
    main()
