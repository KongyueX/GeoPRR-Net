"""Evaluate fixed half interpolation of task-trained anchors and endpoint readers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROCESS_STARTED = time.perf_counter()
PROTOCOL = "geoprr_v2_task_anchor_half_interpolation_v1"
QUEUE_PROTOCOL = "geoprr_v2_task_anchor_half_frozen_evaluation_v1"
ENDPOINT_PROTOCOL = "geoprr_v2_endpoint_generalization_source_only_v1"
ARMS = ("feedback_half", "local_half")
EXPECTED_MODEL = {"bins": 256, "decoder_dim": 64, "token_dim": 16, "coupling": "joint"}


def load(path):
    return json.loads(path.read_text())


def components(home, arm):
    feedback = arm == "feedback_half"
    anchor = home / ("runs/geoprr_v2_tail/iteration_v6_reference_feedback/readers/feedback_s20262020/best.pt"
                     if feedback else "runs/geoprr_v2_tail/iteration_v5_reference_relation/readers/joint_s20262020/best.pt")
    endpoint = home / ("runs/geoprr_generalization/iteration_v11_endpoints_20261023/matched_training/readers/endpoint/best.pt"
                       if feedback else "runs/geoprr_generalization/iteration_v12_local_endpoints_20261023/matched_training/readers/local_endpoint/best.pt")
    return anchor, endpoint


def half_interpolated_state(model, anchor, endpoint):
    """Average learned parameters and BN moments; retain anchor constructor state."""
    import torch
    from torch import nn

    template = model.state_dict()
    if set(anchor) != set(template) or set(endpoint) != set(template):
        raise ValueError("component state keys differ from the native model")
    parameters = set(dict(model.named_parameters()))
    bn_moments, bn_counters = set(), set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for key, _ in module.named_buffers(recurse=False):
                full = f"{name}.{key}"
                if key in ("running_mean", "running_var"):
                    bn_moments.add(full)
                elif key == "num_batches_tracked":
                    bn_counters.add(full)
    result = {}
    for key, reference in template.items():
        a, b = anchor[key], endpoint[key]
        if a.shape != reference.shape or b.shape != reference.shape or a.dtype != reference.dtype or b.dtype != reference.dtype:
            raise ValueError(f"component shape or dtype differs: {key}")
        if torch.is_floating_point(reference) and (not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
            raise ValueError(f"nonfinite component state: {key}")
        if key in parameters or key in bn_moments:
            if not torch.is_floating_point(reference):
                raise ValueError(f"interpolated state must be floating point: {key}")
            if key.endswith(".running_var") and (bool((a < 0).any()) or bool((b < 0).any())):
                raise ValueError(f"negative BN variance: {key}")
            result[key] = a * .5 + b * .5
            if not bool(torch.isfinite(result[key]).all()):
                raise ValueError(f"nonfinite interpolated state: {key}")
        elif key in bn_counters:
            if torch.is_floating_point(reference) or bool((a < 0).any()) or bool((b < 0).any()):
                raise ValueError(f"invalid BN counter: {key}")
            result[key] = a.clone()
        else:
            if not torch.equal(a, b) or not torch.equal(a, reference):
                raise ValueError(f"constructor-fixed buffer differs: {key}")
            result[key] = a.clone()
    return result, {"learned_parameter_tensors_averaged": len(parameters),
                    "BN_running_moment_tensors_averaged": len(bn_moments),
                    "BN_counter_tensors_copied_from_anchor": len(bn_counters),
                    "constructor_fixed_tensors_copied_from_anchor": len(template) - len(parameters) - len(bn_moments) - len(bn_counters)}


def prepare_checkpoint(home, arm):
    import torch
    from experiments.evaluate_geoprr_v2_generalization import reader_from_checkpoint
    from experiments.run_geoprr_v2_optimization import write_json

    started = time.perf_counter()
    anchor_path, endpoint_path = components(home, arm)
    anchor, endpoint = [torch.load(path, map_location="cpu", weights_only=False)
                        for path in (anchor_path, endpoint_path)]
    feedback = arm == "feedback_half"
    native = ("geoprr_v2_reference_feedback_source_only_v1" if feedback
              else "geoprr_v2_reference_relation_source_only_v1")
    expected_model = {**EXPECTED_MODEL, **({"reference_normalization": "global"} if feedback else {})}
    for checkpoint in (anchor, endpoint):
        config = checkpoint["config"]
        if (config.get("protocol", checkpoint["protocol"]) != checkpoint["protocol"]
                or config["model"] != expected_model
                or config["data"] != {"image_size": 768, "output_size": 192}
                or config["training"]["seed"] != 20262020 or config["training"]["amp"] is not True):
            raise ValueError("component architecture, source configuration or seed differs")
    expected_policy = {"freeze_bn_statistics": False, "mixstyle_module_names": [],
                       "mixstyle_p": .5, "mixstyle_alpha": .1,
                       "source_replacement": {"groups": 128, "states": [0, .005, .02, 1],
                                              "protocol": "geoprr_v2_endpoint_source_v11"}}
    if (anchor["protocol"] != native or anchor["epoch"] != (48 if feedback else 57)
            or endpoint["protocol"] != ENDPOINT_PROTOCOL or endpoint["config"]["reader_protocol"] != native
            or endpoint["config"]["generalization"] != expected_policy or endpoint["config"]["training"]["epochs"] != 4
            or endpoint["epoch"] != 1 or endpoint["best_epoch"] != 1
            or not isinstance(endpoint["best_nmae"], (int, float))
            or not bool(torch.isfinite(torch.tensor(endpoint["best_nmae"])))):
        raise ValueError("requires the frozen native anchor and endpoint source-best epoch1")
    model, metadata = reader_from_checkpoint(anchor)
    if metadata["parameters"] != 7461060 or metadata["state_tensors"] != 496:
        raise ValueError("native model size differs")
    state, state_counts = half_interpolated_state(model, anchor["model"], endpoint["model"])
    model.load_state_dict(state, strict=True)
    transformation = {
        "type": "fixed_half_task_anchor_weight_interpolation", "alpha": .5,
        "anchor_path": str(anchor_path.resolve()), "endpoint_path": str(endpoint_path.resolve()),
        "component_epochs": {"task_anchor": anchor["epoch"], "endpoint_source_best": endpoint["epoch"]},
        "endpoint_selection": "existing source1506 NMAE best; fixed epoch1, no target checkpoint search",
        "state_policy": {"learned_float_parameters": "half anchor plus half endpoint, including CNN, BN affine, decoder and head",
                         "BN_running_mean_and_variance": "half anchor plus half endpoint",
                         "BN_num_batches_tracked": "copy task anchor; evaluation only",
                         "constructor_fixed_buffers": "require exact component and native-template equality; copy anchor"},
        "state_tensor_counts": state_counts, "optimizer_updates": 0, "backward_calls": 0,
        "BN_recalibration": False, "new_training": False,
        "reference": "https://arxiv.org/abs/2109.01903v3",
        "reference_scope": "WiSE-FT motivates weight interpolation; this anchor is task-trained, not a zero-shot model",
    }
    checkpoint = {"protocol": PROTOCOL,
        "config": {"protocol": PROTOCOL, "reader_protocol": native, "model": expected_model,
                   "data": anchor["config"]["data"], "training": {"seed": 20262020, "amp": True}},
        "model": state, "epoch": None, "best_epoch": None, "best_nmae": None,
        "posthoc_interpolation": transformation}
    directory = home / "runs/geoprr_generalization/iteration_v14_anchor_half_20261023/frozen_evaluation/checkpoints" / arm
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, directory / "half.pt")
    write_json(directory / "metadata.json", {"protocol": PROTOCOL, "arm": arm,
        "checkpoint_transformation": transformation, "model_metadata": metadata,
        "parameters": 7461060, "state_tensors": 496,
        "cnn_image_forwards": 0, "backward_calls": 0, "optimizer_updates": 0,
        "preparation_seconds": time.perf_counter() - started,
        "full_process_seconds": time.perf_counter() - PROCESS_STARTED})


def evaluate_source(home, checkpoint_path, destination):
    import torch
    from experiments.evaluate_geoprr_v2_generalization import reader_from_checkpoint
    from experiments.geoprr_v2_reference_relation import ReferenceRelationDataset
    from experiments.run_geoprr_v2_optimization import write_json
    from experiments.train_geoprr_v2 import DEFAULT_DATA_ROOT, device_batch, make_loader
    from experiments.train_geoprr_v2_reference_relation import save_predictions, validation_rows, validation_summary

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"source evaluation output is not empty: {destination}")
    started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["protocol"] != PROTOCOL or any(checkpoint[key] is not None for key in ("epoch", "best_epoch", "best_nmae")):
        raise ValueError("requires the fixed posthoc checkpoint without a new training identity")
    model, metadata = reader_from_checkpoint(checkpoint)
    device = torch.device("cuda")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    validation = ReferenceRelationDataset(home / "runs/geoprr_v2/manifests/validation.jsonl",
        data_root=DEFAULT_DATA_ROOT, data_config=checkpoint["config"]["data"], augment=False)
    if len(validation) != 1506:
        raise ValueError("source validation population differs")
    loader = make_loader(validation, batch_size=16, workers=4, shuffle=False, seed=20262020, device=device)
    counts = {"model_forward_calls": 0, "cnn_forward_calls": 0, "cnn_image_forwards": 0}

    def count_backbone(_module, inputs, _output):
        counts["cnn_forward_calls"] += 1
        counts["cnn_image_forwards"] += len(inputs[0])

    hook = model.backbone.register_forward_hook(count_backbone)
    rows, forward_seconds = [], 0.
    torch.cuda.synchronize(device)
    pipeline_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for raw_batch in loader:
                batch = device_batch(raw_batch, device)
                torch.cuda.synchronize(device)
                forward_started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    prediction = model(batch["image"])
                torch.cuda.synchronize(device)
                forward_seconds += time.perf_counter() - forward_started
                counts["model_forward_calls"] += 1
                rows.extend(validation_rows(prediction, batch))
    finally:
        hook.remove()
    pipeline_seconds = time.perf_counter() - pipeline_started
    if ([row["sample_id"] for row in rows] != [row["sample_id"] for row in validation.rows]
            or counts != {"model_forward_calls": 95, "cnn_forward_calls": 95, "cnn_image_forwards": 1506}):
        raise RuntimeError("source population or actual CNN exposure differs")
    destination.mkdir(parents=True, exist_ok=True)
    save_predictions(destination / "predictions.csv.gz", rows)
    write_json(destination / "summary.json", {"protocol": PROTOCOL, "status": "complete",
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": None, "checkpoint_best_epoch": None,
        "checkpoint_source_best_nmae": None, "checkpoint_transformation": checkpoint["posthoc_interpolation"],
        "model_metadata": metadata, "config": checkpoint["config"], "declared_samples": 1506,
        "source_model_inputs": "original validation ROI RGB only; geometry and scalar labels used for scoring",
        "summary": validation_summary(rows), "forward_counts": counts,
        "backward_calls": 0, "optimizer_updates": 0, "training_or_adaptation_during_evaluation": False,
        "failure_policy": "retain all1506; original validation_rows assigns normalized error1 to input/invalid prediction failures",
        "timing": {"batch_size": 16, "workers": 4, "backbone_amp": True,
                   "pipeline_seconds": pipeline_seconds, "forward_seconds": forward_seconds,
                   "evaluation_seconds": time.perf_counter() - started,
                   "full_process_seconds": time.perf_counter() - PROCESS_STARTED},
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "torch": torch.__version__, "cuda": torch.version.cuda})


def run(home):
    from experiments.run_geoprr_v2_optimization import OptimizationRun, process_running, write_json
    from experiments.run_geoprr_v2_render_probe import compute_processes

    repo = Path(__file__).resolve().parents[1]
    output = home / "runs/geoprr_generalization/iteration_v14_anchor_half_20261023/frozen_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ("iteration_v11_endpoints_20261023", "iteration_v12_local_endpoints_20261023",
                     "iteration_v13_cnn_weight_freeze_20261023"):
            state = load(home / f"runs/geoprr_generalization/{name}/matched_training/queue_state.json")
            if state["status"] != "complete" or process_running(state["pipeline_pid"]):
                raise RuntimeError("prior endpoint or CNN-freeze stage is not complete and idle")
        if compute_processes():
            raise RuntimeError("GPU has an active compute process")
        revision = os.environ["GEOPRR_GENERALIZATION_REVISION"]
        protocol = {"protocol": QUEUE_PROTOCOL, "code_revision": revision, "arms": list(ARMS), "alpha": .5,
            "components": {arm: [str(path) for path in components(home, arm)] for arm in ARMS},
            "checkpoint_protocol": PROTOCOL, "new_training": False, "optimizer_updates": 0,
            "selection": "two fixed source-selected epoch1 endpoints; no alpha or target checkpoint search",
            "populations": {"source": 1506, "industrial": 1395, "rf100": 151},
            "expected_total_CNN_images": 6104, "expected_total_CNN_calls": 386,
            "external_scope": "previously viewed development; original ROI pipeline; no target supervision",
            "reference": "https://arxiv.org/abs/2109.01903v3",
            "reference_scope": "task-trained anchor interpolation; not preservation of a zero-shot foundation model"}
        path = output / "protocol.json"
        if path.exists() and load(path) != protocol:
            raise ValueError("stored interpolation protocol differs")
        write_json(path, protocol)
        snapshot = output / "code_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("run_geoprr_v2_anchor_interpolation.py", "evaluate_geoprr_v2_generalization.py",
                         "geoprr_v2_reference_relation.py", "geoprr_v2_data.py", "train_geoprr_v2_reference_relation.py"):
                shutil.copy2(repo / "experiments" / name, snapshot / name)
        queue = OptimizationRun(home, output, [20262020])
        queue.state.update(protocol=QUEUE_PROTOCOL, code_revision=revision)
        (output / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        queue.save()
        summaries = {}
        try:
            for arm in ARMS:
                directory = output / "checkpoints" / arm
                queue.state["current_step"] = f"prepare_{arm}"
                queue.save()
                queue.step(f"prepare_{arm}", [sys.executable, "-m", "experiments.run_geoprr_v2_anchor_interpolation",
                           "--home", str(home), "--prepare-arm", arm], directory / "metadata.json",
                           environment_overrides={"CUDA_VISIBLE_DEVICES": ""})
                summaries[arm] = {"checkpoint": load(directory / "metadata.json"), "evaluations": {}}
                for dataset in ("source", "industrial", "rf100"):
                    if compute_processes():
                        raise RuntimeError("GPU became busy before frozen evaluation")
                    step = f"eval_{arm}_{dataset}"
                    destination = output / "evaluations" / step
                    command = [sys.executable, "-m", "experiments.run_geoprr_v2_anchor_interpolation",
                               "--home", str(home), "--source-eval", "--checkpoint", str(directory / "half.pt"),
                               "--output", str(destination)] if dataset == "source" else [
                                   sys.executable, "-m", "experiments.evaluate_geoprr_v2_generalization",
                                   "--checkpoint", str(directory / "half.pt"), "--dataset-root",
                                   str(home / "datasets" / ("Industrial-1395" if dataset == "industrial" else "RF100-VL")),
                                   "--dataset-name", dataset, "--output", str(destination),
                                   "--batch-size", "16", "--workers", "4", "--device", "cuda"]
                    queue.state["current_step"] = step
                    queue.save()
                    queue.step(step, command, destination / "summary.json")
                    summaries[arm]["evaluations"][dataset] = load(destination / "summary.json")
                write_json(output / "screen_summary.json", {"scope": "partial fixed-half frozen evaluation", "arms": summaries})
            counts = [value["forward_counts"] for arm in summaries.values() for value in arm["evaluations"].values()]
            total_images = sum(value["cnn_image_forwards"] for value in counts)
            total_calls = sum(value["cnn_forward_calls"] for value in counts)
            if (total_images, total_calls) != (6104, 386):
                raise RuntimeError("completed frozen evaluation CNN counts differ")
            write_json(output / "screen_summary.json", {"scope": "complete fixed-half viewed development evaluation; no automatic promotion",
                "arms": summaries, "CNN_image_forwards": total_images, "CNN_forward_calls": total_calls,
                "backward_calls": 0, "optimizer_updates": 0})
            queue.state.update(status="complete", current_step=None)
            queue.save()
        except Exception as error:
            queue.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            queue.save()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--prepare-arm", choices=ARMS)
    parser.add_argument("--source-eval", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_arm:
        prepare_checkpoint(args.home, args.prepare_arm)
    elif args.source_eval:
        if args.checkpoint is None or args.output is None:
            parser.error("source evaluation requires its checkpoint and output")
        evaluate_source(args.home, args.checkpoint, args.output)
    else:
        run(args.home)


if __name__ == "__main__":
    main()
