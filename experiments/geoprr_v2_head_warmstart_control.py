"""One bounded source head-warmstart control with a fixed five-epoch joint stage."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.geoprr_v2_needle_holdout_control import (
    CONFIG as POPULATION_CONFIG, PREPARED_RELATIVE, PROTOCOL as PREPARATION_PROTOCOL,
    SEEN, _save_evaluation, conditional_ood, learning_gate, source_populations,
    validate_prepared, write_json,
)


PROTOCOL = "geoprr_v2_head_warmstart_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_head_warmstart_control_20261025"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_source_head_warmstart_control_preflight_20261025/result.json"
CONFIG = deepcopy(POPULATION_CONFIG)
CONFIG.update(protocol=PROTOCOL, preparation_protocol=PREPARATION_PROTOCOL,
    kind="one bounded source procedure control; not formal V32 or LP-FT paper replication")
CONFIG["LP"] = {
    "cache_images":5496, "cache_CNN_calls":86, "cache_shuffle":False, "loader_seed":20262020,
    "encoder_requires_grad":False, "encoder_mode":"eval including BN affine and buffers",
    "semantic_dim":1280, "cache_dtype":"float16", "cache_device_memory_only":True,
    "cache_saved_or_exported":False, "optimizer_attempts":100, "full_batch_features":5496,
    "point_calls":100, "point_feature_presentations":549600, "point_backward_calls":100,
    "CNN_backward_calls":0, "optimizer":"fresh point-only AdamW", "learning_rate":.0003,
    "weight_decay":.0001, "gradient_clip":5., "smooth_l1_beta":.05,
    "fresh_default_GradScaler":True, "losses":"before each of exactly100 updates; no extra forward",
    "budget_qualification":"fixed resource cap, not paper default or evidence of convergence",
}
CONFIG["FT_transition"] = {
    "head":"retain the same LP-trained point parameters", "encoder":"all parameters trainable; normal source BN",
    "RNG":"restore only stochastic RNG saved immediately after fresh initialization",
    "optimizer":"discard LP optimizer and scaler; fresh full AdamW and default GradScaler",
}
CONFIG["primary_comparison"] = {
    "baseline":"completed directFT5 on the same292 source-ID rows",
    "baseline_mean":.06132800787729562, "baseline_tail":.25669996334860723,
    "retain":"original gate passes and ID mean <= baseline mean and ID tail < baseline tail",
    "automatic_full_training":False,
}
CONFIG["checkpoint"] = "initial RNG/state, full LP state, full state after each FT epoch; lastFT5 only evaluation"
PREFLIGHT_COUNTS = {"CNN_calls":3, "CNN_images":8, "RGB_unique_images":8,
    "point_calls":3, "point_examples":8, "point_backward_calls":2,
    "CNN_backward_calls":1, "optimizer_updates":0}


def lp_mode(model):
    model.eval()
    model.encoder.requires_grad_(False)
    model.point.requires_grad_(True).train()


def enter_finetuning(model, initial_rng, restore):
    """Keep learned parameters and restore the initialization's stochastic boundary."""
    model.zero_grad(set_to_none=True)
    model.requires_grad_(True).train()
    restore(initial_rng)


def detached_semantic(model, image):
    return model.encoder(image).detach()


def point_prediction(model, semantic):
    return model.point(semantic).squeeze(-1).float().sigmoid()


def new_optimizer(model, *, point_only):
    import torch
    parameters = model.point.parameters() if point_only else model.parameters()
    return torch.optim.AdamW(parameters, lr=.0003, weight_decay=.0001)


def state_equal(module, saved):
    import torch
    current = module.state_dict()
    return set(current) == set(saved) and all(torch.equal(current[key], value) for key,value in saved.items())


def initial_totals():
    keys = ("cache_cnn_image_forwards", "cache_cnn_forward_calls", "cache_point_forward_calls",
        "cache_CNN_backward_calls", "lp_point_forward_calls", "lp_point_image_forwards",
        "lp_point_backward_calls", "lp_CNN_backward_calls", "lp_optimizer_updates", "lp_amp_skipped_updates",
        "train_batches", "train_cnn_image_forwards", "train_cnn_forward_calls", "backward_calls",
        "optimizer_updates", "amp_skipped_updates", "train_point_forward_calls", "train_point_image_forwards",
        "validation_images", "validation_cnn_forward_calls", "validation_point_forward_calls",
        "validation_point_image_forwards")
    result = dict.fromkeys(keys, 0)
    result.update(cache_seconds=0., lp_seconds=0., training_seconds=0., validation_seconds=0.)
    return result


def count_summary(totals):
    return {
        "cnn_image_forwards":totals["cache_cnn_image_forwards"]+totals["train_cnn_image_forwards"]+totals["validation_images"],
        "cnn_forward_calls":totals["cache_cnn_forward_calls"]+totals["train_cnn_forward_calls"]+totals["validation_cnn_forward_calls"],
        "point_image_forwards":totals["lp_point_image_forwards"]+totals["train_point_image_forwards"]+totals["validation_point_image_forwards"],
        "point_forward_calls":totals["lp_point_forward_calls"]+totals["train_point_forward_calls"]+totals["validation_point_forward_calls"],
        "source_backward_calls":totals["lp_point_backward_calls"]+totals["backward_calls"],
        "CNN_backward_calls":totals["backward_calls"], "point_only_backward_calls":totals["lp_point_backward_calls"],
        "LP_optimizer_updates":totals["lp_optimizer_updates"], "LP_amp_skipped_updates":totals["lp_amp_skipped_updates"],
        "FT_optimizer_updates":totals["optimizer_updates"], "FT_amp_skipped_updates":totals["amp_skipped_updates"],
        "external_images":0, "external_CNN_calls":0, "external_point_calls":0,
        "external_backward_calls":0, "external_optimizer_updates":0,
    }


def validate_totals(totals, gate_passed):
    expected = {"cache_cnn_image_forwards":5496, "cache_cnn_forward_calls":86, "cache_point_forward_calls":0,
        "cache_CNN_backward_calls":0, "lp_point_forward_calls":100, "lp_point_image_forwards":549600,
        "lp_point_backward_calls":100, "lp_CNN_backward_calls":0, "train_batches":430,
        "train_cnn_image_forwards":27480, "train_cnn_forward_calls":430, "backward_calls":430,
        "train_point_forward_calls":430, "train_point_image_forwards":27480,
        "validation_images":512 if gate_passed else 292,
        "validation_cnn_forward_calls":9 if gate_passed else 5,
        "validation_point_forward_calls":9 if gate_passed else 5,
        "validation_point_image_forwards":512 if gate_passed else 292}
    if (any(totals[key] != value for key,value in expected.items())
            or totals["lp_optimizer_updates"]+totals["lp_amp_skipped_updates"] != 100
            or totals["optimizer_updates"]+totals["amp_skipped_updates"] != 430):
        raise ValueError("the bounded LP/FT control differs from its separate CNN and point budgets")
    summary = count_summary(totals)
    if (summary["cnn_image_forwards"], summary["cnn_forward_calls"], summary["point_image_forwards"],
            summary["point_forward_calls"], summary["source_backward_calls"]) != (
                (33488,525,577592,539,530) if gate_passed else (33268,521,577372,535,530)):
        raise ValueError("the complete source control budget differs")


def primary_comparison(gate):
    baseline = CONFIG["primary_comparison"]
    mean, tail = gate["model"]["nmae"], gate["model"]["tail10_nmae"]
    return {"baseline":baseline["baseline"], "baseline_mean":baseline["baseline_mean"],
        "baseline_tail":baseline["baseline_tail"], "ID_mean":mean, "ID_tail":tail,
        "original_learning_gate_passed":gate["passed"],
        "retain_this_warmstart_candidate":bool(gate["passed"] and mean <= baseline["baseline_mean"]
                                               and tail < baseline["baseline_tail"]),
        "automatic_full_training":False, "extra_LP_compute_included":True}


def phase_cost(device, started):
    import torch
    torch.cuda.synchronize(device)
    return {"seconds":time.perf_counter()-started,
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(device),
        "individual_point_forward_seconds":None}


def runtime(device, started, phases):
    import torch
    peaks = list(phases.values())
    return {"process_seconds":time.perf_counter()-started,
        "full_process_seconds":time.perf_counter()-PROCESS_STARTED,
        "full_process_boundary":"this module entry before any torch import through saved result artifacts; excludes final completion JSON",
        "device":str(device), "gpu":torch.cuda.get_device_name(device), "torch":torch.__version__,
        "cuda":torch.version.cuda, "peak_cuda_memory_bytes":max(
            [torch.cuda.max_memory_allocated(device)]+[phase["peak_cuda_memory_bytes"] for phase in peaks]),
        "peak_cuda_reserved_bytes":max(
            [torch.cuda.max_memory_reserved(device)]+[phase["peak_cuda_reserved_bytes"] for phase in peaks]),
        "phases":phases, "supervisor_wall_record":"runner screen_summary.json"}


def preflight_indices(prepared):
    rows = prepared["source_contract"]["populations"]["train"]
    lp, ft = [], []
    for alias in SEEN:
        parent = min(row["metadata"]["needle_control"]["layout"] for row in rows
                     if row["metadata"]["needle_control"]["view"] == "original"
                     and row["metadata"]["needle_control"]["pointer"] == alias)
        for view,slot,destination in (("original",0,lp),("original",3,ft),("roll",3,ft)):
            matches = [i for i,row in enumerate(rows) if row["metadata"]["needle_control"]["view"] == view
                and row["metadata"]["needle_control"]["slot"] == slot
                and row["metadata"]["needle_control"]["pointer"] == alias
                and row["metadata"]["needle_control"]["layout" if view == "original" else "parent_layout"] == parent]
            if len(matches) != 1:
                raise ValueError("the fixed LP/FT preflight source selection is ambiguous")
            destination.extend(matches)
    indices = {"lp":lp, "ft":ft, "id":[subset[0] for subset in prepared["pref_indices"]["id"]]}
    selected = [rows[i] for i in lp+ft]+[
        prepared["source_contract"]["populations"]["id"][i] for i in indices["id"]]
    if (len({row["sample_id"] for row in selected}) != 8
            or len({row.get("roi_path",row.get("image_path")) for row in selected}) != 8):
        raise ValueError("preflight requires eight distinct existing source RGBs")
    return indices


def preflight(home, output):
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_data import _resolve_image, crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import (
        create_model, encoder_counter, gradient_evidence, model_metadata, point_counter,
    )
    from experiments.train_geoprr_v2 import device_batch, seed_everything

    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("the new preflight already has complete or partial outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "started.json", {"protocol":PROTOCOL, "status":"started"})
    prepared = validate_prepared(home)
    started, device = time.perf_counter(), torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training, source_id = source_populations(home, prepared)
    indices = preflight_indices(prepared)
    batches = {name:device_batch(default_collate([training[i] for i in indices[name]]),device)
               for name in ("lp","ft")}
    samples = [source_id[i] for i in indices["id"]]
    for i,sample in zip(indices["id"],samples,strict=True):
        row = source_id.rows[i]
        image = cv2.imread(str(_resolve_image(row,source_id.base.manifest_root,source_id.base.dataset_root)))
        roi,_ = crop_and_resize(image,None if "roi_path" in row else row.get("metadata",{}).get("dial_bbox"),512)
        rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        expected = torch.from_numpy(rgb.transpose(2,0,1).copy())
        expected = (expected-expected.new_tensor([.485,.456,.406])[:,None,None])/expected.new_tensor([.229,.224,.225])[:,None,None]
        if sample["input_failure"] or sample["image"].shape != (3,512,512) or not torch.equal(sample["image"],expected):
            raise ValueError("source-ID input differs from the existing512 pipeline")
    batches["id"] = device_batch(default_collate(samples),device)
    model = create_model(pretrained=True).to(device)
    metadata = model_metadata(model)
    original = {key:value.detach().clone() for key,value in model.state_dict().items()}
    encoder_initial = {key:value.detach().clone() for key,value in model.encoder.state_dict().items()}
    counters = {key:value.clone() for key,value in model.encoder.state_dict().items() if key.endswith("num_batches_tracked")}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key:value.cpu() for key,value in original.items()},strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key],value.cpu()) for key,value in original.items())
    del reloaded
    if (len(counters),metadata["parameters"],metadata["parameter_tensors"],metadata["state_tensors"],
            len(encoder_initial),sum(1 for _ in model.encoder.parameters())) != (77,8435793,233,464,462,231):
        raise ValueError("the unchanged full reader differs from the frozen architecture")
    cnn_counts,cnn_hook = encoder_counter(model)
    point_counts,point_hook = point_counter(model)
    point_backs,cnn_backs = [],[]
    backward_hook = model.point.weight.register_hook(lambda gradient:point_backs.append(True))
    observed_inputs,observed_pool,observed_point = [],[],[]
    def input_shape(_module,inputs):
        observed_inputs.append({"shape":list(inputs[0].shape),"dtype":str(inputs[0].dtype)})
    def pool_shape(_module,inputs,result):
        observed_pool.append({"before":list(inputs[0].shape),"after":list(result.shape)})
    def point_shape(_module,inputs,result):
        observed_point.append({"semantic_shape":list(inputs[0].shape),"point_logit_shape":list(result.shape),
            "semantic_dtype":str(inputs[0].dtype),"point_logit_dtype":str(result.dtype)})
    hooks = [cnn_hook,point_hook,backward_hook,model.encoder.register_forward_pre_hook(input_shape),
             model.encoder.global_pool.register_forward_hook(pool_shape),model.point.register_forward_hook(point_shape)]
    phases = {"initialization":phase_cost(device,started)}
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    try:
        lp_mode(model)
        with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.float16):
            features = detached_semantic(model,batches["lp"]["image"])
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            prediction = point_prediction(model,features)
            lp_loss = F.smooth_l1_loss(prediction,batches["lp"]["y"],beta=.05)
        (lp_loss*128.).backward()
        for parameter in model.point.parameters():
            parameter.grad.div_(128.)
        lp_encoder_grad_none = all(parameter.grad is None for parameter in model.encoder.parameters())
        lp_point_finite_nonzero = all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.point.parameters()) and any(bool(parameter.grad.abs().sum()>0) for parameter in model.point.parameters())
        lp_encoder_unchanged = state_equal(model.encoder,encoder_initial)
        lp_BN_unchanged = all(torch.equal(model.encoder.state_dict()[key],value) for key,value in counters.items())
        detached = not features.requires_grad and features.grad_fn is None and features.dtype == torch.float16
        lp_scalar = prediction.shape == (2,) and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all())
        del features
        enter_finetuning(model,None,lambda _state:None)
        grads_cleared = all(parameter.grad is None for parameter in model.parameters())
        hooks.append(model.encoder.conv_stem.weight.register_hook(lambda gradient:cnn_backs.append(True)))
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            prediction = model(batches["ft"]["image"])
            ft_loss = F.smooth_l1_loss(prediction,batches["ft"]["y"],beta=.05)
        (ft_loss*128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite,nonzero = gradient_evidence(model)
        ft_scalar = prediction.shape == (4,) and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all())
        if not all(torch.equal(model.encoder.state_dict()[key],value+1) for key,value in counters.items()):
            raise ValueError("all77 BN counters must advance exactly once during FT preflight")
        model.eval()
        before_eval = {key:value.clone() for key,value in model.named_buffers()}
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
            prediction = model(batches["id"]["image"])
        id_scalar = prediction.shape == (2,) and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all()) and bool(((prediction>=0)&(prediction<=1)).all())
        buffers_unchanged = all(torch.equal(value,before_eval[key]) for key,value in model.named_buffers())
        parameters_unchanged = all(torch.equal(value,original[key]) for key,value in model.named_parameters())
    finally:
        for hook in hooks:
            hook.remove()
    expected_inputs = [{"shape":[n,3,512,512],"dtype":"torch.float32"} for n in (2,4,2)]
    expected_pool = [{"before":[n,960,16,16],"after":[n,960,1,1]} for n in (2,4,2)]
    expected_point = [{"semantic_shape":[n,1280],"point_logit_shape":[n,1],
        "semantic_dtype":"torch.float16","point_logit_dtype":"torch.float16"} for n in (2,4,2)]
    if (cnn_counts != {"cnn_forward_calls":3,"cnn_image_forwards":8}
            or point_counts != {"point_forward_calls":3,"point_image_forwards":8}
            or len(point_backs) != 2 or len(cnn_backs) != 1 or observed_inputs != expected_inputs
            or observed_pool != expected_pool or observed_point != expected_point or not strict_reload
            or not all((lp_encoder_grad_none,lp_point_finite_nonzero,lp_encoder_unchanged,lp_BN_unchanged,detached,
                        grads_cleared,finite,all(nonzero.values()),parameters_unchanged,buffers_unchanged,
                        lp_scalar,ft_scalar,id_scalar,bool(torch.isfinite(lp_loss)),bool(torch.isfinite(ft_loss))))):
        raise ValueError("the source-only LP/FT zero-update preflight failed")
    phases["preflight"] = phase_cost(device,tick)
    write_json(output,{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "status":"passed","model_metadata":metadata,"initialization":model.initialization_evidence,
        "LP_sample_ids":batches["lp"]["sample_id"],"FT_sample_ids":batches["ft"]["sample_id"],
        "source_ID_identity_sample_ids":batches["id"]["sample_id"],"source_ID_pipeline_identity":True,
        "observed_encoder_inputs":observed_inputs,"observed_pool_shapes":observed_pool,
        "observed_point_shapes_and_dtypes":observed_point,"LP_encoder_grad_none":lp_encoder_grad_none,
        "LP_encoder_state_tensors_unchanged":462,"LP_BN_counter_increment":0,"LP_cache_detached_F16":detached,
        "LP_point_gradients_finite_nonzero":lp_point_finite_nonzero,"grads_cleared_between_phases":grads_cleared,
        "FT_all233_parameter_tensors_trainable":all(parameter.requires_grad for parameter in model.parameters()),
        "source_BN_counter_tensors_checked":77,"source_BN_counter_increment_FT":1,
        "source_BN_counter_additional_increment_eval":0,"eval_model_buffers_unchanged":buffers_unchanged,
        "parameter_tensors_unchanged_without_optimizer":233,"strict_model_reload":strict_reload,"full_state_tensors":464,
        "all_gradients_finite":finite,"gradient_groups_nonzero":nonzero,
        "LP_smooth_l1":float(lp_loss.detach()),"FT_smooth_l1":float(ft_loss.detach()),"preflight_backward_loss_scale":128.,
        "counts":PREFLIGHT_COUNTS,"source_OOD_RGB_reads":0,"source_OOD_forward_calls":0,
        "external_images_calls_backward_updates":0,"formal_fresh_rebuild_required":True,
        "runtime":runtime(device,started,phases)})


def extract_training_cache(model, training, device):
    import torch
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter, point_counter
    from experiments.train_geoprr_v2 import device_batch, make_loader
    lp_mode(model)
    counts,hook = encoder_counter(model)
    points,point_hook = point_counter(model)
    loader = make_loader(training,batch_size=64,workers=4,shuffle=False,seed=20262020,device=device)
    features,targets = [],[]
    offset = batches = 0
    last_batch = None
    try:
        with torch.no_grad():
            for raw in loader:
                batch = device_batch(raw,device)
                n = len(batch["y"])
                if list(batch["sample_id"]) != [str(row["sample_id"]) for row in training.rows[offset:offset+n]]:
                    raise ValueError("the one-pass LP cache changed existing source row order")
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    semantic = detached_semantic(model,batch["image"])
                if (semantic.shape != (n,1280) or semantic.dtype != torch.float16
                        or semantic.requires_grad or semantic.grad_fn is not None or semantic.device != model.point.weight.device
                        or not bool(torch.isfinite(semantic).all())):
                    raise ValueError("LP cache must contain detached finite device F16 semantic1280 features")
                features.append(semantic)
                targets.append(batch["y"].detach())
                offset += n
                batches += 1
                last_batch = n
    finally:
        hook.remove()
        point_hook.remove()
    if ((batches,last_batch,offset) != (86,56,5496)
            or counts != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
            or points != {"point_forward_calls":0,"point_image_forwards":0}):
        raise ValueError("LP cache differs from one no-shuffle5496 RGB pass and zero point forwards")
    return torch.cat(features).detach(),torch.cat(targets).detach(),counts


def fit_cached_head(model, features, targets, optimizer, scaler):
    import torch
    from torch.nn import functional as F
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter, point_counter
    points,hook = point_counter(model)
    counts,cnn_hook = encoder_counter(model)
    backwards = []
    backward_hook = model.point.weight.register_hook(lambda gradient:backwards.append(True))
    losses,updates,skips = [],0,0
    try:
        for _attempt in range(100):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                prediction = point_prediction(model,features)
                loss = F.smooth_l1_loss(prediction,targets,beta=.05)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite fixed full-batch LP source loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.point.parameters(),5.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < old_scale
            updates += int(not skipped)
            skips += int(skipped)
    finally:
        hook.remove()
        cnn_hook.remove()
        backward_hook.remove()
    if (len(losses) != 100 or len(backwards) != 100 or updates+skips != 100
            or points != {"point_forward_calls":100,"point_image_forwards":549600}
            or counts != {"cnn_forward_calls":0,"cnn_image_forwards":0}
            or any(parameter.grad is not None for parameter in model.encoder.parameters())):
        raise ValueError("LP differs from exactly100 full-cache point-only optimizer attempts")
    return {"losses_before_each_update":losses,"point_counts":points,"point_backward_calls":len(backwards),
        "CNN_backward_calls":0,"optimizer_updates":updates,"amp_skipped_updates":skips}


def evaluate_conditional_ood(gate, home, prepared, evaluate):
    """Construct the held-out source dataset only inside the original gate callback."""
    def open_and_evaluate():
        from experiments.geoprr_v2_high_resolution_direct import ScalarRGBDataset
        dataset = ScalarRGBDataset(home/PREPARED_RELATIVE/prepared["manifests"]["ood"],
                                   data_root=home/"datasets",augment=False)
        return evaluate("source_ood",dataset,220,4)[1]
    return conditional_ood(gate,open_and_evaluate)


def control(home, output):
    import torch
    from torch.nn import functional as F
    from experiments.geoprr_v2_high_resolution_direct import (
        create_model, encoder_counter, evaluate_population, model_metadata, point_counter, synchronize,
    )
    from experiments.train_geoprr_v2 import (
        device_batch, make_loader, restore_rng, rng_state, save_checkpoint, seed_everything,
    )
    if output.exists():
        raise FileExistsError("this bounded control cannot resume or overwrite complete or partial outputs")
    prepared = validate_prepared(home)
    witness = json.loads((home/PREFLIGHT_RELATIVE).read_text())
    if (witness["status"] != "passed" or witness["protocol"] != PROTOCOL or witness["config"] != CONFIG
            or witness["source_contract"] != prepared["source_contract"] or witness["counts"] != PREFLIGHT_COUNTS):
        raise ValueError("the fresh control requires its one completed source-only LP/FT preflight")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"started.json",{"protocol":PROTOCOL,"status":"started"})
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = source_populations(home,prepared)
    model = create_model(pretrained=True).to(device)
    initial_rng = rng_state()
    metadata,initialization = model_metadata(model),model.initialization_evidence
    encoder_initial = {key:value.detach().clone() for key,value in model.encoder.state_dict().items()}
    point_initial = {key:value.detach().clone() for key,value in model.point.state_dict().items()}
    if (metadata["parameters"],metadata["parameter_tensors"],metadata["state_tensors"],len(encoder_initial),
            sum(1 for _ in model.encoder.parameters())) != (8435793,233,464,462,231):
        raise ValueError("the complete fresh reader differs from its frozen architecture")
    totals,history,phases = initial_totals(),[],{}
    base = {"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "initialization":initialization,"model_metadata":metadata,"initial_rng":initial_rng}
    write_json(output/"run.json",{key:value for key,value in base.items() if key != "initial_rng"} | {
        "fresh_default_LP_and_FT_GradScalers":True,"preflight_state_reused":False,
        "source_unique_images":5496,"cache_feature_values_saved_or_exported":False})
    save_checkpoint(output/"initial.pt",base | {"phase":"fresh_initialization","model":model.state_dict(),"rng":initial_rng})
    phases["initialization"] = phase_cost(device,started)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    features,targets,cache_counts = extract_training_cache(model,training,device)
    if not state_equal(model.encoder,encoder_initial):
        raise ValueError("one LP cache extraction changed an encoder state tensor")
    phases["LP_cache"] = phase_cost(device,tick)
    totals.update(cache_cnn_image_forwards=cache_counts["cnn_image_forwards"],
                  cache_cnn_forward_calls=cache_counts["cnn_forward_calls"],cache_seconds=phases["LP_cache"]["seconds"])
    optimizer = new_optimizer(model,point_only=True)
    scaler = torch.amp.GradScaler("cuda",enabled=True)
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    lp_record = fit_cached_head(model,features,targets,optimizer,scaler)
    phases["LP_point"] = phase_cost(device,tick)
    if not state_equal(model.encoder,encoder_initial):
        raise ValueError("LP changed one of the462 cached encoder state tensors")
    lp_record.update(encoder_state_tensors_unchanged=462,point_parameters_changed=not state_equal(model.point,point_initial),
                     cost=phases["LP_point"],cache_cost=phases["LP_cache"],cache_features_saved_or_exported=False)
    totals.update(lp_point_forward_calls=lp_record["point_counts"]["point_forward_calls"],
        lp_point_image_forwards=lp_record["point_counts"]["point_image_forwards"],
        lp_point_backward_calls=lp_record["point_backward_calls"],lp_optimizer_updates=lp_record["optimizer_updates"],
        lp_amp_skipped_updates=lp_record["amp_skipped_updates"],lp_seconds=phases["LP_point"]["seconds"])
    write_json(output/"LP.json",lp_record)
    save_checkpoint(output/"lp.pt",base | {"phase":"LP_complete","model":model.state_dict(),
        "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"rng":rng_state(),"history":[],
        "LP":lp_record,"totals":dict(totals),"phases":dict(phases),"source_feature_values_exported":False})
    del optimizer,scaler,features,targets,encoder_initial,point_initial
    enter_finetuning(model,initial_rng,restore_rng)
    optimizer = new_optimizer(model,point_only=False)
    scaler = torch.amp.GradScaler("cuda",enabled=True)
    for epoch in range(1,6):
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        model.train()
        loader = make_loader(training,batch_size=64,workers=4,shuffle=True,seed=20262020+epoch,device=device)
        counts,hook = encoder_counter(model)
        points,point_hook = point_counter(model)
        cnn_backs,point_backs = [],[]
        backward_hook = model.encoder.conv_stem.weight.register_hook(lambda gradient:cnn_backs.append(True))
        point_backward_hook = model.point.weight.register_hook(lambda gradient:point_backs.append(True))
        batches = updates = skips = 0
        last_batch = None
        loss_sum = error_sum = 0.
        try:
            for raw in loader:
                batch = device_batch(raw,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction,batch["y"],beta=.05)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite source loss at FT epoch{epoch} batch{batches+1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                last_batch = len(batch["y"])
                batches += 1
                updates += int(not skipped)
                skips += int(skipped)
                loss_sum += float(loss.detach())*last_batch
                error_sum += float((prediction.detach()-batch["y"]).abs().sum())
        finally:
            for handle in (hook,point_hook,backward_hook,point_backward_hook):
                handle.remove()
        synchronize(device)
        cost = phase_cost(device,tick)
        phases[f"FT_epoch_{epoch}"] = cost
        if ((batches,last_batch,len(cnn_backs),len(point_backs)) != (86,56,86,86)
                or counts != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or points != {"point_forward_calls":86,"point_image_forwards":5496}
                or updates+skips != 86 or optimizer.param_groups[0]["lr"] != .0003):
            raise RuntimeError("FT epoch differs from the fixed5496 image/86 joint-backward budget")
        increments = {"train_batches":batches,"train_cnn_image_forwards":5496,"train_cnn_forward_calls":86,
            "backward_calls":len(cnn_backs),"optimizer_updates":updates,"amp_skipped_updates":skips,
            "train_point_forward_calls":86,"train_point_image_forwards":5496,"training_seconds":cost["seconds"]}
        for key,value in increments.items():
            totals[key] += value
        record = {"epoch":epoch,"learning_rate":.0003,"source_training_smooth_l1":loss_sum/5496,
            "source_training_nmae":error_sum/5496,"train_counts":counts,"train_point_counts":points,
            "CNN_backward_calls":len(cnn_backs),"point_backward_calls":len(point_backs),
            "optimizer_updates":updates,"amp_skipped_updates":skips,"last_batch_size":last_batch,
            "cost":cost,"validation_calls":0}
        history.append(record)
        checkpoint = base | {"phase":"FT","model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":epoch,"history":history,"LP":lp_record,
            "totals":dict(totals),"phases":dict(phases),"checkpointed_process_seconds":time.perf_counter()-started}
        save_checkpoint(output/f"ft_epoch_{epoch}.pt",checkpoint)
        (output/"history.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history))
        print(json.dumps({"event":"epoch_complete","epoch":epoch,"CNN_images":5496,"CNN_calls":86,
            "point_calls":86,"CNN_backward_attempts":86,"validation_calls":0}),flush=True)

    def evaluate(name,dataset,images,calls):
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        points,hook = point_counter(model)
        try:
            rows,cost = evaluate_population(model,dataset,device)
        finally:
            hook.remove()
        phases[name] = phase_cost(device,tick)
        if (len(rows) != images or cost["counts"]["cnn_image_forwards"] != images
                or cost["counts"]["cnn_forward_calls"] != calls
                or points != {"point_forward_calls":calls,"point_image_forwards":images}):
            raise RuntimeError("source evaluation differs from its frozen complete denominator")
        cost.update(point_counts=points,phase_cost=phases[name])
        for key,value in {"validation_images":images,"validation_cnn_forward_calls":calls,
            "validation_point_forward_calls":calls,"validation_point_image_forwards":images,
            "validation_seconds":cost["pipeline_seconds"]}.items():
            totals[key] += value
        return rows,_save_evaluation(output,name,rows,cost)

    id_rows,id_report = evaluate("source_id",source_id,292,5)
    gate,baseline = learning_gate(id_rows)
    _save_evaluation(output,"constant_source_id",baseline,{"CNN_calls":0,"CNN_images":0})
    write_json(output/"learning_gate.json",gate)
    comparison = primary_comparison(gate)
    write_json(output/"primary_comparison.json",comparison)
    ood_report = evaluate_conditional_ood(gate,home,prepared,evaluate)
    validate_totals(totals,gate["passed"])
    checkpoint.update(totals=dict(totals),rng=rng_state(),source_id_rows=id_rows,learning_gate=gate,
        primary_comparison=comparison,source_ood=ood_report,phases=dict(phases),
        checkpointed_process_seconds=time.perf_counter()-started)
    save_checkpoint(output/"last.pt",checkpoint)
    reason = "source_ood_observed" if gate["passed"] else "learning_gate_failed"
    write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "status":"complete","epochs":5,"completion_reason":reason,"learning_gate":gate,
        "primary_comparison":comparison,"source_id":id_report,"source_ood":ood_report,"LP":lp_record,
        "totals":totals,"counts":count_summary(totals),"model_metadata":metadata,"initialization":initialization,
        "checkpoint":"last.pt","source_OOD_RGB_images":220 if gate["passed"] else 0,
        "source_OOD_forward_calls":4 if gate["passed"] else 0,
        "external_counts":{"images":0,"CNN_calls":0,"point_calls":0,"backward_calls":0,"optimizer_updates":0},
        "runtime":runtime(device,started,phases)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("preflight","control"))
    parser.add_argument("--home",type=Path,default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        preflight(args.home,args.output or args.home/PREFLIGHT_RELATIVE)
    else:
        control(args.home,args.output or args.home/OUTPUT_RELATIVE/"reader")


if __name__ == "__main__":
    main()
