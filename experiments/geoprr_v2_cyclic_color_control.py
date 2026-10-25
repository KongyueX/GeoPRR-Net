"""One fixed source-only C3 color control with conditional held-out RG evaluation."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path

from experiments.geoprr_v2_needle_holdout_control import (
    CONFIG as POPULATION_CONFIG, PREPARED_RELATIVE, PROTOCOL as PREPARATION_PROTOCOL,
    SEEN, _save_evaluation, learning_gate, source_populations, validate_prepared, write_json,
)
from experiments.geoprr_v2_head_warmstart_control import phase_cost


PROTOCOL = "geoprr_v2_cyclic_color_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_cyclic_color_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_cyclic_color_control_preflight_20261026/result.json"
TRAIN_ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
NATIVE_ORDER = (0, 1, 2)
COUNTERFACTUAL_ORDER = (1, 0, 2)
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
CONFIG = deepcopy(POPULATION_CONFIG)
CONFIG.update(protocol=PROTOCOL, preparation_protocol=PREPARATION_PROTOCOL,
    kind="one bounded source-only color-cue control; not formal V32 or real external evaluation")
CONFIG["source"].update(OOD_images=0, OOD_calls=0,
    population_packet=PREPARED_RELATIVE, counterfactual_images=292, counterfactual_calls=5)
CONFIG["training"].update(epochs=15, max_epochs=15, validation_before_epoch5=False,
    intermediate_validation=False, resume="exact new checkpoint at next complete epoch only")
CONFIG["operator"] = {"train_orders":[list(order) for order in TRAIN_ORDERS],
    "order_index":"(dataset_index + epoch - 1) % 3", "orders_per_epoch":[1832,1832,1832],
    "each_item_exposure_15_epochs":[5,5,5], "native_order":list(NATIVE_ORDER),
    "counterfactual_order":list(COUNTERFACTUAL_ORDER), "mean":list(MEAN), "std":list(STD),
    "placement":"after original photo transform and before encoder, in-memory FP32 normalized RGB",
    "normalized_affine":"zprime_c=(std_perm/std_c)*z_perm+(mean_perm-mean_c)/std_c",
    "identity_returns_original_tensor":True, "RNG_draws":0, "CNN_views_per_presentation":1,
    "extra_loss":None, "clip_grayscale_resize_pixel_cache":False}
CONFIG["native_primary"] = {"baseline":"saved directFT5 source-ID292, no new baseline forward",
    "mean_max":.06132800787729562, "tail_strict_max":.25669996334860723,
    "maximum_max":.5787063241004944, "counts_over_1_2_5_max":[245,209,114]}
CONFIG["counterfactual_gate"] = {"mean_max":.12504463703046659, "tail_max":.23692208052865073,
    "mean_absolute_scalar_drift_max":.01, "tail30_absolute_scalar_drift_max":.05,
    "failed_pair_drift":1., "condition":"native learning AND native primary pass; same last15 only"}
CONFIG["checkpoint"] = "latest full state after each complete epoch; last15 only; no best or scheduler"
CONFIG["budget_qualification"] = "15 epochs is 3x directFT5 raw-image/update budget; no matched-budget causal augmentation claim"
PREFLIGHT_COUNTS = {"CNN_calls":3,"CNN_images":8,"RGB_unique_images":8,
    "point_calls":3,"point_examples":8,"CNN_backward_calls":1,"combined_backward_calls":1,"optimizer_updates":0}


def normalized_channel_permutation(image, order):
    """Represent a raw RGB permutation under the unchanged ImageNet normalization."""
    import torch
    order = tuple(order)
    if sorted(order) != [0,1,2] or image.ndim not in (3,4) or image.shape[-3] != 3:
        raise ValueError("the color operator requires one three-channel permutation and CHW/NCHW input")
    if image.dtype != torch.float32:
        raise ValueError("the color operator requires the existing FP32 normalized RGB input")
    if order == NATIVE_ORDER:
        return image
    shape = (3,1,1) if image.ndim == 3 else (1,3,1,1)
    mean, std = image.new_tensor(MEAN), image.new_tensor(STD)
    indices = list(order)
    ratio = (std[indices]/std).reshape(shape)
    offset = ((mean[indices]-mean)/std).reshape(shape)
    return image[...,indices,:,:]*ratio+offset


def training_order_index(dataset_index, epoch):
    if dataset_index < 0 or not 1 <= epoch <= 15:
        raise ValueError("the frozen color schedule has nonnegative indices and epochs1 through15")
    return (dataset_index+epoch-1)%3


class ChannelOrderDataset:
    """Apply one deterministic color order to each existing source presentation."""
    def __init__(self, base, *, epoch=None, order=None):
        if (epoch is None) == (order is None):
            raise ValueError("select exactly the fixed training epoch or fixed evaluation order")
        self.base, self.epoch, self.order = base, epoch, order
        self.rows = base.rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        order_index = training_order_index(index,self.epoch) if self.epoch is not None else None
        order = TRAIN_ORDERS[order_index] if order_index is not None else self.order
        result = {**sample,"image":normalized_channel_permutation(sample["image"],order)}
        if order_index is not None:
            result["channel_order_index"] = order_index
        return result


def native_primary(gate):
    summary, limits = gate["model"], CONFIG["native_primary"]
    counts = [292-summary[f"count_at_{threshold}_percent_fs"] for threshold in (1,2,5)]
    predicates = {"mean":summary["nmae"] <= limits["mean_max"],
        "tail":summary["tail10_nmae"] < limits["tail_strict_max"],
        "maximum":summary["maximum_error"] <= limits["maximum_max"],
        "counts_over_1_2_5":all(value <= limit for value,limit in zip(counts,limits["counts_over_1_2_5_max"],strict=True))}
    return {"passed":all(predicates.values()),"predicates":predicates,"limits":limits,
        "mean":summary["nmae"],"tail":summary["tail10_nmae"],"maximum":summary["maximum_error"],
        "counts_over_1_2_5":counts,"budget_qualification":CONFIG["budget_qualification"]}


def counterfactual_gate(native, counterfactual, summary):
    if len(native) != 292 or len(counterfactual) != 292 or len({r["sample_id"] for r in native}) != 292:
        raise ValueError("paired color evaluation requires all292 distinct source-ID rows")
    drifts, rescued, regressed, unchanged = [], 0, 0, 0
    threshold_pairs = {str(t):{"rescues":0,"regressions":0} for t in (1,2,5)}
    for before,after in zip(native,counterfactual,strict=True):
        if any(before[key] != after[key] for key in ("sample_id","group_id","target")):
            raise ValueError("counterfactual scalar rows are not exactly aligned to native rows and labels")
        valid_pair = before["valid"] and after["valid"]
        drift = abs(after["prediction"]-before["prediction"]) if valid_pair else 1.
        if not math.isfinite(drift):
            raise ValueError("saved scalar drift is nonfinite")
        drifts.append(drift)
        rescued += after["absolute_error"] < before["absolute_error"]
        regressed += after["absolute_error"] > before["absolute_error"]
        unchanged += after["absolute_error"] == before["absolute_error"]
        for threshold in (1,2,5):
            old = before["valid"] and before["absolute_error"] <= threshold/100
            new = after["valid"] and after["absolute_error"] <= threshold/100
            threshold_pairs[str(threshold)]["rescues"] += int(new and not old)
            threshold_pairs[str(threshold)]["regressions"] += int(old and not new)
    mean,tail = sum(drifts)/292,sum(sorted(drifts,reverse=True)[:30])/30
    limits = CONFIG["counterfactual_gate"]
    predicates = {"mean":summary["nmae"] <= limits["mean_max"],
        "tail":summary["tail10_nmae"] <= limits["tail_max"],
        "mean_drift":mean <= limits["mean_absolute_scalar_drift_max"],
        "tail_drift":tail <= limits["tail30_absolute_scalar_drift_max"]}
    return {"passed":all(predicates.values()),"predicates":predicates,"limits":limits,
        "denominator":292,"tail_count":30,"mean_absolute_scalar_drift":mean,
        "tail30_absolute_scalar_drift":tail,"maximum_absolute_scalar_drift":max(drifts),
        "paired_error_rescues":rescued,"paired_error_regressions":regressed,"paired_error_unchanged":unchanged,
        "paired_thresholds_percent_fs":threshold_pairs,
        "failed_pairs":sum(not (a["valid"] and b["valid"]) for a,b in zip(native,counterfactual,strict=True))}


def conditional_counterfactual(learning, primary, source_id, evaluate):
    if not (learning["passed"] and primary["passed"]):
        return None
    return evaluate("source_counterfactual",ChannelOrderDataset(source_id,order=COUNTERFACTUAL_ORDER),292,5)


def initial_totals():
    result = dict.fromkeys(("train_batches","train_cnn_image_forwards","train_cnn_forward_calls",
        "backward_calls","point_backward_calls","optimizer_updates","amp_skipped_updates",
        "train_point_forward_calls","train_point_image_forwards","validation_images",
        "validation_cnn_forward_calls","validation_point_forward_calls","validation_point_image_forwards"),0)
    result.update(training_seconds=0.,validation_seconds=0.)
    return result


def count_summary(totals):
    return {"cnn_image_forwards":totals["train_cnn_image_forwards"]+totals["validation_images"],
        "cnn_forward_calls":totals["train_cnn_forward_calls"]+totals["validation_cnn_forward_calls"],
        "point_image_forwards":totals["train_point_image_forwards"]+totals["validation_point_image_forwards"],
        "point_forward_calls":totals["train_point_forward_calls"]+totals["validation_point_forward_calls"],
        "CNN_backward_calls":totals["backward_calls"],"point_backward_calls":totals["point_backward_calls"],
        "source_backward_calls":totals["backward_calls"],"optimizer_updates":totals["optimizer_updates"],
        "amp_skipped_updates":totals["amp_skipped_updates"],"source_OOD_RGB_images":0,"source_OOD_CNN_calls":0,
        "external_images":0,"external_CNN_calls":0,"external_point_calls":0,
        "external_backward_calls":0,"external_optimizer_updates":0}


def validate_training_totals(totals, epoch):
    expected = {"train_batches":86*epoch,"train_cnn_image_forwards":5496*epoch,
        "train_cnn_forward_calls":86*epoch,"backward_calls":86*epoch,"point_backward_calls":86*epoch,
        "train_point_forward_calls":86*epoch,"train_point_image_forwards":5496*epoch}
    if (any(totals[key] != value for key,value in expected.items())
            or totals["optimizer_updates"]+totals["amp_skipped_updates"] != 86*epoch):
        raise ValueError("saved color training totals differ from the complete-epoch exposure")


def validate_totals(totals, counterfactual_eligible):
    validate_training_totals(totals,15)
    images,calls = (584,10) if counterfactual_eligible else (292,5)
    if (totals["validation_images"],totals["validation_cnn_forward_calls"],
            totals["validation_point_forward_calls"],totals["validation_point_image_forwards"]) != (images,calls,calls,images):
        raise ValueError("native/conditional color evaluation differs from its frozen denominator")
    summary = count_summary(totals)
    if (summary["cnn_image_forwards"],summary["cnn_forward_calls"],summary["point_image_forwards"],
            summary["point_forward_calls"]) != ((83024,1300,83024,1300) if counterfactual_eligible else (82732,1295,82732,1295)):
        raise ValueError("complete color control differs from its frozen image and call budget")


def validate_resume(saved, contract, optimizer_contract=None, scaler_contract=None):
    required = {"protocol","config","source_contract","model","optimizer","scaler","rng","epoch",
        "history","totals","initialization","model_metadata","phases","checkpointed_process_seconds","phase"}
    if not required <= set(saved):
        raise ValueError("resume requires the complete new model/optimizer/scaler/RNG/history/totals checkpoint")
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["source_contract"] != contract
            or saved["phase"] != "training" or not 1 <= saved["epoch"] <= 15):
        raise ValueError("resume protocol, configuration, source data or next-epoch boundary differs")
    history,epoch,totals = saved["history"],saved["epoch"],saved["totals"]
    if [row["epoch"] for row in history] != list(range(1,epoch+1)):
        raise ValueError("resume history does not cover every completed epoch")
    for row in history:
        if (row["train_counts"] != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or row["train_point_counts"] != {"point_forward_calls":86,"point_image_forwards":5496}
                or row["CNN_backward_calls"] != 86 or row["point_backward_calls"] != 86
                or row["optimizer_updates"]+row["amp_skipped_updates"] != 86
                or row["last_batch_size"] != 56 or row["order_exposures"] != [1832,1832,1832]
                or row["validation_calls"] != 0 or row["learning_rate"] != .0003):
            raise ValueError("resume history changed a fixed epoch or C3 exposure boundary")
    validate_training_totals(totals,epoch)
    if (any(totals[key] != 0 for key in ("validation_images","validation_cnn_forward_calls",
            "validation_point_forward_calls","validation_point_image_forwards"))
            or totals["optimizer_updates"] != sum(row["optimizer_updates"] for row in history)
            or totals["amp_skipped_updates"] != sum(row["amp_skipped_updates"] for row in history)
            or totals["training_seconds"] != sum(row["cost"]["seconds"] for row in history)):
        raise ValueError("resume totals differ from the complete saved history")
    if (set(saved["rng"]) != {"python","numpy","torch","cuda"}
            or not {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"} <= set(saved["scaler"])
            or not saved["model"] or not saved["optimizer"].get("param_groups")):
        raise ValueError("resume lacks the full stochastic, AMP or optimizer state")
    if optimizer_contract is not None and saved["optimizer"]["param_groups"] != optimizer_contract["param_groups"]:
        raise ValueError("resume optimizer parameters or fixed hyperparameters differ")
    if scaler_contract is not None and any(saved["scaler"][key] != scaler_contract[key]
            for key in ("growth_factor","backoff_factor","growth_interval")):
        raise ValueError("resume AMP scaler differs from the fixed default growth policy")


def preflight_indices(prepared):
    train = list(prepared["pref_indices"]["train"])
    rows = prepared["source_contract"]["populations"]["id"]
    groups = [sorted((i for i,row in enumerate(rows) if row["metadata"]["needle_control"]["pointer"] == alias),
                     key=lambda i:rows[i]["sample_id"])[:2] for alias in SEEN]
    selected = [prepared["source_contract"]["populations"]["train"][i] for i in train]+[rows[i] for group in groups for i in group]
    if (len(train) != 4 or any(len(group) != 2 for group in groups)
            or len({row["sample_id"] for row in selected}) != 8
            or len({row.get("roi_path",row.get("image_path")) for row in selected}) != 8):
        raise ValueError("the fixed preflight requires eight distinct existing source RGBs")
    return {"train":train,"id":groups}


def runtime(device, started, phases, prior_seconds=0.):
    import torch
    return {"process_seconds":time.perf_counter()-started,
        "full_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED,
        "prior_checkpointed_process_seconds":prior_seconds,
        "full_process_boundary":"module entry before torch import through saved result artifacts; excludes final completion JSON and downtime",
        "device":str(device),"gpu":torch.cuda.get_device_name(device),"torch":torch.__version__,"cuda":torch.version.cuda,
        "peak_cuda_memory_bytes":max([torch.cuda.max_memory_allocated(device)]+[p["peak_cuda_memory_bytes"] for p in phases.values()]),
        "peak_cuda_reserved_bytes":max([torch.cuda.max_memory_reserved(device)]+[p["peak_cuda_reserved_bytes"] for p in phases.values()]),
        "individual_point_forward_seconds":None,"phases":phases,"supervisor_wall_record":"runner screen_summary.json"}


def preflight(home, output):
    import cv2
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_data import _resolve_image, crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import create_model,encoder_counter,gradient_evidence,model_metadata,point_counter
    from experiments.train_geoprr_v2 import device_batch,seed_everything

    prepared = validate_prepared(home)
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("the new color preflight already has complete or partial outputs")
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.parent/"started.json",{"protocol":PROTOCOL,"status":"started"})
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = source_populations(home,prepared)
    indices = preflight_indices(prepared)
    samples = [training[i] for i in indices["train"]]
    original_training = [sample["image"] for sample in samples]
    explicit_orders = (0,1,2,0)
    transformed = [{**sample,"image":normalized_channel_permutation(sample["image"],TRAIN_ORDERS[order])}
                   for sample,order in zip(samples,explicit_orders,strict=True)]
    inverse_checks = []
    for image in original_training:
        swapped = normalized_channel_permutation(image,COUNTERFACTUAL_ORDER)
        inverse_checks.append(torch.allclose(normalized_channel_permutation(swapped,COUNTERFACTUAL_ORDER),image,rtol=2e-6,atol=2e-6))
    if not all(inverse_checks) or any(s["image"].dtype != torch.float32 or s["image"].shape != (3,512,512)
            or not bool(torch.isfinite(s["image"]).all()) for s in transformed):
        raise ValueError("the in-memory C3/held-out operator tensor witness failed")
    train_batch = device_batch(default_collate(transformed),device)
    validation_batches,identity_ids = [],[]
    for subset in indices["id"]:
        native_samples = [source_id[i] for i in subset]
        for i,sample in zip(subset,native_samples,strict=True):
            row = source_id.rows[i]
            image = cv2.imread(str(_resolve_image(row,source_id.base.manifest_root,source_id.base.dataset_root)))
            roi,_ = crop_and_resize(image,None if "roi_path" in row else row.get("metadata",{}).get("dial_bbox"),512)
            rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            expected = torch.from_numpy(rgb.transpose(2,0,1).copy())
            expected = (expected-expected.new_tensor(MEAN)[:,None,None])/expected.new_tensor(STD)[:,None,None]
            if (sample["input_failure"] or not torch.equal(sample["image"],expected)
                    or normalized_channel_permutation(sample["image"],NATIVE_ORDER) is not sample["image"]):
                raise ValueError("native source-ID input differs from the unchanged512 RGB pipeline")
        validation_batches.append(device_batch(default_collate(native_samples),device))
        identity_ids.append([sample["sample_id"] for sample in native_samples])
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    original = {key:value.detach().clone() for key,value in model.state_dict().items()}
    counters = {key:value.clone() for key,value in original.items() if key.endswith("num_batches_tracked")}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key:value.cpu() for key,value in original.items()},strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key],value.cpu()) for key,value in original.items())
    del reloaded
    if (len(counters),metadata["parameters"],metadata["parameter_tensors"],metadata["state_tensors"],
            initialization["retained_complete_encoder_state_tensors"]) != (77,8435793,233,464,462):
        raise ValueError("the unchanged fresh complete MNV4 scalar reader differs")
    cnn_counts,cnn_hook = encoder_counter(model)
    point_counts,point_hook = point_counter(model)
    cnn_backs,point_backs,observed_inputs,observed_pool,observed_point = [],[],[],[],[]
    def input_shape(_module,inputs):
        observed_inputs.append({"shape":list(inputs[0].shape),"dtype":str(inputs[0].dtype)})
    def pool_shape(_module,inputs,result):
        observed_pool.append({"before":list(inputs[0].shape),"after":list(result.shape)})
    def point_shape(_module,inputs,result):
        observed_point.append({"semantic_shape":list(inputs[0].shape),"point_logit_shape":list(result.shape),
            "semantic_dtype":str(inputs[0].dtype),"point_logit_dtype":str(result.dtype)})
    hooks = [cnn_hook,point_hook,model.encoder.conv_stem.weight.register_hook(lambda g:cnn_backs.append(True)),
        model.point.weight.register_hook(lambda g:point_backs.append(True)),model.encoder.register_forward_pre_hook(input_shape),
        model.encoder.global_pool.register_forward_hook(pool_shape),model.point.register_forward_hook(point_shape)]
    phases = {"initialization":phase_cost(device,started)}
    torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    try:
        model.train()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            prediction = model(train_batch["image"])
            loss = F.smooth_l1_loss(prediction,train_batch["y"],beta=.05)
        train_scalar = prediction.shape == (4,) and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all())
        (loss*128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite,nonzero = gradient_evidence(model)
        if not all(torch.equal(model.state_dict()[key],value+1) for key,value in counters.items()):
            raise ValueError("all77 BN counters must advance exactly once in training preflight")
        model.eval()
        before_eval = {key:value.clone() for key,value in model.named_buffers()}
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
            for batch in validation_batches:
                prediction = model(batch["image"])
                if (prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all())
                        or not bool(((prediction>=0)&(prediction<=1)).all())):
                    raise ValueError("native source-ID preflight did not return finite FP32 scalars")
        buffers_unchanged = all(torch.equal(value,before_eval[key]) for key,value in model.named_buffers())
        parameters_unchanged = all(torch.equal(value,original[key]) for key,value in model.named_parameters())
    finally:
        for hook in hooks:
            hook.remove()
    expected_inputs = [{"shape":[n,3,512,512],"dtype":"torch.float32"} for n in (4,2,2)]
    expected_pool = [{"before":[n,960,16,16],"after":[n,960,1,1]} for n in (4,2,2)]
    expected_point = [{"semantic_shape":[n,1280],"point_logit_shape":[n,1],"semantic_dtype":"torch.float16",
                       "point_logit_dtype":"torch.float16"} for n in (4,2,2)]
    if (cnn_counts != {"cnn_forward_calls":3,"cnn_image_forwards":8} or point_counts != {"point_forward_calls":3,"point_image_forwards":8}
            or len(cnn_backs) != 1 or len(point_backs) != 1 or observed_inputs != expected_inputs
            or observed_pool != expected_pool or observed_point != expected_point or not strict_reload
            or not all((train_scalar,finite,all(nonzero.values()),parameters_unchanged,buffers_unchanged,bool(torch.isfinite(loss))))):
        raise ValueError("the source-only color zero-update preflight failed")
    phases["preflight"] = phase_cost(device,tick)
    write_json(output,{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "status":"passed","model_metadata":metadata,"initialization":initialization,"counts":PREFLIGHT_COUNTS,
        "probe_sample_ids":train_batch["sample_id"],"train_order_indices":list(explicit_orders),
        "source_ID_identity_sample_ids":identity_ids,"source_ID_pipeline_identity":True,
        "operator_FP32_shapes_finite":True,"heldout_operator_inverse_on_existing_train_CPU":all(inverse_checks),
        "heldout_operator_extra_RGB_reads":0,"heldout_operator_extra_CNN_calls":0,
        "observed_encoder_inputs":observed_inputs,"observed_pool_shapes":observed_pool,"observed_point_shapes_and_dtypes":observed_point,
        "source_BN_counter_tensors_checked":77,"source_BN_counter_increment_training":1,"source_BN_counter_additional_increment_eval":0,
        "parameter_tensors_unchanged_without_optimizer":233,"strict_model_reload":strict_reload,"full_state_tensors":464,
        "all_gradients_finite":finite,"gradient_groups_nonzero":nonzero,"source_smooth_l1":float(loss.detach()),
        "preflight_backward_loss_scale":128.,"source_OOD_RGB_reads":0,"source_OOD_forward_calls":0,
        "external_images_calls_backward_updates":0,"formal_fresh_rebuild_required":True,"runtime":runtime(device,started,phases)})


def load_evaluation(output, name, prepared, order):
    """Reuse a complete scalar observation; partial artifacts never authorize a new forward."""
    from experiments.geoprr_v2_high_resolution_direct import summarize
    paths = [output/f"{name}_{suffix}" for suffix in ("predictions.csv","groups.csv","summary.json")]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileExistsError("partial saved scalar evaluation cannot be forwarded again")
    report = json.loads(paths[2].read_text())
    if (report.get("protocol") != PROTOCOL or report.get("config") != CONFIG
            or report.get("source_contract") != prepared["source_contract"]
            or report.get("condition_order") != list(order) or report.get("epoch") != 15):
        raise ValueError("saved scalar evaluation protocol/config/data/last15 condition differs")
    with paths[0].open(newline="",encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("target","absolute_error"):
            row[key] = float(row[key])
        row["prediction"] = None if row["prediction"] == "" else float(row["prediction"])
        for key in ("valid","input_valid"):
            if row[key] not in ("True","False"):
                raise ValueError("saved scalar validity field differs")
            row[key] = row[key] == "True"
        for key in ("invalid_mass","outside_arc_mass"):
            row[key] = None if row[key] == "" else float(row[key])
    expected = prepared["source_contract"]["populations"]["id"]
    if (len(rows) != 292 or [r["sample_id"] for r in rows] != [str(r["sample_id"]) for r in expected]
            or report["summary"] != summarize(rows)):
        raise ValueError("saved scalar rows do not match the complete source-ID summary")
    return rows,report


def control(home, output):
    import torch
    from torch.nn import functional as F
    from experiments.geoprr_v2_high_resolution_direct import create_model,encoder_counter,evaluate_population,model_metadata,point_counter,synchronize
    from experiments.train_geoprr_v2 import device_batch,make_loader,restore_rng,rng_state,save_checkpoint,seed_everything

    prepared = validate_prepared(home)
    witness = json.loads((home/PREFLIGHT_RELATIVE).read_text())
    if (witness["status"] != "passed" or witness["protocol"] != PROTOCOL or witness["config"] != CONFIG
            or witness["source_contract"] != prepared["source_contract"] or witness["counts"] != PREFLIGHT_COUNTS):
        raise ValueError("formal color control requires its exact completed zero-update preflight")
    latest = output/"latest.pt"
    resuming = output.exists()
    if resuming and ((output/"completed.json").exists() or not latest.is_file() or (output/"active_epoch.json").exists()):
        raise FileExistsError("complete or partial color outputs cannot be overwritten; resume requires an intact complete-epoch latest")
    if not resuming:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"started.json",{"protocol":PROTOCOL,"status":"started"})
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = source_populations(home,prepared)
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda",enabled=True)
    base = {"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "initialization":initialization,"model_metadata":metadata,"fresh_default_GradScaler":True,
        "preflight_state_reused":False,"source_unique_images":5496,"source_presentations_per_epoch":5496}
    totals,history,start,prior_seconds = initial_totals(),[],0,0.
    phases = {"initialization":phase_cost(device,started)}
    if resuming:
        checkpoint = torch.load(latest,map_location="cpu",weights_only=False)
        validate_resume(checkpoint,prepared["source_contract"],optimizer.state_dict(),scaler.state_dict())
        if checkpoint["model_metadata"] != metadata or checkpoint["initialization"] != initialization:
            raise ValueError("resume cached initialization or unchanged model metadata differs")
        saved_history = "".join(json.dumps(row,allow_nan=False)+"\n" for row in checkpoint["history"])
        if not (output/"history.jsonl").is_file() or (output/"history.jsonl").read_text() != saved_history:
            raise ValueError("resume history artifact differs from the full saved checkpoint")
        model.load_state_dict(checkpoint["model"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        totals,history,start = checkpoint["totals"],checkpoint["history"],checkpoint["epoch"]
        prior_seconds = checkpoint["checkpointed_process_seconds"]
        phases = checkpoint["phases"]
        phases[f"resume_initialization_after_epoch_{start}"] = phase_cost(device,started)
    else:
        write_json(output/"run.json",base)
    for epoch in range(start+1,16):
        write_json(output/"active_epoch.json",{"protocol":PROTOCOL,"epoch":epoch,"last_complete_epoch":epoch-1})
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        model.train()
        dataset = ChannelOrderDataset(training,epoch=epoch)
        loader = make_loader(dataset,batch_size=64,workers=4,shuffle=True,seed=20262020+epoch,device=device)
        counts,hook = encoder_counter(model)
        points,point_hook = point_counter(model)
        cnn_backs,point_backs = [],[]
        cnn_backward = model.encoder.conv_stem.weight.register_hook(lambda g:cnn_backs.append(True))
        point_backward = model.point.weight.register_hook(lambda g:point_backs.append(True))
        batches = updates = skips = 0
        last_batch = None
        loss_sum = error_sum = 0.
        exposures = [0,0,0]
        try:
            for raw in loader:
                for index in raw["channel_order_index"].tolist():
                    exposures[index] += 1
                batch = device_batch(raw,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction,batch["y"],beta=.05)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite color source loss at epoch{epoch} batch{batches+1}")
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
            for handle in (hook,point_hook,cnn_backward,point_backward):
                handle.remove()
        synchronize(device)
        cost = phase_cost(device,tick)
        phases[f"train_epoch_{epoch}"] = cost
        if ((batches,last_batch,len(cnn_backs),len(point_backs)) != (86,56,86,86)
                or counts != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or points != {"point_forward_calls":86,"point_image_forwards":5496}
                or exposures != [1832,1832,1832] or updates+skips != 86 or optimizer.param_groups[0]["lr"] != .0003):
            raise RuntimeError("color epoch differs from fixed5496 images/86 attempts/C3 exposure")
        increments = {"train_batches":batches,"train_cnn_image_forwards":5496,"train_cnn_forward_calls":86,
            "backward_calls":len(cnn_backs),"point_backward_calls":len(point_backs),"optimizer_updates":updates,
            "amp_skipped_updates":skips,"train_point_forward_calls":86,"train_point_image_forwards":5496,
            "training_seconds":cost["seconds"]}
        for key,value in increments.items():
            totals[key] += value
        record = {"epoch":epoch,"learning_rate":.0003,"source_training_smooth_l1":loss_sum/5496,
            "source_training_nmae":error_sum/5496,"train_counts":counts,"train_point_counts":points,
            "CNN_backward_calls":len(cnn_backs),"point_backward_calls":len(point_backs),"optimizer_updates":updates,
            "amp_skipped_updates":skips,"last_batch_size":last_batch,"order_exposures":exposures,"cost":cost,"validation_calls":0}
        history.append(record)
        checkpoint = base | {"phase":"training","model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":epoch,"history":history,"totals":dict(totals),
            "phases":dict(phases),"checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
        save_checkpoint(latest,checkpoint)
        (output/"history.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history))
        (output/"active_epoch.json").unlink()
        print(json.dumps({"event":"epoch_complete","epoch":epoch,"CNN_images":5496,"CNN_calls":86,
            "point_calls":86,"CNN_backward_attempts":86,"order_exposures":exposures,"validation_calls":0}),flush=True)

    def evaluate(name,dataset,images,calls):
        order = NATIVE_ORDER if name == "source_id" else COUNTERFACTUAL_ORDER
        reused = load_evaluation(output,name,prepared,order)
        if reused is None:
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            points,hook = point_counter(model)
            try:
                rows,cost = evaluate_population(model,dataset,device)
            finally:
                hook.remove()
            phase = phase_cost(device,tick)
            if (len(rows) != images or cost["counts"]["cnn_image_forwards"] != images
                    or cost["counts"]["cnn_forward_calls"] != calls
                    or points != {"point_forward_calls":calls,"point_image_forwards":images}):
                raise RuntimeError("source color evaluation differs from the frozen292/5 complete denominator")
            cost.update(point_counts=points,phase_cost=phase)
            report = _save_evaluation(output,name,rows,cost)
            report.update(protocol=PROTOCOL,config=CONFIG,source_contract=prepared["source_contract"],condition_order=list(order),epoch=15)
            write_json(output/f"{name}_summary.json",report)
        else:
            rows,report = reused
            cost = report["cost"]
            if (cost["counts"]["cnn_image_forwards"] != images or cost["counts"]["cnn_forward_calls"] != calls
                    or cost["point_counts"] != {"point_forward_calls":calls,"point_image_forwards":images}):
                raise ValueError("reused evaluation cost differs from the fixed292/5 budget")
        phases[name] = cost["phase_cost"]
        for key,value in {"validation_images":images,"validation_cnn_forward_calls":calls,
            "validation_point_forward_calls":calls,"validation_point_image_forwards":images,
            "validation_seconds":cost["pipeline_seconds"]}.items():
            totals[key] += value
        return rows,report

    id_rows,id_report = evaluate("source_id",ChannelOrderDataset(source_id,order=NATIVE_ORDER),292,5)
    learning,baseline = learning_gate(id_rows)
    _save_evaluation(output,"constant_source_id",baseline,{"CNN_calls":0,"CNN_images":0})
    primary = native_primary(learning)
    write_json(output/"native_learning_gate.json",learning)
    write_json(output/"native_primary.json",primary)
    eligible = learning["passed"] and primary["passed"]
    observed = conditional_counterfactual(learning,primary,source_id,evaluate)
    cf_gate,cf_report = None,None
    if observed is not None:
        cf_rows,cf_report = observed
        cf_gate = counterfactual_gate(id_rows,cf_rows,cf_report["summary"])
        write_json(output/"counterfactual_gate.json",cf_gate)
    elif any(output.glob("source_counterfactual_*")):
        raise ValueError("ineligible native condition has unexpected counterfactual artifacts")
    validate_totals(totals,eligible)
    verdict = "SOURCE_COLOR_CONTROL_PASSED" if eligible and cf_gate["passed"] else "NOT_RETAINED"
    final = base | {"phase":"evaluated","model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":15,"history":history,"totals":dict(totals),
        "phases":phases,"native_learning_gate":learning,"native_primary":primary,"counterfactual_gate":cf_gate,
        "checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
    save_checkpoint(output/"last.pt",final)
    write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "status":"complete","epochs":15,"completion_reason":"counterfactual_observed" if eligible else "native_learning_or_primary_failed",
        "native_learning_gate":learning,"native_primary":primary,"counterfactual_eligible":eligible,
        "counterfactual_gate":cf_gate,"source_id":id_report,"source_counterfactual":cf_report,"verdict":verdict,
        "totals":totals,"counts":count_summary(totals),"model_metadata":metadata,"initialization":initialization,
        "checkpoint":"last.pt","source_OOD_RGB_images":0,"source_OOD_forward_calls":0,
        "external_counts":{"images":0,"CNN_calls":0,"point_calls":0,"backward_calls":0,"optimizer_updates":0},
        "budget_qualification":CONFIG["budget_qualification"],"runtime":runtime(device,started,phases,prior_seconds)})


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
