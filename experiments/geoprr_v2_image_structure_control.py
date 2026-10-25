"""One fixed thirty-epoch source control using a multichannel gradient measurement."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_cyclic_color_control as shared
from experiments import geoprr_v2_mean_tail_control as guards


PROTOCOL = "geoprr_v2_image_structure_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_image_structure_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_image_structure_control_preflight_20261026/result.json"
PREPARED_RELATIVE = shared.PREPARED_RELATIVE
EPOCHS = 30
KX = ((-1,0,1),(-2,0,2),(-1,0,1))
KY = ((-1,-2,-1),(0,0,0),(1,2,1))
MEAN,STD = shared.MEAN,shared.STD
OPERATOR_CONTRACT = {
    "public_input":"existing normalized native-ROI RGB float32 NCHW; original photo/LINEAR512/ImageNet",
    "encoder_input":"fixed RGB gradient second-moment measurement, then original ImageNet normalization",
    "math":"I=X*std+mean; reflectpad1; groupedSobel; E=[sum(dx^2)/48,sum(dy^2)/48,.5+sum(dx*dy)/96]; (E-mean)/std",
    "Kx":[list(row) for row in KX],"Ky":[list(row) for row in KY],"kernel_shape":[6,1,3,3],"groups":3,
    "gradient_order":["dxR","dyR","dxG","dyG","dxB","dyB"],
    "padding":"one-pixel torch reflect; no duplicated border pixel; same H/W and ROI coordinates",
    "precision":"inverse normalization, pad, grouped filter, products, sums and measurement normalization float32 with autocast disabled",
    "fixed_filter_tf32":"disabled for the CUDA grouped convolution only via scoped cudnn.flags; original enabled/benchmark/benchmark_limit/deterministic preserved and flags restored",
    "fixed_scales":[48,96],"new_trainable_parameters":0,"persistent_kernel_buffers":1,
    "kernel_dtype":"torch.float32","mean_std_registered_buffers":False,
    "bounds_verification_atol":2e-6,"clip_fit_rescale_cache":False,
    "cached_weight_scope":"encoder cached weight/state values retained; input feature semantics change",
    "timing":"CUDA start/end events for every operator forward, summed after phase synchronization; includes check/event overhead in full wall",
}
CONFIG = deepcopy(guards.CONFIG)
CONFIG.update(protocol=PROTOCOL,kind="one fixed observable image-measurement source control; not physical calibration or a new theory")
CONFIG.pop("objective")
CONFIG["training"].update(epochs=30,max_epochs=30,loss="original batch-mean SmoothL1 beta=.05",resume="exact new protocol/operator/full state at next complete epoch only")
CONFIG["operator"] = {"training_RGB_once_per_epoch":5496,"each_raw_RGB_exposures":30,
    "training_color_permutation":False,"counterfactual_raw_RGB_order":[1,0,2]}
CONFIG["input_operator"] = OPERATOR_CONTRACT
CONFIG["model"].update(factory="experiments.geoprr_v2_image_structure_control.create_model(pretrained=True)",
    cached_reader_factory=CONFIG["model"]["factory"],state_tensors=465,new_fixed_kernel_state_tensors=1,
    inference_input="native normalized RGB; fixed measurement is inside reader before the encoder")
CONFIG["checkpoint"] = "latest full state after each complete epoch; last30 only; no intermediate validation, scheduler or best"
CONFIG["budget_qualification"] = ("30E is twice identity15 scheduled train images/backward attempts, fixed in advance for capped input adaptation; "
    "no matched-budget or convergence claim; fixed-filter GPU compute, event overhead, phase wall and peaks are included")
PREFLIGHT_COUNTS = {"CNN_calls":3,"CNN_images":8,"RGB_unique_images":8,"point_calls":3,"point_examples":8,
    "fixed_filter_calls":3,"fixed_filter_images":8,"CNN_backward_calls":1,"combined_backward_calls":1,"optimizer_updates":0}
write_json = shared.write_json
validate_prepared = shared.validate_prepared


class FixedImageStructure(nn.Module):
    """One fixed grouped Sobel pass and local channel-summed second moments."""
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel",torch.tensor((KX,KY)*3,dtype=torch.float32).reshape(6,1,3,3))
        self.reset_phase()

    def reset_phase(self):
        self.calls,self.images = 0,0
        self.events,self.shapes = [],set()
        self.checks = None
        self.shape_preserved,self.float32_input,self.float32_output = True,True,True
        self.last_device = None

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or image.dtype != torch.float32:
            raise ValueError("the fixed measurement requires the existing float32 NCHW RGB interface")
        start=end=None
        if image.device.type == "cuda":
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast(device_type=image.device.type,enabled=False):
            mean,std = image.new_tensor(MEAN).view(1,3,1,1),image.new_tensor(STD).view(1,3,1,1)
            intensity = image*std+mean
            padded = F.pad(intensity,(1,1,1,1),mode="reflect")
            filter_flags = torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,benchmark_limit=torch.backends.cudnn.benchmark_limit,
                deterministic=torch.backends.cudnn.deterministic,allow_tf32=False) if image.device.type == "cuda" else nullcontext()
            with filter_flags:
                gradient = F.conv2d(padded,self.kernel,groups=3)
            dx,dy = gradient[:,0::2],gradient[:,1::2]
            measurement = torch.cat(((dx*dx).sum(1,keepdim=True)/48,
                (dy*dy).sum(1,keepdim=True)/48,.5+(dx*dy).sum(1,keepdim=True)/96),dim=1)
            result = (measurement-mean)/std
            checks = torch.stack((torch.isfinite(measurement).all(),
                ((measurement>=-2e-6)&(measurement<=1+2e-6)).all(),torch.isfinite(result).all())).detach()
            self.checks = checks if self.checks is None else self.checks & checks
        if end is not None:
            end.record()
            self.events.append((start,end))
        self.calls += 1
        self.images += len(image)
        self.shapes.add(tuple(image.shape))
        self.last_device = image.device
        self.shape_preserved &= result.shape == image.shape
        self.float32_input &= image.dtype == torch.float32
        self.float32_output &= measurement.dtype == result.dtype == torch.float32
        return result

    def phase_report(self):
        if self.last_device is not None and self.last_device.type == "cuda":
            torch.cuda.synchronize(self.last_device)
        checks = [True,True,True] if self.checks is None else self.checks.cpu().tolist()
        return {"fixed_filter_calls":self.calls,"fixed_filter_images":self.images,
            "operator_forward_gpu_seconds":sum(start.elapsed_time(end) for start,end in self.events)/1000 if self.events else None,
            "operator_timing_scope":"CUDA events per forward, after phase synchronization" if self.events else "CPU operator GPU time not measured",
            "input_shapes":[list(shape) for shape in sorted(self.shapes)],"shape_preserved":bool(self.shape_preserved),
            "input_float32":bool(self.float32_input),"measurement_and_encoder_input_float32":bool(self.float32_output),
            "measurement_finite":checks[0],"measurement_bounded":checks[1],"encoder_input_finite":checks[2]}


class ImageStructureReader(nn.Module):
    """Retain the complete scalar reader and apply the fixed measurement once before it."""
    def __init__(self, reader):
        super().__init__()
        self.reader = reader
        self.measurement = FixedImageStructure()
        self.semantic_dim = reader.semantic_dim
        if hasattr(reader,"initialization_evidence"):
            self.initialization_evidence = deepcopy(reader.initialization_evidence)
            self.initialization_evidence.update(fixed_measurement_kernel_states=1,
                input_semantics="fixed measurement; cached encoder state values retained")

    @property
    def encoder(self):
        return self.reader.encoder

    @property
    def point(self):
        return self.reader.point

    def forward(self, image):
        return self.reader(self.measurement(image))


def create_model(*, pretrained):
    from experiments.geoprr_v2_high_resolution_direct import create_model as original_factory
    return ImageStructureReader(original_factory(pretrained=pretrained))


def model_metadata(model):
    from experiments.geoprr_v2_high_resolution_direct import model_metadata as original_metadata
    result = original_metadata(model)
    result.update(operator_contract=OPERATOR_CONTRACT,operator_parameter_tensors=sum(1 for _ in model.measurement.parameters()),
        operator_state_tensors=len(model.measurement.state_dict()),operator_kernel_shape=list(model.measurement.kernel.shape),
        public_input_representation="native normalized RGB",encoder_input_representation="normalized fixed image structure measurement")
    return result


def phase_cost(device, started):
    return shared.phase_cost(device,started)


def runtime(device, started, phases, prior_seconds=0.):
    result = shared.runtime(device,started,phases,prior_seconds)
    result.update(full_process_seconds=prior_seconds+time.perf_counter()-PROCESS_STARTED,
        full_process_boundary="image-structure module entry before torch import through saved source metadata; excludes final completion JSON and downtime",
        individual_point_forward_seconds=None,operator_timing_scope="CUDA events per forward, summed after each phase sync; operator overhead included in full phase wall")
    return result


def validate_operator_report(report, images, calls):
    if (report["fixed_filter_images"] != images or report["fixed_filter_calls"] != calls
            or report["operator_forward_gpu_seconds"] is None
            or not math.isfinite(report["operator_forward_gpu_seconds"]) or report["operator_forward_gpu_seconds"]<0
            or not all(report[key] for key in ("shape_preserved","input_float32","measurement_and_encoder_input_float32",
                "measurement_finite","measurement_bounded","encoder_input_finite"))):
        raise ValueError("the actual fixed-filter count, float32 input/field or finite/bounded contract differs")


def initial_totals():
    result = shared.initial_totals()
    result.update(train_fixed_filter_calls=0,train_fixed_filter_images=0,train_operator_forward_gpu_seconds=0.,
        validation_fixed_filter_calls=0,validation_fixed_filter_images=0,validation_operator_forward_gpu_seconds=0.)
    return result


def count_summary(totals):
    result = shared.count_summary(totals)
    result.update(fixed_filter_calls=totals["train_fixed_filter_calls"]+totals["validation_fixed_filter_calls"],
        fixed_filter_images=totals["train_fixed_filter_images"]+totals["validation_fixed_filter_images"],
        operator_forward_gpu_seconds=totals["train_operator_forward_gpu_seconds"]+totals["validation_operator_forward_gpu_seconds"])
    return result


def validate_training_totals(totals, epoch):
    shared.validate_training_totals(totals,epoch)
    if (totals["train_fixed_filter_calls"],totals["train_fixed_filter_images"]) != (86*epoch,5496*epoch):
        raise ValueError("training filter exposure differs from the one-filter-per-encoder contract")


def validate_totals(totals, counterfactual_eligible):
    validate_training_totals(totals,30)
    images,calls = (584,10) if counterfactual_eligible else (292,5)
    if any(totals[key] != value for key,value in {"validation_images":images,"validation_cnn_forward_calls":calls,
            "validation_point_forward_calls":calls,"validation_point_image_forwards":images,
            "validation_fixed_filter_calls":calls,"validation_fixed_filter_images":images}.items()):
        raise ValueError("native/CF encoder, point and filter exposure differs from the full292-view budget")
    summary = count_summary(totals)
    expected_images,expected_calls = (165464,2590) if counterfactual_eligible else (165172,2585)
    if any(summary[key] != expected_images for key in ("cnn_image_forwards","point_image_forwards","fixed_filter_images")) or any(
            summary[key] != expected_calls for key in ("cnn_forward_calls","point_forward_calls","fixed_filter_calls")):
        raise ValueError("the fixed30E control exceeds or loses its declared total exposure")


def validate_resume(saved, contract, optimizer_contract=None, scaler_contract=None):
    required = {"protocol","config","source_contract","operator_contract","model","optimizer","scaler","rng","epoch",
        "history","totals","initialization","model_metadata","phases","checkpointed_process_seconds","phase"}
    if not required <= set(saved):
        raise ValueError("resume requires the complete new fixed-operator and full stochastic training state")
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["source_contract"] != contract
            or saved["operator_contract"] != OPERATOR_CONTRACT or saved["phase"] != "training" or not 1<=saved["epoch"]<=30):
        raise ValueError("resume new protocol/config/data/operator or next-complete-epoch boundary differs")
    kernel = saved["model"].get("measurement.kernel")
    expected_kernel = torch.tensor((KX,KY)*3,dtype=torch.float32).reshape(6,1,3,3)
    if (not isinstance(kernel,torch.Tensor) or kernel.dtype != torch.float32
            or kernel.shape != expected_kernel.shape or not torch.equal(kernel.cpu(),expected_kernel)):
        raise ValueError("resume fixed kernel state differs from the frozen float32 operator")
    history,epoch,totals = saved["history"],saved["epoch"],saved["totals"]
    if [row["epoch"] for row in history] != list(range(1,epoch+1)):
        raise ValueError("resume history does not contain every completed source epoch")
    for row in history:
        if (row["train_counts"] != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or row["train_point_counts"] != {"point_forward_calls":86,"point_image_forwards":5496}
                or row["CNN_backward_calls"] != 86 or row["point_backward_calls"] != 86
                or row["optimizer_updates"]+row["amp_skipped_updates"] != 86
                or row["last_batch_size"] != 56 or row["validation_calls"] != 0 or row["learning_rate"] != .0003):
            raise ValueError("resume history changed a fixed30E epoch boundary")
        validate_operator_report(row["operator"],5496,86)
    validate_training_totals(totals,epoch)
    if (any(totals[key] != 0 for key in ("validation_images","validation_cnn_forward_calls","validation_point_forward_calls",
            "validation_point_image_forwards","validation_fixed_filter_calls","validation_fixed_filter_images","validation_operator_forward_gpu_seconds"))
            or any(totals[key] != sum(row[key] for row in history) for key in ("optimizer_updates","amp_skipped_updates"))
            or totals["training_seconds"] != sum(row["cost"]["seconds"] for row in history)
            or totals["train_operator_forward_gpu_seconds"] != sum(row["operator"]["operator_forward_gpu_seconds"] for row in history)):
        raise ValueError("resume totals differ from the saved phase/operator history")
    if (set(saved["rng"]) != {"python","numpy","torch","cuda"} or not saved["model"]
            or not saved["optimizer"].get("param_groups")
            or not {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"} <= set(saved["scaler"])):
        raise ValueError("resume lacks complete model, optimizer, scaler or stochastic state")
    if optimizer_contract is not None and saved["optimizer"]["param_groups"] != optimizer_contract["param_groups"]:
        raise ValueError("resume optimizer differs from the frozen parameter ordering and recipe")
    if scaler_contract is not None and any(saved["scaler"][key] != scaler_contract[key] for key in ("growth_factor","backoff_factor","growth_interval")):
        raise ValueError("resume scaler differs from the fresh default growth policy")


def load_reference(home, prepared):
    reference = guards.load_reference(home,prepared)
    counts = [sum(row["absolute_error"]<=.01 for row in reference[view]) for view in ("native","counterfactual")]
    if counts != [102,104]:
        raise ValueError("the identity reference's native102 and CF104 accurate source sets differ")
    reference["metadata"].update(native_accurate_count=102,counterfactual_accurate_count=104)
    return reference


def counterfactual_joint(reference, native, rows, summary):
    result = guards.joint_view_gate(reference["counterfactual"],rows,summary,view="counterfactual")
    diagnostic = shared.counterfactual_gate(native,rows,summary)
    for key in ("denominator","tail_count","mean_absolute_scalar_drift","tail30_absolute_scalar_drift",
                "maximum_absolute_scalar_drift","paired_error_rescues","paired_error_regressions",
                "paired_error_unchanged","paired_thresholds_percent_fs","failed_pairs"):
        result[key] = diagnostic[key]
    result.update(previous_color_control_diagnostic=diagnostic,drift_thresholds_are_acceptance=False)
    return result


def preflight(home, output):
    import cv2
    import numpy as np
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_data import _resolve_image,crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter,gradient_evidence,point_counter
    from experiments.train_geoprr_v2 import device_batch,seed_everything

    prepared = validate_prepared(home)
    reference = load_reference(home,prepared)
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("the image-structure preflight already has complete or partial outputs")
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.parent/"started.json",{"protocol":PROTOCOL,"status":"started"})
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = shared.source_populations(home,prepared)
    indices = shared.preflight_indices(prepared)
    train_batch = device_batch(default_collate([training[i] for i in indices["train"]]),device)
    validation_batches,identity_ids = [],[]
    for subset in indices["id"]:
        samples = [source_id[i] for i in subset]
        for index,sample in zip(subset,samples,strict=True):
            row = source_id.rows[index]
            image = cv2.imread(str(_resolve_image(row,source_id.base.manifest_root,source_id.base.dataset_root)))
            roi,_ = crop_and_resize(image,None if "roi_path" in row else row.get("metadata",{}).get("dial_bbox"),512)
            rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            expected = torch.from_numpy(rgb.transpose(2,0,1).copy())
            expected = (expected-expected.new_tensor(MEAN)[:,None,None])/expected.new_tensor(STD)[:,None,None]
            if sample["input_failure"] or sample["image"].shape != (3,512,512) or not torch.equal(sample["image"],expected):
                raise ValueError("the source dataset must retain the existing native512 normalized RGB input")
        validation_batches.append(device_batch(default_collate(samples),device))
        identity_ids.append([sample["sample_id"] for sample in samples])
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    original = {key:value.detach().clone() for key,value in model.state_dict().items()}
    counters = {key:value.clone() for key,value in original.items() if key.endswith("num_batches_tracked")}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key:value.cpu() for key,value in original.items()},strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key],value.cpu()) for key,value in original.items())
    del reloaded
    if ((len(counters),metadata["parameters"],metadata["trainable_parameters"],metadata["parameter_tensors"],metadata["state_tensors"],
            metadata["encoder_state_tensors"],metadata["operator_parameter_tensors"],metadata["operator_state_tensors"])
            != (77,8435793,8435793,233,465,462,0,1)
            or initialization["retained_complete_encoder_state_tensors"] != 462
            or not initialization["all_cached_encoder_state_tensors_retained"]
            or model.measurement.kernel.requires_grad or model.measurement.kernel.dtype != torch.float32):
        raise ValueError("the complete cached reader plus one fixed kernel differs from its declared state contract")
    cnn_counts,cnn_hook = encoder_counter(model)
    point_counts,point_hook = point_counter(model)
    cnn_backs,point_backs,observed_inputs,observed_pool,observed_point = [],[],[],[],[]
    def input_shape(_module,inputs):
        observed_inputs.append({"shape":list(inputs[0].shape),"dtype":str(inputs[0].dtype),"representation":"image structure measurement"})
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
    model.measurement.reset_phase()
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
            raise ValueError("all77 source BN counters must increment once in training preflight")
        model.eval()
        before_eval = {key:value.clone() for key,value in model.named_buffers()}
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
            for batch in validation_batches:
                prediction = model(batch["image"])
                if (prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all())
                        or not bool(((prediction>=0)&(prediction<=1)).all())):
                    raise ValueError("native source preflight must return finite FP32 reading scalars")
        buffers_unchanged = all(torch.equal(value,before_eval[key]) for key,value in model.named_buffers())
        parameters_unchanged = all(torch.equal(value,original[key]) for key,value in model.named_parameters())
        kernel_unchanged = torch.equal(model.measurement.kernel,original["measurement.kernel"])
    finally:
        for hook in hooks:
            hook.remove()
    operator = model.measurement.phase_report()
    validate_operator_report(operator,8,3)
    expected_inputs = [{"shape":[n,3,512,512],"dtype":"torch.float32","representation":"image structure measurement"} for n in (4,2,2)]
    expected_pool = [{"before":[n,960,16,16],"after":[n,960,1,1]} for n in (4,2,2)]
    expected_point = [{"semantic_shape":[n,1280],"point_logit_shape":[n,1],"semantic_dtype":"torch.float16","point_logit_dtype":"torch.float16"} for n in (4,2,2)]
    if (cnn_counts != {"cnn_forward_calls":3,"cnn_image_forwards":8} or point_counts != {"point_forward_calls":3,"point_image_forwards":8}
            or len(cnn_backs) != 1 or len(point_backs) != 1 or observed_inputs != expected_inputs
            or observed_pool != expected_pool or observed_point != expected_point or not strict_reload
            or not all((train_scalar,finite,all(nonzero.values()),parameters_unchanged,buffers_unchanged,kernel_unchanged,bool(torch.isfinite(loss))))):
        raise ValueError("the fixed-measurement zero-update preflight failed")
    phases["preflight"] = phase_cost(device,tick)
    write_json(output,{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "operator_contract":OPERATOR_CONTRACT,"status":"passed","counts":PREFLIGHT_COUNTS,
        "model_metadata":metadata,"initialization":initialization,"probe_sample_ids":train_batch["sample_id"],
        "source_ID_identity_sample_ids":identity_ids,"source_dataset_native_RGB_identity":True,
        "encoder_input_representation":"fixed measurement, intentionally distinct from raw normalized RGB",
        "observed_encoder_inputs":observed_inputs,"observed_pool_shapes":observed_pool,"observed_point_shapes_and_dtypes":observed_point,
        "operator":operator,"source_BN_counter_tensors_checked":77,"source_BN_counter_increment_training":1,
        "source_BN_counter_additional_increment_eval":0,"parameter_tensors_unchanged_without_optimizer":233,
        "kernel_unchanged":bool(kernel_unchanged),"strict_model_reload":strict_reload,"full_state_tensors":465,
        "all_gradients_finite":finite,"gradient_groups_nonzero":nonzero,"source_preflight_mean_smooth_l1":float(loss.detach()),
        "preflight_backward_loss_scale":128.,"reference_qualification":reference["metadata"],
        "source_OOD_RGB_reads":0,"source_OOD_forward_calls":0,"external_images_calls_backward_updates":0,
        "formal_fresh_rebuild_required":True,"runtime":runtime(device,started,phases)})


def load_evaluation(output, name, prepared, order):
    from experiments.geoprr_v2_high_resolution_direct import summarize
    paths = [output/f"{name}_{suffix}" for suffix in ("predictions.csv","groups.csv","summary.json")]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileExistsError("partial saved image-structure scalar evaluation cannot be forwarded again")
    report = json.loads(paths[2].read_text())
    if (report.get("protocol") != PROTOCOL or report.get("config") != CONFIG
            or report.get("source_contract") != prepared["source_contract"] or report.get("operator_contract") != OPERATOR_CONTRACT
            or report.get("condition_order") != list(order) or report.get("epoch") != 30):
        raise ValueError("saved scalar observation differs from the exact new last30 input/operator contract")
    rows = guards._read_scalar_rows(paths[0])
    for row in rows:
        for key in ("invalid_mass","outside_arc_mass"):
            row[key] = None if row[key] == "" else float(row[key])
    expected_ids = [str(row["sample_id"]) for row in prepared["source_contract"]["populations"]["id"]]
    calculated = summarize(rows)
    same_summary = set(calculated) == set(report["summary"]) and all(
        math.isclose(value,report["summary"][key],rel_tol=0.,abs_tol=1e-12) if isinstance(value,float)
        else value == report["summary"][key] for key,value in calculated.items())
    if len(rows) != 292 or [row["sample_id"] for row in rows] != expected_ids or not same_summary:
        raise ValueError("saved last30 scalar rows or summary differ from the complete source-ID view")
    return rows,report


def control(home, output):
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter,evaluate_population,point_counter
    from experiments.train_geoprr_v2 import device_batch,make_loader,restore_rng,rng_state,save_checkpoint,seed_everything

    prepared = validate_prepared(home)
    reference = load_reference(home,prepared)
    witness = json.loads((home/PREFLIGHT_RELATIVE).read_text())
    if (witness["status"] != "passed" or witness["protocol"] != PROTOCOL or witness["config"] != CONFIG
            or witness["source_contract"] != prepared["source_contract"] or witness["operator_contract"] != OPERATOR_CONTRACT
            or witness["counts"] != PREFLIGHT_COUNTS or not witness["source_dataset_native_RGB_identity"]):
        raise ValueError("the fresh30E control requires its own exact zero-update measurement preflight")
    latest = output/"latest.pt"
    resuming = output.exists()
    if resuming and ((output/"completed.json").exists() or not latest.is_file() or (output/"active_epoch.json").exists()):
        raise FileExistsError("partial/complete outputs cannot be overwritten; resume requires an intact complete-epoch latest")
    if not resuming:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"started.json",{"protocol":PROTOCOL,"status":"started"})
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = shared.source_populations(home,prepared)
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    if (metadata["parameters"],metadata["trainable_parameters"],metadata["parameter_tensors"],metadata["state_tensors"],
            metadata["encoder_state_tensors"],metadata["operator_parameter_tensors"],metadata["operator_state_tensors"]) != (8435793,8435793,233,465,462,0,1):
        raise ValueError("the fresh field-input model differs from its complete cached/one-kernel architecture")
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda",enabled=True)
    base = {"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],"operator_contract":OPERATOR_CONTRACT,
        "initialization":initialization,"model_metadata":metadata,"fresh_default_GradScaler":True,
        "preflight_state_reused":False,"source_unique_images":5496,"source_presentations_per_epoch":5496,
        "reference_qualification":reference["metadata"]}
    totals,history,start,prior_seconds = initial_totals(),[],0,0.
    phases = {"initialization":phase_cost(device,started)}
    if resuming:
        checkpoint = torch.load(latest,map_location="cpu",weights_only=False)
        validate_resume(checkpoint,prepared["source_contract"],optimizer.state_dict(),scaler.state_dict())
        if checkpoint["model_metadata"] != metadata or checkpoint["initialization"] != initialization:
            raise ValueError("resume initialization or actual model/input metadata differs")
        saved_history = "".join(json.dumps(row,allow_nan=False)+"\n" for row in checkpoint["history"])
        if not (output/"history.jsonl").is_file() or (output/"history.jsonl").read_text() != saved_history:
            raise ValueError("the saved history artifact differs from its full complete-epoch checkpoint")
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
    for epoch in range(start+1,31):
        write_json(output/"active_epoch.json",{"protocol":PROTOCOL,"epoch":epoch,"last_complete_epoch":epoch-1})
        torch.cuda.reset_peak_memory_stats(device)
        model.measurement.reset_phase()
        tick = time.perf_counter()
        model.train()
        loader = make_loader(training,batch_size=64,workers=4,shuffle=True,seed=20262020+epoch,device=device)
        counts,hook = encoder_counter(model)
        points,point_hook = point_counter(model)
        cnn_backs,point_backs = [],[]
        cnn_backward = model.encoder.conv_stem.weight.register_hook(lambda g:cnn_backs.append(True))
        point_backward = model.point.weight.register_hook(lambda g:point_backs.append(True))
        batches=updates=skips=0
        last_batch=None
        loss_sum=error_sum=0.
        try:
            for raw in loader:
                batch = device_batch(raw,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    prediction = model(batch["image"])
                    loss = F.smooth_l1_loss(prediction,batch["y"],beta=.05)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite source mean loss at epoch{epoch} batch{batches+1}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale()<old_scale
                last_batch = len(batch["y"])
                batches += 1
                updates += int(not skipped)
                skips += int(skipped)
                loss_sum += float(loss.detach())*last_batch
                error_sum += float((prediction.detach()-batch["y"]).abs().sum())
        finally:
            for handle in (hook,point_hook,cnn_backward,point_backward):
                handle.remove()
        operator = model.measurement.phase_report()
        validate_operator_report(operator,5496,86)
        cost = phase_cost(device,tick)
        phases[f"train_epoch_{epoch}"] = cost
        if ((batches,last_batch,len(cnn_backs),len(point_backs)) != (86,56,86,86)
                or counts != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or points != {"point_forward_calls":86,"point_image_forwards":5496}
                or updates+skips != 86 or optimizer.param_groups[0]["lr"] != .0003):
            raise RuntimeError("a30E source epoch differs from its5496-image/86-attempt schedule")
        increments = {"train_batches":batches,"train_cnn_image_forwards":5496,"train_cnn_forward_calls":86,
            "backward_calls":len(cnn_backs),"point_backward_calls":len(point_backs),"optimizer_updates":updates,
            "amp_skipped_updates":skips,"train_point_forward_calls":86,"train_point_image_forwards":5496,
            "train_fixed_filter_calls":86,"train_fixed_filter_images":5496,
            "train_operator_forward_gpu_seconds":operator["operator_forward_gpu_seconds"],"training_seconds":cost["seconds"]}
        for key,value in increments.items():
            totals[key] += value
        record = {"epoch":epoch,"learning_rate":.0003,"source_training_smooth_l1":loss_sum/5496,
            "source_training_nmae":error_sum/5496,"train_counts":counts,"train_point_counts":points,
            "CNN_backward_calls":len(cnn_backs),"point_backward_calls":len(point_backs),"optimizer_updates":updates,
            "amp_skipped_updates":skips,"last_batch_size":last_batch,"operator":operator,"cost":cost,"validation_calls":0}
        history.append(record)
        checkpoint = base | {"phase":"training","model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":epoch,"history":history,"totals":dict(totals),
            "phases":dict(phases),"checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
        save_checkpoint(latest,checkpoint)
        (output/"history.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history))
        (output/"active_epoch.json").unlink()
        print(json.dumps({"event":"epoch_complete","epoch":epoch,"CNN_images":5496,"CNN_calls":86,"point_calls":86,
            "fixed_filter_calls":86,"fixed_filter_images":5496,"CNN_backward_attempts":86,"validation_calls":0}),flush=True)

    def evaluate(name,dataset,images,calls):
        order = shared.NATIVE_ORDER if name == "source_id" else shared.COUNTERFACTUAL_ORDER
        reused = load_evaluation(output,name,prepared,order)
        if reused is None:
            torch.cuda.reset_peak_memory_stats(device)
            model.measurement.reset_phase()
            tick = time.perf_counter()
            points,hook = point_counter(model)
            try:
                rows,cost = evaluate_population(model,dataset,device)
            finally:
                hook.remove()
            operator = model.measurement.phase_report()
            validate_operator_report(operator,images,calls)
            phase = phase_cost(device,tick)
            if (len(rows) != images or cost["counts"]["cnn_image_forwards"] != images
                    or cost["counts"]["cnn_forward_calls"] != calls
                    or points != {"point_forward_calls":calls,"point_image_forwards":images}):
                raise RuntimeError("the last30 field evaluation differs from the fixed292-image/5-call view")
            cost.update(point_counts=points,phase_cost=phase,operator=operator)
            report = shared._save_evaluation(output,name,rows,cost)
            report.update(protocol=PROTOCOL,config=CONFIG,source_contract=prepared["source_contract"],operator_contract=OPERATOR_CONTRACT,
                condition_order=list(order),epoch=30)
            write_json(output/f"{name}_summary.json",report)
        else:
            rows,report = reused
            cost = report["cost"]
            validate_operator_report(cost["operator"],images,calls)
            if (cost["counts"]["cnn_image_forwards"] != images or cost["counts"]["cnn_forward_calls"] != calls
                    or cost["point_counts"] != {"point_forward_calls":calls,"point_image_forwards":images}):
                raise ValueError("reused scalar observation has a different encoder/point budget")
        phases[name] = cost["phase_cost"]
        for key,value in {"validation_images":images,"validation_cnn_forward_calls":calls,
            "validation_point_forward_calls":calls,"validation_point_image_forwards":images,
            "validation_fixed_filter_calls":calls,"validation_fixed_filter_images":images,
            "validation_operator_forward_gpu_seconds":cost["operator"]["operator_forward_gpu_seconds"],
            "validation_seconds":cost["pipeline_seconds"]}.items():
            totals[key] += value
        return rows,report

    id_rows,id_report = evaluate("source_id",shared.ChannelOrderDataset(source_id,order=shared.NATIVE_ORDER),292,5)
    learning,baseline = shared.learning_gate(id_rows)
    shared._save_evaluation(output,"constant_source_id",baseline,{"CNN_calls":0,"CNN_images":0,"fixed_filter_calls":0})
    primary = guards.joint_view_gate(reference["native"],id_rows,learning["model"],view="native")
    write_json(output/"native_learning_gate.json",learning)
    write_json(output/"native_primary.json",primary)
    eligible = bool(learning["passed"] and primary["passed"])
    cf_gate=cf_report=None
    if eligible:
        cf_rows,cf_report = evaluate("source_counterfactual",shared.ChannelOrderDataset(source_id,order=shared.COUNTERFACTUAL_ORDER),292,5)
        cf_gate = counterfactual_joint(reference,id_rows,cf_rows,cf_report["summary"])
        write_json(output/"counterfactual_gate.json",cf_gate)
    elif any(output.glob("source_counterfactual_*")):
        raise ValueError("ineligible native source condition has unexpected CF artifacts")
    validate_totals(totals,eligible)
    verdict = "IMAGE_STRUCTURE_SOURCE_CONTROL_PASSED" if eligible and cf_gate["passed"] else "NOT_RETAINED"
    final = base | {"phase":"evaluated","model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":30,"history":history,"totals":dict(totals),
        "phases":phases,"native_learning_gate":learning,"native_primary":primary,"counterfactual_gate":cf_gate,
        "checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
    save_checkpoint(output/"last.pt",final)
    write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "operator_contract":OPERATOR_CONTRACT,"status":"complete","epochs":30,
        "completion_reason":"counterfactual_observed" if eligible else "native_learning_or_primary_failed",
        "native_learning_gate":learning,"native_primary":primary,"counterfactual_eligible":eligible,"counterfactual_gate":cf_gate,
        "source_id":id_report,"source_counterfactual":cf_report,"verdict":verdict,"totals":totals,"counts":count_summary(totals),
        "model_metadata":metadata,"initialization":initialization,"checkpoint":"last.pt","reference_qualification":reference["metadata"],
        "source_training_smooth_l1_semantics":"original primitive batch-mean SmoothL1 beta=.05; no mean-tail aggregation",
        "source_dataset_native_RGB_identity":True,"encoder_input_representation":"fixed image structure measurement",
        "drift_thresholds_are_acceptance":False,"source_OOD_RGB_images":0,"source_OOD_forward_calls":0,
        "external_counts":{"images":0,"CNN_calls":0,"point_calls":0,"backward_calls":0,"optimizer_updates":0},
        "budget_qualification":CONFIG["budget_qualification"],"provided_label_scope":guards.identity.AUDIT_LABEL_SCOPE,
        "runtime":runtime(device,started,phases,prior_seconds)})


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
