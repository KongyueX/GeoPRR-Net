"""Frozen source state-correspondence supervision with a conditional cyclic control."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import resource
import signal

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_cyclic_color_control as shared
from experiments import geoprr_v2_matched_identity_control as identity
from experiments import geoprr_v2_mean_tail_control as guards
from experiments import geoprr_v2_needle_holdout_control as population


PROTOCOL = "geoprr_v2_state_change_supervision_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_state_change_supervision_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_state_change_supervision_preflight_20261026/result.json"
PREPARED_RELATIVE = shared.PREPARED_RELATIVE
PROTOCOL_RELATIVE = "refine-logs/geoprr_research_reset_20261025/state_change_supervision_source_control_protocol.json"
ARM_DIRECTORIES = {"correct":"matched_state", "cyclic":"cycled_state"}
TARGET_FORMULA = {
    "input":"unchanged native ROI / INTER_LINEAR512 / BGRtoRGB / float64 divided by255; no photo or ImageNet normalization",
    "D":"mean_c((I_s - mean_t(I_t))**2), float64",
    "pool":"sum nonoverlapping32x32 blocks to16x16, then divide by all D_s mass",
    "dtype":"float32", "shape":[5496,16,16], "payload_bytes":5627904,
    "state_mean_strict_min":(1/255)**2, "sum_atol":1e-6, "next_slot_TV_min":.05,
    "cyclic":"same(view,layout), assigned slot=(slot+1)%4; RGB/y/order unchanged",
    "physical_geometry_or_teacher":False,
}
CONFIG = deepcopy(population.CONFIG)
CONFIG.update(protocol=PROTOCOL, kind="bounded source image-derived state-correspondence control")
CONFIG["source"].update(population_packet=PREPARED_RELATIVE, OOD_images=0, OOD_calls=0,
    counterfactual_images=292, counterfactual_calls=5, target_quartets=1374)
CONFIG["training"].update(epochs=15,max_epochs=15,intermediate_validation=False,
    resume="same new protocol/config/source/aux-target and complete epoch full state only",
    loss="mean SmoothL1beta.05 + (.05/log256) mean KL(q_assigned || softmax(aux.float))")
CONFIG["model"].update(factory="experiments.geoprr_v2_state_change_supervision_control.create_model",
    parameters=8436753,parameter_tensors=234,state_tensors=465,
    auxiliary="training-only zero parameter[1,960,1,1], no bias or initialization RNG")
CONFIG["auxiliary_target"] = deepcopy(TARGET_FORMULA)
CONFIG["objective"] = {"scalar_beta":.05,"auxiliary_coefficient":.05/math.log(256),
    "KL":"mean over images of spatial sum q*(logq-logsoftmax(logits.float)); 0log0=0; float32",
    "scalar_and_auxiliary_history_separate":True}
CONFIG["checkpoint"] = "last15 only; no intermediate validation, best, scheduler or extension"
CONFIG["reference"] = {"protocol":identity.PROTOCOL,"science_revision":guards.REFERENCE_REVISION,
    "relative_reader":guards.REFERENCE_RELATIVE,"statistics":deepcopy(guards.REFERENCE_STATISTICS),
    "accurate1_native":102,"accurate1_counterfactual":104,
    "provided_label_scope":identity.AUDIT_LABEL_SCOPE}
CONFIG["acceptance"] = {
    "correct_native":"original constant0.5 half mean/tail learning gate AND identity native joint",
    "correct_CF":"identity CF joint; drift thresholds diagnostic only",
    "correspondence":"each view correct mean<=cyclic, tail<cyclic, max<=cyclic, over5<=cyclic, no new>5 on cyclic<=1%",
    "cyclic_if_correct_fails":False,"cyclic_CF_even_if_native_fails":True,
}
PREFLIGHT_COUNTS = {"CNN_calls":3,"CNN_images":8,"RGB_unique_images":8,
    "RGB_decode_calls":12,"CPU_native_identity_redecodes":4,"point_calls":3,"point_examples":8,
    "auxiliary_calls":1,"auxiliary_maps":4,"combined_backward_calls":1,
    "CNN_backward_calls":1,"point_backward_calls":1,"auxiliary_backward_calls":1,"optimizer_updates":0}
write_json = population.write_json


class SourceObservationRejected(ValueError):
    """A frozen necessary source predicate failed before any model construction."""


def require_source(condition, predicate):
    if not condition:
        raise SourceObservationRejected(predicate)


def precheck(home):
    from experiments.run_geoprr_v2_state_change_supervision_control import precheck as check
    return check(home)


def read_packet(home):
    """Read only the existing train manifest and its saved private packet contract."""
    root = home/PREPARED_RELATIVE
    prepared = json.loads((root/"prepared.json").read_text())
    if (prepared["protocol"] != population.PROTOCOL or prepared["config"] != population.CONFIG
            or prepared["status"] != "complete" or prepared["counts"] != population.EXPECTED_COUNTS
            or prepared["manifests"] != population.MANIFESTS):
        raise ValueError("the existing source packet metadata differs from its frozen contract")
    rows = population.read_rows(root/prepared["manifests"]["train"])
    if (len(rows) != 5496 or rows != prepared["source_contract"]["populations"]["train"]
            or len({row["sample_id"] for row in rows}) != 5496):
        raise ValueError("the single prepared train manifest differs from the exact saved source order")
    return prepared,rows


def quartet_indices(rows):
    groups = {}
    for index,row in enumerate(rows):
        state = row.get("metadata",{}).get("needle_control",{})
        key = (state.get("view"),state.get("layout"))
        require_source(key[0] in ("original","roll") and isinstance(key[1],str) and bool(key[1]),
                       "source_view_and_layout")
        groups.setdefault(key,[]).append(index)
    require_source(len(groups) == 1374,"quartet_count1374")
    original,rolled = {},{}
    for key,indices in groups.items():
        require_source(len(indices) == 4,"four_rows_per_quartet")
        ordered = sorted(indices,key=lambda i:rows[i]["metadata"]["needle_control"]["slot"])
        entries = [rows[i] for i in ordered]
        states = [r["metadata"]["needle_control"] for r in entries]
        require_source([s["slot"] for s in states] == [0,1,2,3]
            and len({r["sample_id"] for r in entries}) == 4 and len({r["group_id"] for r in entries}) == 1,
            "distinct_four_slots_one_group")
        require_source(len({s.get("pointer") for s in states}) == 1
            and states[0].get("pointer") in population.SEEN,"source_seen_pointer_only")
        if key[0] == "original":
            original[key[1]] = ordered
        else:
            parent = states[0].get("parent_layout")
            require_source(all(s.get("parent_layout") == parent and s.get("parent_equal") is True for s in states)
                and parent not in rolled,"roll_parent_closure")
            rolled[parent] = ordered
        groups[key] = ordered
    require_source(len(original) == len(rolled) == 687 and set(original) == set(rolled),"687_parent_closures")
    for parent,indices in rolled.items():
        for i,j in zip(original[parent],indices,strict=True):
            a,b = rows[i]["metadata"]["needle_control"],rows[j]["metadata"]["needle_control"]
            require_source(b["parent_sample_id"] == rows[i]["sample_id"]
                and all(a[k] == b[k] for k in ("slot","state","pointer")),"same_parent_slot_state")
    return list(groups.values())


def next_slot_indices(quartets, size):
    mapping = [-1]*size
    for indices in quartets:
        for slot,index in enumerate(indices):
            require_source(mapping[index] == -1,"one_target_binding_per_row")
            mapping[index] = indices[(slot+1)%4]
    require_source(all(index >= 0 for index in mapping),"complete_target_binding")
    return mapping


def q_from_state_energy(energy):
    """New target operator: exact sum-pooling and statewise mass normalization."""
    import numpy as np
    energy = np.asarray(energy,dtype=np.float64)
    require_source(energy.shape == (4,512,512),"D_shape4_512_512")
    require_source(bool(np.isfinite(energy).all() and (energy >= 0).all()),"D_finite_nonnegative")
    mass = energy.sum(axis=(1,2))
    require_source(bool((mass/(512*512) > (1/255)**2).all()),"each_state_energy_above_quantization_squared")
    pooled = energy.reshape(4,16,32,16,32).sum(axis=(2,4))
    q = (pooled/mass[:,None,None]).astype(np.float32)
    require_source(bool(np.isfinite(q).all() and (q >= 0).all()
        and (np.abs(q.sum(axis=(1,2),dtype=np.float64)-1) <= 1e-6).all()),"q_finite_nonnegative_sum1")
    tv = .5*np.abs(q.astype(np.float64)-q[[1,2,3,0]].astype(np.float64)).sum(axis=(1,2))
    require_source(bool((tv >= .05).all()),"each_next_slot_total_variation_at_least_005")
    return q,{"minimum_state_mean_energy":float((mass/(512*512)).min()),
        "minimum_next_slot_TV":float(tv.min()),"maximum_q_sum_error":float(np.abs(q.sum(axis=(1,2),dtype=np.float64)-1).max())}


def observe_quartet(raw):
    """The frozen necessary image predicates followed by the new state target."""
    import numpy as np
    raw = np.asarray(raw,dtype=np.float64)
    require_source(raw.shape == (4,512,512,3),"raw_quartet_shape")
    mean = raw.mean(axis=0)
    residual = raw-mean
    variance = np.mean(residual*residual,axis=(0,3))
    ss = float(np.sum(residual*residual))
    finite = bool(np.isfinite(raw).all() and np.isfinite(variance).all()
        and raw.min() >= 0 and raw.max() <= 1 and variance.min() >= -1e-12 and variance.max() <= .25+1e-12)
    require_source(finite,"finite_bounded_raw_and_variance")
    pairs = [float(np.sqrt(np.mean((raw[i]-raw[j])**2))) for i in range(4) for j in range(i+1,4)]
    require_source(all(v > 1/255 for v in pairs),"all_six_pairs_above_quantization_unit")
    require_source(ss > 0,"positive_total_variance")
    k = math.ceil(.1*variance.size)
    concentration = float(np.sort(variance.ravel())[-k:].sum()/variance.sum())
    require_source(concentration >= .5,"localized_top10_mass_at_least_half")
    affine_sse = 0.
    for channel in range(3):
        base = mean[:,:,channel]
        centered = base-float(base.mean())
        denominator = float(np.sum(centered*centered))
        for state in range(4):
            target = raw[state,:,:,channel]
            target_mean = float(target.mean())
            gain = float(np.sum(centered*(target-target_mean))/denominator) if denominator > 0 else 0.
            affine_sse += float(np.sum((target-(gain*centered+target_mean))**2))
    explained = 1-affine_sse/ss
    require_source(math.isfinite(explained) and explained < .5,"global_affine_explains_less_than_half")
    q,report = q_from_state_energy(np.mean(residual*residual,axis=3))
    report.update(minimum_pair_RMSE=min(pairs),pairs_above_quantization_unit=sum(v > 1/255 for v in pairs),
        total_mean_pixel_variance=float(variance.mean()),top10percent_variance_mass=concentration,
        global_affine_explained_fraction=explained)
    return q,report


def native_context(row,manifest_root,dataset_root):
    """Read lazy native headers before the bounded decoder, preserving path semantics."""
    from PIL import Image
    from experiments.geoprr_v2_data import _resolve_image
    kind = "roi_path" if "roi_path" in row else "image_path"
    path = _resolve_image(row,manifest_root,dataset_root)
    with Image.open(path) as header:
        width,height = header.size
        image_format,mode = header.format,header.mode
    require_source(image_format in ("PNG","JPEG") and mode in ("RGB","RGBA")
        and width > 0 and height > 0,"supported_native_PNG_or_JPEG_header")
    require_source(width*height <= 16777216,"native_pixel_cap16777216")
    md = row.get("metadata",{})
    require_source(md.get("image_width") == width and md.get("image_height") == height,"saved_native_dimensions_match")
    bbox = None if kind == "roi_path" else md.get("dial_bbox")
    if bbox is None:
        crop = [0,0,width,height]
    else:
        require_source(isinstance(bbox,list) and len(bbox) == 4
            and all(isinstance(v,(int,float)) and math.isfinite(v) for v in bbox),"finite_saved_native_bbox")
        crop = [max(0,math.floor(bbox[0])),max(0,math.floor(bbox[1])),min(width,math.ceil(bbox[2])),min(height,math.ceil(bbox[3]))]
    require_source(crop[2] > crop[0] and crop[3] > crop[1],"native_crop_intersection")
    return {"path":path,"kind":kind,"width":width,"height":height,"format":image_format,
        "mode":mode,"bbox":bbox,"crop":crop}


def zero_model_counts():
    return {"model_constructions":0,"CNN_calls":0,"CNN_images":0,"point_calls":0,
        "auxiliary_calls":0,"combined_backward_calls":0,"optimizer_updates":0,"GPU_calls":0,
        "ID_OOD_external_RGB_reads":0,"new_RGB_files":0}


def prepare(home,output):
    frozen = precheck(home)
    if output != home/OUTPUT_RELATIVE/"preparation":
        raise ValueError("source preparation has one frozen independent destination")
    if output.exists():
        raise FileExistsError("complete or partial source target preparation cannot be replaced")
    prepared,rows = read_packet(home)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"started.json",{"protocol":PROTOCOL,"config":CONFIG,"status":"started","counts":zero_model_counts()})
    counts = zero_model_counts() | {"source_train_rows":5496,"source_train_manifest_reads":1,
        "source_RGB_decode_calls":0,"native_header_files":0,"quartet_operator_calls":0,
        "processed_quartets":0,"accepted_quartets":0,"target_maps_written":0}
    cost = {"decode_and_native_preprocessing_seconds":0.,"target_build_seconds":0.,
        "header_and_alignment_seconds":0.,"point_forward_seconds":None,"GPU_seconds":0.}
    current = None
    def budget(*_):
        require_source(time.perf_counter()-PROCESS_STARTED < 900,"CPU_preparation_wall_cap900")
    previous = signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(SourceObservationRejected("CPU_preparation_wall_cap900")))
    signal.setitimer(signal.ITIMER_REAL,max(.001,900-(time.perf_counter()-PROCESS_STARTED)))
    try:
        import cv2
        import numpy as np
        from experiments.geoprr_v2_data import crop_and_resize
        budget()
        quartets = quartet_indices(rows)
        cycle = next_slot_indices(quartets,len(rows))
        target_contract = {"protocol":PROTOCOL,"frozen_protocol":frozen,"source_contract":prepared["source_contract"],
            "formula":TARGET_FORMULA,"manifest_relative":prepared["manifests"]["train"],
            "row_sample_ids":[r["sample_id"] for r in rows],"cyclic_row_indices":cycle,"quartet_row_indices":quartets}
        write_json(output/"target_contract.json",target_contract)
        targets = np.lib.format.open_memmap(output/"qcache.npy",mode="w+",dtype=np.float32,shape=(5496,16,16))
        extrema = {}
        for current,indices in enumerate(quartets):
            budget()
            tick = time.perf_counter()
            context = []
            try:
                for index in indices:
                    counts["native_header_files"] += 1
                    context.append(native_context(rows[index],home/PREPARED_RELATIVE/"manifests",home/"datasets"))
                signature = ("kind","width","height","format","mode","bbox","crop")
                require_source(all(all(item[k] == context[0][k] for k in signature) for item in context),"exact_quartet_input_grid_ROI_crop")
            finally:
                cost["header_and_alignment_seconds"] += time.perf_counter()-tick
            frames = []
            for index,item in zip(indices,context,strict=True):
                budget()
                require_source(counts["source_RGB_decode_calls"] < 5496,"source_decode_cap5496")
                tick = time.perf_counter()
                try:
                    counts["source_RGB_decode_calls"] += 1
                    image = cv2.imread(str(item["path"]),cv2.IMREAD_COLOR)
                    require_source(image is not None and image.dtype == np.uint8
                        and image.shape == (item["height"],item["width"],3),"native_uint8_RGB_shape_agrees")
                    roi,_ = crop_and_resize(image,item["bbox"],512)
                    frames.append(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).astype(np.float64)/255)
                finally:
                    cost["decode_and_native_preprocessing_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            try:
                counts["quartet_operator_calls"] += 1
                q,report = observe_quartet(np.stack(frames))
            finally:
                cost["target_build_seconds"] += time.perf_counter()-tick
            targets[indices] = q
            targets.flush()
            counts["processed_quartets"] += 1
            counts["accepted_quartets"] += 1
            counts["target_maps_written"] += 4
            for key,value in report.items():
                old = extrema.setdefault(key,{"minimum":value,"maximum":value})
                old.update(minimum=min(old["minimum"],value),maximum=max(old["maximum"],value))
            write_json(output/"partial.json",{"protocol":PROTOCOL,"status":"preparing","counts":counts,"cost":cost,
                "last_accepted_quartet_index":current,"target_contract_file":"target_contract.json"})
        budget()
        require_source(counts["source_RGB_decode_calls"] == 5496 and counts["accepted_quartets"] == 1374,"complete_source_target_cache")
        targets.flush()
        cost.update(full_entry_seconds=time.perf_counter()-PROCESS_STARTED,
            process_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            target_payload_bytes=int(targets.nbytes),target_cache_storage_bytes=(output/"qcache.npy").stat().st_size)
        write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"status":"complete","admitted":True,
            "verdict":"SOURCE_OBSERVATION_ADMITTED","source_contract":prepared["source_contract"],
            "target_contract":target_contract,"counts":counts,"cost":cost,"quartet_scalar_extrema":extrema,
            "source_target_scope":"image residual energy, including shadows/JPEG/other-state ghosts; no physical geometry"})
    except SourceObservationRejected as error:
        counts["processed_quartets"] = counts["accepted_quartets"]+int(current is not None)
        cost.update(full_entry_seconds=time.perf_counter()-PROCESS_STARTED,
            process_peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rejected = {"protocol":PROTOCOL,"config":CONFIG,"status":"complete","admitted":False,
            "verdict":"NOT_ADMITTED_SOURCE_OBSERVATION","failed_predicate":str(error),"first_failed_quartet_index":current,
            "source_contract":prepared["source_contract"],"counts":counts,"cost":cost,
            "preflight":None,"correct":None,"cyclic":None,"trained_model_result":None}
        write_json(output/"partial.json",rejected)
        write_json(output/"completed.json",rejected)
    except Exception as error:
        write_json(output/"partial.json",{"protocol":PROTOCOL,"status":"failed","error_class":type(error).__name__,
            "counts":counts,"cost":cost,"first_failed_quartet_index":current})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


def load_preparation(home,prepared):
    import numpy as np
    root = home/OUTPUT_RELATIVE/"preparation"
    saved = json.loads((root/"completed.json").read_text())
    contract = json.loads((root/"target_contract.json").read_text())
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["status"] != "complete"
            or not saved["admitted"] or saved["source_contract"] != prepared["source_contract"]
            or saved["target_contract"] != contract or contract["protocol"] != PROTOCOL
            or contract["frozen_protocol"] != precheck(home) or contract["formula"] != TARGET_FORMULA
            or contract["source_contract"] != prepared["source_contract"]
            or saved["counts"]["source_RGB_decode_calls"] != 5496 or saved["counts"]["accepted_quartets"] != 1374):
        raise ValueError("the full-source target cache is not admitted under the same new contract")
    rows = prepared["source_contract"]["populations"]["train"]
    if (contract["row_sample_ids"] != [r["sample_id"] for r in rows]
            or contract["quartet_row_indices"] != quartet_indices(rows)
            or contract["cyclic_row_indices"] != next_slot_indices(contract["quartet_row_indices"],5496)):
        raise ValueError("the saved source target row/state correspondence differs")
    targets = np.load(root/"qcache.npy",mmap_mode="r",allow_pickle=False)
    if (targets.dtype != np.float32 or targets.shape != (5496,16,16) or targets.nbytes != 5627904
            or not np.isfinite(targets).all() or (targets < 0).any()
            or not (np.abs(targets.sum(axis=(1,2),dtype=np.float64)-1) <= 1e-6).all()):
        raise ValueError("the saved target payload is not the complete finite float32 probability cache")
    for indices in contract["quartet_row_indices"]:
        q = targets[indices].astype(np.float64)
        if not (.5*np.abs(q-q[[1,2,3,0]]).sum(axis=(1,2)) >= .05).all():
            raise ValueError("saved cache state contrast differs from its admission contract")
    return saved,contract,targets


class StateTargetDataset:
    def __init__(self,base,targets,cycle,*,arm):
        if arm not in ARM_DIRECTORIES:
            raise ValueError("the frozen arm must be correct or cyclic")
        if len(base) != len(targets) or len(cycle) != len(base):
            raise ValueError("source/cache row correspondence has changed length")
        self.base,self.targets,self.cycle,self.arm = base,targets,cycle,arm
        self.rows = base.rows

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        assigned = index if self.arm == "correct" else self.cycle[index]
        sample = self.base[index]
        return {**sample,"q":torch.tensor(self.targets[assigned],dtype=torch.float32),
            "source_row_index":index,"assigned_target_row_index":assigned}


class AuxiliaryCost:
    def __init__(self):
        self.calls,self.maps,self.hook_seconds = 0,0,0.
        self.events = {"auxiliary_convolution":[],"KL_logsoftmax":[]}

    def start(self,name,device):
        if device.type != "cuda":
            return None
        pair = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        pair[0].record()
        self.events[name].append(pair)
        return pair[1]

    def finish(self,end):
        if end is not None:
            end.record()

    def report(self,device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"auxiliary_calls":self.calls,"auxiliary_maps":self.maps,"temporary_hook_host_seconds":self.hook_seconds,
            **{name+"_GPU_seconds":sum(a.elapsed_time(b) for a,b in pairs)/1000 if pairs else None
                for name,pairs in self.events.items()},
            "timing_scope":"CUDA events include per-call event overhead; CPU-only GPU timing unknownnull"}


class StateChangeReader(nn.Module):
    """Retain the scalar reader and expose its local pool input only during training."""
    def __init__(self,reader):
        super().__init__()
        self.reader = reader
        self.aux_weight = nn.Parameter(torch.zeros((1,960,1,1)))
        self.semantic_dim = reader.semantic_dim
        if hasattr(reader,"initialization_evidence"):
            self.initialization_evidence = deepcopy(reader.initialization_evidence)
            self.initialization_evidence.update(auxiliary_zero_initialized=True,auxiliary_initialization_RNG_draws=0,
                auxiliary_parameter_tensors=1,auxiliary_parameters=960,auxiliary_bias=False)

    @property
    def encoder(self):
        return self.reader.encoder

    @property
    def point(self):
        return self.reader.point

    def forward(self,image):
        return self.reader(image)

    def forward_with_auxiliary(self,image,cost=None):
        if not self.training:
            raise ValueError("auxiliary execution is training-only")
        captured = []
        def capture(_module,inputs):
            hook_started = time.perf_counter()
            if len(inputs) != 1 or inputs[0].shape != (len(image),960,16,16):
                raise ValueError("the original global_pool must have one960x16x16 input")
            captured.append(inputs[0])
            if cost:
                cost.hook_seconds += time.perf_counter()-hook_started
        tick = time.perf_counter()
        handle = self.encoder.global_pool.register_forward_pre_hook(capture)
        registration = time.perf_counter()-tick
        try:
            prediction = self(image)
            if len(captured) != 1:
                raise ValueError("the original scalar forward must call global_pool exactly once")
            end = cost.start("auxiliary_convolution",image.device) if cost else None
            logits = F.conv2d(captured[0],self.aux_weight)
            if cost:
                cost.finish(end)
                cost.calls += 1
                cost.maps += len(image)
            return prediction,logits,{"pool_shape":list(captured[0].shape),"pool_dtype":str(captured[0].dtype),
                "auxiliary_logits_dtype":str(logits.dtype)}
        finally:
            tick = time.perf_counter()
            handle.remove()
            if cost:
                cost.hook_seconds += registration+time.perf_counter()-tick


def create_model(*,pretrained):
    from experiments.geoprr_v2_high_resolution_direct import create_model as original_factory,model_metadata
    model = StateChangeReader(original_factory(pretrained=pretrained))
    metadata = model_metadata(model)
    if (metadata["parameters"],metadata["parameter_tensors"],metadata["state_tensors"],
            metadata["encoder_state_tensors"],metadata["point_parameters"]) != (8436753,234,465,462,1281):
        raise ValueError("the full scalar reader plus one960-weight auxiliary tensor differs")
    return model


def source_objective(prediction,target,logits,q,cost=None):
    with torch.autocast(device_type=logits.device.type,enabled=False):
        q = q.float().flatten(1)
        flat = logits.float().flatten(1)
        if (flat.shape != q.shape or flat.shape[1] != 256 or not bool(torch.isfinite(q).all())
                or not bool((q >= 0).all()) or not bool(((q.sum(1)-1).abs() <= 1e-6).all())):
            raise ValueError("the auxiliary target must be one finite normalized16x16 probability map per image")
        end = cost.start("KL_logsoftmax",logits.device) if cost else None
        log_probability = F.log_softmax(flat,dim=1)
        log_q = torch.where(q > 0,q.log(),torch.zeros_like(q))
        kl = (q*(log_q-log_probability)).sum(1).mean()
        if cost:
            cost.finish(end)
        primitive = F.smooth_l1_loss(prediction.float(),target.float(),beta=.05)
        composite = primitive+(.05/math.log(256))*kl
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value)) for value in (primitive,kl,composite)):
        raise FloatingPointError("the scalar primitive, KL and composite must all be finite float32")
    return composite,{"scalar_mean_smooth_l1":primitive,"auxiliary_mean_KL":kl,"composite":composite,
        "log_softmax_dtype":str(log_probability.dtype),"q_dtype":str(q.dtype)}


def load_reference(home,prepared):
    """Check saved scalar observations and the frozen numbers without any old forward."""
    from experiments.geoprr_v2_high_resolution_direct import summarize
    result = guards.load_reference(home,prepared)
    for view,key,accurate in (("native","source_id",102),("counterfactual","source_counterfactual",104)):
        summary = summarize(result[view])
        expected = CONFIG["reference"]["statistics"][view]
        if (summary["samples"] != 292 or sum(row["absolute_error"] <= .01 for row in result[view]) != accurate
                or any(abs(summary[k]-expected[k]) > 1e-12 for k in ("nmae","tail10_nmae","maximum_error"))
                or sum(row["absolute_error"] > .05 for row in result[view]) != expected["errors_above_5_percent_fs"]):
            raise ValueError("the saved identity15 reference does not reproduce its frozen provided-label numbers")
        saved = json.loads((home/guards.REFERENCE_RELATIVE/"completed.json").read_text())
        if saved[key]["summary"] != summary:
            raise ValueError("the identity reference saved summary differs from its complete scalar rows")
    return result


def view_joint(reference,candidate,summary,*,view,strict_tail):
    from experiments.geoprr_v2_high_resolution_direct import summarize
    paired = guards.paired_reference_guard(reference,candidate)
    base = summarize(reference)
    predicates = {"mean":summary["nmae"] <= base["nmae"],
        "tail":summary["tail10_nmae"] < base["tail10_nmae"] if strict_tail else summary["tail10_nmae"] <= base["tail10_nmae"],
        "maximum":summary["maximum_error"] <= base["maximum_error"],
        "over5":paired["paired_thresholds_percent_fs"]["5"]["candidate_errors_above"] <= paired["paired_thresholds_percent_fs"]["5"]["reference_errors_above"],
        "no_new_over5_on_reference_accurate":paired["new_over5_on_reference_accurate_count"] == 0}
    return {"passed":all(predicates.values()),"view":view,"predicates":predicates,
        "reference_summary":base,"candidate_summary":summary,"paired_reference":paired}


def conditional_stages(arm,learning,native_joint):
    if arm not in ARM_DIRECTORIES:
        raise ValueError("unknown frozen arm")
    return {"native":True,"counterfactual":arm == "cyclic" or bool(learning["passed"] and native_joint["passed"])}


def compare_arms(correct,cyclic):
    if not correct["cyclic_eligible"]:
        if cyclic is not None:
            raise ValueError("a correct-arm rejection cannot be followed by a cyclic model")
        return {"eligible":False,"passed":False,"native":None,"counterfactual":None}
    if cyclic is None:
        raise ValueError("correct native+CF admission requires the full predeclared cyclic control")
    result = {"eligible":True}
    for view,key in (("native","source_id"),("counterfactual","source_counterfactual")):
        after,before = correct[key],cyclic[key]
        if after is None or before is None or after["summary"]["samples"] != 292 or before["summary"]["samples"] != 292:
            raise ValueError("correspondence comparison requires both complete292-row views for each arm")
        result[view] = view_joint(before["rows"],after["rows"],after["summary"],view=view,strict_tail=True)
    result["passed"] = all(result[view]["passed"] for view in ("native","counterfactual"))
    return result


def initial_totals():
    totals = shared.initial_totals()
    totals.update(auxiliary_forward_calls=0,auxiliary_map_forwards=0,auxiliary_backward_calls=0,
        combined_backward_calls=0)
    return totals


def count_summary(totals):
    return shared.count_summary(totals) | {"auxiliary_forward_calls":totals["auxiliary_forward_calls"],
        "auxiliary_map_forwards":totals["auxiliary_map_forwards"],
        "auxiliary_backward_calls":totals["auxiliary_backward_calls"],
        "combined_backward_calls":totals["combined_backward_calls"]}


def validate_training_totals(totals,epoch):
    shared.validate_training_totals(totals,epoch)
    if any(totals[key] != value*epoch for key,value in {
            "auxiliary_forward_calls":86,"auxiliary_map_forwards":5496,
            "auxiliary_backward_calls":86,"combined_backward_calls":86}.items()):
        raise ValueError("auxiliary and combined backward counts differ from complete source epochs")


def validate_totals(totals,counterfactual_observed):
    validate_training_totals(totals,15)
    shared.validate_totals(totals,counterfactual_observed)


def validate_resume(saved,source_contract,target_contract,arm,model_state=None,optimizer_contract=None,scaler_contract=None):
    required = {"protocol","config","source_contract","target_contract","arm","model","optimizer","scaler","rng",
        "epoch","history","totals","initialization","model_metadata","phases","checkpointed_process_seconds","phase"}
    if not required <= set(saved):
        raise ValueError("resume requires complete new model+aux/optimizer/scaler/RNG/history/totals")
    if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["source_contract"] != source_contract
            or saved["target_contract"] != target_contract or saved["arm"] != arm or saved["phase"] != "training"
            or not 1 <= saved["epoch"] <= 15):
        raise ValueError("resume protocol/config/source/target/arm or complete-epoch phase differs")
    epoch,history,totals = saved["epoch"],saved["history"],saved["totals"]
    if [row["epoch"] for row in history] != list(range(1,epoch+1)):
        raise ValueError("resume history must contain every completed epoch exactly once")
    for row in history:
        if (row["train_counts"] != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or row["train_point_counts"] != {"point_forward_calls":86,"point_image_forwards":5496}
                or (row["CNN_backward_calls"],row["point_backward_calls"],row["auxiliary_backward_calls"],row["combined_backward_calls"]) != (86,86,86,86)
                or (row["auxiliary_cost"]["auxiliary_calls"],row["auxiliary_cost"]["auxiliary_maps"]) != (86,5496)
                or row["optimizer_updates"]+row["amp_skipped_updates"] != 86 or row["last_batch_size"] != 56
                or row["validation_calls"] != 0 or row["learning_rate"] != .0003
                or row.get("all_source_rows_once") is not True or row.get("assigned_targets_match_arm") is not True
                or any(not math.isfinite(row[key]) for key in ("source_training_scalar_mean_smooth_l1","source_training_auxiliary_mean_KL","source_training_composite"))):
            raise ValueError("resume history differs from the frozen source/auxiliary exposure and loss fields")
    validate_training_totals(totals,epoch)
    if (any(totals[key] != 0 for key in ("validation_images","validation_cnn_forward_calls","validation_point_forward_calls","validation_point_image_forwards"))
            or totals["optimizer_updates"] != sum(row["optimizer_updates"] for row in history)
            or totals["amp_skipped_updates"] != sum(row["amp_skipped_updates"] for row in history)
            or totals["training_seconds"] != sum(row["cost"]["seconds"] for row in history)):
        raise ValueError("resume totals differ from its complete saved history")
    if (set(saved["rng"]) != {"python","numpy","torch","cuda"}
            or not {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"} <= set(saved["scaler"])
            or not saved["optimizer"].get("param_groups") or not saved["model"]):
        raise ValueError("resume lacks the full stochastic, default AMP or optimizer state")
    if model_state is not None:
        if (set(saved["model"]) != set(model_state)
                or any(saved["model"][key].shape != value.shape or saved["model"][key].dtype != value.dtype for key,value in model_state.items())):
            raise ValueError("resume model+auxiliary full state shape or dtype differs")
    if "aux_weight" not in saved["model"] or tuple(saved["model"]["aux_weight"].shape) != (1,960,1,1):
        raise ValueError("resume requires the new single auxiliary weight; old464 states are incompatible")
    if optimizer_contract is not None and saved["optimizer"]["param_groups"] != optimizer_contract["param_groups"]:
        raise ValueError("resume fixed optimizer parameters or hyperparameters differ")
    if scaler_contract is not None and any(saved["scaler"][key] != scaler_contract[key] for key in ("growth_factor","backoff_factor","growth_interval")):
        raise ValueError("resume GradScaler default growth policy differs")


def runtime(device,started,phases,prior_seconds=0.):
    result = shared.runtime(device,started,phases,prior_seconds)
    result.update(full_process_seconds=prior_seconds+time.perf_counter()-PROCESS_STARTED,
        full_process_boundary="new module entry before torch import through saved result metadata; excludes final JSON and downtime",
        individual_point_forward_seconds=None,CPU_decode_seconds=None)
    return result


def load_evaluation(output,name,prepared,target_contract,arm,order):
    from experiments.geoprr_v2_high_resolution_direct import summarize
    paths = [output/f"{name}_{suffix}" for suffix in ("predictions.csv","groups.csv","summary.json")]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileExistsError("partial scalar CSV or missing summary cannot authorize a second forward")
    report = json.loads(paths[2].read_text())
    if (report.get("protocol") != PROTOCOL or report.get("config") != CONFIG
            or report.get("source_contract") != prepared["source_contract"] or report.get("target_contract") != target_contract
            or report.get("arm") != arm or report.get("condition_order") != list(order) or report.get("epoch") != 15):
        raise ValueError("saved scalar observation protocol/config/data/target/arm/last15 differs")
    rows = guards._read_scalar_rows(paths[0])
    expected = prepared["source_contract"]["populations"]["id"]
    if (len(rows) != 292 or [row["sample_id"] for row in rows] != [str(row["sample_id"]) for row in expected]
            or [row["group_id"] for row in rows] != [str(row["group_id"]) for row in expected]
            or report["summary"] != summarize(rows)):
        raise ValueError("saved complete observation rows do not reproduce the exact source-ID summary")
    report["rows"] = rows
    return rows,report


def control(home,output,*,arm,newcpu_passed=False):
    precheck(home)
    if not newcpu_passed:
        raise ValueError("necessary new synthetic CPU checks must pass before model execution")
    if arm not in ARM_DIRECTORIES or output != home/OUTPUT_RELATIVE/ARM_DIRECTORIES[arm]:
        raise ValueError("the arm has one frozen independent destination")
    prepared,train_rows = read_packet(home)
    _,target_contract,targets = load_preparation(home,prepared)
    reference = load_reference(home,prepared)
    witness = json.loads((home/PREFLIGHT_RELATIVE).read_text())
    if (witness["protocol"] != PROTOCOL or witness["config"] != CONFIG or witness["status"] != "passed"
            or witness["source_contract"] != prepared["source_contract"] or witness["target_contract"] != target_contract
            or witness["counts"] != PREFLIGHT_COUNTS or not witness["necessary_NEWCPU_passed"]):
        raise ValueError("formal training requires its same complete new zero-update preflight")
    if arm == "cyclic":
        correct = json.loads((home/OUTPUT_RELATIVE/ARM_DIRECTORIES["correct"]/"completed.json").read_text())
        if (correct["protocol"] != PROTOCOL or correct["config"] != CONFIG or correct["source_contract"] != prepared["source_contract"]
                or correct["target_contract"] != target_contract or correct["arm"] != "correct" or not correct["cyclic_eligible"]
                or correct["status"] != "complete" or correct["epochs"] != 15
                or not all(correct[k]["passed"] for k in ("native_learning_gate","native_joint","counterfactual_joint"))):
            raise ValueError("fresh cyclic training requires completed correct native+CF joint admission")
    latest = output/"latest.pt"
    resuming = output.exists()
    if resuming and ((output/"completed.json").exists() or not latest.is_file() or (output/"active_epoch.json").exists()):
        raise FileExistsError("complete or partial arm output cannot be overwritten; only intact complete-epoch latest can resume")
    if not resuming:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/"started.json",{"protocol":PROTOCOL,"config":CONFIG,"arm":arm,"status":"started"})
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter,evaluate_population,model_metadata,point_counter,synchronize
    from experiments.geoprr_v2_head_warmstart_control import phase_cost
    from experiments.train_geoprr_v2 import device_batch,make_loader,restore_rng,rng_state,save_checkpoint,seed_everything
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = population.source_populations(home,prepared)
    if training.rows != train_rows or source_id.rows != prepared["source_contract"]["populations"]["id"]:
        raise ValueError("formal source populations differ from the frozen packet")
    dataset = StateTargetDataset(training,targets,target_contract["cyclic_row_indices"],arm=arm)
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    scaler = torch.amp.GradScaler("cuda",enabled=True)
    base = {"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "target_contract":target_contract,"arm":arm,"initialization":initialization,"model_metadata":metadata,
        "fresh_default_GradScaler":True,"preflight_state_reused":False,"source_unique_images":5496,
        "source_presentations_per_epoch":5496,"auxiliary_assignment":"same_state" if arm == "correct" else "next_slot_same_quartet"}
    totals,history,start,prior_seconds = initial_totals(),[],0,0.
    phases = {"initialization_and_cache_loading":phase_cost(device,started)}
    if resuming:
        checkpoint = torch.load(latest,map_location="cpu",weights_only=False)
        validate_resume(checkpoint,prepared["source_contract"],target_contract,arm,model.state_dict(),optimizer.state_dict(),scaler.state_dict())
        run = json.loads((output/"run.json").read_text())
        if run != base:
            raise ValueError("resume run metadata differs from the same architecture/config/source/target/arm")
        serialized = "".join(json.dumps(row,allow_nan=False)+"\n" for row in checkpoint["history"])
        if not (output/"history.jsonl").is_file() or (output/"history.jsonl").read_text() != serialized:
            raise ValueError("resume history artifact differs from the complete-epoch checkpoint")
        model.load_state_dict(checkpoint["model"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        totals,history,start = checkpoint["totals"],checkpoint["history"],checkpoint["epoch"]
        prior_seconds,phases = checkpoint["checkpointed_process_seconds"],checkpoint["phases"]
        phases[f"resume_initialization_after_epoch_{start}"] = phase_cost(device,started)
    else:
        write_json(output/"run.json",base)
    for epoch in range(start+1,16):
        write_json(output/"active_epoch.json",{"protocol":PROTOCOL,"arm":arm,"epoch":epoch,"last_complete_epoch":epoch-1})
        torch.cuda.reset_peak_memory_stats(device)
        tick = time.perf_counter()
        model.train()
        loader = make_loader(dataset,batch_size=64,workers=4,shuffle=True,seed=20262020+epoch,device=device)
        cnn,cnn_hook = encoder_counter(model)
        point,point_hook = point_counter(model)
        cnn_backs,point_backs,aux_backs = [],[],[]
        hooks = [cnn_hook,point_hook,
            model.encoder.conv_stem.weight.register_hook(lambda g:cnn_backs.append(True)),
            model.point.weight.register_hook(lambda g:point_backs.append(True)),
            model.aux_weight.register_hook(lambda g:aux_backs.append(True))]
        aux_cost = AuxiliaryCost()
        batches = updates = skips = 0
        last_batch = None
        scalar_sum = kl_sum = composite_sum = error_sum = 0.
        source_indices,target_indices = [],[]
        try:
            for raw in loader:
                source_indices.extend(raw["source_row_index"].tolist())
                target_indices.extend(raw["assigned_target_row_index"].tolist())
                batch = device_batch(raw,device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16):
                    prediction,logits,_ = model.forward_with_auxiliary(batch["image"],aux_cost)
                    loss,losses = source_objective(prediction,batch["y"],logits,batch["q"],aux_cost)
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
                scalar_sum += float(losses["scalar_mean_smooth_l1"].detach())*last_batch
                kl_sum += float(losses["auxiliary_mean_KL"].detach())*last_batch
                composite_sum += float(loss.detach())*last_batch
                error_sum += float((prediction.detach()-batch["y"]).abs().sum())
        finally:
            for hook in hooks:
                hook.remove()
        synchronize(device)
        cost = phase_cost(device,tick)
        aux_report = aux_cost.report(device)
        phases[f"train_epoch_{epoch}"] = cost
        expected_targets = [i if arm == "correct" else target_contract["cyclic_row_indices"][i] for i in source_indices]
        if ((batches,last_batch,len(cnn_backs),len(point_backs),len(aux_backs),aux_cost.calls,aux_cost.maps) != (86,56,86,86,86,86,5496)
                or cnn != {"cnn_forward_calls":86,"cnn_image_forwards":5496}
                or point != {"point_forward_calls":86,"point_image_forwards":5496}
                or sorted(source_indices) != list(range(5496)) or target_indices != expected_targets
                or updates+skips != 86 or optimizer.param_groups[0]["lr"] != .0003):
            raise RuntimeError("an epoch differs from fixed5496 RGB/scalar rows, state binding and86 combined attempts")
        increments = {"train_batches":86,"train_cnn_image_forwards":5496,"train_cnn_forward_calls":86,
            "backward_calls":86,"point_backward_calls":86,"auxiliary_backward_calls":86,"combined_backward_calls":86,
            "optimizer_updates":updates,"amp_skipped_updates":skips,"train_point_forward_calls":86,
            "train_point_image_forwards":5496,"auxiliary_forward_calls":86,"auxiliary_map_forwards":5496,
            "training_seconds":cost["seconds"]}
        for key,value in increments.items():
            totals[key] += value
        record = {"epoch":epoch,"learning_rate":.0003,
            "source_training_scalar_mean_smooth_l1":scalar_sum/5496,"source_training_auxiliary_mean_KL":kl_sum/5496,
            "source_training_composite":composite_sum/5496,"source_training_nmae":error_sum/5496,
            "train_counts":cnn,"train_point_counts":point,"CNN_backward_calls":86,"point_backward_calls":86,
            "auxiliary_backward_calls":86,"combined_backward_calls":86,"optimizer_updates":updates,"amp_skipped_updates":skips,
            "last_batch_size":56,"cost":cost,"auxiliary_cost":aux_report,"validation_calls":0,
            "all_source_rows_once":True,"assigned_targets_match_arm":True,"CPU_decode_seconds":None}
        history.append(record)
        checkpoint = base | {"phase":"training","model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scaler":scaler.state_dict(),"rng":rng_state(),"epoch":epoch,"history":history,"totals":dict(totals),
            "phases":dict(phases),"checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
        save_checkpoint(latest,checkpoint)
        (output/"history.jsonl").write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in history))
        (output/"active_epoch.json").unlink()
        print(json.dumps({"epoch":epoch,"CNN_images":5496,"CNN_calls":86,"auxiliary_calls":86,
            "combined_backward_attempts":86,"optimizer_updates":updates,"amp_skips":skips,"validation_calls":0}),flush=True)

    def evaluate(name,order):
        reused = load_evaluation(output,name,prepared,target_contract,arm,order)
        if reused is None:
            torch.cuda.reset_peak_memory_stats(device)
            tick = time.perf_counter()
            point,hook = point_counter(model)
            try:
                values,cost = evaluate_population(model,shared.ChannelOrderDataset(source_id,order=order),device)
            finally:
                hook.remove()
            phase = phase_cost(device,tick)
            if (len(values) != 292 or cost["counts"]["cnn_image_forwards"] != 292 or cost["counts"]["cnn_forward_calls"] != 5
                    or point != {"point_forward_calls":5,"point_image_forwards":292}):
                raise RuntimeError("last15 scalar evaluation changed the complete292/5 denominator")
            cost.update(point_counts=point,phase_cost=phase,auxiliary_calls=0,auxiliary_maps=0,CPU_decode_seconds=None)
            report = population._save_evaluation(output,name,values,cost)
            report.update(protocol=PROTOCOL,config=CONFIG,source_contract=prepared["source_contract"],target_contract=target_contract,
                arm=arm,condition_order=list(order),epoch=15)
            write_json(output/f"{name}_summary.json",report)
            report["rows"] = values
        else:
            values,report = reused
            cost = report["cost"]
            if (cost["counts"]["cnn_image_forwards"] != 292 or cost["counts"]["cnn_forward_calls"] != 5
                    or cost["point_counts"] != {"point_forward_calls":5,"point_image_forwards":292}
                    or cost["auxiliary_calls"] != 0 or cost["auxiliary_maps"] != 0):
                raise ValueError("saved evaluation cost differs from its frozen scalar-only budget")
        phases[name] = cost["phase_cost"]
        for key,value in {"validation_images":292,"validation_cnn_forward_calls":5,
                "validation_point_forward_calls":5,"validation_point_image_forwards":292,
                "validation_seconds":cost["pipeline_seconds"]}.items():
            totals[key] += value
        return values,report

    native_rows,native_report = evaluate("source_id",shared.NATIVE_ORDER)
    learning,baseline = population.learning_gate(native_rows)
    population._save_evaluation(output,"constant_source_id",baseline,{"CNN_calls":0,"CNN_images":0})
    native_joint = view_joint(reference["native"],native_rows,native_report["summary"],view="native",strict_tail=True)
    write_json(output/"native_learning_gate.json",learning)
    write_json(output/"native_joint.json",native_joint)
    stages = conditional_stages(arm,learning,native_joint)
    cf_report,cf_joint,drift = None,None,None
    if stages["counterfactual"]:
        cf_rows,cf_report = evaluate("source_counterfactual",shared.COUNTERFACTUAL_ORDER)
        cf_joint = view_joint(reference["counterfactual"],cf_rows,cf_report["summary"],view="counterfactual",strict_tail=False)
        drift = shared.counterfactual_gate(native_rows,cf_rows,cf_report["summary"])
        drift.update(acceptance=False,role="old thresholds diagnostic only")
        write_json(output/"counterfactual_joint.json",cf_joint)
        write_json(output/"counterfactual_drift_diagnostic.json",drift)
    elif any(output.glob("source_counterfactual_*")):
        raise ValueError("correct native rejection has unexpected counterfactual artifacts")
    validate_totals(totals,stages["counterfactual"])
    cyclic_eligible = bool(arm == "correct" and learning["passed"] and native_joint["passed"] and cf_joint is not None and cf_joint["passed"])
    final = base | {"phase":"evaluated","model":model.state_dict(),"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
        "rng":rng_state(),"epoch":15,"history":history,"totals":dict(totals),"phases":dict(phases),
        "native_learning_gate":learning,"native_joint":native_joint,"counterfactual_joint":cf_joint,
        "checkpointed_process_seconds":prior_seconds+time.perf_counter()-PROCESS_STARTED}
    save_checkpoint(output/"last.pt",final)
    write_json(output/"completed.json",base | {"status":"complete","epochs":15,
        "completion_reason":"counterfactual_observed" if stages["counterfactual"] else "correct_native_learning_or_joint_failed",
        "native_learning_gate":learning,"native_joint":native_joint,"counterfactual_joint":cf_joint,
        "counterfactual_eligible":stages["counterfactual"],"cyclic_eligible":cyclic_eligible,
        "counterfactual_drift_diagnostic":drift,"source_id":native_report,"source_counterfactual":cf_report,
        "verdict":"CORRECT_IDENTITY_JOINT_PASSED" if cyclic_eligible else "NOT_RETAINED",
        "totals":totals,"counts":count_summary(totals),"checkpoint":"last.pt",
        "runtime":runtime(device,started,phases,prior_seconds),"reference_qualification":reference["metadata"]})


def preflight(home,output,*,newcpu_passed=False):
    precheck(home)
    if not newcpu_passed:
        raise ValueError("the necessary new synthetic CPU checks must pass before model execution")
    prepared,train_rows = read_packet(home)
    prep,target_contract,targets = load_preparation(home,prepared)
    reference = load_reference(home,prepared)
    if output != home/PREFLIGHT_RELATIVE:
        raise ValueError("the source-only preflight has one frozen destination")
    if output.parent.exists() and any(output.parent.iterdir()):
        raise FileExistsError("the new zero-update preflight has complete or partial output")
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.parent/"started.json",{"protocol":PROTOCOL,"config":CONFIG,"status":"started"})
    import cv2
    import numpy as np
    from torch.utils.data import default_collate
    from experiments.geoprr_v2_data import _resolve_image,crop_and_resize
    from experiments.geoprr_v2_high_resolution_direct import encoder_counter,gradient_evidence,model_metadata,point_counter
    from experiments.train_geoprr_v2 import device_batch,seed_everything
    from experiments.geoprr_v2_head_warmstart_control import phase_cost
    started,device = time.perf_counter(),torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(20262020)
    training,source_id = population.source_populations(home,prepared)
    if training.rows != train_rows or source_id.rows != prepared["source_contract"]["populations"]["id"]:
        raise ValueError("preflight source population order differs")
    train_indices = [i for i,row in enumerate(train_rows)
        if row["metadata"]["needle_control"]["view"] == "original"
        and row["metadata"]["needle_control"]["layout"] == "geoprr_compositional_v1_g0129"]
    train_indices.sort(key=lambda i:train_rows[i]["metadata"]["needle_control"]["slot"])
    if len(train_indices) != 4 or [train_rows[i]["metadata"]["needle_control"]["slot"] for i in train_indices] != [0,1,2,3]:
        raise ValueError("the preflight must use the previously fixed g0129 original four-state quartet")
    dataset = StateTargetDataset(training,targets,target_contract["cyclic_row_indices"],arm="correct")
    samples = [dataset[i] for i in train_indices]
    train_batch = device_batch(default_collate(samples),device)
    cycle_indices = [target_contract["cyclic_row_indices"][i] for i in train_indices]
    if (cycle_indices != train_indices[1:]+train_indices[:1]
            or any(np.array_equal(targets[i],targets[j]) for i,j in zip(train_indices,cycle_indices,strict=True))):
        raise ValueError("the admitted preflight targets must bind distinct correct and next-state cache rows")
    validation_batches,identity_ids = [],[]
    for subset in shared.preflight_indices(prepared)["id"]:
        values = [source_id[i] for i in subset]
        for index,sample in zip(subset,values,strict=True):
            row = source_id.rows[index]
            image = cv2.imread(str(_resolve_image(row,source_id.base.manifest_root,source_id.base.dataset_root)))
            roi,_ = crop_and_resize(image,None if "roi_path" in row else row.get("metadata",{}).get("dial_bbox"),512)
            rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            expected = torch.from_numpy(rgb.transpose(2,0,1).copy())
            expected = (expected-expected.new_tensor(shared.MEAN)[:,None,None])/expected.new_tensor(shared.STD)[:,None,None]
            if sample["input_failure"] or not torch.equal(sample["image"],expected):
                raise ValueError("source-ID native input differs from the retained512 RGB pipeline")
        validation_batches.append(device_batch(default_collate(values),device))
        identity_ids.append([sample["sample_id"] for sample in values])
    model = create_model(pretrained=True).to(device)
    metadata,initialization = model_metadata(model),model.initialization_evidence
    original = {key:value.detach().clone() for key,value in model.state_dict().items()}
    counters = {key:value.clone() for key,value in original.items() if key.endswith("num_batches_tracked")}
    reloaded = create_model(pretrained=False)
    reloaded.load_state_dict({key:value.cpu() for key,value in original.items()},strict=True)
    strict_reload = all(torch.equal(reloaded.state_dict()[key],value.cpu()) for key,value in original.items())
    del reloaded
    if len(counters) != 77 or not bool((model.aux_weight == 0).all()):
        raise ValueError("the original77 BN states and single zero auxiliary weight differ")
    cnn,cnn_hook = encoder_counter(model)
    point,point_hook = point_counter(model)
    cnn_backs,point_backs,aux_backs,inputs,pools,points = [],[],[],[],[],[]
    hooks = [cnn_hook,point_hook,
        model.encoder.conv_stem.weight.register_hook(lambda g:cnn_backs.append(True)),
        model.point.weight.register_hook(lambda g:point_backs.append(True)),
        model.aux_weight.register_hook(lambda g:aux_backs.append(True)),
        model.encoder.register_forward_pre_hook(lambda m,x:inputs.append({"shape":list(x[0].shape),"dtype":str(x[0].dtype)})),
        model.encoder.global_pool.register_forward_hook(lambda m,x,y:pools.append({"before":list(x[0].shape),"after":list(y.shape),"dtype":str(x[0].dtype)})),
        model.point.register_forward_hook(lambda m,x,y:points.append({"semantic_shape":list(x[0].shape),"point_logit_shape":list(y.shape),
            "semantic_dtype":str(x[0].dtype),"point_logit_dtype":str(y.dtype)}))]
    phases = {"initialization_and_CPU_inputs":phase_cost(device,started)}
    aux_cost = AuxiliaryCost()
    tick = time.perf_counter()
    try:
        model.train()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            prediction,logits,aux_witness = model.forward_with_auxiliary(train_batch["image"],aux_cost)
            loss,losses = source_objective(prediction,train_batch["y"],logits,train_batch["q"],aux_cost)
        scalar_ok = prediction.shape == (4,) and prediction.dtype == torch.float32 and bool(torch.isfinite(prediction).all())
        (loss*128.).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(128.)
        finite,nonzero = gradient_evidence(model)
        nonzero["auxiliary"] = model.aux_weight.grad is not None and bool(model.aux_weight.grad.abs().sum() > 0)
        if not all(torch.equal(model.state_dict()[key],value+1) for key,value in counters.items()):
            raise ValueError("each retained BN counter must advance exactly once in training")
        model.eval()
        before_eval = {key:value.clone() for key,value in model.named_buffers()}
        with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.float16):
            for batch in validation_batches:
                prediction = model(batch["image"])
                if prediction.shape != (2,) or prediction.dtype != torch.float32 or not bool(torch.isfinite(prediction).all()) or not bool(((prediction >= 0)&(prediction <= 1)).all()):
                    raise ValueError("the source-ID eval scalar interface differs")
        parameters_unchanged = all(torch.equal(value,original[key]) for key,value in model.named_parameters())
        buffers_unchanged = all(torch.equal(value,before_eval[key]) for key,value in model.named_buffers())
    finally:
        for hook in hooks:
            hook.remove()
    expected_inputs = [{"shape":[n,3,512,512],"dtype":"torch.float32"} for n in (4,2,2)]
    expected_pools = [{"before":[n,960,16,16],"after":[n,960,1,1],"dtype":"torch.float16"} for n in (4,2,2)]
    expected_points = [{"semantic_shape":[n,1280],"point_logit_shape":[n,1],"semantic_dtype":"torch.float16","point_logit_dtype":"torch.float16"} for n in (4,2,2)]
    if (cnn != {"cnn_forward_calls":3,"cnn_image_forwards":8} or point != {"point_forward_calls":3,"point_image_forwards":8}
            or (len(cnn_backs),len(point_backs),len(aux_backs),aux_cost.calls,aux_cost.maps) != (1,1,1,1,4)
            or inputs != expected_inputs or pools != expected_pools or points != expected_points
            or aux_witness["pool_shape"] != [4,960,16,16] or aux_witness["pool_dtype"] != "torch.float16"
            or aux_witness["auxiliary_logits_dtype"] != "torch.float16"
            or not all((strict_reload,scalar_ok,finite,all(nonzero.values()),parameters_unchanged,buffers_unchanged))):
        raise ValueError("the one zero-update source-only auxiliary preflight failed")
    phases["one_training_two_eval_batches"] = phase_cost(device,tick)
    write_json(output,{"protocol":PROTOCOL,"config":CONFIG,"source_contract":prepared["source_contract"],
        "target_contract":target_contract,"status":"passed","counts":PREFLIGHT_COUNTS,
        "necessary_NEWCPU_passed":True,"model_metadata":metadata,"initialization":initialization,
        "source_ID_pipeline_identity":True,"probe_sample_ids":train_batch["sample_id"],"source_ID_identity_sample_ids":identity_ids,
        "correct_cache_rows":train_indices,"cyclic_cache_rows":cycle_indices,"cache_targets_pairwise_different":True,
        "observed_encoder_inputs":inputs,"observed_pool_shapes":pools,"observed_point_shapes_and_dtypes":points,
        "source_BN_counter_tensors_checked":77,"source_BN_counter_increment_training":1,"source_BN_counter_additional_increment_eval":0,
        "parameter_tensors_unchanged_without_optimizer":234,"strict_model_reload":strict_reload,"full_state_tensors":465,
        "zero_auxiliary_unchanged_without_optimizer":bool((model.aux_weight == 0).all()),
        "all_gradients_finite":finite,"gradient_groups_nonzero":nonzero,
        "scalar_mean_smooth_l1":float(losses["scalar_mean_smooth_l1"].detach()),
        "auxiliary_mean_KL":float(losses["auxiliary_mean_KL"].detach()),"composite":float(loss.detach()),
        "log_softmax_dtype":losses["log_softmax_dtype"],"q_dtype":losses["q_dtype"],
        "preflight_backward_loss_scale":128.,"auxiliary_cost":aux_cost.report(device),
        "formal_fresh_rebuild_required":True,"preparation_counts_separate":prep["counts"],
        "reference_qualification":reference["metadata"],"runtime":runtime(device,started,phases)})


def finalize(home,output):
    precheck(home)
    if output != home/OUTPUT_RELATIVE or (output/"completed.json").exists():
        raise FileExistsError("whole-control completion has one new destination and cannot be replaced")
    preparation = json.loads((output/"preparation"/"completed.json").read_text())
    if preparation["protocol"] != PROTOCOL or preparation["config"] != CONFIG or preparation["status"] != "complete":
        raise ValueError("whole completion requires the same complete preparation result")
    empty = count_summary(initial_totals())
    if not preparation["admitted"]:
        if preparation["verdict"] != "NOT_ADMITTED_SOURCE_OBSERVATION":
            raise ValueError("an unadmitted source result must retain its scientific stopping verdict")
        if any((output/directory).exists() for directory in ARM_DIRECTORIES.values()) or (home/PREFLIGHT_RELATIVE).parent.exists():
            raise ValueError("source rejection cannot include preflight or either model arm")
        write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"status":"complete",
            "source_contract":preparation["source_contract"],"verdict":"NOT_ADMITTED_SOURCE_OBSERVATION",
            "preparation":preparation,"preflight":None,"correct":None,"cyclic":None,
            "trained_model_result":None,"comparison":None,"counts":empty,"correct_counts":empty,"cyclic_counts":empty,
            "preflight_counts_separate":zero_model_counts(),"model_constructions":0,
            "independent_nonauthor_saved_number_CPU_audit_required":True,
            "audit_requires_all_actual_parents_gone_and_GPU_empty":True,
            "runtime":{"full_entry_seconds":time.perf_counter()-PROCESS_STARTED}})
        return
    prepared,_ = read_packet(home)
    _,target_contract,_ = load_preparation(home,prepared)
    reference = load_reference(home,prepared)
    pref = json.loads((home/PREFLIGHT_RELATIVE).read_text())
    if (pref["protocol"] != PROTOCOL or pref["config"] != CONFIG or pref["status"] != "passed"
            or pref["source_contract"] != prepared["source_contract"] or pref["target_contract"] != target_contract
            or pref["counts"] != PREFLIGHT_COUNTS):
        raise ValueError("whole completion requires the one same source-only zero-update preflight")
    def load_arm(arm):
        directory = output/ARM_DIRECTORIES[arm]
        saved = json.loads((directory/"completed.json").read_text())
        if (saved["protocol"] != PROTOCOL or saved["config"] != CONFIG or saved["status"] != "complete"
                or saved["arm"] != arm or saved["epochs"] != 15 or saved["source_contract"] != prepared["source_contract"]
                or saved["target_contract"] != target_contract):
            raise ValueError("completed arm changed protocol/config/source/targets or fixed budget")
        rows,report = load_evaluation(directory,"source_id",prepared,target_contract,arm,shared.NATIVE_ORDER)
        learning,_ = population.learning_gate(rows)
        native_joint = view_joint(reference["native"],rows,report["summary"],view="native",strict_tail=True)
        if saved["native_learning_gate"] != learning or saved["native_joint"] != native_joint:
            raise ValueError("saved native gates differ from the same complete scalar observations")
        cf_expected = conditional_stages(arm,learning,native_joint)["counterfactual"]
        if saved["counterfactual_eligible"] is not cf_expected:
            raise ValueError("saved counterfactual stage differs from its conditional contract")
        saved["source_id"] = report
        if cf_expected:
            cf_rows,cf = load_evaluation(directory,"source_counterfactual",prepared,target_contract,arm,shared.COUNTERFACTUAL_ORDER)
            joint = view_joint(reference["counterfactual"],cf_rows,cf["summary"],view="counterfactual",strict_tail=False)
            if saved["counterfactual_joint"] != joint:
                raise ValueError("saved CF joint differs from its complete scalar observation")
            saved["source_counterfactual"] = cf
        elif saved["source_counterfactual"] is not None or saved["counterfactual_joint"] is not None:
            raise ValueError("an unrun correct CF must retain null metrics")
        expected_cyclic = bool(arm == "correct" and learning["passed"] and native_joint["passed"]
            and cf_expected and saved["counterfactual_joint"]["passed"])
        if saved["cyclic_eligible"] is not expected_cyclic:
            raise ValueError("cyclic admission differs from correct native+CF joint")
        validate_totals(saved["totals"],cf_expected)
        if saved["counts"] != count_summary(saved["totals"]):
            raise ValueError("completed arm counts differ from its saved totals")
        return saved
    correct = load_arm("correct")
    cyclic = load_arm("cyclic") if correct["cyclic_eligible"] else None
    if cyclic is None and (output/ARM_DIRECTORIES["cyclic"]).exists():
        raise ValueError("correct rejection cannot include any cyclic partial or complete output")
    comparison = compare_arms(correct,cyclic)
    verdict = "STATE_CHANGE_SOURCE_CONTROL_PASSED" if comparison["passed"] else "NOT_RETAINED"
    combined = {key:value+(cyclic["counts"][key] if cyclic else 0) for key,value in correct["counts"].items()}
    if cyclic is not None and (combined["cnn_image_forwards"],combined["cnn_forward_calls"],combined["point_image_forwards"],
            combined["point_forward_calls"],combined["auxiliary_map_forwards"],combined["auxiliary_forward_calls"],
            combined["combined_backward_calls"]) != (166048,2600,166048,2600,164880,2580,2580):
        raise ValueError("two-arm completion differs from the predeclared source compute budget")
    write_json(output/"completed.json",{"protocol":PROTOCOL,"config":CONFIG,"status":"complete",
        "source_contract":prepared["source_contract"],"target_contract":target_contract,"verdict":verdict,
        "preparation":preparation,"preflight":pref,"correct":correct,"cyclic":cyclic,"comparison":comparison,
        "counts":combined,"correct_counts":correct["counts"],"cyclic_counts":empty if cyclic is None else cyclic["counts"],
        "preflight_counts_separate":pref["counts"],"model_arms_executed":1+int(cyclic is not None),
        "cyclic_not_run_metrics_are_null":cyclic is None,"independent_nonauthor_saved_number_CPU_audit_required":True,
        "audit_requires_all_actual_parents_gone_and_GPU_empty":True,
        "qualifications":[identity.AUDIT_LABEL_SCOPE,
            "One seed, repeated source and artificial RG swap; no significance, causal proof or new blind/real-domain result.",
            "Residual targets include shadows, reflections, JPEG and other-state ghosts; no physical localization certification.",
            "Correct-versus-cyclic supports only source state correspondence at this fixed budget; no promotion."],
        "runtime":{"full_entry_seconds":time.perf_counter()-PROCESS_STARTED,"point_forward_seconds":None}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","preflight","control","finalize"))
    parser.add_argument("--home",type=Path,default=Path("/root/autodl-tmp/geoprrnet"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--arm",choices=tuple(ARM_DIRECTORIES))
    parser.add_argument("--newcpu-passed",action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        prepare(args.home,args.output or args.home/OUTPUT_RELATIVE/"preparation")
    elif args.stage == "preflight":
        preflight(args.home,args.output or args.home/PREFLIGHT_RELATIVE,newcpu_passed=args.newcpu_passed)
    elif args.stage == "control":
        if args.arm is None:
            parser.error("control requires the frozen correct or cyclic arm")
        control(args.home,args.output or args.home/OUTPUT_RELATIVE/ARM_DIRECTORIES[args.arm],arm=args.arm,newcpu_passed=args.newcpu_passed)
    else:
        finalize(args.home,args.output or args.home/OUTPUT_RELATIVE)


if __name__ == "__main__":
    main()
