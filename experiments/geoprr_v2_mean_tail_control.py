"""One fixed mean-plus-batch-tail source objective under the matched15E identity schedule."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import json
from pathlib import Path

from experiments import geoprr_v2_cyclic_color_control as shared
from experiments import geoprr_v2_matched_identity_control as identity


PROTOCOL = "geoprr_v2_mean_tail_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_mean_tail_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_mean_tail_control_preflight_20261026/result.json"
PREPARED_RELATIVE = shared.PREPARED_RELATIVE
PREFLIGHT_COUNTS = deepcopy(shared.PREFLIGHT_COUNTS)
REFERENCE_RELATIVE = "runs/geoprr_generalization/source_matched_identity_control_20261026/reader"
REFERENCE_REVISION = "3e39174f36b342691f6ee96acdea2036ee5e8d30"
REFERENCE_STATISTICS = {
    "native":{"nmae":.025431218243695233,"tail10_nmae":.11800955331418664,
        "maximum_error":.5379703640937805,"errors_above_5_percent_fs":16},
    "counterfactual":{"nmae":.026353762864682596,"tail10_nmae":.12945961876345488,
        "maximum_error":.5424728989601135,"errors_above_5_percent_fs":17},
}
CONFIG = deepcopy(identity.CONFIG)
CONFIG.update(protocol=PROTOCOL,kind="one matched15E source loss-aggregation control; not formalV32 or a new theory/generalization model")
del CONFIG["matched_comparison"]
CONFIG["objective"] = {
    "per_image":"one scalar SmoothL1(prediction_i, y_i, beta=.05, reduction=none)",
    "aggregation":"0.5 * per_image.mean() + 0.5 * per_image.topk(ceil(.1*actual_B)).mean()",
    "mean_weight":.5,"tail_weight":.5,"tail_fraction":.1,"smooth_l1_beta":.05,
    "k_B1_B4_B56_B64":[1,1,6,7],"upweighted_tail_slots_per_complete_epoch":601,
    "all_samples_have_mean_contribution":True,"extra_forward_or_loss":False,
    "history_source_training_smooth_l1":"sample-count-weighted mean of batch composite risks; not primitive mean SmoothL1 or population tail",
    "source_training_nmae":"unchanged unweighted scalar absolute error",
    "source_preflight_mean_tail_objective":"same composite aggregation for the existing four training predictions, k1",
    "qualification":"batch-local SmoothL1 tail differs from full-population absolute-error tail; no trimming or deletion",
}
CONFIG["training"]["loss"] = "fixed .5 mean + .5 top-ceil10-percent per-image SmoothL1"
CONFIG["native_primary"] = {"reference":"saved matched identity15E native292; no prior model forward",
    "reference_statistics":REFERENCE_STATISTICS["native"],"tail_comparison":"strictly smaller",
    "ordinary_guard":"zero new >5-percent-FS errors among the aligned reference native error<=.01 rows",
    "counts_at_1_and_2_percent_are_diagnostic_only":True}
CONFIG["counterfactual_gate"].update(role="previous color-control diagnostic only; acceptance is joint_counterfactual_gate",
    thresholds_are_acceptance=False)
CONFIG["joint_counterfactual_gate"] = {"reference_statistics":REFERENCE_STATISTICS["counterfactual"],
    "tail_comparison":"no larger","ordinary_guard":"zero new >5-percent-FS errors among aligned reference CF error<=.01 rows",
    "previous_color_drift_thresholds":"diagnostic only; not acceptance"}
CONFIG["reference"] = {"protocol":identity.PROTOCOL,"science_revision":REFERENCE_REVISION,
    "relative_reader":REFERENCE_RELATIVE,"statistics":REFERENCE_STATISTICS,
    "provided_label_scope":identity.AUDIT_LABEL_SCOPE}
CONFIG["budget_qualification"] = ("same identity15E scheduled82440 train images/1290 backward attempts; "
    "actual updates and AMP skips reported separately; only the fixed per-batch loss aggregation changes")
write_json = shared.write_json
validate_prepared = shared.validate_prepared
count_summary = shared.count_summary
validate_totals = shared.validate_totals


def tail_count(batch_size):
    return (batch_size+9)//10


def aggregate_mean_tail(per_image):
    """Give every image mean weight and the batch's largest ceil10-percent additional weight."""
    k = tail_count(len(per_image))
    return .5*per_image.mean()+.5*per_image.topk(k,sorted=False).values.mean()


def mean_tail_smooth_l1(prediction, target, primitive, *, beta=.05):
    if beta != .05:
        raise ValueError("the frozen source objective requires SmoothL1 beta=.05")
    return aggregate_mean_tail(primitive(prediction,target,beta=.05,reduction="none"))


def _read_scalar_rows(path):
    with path.open(newline="",encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("target","absolute_error"):
            row[key] = float(row[key])
        row["prediction"] = None if row["prediction"] == "" else float(row["prediction"])
        for key in ("valid","input_valid"):
            if row[key] not in ("True","False"):
                raise ValueError("saved identity scalar validity field differs")
            row[key] = row[key] == "True"
    return rows


def aligned_pairs(reference, candidate):
    if (len(reference) != 292 or len(candidate) != 292
            or len({row["sample_id"] for row in reference}) != 292
            or len({row["sample_id"] for row in candidate}) != 292):
        raise ValueError("the source paired guard requires the same complete292 distinct rows")
    pairs = list(zip(reference,candidate,strict=True))
    if any(any(before[key] != after[key] for key in ("sample_id","group_id","target")) for before,after in pairs):
        raise ValueError("source reference and new scalar IDs, groups or provided labels are not exactly aligned")
    return pairs


def paired_reference_guard(reference, candidate):
    pairs = aligned_pairs(reference,candidate)
    thresholds,groups = {},{}
    for threshold in (1,2,5):
        boundary = threshold/100
        new = [after for before,after in pairs if before["absolute_error"] <= boundary < after["absolute_error"]]
        rescues = [after for before,after in pairs if after["absolute_error"] <= boundary < before["absolute_error"]]
        thresholds[str(threshold)] = {"new_errors":len(new),"rescues":len(rescues),
            "new_error_ids":[row["sample_id"] for row in new],"rescue_ids":[row["sample_id"] for row in rescues],
            "new_error_groups":sorted({row["group_id"] for row in new}),"rescue_groups":sorted({row["group_id"] for row in rescues}),
            "reference_errors_above":sum(before["absolute_error"] > boundary for before,_ in pairs),
            "candidate_errors_above":sum(after["absolute_error"] > boundary for _,after in pairs)}
    ordinary = [after for before,after in pairs if before["absolute_error"] <= .01 and after["absolute_error"] > .05]
    for before,after in pairs:
        group = groups.setdefault(after["group_id"],{"samples":0,"reference_error_sum":0.,"candidate_error_sum":0.,
            "reference_errors_above_1_2_5":[0,0,0],"candidate_errors_above_1_2_5":[0,0,0]})
        group["samples"] += 1
        group["reference_error_sum"] += before["absolute_error"]
        group["candidate_error_sum"] += after["absolute_error"]
        for i,threshold in enumerate((1,2,5)):
            group["reference_errors_above_1_2_5"][i] += int(before["absolute_error"] > threshold/100)
            group["candidate_errors_above_1_2_5"][i] += int(after["absolute_error"] > threshold/100)
    return {"denominator":292,"paired_thresholds_percent_fs":thresholds,
        "reference_accurate_at_1_percent_count":sum(before["absolute_error"] <= .01 for before,_ in pairs),
        "new_over5_on_reference_accurate_count":len(ordinary),
        "new_over5_on_reference_accurate_ids":[row["sample_id"] for row in ordinary],
        "new_over5_on_reference_accurate_groups":sorted({row["group_id"] for row in ordinary}),
        "error_improved_ids":[after["sample_id"] for before,after in pairs if after["absolute_error"] < before["absolute_error"]],
        "error_worsened_ids":[after["sample_id"] for before,after in pairs if after["absolute_error"] > before["absolute_error"]],
        "group_differences":groups}


def joint_view_gate(reference, candidate, summary, *, view):
    paired = paired_reference_guard(reference,candidate)
    limits = REFERENCE_STATISTICS[view]
    predicates = {"mean":summary["nmae"] <= limits["nmae"],
        "tail":summary["tail10_nmae"] < limits["tail10_nmae"] if view == "native" else summary["tail10_nmae"] <= limits["tail10_nmae"],
        "maximum":summary["maximum_error"] <= limits["maximum_error"],
        "errors_above_5_percent_fs":paired["paired_thresholds_percent_fs"]["5"]["candidate_errors_above"] <= limits["errors_above_5_percent_fs"],
        "no_new_over5_on_reference_accurate":paired["new_over5_on_reference_accurate_count"] == 0}
    return {"passed":all(predicates.values()),"view":view,"predicates":predicates,"limits":limits,
        "mean":summary["nmae"],"tail":summary["tail10_nmae"],"maximum":summary["maximum_error"],
        "counts_over_1_2_5":[paired["paired_thresholds_percent_fs"][str(t)]["candidate_errors_above"] for t in (1,2,5)],
        "paired_reference":paired,"reference_science_revision":REFERENCE_REVISION,
        "counts_at_1_and_2_percent_are_diagnostic_only":True}


def load_reference(home, prepared):
    root = home/REFERENCE_RELATIVE
    saved = json.loads((root/"completed.json").read_text())
    if (saved["protocol"] != identity.PROTOCOL or saved["config"] != identity.CONFIG
            or saved["status"] != "complete" or saved["epochs"] != 15
            or saved["source_contract"] != prepared["source_contract"]
            or not saved["native_learning_gate"]["passed"] or not saved["native_primary"]["passed"]
            or not saved["counterfactual_eligible"] or saved["source_counterfactual"] is None
            or saved["source_id"]["summary"]["samples"] != 292
            or saved["source_counterfactual"]["summary"]["samples"] != 292):
        raise ValueError("the frozen identity reference lacks exact source identity or native/CF observation qualification")
    native = _read_scalar_rows(root/"source_id_predictions.csv")
    counterfactual = _read_scalar_rows(root/"source_counterfactual_predictions.csv")
    aligned_pairs(native,counterfactual)
    expected_ids = [str(row["sample_id"]) for row in prepared["source_contract"]["populations"]["id"]]
    if [row["sample_id"] for row in native] != expected_ids:
        raise ValueError("saved identity scalar rows differ from the same prepared source-ID order")
    return {"native":native,"counterfactual":counterfactual,
        "metadata":{"protocol":identity.PROTOCOL,"science_revision":REFERENCE_REVISION,
            "source_contract_equal":True,"native_rows":292,"counterfactual_rows":292,
            "reference_optimizer_updates":saved["totals"]["optimizer_updates"],
            "reference_amp_skipped_updates":saved["totals"]["amp_skipped_updates"],
            "provided_label_scope":identity.AUDIT_LABEL_SCOPE}}


@contextmanager
def mean_tail_context(reference):
    """Bind this process's loss and joint gates, restoring all shared references afterward."""
    from torch.nn import functional as functional
    primitive = functional.smooth_l1_loss
    values = {"PROTOCOL":PROTOCOL,"CONFIG":CONFIG,"OUTPUT_RELATIVE":OUTPUT_RELATIVE,
        "PREFLIGHT_RELATIVE":PREFLIGHT_RELATIVE,"TRAIN_ORDERS":identity.IDENTITY_ORDERS,"PROCESS_STARTED":PROCESS_STARTED}
    previous = {name:getattr(shared,name) for name in values}
    learning,primary,counterfactual = shared.learning_gate,shared.native_primary,shared.counterfactual_gate
    native_rows = None
    def objective(prediction,target,*,beta=.05):
        return mean_tail_smooth_l1(prediction,target,primitive,beta=beta)
    def capture_learning(rows):
        nonlocal native_rows
        native_rows = rows
        return learning(rows)
    def native_joint(gate):
        return joint_view_gate(reference["native"],native_rows,gate["model"],view="native")
    def counterfactual_joint(native,rows,summary):
        result = joint_view_gate(reference["counterfactual"],rows,summary,view="counterfactual")
        diagnostic = counterfactual(native,rows,summary)
        for key in ("denominator","tail_count","mean_absolute_scalar_drift","tail30_absolute_scalar_drift",
                    "maximum_absolute_scalar_drift","paired_error_rescues","paired_error_regressions",
                    "paired_error_unchanged","paired_thresholds_percent_fs","failed_pairs"):
            result[key] = diagnostic[key]
        result.update(previous_color_control_diagnostic=diagnostic,drift_thresholds_are_acceptance=False)
        return result
    try:
        for name,value in values.items():
            setattr(shared,name,value)
        functional.smooth_l1_loss = objective
        shared.learning_gate,shared.native_primary,shared.counterfactual_gate = capture_learning,native_joint,counterfactual_joint
        yield
    finally:
        functional.smooth_l1_loss = primitive
        shared.learning_gate,shared.native_primary,shared.counterfactual_gate = learning,primary,counterfactual
        for name,value in previous.items():
            setattr(shared,name,value)


def preflight(home, output):
    reference = load_reference(home,validate_prepared(home))
    with mean_tail_context(reference):
        shared.preflight(home,output)
    metadata_started = time.perf_counter()
    saved = json.loads(output.read_text())
    nominal = saved.pop("train_order_indices")
    objective = saved.pop("source_smooth_l1")
    saved.update(identity_actual_rgb=True,train_nominal_slots=nominal,train_actual_orders=[list(shared.NATIVE_ORDER)]*4,
        objective_config=CONFIG["objective"],source_preflight_mean_tail_objective=objective,preflight_tail_k=1,
        train_exposure_semantics="slots0/1/2/0 all native RGB; only the batch loss aggregation changes",
        reference_qualification=reference["metadata"])
    saved["runtime"].update(full_process_seconds=time.perf_counter()-PROCESS_STARTED,
        mean_tail_metadata_seconds=time.perf_counter()-metadata_started,
        individual_operator_seconds=None,individual_loss_aggregation_seconds=None,
        full_process_boundary="mean-tail module entry before torch import through saved metadata; excludes final result JSON")
    write_json(output,saved)


def control(home, output):
    reference = load_reference(home,validate_prepared(home))
    with mean_tail_context(reference):
        shared.control(home,output)
    metadata_started = time.perf_counter()
    completed = json.loads((output/"completed.json").read_text())
    passed = bool(completed["native_learning_gate"]["passed"] and completed["native_primary"]["passed"]
        and completed["counterfactual_eligible"] and completed["counterfactual_gate"]["passed"])
    completed.update(verdict="MEAN_TAIL_SOURCE_CONTROL_PASSED" if passed else "NOT_RETAINED",
        identity_actual_rgb=True,objective_config=CONFIG["objective"],reference_qualification=reference["metadata"],
        actual_RGB_GBR_BRG_per_epoch=[5496,0,0],actual_RGB_GBR_BRG_total=[82440,0,0],
        each_item_actual_RGB_exposure_15_epochs=15,
        history_order_exposures_semantics="[1832,1832,1832] counts nominal slots, all mapped to native RGB",
        source_training_smooth_l1_semantics=CONFIG["objective"]["history_source_training_smooth_l1"],
        drift_thresholds_are_acceptance=False,provided_label_scope=identity.AUDIT_LABEL_SCOPE)
    completed["runtime"].update(full_process_seconds=completed["runtime"]["prior_checkpointed_process_seconds"]+time.perf_counter()-PROCESS_STARTED,
        mean_tail_metadata_seconds=time.perf_counter()-metadata_started,
        individual_operator_seconds=None,individual_loss_aggregation_seconds=None,
        full_process_boundary="mean-tail module entry before torch import through saved joint metadata; excludes final completion JSON and downtime")
    write_json(output/"completed.json",completed)


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
