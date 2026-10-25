"""One fresh matched15E native-RGB deletion control against frozen C3 statistics."""
from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

from experiments import geoprr_v2_cyclic_color_control as shared


PROTOCOL = "geoprr_v2_matched_identity_source_control_v1"
OUTPUT_RELATIVE = "runs/geoprr_generalization/source_matched_identity_control_20261026"
PREFLIGHT_RELATIVE = "diagnostics/geoprr_matched_identity_control_preflight_20261026/result.json"
PREPARED_RELATIVE = shared.PREPARED_RELATIVE
PREFLIGHT_COUNTS = deepcopy(shared.PREFLIGHT_COUNTS)
REFERENCE_PROTOCOL = shared.PROTOCOL
REFERENCE_CONFIG = deepcopy(shared.CONFIG)
REFERENCE_RELATIVE = "runs/geoprr_generalization/source_cyclic_color_control_20261026/reader/completed.json"
REFERENCE_REVISION = "45de72cbd7f6c0ac1e507f0e357760e77e2a3933"
IDENTITY_ORDERS = (shared.NATIVE_ORDER,)*3
REFERENCE_STATISTICS = {
    "source_id":{"nmae":.02583311489025332,"tail10_nmae":.11098201560477415,
        "maximum_error":.5509077310562134,"count_at_5_percent_fs":264},
    "source_counterfactual":{"nmae":.02558655711628452,"tail10_nmae":.1071122961739699,
        "maximum_error":.5359558463096619,"count_at_5_percent_fs":261},
}
REFERENCE_DRIFT = {"mean_absolute_scalar_drift":.004645775858379026,
    "tail30_absolute_scalar_drift":.025007127784192562,"maximum_absolute_scalar_drift":.08187556266784668}
AUDIT_LABEL_SCOPE = ("Provided-label saved-number comparison only. The prior C3 audit remains149checks148PASS1FAIL: "
    "the normalized_progress/y field check was unavailable in the physical-label-triplet source contract. "
    "This is inherited raw-label-derivation coverage, not a discovered label conflict; no old audit or labels are recomputed.")
CONFIG = deepcopy(shared.CONFIG)
CONFIG.update(protocol=PROTOCOL,
    kind="one fresh matched15E native-RGB deletion control; not an old checkpoint extension or model promotion")
CONFIG["operator"].update(train_orders=[list(order) for order in IDENTITY_ORDERS],
    order_index="(dataset_index + epoch - 1) % 3; nominal slot only; all three slots map to identity RGB",
    nominal_slots_per_epoch=[1832,1832,1832],nominal_slot_to_actual_RGB=["RGB","RGB","RGB"],
    history_order_exposures="nominal slots only, never distinct actual colors",
    actual_RGB_GBR_BRG_per_epoch=[5496,0,0],each_item_actual_RGB_exposure_15_epochs=15,
    placement="training identity returns the original normalizedFP32 RGB tensor; no affine or pixel change")
del CONFIG["operator"]["orders_per_epoch"]
del CONFIG["operator"]["each_item_exposure_15_epochs"]
CONFIG["budget_qualification"] = ("15vs15 scheduled epochs/82440 train images/1290 backward attempts matched to C3; "
    "actual optimizer updates and AMP skips may differ and are reported separately; one seed, source-only")
CONFIG["matched_comparison"] = {"reference_relative":REFERENCE_RELATIVE,"reference_science_revision":REFERENCE_REVISION,
    "reference_statistics":REFERENCE_STATISTICS,"reference_drift":REFERENCE_DRIFT,
    "risk":"(native_mean + native_ceil30_tail + CF_mean + CF_ceil30_tail) / 4",
    "support":"C3 risk strictly smaller; combined >5 count and both-view maximum no larger; mean and ceil30 scalar drift strictly smaller",
    "eligible":"qualified prior C3 plus identity native learning and primary pass plus both complete292 CF observations",
    "audit_label_scope":AUDIT_LABEL_SCOPE}

write_json = shared.write_json
validate_prepared = shared.validate_prepared
count_summary = shared.count_summary
validate_totals = shared.validate_totals


@contextmanager
def identity_context():
    """Bind the frozen shared execution to this process's protocol and identity mapping."""
    values = {"PROTOCOL":PROTOCOL,"CONFIG":CONFIG,"OUTPUT_RELATIVE":OUTPUT_RELATIVE,
        "PREFLIGHT_RELATIVE":PREFLIGHT_RELATIVE,"TRAIN_ORDERS":IDENTITY_ORDERS,"PROCESS_STARTED":PROCESS_STARTED}
    previous = {name:getattr(shared,name) for name in values}
    try:
        for name,value in values.items():
            setattr(shared,name,value)
        yield
    finally:
        for name,value in previous.items():
            setattr(shared,name,value)


def validate_reference(home, prepared):
    """Check existing metadata identity and qualification without opening prior weights or scalar arrays."""
    saved = json.loads((home/REFERENCE_RELATIVE).read_text())
    qualified = (saved["protocol"] == REFERENCE_PROTOCOL and saved["config"] == REFERENCE_CONFIG
        and saved["status"] == "complete" and saved["epochs"] == 15
        and saved["source_contract"] == prepared["source_contract"]
        and saved["native_learning_gate"]["passed"] and saved["native_primary"]["passed"]
        and saved["counterfactual_eligible"] and saved["counterfactual_gate"]["passed"]
        and saved["verdict"] == "SOURCE_COLOR_CONTROL_PASSED"
        and saved["source_id"]["summary"]["samples"] == 292
        and saved["source_counterfactual"]["summary"]["samples"] == 292)
    if not qualified:
        raise ValueError("the frozen C3 reference is not the same completed, source-qualified292-row control")
    return {"qualified":True,"source_contract_equal":True,"reference_science_revision":REFERENCE_REVISION,
        "reference_relative":REFERENCE_RELATIVE,"source_ID_rows":292,"CF_rows":292,
        "optimizer_updates":saved["totals"]["optimizer_updates"],"amp_skipped_updates":saved["totals"]["amp_skipped_updates"],
        "audit_label_scope":AUDIT_LABEL_SCOPE}


def paired_view_statistics(native, counterfactual):
    return {"R_view":(native["nmae"]+native["tail10_nmae"]+counterfactual["nmae"]+counterfactual["tail10_nmae"])/4,
        "combined_errors_above_5_percent_fs":584-native["count_at_5_percent_fs"]-counterfactual["count_at_5_percent_fs"],
        "maximum_across_both_views":max(native["maximum_error"],counterfactual["maximum_error"])}


def matched_decision(identity, reference_qualified):
    native = identity["source_id"]["summary"]
    counterfactual = identity["source_counterfactual"]
    eligible = bool(reference_qualified and identity["native_learning_gate"]["passed"]
        and identity["native_primary"]["passed"] and identity["counterfactual_eligible"]
        and native["samples"] == 292 and counterfactual is not None
        and counterfactual["summary"]["samples"] == 292 and identity["counterfactual_gate"] is not None)
    reference = paired_view_statistics(REFERENCE_STATISTICS["source_id"],REFERENCE_STATISTICS["source_counterfactual"])
    report = {"eligible":eligible,"verdict":"MATCHED_CF_EVIDENCE_INCOMPLETE","predicates":None,
        "C3":reference | REFERENCE_DRIFT,"identity":None,
        "reference_science_revision":REFERENCE_REVISION,"reference_statistics_source":"frozen protocol constants; no prior model or scalar-array evaluation",
        "scheduled_budget_matched":True,"scheduled_train_images_each":82440,"scheduled_backward_attempts_each":1290,
        "identity_optimizer_updates":identity["totals"]["optimizer_updates"],
        "identity_amp_skipped_updates":identity["totals"]["amp_skipped_updates"],
        "scope":"one-seed paired fixed source views; no significance, blind-test or real-domain conclusion",
        "audit_label_scope":AUDIT_LABEL_SCOPE}
    if not eligible:
        return report
    observed = paired_view_statistics(native,counterfactual["summary"])
    drift = identity["counterfactual_gate"]
    observed.update({key:drift[key] for key in REFERENCE_DRIFT})
    predicates = {"R_view":reference["R_view"] < observed["R_view"],
        "combined_errors_above_5_percent_fs":reference["combined_errors_above_5_percent_fs"] <= observed["combined_errors_above_5_percent_fs"],
        "maximum_across_both_views":reference["maximum_across_both_views"] <= observed["maximum_across_both_views"],
        "mean_absolute_scalar_drift":REFERENCE_DRIFT["mean_absolute_scalar_drift"] < observed["mean_absolute_scalar_drift"],
        "tail30_absolute_scalar_drift":REFERENCE_DRIFT["tail30_absolute_scalar_drift"] < observed["tail30_absolute_scalar_drift"]}
    report.update(identity=observed,predicates=predicates,
        verdict="MATCHED_SUPPORT_FOR_C3" if all(predicates.values()) else "NO_MATCHED_SUPPORT_FOR_C3")
    return report


def preflight(home, output):
    prepared = validate_prepared(home)
    reference = validate_reference(home,prepared)
    with identity_context():
        shared.preflight(home,output)
    metadata_started = time.perf_counter()
    saved = json.loads(output.read_text())
    nominal = saved.pop("train_order_indices")
    saved.update(identity_actual_rgb=True,train_nominal_slots=nominal,
        train_actual_orders=[list(shared.NATIVE_ORDER)]*4,
        train_exposure_semantics="slots0/1/2/0 all native RGB; no training color permutation",
        reference_qualification=reference)
    saved["runtime"].update(full_process_seconds=time.perf_counter()-PROCESS_STARTED,
        identity_metadata_seconds=time.perf_counter()-metadata_started,
        full_process_boundary="matched-identity module entry before torch import through saved metadata; excludes final result JSON")
    write_json(output,saved)


def control(home, output):
    prepared = validate_prepared(home)
    reference = validate_reference(home,prepared)
    with identity_context():
        shared.control(home,output)
    metadata_started = time.perf_counter()
    completed = json.loads((output/"completed.json").read_text())
    decision = matched_decision(completed,reference["qualified"])
    decision.update(C3_optimizer_updates=reference["optimizer_updates"],C3_amp_skipped_updates=reference["amp_skipped_updates"])
    write_json(output/"matched_comparison.json",decision)
    completed.update(verdict=decision["verdict"],matched_comparison=decision,
        reference_qualification=reference,identity_actual_rgb=True,
        actual_RGB_GBR_BRG_per_epoch=[5496,0,0],actual_RGB_GBR_BRG_total=[82440,0,0],
        each_item_actual_RGB_exposure_15_epochs=15,
        history_order_exposures_semantics="[1832,1832,1832] counts nominal slots, all mapped to native RGB",
        identity_observation_predicates={"native_learning_passed":completed["native_learning_gate"]["passed"],
            "native_primary_passed":completed["native_primary"]["passed"],
            "CF_observed":completed["counterfactual_eligible"],
            "CF_old_predicates_passed":None if completed["counterfactual_gate"] is None else completed["counterfactual_gate"]["passed"]},
        audit_label_scope=AUDIT_LABEL_SCOPE)
    completed["runtime"].update(full_process_seconds=completed["runtime"]["prior_checkpointed_process_seconds"]+time.perf_counter()-PROCESS_STARTED,
        identity_metadata_seconds=time.perf_counter()-metadata_started,
        full_process_boundary="matched-identity module entry before torch import through saved matched comparison; excludes final completion JSON and downtime")
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
