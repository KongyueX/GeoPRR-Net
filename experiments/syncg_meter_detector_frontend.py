"""Freeze and verify the public-only shared meter-detection frontend lineage."""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.build_syncg_meter_detector_public import (
    PROTOCOL as CORPUS_PROTOCOL,
    SEAL_PROTOCOL as CORPUS_SEAL_PROTOCOL,
    sha256_file,
    verify_corpus,
    verify_corpus_partition,
)
from experiments.train_syncg_meter_detector import (
    EXPECTED_ULTRALYTICS,
    EXPECTED_PRETRAINED_SHA256,
    EXPERIMENT_PROTOCOL_PATH,
    RUN_INTENT_PROTOCOL,
    SELECTION_CLAIM_PROTOCOL,
    TRAINER_PATH,
    TRAINING_PROTOCOL,
    TRAINING_SEAL_PROTOCOL,
    _metric,
    _runtime_identity,
    _selected_model_identity,
    choose_confidence_threshold,
    selected_box_metrics,
)


FRONTEND_PROTOCOL = "field_blind_shared_meter_frontend_v1"
FRONTEND_SEAL_PROTOCOL = "syncg_public_meter_frontend_seal_v1"
SAFE_OUTPUT_ROOT = Path(r"C:\pointer_read").resolve()
DEFAULT_TRAINING_SUMMARY = (
    SAFE_OUTPUT_ROOT / "syncg_meter_detector_runs/seed_20260819/summary.json"
)
DEFAULT_OUTPUT = SAFE_OUTPUT_ROOT / "syncg_meter_detector_frontend_v1"
EXPECTED_DETECTOR_SOURCE = (
    PROJECT_ROOT / "utils/angleDetect/yoloDetection/yoloDectect.py"
).resolve()
LEGACY_METER_CHECKPOINT_SHA256 = (
    "98b8f40cba170b40f828bd0579e6a8a5feba500fd83261a13651ab428c0956ab"
)
FAILURE_PENALTY = 1.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value.casefold()).issubset(set("0123456789abcdef"))
    )


def _binding(path: Path) -> dict[str, str]:
    value = Path(path).resolve(strict=True)
    return {"path": str(value), "sha256": sha256_file(value)}


def _resolve_binding(value: Any, *, label: str) -> Path:
    _require(isinstance(value, Mapping), f"{label} binding absent")
    path = Path(str(value.get("path") or ""))
    _require(path.is_absolute(), f"{label}.path must be absolute")
    path = path.resolve(strict=True)
    digest = str(value.get("sha256") or "").casefold()
    _require(_is_sha256(digest), f"{label}.sha256 invalid")
    _require(sha256_file(path) == digest, f"{label} hash drift")
    return path


def _safe_output(path: Path) -> Path:
    output = Path(path).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"frontend output must stay below {SAFE_OUTPUT_ROOT}") from error
    _require(output != SAFE_OUTPUT_ROOT, "refusing broad frontend output root")
    return output


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"immutable artifact already exists: {path}")
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_production_source(path: Path) -> None:
    source = Path(path).resolve(strict=True)
    _require(source == EXPECTED_DETECTOR_SOURCE, "shared frontend is not the production detector source")
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    _require("targetDetectModel" in classes, "production detector class targetDetectModel is absent")


def verify_training_lineage(summary_path: Path) -> dict[str, Any]:
    path = Path(summary_path).resolve(strict=True)
    summary = _json(path)
    _require(summary.get("protocol") == TRAINING_PROTOCOL, "detector training protocol drift")
    _require(summary.get("status") == "complete", "detector training did not pass its gate")
    scope = summary.get("scope") or {}
    _require(scope.get("dataset") == "SyncG" and scope.get("split") == "train", "detector training scope drift")
    _require(scope.get("public_data_only") is True, "detector training is not public-only")
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(scope.get(key) is False, f"detector training audit violation: {key}")
    implementation = summary.get("implementation") or {}
    trainer_path = _resolve_binding(implementation.get("trainer"), label="detector trainer")
    experiment_protocol_path = _resolve_binding(
        implementation.get("experiment_protocol"), label="detector experiment protocol"
    )
    _require(trainer_path == TRAINER_PATH.resolve(strict=True), "detector trainer path drift")
    _require(
        experiment_protocol_path == EXPERIMENT_PROTOCOL_PATH.resolve(strict=True),
        "detector experiment protocol path drift",
    )
    _require(
        implementation.get("ultralytics_version") == EXPECTED_ULTRALYTICS,
        "detector Ultralytics version drift",
    )
    experiment_protocol = _json(experiment_protocol_path)
    _require(
        experiment_protocol.get("protocol") == "syncg_public_meter_detector_experiment_v1"
        and experiment_protocol.get("status") == "frozen_before_training",
        "detector experiment protocol identity drift",
    )
    _require(
        (experiment_protocol.get("legacy_checkpoint_rejection") or {}).get("sha256")
        == LEGACY_METER_CHECKPOINT_SHA256,
        "legacy checkpoint rejection identity drift",
    )
    runtime_identity = _runtime_identity(experiment_protocol)
    _require(
        implementation.get("runtime_identity") == runtime_identity,
        "detector training runtime identity drift",
    )
    run = summary.get("run") or {}
    frozen_partition = experiment_protocol.get("partition") or {}
    _require(
        int(run.get("seed", -1)) == int(frozen_partition.get("seed", -2)),
        "detector run seed differs from frozen protocol",
    )

    run_root = path.parent
    seal_path = run_root / "seal.json"
    seal = _json(seal_path.resolve(strict=True))
    _require(
        seal.get("protocol") == TRAINING_SEAL_PROTOCOL and seal.get("status") == "sealed",
        "detector training seal drift",
    )
    _require(seal.get("summary_sha256") == sha256_file(path), "training summary/seal hash drift")
    _require(seal.get("validation_gate_pass") is True, "training validation gate did not pass")
    _require(seal.get("trainer_sha256") == sha256_file(trainer_path), "training seal/trainer drift")
    _require(
        seal.get("experiment_protocol_sha256") == sha256_file(experiment_protocol_path),
        "training seal/experiment protocol drift",
    )
    run_intent_path = _resolve_binding(run.get("intent"), label="detector run intent")
    run_intent = _json(run_intent_path)
    _require(
        run_intent.get("protocol") == RUN_INTENT_PROTOCOL
        and run_intent.get("status") == "frozen_before_first_training_process",
        "detector run intent drift",
    )
    _require(int(run_intent.get("seed", -1)) == int(run.get("seed", -2)), "run intent seed drift")
    _require(run_intent.get("runtime_identity") == runtime_identity, "run intent runtime drift")
    _require(
        run_intent.get("trainer_sha256") == sha256_file(trainer_path)
        and run_intent.get("experiment_protocol_sha256")
        == sha256_file(experiment_protocol_path),
        "run intent implementation drift",
    )
    _require(
        seal.get("run_intent_sha256") == sha256_file(run_intent_path),
        "training seal/run-intent drift",
    )

    selection = summary.get("selection") or {}
    _require(selection.get("partition") == "calibration", "detector selection did not use calibration")
    _require(
        selection.get("validation_model_inference_and_metrics_after_selection_claim") is True,
        "selection chronology absent",
    )
    claim_path = _resolve_binding(selection.get("claim"), label="selection claim")
    claim = _json(claim_path)
    _require(claim.get("protocol") == SELECTION_CLAIM_PROTOCOL, "selection claim protocol drift")
    _require(claim.get("status") == "selection_frozen_before_independent_validation", "selection was not frozen")
    _require(claim.get("trainer_sha256") == sha256_file(trainer_path), "selection claim/trainer drift")
    _require(
        claim.get("experiment_protocol_sha256") == sha256_file(experiment_protocol_path),
        "selection claim/experiment protocol drift",
    )
    _require(int(claim.get("seed", -1)) == int(run.get("seed", -2)), "selection claim seed drift")
    _require(
        _resolve_binding(claim.get("run_intent"), label="selection run intent")
        == run_intent_path,
        "selection claim/run-intent drift",
    )
    _require(claim.get("runtime_identity") == runtime_identity, "selection claim runtime drift")
    _require(
        seal.get("selection_claim_sha256") == sha256_file(claim_path),
        "training seal/selection-claim drift",
    )
    for key in (
        "independent_validation_model_inference_before_claim",
        "independent_validation_performance_metrics_before_claim",
        "field_manifest_opened",
        "field_images_opened",
        "field_labels_opened",
    ):
        _require(claim.get(key) is False, f"selection chronology/scope violation: {key}")
    _require(
        claim.get("independent_validation_format_integrity_preflight_permitted") is True,
        "validation integrity-preflight policy drift",
    )
    validation = summary.get("validation") or {}
    _require(validation.get("partition") == "independent_validation", "independent validation identity drift")
    _require(
        validation.get("model_inference_and_metrics_after_selection_claim") is True,
        "independent validation chronology drift",
    )
    _require(validation.get("gate_pass") is True, "independent validation gate failed")
    gate_checks = validation.get("gate_checks") or {}
    _require(set(gate_checks) == {
        "selected_box_iou50_recall",
        "selected_box_precision",
        "mean_selected_iou",
        "ultralytics_map50",
        "ultralytics_map50_95",
    } and all(gate_checks.values()), "not all detector validation checks passed")

    best_path = _resolve_binding((summary.get("artifacts") or {}).get("best_checkpoint"), label="best checkpoint")
    _require(seal.get("best_checkpoint_sha256") == sha256_file(best_path), "training seal/best checkpoint drift")
    _require(claim.get("checkpoint", {}).get("sha256") == sha256_file(best_path), "selection claim/best checkpoint drift")
    _require(
        _resolve_binding(claim.get("checkpoint"), label="selection checkpoint") == best_path,
        "selection claim/best checkpoint path drift",
    )
    model_identity = selection.get("selected_model_identity")
    _require(
        isinstance(model_identity, Mapping)
        and model_identity.get("task") == "detect"
        and model_identity.get("names") == {"0": "meter"}
        and int(model_identity.get("parameters", 0)) > 0,
        "selected detector model identity drift",
    )
    _require(
        claim.get("selected_model_identity") == model_identity,
        "selection claim/model identity drift",
    )
    from ultralytics import YOLO

    loaded_identity = _selected_model_identity(YOLO(str(best_path)))
    _require(loaded_identity == model_identity, "best.pt loaded model identity drift")
    frozen_selection = experiment_protocol.get("selection") or {}
    _require(
        selection.get("checkpoint_rule") == frozen_selection.get("checkpoint_rule")
        == claim.get("checkpoint_selection"),
        "checkpoint-selection rule binding drift",
    )
    _require(
        selection.get("confidence") == claim.get("confidence_selection"),
        "summary/claim confidence selection drift",
    )
    _require(
        selection.get("ultralytics_metrics") == claim.get("calibration_ultralytics"),
        "summary/claim calibration metrics drift",
    )
    _require(
        math.isclose(
            float(selection.get("padding_fraction")),
            float(frozen_selection.get("fixed_padding_fraction")),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(selection.get("prediction_confidence_floor")),
            float(frozen_selection.get("prediction_confidence_floor")),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(selection.get("prediction_nms_iou")),
            float(frozen_selection.get("prediction_nms_iou")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "frontend prediction settings differ from frozen protocol",
    )
    _require(
        selection.get("runtime_integer_box_quantization")
        == frozen_selection.get("runtime_integer_box_quantization"),
        "runtime box quantization drift",
    )
    training_evidence = run.get("training_evidence")
    _require(
        isinstance(training_evidence, Mapping)
        and dict(training_evidence) == claim.get("training_evidence"),
        "summary/claim training evidence drift",
    )
    results_path = _resolve_binding(
        training_evidence.get("results_csv"), label="training results.csv"
    )
    _resolve_binding(training_evidence.get("args_yaml"), label="training args.yaml")
    _require(
        seal.get("training_results_sha256") == sha256_file(results_path),
        "training seal/results.csv drift",
    )

    corpus = summary.get("corpus") or {}
    corpus_summary_path = _resolve_binding(corpus.get("summary"), label="corpus summary")
    corpus_seal_path = _resolve_binding(corpus.get("seal"), label="corpus seal")
    corpus_summary, corpus_seal = _json(corpus_summary_path), _json(corpus_seal_path)
    _require(corpus_summary.get("protocol") == CORPUS_PROTOCOL and corpus_summary.get("status") == "complete", "corpus protocol drift")
    _require(corpus_seal.get("protocol") == CORPUS_SEAL_PROTOCOL and corpus_seal.get("status") == "sealed", "corpus seal protocol drift")
    _require(corpus_seal.get("summary_sha256") == sha256_file(corpus_summary_path), "corpus summary/seal drift")
    corpus_scope = corpus_summary.get("scope") or {}
    _require(corpus_scope.get("dataset") == "SyncG" and corpus_scope.get("split") == "train", "corpus source drift")
    _require(corpus_scope.get("public_data_only") is True, "corpus is not public-only")
    _require(corpus_summary_path.parent == Path(str(corpus.get("root") or "")).resolve(strict=True), "corpus root binding drift")
    verify_corpus(corpus_summary_path.parent)
    _require(seal.get("corpus_summary_sha256") == sha256_file(corpus_summary_path), "training/corpus summary drift")
    _require(seal.get("corpus_seal_sha256") == sha256_file(corpus_seal_path), "training/corpus seal drift")
    _require(
        _resolve_binding(run_intent.get("corpus_summary"), label="run-intent corpus summary")
        == corpus_summary_path
        and _resolve_binding(run_intent.get("corpus_seal"), label="run-intent corpus seal")
        == corpus_seal_path,
        "run intent/corpus lineage drift",
    )
    _require(
        _resolve_binding(claim.get("corpus_summary"), label="claim corpus summary")
        == corpus_summary_path
        and _resolve_binding(claim.get("corpus_seal"), label="claim corpus seal")
        == corpus_seal_path,
        "selection claim/corpus lineage drift",
    )

    calibration_rows, _, _ = verify_corpus_partition(
        corpus_summary_path.parent, corpus_summary, "calibration"
    )
    validation_rows, _, _ = verify_corpus_partition(
        corpus_summary_path.parent, corpus_summary, "validation"
    )
    calibration_predictions_path = _resolve_binding(
        (summary.get("artifacts") or {}).get("calibration_predictions"),
        label="calibration predictions",
    )
    _require(
        _resolve_binding(claim.get("calibration_predictions"), label="claim calibration predictions")
        == calibration_predictions_path,
        "claim/summary calibration prediction drift",
    )
    calibration_predictions = [
        json.loads(line)
        for line in calibration_predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    _require(
        choose_confidence_threshold(calibration_rows, calibration_predictions)
        == selection.get("confidence"),
        "calibration confidence selection cannot be reproduced",
    )
    validation_predictions_path = _resolve_binding(
        (summary.get("artifacts") or {}).get("validation_predictions"),
        label="validation predictions",
    )
    validation_predictions = [
        json.loads(line)
        for line in validation_predictions_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    threshold = float(selection["confidence"]["selected"]["confidence_threshold"])
    recomputed_validation = selected_box_metrics(
        validation_rows, validation_predictions, threshold=threshold
    )
    _require(
        recomputed_validation == validation.get("selected_box_metrics"),
        "independent-validation selected-box metrics cannot be reproduced",
    )
    gate_spec = experiment_protocol.get("independent_validation_gate") or {}
    _require(validation.get("gate_specification") == gate_spec, "validation gate specification drift")
    validation_ultralytics = validation.get("ultralytics_metrics") or {}
    recomputed_gate = {
        "selected_box_iou50_recall": recomputed_validation["selected_box_iou50_recall"]
        >= float(gate_spec["selected_box_iou50_recall_minimum"]),
        "selected_box_precision": recomputed_validation["selected_box_precision"]
        >= float(gate_spec["selected_box_precision_minimum"]),
        "mean_selected_iou": recomputed_validation["mean_selected_iou"]
        >= float(gate_spec["mean_selected_iou_minimum"]),
        "ultralytics_map50": _metric(validation_ultralytics, "mAP50(B)")
        >= float(gate_spec["ultralytics_map50_minimum"]),
        "ultralytics_map50_95": _metric(validation_ultralytics, "mAP50-95(B)")
        >= float(gate_spec["ultralytics_map50_95_minimum"]),
    }
    _require(recomputed_gate == gate_checks and all(recomputed_gate.values()), "validation gate is not reproducible")

    pretrained_path = _resolve_binding((summary.get("pretrained") or {}).get("checkpoint"), label="public pretrained checkpoint")
    _require(sha256_file(pretrained_path) == EXPECTED_PRETRAINED_SHA256, "public pretrained identity drift")
    _require(
        str((summary.get("pretrained") or {}).get("official_url") or "")
        == "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "public pretrained URL drift",
    )
    _require(
        _resolve_binding(run_intent.get("pretrained_checkpoint"), label="run-intent pretrained")
        == pretrained_path,
        "run intent/pretrained lineage drift",
    )
    return {
        "summary_path": path,
        "summary": summary,
        "seal_path": seal_path,
        "seal": seal,
        "claim_path": claim_path,
        "claim": claim,
        "run_intent_path": run_intent_path,
        "run_intent": run_intent,
        "runtime_identity": runtime_identity,
        "best_checkpoint": best_path,
        "corpus_summary_path": corpus_summary_path,
        "corpus_seal_path": corpus_seal_path,
        "pretrained_path": pretrained_path,
        "calibration_predictions_path": calibration_predictions_path,
        "validation_predictions_path": validation_predictions_path,
    }


def freeze_frontend(training_summary: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    lineage = verify_training_lineage(training_summary)
    output = _safe_output(output_dir)
    _require(not output.exists(), f"refusing to overwrite frozen frontend: {output}")
    output.mkdir(parents=True)
    try:
        source = EXPECTED_DETECTOR_SOURCE.resolve(strict=True)
        _validate_production_source(source)
        summary = lineage["summary"]
        threshold = float(summary["selection"]["confidence"]["selected"]["confidence_threshold"])
        padding = float(summary["selection"]["padding_fraction"])
        _require(math.isfinite(threshold) and 0.0 <= threshold <= 1.0, "invalid calibrated threshold")
        _require(math.isfinite(padding) and 0.0 <= padding <= 0.25, "invalid frozen padding")
        plan = {
            "schema_version": 1,
            "protocol": FRONTEND_PROTOCOL,
            "status": "public_selected_frozen",
            "detector_checkpoint": _binding(lineage["best_checkpoint"]),
            "detector_source": _binding(source),
            "detector_class": "targetDetectModel",
            "confidence_threshold": threshold,
            "padding_fraction": padding,
            "accepted_class_ids": [0],
            "contract": {
                "input": "original_full_scene_bgr_uint8",
                "selection_rule": "highest_confidence_then_xyxy_lexicographic",
                "padding_rule": "fraction_of_detected_box_width_and_height_symmetric_clamped",
                "fallback_to_full_frame": False,
                "caller_bbox_allowed": False,
                "caller_crop_allowed": False,
                "ground_truth_geometry_allowed": False,
                "correction_or_warp_applied": False,
                "same_roi_for_all_methods": True,
                "detector_failure_penalty_nmae": FAILURE_PENALTY,
            },
            "selection_audit": {
                "dataset": "SyncG",
                "split": "train",
                "public_data_only": True,
                "selection_partition": "calibration",
                "validation_partition": "independent_validation",
                "validation_model_inference_and_metrics_only_after_selection_claim": True,
                "validation_format_integrity_preflight_permitted": True,
                "training_protocol": TRAINING_PROTOCOL,
                "training_summary": _binding(lineage["summary_path"]),
                "training_seal": _binding(lineage["seal_path"]),
                "run_intent": _binding(lineage["run_intent_path"]),
                "selection_claim": _binding(lineage["claim_path"]),
                "calibration_predictions": _binding(
                    lineage["calibration_predictions_path"]
                ),
                "validation_predictions": _binding(
                    lineage["validation_predictions_path"]
                ),
                "corpus_protocol": CORPUS_PROTOCOL,
                "corpus_summary": _binding(lineage["corpus_summary_path"]),
                "corpus_seal": _binding(lineage["corpus_seal_path"]),
                "pretrained_checkpoint_sha256": EXPECTED_PRETRAINED_SHA256,
                "runtime_identity": lineage["runtime_identity"],
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
            },
        }
        plan_path = output / "frontend_plan.json"
        _atomic_new_json(plan_path, plan)
        seal = {
            "schema_version": 1,
            "protocol": FRONTEND_SEAL_PROTOCOL,
            "status": "sealed",
            "frontend_plan_sha256": sha256_file(plan_path),
            "detector_checkpoint_sha256": plan["detector_checkpoint"]["sha256"],
            "detector_source_sha256": plan["detector_source"]["sha256"],
            "training_summary_sha256": plan["selection_audit"]["training_summary"]["sha256"],
            "training_seal_sha256": plan["selection_audit"]["training_seal"]["sha256"],
            "run_intent_sha256": plan["selection_audit"]["run_intent"]["sha256"],
            "selection_claim_sha256": plan["selection_audit"]["selection_claim"]["sha256"],
            "calibration_predictions_sha256": plan["selection_audit"][
                "calibration_predictions"
            ]["sha256"],
            "validation_predictions_sha256": plan["selection_audit"][
                "validation_predictions"
            ]["sha256"],
            "corpus_summary_sha256": plan["selection_audit"]["corpus_summary"]["sha256"],
            "corpus_seal_sha256": plan["selection_audit"]["corpus_seal"]["sha256"],
        }
        _atomic_new_json(output / "seal.json", seal)
        return plan_path, plan
    except Exception:
        # Only an incomplete, never-successfully-frozen output can be cleaned.
        if output.exists() and not (output / "seal.json").exists():
            for child in output.iterdir():
                if child.is_file():
                    child.unlink()
            output.rmdir()
        raise


def verify_public_frontend_lineage(plan: Mapping[str, Any], *, plan_path: Path) -> dict[str, Any]:
    """Verify a parsed plan and every public training artifact it transitively binds."""

    _require(plan.get("protocol") == FRONTEND_PROTOCOL, "frontend protocol drift")
    _require(plan.get("status") == "public_selected_frozen", "frontend is not frozen")
    _require(plan.get("detector_class") == "targetDetectModel", "legacy/unknown detector class rejected")
    checkpoint_path = _resolve_binding(plan.get("detector_checkpoint"), label="frontend checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(
        "yolo_findmeter" not in str(checkpoint_path).casefold(),
        "legacy yolo_findMeter checkpoint is explicitly rejected",
    )
    _require(
        checkpoint_sha256 != LEGACY_METER_CHECKPOINT_SHA256,
        "legacy yolo_findMeter checkpoint content is explicitly rejected",
    )
    source_path = _resolve_binding(plan.get("detector_source"), label="frontend source")
    _validate_production_source(source_path)
    threshold = float(plan.get("confidence_threshold"))
    padding = float(plan.get("padding_fraction"))
    _require(math.isfinite(threshold) and 0.0 <= threshold <= 1.0, "frontend threshold invalid")
    _require(math.isfinite(padding) and 0.0 <= padding <= 0.25, "frontend padding invalid")
    _require(plan.get("accepted_class_ids") == [0], "frontend class roster drift")
    expected_contract = {
        "input": "original_full_scene_bgr_uint8",
        "selection_rule": "highest_confidence_then_xyxy_lexicographic",
        "padding_rule": "fraction_of_detected_box_width_and_height_symmetric_clamped",
        "fallback_to_full_frame": False,
        "caller_bbox_allowed": False,
        "caller_crop_allowed": False,
        "ground_truth_geometry_allowed": False,
        "correction_or_warp_applied": False,
        "same_roi_for_all_methods": True,
        "detector_failure_penalty_nmae": FAILURE_PENALTY,
    }
    _require(plan.get("contract") == expected_contract, "frontend contract drift")
    audit = plan.get("selection_audit") or {}
    _require(audit.get("dataset") == "SyncG" and audit.get("split") == "train", "frontend public source drift")
    _require(audit.get("public_data_only") is True, "frontend is not public-only")
    _require(audit.get("selection_partition") == "calibration", "frontend selection partition drift")
    _require(audit.get("validation_partition") == "independent_validation", "frontend validation identity drift")
    _require(
        audit.get("validation_model_inference_and_metrics_only_after_selection_claim") is True,
        "frontend chronology drift",
    )
    _require(
        audit.get("validation_format_integrity_preflight_permitted") is True,
        "frontend integrity-preflight policy drift",
    )
    for key in ("field_manifest_opened", "field_images_opened", "field_labels_opened"):
        _require(audit.get(key) is False, f"frontend scope violation: {key}")
    training_summary_path = _resolve_binding(audit.get("training_summary"), label="frontend training summary")
    training_seal_path = _resolve_binding(audit.get("training_seal"), label="frontend training seal")
    lineage = verify_training_lineage(training_summary_path)
    _require(training_seal_path == lineage["seal_path"], "frontend training seal path drift")
    _require(checkpoint_path == lineage["best_checkpoint"], "frontend does not bind selected best.pt")
    _require(
        math.isclose(
            threshold,
            float(lineage["summary"]["selection"]["confidence"]["selected"]["confidence_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "frontend threshold differs from frozen calibration selection",
    )
    for key, expected in (
        ("selection_claim", lineage["claim_path"]),
        ("corpus_summary", lineage["corpus_summary_path"]),
        ("corpus_seal", lineage["corpus_seal_path"]),
    ):
        _require(_resolve_binding(audit.get(key), label=f"frontend {key}") == expected, f"frontend {key} lineage drift")
    _require(audit.get("pretrained_checkpoint_sha256") == EXPECTED_PRETRAINED_SHA256, "frontend pretrained identity drift")
    _require(
        audit.get("runtime_identity") == lineage["runtime_identity"],
        "frontend runtime identity drift",
    )
    for key, expected in (
        ("run_intent", lineage["run_intent_path"]),
        ("calibration_predictions", lineage["calibration_predictions_path"]),
        ("validation_predictions", lineage["validation_predictions_path"]),
    ):
        _require(
            _resolve_binding(audit.get(key), label=f"frontend {key}") == expected,
            f"frontend {key} lineage drift",
        )

    seal_path = Path(plan_path).resolve(strict=True).with_name("seal.json")
    seal = _json(seal_path.resolve(strict=True))
    _require(seal.get("protocol") == FRONTEND_SEAL_PROTOCOL and seal.get("status") == "sealed", "frontend seal drift")
    expected_hashes = {
        "frontend_plan_sha256": sha256_file(Path(plan_path)),
        "detector_checkpoint_sha256": sha256_file(checkpoint_path),
        "detector_source_sha256": sha256_file(source_path),
        "training_summary_sha256": sha256_file(lineage["summary_path"]),
        "training_seal_sha256": sha256_file(lineage["seal_path"]),
        "run_intent_sha256": sha256_file(lineage["run_intent_path"]),
        "selection_claim_sha256": sha256_file(lineage["claim_path"]),
        "calibration_predictions_sha256": sha256_file(
            lineage["calibration_predictions_path"]
        ),
        "validation_predictions_sha256": sha256_file(
            lineage["validation_predictions_path"]
        ),
        "corpus_summary_sha256": sha256_file(lineage["corpus_summary_path"]),
        "corpus_seal_sha256": sha256_file(lineage["corpus_seal_path"]),
    }
    for key, expected in expected_hashes.items():
        _require(seal.get(key) == expected, f"frontend seal binding drift: {key}")
    return {
        "plan_path": Path(plan_path).resolve(strict=True),
        "plan": dict(plan),
        "seal_path": seal_path,
        "seal": seal,
        "training_lineage": lineage,
    }


def verify_frontend_plan(path: Path) -> tuple[Path, dict[str, Any]]:
    """Public API used by bundle materialization and final blind inference."""

    plan_path = Path(path).resolve(strict=True)
    plan = _json(plan_path)
    verify_public_frontend_lineage(plan, plan_path=plan_path)
    return plan_path, plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frontend-plan", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.freeze:
        path, plan = freeze_frontend(args.training_summary, args.output_dir)
    else:
        _require(args.frontend_plan is not None, "--frontend-plan is required with --verify")
        path, plan = verify_frontend_plan(args.frontend_plan)
    print(
        json.dumps(
            {"status": "verified" if args.verify else "frozen", "path": str(path), "plan": plan},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
