"""Apply the frozen outer-calibration Tiny-to-Strong OCR activation gate.

The gate is deliberately incapable of selecting from the OCR corpus's inner
validation metrics.  Those metrics are retained as diagnostics only.  Formal
activation requires a separately sealed Tiny recognizer component evaluation
on the exact 2,224-image / 100-group GARC *outer calibration* roster.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    canonical_sha256,
    require,
    sha256_file,
    strict_json,
)
from experiments.evaluate_syncg_ocr_garc_calibration import (
    DEFAULT_CORPUS,
    DEFAULT_OUTPUT as DEFAULT_CALIBRATION_REPORT,
    DEFAULT_TINY_SUMMARY,
    EVALUATION_PROTOCOL,
    EXPECTED_CALIBRATION,
    EXPECTED_SEED,
    authenticate_metadata,
)
from experiments.syncg_numeric_ocr import CHECKPOINT_PROTOCOL


DECISION_PROTOCOL: Final[str] = "syncg_strong_numeric_ocr_gate_decision_v2"
UPGRADE_PROTOCOL: Final[str] = "syncg_strong_numeric_ocr_upgrade_v2"
SAFE_OUTPUT_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
DEFAULT_PROTOCOL: Final[Path] = (
    PROJECT_ROOT / "experiments/syncg_strong_numeric_ocr_upgrade_protocol.json"
)
DEFAULT_OUTPUT: Final[Path] = (
    SAFE_OUTPUT_ROOT / "syncg_strong_numeric_ocr_gate_v2/decision.json"
)
ZERO_ACCESS_KEYS: Final[tuple[str, ...]] = (
    "algorithm_fit_images_opened",
    "inner_validation_images_opened",
    "development_excluded_images_opened",
    "development_excluded_annotations_opened",
    "independent_validation_images_opened",
    "independent_validation_annotations_opened",
    "joint_oof_412_19_samples_opened",
    "field_samples_opened",
    "public_test_samples_opened",
    "sealed_samples_opened",
    "confirmatory_samples_opened",
)


def _safe_existing(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_OUTPUT_ROOT}") from error
    return resolved


def _safe_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"decision output must stay below {SAFE_OUTPUT_ROOT}") from error
    require(resolved != SAFE_OUTPUT_ROOT, "refusing broad decision output root")
    return resolved


def _finite_fraction(value: Any, label: str) -> float:
    observed = float(value)
    require(math.isfinite(observed) and 0.0 <= observed <= 1.0, f"{label} is invalid")
    return observed


def _load_protocol(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    protocol = strict_json(resolved)
    require(protocol.get("protocol") == UPGRADE_PROTOCOL, "upgrade protocol drift")
    require(protocol.get("status") == "frozen_before_outer_calibration_metrics", "upgrade protocol is not frozen")
    gate = protocol.get("frozen_activation_gate")
    require(isinstance(gate, Mapping), "frozen activation gate missing")
    require(gate.get("selection_partition") == "garc_outer_calibration", "selection partition drift")
    require(
        gate.get("algorithm_fit_inner_validation_role") == "diagnostic_only_not_a_gate_input",
        "inner validation role drift",
    )
    require(
        gate.get("independent_validation_role") == "unopened_until_final_model_freeze",
        "independent validation role drift",
    )
    thresholds = gate.get("activate_strong_recognizer_if_any")
    require(isinstance(thresholds, Mapping), "recognizer thresholds missing")
    require(
        set(thresholds)
        == {
            "garc_calibration_exact_accuracy_below",
            "garc_calibration_character_accuracy_below",
            "garc_calibration_parseable_fraction_below",
        },
        "activation threshold schema drift",
    )
    for key, value in thresholds.items():
        _finite_fraction(value, key)
    bindings = protocol.get("implementation_bindings")
    require(isinstance(bindings, Mapping), "implementation bindings missing")
    expected_sources = {
        "calibration_evaluator": (
            PROJECT_ROOT / "experiments/evaluate_syncg_ocr_garc_calibration.py"
        ).resolve(strict=True),
        "activation_decider": Path(__file__).resolve(strict=True),
    }
    for key, source in expected_sources.items():
        binding = bindings.get(key)
        require(isinstance(binding, Mapping), f"missing implementation binding: {key}")
        declared = (PROJECT_ROOT / str(binding.get("path") or "")).resolve(strict=True)
        require(declared == source, f"{key} path drift")
        require(binding.get("sha256") == sha256_file(source), f"{key} source hash drift")
    return resolved, protocol


def _expected_inputs(protocol: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
    expected = protocol.get("frozen_inputs")
    require(isinstance(expected, Mapping), "frozen inputs missing")
    require(int(expected.get("tiny_seed", -1)) == EXPECTED_SEED, "frozen Tiny seed drift")
    corpus = metadata["training_corpus"]
    require(
        Path(str(expected.get("aligned_corpus", {}).get("root") or "")).resolve(strict=True)
        == Path(str(corpus["root"])).resolve(strict=True),
        "frozen corpus root drift",
    )
    for key in ("summary_sha256", "seal_sha256", "samples_sha256", "tokens_sha256"):
        require(expected.get("aligned_corpus", {}).get(key) == corpus[key], f"frozen corpus {key} drift")
    require(
        int(expected.get("aligned_corpus", {}).get("samples", -1)) == 12_176
        and int(expected.get("aligned_corpus", {}).get("groups", -1)) == 551,
        "frozen fit inventory drift",
    )
    roster = metadata["calibration_roster"]
    require(
        Path(str(expected.get("garc_outer_calibration", {}).get("path") or "")).resolve(strict=True)
        == Path(str(roster["path"])).resolve(strict=True),
        "frozen calibration roster path drift",
    )
    for key in ("sha256", "sample_ids_sha256", "group_ids_sha256"):
        require(expected.get("garc_outer_calibration", {}).get(key) == roster[key], f"frozen calibration {key} drift")
    require(
        (
            int(expected.get("garc_outer_calibration", {}).get("samples", -1)),
            int(expected.get("garc_outer_calibration", {}).get("groups", -1)),
        )
        == EXPECTED_CALIBRATION,
        "frozen calibration inventory drift",
    )
    require(
        expected.get("parent_garc_protocol_sha256")
        == metadata["parent_garc_protocol"]["sha256"],
        "frozen parent GARC protocol drift",
    )


def _load_report(
    path: Path, *, metadata: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = _safe_existing(path, label="GARC calibration component report")
    report = strict_json(resolved)
    require(report.get("protocol") == EVALUATION_PROTOCOL, "calibration report protocol drift")
    require(report.get("status") == "formal_calibration_component_evaluation_complete", "calibration report incomplete")
    require(report.get("mode") == "formal", "calibration report is not formal")
    require(report.get("partition") == "calibration", "calibration report partition drift")
    require(report.get("recognizer_kind") == "tiny", "activation report is not Tiny")
    require(
        report.get("evaluation_kind") == "recognizer_oracle_text_boxes_component_only",
        "calibration report is not the recognizer component evaluation",
    )
    for key in (
        "training_corpus", "tiny_summary", "tiny_checkpoint", "parent_garc_protocol",
        "calibration_roster", "disjointness",
    ):
        require(report.get(key) == metadata[key], f"calibration report {key} binding drift")
    code = report.get("code")
    evaluator = (PROJECT_ROOT / "experiments/evaluate_syncg_ocr_garc_calibration.py").resolve(strict=True)
    require(isinstance(code, Mapping), "calibration evaluator binding missing")
    require(Path(str(code.get("path") or "")).resolve(strict=True) == evaluator, "calibration evaluator path drift")
    require(code.get("sha256") == sha256_file(evaluator), "calibration evaluator source drift")
    component = report.get("component_corpus")
    require(isinstance(component, Mapping), "calibration component corpus missing")
    require(
        (int(component.get("samples", -1)), int(component.get("groups", -1)))
        == EXPECTED_CALIBRATION,
        "calibration component inventory drift",
    )
    require(
        component.get("sample_ids_sha256") == metadata["calibration_roster"]["sample_ids_sha256"]
        and component.get("group_ids_sha256") == metadata["calibration_roster"]["group_ids_sha256"],
        "calibration component roster drift",
    )
    for key in ("token_ids_sha256", "annotation_content_inventory_sha256"):
        value = str(component.get(key) or "")
        require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"bad {key}")
    require(int(component.get("tokens", 0)) > 0, "empty calibration token inventory")
    disjoint = report["disjointness"]
    require(int(disjoint.get("sample_overlap", -1)) == 0, "fit/calibration sample overlap")
    require(int(disjoint.get("group_overlap", -1)) == 0, "fit/calibration group overlap")
    audit = report.get("data_access_audit")
    require(isinstance(audit, Mapping), "calibration data-access audit missing")
    require(int(audit.get("calibration_images_opened", -1)) == EXPECTED_CALIBRATION[0], "calibration image count drift")
    require(int(audit.get("calibration_annotations_opened", -1)) == EXPECTED_CALIBRATION[0], "calibration annotation count drift")
    for key in ZERO_ACCESS_KEYS:
        require(int(audit.get(key, -1)) == 0, f"forbidden data access: {key}")
    metrics = report.get("metrics")
    require(isinstance(metrics, Mapping), "calibration metrics missing")
    observed = {
        "exact_accuracy": _finite_fraction(metrics.get("exact_accuracy"), "calibration exact accuracy"),
        "character_accuracy": _finite_fraction(metrics.get("character_accuracy"), "calibration character accuracy"),
        "parseable_fraction": _finite_fraction(metrics.get("parseable_fraction"), "calibration parseable fraction"),
        "tokens": int(metrics.get("tokens", -1)),
        "source_images": int(metrics.get("source_images", -1)),
    }
    require(observed["tokens"] == int(component["tokens"]), "calibration metric token count drift")
    require(observed["source_images"] == EXPECTED_CALIBRATION[0], "calibration metric image count drift")
    return resolved, report, observed


def _inner_diagnostic(metadata: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(metadata["tiny_checkpoint"]["path"])
    value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(value, Mapping), "Tiny checkpoint is not a mapping")
    require(value.get("protocol") == CHECKPOINT_PROTOCOL, "Tiny checkpoint protocol drift")
    validation = value.get("metrics", {}).get("validation")
    require(isinstance(validation, Mapping), "Tiny inner validation metrics missing")
    return {
        "role": "diagnostic_only_not_used_by_activation_checks",
        "partition": "inner_validation_within_garc_algorithm_fit",
        "exact_accuracy": _finite_fraction(validation.get("exact_accuracy"), "inner exact accuracy"),
        "character_accuracy": _finite_fraction(validation.get("character_accuracy"), "inner character accuracy"),
        "parseable_fraction": _finite_fraction(validation.get("parseable_fraction"), "inner parseable fraction"),
        "tokens": int(validation.get("tokens", -1)),
        "source_images": int(validation.get("source_images", -1)),
    }


def decide(
    *, corpus_root: Path, tiny_summary_path: Path, calibration_report_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> Mapping[str, Any]:
    protocol_file, protocol = _load_protocol(protocol_path)
    metadata = authenticate_metadata(
        corpus_root=corpus_root, tiny_summary_path=tiny_summary_path
    )
    _expected_inputs(protocol, metadata)
    report_file, _, observed = _load_report(calibration_report_path, metadata=metadata)
    thresholds = protocol["frozen_activation_gate"]["activate_strong_recognizer_if_any"]
    checks = {
        "garc_calibration_exact_accuracy_below": observed["exact_accuracy"]
        < float(thresholds["garc_calibration_exact_accuracy_below"]),
        "garc_calibration_character_accuracy_below": observed["character_accuracy"]
        < float(thresholds["garc_calibration_character_accuracy_below"]),
        "garc_calibration_parseable_fraction_below": observed["parseable_fraction"]
        < float(thresholds["garc_calibration_parseable_fraction_below"]),
    }
    triggered = [key for key, value in checks.items() if value]
    return {
        "schema_version": 2,
        "protocol": DECISION_PROTOCOL,
        "status": "strong_recognizer_required" if triggered else "tiny_component_gate_pass",
        "selection_partition": "garc_outer_calibration",
        "upgrade_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "training_corpus": metadata["training_corpus"],
        "tiny_summary": metadata["tiny_summary"],
        "tiny_checkpoint": metadata["tiny_checkpoint"],
        "calibration_component_report": {
            "path": str(report_file), "sha256": sha256_file(report_file),
            "protocol": EVALUATION_PROTOCOL,
        },
        "calibration_roster": metadata["calibration_roster"],
        "garc_outer_calibration": observed,
        "algorithm_fit_inner_validation": _inner_diagnostic(metadata),
        "frozen_gate": protocol["frozen_activation_gate"],
        "frozen_gate_sha256": canonical_sha256(protocol["frozen_activation_gate"]),
        "component_checks": checks,
        "triggered_component_checks": triggered,
        "train_strong_recognizer": bool(triggered),
        "gpu_work_started": False,
        "data_scope": (
            "decision reads authenticated metadata/checkpoints/report only; activation metrics "
            "come exclusively from GARC outer calibration"
        ),
        "data_access_audit": {
            "images_opened_by_decision": 0,
            "annotations_opened_by_decision": 0,
            "independent_validation_opened_by_decision": 0,
            "development_excluded_opened_by_decision": 0,
            "joint_oof_412_19_opened_by_decision": 0,
            "field_test_sealed_confirmatory_opened_by_decision": 0,
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    resolved = _safe_output(path)
    require(not resolved.exists(), f"refusing to overwrite gate decision: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--tiny-summary", type=Path, default=DEFAULT_TINY_SUMMARY)
    parser.add_argument("--calibration-report", type=Path, default=DEFAULT_CALIBRATION_REPORT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = decide(
        corpus_root=args.corpus,
        tiny_summary_path=args.tiny_summary,
        calibration_report_path=args.calibration_report,
        protocol_path=args.protocol,
    )
    _atomic_json(args.output, decision)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = ["DECISION_PROTOCOL", "DEFAULT_OUTPUT", "decide"]
