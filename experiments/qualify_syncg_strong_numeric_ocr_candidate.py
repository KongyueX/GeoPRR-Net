"""Qualify (but never finally select) Strong OCR on GARC outer calibration.

The artifact produced here is a prerequisite for presenting Strong to the
existing GARC calibration-only end-to-end selector.  It is not permitted to
override that selector and it never opens independent validation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.automatic_numeric_range_public_protocol import (
    canonical_sha256,
    require,
    sha256_file,
    strict_json,
)
from experiments.decide_syncg_strong_numeric_ocr_upgrade import (
    DECISION_PROTOCOL as ACTIVATION_DECISION_PROTOCOL,
    ZERO_ACCESS_KEYS,
)
from experiments.evaluate_syncg_ocr_garc_calibration import EVALUATION_PROTOCOL


QUALIFICATION_PROTOCOL: Final[str] = "syncg_strong_numeric_ocr_candidate_qualification_v1"
UPGRADE_PROTOCOL: Final[str] = "syncg_strong_numeric_ocr_upgrade_v2"
SAFE_OUTPUT_ROOT: Final[Path] = Path(r"C:\pointer_read").resolve()
DEFAULT_TINY_REPORT: Final[Path] = (
    SAFE_OUTPUT_ROOT / "syncg_garc_calibration_tiny_ocr_seed_20260817_v1/summary.json"
)
DEFAULT_STRONG_REPORT: Final[Path] = (
    SAFE_OUTPUT_ROOT / "syncg_garc_calibration_strong_ocr_seed_20260818_v1/summary.json"
)
DEFAULT_ACTIVATION: Final[Path] = SAFE_OUTPUT_ROOT / "syncg_strong_numeric_ocr_gate_v2/decision.json"
DEFAULT_PROTOCOL: Final[Path] = PROJECT_ROOT / "experiments/syncg_strong_numeric_ocr_upgrade_protocol.json"
DEFAULT_OUTPUT: Final[Path] = (
    SAFE_OUTPUT_ROOT / "syncg_strong_numeric_ocr_candidate_qualification_v1/decision.json"
)


def garc_selector_semantic_binding() -> dict[str, Any]:
    """Bind only the final selector callable/rules, not unrelated file edits."""

    from experiments.evaluate_garc_full_auto_public import (
        STRONG_GARC_PAIR_GAIN_MIN,
        STRONG_MAX_COVERAGE_REGRESSION,
        TOPK_GARC_PAIR_GAIN_MIN,
        select_recognizer,
    )

    payload = {
        "callable": "experiments.evaluate_garc_full_auto_public.select_recognizer",
        "output_protocol": "garc_numeric_recognizer_selection_v2",
        "source": inspect.getsource(select_recognizer),
        "thresholds": {
            "topk_pair_gain_min": float(TOPK_GARC_PAIR_GAIN_MIN),
            "strong_pair_gain_min": float(STRONG_GARC_PAIR_GAIN_MIN),
            "strong_max_coverage_regression": float(STRONG_MAX_COVERAGE_REGRESSION),
        },
    }
    return {**payload, "semantic_sha256": canonical_sha256(payload)}


def _safe_existing(path: Path, *, label: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must stay below {SAFE_OUTPUT_ROOT}") from error
    return resolved


def _fraction(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result) and 0.0 <= result <= 1.0, f"invalid {label}")
    return result


def _report(path: Path, *, kind: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = _safe_existing(path, label=f"{kind} calibration report")
    value = strict_json(resolved)
    require(value.get("protocol") == EVALUATION_PROTOCOL, f"{kind} report protocol drift")
    require(value.get("status") == "formal_calibration_component_evaluation_complete", f"{kind} report incomplete")
    require(value.get("mode") == "formal", f"{kind} report is not formal")
    require(value.get("partition") == "calibration", f"{kind} report partition drift")
    require(value.get("recognizer_kind") == kind, f"{kind} report recognizer drift")
    require(value.get("evaluation_kind") == "recognizer_oracle_text_boxes_component_only", f"{kind} report role drift")
    require(value.get(f"{kind}_summary") == value.get("recognizer_summary"), f"{kind} summary alias drift")
    require(value.get(f"{kind}_checkpoint") == value.get("recognizer_checkpoint"), f"{kind} checkpoint alias drift")
    expected_seed = 20260817 if kind == "tiny" else 20260818
    summary_binding = value["recognizer_summary"]
    checkpoint_binding = value["recognizer_checkpoint"]
    require(int(summary_binding.get("seed", -1)) == expected_seed, f"{kind} seed drift")
    summary_file = _safe_existing(Path(str(summary_binding.get("path") or "")), label=f"{kind} summary")
    checkpoint_file = _safe_existing(Path(str(checkpoint_binding.get("path") or "")), label=f"{kind} checkpoint")
    require(summary_binding.get("sha256") == sha256_file(summary_file), f"{kind} summary hash drift")
    require(checkpoint_binding.get("sha256") == sha256_file(checkpoint_file), f"{kind} checkpoint hash drift")
    code = value.get("code")
    evaluator = (PROJECT_ROOT / "experiments/evaluate_syncg_ocr_garc_calibration.py").resolve(strict=True)
    require(isinstance(code, Mapping), f"{kind} evaluator binding missing")
    require(Path(str(code.get("path") or "")).resolve(strict=True) == evaluator, f"{kind} evaluator path drift")
    require(code.get("sha256") == sha256_file(evaluator), f"{kind} evaluator source drift")
    component = value.get("component_corpus")
    require(isinstance(component, Mapping), f"{kind} component corpus missing")
    require((int(component.get("samples", -1)), int(component.get("groups", -1))) == (2224, 100), f"{kind} component inventory drift")
    roster = value.get("calibration_roster")
    require(isinstance(roster, Mapping), f"{kind} roster missing")
    require((int(roster.get("samples", -1)), int(roster.get("groups", -1))) == (2224, 100), f"{kind} roster inventory drift")
    require(component.get("sample_ids_sha256") == roster.get("sample_ids_sha256"), f"{kind} sample roster drift")
    require(component.get("group_ids_sha256") == roster.get("group_ids_sha256"), f"{kind} group roster drift")
    disjoint = value.get("disjointness")
    require(isinstance(disjoint, Mapping), f"{kind} disjointness missing")
    require(int(disjoint.get("sample_overlap", -1)) == 0 and int(disjoint.get("group_overlap", -1)) == 0, f"{kind} fit/calibration overlap")
    audit = value.get("data_access_audit")
    require(isinstance(audit, Mapping), f"{kind} access audit missing")
    require(int(audit.get("calibration_images_opened", -1)) == 2224, f"{kind} image inventory drift")
    require(int(audit.get("calibration_annotations_opened", -1)) == 2224, f"{kind} annotation inventory drift")
    for key in ZERO_ACCESS_KEYS:
        require(int(audit.get(key, -1)) == 0, f"{kind} forbidden access: {key}")
    metrics = value.get("metrics")
    require(isinstance(metrics, Mapping), f"{kind} metrics missing")
    observed = {
        "exact_accuracy": _fraction(metrics.get("exact_accuracy"), f"{kind} exact accuracy"),
        "character_accuracy": _fraction(metrics.get("character_accuracy"), f"{kind} character accuracy"),
        "parseable_fraction": _fraction(metrics.get("parseable_fraction"), f"{kind} parseable fraction"),
        "tokens": int(metrics.get("tokens", -1)),
        "source_images": int(metrics.get("source_images", -1)),
    }
    require(observed["tokens"] == int(component.get("tokens", -2)), f"{kind} token metric drift")
    require(observed["source_images"] == 2224, f"{kind} source image metric drift")
    return resolved, value, observed


def qualify(
    *, tiny_report_path: Path, strong_report_path: Path, activation_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve(strict=True)
    protocol = strict_json(protocol_file)
    require(protocol.get("protocol") == UPGRADE_PROTOCOL, "upgrade protocol drift")
    require(protocol.get("status") == "frozen_before_outer_calibration_metrics", "upgrade protocol not frozen")
    bindings = protocol.get("implementation_bindings")
    require(isinstance(bindings, Mapping), "implementation bindings missing")
    expected_sources = {
        "calibration_evaluator": PROJECT_ROOT / "experiments/evaluate_syncg_ocr_garc_calibration.py",
        "strong_candidate_qualifier": Path(__file__),
        "final_garc_selector": PROJECT_ROOT / "experiments/evaluate_garc_full_auto_public.py",
    }
    for name, source in expected_sources.items():
        source = Path(source).resolve(strict=True)
        binding = bindings.get(name)
        require(isinstance(binding, Mapping), f"missing implementation binding: {name}")
        require((PROJECT_ROOT / str(binding.get("path") or "")).resolve(strict=True) == source, f"{name} path drift")
        require(binding.get("sha256") == sha256_file(source), f"{name} source hash drift")
    selector_binding = bindings.get("final_garc_selector")
    require(isinstance(selector_binding, Mapping), "missing final selector binding")
    selector_semantics = garc_selector_semantic_binding()
    require(
        selector_binding.get("semantic_sha256") == selector_semantics["semantic_sha256"],
        "final GARC selector semantics drift",
    )
    require(
        selector_binding.get("output_protocol") == selector_semantics["output_protocol"],
        "final GARC selector output protocol drift",
    )
    rule = protocol.get("candidate_qualification")
    require(isinstance(rule, Mapping), "candidate qualification rule missing")
    require(rule.get("selection_role") == "candidate_eligibility_only_not_final_selection", "candidate role drift")
    require(rule.get("partition") == "garc_outer_calibration", "candidate partition drift")
    thresholds = rule.get("thresholds")
    require(isinstance(thresholds, Mapping), "candidate thresholds missing")
    exact_min = _fraction(thresholds.get("strong_exact_accuracy_gain_min"), "exact gain threshold")
    char_regression_max = _fraction(thresholds.get("maximum_character_accuracy_regression"), "character regression threshold")
    parse_regression_max = _fraction(thresholds.get("maximum_parseable_fraction_regression"), "parseable regression threshold")

    activation_file = _safe_existing(activation_path, label="Tiny activation decision")
    activation = strict_json(activation_file)
    require(activation.get("protocol") == ACTIVATION_DECISION_PROTOCOL, "activation protocol drift")
    require(activation.get("status") == "strong_recognizer_required", "Strong was not activated")
    require(activation.get("train_strong_recognizer") is True, "activation does not require Strong")
    tiny_file, tiny, tiny_metrics = _report(tiny_report_path, kind="tiny")
    strong_file, strong, strong_metrics = _report(strong_report_path, kind="strong")
    frozen = protocol.get("frozen_inputs", {})
    for key in ("summary_sha256", "seal_sha256", "samples_sha256", "tokens_sha256"):
        require(tiny.get("training_corpus", {}).get(key) == frozen.get("aligned_corpus", {}).get(key), f"frozen corpus {key} drift")
    for key in ("sha256", "sample_ids_sha256", "group_ids_sha256", "samples", "groups"):
        require(tiny.get("calibration_roster", {}).get(key) == frozen.get("garc_outer_calibration", {}).get(key), f"frozen calibration {key} drift")
    require(int(frozen.get("tiny_seed", -1)) == 20260817, "frozen Tiny seed drift")
    require(int(frozen.get("strong_seed_if_activated", -1)) == 20260818, "frozen Strong seed drift")
    require(
        activation.get("calibration_component_report", {}).get("sha256") == sha256_file(tiny_file),
        "activation/Tiny report hash drift",
    )
    require(activation.get("training_corpus") == tiny.get("training_corpus"), "activation/Tiny corpus drift")
    require(activation.get("tiny_summary") == tiny.get("tiny_summary"), "activation/Tiny summary drift")
    require(activation.get("tiny_checkpoint") == tiny.get("tiny_checkpoint"), "activation/Tiny checkpoint drift")
    for key in (
        "training_corpus", "parent_garc_protocol", "calibration_roster",
        "source_manifest", "disjointness", "component_corpus",
    ):
        require(tiny.get(key) == strong.get(key), f"Tiny/Strong {key} drift")

    deltas = {
        "exact_accuracy_gain": strong_metrics["exact_accuracy"] - tiny_metrics["exact_accuracy"],
        "character_accuracy_regression": tiny_metrics["character_accuracy"] - strong_metrics["character_accuracy"],
        "parseable_fraction_regression": tiny_metrics["parseable_fraction"] - strong_metrics["parseable_fraction"],
    }
    checks = {
        "strong_exact_accuracy_gain_min": deltas["exact_accuracy_gain"] >= exact_min,
        "character_accuracy_non_degradation": deltas["character_accuracy_regression"] <= char_regression_max,
        "parseable_fraction_non_degradation": deltas["parseable_fraction_regression"] <= parse_regression_max,
        "same_outer_calibration_roster_and_tokens": True,
        "independent_validation_access_zero": True,
    }
    eligible = all(checks.values())
    return {
        "schema_version": 1,
        "protocol": QUALIFICATION_PROTOCOL,
        "status": "strong_candidate_qualified" if eligible else "strong_candidate_rejected",
        "selection_role": "candidate_eligibility_only_not_final_selection",
        "strong_candidate_eligible_for_garc_calibration": eligible,
        "upgrade_protocol": {"path": str(protocol_file), "sha256": sha256_file(protocol_file)},
        "activation_decision": {"path": str(activation_file), "sha256": sha256_file(activation_file)},
        "tiny_report": {"path": str(tiny_file), "sha256": sha256_file(tiny_file)},
        "strong_report": {"path": str(strong_file), "sha256": sha256_file(strong_file)},
        "training_corpus": tiny["training_corpus"],
        "calibration_roster": tiny["calibration_roster"],
        "tiny_metrics": tiny_metrics,
        "strong_metrics": strong_metrics,
        "metric_deltas": deltas,
        "thresholds": dict(thresholds),
        "checks": checks,
        "downstream_contract": {
            "if_qualified": "Strong may be presented as an optional candidate to garc_numeric_recognizer_selection_v2",
            "if_rejected": "GARC final selector must be invoked without a Strong candidate",
            "final_authority": "garc_numeric_recognizer_selection_v2",
            "strong_is_never_selected_by_this_artifact": True,
        },
        "data_access_audit": {
            "images_opened_by_qualifier": 0,
            "annotations_opened_by_qualifier": 0,
            "independent_validation_opened": 0,
            "development_excluded_opened": 0,
            "joint_oof_412_19_opened": 0,
            "field_test_sealed_confirmatory_opened": 0,
        },
        "code": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    try:
        output.relative_to(SAFE_OUTPUT_ROOT)
    except ValueError as error:
        raise ValueError("qualification output escaped C:\\pointer_read") from error
    require(not output.exists(), f"refusing to overwrite qualification: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-report", type=Path, default=DEFAULT_TINY_REPORT)
    parser.add_argument("--strong-report", type=Path, default=DEFAULT_STRONG_REPORT)
    parser.add_argument("--activation", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = qualify(
        tiny_report_path=args.tiny_report,
        strong_report_path=args.strong_report,
        activation_path=args.activation,
        protocol_path=args.protocol,
    )
    _write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT",
    "QUALIFICATION_PROTOCOL",
    "garc_selector_semantic_binding",
    "qualify",
]
