"""Prepare every public-only input required by the five-method bundle freezer.

This is the missing bridge between completed public experiments and
``materialize_field_blind_bundles.freeze_spec``.  It is deliberately unable to
accept a field dataset, image manifest, labels, crop, ScaleMark, or numeric
range.  It consumes only four exact authorities:

* the completed GARC public summary;
* the completed/sealed paper-result summary;
* the public SyncG meter-frontend plan; and
* the repository's audited runtime implementations.

The builder authenticates the validation-sealed GARC bundle, recovers the
calibration-sealed pure-V5 bundle selected before independent validation,
reuses PEPD seed 20260720, and derives VDN seed 20260720 plus Original
Transformer progress identities from their sealed same-input public runs.  It
then writes five source-bundle identities, five runtime-factory configs, a
complete exact-file catalog, and invokes the existing ``freeze_spec`` entry
point.  PEPD/VDN/Transformer are configured to reuse the selected GARC range;
no ground-truth range is accepted anywhere in this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import garc_pepd_progress_factory as pepd_factory
from experiments import garc_vdn_progress_factory as vdn_factory
from experiments import materialize_field_blind_bundles as materializer
from experiments.field_blind_multimethod import METHOD_ROLES
from experiments.garc_full_auto_public import load_plan, load_prediction_bundle
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    REFERENCE_MODE_AUTO,
    REFERENCE_MODE_NATIVE,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    sha256_file,
)
from experiments.v5_unified_full_auto_progress_providers import (
    PROTOCOL as PROGRESS_PROVIDER_PROTOCOL,
)
from experiments.v5_unified_legacy_adapters import FrozenLegacyTaskConfig
from experiments.v5_unified_two_stage_retest import (
    assert_label_free,
    canonical_json_bytes,
    canonical_json_sha256,
    strict_json_load,
)


PROTOCOL: Final[str] = "field_blind_bundle_input_builder_v1"
CATALOG_PROTOCOL: Final[str] = "field_blind_runtime_artifact_catalog_v1"
INVENTORY_PROTOCOL: Final[str] = "field_blind_source_bundle_inventory_v1"
TARGET_SEED: Final[int] = 20260720
EXTERNAL_SCORE_PROTOCOL: Final[str] = "garc_external_progress_412_score_v1"
EXTERNAL_PREDICTION_PROTOCOL: Final[str] = (
    "garc_external_progress_412_predictions_v1"
)
EXTERNAL_PROTOCOL: Final[str] = "garc_external_progress_412_comparison_v1"
RECOGNIZER_SELECTION_PROTOCOL: Final[str] = "garc_numeric_recognizer_selection_v2"
CALIBRATION_PROTOCOL: Final[str] = "garc_full_auto_public_acceptance_v1"

DEFAULT_GARC_SUMMARY: Final[Path] = Path(
    r"C:\pointer_read\garc_full_auto_formal_v1\summary.json"
)
DEFAULT_PAPER_ROOT: Final[Path] = Path(r"C:\pointer_read\paper_final_results_v2")
DEFAULT_FRONTEND_PLAN: Final[Path] = Path(
    r"C:\pointer_read\syncg_meter_detector_frontend_v1\frontend_plan.json"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path(
    r"C:\pointer_read\blind_bundle_materialization_v1"
)
RUNTIME_FACTORY: Final[Path] = Path(__file__).with_name(
    "field_blind_runtime_factory.py"
).resolve()
ADAPTER_SOURCE: Final[Path] = Path(__file__).with_name(
    "v5_unified_full_auto_adapter.py"
).resolve()
PROGRESS_WRAPPER_SOURCE: Final[Path] = Path(__file__).with_name(
    "v5_unified_full_auto_progress_providers.py"
).resolve()
DIRECTION_ADAPTER_SOURCE: Final[Path] = Path(__file__).with_name(
    "v5_unified_direction_adapters.py"
).resolve()
LEGACY_ADAPTER_SOURCE: Final[Path] = Path(__file__).with_name(
    "v5_unified_legacy_adapters.py"
).resolve()
REFERENCE_LOADER_SOURCE: Final[Path] = Path(__file__).with_name(
    "evaluate_vdn_baseline.py"
).resolve()
PRODUCTION_PIPELINE_SOURCE: Final[Path] = (
    PROJECT_ROOT / "utils" / "angleDetect" / "zeroShotMeter.py"
).resolve()


class InputPreparationError(ValueError):
    """Fail-closed public-authority or immutable-output error."""


class SchemaGapError(InputPreparationError):
    """The future upstream result exists but lacks required bindings."""

    def __init__(self, gaps: Sequence[str]):
        self.gaps = tuple(sorted({str(value) for value in gaps}))
        super().__init__("upstream schema gaps: " + ", ".join(self.gaps))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputPreparationError(message)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").casefold()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _allowed_roots() -> tuple[Path, ...]:
    """Exact roots that may contain public/model authority files.

    In particular, the repository's ``data`` tree is intentionally absent.
    Keeping this as a function lets synthetic unit tests patch the allow-list
    without exposing a command-line escape hatch in the formal builder.
    """

    return (
        Path(r"C:\pointer_read"),
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "utils",
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath(
            [str(path).casefold(), str(root).casefold()]
        ) == str(root).casefold()
    except ValueError:
        return False


def _enforce_allowed_root(path: Path, *, label: str) -> None:
    allowed = tuple(Path(root).absolute() for root in _allowed_roots())
    _require(
        any(_inside(Path(path).absolute(), root) for root in allowed),
        f"{label} is outside the public/model authority roots: {path}",
    )


def _guard(path: Path, *, label: str, must_exist: bool = True) -> Path:
    """Lexically guard, root-allow-list, then recheck the resolved target."""

    guarded = materializer._guard_metadata_path(Path(path), label=label)
    _enforce_allowed_root(guarded, label=label)
    if not must_exist:
        return guarded
    resolved = guarded.resolve(strict=True)
    _enforce_allowed_root(resolved, label=f"resolved {label}")
    _require(resolved.is_file(), f"{label} is not an exact file: {resolved}")
    return resolved


def _guard_directory(path: Path, *, label: str) -> Path:
    guarded = materializer._guard_metadata_path(Path(path), label=label)
    _enforce_allowed_root(guarded, label=label)
    resolved = guarded.resolve(strict=True)
    _enforce_allowed_root(resolved, label=f"resolved {label}")
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def _json(path: Path, *, label: str) -> dict[str, Any]:
    source = _guard(path, label=label)
    value = strict_json_load(source)
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _binding(
    value: Any,
    *,
    label: str,
    relative_root: Path | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise SchemaGapError([f"{label}.{{path,sha256}}"])
    raw_text = str(value.get("path") or "")
    digest = str(value.get("sha256") or "").casefold()
    if not raw_text or not _is_sha256(digest):
        raise SchemaGapError([f"{label}.{{path,sha256}}"])
    raw = Path(raw_text)
    if not raw.is_absolute():
        if relative_root is None:
            raise InputPreparationError(f"{label}.path must be absolute")
        raw = relative_root / raw
    path = _guard(raw, label=label)
    _require(sha256_file(path) == digest, f"{label} hash drift")
    return {"path": str(path), "sha256": digest}


def _directory_summary(
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], Path]:
    if not isinstance(value, Mapping):
        raise SchemaGapError([f"{label}.{{path,summary_sha256,bundle_sha256}}"])
    raw = Path(str(value.get("path") or ""))
    if not raw.is_absolute():
        raise InputPreparationError(f"{label}.path must be absolute")
    root = _guard_directory(raw, label=label)
    summary_path = _guard(root / "summary.json", label=f"{label} summary")
    seal_path = _guard(root / "seal.json", label=f"{label} seal")
    _require(
        sha256_file(summary_path) == str(value.get("summary_sha256") or "").casefold(),
        f"{label} summary hash drift",
    )
    summary = _json(summary_path, label=f"{label} summary")
    seal = _json(seal_path, label=f"{label} seal")
    _require(
        seal.get("protocol") == EXTERNAL_PREDICTION_PROTOCOL
        and seal.get("status") == "sealed"
        and str(seal.get("summary_sha256") or "").casefold()
        == sha256_file(summary_path),
        f"{label} summary seal drift",
    )
    _require(
        str(seal.get("bundle_sha256") or "").casefold()
        == str(value.get("bundle_sha256") or "").casefold(),
        f"{label} bundle seal drift",
    )
    return root, summary, seal_path


def _write_new_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"immutable builder artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    payload = canonical_json_bytes(value)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _path_from_protocol(value: Any, *, label: str) -> Path:
    return Path(
        _binding(value, label=label, relative_root=PROJECT_ROOT)["path"]
    )


def _walk_exact_bindings(
    value: Any,
    *,
    location: str,
    relative_root: Path | None = None,
) -> list[Path]:
    """Resolve only explicit ``{path, sha256}`` records; never enumerate."""

    found: list[Path] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            found.append(
                Path(
                    _binding(
                        value,
                        label=location,
                        relative_root=relative_root,
                    )["path"]
                )
            )
        for key, nested in value.items():
            found.extend(
                _walk_exact_bindings(
                    nested,
                    location=f"{location}.{key}",
                    relative_root=relative_root,
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            found.extend(
                _walk_exact_bindings(
                    nested,
                    location=f"{location}[{index}]",
                    relative_root=relative_root,
                )
            )
    return found


def _schema_gaps(
    garc: Mapping[str, Any],
    paper: Mapping[str, Any],
    seal: Mapping[str, Any],
    frontend: Mapping[str, Any],
) -> list[str]:
    gaps: list[str] = []

    def need_binding(container: Mapping[str, Any], key: str, location: str) -> None:
        value = container.get(key)
        if not (
            isinstance(value, Mapping)
            and bool(str(value.get("path") or ""))
            and _is_sha256(value.get("sha256"))
        ):
            gaps.append(f"{location}.{{path,sha256}}")

    need_binding(garc, "validation", "garc.validation")
    selected = garc.get("selected")
    if not isinstance(selected, Mapping):
        gaps.append("garc.selected.{plan,plan_sha256}")
    elif not bool(str(selected.get("plan") or "")) or not _is_sha256(
        selected.get("plan_sha256")
    ):
        gaps.append("garc.selected.{plan,plan_sha256}")
    need_binding(garc, "recognizer_selection", "garc.recognizer_selection")
    sources = paper.get("sources")
    if not isinstance(sources, Mapping):
        gaps.extend(
            ["paper.sources.garc_summary", "paper.sources.external_comparison"]
        )
    else:
        need_binding(sources, "garc_summary", "paper.sources.garc_summary")
        need_binding(
            sources,
            "external_comparison",
            "paper.sources.external_comparison",
        )
    artifacts = seal.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("summary"), Mapping
    ):
        gaps.append("paper_seal.artifacts.summary")
    need_binding(frontend, "detector_checkpoint", "frontend.detector_checkpoint")
    need_binding(frontend, "detector_source", "frontend.detector_source")
    selection = frontend.get("selection_audit")
    lineage = (
        "training_summary",
        "training_seal",
        "selection_claim",
        "corpus_summary",
        "corpus_seal",
    )
    if not isinstance(selection, Mapping):
        gaps.append("frontend.selection_audit")
    else:
        for key in lineage:
            need_binding(selection, key, f"frontend.selection_audit.{key}")
    return sorted(set(gaps))


def _candidate_for_selected_recognizer(
    decision: Mapping[str, Any],
) -> Mapping[str, Any]:
    recognizer = str(decision.get("selected_recognizer") or "")
    consensus = str(decision.get("selected_consensus") or "")
    key = (
        "strong_topk"
        if recognizer == "strong"
        else "tiny_topk" if consensus == "topk" else "tiny_top1"
    )
    value = decision.get(key)
    if not isinstance(value, Mapping):
        raise SchemaGapError([f"garc.recognizer_selection.{key}"])
    return value


def _identity_from_external_summary(
    summary: Mapping[str, Any],
    *,
    route: str,
    label: str,
) -> dict[str, Any]:
    identities = summary.get("provider_identities")
    if not isinstance(identities, Mapping) or not isinstance(
        identities.get(route), Mapping
    ):
        raise SchemaGapError([f"{label}.provider_identities.{route}"])
    record = identities[route]
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise SchemaGapError([f"{label}.provider_identities.{route}.identity"])
    normalized = dict(identity)
    _require(
        canonical_json_sha256(normalized)
        == str(record.get("identity_sha256") or "").casefold(),
        f"{label} provider identity hash drift",
    )
    return normalized


@dataclass(frozen=True)
class PreparedAuthorities:
    garc_summary: Path
    paper_summary: Path
    paper_seal: Path
    frontend_plan: Path
    frontend_seal: Path
    final_plan_path: Path
    final_plan: Mapping[str, Any]
    base_v5_plan_path: Path
    base_v5_plan: Mapping[str, Any]
    garc_bundle_path: Path
    garc_bundle: FrozenFullAutoBundle
    v5_bundle_path: Path
    v5_bundle: FrozenFullAutoBundle
    pepd_progress: FrozenComponentBinding
    vdn_identity: Mapping[str, Any]
    vdn_checkpoint: Path
    vdn_verification: Path
    vdn_reference_detector: Path
    transformer_identity: Mapping[str, Any]
    transformer_checkpoint: Path
    transformer_pointer_segmentation: Path
    transformer_task_config: Mapping[str, Any]
    external_protocol_path: Path
    artifact_candidates: tuple[Path, ...]


def _load_authorities(
    *,
    garc_summary: Path,
    paper_summary: Path,
    paper_seal: Path,
    frontend_plan: Path,
) -> PreparedAuthorities:
    """Authenticate public metadata only; no model or image is instantiated."""

    garc_path = _guard(garc_summary, label="GARC summary")
    paper_path = _guard(paper_summary, label="paper summary")
    seal_path = _guard(paper_seal, label="paper seal")
    frontend_path = _guard(frontend_plan, label="public meter frontend plan")
    garc_value = _json(garc_path, label="GARC summary")
    paper_value = _json(paper_path, label="paper summary")
    seal_value = _json(seal_path, label="paper seal")
    frontend_value = _json(frontend_path, label="public meter frontend plan")
    gaps = _schema_gaps(garc_value, paper_value, seal_value, frontend_value)
    if gaps:
        raise SchemaGapError(gaps)

    garc_auth = materializer._authenticate_garc_summary(garc_path)
    materializer._authenticate_paper(paper_path, seal_path, garc_auth)
    frontend_binding = {"path": str(frontend_path), "sha256": sha256_file(frontend_path)}
    materializer._load_frontend_guarded(frontend_binding)
    frontend_seal = _guard(
        frontend_path.parent / "seal.json", label="public meter frontend seal"
    )

    final_plan_binding = _binding(
        {
            "path": garc_value["selected"]["plan"],
            "sha256": garc_value["selected"]["plan_sha256"],
        },
        label="selected GARC plan",
    )
    final_plan_path, final_plan = load_plan(Path(final_plan_binding["path"]))
    _require(final_plan["execution_mode"] == EXECUTION_FORMAL, "GARC plan is not formal")
    validation_value = _json(
        Path(garc_auth["validation"]["path"]), label="GARC validation report"
    )
    _require(
        _binding(
            validation_value.get("parent_plan"),
            label="GARC validation parent plan",
        )
        == final_plan_binding,
        "validation-sealed GARC bundle uses another selected plan",
    )

    selection_binding = _binding(
        garc_value["recognizer_selection"], label="GARC recognizer selection"
    )
    selection = _json(
        Path(selection_binding["path"]), label="GARC recognizer selection"
    )
    _require(
        selection.get("protocol") == RECOGNIZER_SELECTION_PROTOCOL
        and selection.get("status") == "frozen_calibration_only_selection",
        "GARC recognizer selection authority drift",
    )
    candidate = _candidate_for_selected_recognizer(selection)
    base_plan_binding = _binding(
        {"path": candidate.get("plan"), "sha256": candidate.get("plan_sha256")},
        label="selected pure-V5 plan",
    )
    base_calibration_binding = _binding(
        {
            "path": candidate.get("calibration"),
            "sha256": candidate.get("calibration_sha256"),
        },
        label="selected pure-V5 calibration",
    )
    _require(
        str(Path(selection.get("selected_plan") or "").resolve())
        == base_plan_binding["path"],
        "recognizer selection/base V5 plan drift",
    )
    _require(
        str(Path(selection.get("selected_calibration") or "").resolve())
        == base_calibration_binding["path"],
        "recognizer selection/base V5 calibration drift",
    )
    base_plan_path, base_plan = load_plan(Path(base_plan_binding["path"]))
    _require(base_plan["execution_mode"] == EXECUTION_FORMAL, "V5 plan is not formal")
    _require(base_plan["garc"]["geometry_mode"] == "v5", "base plan is not pure V5")
    _require(
        base_plan["garc"]["recognizer_kind"]
        == final_plan["garc"]["recognizer_kind"]
        == selection["selected_recognizer"]
        and base_plan["garc"]["consensus_mode"]
        == final_plan["garc"]["consensus_mode"]
        == selection["selected_consensus"],
        "V5 and final GARC do not share the selected OCR/consensus",
    )
    for plan_name, plan in (("GARC", final_plan), ("V5", base_plan)):
        _require(
            plan["garc"].get("geometry_provider") == "enhanced_v5_oof_fold"
            and int(plan["garc"]["geometry_fold"].get("pepd_seed", -1))
            == TARGET_SEED,
            f"{plan_name} plan is not the frozen seed-{TARGET_SEED} route",
        )
    _require(
        base_plan["progress_component"]["binding"]
        == final_plan["progress_component"]["binding"],
        "V5/GARC seed-20 progress binding drift",
    )
    _require(
        base_plan["garc"]["artifacts"]["detector"]
        == final_plan["garc"]["artifacts"]["detector"]
        and base_plan["garc"]["artifacts"]["recognizer"]
        == final_plan["garc"]["artifacts"]["recognizer"],
        "V5/GARC selected OCR checkpoints drift",
    )

    pepd_seed, pepd_row = pepd_factory._seed_from_plan(base_plan)
    _require(pepd_seed == TARGET_SEED, "base V5 does not use PEPD seed 20260720")
    base_progress = FrozenComponentBinding.from_record(
        base_plan["progress_component"]["binding"]
    )
    _require(
        base_progress.artifact_sha256.get("checkpoint")
        == pepd_row["checkpoint_sha256"],
        "PEPD seed-20 checkpoint binding drift",
    )

    calibration = _json(
        Path(base_calibration_binding["path"]), label="pure-V5 calibration"
    )
    _require(
        calibration.get("protocol") == CALIBRATION_PROTOCOL
        and calibration.get("status") == "frozen_range_acceptance"
        and calibration.get("mode") == "formal"
        and calibration.get("claim_eligible") is True,
        "pure-V5 calibration is not formal/eligible",
    )
    _require(
        _binding(calibration.get("parent_plan"), label="V5 calibration parent plan")
        == base_plan_binding,
        "V5 calibration parent-plan drift",
    )
    prediction = calibration.get("prediction_bundle")
    if not isinstance(prediction, Mapping) or not bool(str(prediction.get("path") or "")):
        raise SchemaGapError(["garc.recognizer_selection.selected_calibration.prediction_bundle"])
    prediction_root_raw = Path(str(prediction["path"]))
    _require(prediction_root_raw.is_absolute(), "V5 prediction root must be absolute")
    prediction_root = _guard_directory(
        prediction_root_raw, label="pure-V5 calibration prediction root"
    )
    v5_prediction_summary, _, _, v5_calibration_bundle = load_prediction_bundle(
        plan_path=base_plan_path,
        prediction_root=prediction_root,
        expected_partition="calibration",
        allow_smoke=False,
    )
    _require(
        sha256_file(prediction_root / "summary.json")
        == str(prediction.get("summary_sha256") or "").casefold(),
        "V5 calibration prediction summary drift",
    )
    v5_bundle_binding = _binding(
        {
            "path": str(
                prediction_root
                / str(v5_prediction_summary["artifacts"]["bundle"]["path"])
            ),
            "sha256": v5_prediction_summary["artifacts"]["bundle"]["sha256"],
        },
        label="calibration-sealed pure-V5 bundle",
    )
    _require(
        v5_calibration_bundle.progress_binding.binding_sha256
        == base_progress.binding_sha256,
        "V5 calibration bundle progress drift",
    )
    _require(
        v5_calibration_bundle.range_binding_sha256
        == calibration["prediction_bundle"]["range_binding_sha256"],
        "V5 calibration range binding drift",
    )

    garc_bundle_path = Path(garc_auth["runtime_bundle"]["path"])
    garc_bundle = FrozenFullAutoBundle.load(garc_bundle_path)
    validation_predictions = validation_value.get("validation_predictions") or {}
    validation_prediction_raw = Path(
        str(validation_predictions.get("path") or "")
    )
    _require(
        validation_prediction_raw.is_absolute(),
        "GARC validation prediction root must be absolute",
    )
    validation_prediction_root = _guard_directory(
        validation_prediction_raw,
        label="GARC validation prediction root",
    )
    _require(
        validation_prediction_root == garc_bundle_path.parent,
        "GARC validation bundle escaped the validated prediction root",
    )
    _require(
        garc_bundle.progress_binding.as_record()
        == final_plan["progress_component"]["binding"],
        "validation-sealed GARC progress differs from selected plan",
    )
    _require(
        garc_bundle.reference_detector_sha256
        == final_plan["reference"]["detector_sha256"],
        "validation-sealed GARC reference detector drift",
    )
    _require(
        str(validation_predictions.get("range_binding_sha256") or "").casefold()
        == garc_bundle.range_binding_sha256,
        "validation report/GARC bundle range binding drift",
    )

    external_binding = _binding(
        paper_value["sources"]["external_comparison"],
        label="paper-bound external comparison",
    )
    external_score = _json(
        Path(external_binding["path"]), label="paper-bound external comparison"
    )
    _require(
        external_score.get("protocol") == EXTERNAL_SCORE_PROTOCOL
        and external_score.get("status") == "complete",
        "external comparison is incomplete",
    )
    _require(
        str((external_score.get("handoff") or {}).get("range_binding_sha256") or "").casefold()
        == garc_bundle.range_binding_sha256,
        "paper external comparison used another GARC range binding",
    )
    eligibility = external_score.get("claim_eligibility") or {}
    _require(
        eligibility.get("vdn_strict_progress_oof_claim") is True
        and eligibility.get("original_transformer_strict_oof_claim") is False,
        "external comparator claim roles drift",
    )
    external_protocol_binding = _binding(
        external_score.get("frozen_protocol"), label="external frozen protocol"
    )
    external_protocol_path = Path(external_protocol_binding["path"])
    external_protocol = _json(
        external_protocol_path, label="external frozen protocol"
    )
    _require(
        external_protocol.get("protocol") == EXTERNAL_PROTOCOL
        and external_protocol.get("status") == "frozen_before_external_inference",
        "external protocol drift",
    )
    sources = external_score.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {
        "vdn_official200",
        "original_transformer",
    }:
        raise SchemaGapError(
            [
                "paper.external_comparison.sources.vdn_official200",
                "paper.external_comparison.sources.original_transformer",
            ]
        )
    vdn_prediction_root, vdn_prediction_summary, vdn_prediction_seal = _directory_summary(
        sources["vdn_official200"], label="VDN public prediction bundle"
    )
    (
        transformer_prediction_root,
        transformer_prediction_summary,
        transformer_prediction_seal,
    ) = _directory_summary(
        sources["original_transformer"], label="Transformer public prediction bundle"
    )
    for method, summary in (
        ("vdn_official200", vdn_prediction_summary),
        ("original_transformer", transformer_prediction_summary),
    ):
        _require(
            summary.get("protocol") == EXTERNAL_PREDICTION_PROTOCOL
            and summary.get("status") == "predictions_sealed"
            and summary.get("method") == method,
            f"{method} public prediction authority drift",
        )
        _require(
            _binding(summary.get("frozen_protocol"), label=f"{method} frozen protocol")
            == external_protocol_binding,
            f"{method} used another external protocol",
        )

    vdn_identity = _identity_from_external_summary(
        vdn_prediction_summary,
        route=str(TARGET_SEED),
        label="VDN public predictions",
    )
    transformer_identity = _identity_from_external_summary(
        transformer_prediction_summary,
        route="global",
        label="Transformer public predictions",
    )
    _require(
        vdn_identity.get("protocol") == PROGRESS_PROVIDER_PROTOCOL
        and vdn_identity.get("provider") == "vdn_official200",
        "VDN provider identity drift",
    )
    _require(
        transformer_identity.get("protocol") == PROGRESS_PROVIDER_PROTOCOL
        and transformer_identity.get("provider")
        == "original_transformer_native_progress",
        "Transformer provider identity drift",
    )

    vdn_section = external_protocol.get("vdn_official200")
    transformer_section = external_protocol.get("original_transformer")
    if not isinstance(vdn_section, Mapping) or not isinstance(
        transformer_section, Mapping
    ):
        raise SchemaGapError(
            [
                "external_protocol.vdn_official200",
                "external_protocol.original_transformer",
            ]
        )
    routes = vdn_section.get("routes")
    if not isinstance(routes, Mapping) or not isinstance(
        routes.get(str(TARGET_SEED)), Mapping
    ):
        raise SchemaGapError(
            [f"external_protocol.vdn_official200.routes.{TARGET_SEED}"]
        )
    vdn_route = routes[str(TARGET_SEED)]
    vdn_checkpoint = _path_from_protocol(
        vdn_route.get("checkpoint"), label="VDN seed-20 checkpoint"
    )
    vdn_verification = _path_from_protocol(
        vdn_route.get("verification"), label="VDN seed-20 verification"
    )
    vdn_reference = _path_from_protocol(
        vdn_section.get("automatic_reference_detector"),
        label="VDN automatic reference detector",
    )
    vdn_rows = vdn_factory.authoritative_rows()
    vdn_row = vdn_rows[TARGET_SEED]
    _require(
        sha256_file(vdn_checkpoint) == vdn_row["checkpoint_sha256"]
        == vdn_identity.get("checkpoint_sha256")
        and sha256_file(vdn_verification) == vdn_row["verification_sha256"]
        == vdn_identity.get("verification_sha256"),
        "VDN seed-20 public result/factory binding drift",
    )
    _require(
        sha256_file(vdn_reference)
        == vdn_identity.get("reference_detector_sha256")
        == garc_bundle.reference_detector_sha256,
        "VDN/GARC automatic reference detector drift",
    )
    _require(
        vdn_identity.get("direction_adapter_source_sha256")
        == sha256_file(DIRECTION_ADAPTER_SOURCE)
        and vdn_identity.get("wrapper_source_sha256")
        == sha256_file(PROGRESS_WRAPPER_SOURCE)
        and (vdn_identity.get("automatic_reference") or {}).get(
            "detector_loader_source_sha256"
        )
        == sha256_file(REFERENCE_LOADER_SOURCE),
        "VDN runtime source identity drift",
    )

    transformer_checkpoint = _path_from_protocol(
        transformer_section.get("transformer_checkpoint"),
        label="Original Transformer checkpoint",
    )
    pointer_section = transformer_section.get("pointer_segmentation")
    if not isinstance(pointer_section, Mapping):
        raise SchemaGapError(
            ["external_protocol.original_transformer.pointer_segmentation"]
        )
    transformer_pointer = _path_from_protocol(
        pointer_section, label="Transformer pointer segmentation"
    )
    transformer_components = transformer_identity.get("component_sha256") or {}
    _require(
        transformer_components.get("original_transformer")
        == sha256_file(transformer_checkpoint)
        and transformer_components.get("pointer_segmentation")
        == sha256_file(transformer_pointer)
        and transformer_components.get("production_pipeline_source")
        == sha256_file(PRODUCTION_PIPELINE_SOURCE),
        "Transformer checkpoint/source identity drift",
    )
    _require(
        transformer_identity.get("legacy_adapter_source_sha256")
        == sha256_file(LEGACY_ADAPTER_SOURCE)
        and transformer_identity.get("wrapper_source_sha256")
        == sha256_file(PROGRESS_WRAPPER_SOURCE)
        and transformer_identity.get("reference_detector_loaded") is False
        and transformer_identity.get("reference_detector_invoked") is False,
        "Transformer native-progress identity drift",
    )
    transformer_config = asdict(FrozenLegacyTaskConfig())
    _require(
        transformer_identity.get("task_config_sha256")
        == canonical_json_sha256(transformer_config),
        "Transformer task configuration differs from the public comparison",
    )

    pepd_sources = dict(base_progress.source_sha256)
    pepd_progress = FrozenComponentBinding(
        name=base_progress.name,
        provider_protocol=base_progress.provider_protocol,
        provider_identity=base_progress.provider_identity,
        artifact_sha256=base_progress.artifact_sha256,
        source_sha256={
            **pepd_sources,
            "reference_detector_loader": sha256_file(REFERENCE_LOADER_SOURCE),
        },
        frozen=True,
        verified_complete=True,
        synthetic=False,
    )

    candidates: set[Path] = {
        garc_path,
        paper_path,
        seal_path,
        frontend_path,
        frontend_seal,
        final_plan_path,
        base_plan_path,
        Path(selection_binding["path"]),
        Path(base_calibration_binding["path"]),
        prediction_root / "summary.json",
        prediction_root / "seal.json",
        Path(v5_bundle_binding["path"]),
        garc_bundle_path,
        garc_bundle_path.parent / "summary.json",
        garc_bundle_path.parent / "seal.json",
        Path(external_binding["path"]),
        external_protocol_path,
        vdn_prediction_root / "summary.json",
        vdn_prediction_seal,
        transformer_prediction_root / "summary.json",
        transformer_prediction_seal,
        vdn_checkpoint,
        vdn_verification,
        vdn_reference,
        transformer_checkpoint,
        transformer_pointer,
        RUNTIME_FACTORY,
        ADAPTER_SOURCE,
        pepd_factory.SOURCE,
        pepd_factory.WRAPPER_SOURCE,
        pepd_factory.DIRECTION_ADAPTER_SOURCE,
        pepd_factory.HANDOFF,
        Path(pepd_row["checkpoint"]),
        Path(pepd_row["verification"]),
        pepd_factory.REFERENCE_DETECTOR,
        vdn_factory.SOURCE,
        vdn_factory.WRAPPER_SOURCE,
        vdn_factory.DIRECTION_ADAPTER_SOURCE,
        vdn_factory.OOF_SUMMARY,
        PROGRESS_WRAPPER_SOURCE,
        DIRECTION_ADAPTER_SOURCE,
        LEGACY_ADAPTER_SOURCE,
        REFERENCE_LOADER_SOURCE,
        PRODUCTION_PIPELINE_SOURCE,
    }
    for value, location, root in (
        (garc_value, "garc_summary", None),
        (validation_value, "garc_validation", None),
        (paper_value, "paper_summary", None),
        (seal_value, "paper_seal", None),
        (frontend_value, "frontend_plan", None),
        (final_plan, "selected_garc_plan", None),
        (base_plan, "pure_v5_plan", None),
        (external_score, "external_score", None),
        (external_protocol, "external_protocol", PROJECT_ROOT),
        (vdn_prediction_summary, "vdn_predictions", vdn_prediction_root),
        (
            transformer_prediction_summary,
            "transformer_predictions",
            transformer_prediction_root,
        ),
    ):
        candidates.update(
            _walk_exact_bindings(value, location=location, relative_root=root)
        )
    checked_candidates = tuple(
        sorted(
            {_guard(path, label="runtime artifact candidate") for path in candidates},
            key=lambda path: str(path).casefold(),
        )
    )
    return PreparedAuthorities(
        garc_summary=garc_path,
        paper_summary=paper_path,
        paper_seal=seal_path,
        frontend_plan=frontend_path,
        frontend_seal=frontend_seal,
        final_plan_path=final_plan_path,
        final_plan=final_plan,
        base_v5_plan_path=base_plan_path,
        base_v5_plan=base_plan,
        garc_bundle_path=garc_bundle_path,
        garc_bundle=garc_bundle,
        v5_bundle_path=Path(v5_bundle_binding["path"]),
        v5_bundle=v5_calibration_bundle,
        pepd_progress=pepd_progress,
        vdn_identity=vdn_identity,
        vdn_checkpoint=vdn_checkpoint,
        vdn_verification=vdn_verification,
        vdn_reference_detector=vdn_reference,
        transformer_identity=transformer_identity,
        transformer_checkpoint=transformer_checkpoint,
        transformer_pointer_segmentation=transformer_pointer,
        transformer_task_config=transformer_config,
        external_protocol_path=external_protocol_path,
        artifact_candidates=checked_candidates,
    )


class _ArtifactCatalog:
    def __init__(self) -> None:
        self._by_path: dict[Path, str] = {}

    def add(self, path: Path, *, label: str = "runtime artifact") -> None:
        checked = _guard(path, label=label)
        digest = sha256_file(checked)
        previous = self._by_path.get(checked)
        _require(previous in (None, digest), f"{label} path hash changed")
        self._by_path[checked] = digest

    def add_staged(self, staged: Path, final: Path, *, label: str) -> None:
        source = Path(staged).resolve(strict=True)
        _require(source.is_file(), f"{label} staging artifact is not a file")
        target = _guard(Path(final), label=label, must_exist=False)
        self._by_path[target] = sha256_file(source)

    @property
    def records(self) -> list[dict[str, str]]:
        return [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(
                self._by_path.items(), key=lambda row: str(row[0]).casefold()
            )
        ]

    @property
    def paths(self) -> list[Path]:
        return [Path(record["path"]) for record in self.records]

    @property
    def hashes(self) -> set[str]:
        return set(self._by_path.values())


def _make_vdn_binding(authority: PreparedAuthorities) -> FrozenComponentBinding:
    return FrozenComponentBinding(
        name="progress",
        provider_protocol=PROGRESS_PROVIDER_PROTOCOL,
        provider_identity=authority.vdn_identity,
        artifact_sha256={
            "checkpoint": sha256_file(authority.vdn_checkpoint),
            "verification": sha256_file(authority.vdn_verification),
            "reference_detector": sha256_file(authority.vdn_reference_detector),
        },
        source_sha256={
            "factory": sha256_file(vdn_factory.SOURCE),
            "progress_wrapper": sha256_file(PROGRESS_WRAPPER_SOURCE),
            "direction_adapter": sha256_file(DIRECTION_ADAPTER_SOURCE),
            "reference_detector_loader": sha256_file(REFERENCE_LOADER_SOURCE),
            "oof_summary": sha256_file(vdn_factory.OOF_SUMMARY),
        },
        frozen=True,
        verified_complete=True,
        synthetic=False,
    )


def _make_transformer_binding(
    authority: PreparedAuthorities,
) -> FrozenComponentBinding:
    return FrozenComponentBinding(
        name="progress",
        provider_protocol=PROGRESS_PROVIDER_PROTOCOL,
        provider_identity=authority.transformer_identity,
        artifact_sha256={
            "pointer_segmentation": sha256_file(
                authority.transformer_pointer_segmentation
            ),
            "original_transformer": sha256_file(
                authority.transformer_checkpoint
            ),
        },
        source_sha256={
            "progress_wrapper": sha256_file(PROGRESS_WRAPPER_SOURCE),
            "legacy_adapter": sha256_file(LEGACY_ADAPTER_SOURCE),
            "production_pipeline": sha256_file(PRODUCTION_PIPELINE_SOURCE),
        },
        frozen=True,
        verified_complete=True,
        synthetic=False,
    )


def _source_bundle(
    *,
    method_name: str,
    progress: FrozenComponentBinding,
    numeric_range: FrozenComponentBinding,
    reference_mode: str,
    reference_detector_sha256: str | None,
) -> FrozenFullAutoBundle:
    return FrozenFullAutoBundle.create(
        method_name=method_name,
        progress_binding=progress,
        range_binding=numeric_range,
        factory_source_sha256=sha256_file(RUNTIME_FACTORY),
        reference_mode=reference_mode,
        reference_detector_sha256=reference_detector_sha256,
        execution_mode=EXECUTION_FORMAL,
    )


def _write_prepared_inputs(
    *,
    authority: PreparedAuthorities,
    staging_root: Path,
    final_root: Path,
) -> tuple[
    dict[str, Path],
    dict[str, Path],
    list[Path],
    Path,
    Path,
]:
    staging = Path(staging_root).resolve(strict=True)
    final = _guard(final_root, label="builder output root", must_exist=False)
    catalog = _ArtifactCatalog()
    for path in authority.artifact_candidates:
        catalog.add(path)

    component_stage = staging / "components"
    component_final = final / "components"
    pepd_component = component_stage / "pepd_seed_20260720.progress.json"
    vdn_component = component_stage / "vdn_seed_20260720.progress.json"
    transformer_component = component_stage / "original_transformer.progress.json"
    authority.pepd_progress.write(pepd_component)
    vdn_progress = _make_vdn_binding(authority)
    vdn_progress.write(vdn_component)
    transformer_progress = _make_transformer_binding(authority)
    transformer_progress.write(transformer_component)

    garc_range = authority.garc_bundle.range_binding
    reference_sha = authority.garc_bundle.reference_detector_sha256
    _require(reference_sha is not None, "GARC automatic reference binding absent")
    generated = {
        "v5_complete": _source_bundle(
            method_name="V5-source+auto-ref",
            progress=authority.v5_bundle.progress_binding,
            numeric_range=authority.v5_bundle.range_binding,
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=authority.v5_bundle.reference_detector_sha256,
        ),
        "pepd_shared_range": _source_bundle(
            method_name="PEPD-source+auto-ref",
            progress=authority.pepd_progress,
            numeric_range=garc_range,
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=reference_sha,
        ),
        "vdn_shared_range": _source_bundle(
            method_name="VDN-source+auto-ref",
            progress=vdn_progress,
            numeric_range=garc_range,
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=sha256_file(
                authority.vdn_reference_detector
            ),
        ),
        "transformer_shared_range": _source_bundle(
            method_name="Original-Transformer-source",
            progress=transformer_progress,
            numeric_range=garc_range,
            reference_mode=REFERENCE_MODE_NATIVE,
            reference_detector_sha256=None,
        ),
    }
    source_stage = staging / "source_bundles"
    source_final = final / "source_bundles"
    source_bundles: dict[str, Path] = {
        "garc_final": authority.garc_bundle_path,
    }
    for role, bundle in generated.items():
        stage_path = source_stage / f"{role}.bundle.json"
        bundle.write(stage_path)
        source_bundles[role] = source_final / stage_path.name

    final_components = {
        "pepd_shared_range": component_final / pepd_component.name,
        "vdn_shared_range": component_final / vdn_component.name,
        "transformer_shared_range": component_final / transformer_component.name,
    }
    pepd_progress_factory = {
        "path": str(pepd_factory.SOURCE.resolve(strict=True)),
        "sha256": sha256_file(pepd_factory.SOURCE),
        "function": "build_progress_provider",
    }
    vdn_progress_factory = {
        "path": str(vdn_factory.SOURCE.resolve(strict=True)),
        "sha256": sha256_file(vdn_factory.SOURCE),
        "function": "build_progress_provider",
    }
    final_plan_binding = {
        "path": str(authority.final_plan_path),
        "sha256": sha256_file(authority.final_plan_path),
    }
    base_plan_binding = {
        "path": str(authority.base_v5_plan_path),
        "sha256": sha256_file(authority.base_v5_plan_path),
    }
    configs: dict[str, dict[str, Any]] = {
        "garc_final": {
            "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
            "mode": "garc_plan",
            "source_plan": final_plan_binding,
        },
        "v5_complete": {
            "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
            "mode": "v5_plan",
            "source_plan": base_plan_binding,
        },
        "pepd_shared_range": {
            "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
            "mode": "shared_garc_range",
            "source_plan": final_plan_binding,
            "progress_kind": "external_progress_factory",
            "progress_binding": {
                "path": str(final_components["pepd_shared_range"]),
                "sha256": authority.pepd_progress.binding_sha256,
            },
            "progress_factory": pepd_progress_factory,
        },
        "vdn_shared_range": {
            "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
            "mode": "shared_garc_range",
            "source_plan": final_plan_binding,
            "progress_kind": "external_progress_factory",
            "progress_binding": {
                "path": str(final_components["vdn_shared_range"]),
                "sha256": vdn_progress.binding_sha256,
            },
            "progress_factory": vdn_progress_factory,
        },
        "transformer_shared_range": {
            "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
            "mode": "shared_garc_range",
            "source_plan": final_plan_binding,
            "progress_kind": "original_transformer",
            "pointer_segmentation": {
                "path": str(authority.transformer_pointer_segmentation),
                "sha256": sha256_file(
                    authority.transformer_pointer_segmentation
                ),
            },
            "original_transformer": {
                "path": str(authority.transformer_checkpoint),
                "sha256": sha256_file(authority.transformer_checkpoint),
            },
            "device": str(authority.final_plan["garc"]["device"]),
            "task_config": dict(authority.transformer_task_config),
        },
    }
    config_stage = staging / "factory_configs"
    config_final = final / "factory_configs"
    factory_configs: dict[str, Path] = {}
    for role in METHOD_ROLES:
        assert_label_free(configs[role], location=f"bundle_input_builder.{role}")
        stage_path = config_stage / f"{role}.factory.json"
        _write_new_json(stage_path, configs[role])
        factory_configs[role] = config_final / stage_path.name

    for stage_path, final_path, label in (
        (pepd_component, final_components["pepd_shared_range"], "PEPD component"),
        (vdn_component, final_components["vdn_shared_range"], "VDN component"),
        (
            transformer_component,
            final_components["transformer_shared_range"],
            "Transformer component",
        ),
    ):
        catalog.add_staged(stage_path, final_path, label=label)
    for role in generated:
        catalog.add_staged(
            source_stage / f"{role}.bundle.json",
            source_bundles[role],
            label=f"{role} source bundle",
        )
    catalog.add(authority.garc_bundle_path, label="validation-sealed GARC bundle")
    for role in METHOD_ROLES:
        catalog.add_staged(
            config_stage / f"{role}.factory.json",
            factory_configs[role],
            label=f"{role} factory config",
        )

    prospective_sources = {
        "garc_final": authority.garc_bundle,
        **generated,
    }
    required_hashes: set[str] = set()
    for role, source in prospective_sources.items():
        prospective = FrozenFullAutoBundle.create(
            method_name=materializer.ROLE_METHOD_NAMES[role],
            progress_binding=source.progress_binding,
            range_binding=(
                source.range_binding if role == "v5_complete" else garc_range
            ),
            factory_source_sha256=sha256_file(RUNTIME_FACTORY),
            reference_mode=source.reference_mode,
            reference_detector_sha256=source.reference_detector_sha256,
            execution_mode=EXECUTION_FORMAL,
        )
        required_hashes.update(materializer._required_runtime_hashes(prospective))
    missing_hashes = sorted(required_hashes - catalog.hashes)
    _require(
        not missing_hashes,
        "runtime catalog cannot resolve component/provider hashes: "
        + ", ".join(missing_hashes),
    )

    inventory_path = staging / "source_bundle_inventory.json"
    inventory_final = final / inventory_path.name
    inventory = {
        "schema_version": 1,
        "protocol": INVENTORY_PROTOCOL,
        "status": "five_public_source_bundles_resolved",
        "target_seed": TARGET_SEED,
        "methods": {
            role: {
                "path": str(source_bundles[role]),
                "sha256": (
                    sha256_file(authority.garc_bundle_path)
                    if role == "garc_final"
                    else FrozenFullAutoBundle.load(
                        source_stage / f"{role}.bundle.json"
                    ).bundle_sha256
                ),
                "authority": (
                    "GARC validation-sealed bundle"
                    if role == "garc_final"
                    else "public-result-derived immutable source bundle"
                ),
            }
            for role in METHOD_ROLES
        },
        "contracts": {
            "garc_final_uses_validation_sealed_selected_plan": True,
            "v5_uses_selected_ocr_consensus_and_pure_v5_geometry": True,
            "pepd_seed": TARGET_SEED,
            "vdn_seed": TARGET_SEED,
            "transformer_is_fixed_checkpoint_sensitivity": True,
            "pepd_vdn_transformer_share_garc_range_at_materialization": True,
            "ground_truth_or_manual_range_used": False,
        },
        "audit": {
            "providers_instantiated": False,
            "images_opened": 0,
            "directory_enumerations": 0,
            "restricted_namespace_artifacts_opened": 0,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
        },
    }
    assert_label_free(inventory, location="bundle_input_inventory")
    _write_new_json(inventory_path, inventory)
    catalog.add_staged(inventory_path, inventory_final, label="source inventory")

    catalog_path = staging / "runtime_artifact_catalog.json"
    catalog_final = final / catalog_path.name
    catalog_value = {
        "schema_version": 1,
        "protocol": CATALOG_PROTOCOL,
        "status": "public_runtime_artifacts_resolved",
        "artifacts": catalog.records,
        "coverage": {
            "required_component_hashes": len(required_hashes),
            "all_required_component_hashes_resolved": True,
            "unresolved_hashes": [],
        },
        "audit": {
            "public_model_metadata_only": True,
            "providers_instantiated": False,
            "images_opened": 0,
            "directory_enumerations": 0,
            "restricted_namespace_artifacts_opened": 0,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
        },
    }
    assert_label_free(catalog_value, location="runtime_artifact_catalog")
    _write_new_json(catalog_path, catalog_value)
    catalog.add_staged(catalog_path, catalog_final, label="runtime catalog manifest")
    return (
        source_bundles,
        factory_configs,
        catalog.paths,
        inventory_final,
        catalog_final,
    )


def build_inputs(
    *,
    garc_summary: Path,
    paper_summary: Path,
    paper_seal: Path,
    frontend_plan: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build all 13 unresolved inputs and invoke the existing freeze-spec."""

    output = _guard(
        output_root, label="bundle-input builder output", must_exist=False
    )
    _require(not output.exists(), f"immutable builder output exists: {output}")
    authority = _load_authorities(
        garc_summary=garc_summary,
        paper_summary=paper_summary,
        paper_seal=paper_seal,
        frontend_plan=frontend_plan,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    _require(not staging.exists(), f"builder staging root exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    moved = False
    try:
        (
            source_bundles,
            factory_configs,
            runtime_artifacts,
            inventory_path,
            catalog_path,
        ) = _write_prepared_inputs(
            authority=authority,
            staging_root=staging,
            final_root=output,
        )
        os.replace(staging, output)
        moved = True
        spec_path = output / "spec.json"
        materializer.freeze_spec(
            garc_summary=authority.garc_summary,
            paper_summary=authority.paper_summary,
            paper_seal=authority.paper_seal,
            frontend_plan=authority.frontend_plan,
            runtime_factory=RUNTIME_FACTORY,
            source_bundles=source_bundles,
            factory_configs=factory_configs,
            runtime_artifacts=runtime_artifacts,
            output=spec_path,
        )
        checked = materializer.validate_spec(spec_path)
        _require(set(checked["sources"]) == set(METHOD_ROLES), "frozen spec roster drift")
        result = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "complete",
            "created_at": _now(),
            "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
            "source_bundle_inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
            },
            "runtime_artifact_catalog": {
                "path": str(catalog_path),
                "sha256": sha256_file(catalog_path),
            },
            "methods": list(METHOD_ROLES),
            "target_seed": TARGET_SEED,
            "audit": {
                "providers_instantiated": False,
                "images_opened": 0,
                "directory_enumerations": 0,
                "restricted_namespace_artifacts_opened": 0,
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
                "feishu_messages_sent": 0,
            },
        }
        assert_label_free(result, location="bundle_input_builder_result")
        return result
    except Exception:
        # Preserve a failed attempt without overwriting the immutable target.
        current = output if moved else staging
        if current.exists():
            failed = output.with_name(f".{output.name}.failed.{os.getpid()}")
            if not failed.exists():
                os.replace(current, failed)
        raise


def preflight(
    *,
    garc_summary: Path,
    paper_summary: Path,
    paper_seal: Path,
    frontend_plan: Path,
    output_root: Path,
) -> dict[str, Any]:
    inputs = {
        "garc_summary": Path(garc_summary),
        "paper_summary": Path(paper_summary),
        "paper_seal": Path(paper_seal),
        "frontend_plan": Path(frontend_plan),
    }
    missing: list[dict[str, str]] = []
    for name, raw in inputs.items():
        path = _guard(raw, label=name, must_exist=False)
        if not path.is_file():
            missing.append({"requirement": name, "path": str(path)})
    output = _guard(output_root, label="builder output root", must_exist=False)
    if output.exists():
        missing.append({"requirement": "unused_output_root", "path": str(output)})
    schema_gaps: list[str] = []
    validation_error: str | None = None
    if not missing:
        try:
            _load_authorities(
                garc_summary=inputs["garc_summary"],
                paper_summary=inputs["paper_summary"],
                paper_seal=inputs["paper_seal"],
                frontend_plan=inputs["frontend_plan"],
            )
        except SchemaGapError as error:
            schema_gaps.extend(error.gaps)
        except Exception as error:  # surfaced as exact fail-closed diagnostic
            validation_error = f"{type(error).__name__}: {error}"
    ready = not missing and not schema_gaps and validation_error is None
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "missing": missing,
        "schema_gaps": sorted(set(schema_gaps)),
        "validation_error": validation_error,
        "would_create": {
            "source_bundles": 5,
            "factory_configs": 5,
            "runtime_artifact_catalog": 1,
            "freeze_spec": str(output / "spec.json"),
        },
        "audit": {
            "writes": 0,
            "providers_instantiated": False,
            "images_opened": 0,
            "directory_enumerations": 0,
            "restricted_namespace_artifacts_opened": 0,
            "field_manifest_opened": False,
            "field_images_opened": False,
            "field_labels_opened": False,
            "feishu_messages_sent": 0,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "build"):
        command = commands.add_parser(name)
        command.add_argument("--garc-summary", type=Path, default=DEFAULT_GARC_SUMMARY)
        command.add_argument(
            "--paper-summary",
            type=Path,
            default=DEFAULT_PAPER_ROOT / "summary.json",
        )
        command.add_argument(
            "--paper-seal", type=Path, default=DEFAULT_PAPER_ROOT / "seal.json"
        )
        command.add_argument(
            "--frontend-plan", type=Path, default=DEFAULT_FRONTEND_PLAN
        )
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        value = preflight(
            garc_summary=args.garc_summary,
            paper_summary=args.paper_summary,
            paper_seal=args.paper_seal,
            frontend_plan=args.frontend_plan,
            output_root=args.output_root,
        )
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0 if value["ready"] else 2
    result = build_inputs(
        garc_summary=args.garc_summary,
        paper_summary=args.paper_summary,
        paper_seal=args.paper_seal,
        frontend_plan=args.frontend_plan,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOG_PROTOCOL",
    "DEFAULT_FRONTEND_PLAN",
    "DEFAULT_GARC_SUMMARY",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PAPER_ROOT",
    "INVENTORY_PROTOCOL",
    "InputPreparationError",
    "PROTOCOL",
    "PreparedAuthorities",
    "SchemaGapError",
    "TARGET_SEED",
    "build_inputs",
    "preflight",
]
