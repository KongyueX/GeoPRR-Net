from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import materialize_field_blind_bundles as materializer
from experiments.field_blind_multimethod import METHOD_ROLES, load_method_roster
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    REFERENCE_MODE_AUTO,
    REFERENCE_MODE_NATIVE,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FACTORY = ROOT / "experiments" / "field_blind_runtime_factory.py"
ADAPTER_SOURCE = ROOT / "experiments" / "v5_unified_full_auto_adapter.py"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


class FieldBlindBundleMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "public-artifacts"
        self.artifacts.mkdir()
        self.progress_artifacts: dict[str, Path] = {}
        self.progress_sources: dict[str, Path] = {}
        self.source_bundles: dict[str, Path] = {}
        self.configs: dict[str, Path] = {}

        self.range_artifact = self.artifacts / "garc-range.bin"
        self.range_artifact.write_bytes(b"garc-range")
        self.v5_range_artifact = self.artifacts / "v5-range.bin"
        self.v5_range_artifact.write_bytes(b"v5-range")
        self.range_source = self.artifacts / "range-provider.py"
        self.range_source.write_text("RANGE_SOURCE = True\n", encoding="utf-8")
        self.reference_detector = self.artifacts / "reference-detector.pt"
        self.reference_detector.write_bytes(b"reference")
        self.pointer = self.artifacts / "pointer.pt"
        self.pointer.write_bytes(b"pointer")
        self.transformer = self.artifacts / "transformer.pt"
        self.transformer.write_bytes(b"transformer")
        self.external_factory = self.artifacts / "external-progress-factory.py"
        self.external_factory.write_text(
            "def build_progress_provider(_):\n    raise RuntimeError('metadata fixture')\n",
            encoding="utf-8",
        )

        self.garc_plan = self.artifacts / "garc-plan.json"
        _write_json(self.garc_plan, {"protocol": "synthetic_public_garc_plan_v1"})
        self.v5_plan = self.artifacts / "v5-plan.json"
        _write_json(self.v5_plan, {"protocol": "synthetic_public_v5_plan_v1"})

        common_range = FrozenComponentBinding(
            name="automatic_numeric_range",
            provider_protocol="garc_range_v1",
            provider_identity={"protocol": "garc_range_v1", "frozen": True},
            artifact_sha256={"checkpoint": sha256_file(self.range_artifact)},
            source_sha256={"provider": sha256_file(self.range_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        v5_range = FrozenComponentBinding(
            name="automatic_numeric_range",
            provider_protocol="v5_range_v1",
            provider_identity={"protocol": "v5_range_v1", "frozen": True},
            artifact_sha256={"checkpoint": sha256_file(self.v5_range_artifact)},
            source_sha256={"provider": sha256_file(self.range_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        source_names = {
            "garc_final": "GARC-source+auto-ref",
            "v5_complete": "V5-source+auto-ref",
            "pepd_shared_range": "PEPD-source+auto-ref",
            "vdn_shared_range": "VDN-source+auto-ref",
            "transformer_shared_range": "Original-Transformer-source",
        }
        for role in METHOD_ROLES:
            progress_artifact = self.artifacts / f"{role}-progress.bin"
            progress_artifact.write_bytes(role.encode("utf-8"))
            progress_source = self.artifacts / f"{role}-progress.py"
            progress_source.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
            self.progress_artifacts[role] = progress_artifact
            self.progress_sources[role] = progress_source
            progress = FrozenComponentBinding(
                name="progress",
                provider_protocol=f"{role}_progress_v1",
                provider_identity={"protocol": f"{role}_progress_v1"},
                artifact_sha256={"checkpoint": sha256_file(progress_artifact)},
                source_sha256={"provider": sha256_file(progress_source)},
                frozen=True,
                verified_complete=True,
                synthetic=False,
            )
            auto = role != "transformer_shared_range"
            bundle = FrozenFullAutoBundle.create(
                method_name=source_names[role],
                progress_binding=progress,
                range_binding=v5_range if role == "v5_complete" else common_range,
                factory_source_sha256=sha256_file(RUNTIME_FACTORY),
                reference_mode=REFERENCE_MODE_AUTO if auto else REFERENCE_MODE_NATIVE,
                reference_detector_sha256=(
                    sha256_file(self.reference_detector) if auto else None
                ),
                execution_mode=EXECUTION_FORMAL,
            )
            bundle_path = self.artifacts / f"{role}-source-bundle.json"
            bundle.write(bundle_path)
            self.source_bundles[role] = bundle_path

        self.component_files: dict[str, Path] = {}
        for role in ("pepd_shared_range", "vdn_shared_range"):
            component = FrozenFullAutoBundle.load(
                self.source_bundles[role]
            ).progress_binding
            component_path = self.artifacts / f"{role}-component.json"
            component.write(component_path)
            self.component_files[role] = component_path

        for role in METHOD_ROLES:
            if role == "v5_complete":
                config = {
                    "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
                    "mode": "v5_plan",
                    "source_plan": _binding(self.v5_plan),
                }
            elif role in {"garc_final"}:
                config = {
                    "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
                    "mode": "garc_plan",
                    "source_plan": _binding(self.garc_plan),
                }
            elif role in {"pepd_shared_range", "vdn_shared_range"}:
                config = {
                    "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
                    "mode": "shared_garc_range",
                    "source_plan": _binding(self.garc_plan),
                    "progress_kind": "external_progress_factory",
                    "progress_binding": _binding(self.component_files[role]),
                    "progress_factory": {
                        **_binding(self.external_factory),
                        "function": "build_progress_provider",
                    },
                }
            else:
                config = {
                    "protocol": materializer.RUNTIME_FACTORY_PROTOCOL,
                    "mode": "shared_garc_range",
                    "source_plan": _binding(self.garc_plan),
                    "progress_kind": "original_transformer",
                    "pointer_segmentation": _binding(self.pointer),
                    "original_transformer": _binding(self.transformer),
                    "device": "cpu",
                }
            config_path = self.artifacts / f"{role}-config.json"
            _write_json(config_path, config)
            self.configs[role] = config_path

        prediction_root = self.artifacts / "garc-validation-predictions"
        prediction_root.mkdir()
        garc_runtime_bundle = prediction_root / "bundle.json"
        garc_runtime_bundle.write_bytes(self.source_bundles["garc_final"].read_bytes())
        prediction_summary = prediction_root / "summary.json"
        _write_json(
            prediction_summary,
            {"artifacts": {"bundle": {"path": "bundle.json", "sha256": sha256_file(garc_runtime_bundle)}}},
        )
        # The source bundle path itself must be the path sealed by GARC.
        self.source_bundles["garc_final"] = garc_runtime_bundle
        validation = self.artifacts / "garc-validation.json"
        _write_json(
            validation,
            {
                "protocol": "garc_full_auto_public_validation_v1",
                "status": "independent_validation_complete",
                "mode": "formal",
                "claim_eligible": True,
                "validation_predictions": {
                    "path": str(prediction_root.resolve()),
                    "summary_sha256": sha256_file(prediction_summary),
                },
            },
        )
        self.garc_summary = self.artifacts / "garc-summary.json"
        _write_json(
            self.garc_summary,
            {
                "schema_version": 1,
                "protocol": materializer.GARC_SUMMARY_PROTOCOL,
                "status": "complete",
                "selected": {
                    "plan": str(self.garc_plan.resolve()),
                    "plan_sha256": sha256_file(self.garc_plan),
                },
                "validation": _binding(validation),
                "audit": {"restricted_namespace_images_opened": 0},
            },
        )
        self.paper_summary = self.artifacts / "paper-summary.json"
        _write_json(
            self.paper_summary,
            {
                "schema_version": 1,
                "protocol": materializer.PAPER_RESULTS_PROTOCOL,
                "status": "complete",
                "cohort": {"samples": 412, "groups": 19},
                "sources": {"garc_summary": _binding(self.garc_summary)},
                "audit": {
                    "public_truth_used_for_scoring_only": True,
                    "all_prediction_and_score_seals_verified_before_public_truth_opened": True,
                    "restricted_namespace_artifacts_opened": 0,
                },
            },
        )
        self.paper_seal = self.artifacts / "paper-seal.json"
        _write_json(
            self.paper_seal,
            {
                "schema_version": 1,
                "protocol": materializer.PAPER_RESULTS_PROTOCOL,
                "status": "sealed",
                "artifacts": {"summary": _binding(self.paper_summary)},
            },
        )

        detector_checkpoint = self.artifacts / "meter-detector.pt"
        detector_checkpoint.write_bytes(b"meter-detector")
        detector_source = self.artifacts / "meter-detector.py"
        detector_source.write_text("class targetDetectModel: pass\n", encoding="utf-8")
        self.frontend = self.artifacts / "frontend.json"
        _write_json(
            self.frontend,
            {
                "schema_version": 1,
                "protocol": "field_blind_shared_meter_frontend_v1",
                "status": "public_selected_frozen",
                "detector_checkpoint": _binding(detector_checkpoint),
                "detector_source": _binding(detector_source),
                "detector_class": "targetDetectModel",
                "confidence_threshold": 0.25,
                "padding_fraction": 0.05,
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
                    "detector_failure_penalty_nmae": 1.0,
                },
                "selection_audit": {
                    "public_data_only": True,
                    "field_manifest_opened": False,
                    "field_images_opened": False,
                    "field_labels_opened": False,
                },
            },
        )
        self.detector_checkpoint = detector_checkpoint
        self.detector_source = detector_source
        # This suite exercises bundle materialization, not the detector
        # lineage verifier.  The latter has its own public/synthetic checks and
        # formal artifacts do not exist during this metadata-only fixture.
        self.frontend_verifier = patch(
            "experiments.syncg_meter_detector_frontend.verify_frontend_plan",
            side_effect=lambda path: (
                Path(path).resolve(strict=True),
                json.loads(Path(path).read_text(encoding="utf-8")),
            ),
        )
        self.frontend_verifier.start()

    def tearDown(self) -> None:
        self.frontend_verifier.stop()
        self.temporary.cleanup()

    def _runtime_artifacts(self) -> list[Path]:
        paths = {
            RUNTIME_FACTORY,
            ADAPTER_SOURCE,
            self.range_artifact,
            self.v5_range_artifact,
            self.range_source,
            self.reference_detector,
            self.pointer,
            self.transformer,
            self.external_factory,
            self.garc_plan,
            self.v5_plan,
            *self.progress_artifacts.values(),
            *self.progress_sources.values(),
            *self.component_files.values(),
        }
        return sorted(paths, key=lambda path: str(path))

    def _freeze(self) -> Path:
        spec = self.root / "spec.json"
        return materializer.freeze_spec(
            garc_summary=self.garc_summary,
            paper_summary=self.paper_summary,
            paper_seal=self.paper_seal,
            frontend_plan=self.frontend,
            runtime_factory=RUNTIME_FACTORY,
            source_bundles=self.source_bundles,
            factory_configs=self.configs,
            runtime_artifacts=self._runtime_artifacts(),
            output=spec,
        )

    def test_materializes_exact_five_role_roster_without_image_access(self) -> None:
        spec = self._freeze()
        output = self.root / "materialized"
        result = materializer.materialize(spec_path=spec, output_root=output)
        self.assertTrue(result["verified"])
        _, roster, bundles = load_method_roster(output / "method_roster.json")
        self.assertEqual(set(bundles), set(METHOD_ROLES))
        garc_range = bundles["garc_final"].range_binding_sha256
        for role in (
            "pepd_shared_range",
            "vdn_shared_range",
            "transformer_shared_range",
        ):
            self.assertEqual(bundles[role].range_binding_sha256, garc_range)
        self.assertNotEqual(bundles["v5_complete"].range_binding_sha256, garc_range)
        self.assertEqual(
            roster["shared_frontend"]["detector_checkpoint"]["sha256"],
            sha256_file(self.detector_checkpoint),
        )
        self.assertFalse(roster["audit"]["field_images_opened"])

    def test_missing_preflight_is_exact_and_does_not_create_inputs(self) -> None:
        absent = self.root / "absent"
        paths = {
            "spec_path": absent / "spec.json",
            "garc_summary": absent / "garc.json",
            "paper_summary": absent / "paper.json",
            "paper_seal": absent / "seal.json",
        }
        result = materializer.preflight_requirements(**paths)
        self.assertFalse(result["ready"])
        self.assertEqual(
            [row["requirement"] for row in result["missing"]],
            [
                "materialization_spec",
                "garc_summary",
                "paper_results_summary",
                "paper_results_seal",
            ],
        )
        self.assertFalse(absent.exists())
        self.assertEqual(result["directory_enumerations"], 0)
        self.assertEqual(
            len(result["unresolved_dependencies_blocked_by_missing_spec"]),
            13,
        )

    def test_garc_source_bundle_must_equal_validation_sealed_bundle(self) -> None:
        wrong = self.artifacts / "wrong-garc-source.json"
        wrong.write_bytes((self.artifacts / "v5_complete-source-bundle.json").read_bytes())
        sources = dict(self.source_bundles)
        sources["garc_final"] = wrong
        with self.assertRaisesRegex(Exception, "garc_final source bundle"):
            materializer.freeze_spec(
                garc_summary=self.garc_summary,
                paper_summary=self.paper_summary,
                paper_seal=self.paper_seal,
                frontend_plan=self.frontend,
                runtime_factory=RUNTIME_FACTORY,
                source_bundles=sources,
                factory_configs=self.configs,
                runtime_artifacts=self._runtime_artifacts(),
                output=self.root / "bad-spec.json",
            )

    def test_missing_runtime_hash_fails_before_output_materialization(self) -> None:
        artifacts = [
            path for path in self._runtime_artifacts() if path != self.reference_detector
        ]
        with self.assertRaisesRegex(Exception, "runtime artifact"):
            materializer.freeze_spec(
                garc_summary=self.garc_summary,
                paper_summary=self.paper_summary,
                paper_seal=self.paper_seal,
                frontend_plan=self.frontend,
                runtime_factory=RUNTIME_FACTORY,
                source_bundles=self.source_bundles,
                factory_configs=self.configs,
                runtime_artifacts=artifacts,
                output=self.root / "missing-artifact-spec.json",
            )

    def test_restricted_external_namespace_is_rejected_lexically(self) -> None:
        forbidden = Path(r"C:\pointer_read\field\do-not-open.json")
        with self.assertRaisesRegex(Exception, "restricted external namespace"):
            materializer.preflight_requirements(spec_path=forbidden)


if __name__ == "__main__":
    unittest.main()
