from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import field_blind_multimethod as multi
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    EVIDENCE_ROLE_PRIMARY,
    OUTPUT_MODE_FULL_READING,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
)
from experiments.v5_unified_two_stage_retest import REFERENCE_MODE_NATIVE


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeDetector:
    def __init__(self, rows: list[tuple[float, list[list[float]], int]]) -> None:
        self.rows = rows
        self.calls = 0

    def target_detection(self, image, confidence=None):
        del image, confidence
        self.calls += 1
        return (
            [row[0] for row in self.rows],
            [np.asarray(row[1], dtype=np.float32) for row in self.rows],
            [np.zeros((1, 1, 3), dtype=np.uint8) for _ in self.rows],
            [row[2] for row in self.rows],
        )


class FieldBlindMultimethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "must-not-open-full-scene.jsonl"
        self.labels = self.root / "must-not-open-labels.jsonl"
        self.dataset_identity = self.root / "dataset-identity.json"
        _write_json(
            self.dataset_identity,
            {
                "schema_version": 1,
                "protocol": multi.DATASET_IDENTITY_PROTOCOL,
                "status": "owner_frozen",
                "definition_authority": "dataset_owner",
                "cohort_definition": {
                    "declared_images": 1201,
                    "deduplicated_before_freeze": True,
                    "declared_as_frozen_unseen_blind_test": True,
                    "source_unit": "original_full_scene_photograph",
                },
                "unlabeled_manifest": {
                    "path": str(self.manifest.resolve()),
                    "sha256": "a" * 64,
                    "rows": 1201,
                    "input_role": "original_full_scene",
                    "contains_labels": False,
                    "contains_manual_or_gt_range": False,
                    "contains_manual_or_gt_crop": False,
                },
                "labels": {
                    "path": str(self.labels.resolve()),
                    "sha256": "b" * 64,
                    "rows": 1201,
                },
                "authorization": {
                    "one_shot_image_inference": True,
                    "one_shot_scoring_after_prediction_seal": True,
                    "no_tuning_after_result": True,
                },
            },
        )
        self.paper = self.root / "paper-results.json"
        _write_json(
            self.paper,
            {
                "schema_version": 1,
                "protocol": multi.PAPER_RESULTS_PROTOCOL,
                "status": "complete",
                "audit": {
                    "all_prediction_and_score_seals_verified_before_public_truth_opened": True,
                    "bottom_up_metrics_match_all_sealed_summaries": True,
                    "manual_or_ground_truth_numeric_range_used_by_model": False,
                    "restricted_namespace_artifacts_opened": 0,
                    "field_samples_opened": 0,
                    "test_samples_opened": 0,
                    "sealed_samples_opened": 0,
                    "confirmatory_samples_opened": 0,
                    "inference_started": False,
                    "training_started": False,
                },
            },
        )
        self.paper_seal = self.root / "seal.json"
        _write_json(
            self.paper_seal,
            {
                "schema_version": 1,
                "protocol": multi.PAPER_RESULTS_PROTOCOL,
                "status": "sealed",
                "artifacts": {"summary": {"sha256": _sha(self.paper)}},
            },
        )
        self.paper_tables_root = self.root / "generated_tables"
        self.paper_tables_root.mkdir()
        self.paper_tables_manifest = self.paper_tables_root / "manifest.json"
        _write_json(
            self.paper_tables_manifest,
            {
                "schema_version": 1,
                "protocol": "paper_result_latex_tables_v1",
                "status": "complete",
                "source": {
                    "path": str(self.root.resolve()),
                    "summary_sha256": _sha(self.paper),
                    "seal_sha256": _sha(self.paper_seal),
                },
            },
        )
        _write_json(
            self.paper_tables_root / "seal.json",
            {
                "schema_version": 1,
                "protocol": "paper_result_latex_tables_v1",
                "status": "sealed",
                "artifacts": {
                    "manifest": {"sha256": _sha(self.paper_tables_manifest)}
                },
            },
        )
        self.detector_checkpoint = self.root / "meter-detector.pt"
        self.detector_checkpoint.write_bytes(b"frozen-meter-detector")
        detector_source = (
            Path(__file__).resolve().parents[1]
            / "utils/angleDetect/yoloDetection/yoloDectect.py"
        ).resolve()
        self.frontend = self.root / "frontend.json"
        self.frontend_value = {
            "schema_version": 1,
            "protocol": multi.FRONTEND_PROTOCOL,
            "status": "public_selected_frozen",
            "detector_checkpoint": {
                "path": str(self.detector_checkpoint.resolve()),
                "sha256": _sha(self.detector_checkpoint),
            },
            "detector_source": {
                "path": str(detector_source),
                "sha256": _sha(detector_source),
            },
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
        }
        _write_json(self.frontend, self.frontend_value)
        # These tests cover the generic one-shot protocol.  Authenticate the
        # synthetic plan at the seam while retaining all generic schema/manual
        # input checks in ``load_frontend_plan``.  Full SyncG lineage checking
        # is covered separately and remains mandatory in production.
        self.frontend_verifier = patch(
            "experiments.syncg_meter_detector_frontend.verify_frontend_plan",
            side_effect=lambda path: (
                Path(path).resolve(strict=True),
                json.loads(Path(path).read_text(encoding="utf-8")),
            ),
        )
        self.frontend_verifier.start()

        self.progress_artifact = self.root / "progress.pt"
        self.range_artifact = self.root / "garc-range.pt"
        self.v5_range_artifact = self.root / "v5-range.pt"
        self.progress_source = self.root / "progress.py"
        self.range_source = self.root / "range.py"
        self.adapter_source = (
            Path(__file__).resolve().parents[1]
            / "experiments/v5_unified_full_auto_adapter.py"
        ).resolve()
        for path, payload in (
            (self.progress_artifact, b"progress"),
            (self.range_artifact, b"garc-range"),
            (self.v5_range_artifact, b"v5-range"),
            (self.progress_source, b"def progress(): return 1\n"),
            (self.range_source, b"def numeric_range(): return 1\n"),
        ):
            path.write_bytes(payload)
        common_range = FrozenComponentBinding(
            name="automatic_numeric_range",
            provider_protocol="garc_range_v1",
            provider_identity={"protocol": "garc_range_v1"},
            artifact_sha256={"checkpoint": _sha(self.range_artifact)},
            source_sha256={"provider": _sha(self.range_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        v5_range = FrozenComponentBinding(
            name="automatic_numeric_range",
            provider_protocol="v5_range_v1",
            provider_identity={"protocol": "v5_range_v1"},
            artifact_sha256={"checkpoint": _sha(self.v5_range_artifact)},
            source_sha256={"provider": _sha(self.range_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        names = {
            "garc_final": "GARC-final",
            "v5_complete": "V5-complete",
            "pepd_shared_range": "PEPD-shared-range",
            "vdn_shared_range": "VDN-shared-range",
            "transformer_shared_range": "Original-Transformer-shared-range",
        }
        methods = {}
        for role in multi.METHOD_ROLES:
            factory = self.root / f"{role}-factory.py"
            factory.write_text(
                "def build_field_blind_full_auto_providers(_, __):\n"
                "    raise RuntimeError('metadata-only fixture')\n",
                encoding="utf-8",
                newline="\n",
            )
            progress = FrozenComponentBinding(
                name="progress",
                provider_protocol=f"{role}_progress_v1",
                provider_identity={"protocol": f"{role}_progress_v1"},
                artifact_sha256={"checkpoint": _sha(self.progress_artifact)},
                source_sha256={"provider": _sha(self.progress_source)},
                frozen=True,
                verified_complete=True,
                synthetic=False,
            )
            numeric_range = v5_range if role == "v5_complete" else common_range
            bundle = FrozenFullAutoBundle.create(
                method_name=names[role],
                progress_binding=progress,
                range_binding=numeric_range,
                factory_source_sha256=_sha(factory),
                reference_mode=REFERENCE_MODE_NATIVE,
                execution_mode=EXECUTION_FORMAL,
            )
            bundle_path = self.root / f"{role}-bundle.json"
            bundle.write(bundle_path)
            runtime_paths = [
                self.progress_artifact,
                self.progress_source,
                self.range_source,
                self.v5_range_artifact if role == "v5_complete" else self.range_artifact,
                factory,
                self.adapter_source,
            ]
            methods[role] = {
                "claim_tier": multi.CLAIM_TIERS[role],
                "bundle": {"path": str(bundle_path.resolve()), "sha256": _sha(bundle_path)},
                "factory": {"path": str(factory.resolve()), "sha256": _sha(factory)},
                "factory_function": "build_field_blind_full_auto_providers",
                "factory_config": {
                    "protocol": "metadata_only_fixture_v1",
                    "role": role,
                },
                "runtime_artifacts": [
                    {"path": str(path.resolve()), "sha256": _sha(path)}
                    for path in runtime_paths
                ],
                "output_mode": OUTPUT_MODE_FULL_READING,
                "evidence_role": EVIDENCE_ROLE_PRIMARY,
                "range_binding_sha256": bundle.range_binding_sha256,
            }
        self.roster = self.root / "roster.json"
        self.roster_value = {
            "schema_version": 1,
            "protocol": multi.ROSTER_PROTOCOL,
            "status": "all_full_reading_methods_frozen",
            "methods": methods,
            "comparison_contract": {
                "same_full_scene_frontend": True,
                "same_canonical_roi_per_sample": True,
                "all_predictions_sealed_together_before_labels": True,
                "progress_only_outputs_eligible": False,
                "garc_is_only_primary_final_model": True,
                "v5_is_internal_end_to_end_ablation": True,
                "pepd_vdn_share_garc_range_for_backbone_control": True,
                "transformer_is_sensitivity_only": True,
            },
            "audit": {
                "public_data_only": True,
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
            },
        }
        _write_json(self.roster, self.roster_value)
        self.protocol = self.root / "protocol.json"
        self.run_root = self.root / "formal-run"

    def tearDown(self) -> None:
        self.frontend_verifier.stop()
        self.temporary.cleanup()

    def _freeze(self):
        return multi.freeze_protocol(
            dataset_identity_path=self.dataset_identity,
            paper_results_path=self.paper,
            paper_tables_root=self.paper_tables_root,
            frontend_plan_path=self.frontend,
            method_roster_path=self.roster,
            output_path=self.protocol,
            run_root=self.run_root,
        )

    def test_freeze_and_preflight_open_no_field_files(self) -> None:
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.labels.exists())
        frozen = self._freeze()
        self.assertEqual(frozen["status"], "frozen_authorized_not_started")
        preflight = multi.static_preflight(self.protocol)
        self.assertEqual(
            preflight["status"],
            "validated_without_field_manifest_label_or_image_access",
        )
        self.assertEqual(preflight["method_roles"], list(multi.METHOD_ROLES))
        self.assertTrue(preflight["same_roi_for_all_methods"])
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.labels.exists())

    def test_frontend_rejects_manual_or_gt_crop_configuration(self) -> None:
        value = dict(self.frontend_value)
        value["manual_bbox"] = [1, 2, 3, 4]
        _write_json(self.frontend, value)
        with self.assertRaisesRegex(ValueError, "manual/GT crop"):
            multi.load_frontend_plan(self.frontend)

    def test_roster_is_exact_and_missing_method_fails_closed(self) -> None:
        value = json.loads(self.roster.read_text(encoding="utf-8"))
        del value["methods"]["vdn_shared_range"]
        _write_json(self.roster, value)
        with self.assertRaisesRegex(ValueError, "exactly five"):
            multi.load_method_roster(self.roster)

    def test_progress_controls_must_share_garc_automatic_range(self) -> None:
        value = json.loads(self.roster.read_text(encoding="utf-8"))
        value["methods"]["vdn_shared_range"]["range_binding_sha256"] = "f" * 64
        _write_json(self.roster, value)
        with self.assertRaisesRegex(ValueError, "range binding drift"):
            multi.load_method_roster(self.roster)

    def test_shared_roi_is_one_detection_with_deterministic_tie_break_and_padding(self) -> None:
        detector = _FakeDetector(
            [
                (0.9, [[30, 20], [70, 20], [70, 60], [30, 60]], 0),
                (0.9, [[10, 10], [50, 10], [50, 50], [10, 50]], 0),
            ]
        )
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        roi, record = multi.select_shared_roi(
            detector,
            image,
            confidence_threshold=0.25,
            padding_fraction=0.05,
            accepted_class_ids=[0],
        )
        self.assertEqual(detector.calls, 1)
        self.assertTrue(record["status"])
        self.assertEqual(record["detected_xyxy"], [10, 10, 50, 50])
        self.assertEqual(record["padded_xyxy"], [8, 8, 52, 52])
        self.assertEqual(roi.shape, (44, 44, 3))

    def test_detector_failure_has_no_full_frame_fallback(self) -> None:
        detector = _FakeDetector([])
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        roi, record = multi.select_shared_roi(
            detector,
            image,
            confidence_threshold=0.25,
            padding_fraction=0.05,
            accepted_class_ids=[0],
        )
        self.assertIsNone(roi)
        self.assertFalse(record["fallback_to_full_frame"])
        self.assertEqual(record["detector_invocations"], 1)

    def test_full_scene_manifest_rejects_any_crop_field_and_duplicate_bytes(self) -> None:
        base = {
            "sample_id": "a",
            "group_id": "meter-a",
            "image_path": str((self.root / "a.jpg").resolve()),
            "image_sha256": "c" * 64,
            "frame_sha256": "c" * 64,
        }
        with self.assertRaisesRegex(ValueError, "schema drift"):
            multi.validate_full_scene_manifest([{**base, "bbox": [0, 0, 10, 10]}])
        second = {
            **base,
            "sample_id": "b",
            "group_id": "meter-b",
            "image_path": str((self.root / "b.jpg").resolve()),
        }
        with self.assertRaisesRegex(ValueError, "duplicate full-scene image bytes"):
            multi.validate_full_scene_manifest([base, second])


if __name__ == "__main__":
    unittest.main()
