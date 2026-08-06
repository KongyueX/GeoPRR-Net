import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import garc_field_blind_guard as guard
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    ROI_CONTRACT_SHA256,
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


class GARCFieldBlindGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "formal-run"
        self.guard_path = self.root / "guard.json"

        self.progress_artifact = self.root / "progress.pt"
        self.range_artifact = self.root / "range.pt"
        self.progress_source = self.root / "progress_source.py"
        self.range_source = self.root / "range_source.py"
        self.factory = self.root / "factory.py"
        self.config = self.root / "config.json"
        self.selection = self.root / "selection.json"
        self.paper_results = self.root / "paper-results.json"
        self.bundle_path = self.root / "garc_bundle.json"
        for path, payload in (
            (self.progress_artifact, b"progress-weights"),
            (self.range_artifact, b"range-weights"),
            (self.progress_source, b"def progress(): return 1\n"),
            (self.range_source, b"def numeric_range(): return 1\n"),
            (self.factory, b"def build_full_auto_providers(_): return {}\n"),
        ):
            path.write_bytes(payload)
        _write_json(
            self.config,
            {
                "method": "GARC-final",
                "input": "image_only",
                "range_mode": "automatic_numeric_ocr",
            },
        )

        progress = FrozenComponentBinding(
            name="progress",
            provider_protocol="garc_progress_final_v1",
            provider_identity={"protocol": "garc_progress_final_v1", "seed": 7},
            artifact_sha256={
                "checkpoint": _sha(self.progress_artifact),
                "effective_config": _sha(self.config),
            },
            source_sha256={"provider": _sha(self.progress_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        numeric_range = FrozenComponentBinding(
            name="automatic_numeric_range",
            provider_protocol="garc_range_final_v1",
            provider_identity={"protocol": "garc_range_final_v1", "seed": 11},
            artifact_sha256={"checkpoint": _sha(self.range_artifact)},
            source_sha256={"provider": _sha(self.range_source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        bundle = FrozenFullAutoBundle.create(
            method_name="GARC-final",
            progress_binding=progress,
            range_binding=numeric_range,
            factory_source_sha256=_sha(self.factory),
            reference_mode=REFERENCE_MODE_NATIVE,
            execution_mode=EXECUTION_FORMAL,
        )
        bundle.write(self.bundle_path)
        _write_json(
            self.selection,
            {
                "schema_version": 1,
                "protocol": guard.SELECTION_PROTOCOL,
                "status": "final_selected",
                "method_name": bundle.method_name,
                "selected_bundle_sha256": bundle.bundle_sha256,
                "selected_config_sha256": _sha(self.config),
                "selection_complete": True,
                "selection_gate_passed": True,
                "public_data_only": True,
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
                "no_post_selection_tuning": True,
            },
        )
        _write_json(
            self.paper_results,
            {
                "schema_version": 1,
                "protocol": guard.PAPER_RESULTS_PROTOCOL,
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
        files = [
            ("final_model_bundle", self.bundle_path),
            ("provider_factory_source", self.factory),
            ("final_inference_config", self.config),
            ("public_selection_summary", self.selection),
            ("paper_results_summary", self.paper_results),
            ("model_artifact", self.progress_artifact),
            ("model_artifact", self.range_artifact),
            ("component_source", self.progress_source),
            ("component_source", self.range_source),
        ]
        self.finalization = self.root / "finalization.json"
        _write_json(
            self.finalization,
            {
                "schema_version": 1,
                "protocol": guard.FINALIZATION_PROTOCOL,
                "status": "final_frozen",
                "method_name": bundle.method_name,
                "public_selection_summary": {
                    "path": str(self.selection.resolve()),
                    "sha256": _sha(self.selection),
                },
                "paper_results_summary": {
                    "path": str(self.paper_results.resolve()),
                    "sha256": _sha(self.paper_results),
                },
                "inference_contract": {
                    "input": "image_only",
                    "caller_range_allowed": False,
                    "caller_scalemark_allowed": False,
                    "caller_geometry_allowed": False,
                    "caller_reference_allowed": False,
                    "ground_truth_available": False,
                    "outputs": [
                        "prediction_progress",
                        "predicted_scale_start",
                        "predicted_scale_end",
                        "range_confidence",
                    ],
                },
                "frozen_files": [
                    {
                        "role": role,
                        "path": str(path.resolve()),
                        "sha256": _sha(path),
                    }
                    for role, path in files
                ],
            },
        )

        self.manifest = self.root / "must-not-open-manifest.jsonl"
        self.labels = self.root / "must-not-open-labels.jsonl"
        self.dataset_identity = self.root / "dataset-identity.json"
        self._write_dataset_identity(
            manifest_sha="a" * 64,
            labels_sha="b" * 64,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_dataset_identity(self, *, manifest_sha: str, labels_sha: str) -> None:
        _write_json(
            self.dataset_identity,
            {
                "schema_version": 1,
                "protocol": guard.DATASET_IDENTITY_PROTOCOL,
                "status": "owner_frozen",
                "definition_authority": "dataset_owner",
                "cohort_definition": {
                    "declared_images": 1201,
                    "deduplicated_before_freeze": True,
                    "declared_as_frozen_unseen_blind_test": True,
                    "paper_statement_authorized": True,
                },
                "unlabeled_manifest": {
                    "path": str(self.manifest.resolve()),
                    "sha256": manifest_sha,
                    "rows": 1201,
                    "unique_image_sha256": 1201,
                    "contains_labels": False,
                    "contains_manual_or_gt_range": False,
                },
                "labels": {
                    "path": str(self.labels.resolve()),
                    "sha256": labels_sha,
                    "rows": 1201,
                },
                "authorization": {
                    "one_shot_image_inference": True,
                    "one_shot_scoring_after_prediction_seal": True,
                    "no_tuning_after_result": True,
                },
            },
        )

    def _freeze(self) -> dict:
        return guard.freeze_guard(
            dataset_identity_path=self.dataset_identity,
            finalization_path=self.finalization,
            output_path=self.guard_path,
            run_root=self.run_root,
        )

    def test_freeze_and_preflight_do_not_require_field_files(self) -> None:
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.labels.exists())
        frozen = self._freeze()
        self.assertFalse(frozen["chronology"]["field_manifest_opened_during_freeze"])
        self.assertFalse(frozen["chronology"]["field_labels_opened_during_freeze"])
        result = guard.static_preflight(self.guard_path)
        self.assertEqual(result["status"], "validated_without_field_data_access")
        self.assertEqual(result["declared_images"], 1201)
        self.assertFalse(self.manifest.exists())
        self.assertFalse(self.labels.exists())

    def test_public_only_finalization_builder_is_self_verifying(self) -> None:
        output = self.root / "built-finalization.json"
        result = guard.freeze_finalization(
            bundle_path=self.bundle_path,
            factory_path=self.factory,
            config_path=self.config,
            public_selection_path=self.selection,
            paper_results_path=self.paper_results,
            model_artifacts=[self.progress_artifact, self.range_artifact],
            component_sources=[self.progress_source, self.range_source],
            output_path=output,
        )
        self.assertEqual(result["status"], "final_frozen")
        self.assertTrue(output.is_file())
        verified = guard._load_finalization(output)
        self.assertEqual(verified["method_name"], "GARC-final")

    def test_incomplete_paper_results_block_finalization_and_preflight(self) -> None:
        paper = json.loads(self.paper_results.read_text(encoding="utf-8"))
        paper["status"] = "not_ready"
        _write_json(self.paper_results, paper)
        finalization = json.loads(self.finalization.read_text(encoding="utf-8"))
        for row in finalization["frozen_files"]:
            if row["role"] == "paper_results_summary":
                row["sha256"] = _sha(self.paper_results)
        finalization["paper_results_summary"]["sha256"] = _sha(self.paper_results)
        _write_json(self.finalization, finalization)
        with self.assertRaisesRegex(ValueError, "paper results are incomplete"):
            self._freeze()

    def test_paper_results_that_used_manual_range_block_finalization(self) -> None:
        paper = json.loads(self.paper_results.read_text(encoding="utf-8"))
        paper["audit"]["manual_or_ground_truth_numeric_range_used_by_model"] = True
        _write_json(self.paper_results, paper)
        with self.assertRaisesRegex(ValueError, "manual/GT numeric range"):
            guard.freeze_finalization(
                bundle_path=self.bundle_path,
                factory_path=self.factory,
                config_path=self.config,
                public_selection_path=self.selection,
                paper_results_path=self.paper_results,
                model_artifacts=[self.progress_artifact, self.range_artifact],
                component_sources=[self.progress_source, self.range_source],
                output_path=self.root / "blocked-finalization.json",
            )

    def test_rejects_manual_scale_or_scalemark_in_final_config(self) -> None:
        _write_json(
            self.config,
            {"method": "GARC-final", "scaleStart": 0, "ScaleMark": [[1, 2]]},
        )
        finalization = json.loads(self.finalization.read_text(encoding="utf-8"))
        for row in finalization["frozen_files"]:
            if row["role"] == "final_inference_config":
                row["sha256"] = _sha(self.config)
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["selected_config_sha256"] = _sha(self.config)
        _write_json(self.selection, selection)
        for row in finalization["frozen_files"]:
            if row["role"] == "public_selection_summary":
                row["sha256"] = _sha(self.selection)
        finalization["public_selection_summary"]["sha256"] = _sha(self.selection)
        _write_json(self.finalization, finalization)
        with self.assertRaisesRegex(ValueError, "forbidden|label-derived"):
            self._freeze()

    def test_rejects_selection_that_opened_field_images(self) -> None:
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["field_images_opened"] = True
        _write_json(self.selection, selection)
        finalization = json.loads(self.finalization.read_text(encoding="utf-8"))
        for row in finalization["frozen_files"]:
            if row["role"] == "public_selection_summary":
                row["sha256"] = _sha(self.selection)
        finalization["public_selection_summary"]["sha256"] = _sha(self.selection)
        _write_json(self.finalization, finalization)
        with self.assertRaisesRegex(ValueError, "field images"):
            self._freeze()

    def test_rejects_config_that_is_not_bound_into_model_bundle(self) -> None:
        _write_json(
            self.config,
            {"method": "GARC-final", "input": "image_only", "threshold": 0.73},
        )
        selection = json.loads(self.selection.read_text(encoding="utf-8"))
        selection["selected_config_sha256"] = _sha(self.config)
        _write_json(self.selection, selection)
        finalization = json.loads(self.finalization.read_text(encoding="utf-8"))
        for row in finalization["frozen_files"]:
            if row["role"] == "final_inference_config":
                row["sha256"] = _sha(self.config)
            elif row["role"] == "public_selection_summary":
                row["sha256"] = _sha(self.selection)
        finalization["public_selection_summary"]["sha256"] = _sha(self.selection)
        _write_json(self.finalization, finalization)
        with self.assertRaisesRegex(ValueError, "not bound"):
            self._freeze()

    def test_public_artifact_drift_blocks_preflight(self) -> None:
        self._freeze()
        self.progress_artifact.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            guard.static_preflight(self.guard_path)

    def test_copying_guard_cannot_create_a_fresh_one_shot_ledger(self) -> None:
        self._freeze()
        copied = self.root / "copied-guard.json"
        copied.write_bytes(self.guard_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "copied or moved"):
            guard.static_preflight(copied)

    def test_claim_is_burned_before_invalid_manifest_is_opened(self) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "sample_id": "field-1",
                    "group_id": "meter-1",
                    "image_path": str((self.root / "image.jpg").resolve()),
                    "image_sha256": "c" * 64,
                    "canonical_roi_sha256": "c" * 64,
                    "frame_sha256": "d" * 64,
                    "roi_contract_sha256": "e" * 64,
                    "scaleStart": 0,
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self._write_dataset_identity(
            manifest_sha=_sha(self.manifest),
            labels_sha="b" * 64,
        )
        self._freeze()
        with self.assertRaisesRegex(ValueError, "label-derived"):
            guard._claim_and_validate_manifest(guard_path=self.guard_path)
        ledgers = guard._ledger_paths(self.guard_path)
        self.assertTrue(ledgers["inference_claim"].is_file())
        self.assertFalse(ledgers["inference_ready"].exists())
        with self.assertRaisesRegex(FileExistsError, "ledger"):
            guard._claim_and_validate_manifest(guard_path=self.guard_path)

    def test_runtime_manifest_rejects_duplicate_image_hashes(self) -> None:
        rows = []
        for index in range(1201):
            rows.append(
                {
                    "sample_id": f"field-{index:04d}",
                    "group_id": "meter-1",
                    "image_path": str((self.root / f"image-{index:04d}.jpg").resolve()),
                    "image_sha256": "c" * 64,
                    "canonical_roi_sha256": "c" * 64,
                    "frame_sha256": f"{index:064x}"[-64:],
                    "roi_contract_sha256": ROI_CONTRACT_SHA256,
                }
            )
        self.manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        self._write_dataset_identity(
            manifest_sha=_sha(self.manifest),
            labels_sha="b" * 64,
        )
        self._freeze()
        with self.assertRaisesRegex(ValueError, "duplicate image bytes"):
            guard._claim_and_validate_manifest(guard_path=self.guard_path)
        self.assertTrue(
            guard._ledger_paths(self.guard_path)["inference_claim"].is_file()
        )


if __name__ == "__main__":
    unittest.main()
