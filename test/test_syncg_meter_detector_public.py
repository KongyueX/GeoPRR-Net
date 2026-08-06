from __future__ import annotations

import argparse
import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

from experiments.build_syncg_meter_detector_public import (
    DEFAULT_OUTPUT,
    _normalized_box,
    build_corpus,
    verify_corpus,
)
from experiments.syncg_meter_detector_frontend import (
    EXPECTED_DETECTOR_SOURCE,
    _validate_production_source,
    verify_public_frontend_lineage,
)
from experiments.field_blind_multimethod import select_shared_roi
from experiments.train_syncg_meter_detector import choose_confidence_threshold
from experiments.train_syncg_meter_detector import (
    _load_experiment_protocol,
    _run_intent,
    _runtime_identity,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SyncGPublicMeterDetectorTests(unittest.TestCase):
    def test_public_bbox_clips_to_visible_frame(self) -> None:
        normalized, line, clipped = _normalized_box(
            [-2, 20, 102, 80], width=100, height=100, sample_id="synthetic"
        )
        self.assertEqual(normalized, [0.5, 0.5, 1.0, 0.6])
        self.assertEqual(clipped, [0.0, 20.0, 100.0, 80.0])
        self.assertEqual(line, "0 0.5000000000 0.5000000000 1.0000000000 0.6000000000")

    def test_threshold_selection_is_deterministic_and_recall_constrained(self) -> None:
        samples = [
            {"sample_id": "a", "dial_bbox_xyxy": [10, 10, 90, 90]},
            {"sample_id": "b", "dial_bbox_xyxy": [0, 0, 50, 50]},
        ]
        predictions = [
            {"sample_id": "a", "candidates": [{"confidence": 0.9, "xyxy": [10, 10, 90, 90]}]},
            {"sample_id": "b", "candidates": [{"confidence": 0.8, "xyxy": [0, 0, 50, 50]}]},
        ]
        choice = choose_confidence_threshold(samples, predictions)
        self.assertEqual(choice["branch"], "recall_constraint_feasible")
        self.assertEqual(choice["selected"]["confidence_threshold"], 0.8)
        self.assertEqual(choice["selected"]["selected_box_iou50_recall"], 1.0)

    def test_corpus_builder_rejects_nonfrozen_split_before_materialization(self) -> None:
        output = Path(r"C:\pointer_read") / f"unit_bad_detector_split_{uuid.uuid4().hex}"
        with self.assertRaisesRegex(ValueError, "split seed differs"):
            build_corpus(
                output_dir=output,
                split_seed=20260820,
                calibration_fraction=0.10,
                validation_fraction=0.10,
            )
        self.assertFalse(output.exists())

    def test_runtime_sources_and_checkpoint_rule_are_frozen(self) -> None:
        protocol = _load_experiment_protocol()
        identity = _runtime_identity(protocol)
        self.assertEqual(identity["ultralytics_version"], "8.4.102")
        self.assertEqual(
            protocol["selection"]["checkpoint_rule"],
            "maximum calibration mAP50-95; Ultralytics latest epoch attaining the maximum is retained on exact ties",
        )
        self.assertEqual(
            identity["requirements_lock"]["sha256"],
            protocol["model"]["runtime"]["requirements_lock_sha256"],
        )

    def test_run_intent_rejects_cross_corpus_resume(self) -> None:
        protocol = _load_experiment_protocol()
        with tempfile.TemporaryDirectory(dir=r"C:\pointer_read") as temporary:
            root = Path(temporary)
            corpora = []
            for name in ("corpus_a", "corpus_b"):
                corpus = root / name
                corpus.mkdir()
                (corpus / "summary.json").write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
                (corpus / "seal.json").write_text(f'{{"name":"{name}"}}\n', encoding="utf-8")
                (corpus / "dataset_train_cal.yaml").write_text("names: {0: meter}\n", encoding="utf-8")
                corpora.append(corpus)
            pretrained = root / "pretrained.pt"
            pretrained.write_bytes(b"public")
            output = root / "runs"
            output.mkdir()
            intent_path = output / "seed_20260819.run_intent.json"
            args = argparse.Namespace(seed=20260819, device="0", workers=4, resume=False)
            _run_intent(
                path=intent_path,
                protocol=protocol,
                corpus_root=corpora[0],
                output_root=output,
                pretrained=pretrained,
                runtime_identity={"synthetic": True},
                args=args,
            )
            args.resume = True
            with self.assertRaisesRegex(ValueError, "run intent differs"):
                _run_intent(
                    path=intent_path,
                    protocol=protocol,
                    corpus_root=corpora[1],
                    output_root=output,
                    pretrained=pretrained,
                    runtime_identity={"synthetic": True},
                    args=args,
                )

    def test_shared_roi_rejects_misaligned_detector_outputs(self) -> None:
        class BadDetector:
            def target_detection(self, image, confidence):
                del image, confidence
                return [0.9], [], [], [0]

        import numpy as np

        with self.assertRaisesRegex(ValueError, "misaligned candidate arrays"):
            select_shared_roi(
                BadDetector(),
                np.zeros((64, 64, 3), dtype=np.uint8),
                confidence_threshold=0.5,
                padding_fraction=0.05,
                accepted_class_ids=[0],
            )

    def test_event_wrappers_bind_atomic_publish_and_child_identity(self) -> None:
        event_source = (
            Path(__file__).resolve().parents[1]
            / "experiments/run_syncg_meter_detector_after_paper_event_driven.ps1"
        ).read_text(encoding="utf-8")
        training_source = (
            Path(__file__).resolve().parents[1]
            / "experiments/run_syncg_meter_detector_training.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("NotifyFilters]::DirectoryName", event_source)
        self.assertIn("syncg_public_meter_detector_event_chain_lock_v1", event_source)
        self.assertIn("Send-ProgressBestEffort", event_source)
        self.assertIn("child_pending", training_source)
        self.assertIn("child_command_sha256", training_source)

    def test_legacy_checkpoint_name_is_rejected_before_lineage_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "yolo_findMeter.pt"
            checkpoint.write_bytes(b"legacy")
            plan_path = root / "frontend_plan.json"
            plan_path.write_text("{}\n", encoding="utf-8")
            plan = {
                "protocol": "field_blind_shared_meter_frontend_v1",
                "status": "public_selected_frozen",
                "detector_class": "targetDetectModel",
                "detector_checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": _sha(checkpoint),
                },
            }
            with self.assertRaisesRegex(ValueError, "legacy yolo_findMeter"):
                verify_public_frontend_lineage(plan, plan_path=plan_path)

    def test_production_runtime_source_exposes_expected_class(self) -> None:
        _validate_production_source(EXPECTED_DETECTOR_SOURCE)

    @unittest.skipUnless(DEFAULT_OUTPUT.is_dir(), "sealed public detector corpus is absent")
    def test_materialized_public_corpus_verifies(self) -> None:
        summary = verify_corpus(DEFAULT_OUTPUT)
        self.assertEqual(summary["inventory"]["images"], 16_000)
        self.assertEqual(summary["inventory"]["physical_groups"], 725)
        self.assertTrue(summary["split"]["group_disjoint"])


if __name__ == "__main__":
    unittest.main()
