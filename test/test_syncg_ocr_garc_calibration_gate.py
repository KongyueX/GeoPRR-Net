from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.automatic_numeric_range_public_protocol import sha256_file
from experiments.decide_syncg_strong_numeric_ocr_upgrade import (
    DECISION_PROTOCOL,
    _load_report,
    decide,
)
from experiments.evaluate_syncg_ocr_garc_calibration import (
    EVALUATION_PROTOCOL,
    EXPECTED_SEED,
    _checkpoint_inputs,
    _strong_checkpoint_inputs,
    authenticate_metadata,
)
from experiments.syncg_numeric_ocr import CHECKPOINT_PROTOCOL, PROTOCOL, VOCABULARY
from experiments.syncg_strong_numeric_ocr import STRONG_CHECKPOINT_PROTOCOL


SAFE_ROOT = Path(r"C:\pointer_read")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT_ROOT / "experiments/evaluate_syncg_ocr_garc_calibration.py"
PROTOCOL_PATH = PROJECT_ROOT / "experiments/syncg_strong_numeric_ocr_upgrade_protocol.json"


def corpus_identity() -> dict[str, object]:
    return {
        "root": r"C:\pointer_read\aligned-fixture",
        "summary_sha256": "1" * 64,
        "seal_sha256": "2" * 64,
        "samples_sha256": "3" * 64,
        "tokens_sha256": "4" * 64,
        "samples": 12176,
        "groups": 551,
        "alignment_protocol": "syncg_public_numeric_ocr_garc_aligned_v1",
    }


def metadata(checkpoint: Path, summary: Path) -> dict[str, object]:
    return {
        "training_corpus": corpus_identity(),
        "tiny_summary": {"path": str(summary), "sha256": sha256_file(summary), "seed": EXPECTED_SEED},
        "tiny_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "parent_garc_protocol": {"path": "public-protocol", "sha256": "5" * 64, "identity": "garc"},
        "calibration_roster": {
            "path": "calibration.label_free.jsonl",
            "sha256": "6" * 64,
            "samples": 2224,
            "groups": 100,
            "sample_ids_sha256": "7" * 64,
            "group_ids_sha256": "8" * 64,
        },
        "source_manifest": {"path": "syncg_train.jsonl", "sha256": "9" * 64},
        "disjointness": {
            "training_sample_ids_sha256": "a" * 64,
            "training_group_ids_sha256": "b" * 64,
            "calibration_sample_ids_sha256": "7" * 64,
            "calibration_group_ids_sha256": "8" * 64,
            "sample_overlap": 0,
            "group_overlap": 0,
        },
    }


def audit() -> dict[str, int]:
    return {
        "calibration_images_opened": 2224,
        "calibration_annotations_opened": 2224,
        "algorithm_fit_images_opened": 0,
        "inner_validation_images_opened": 0,
        "development_excluded_images_opened": 0,
        "development_excluded_annotations_opened": 0,
        "independent_validation_images_opened": 0,
        "independent_validation_annotations_opened": 0,
        "joint_oof_412_19_samples_opened": 0,
        "field_samples_opened": 0,
        "public_test_samples_opened": 0,
        "sealed_samples_opened": 0,
        "confirmatory_samples_opened": 0,
    }


class TinyCheckpointCompatibilityTests(unittest.TestCase):
    def test_new_seed_summary_and_checkpoint_schema_is_accepted(self) -> None:
        SAFE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SAFE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / "recognizer.pt"
            expected = corpus_identity()
            checkpoint_corpus = {key: expected[key] for key in (
                "root", "summary_sha256", "samples_sha256", "tokens_sha256"
            )}
            torch.save({
                "protocol": CHECKPOINT_PROTOCOL,
                "status": "complete",
                "mode": "formal",
                "component": "recognizer",
                "seed": EXPECTED_SEED,
                "vocabulary": list(VOCABULARY),
                "corpus": checkpoint_corpus,
                "state_dict": {},
                "metrics": {"validation": {
                    "exact_accuracy": 0.9,
                    "character_accuracy": 0.95,
                    "parseable_fraction": 1.0,
                    "tokens": 10,
                    "source_images": 2,
                }},
            }, checkpoint)
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "protocol": PROTOCOL,
                "status": "complete",
                "mode": "formal",
                "seed": EXPECTED_SEED,
                "component_selection": "both",
                "corpus": checkpoint_corpus,
                "artifacts": {"recognizer": str(checkpoint), "detector": str(root / "detector.pt")},
                "artifact_sha256": {"recognizer": sha256_file(checkpoint)},
            }), encoding="utf-8")
            _, value, resolved, loaded = _checkpoint_inputs(summary, corpus_identity=expected)
            self.assertEqual(value["seed"], EXPECTED_SEED)
            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(loaded["metrics"]["validation"]["tokens"], 10)

    def test_strong_seed_summary_and_checkpoint_schema_is_accepted(self) -> None:
        SAFE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SAFE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / "strong.pt"
            expected = corpus_identity()
            checkpoint_corpus = {key: expected[key] for key in (
                "root", "summary_sha256", "samples_sha256", "tokens_sha256"
            )}
            torch.save({
                "protocol": STRONG_CHECKPOINT_PROTOCOL,
                "status": "complete",
                "mode": "formal",
                "component": "recognizer",
                "seed": 20260818,
                "vocabulary": list(VOCABULARY),
                "corpus": checkpoint_corpus,
                "state_dict": {},
                "metrics": {"validation": {"tokens": 10}},
            }, checkpoint)
            summary = root / "summary.json"
            summary.write_text(json.dumps({
                "protocol": STRONG_CHECKPOINT_PROTOCOL,
                "status": "complete",
                "mode": "formal",
                "seed": 20260818,
                "corpus": checkpoint_corpus,
                "artifact": str(checkpoint),
                "artifact_sha256": sha256_file(checkpoint),
            }), encoding="utf-8")
            _, value, resolved, loaded = _strong_checkpoint_inputs(summary, corpus_identity=expected)
            self.assertEqual(value["seed"], 20260818)
            self.assertEqual(resolved, checkpoint.resolve())
            self.assertEqual(loaded["component"], "recognizer")


class CalibrationMetadataBoundaryTests(unittest.TestCase):
    def test_metadata_authentication_loads_only_outer_calibration_roster(self) -> None:
        expected_corpus = corpus_identity()
        fake_summary = {
            "parent_garc_protocol": {"path": "parent.json", "sha256": "5" * 64},
            "garc_partition_bindings": {"calibration": {
                "path": "calibration.jsonl", "sha256": "6" * 64,
                "samples": 2224, "groups": 100,
                "sample_ids_sha256": "7" * 64, "group_ids_sha256": "8" * 64,
            }},
            "alignment_audit": {
                "algorithm_fit_exact_coverage": {
                    "sample_ids_sha256": "a" * 64, "group_ids_sha256": "b" * 64,
                },
                "outer_exclusion": {"calibration": {"sample_overlap": 0, "group_overlap": 0}},
            },
        }
        loaded_partitions: list[str] = []

        def fake_roster(_path: Path, partition: str):
            loaded_partitions.append(partition)
            return {}, Path("calibration.jsonl"), [], {
                "samples": 2224, "groups": 100,
                "sample_ids_sha256": "7" * 64, "group_ids_sha256": "8" * 64,
            }

        fake_tiny = Path("tiny.json")
        fake_checkpoint = Path("tiny.pt")
        with patch("experiments.evaluate_syncg_ocr_garc_calibration._corpus_identity", return_value=(Path("corpus"), fake_summary, expected_corpus)), \
             patch("experiments.evaluate_syncg_ocr_garc_calibration._checkpoint_inputs", return_value=(fake_tiny, {"seed": EXPECTED_SEED}, fake_checkpoint, {})), \
             patch("experiments.evaluate_syncg_ocr_garc_calibration.load_frozen_protocol", return_value=(Path("parent.json"), {"protocol": "garc"})), \
             patch("experiments.evaluate_syncg_ocr_garc_calibration.sha256_file", side_effect=lambda path: {
                 "parent.json": "5" * 64, "calibration.jsonl": "6" * 64,
                 "tiny.json": "c" * 64, "tiny.pt": "d" * 64,
                 "manifest.jsonl": "e" * 64,
             }[str(path)]), \
             patch("experiments.evaluate_syncg_ocr_garc_calibration.load_partition_roster", side_effect=fake_roster), \
             patch("experiments.evaluate_syncg_ocr_garc_calibration.verify_bound_file", return_value=Path("manifest.jsonl")):
            result = authenticate_metadata(corpus_root=Path("corpus"), tiny_summary_path=fake_tiny)

        self.assertEqual(loaded_partitions, ["calibration"])
        self.assertEqual(result["calibration_roster"]["samples"], 2224)
        self.assertEqual(result["calibration_roster"]["groups"], 100)
        self.assertTrue(all(value == 0 for value in result["data_access_audit"].values()))


class GateReportBindingTests(unittest.TestCase):
    def _write_report(self, root: Path, md: dict[str, object], **metric_overrides: float) -> Path:
        metrics = {
            "exact_accuracy": 0.91,
            "character_accuracy": 0.99,
            "parseable_fraction": 0.999,
            "tokens": 30000,
            "source_images": 2224,
            **metric_overrides,
        }
        report = {
            "protocol": EVALUATION_PROTOCOL,
            "status": "formal_calibration_component_evaluation_complete",
            "mode": "formal",
            "partition": "calibration",
            "recognizer_kind": "tiny",
            "evaluation_kind": "recognizer_oracle_text_boxes_component_only",
            **md,
            "component_corpus": {
                "samples": 2224, "groups": 100, "tokens": 30000,
                "sample_ids_sha256": md["calibration_roster"]["sample_ids_sha256"],
                "group_ids_sha256": md["calibration_roster"]["group_ids_sha256"],
                "token_ids_sha256": "e" * 64,
                "annotation_content_inventory_sha256": "f" * 64,
            },
            "metrics": metrics,
            "data_access_audit": audit(),
            "code": {"path": str(EVALUATOR.resolve()), "sha256": sha256_file(EVALUATOR)},
        }
        path = root / "calibration-report.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_report_binds_exact_outer_calibration_and_zero_restricted_access(self) -> None:
        SAFE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SAFE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / "tiny.pt"; checkpoint.write_bytes(b"checkpoint")
            summary = root / "summary.json"; summary.write_text("{}", encoding="utf-8")
            md = metadata(checkpoint, summary)
            report = self._write_report(root, md)
            _, _, observed = _load_report(report, metadata=md)
            self.assertEqual(observed["source_images"], 2224)
            self.assertEqual(observed["tokens"], 30000)

    def test_report_fails_closed_on_independent_validation_access(self) -> None:
        SAFE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SAFE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / "tiny.pt"; checkpoint.write_bytes(b"checkpoint")
            summary = root / "summary.json"; summary.write_text("{}", encoding="utf-8")
            md = metadata(checkpoint, summary)
            report = self._write_report(root, md)
            value = json.loads(report.read_text(encoding="utf-8"))
            value["data_access_audit"]["independent_validation_images_opened"] = 1
            report.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "independent_validation_images_opened"):
                _load_report(report, metadata=md)

    def test_decision_uses_outer_calibration_not_inner_diagnostic(self) -> None:
        fake_metadata = {
            "training_corpus": corpus_identity(),
            "tiny_summary": {"path": "summary", "sha256": "1" * 64, "seed": EXPECTED_SEED},
            "tiny_checkpoint": {"path": "checkpoint", "sha256": "2" * 64},
            "calibration_roster": {"path": "roster", "sha256": "3" * 64},
        }
        protocol = {
            "frozen_activation_gate": {
                "activate_strong_recognizer_if_any": {
                    "garc_calibration_exact_accuracy_below": 0.92,
                    "garc_calibration_character_accuracy_below": 0.98,
                    "garc_calibration_parseable_fraction_below": 0.995,
                }
            }
        }
        outer = {
            "exact_accuracy": 0.91, "character_accuracy": 0.99,
            "parseable_fraction": 0.999, "tokens": 10, "source_images": 2224,
        }
        inner = {
            "role": "diagnostic_only_not_used_by_activation_checks",
            "exact_accuracy": 1.0, "character_accuracy": 1.0,
            "parseable_fraction": 1.0, "tokens": 10, "source_images": 1,
        }
        with patch("experiments.decide_syncg_strong_numeric_ocr_upgrade._load_protocol", return_value=(PROTOCOL_PATH, protocol)), \
             patch("experiments.decide_syncg_strong_numeric_ocr_upgrade.authenticate_metadata", return_value=fake_metadata), \
             patch("experiments.decide_syncg_strong_numeric_ocr_upgrade._expected_inputs"), \
             patch("experiments.decide_syncg_strong_numeric_ocr_upgrade._load_report", return_value=(Path("report"), {}, outer)), \
             patch("experiments.decide_syncg_strong_numeric_ocr_upgrade._inner_diagnostic", return_value=inner), \
             patch("experiments.decide_syncg_strong_numeric_ocr_upgrade.sha256_file", return_value="a" * 64):
            result = decide(
                corpus_root=Path("corpus"), tiny_summary_path=Path("summary"),
                calibration_report_path=Path("report"), protocol_path=PROTOCOL_PATH,
            )
        self.assertEqual(result["protocol"], DECISION_PROTOCOL)
        self.assertTrue(result["train_strong_recognizer"])
        self.assertEqual(result["triggered_component_checks"], ["garc_calibration_exact_accuracy_below"])
        self.assertEqual(result["algorithm_fit_inner_validation"]["role"], "diagnostic_only_not_used_by_activation_checks")


class FrozenProtocolTests(unittest.TestCase):
    def test_protocol_freezes_exact_calibration_and_reserves_final_evidence(self) -> None:
        value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertEqual(value["protocol"], "syncg_strong_numeric_ocr_upgrade_v2")
        self.assertEqual(value["status"], "frozen_before_outer_calibration_metrics")
        calibration = value["frozen_inputs"]["garc_outer_calibration"]
        self.assertEqual((calibration["samples"], calibration["groups"]), (2224, 100))
        self.assertEqual(calibration["sample_ids_sha256"], "e610210b7381592a347b39a1d2875f9fec91ff49f741c031ddc8465ecfc40302")
        self.assertEqual(calibration["group_ids_sha256"], "d6e1ae0648590d1b5e61db492196aabf0f32d296062363fb01f348314ee9082a")
        gate = value["frozen_activation_gate"]
        self.assertEqual(gate["selection_partition"], "garc_outer_calibration")
        self.assertEqual(gate["algorithm_fit_inner_validation_role"], "diagnostic_only_not_a_gate_input")
        self.assertEqual(gate["independent_validation_role"], "unopened_until_final_model_freeze")
        serialized = json.dumps(value).lower()
        self.assertNotIn("seed_20260806", serialized)
        self.assertIsNone(value["observed_results"])
        for name, relative in {
            "calibration_evaluator": "experiments/evaluate_syncg_ocr_garc_calibration.py",
            "activation_decider": "experiments/decide_syncg_strong_numeric_ocr_upgrade.py",
            "strong_candidate_qualifier": "experiments/qualify_syncg_strong_numeric_ocr_candidate.py",
            "final_garc_selector": "experiments/evaluate_garc_full_auto_public.py",
        }.items():
            binding = value["implementation_bindings"][name]
            self.assertEqual(binding["path"], relative)
            self.assertEqual(binding["sha256"], sha256_file(PROJECT_ROOT / relative))


if __name__ == "__main__":
    unittest.main()
