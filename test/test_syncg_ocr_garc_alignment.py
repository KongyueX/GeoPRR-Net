from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.audit_syncg_ocr_garc_overlap import build_overlap_matrix, overlap_record
from experiments.automatic_numeric_range_public_protocol import (
    canonical_bytes,
    canonical_sha256,
    sha256_file,
)
from experiments.build_garc_aligned_syncg_numeric_ocr_public import (
    ALIGNMENT_PROTOCOL,
    _partition_stats,
    assert_outer_exclusion,
    assign_inner_partitions,
    verify_local_artifacts,
)
from experiments.syncg_numeric_ocr import PROTOCOL as OCR_PROTOCOL
from experiments.syncg_numeric_ocr import CHECKPOINT_PROTOCOL
from experiments.syncg_strong_numeric_ocr import STRONG_CHECKPOINT_PROTOCOL
from experiments.verify_garc_ocr_training_evidence import verify_training_evidence


def row(sample: str, group: str, partition: str = "train") -> dict[str, str]:
    return {"sample_id": sample, "group_id": group, "partition": partition}


class OverlapAuditTests(unittest.TestCase):
    def test_overlap_record_detects_exact_sample_and_group_exposure(self) -> None:
        left = [row("a", "g1"), row("b", "g2"), row("c", "g3")]
        right = [row("b", "g2", "independent_validation"), row("z", "g3", "independent_validation")]
        result = overlap_record(left, right)
        self.assertEqual(result["sample_overlap"], 1)
        self.assertEqual(result["group_overlap"], 2)

    def test_matrix_keeps_inner_partitions_separate(self) -> None:
        ocr = [row("a", "g1", "train"), row("b", "g2", "validation")]
        garc = {"independent_validation": [row("a", "g1", "independent_validation")]}
        matrix = build_overlap_matrix(ocr, garc)
        self.assertEqual(matrix["train"]["independent_validation"]["group_overlap"], 1)
        self.assertEqual(matrix["validation"]["independent_validation"]["group_overlap"], 0)
        self.assertEqual(matrix["all"]["independent_validation"]["sample_overlap"], 1)


class AlignedBuilderTests(unittest.TestCase):
    def test_outer_group_overlap_is_rejected_even_without_sample_overlap(self) -> None:
        fit = [row("fit-a", "same-group", "algorithm_fit")]
        outer = {
            "calibration": [row("outer-a", "same-group", "calibration")],
            "development_excluded": [row("outer-b", "g2", "development_excluded")],
            "independent_validation": [row("outer-c", "g3", "independent_validation")],
        }
        with self.assertRaisesRegex(ValueError, "group overlap"):
            assert_outer_exclusion(fit, outer)

    def test_inner_group_split_is_order_independent_and_disjoint(self) -> None:
        rows = [row(f"s{i}", f"g{i}", "algorithm_fit") for i in range(30)]
        first = assign_inner_partitions(
            rows,
            seed=20260807,
            calibration_fraction=0.10,
            validation_fraction=0.10,
        )
        second = assign_inner_partitions(
            list(reversed(rows)),
            seed=20260807,
            calibration_fraction=0.10,
            validation_fraction=0.10,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first), {f"s{i}" for i in range(30)})
        self.assertEqual(set(first.values()), {"train", "calibration", "validation"})

    def test_local_verifier_rejects_tampered_sample_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [
                {
                    **row("s1", "g1", "train"),
                    "image_path": "datasets/SyncG/syncG/images/train/s1.jpg",
                    "annotation_path": "datasets/SyncG/syncG/annotations/train/s1.json",
                    "image_width": 10,
                    "image_height": 10,
                    "dial_bbox": [0.0, 0.0, 10.0, 10.0],
                    "tokens": [{"token_id": "s1:text:0", "text": "0", "bbox_roi_normalized": [0.1, 0.1, 0.2, 0.2]}],
                },
                {
                    **row("s2", "g2", "calibration"),
                    "image_path": "datasets/SyncG/syncG/images/train/s2.jpg",
                    "annotation_path": "datasets/SyncG/syncG/annotations/train/s2.json",
                    "image_width": 10,
                    "image_height": 10,
                    "dial_bbox": [0.0, 0.0, 10.0, 10.0],
                    "tokens": [{"token_id": "s2:text:0", "text": "1", "bbox_roi_normalized": [0.1, 0.1, 0.2, 0.2]}],
                },
                {
                    **row("s3", "g3", "validation"),
                    "image_path": "datasets/SyncG/syncG/images/train/s3.jpg",
                    "annotation_path": "datasets/SyncG/syncG/annotations/train/s3.json",
                    "image_width": 10,
                    "image_height": 10,
                    "dial_bbox": [0.0, 0.0, 10.0, 10.0],
                    "tokens": [{"token_id": "s3:text:0", "text": "2", "bbox_roi_normalized": [0.1, 0.1, 0.2, 0.2]}],
                },
            ]
            tokens = [
                {
                    "token_id": f"s{i}:text:0",
                    "sample_id": f"s{i}",
                    "group_id": f"g{i}",
                    "partition": partition,
                    "image_path": f"datasets/SyncG/syncG/images/train/s{i}.jpg",
                    "annotation_path": f"datasets/SyncG/syncG/annotations/train/s{i}.json",
                    "text": str(i - 1),
                    "numeric_value": float(i - 1),
                    "bbox_original": [1.0, 1.0, 2.0, 2.0],
                    "bbox_roi_normalized": [0.1, 0.1, 0.2, 0.2],
                }
                for i, partition in ((1, "train"), (2, "calibration"), (3, "validation"))
            ]
            samples_path = root / "samples.jsonl"
            tokens_path = root / "tokens.jsonl"
            samples_path.write_bytes(b"".join(canonical_bytes(value) for value in samples))
            tokens_path.write_bytes(b"".join(canonical_bytes(value) for value in tokens))
            outer = {
                name: {
                    "samples": 1,
                    "groups": 1,
                    "sample_overlap": 0,
                    "group_overlap": 0,
                    "sample_ids_sha256": canonical_sha256([f"outer-{name}"]),
                    "group_ids_sha256": canonical_sha256([f"outer-group-{name}"]),
                }
                for name in ("calibration", "development_excluded", "independent_validation")
            }
            partition_stats = _partition_stats(samples, tokens)
            summary = {
                "schema_version": 2,
                "protocol": OCR_PROTOCOL,
                "alignment_protocol": ALIGNMENT_PROTOCOL,
                "status": "complete",
                "parent_garc_protocol": {"path": "unused", "sha256": "a" * 64},
                "garc_partition_bindings": {
                    "algorithm_fit": {"sha256": "b" * 64, "group_ids_sha256": "c" * 64}
                },
                "split": {"partitions": partition_stats},
                "inventory": {"samples": 3, "groups": 3, "tokens": 3},
                "alignment_audit": {
                    "outer_exclusion": outer,
                    "all_outer_group_overlap_zero": True,
                    "all_outer_sample_overlap_zero": True,
                },
                "artifacts": {
                    "samples": "samples.jsonl",
                    "samples_sha256": sha256_file(samples_path),
                    "tokens": "tokens.jsonl",
                    "tokens_sha256": sha256_file(tokens_path),
                },
            }
            summary_path = root / "summary.json"
            summary_path.write_bytes(canonical_bytes(summary, pretty=True))
            seal = {
                "schema_version": 2,
                "protocol": OCR_PROTOCOL,
                "alignment_protocol": ALIGNMENT_PROTOCOL,
                "status": "sealed",
                "summary_sha256": sha256_file(summary_path),
                "samples_sha256": summary["artifacts"]["samples_sha256"],
                "tokens_sha256": summary["artifacts"]["tokens_sha256"],
                "split_sha256": canonical_sha256(summary["split"]),
                "parent_garc_protocol_sha256": "a" * 64,
                "algorithm_fit_manifest_sha256": "b" * 64,
                "algorithm_fit_group_ids_sha256": "c" * 64,
                "outer_exclusion_sha256": canonical_sha256(outer),
            }
            (root / "seal.json").write_bytes(canonical_bytes(seal, pretty=True))
            verify_local_artifacts(root)
            samples_path.write_bytes(samples_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "sample artifact hash drift"):
                verify_local_artifacts(root)


class TrainingEvidenceTests(unittest.TestCase):
    def _fixture(self, root: Path, *, strong_corpus_drift: bool = False) -> tuple[Path, Path]:
        identity = {
            "root": "C:\\pointer_read\\syncg_numeric_ocr_garc_aligned_v1",
            "summary_sha256": "1" * 64,
            "samples_sha256": "2" * 64,
            "tokens_sha256": "3" * 64,
        }
        tiny_artifacts: dict[str, str] = {}
        tiny_hashes: dict[str, str] = {}
        for component in ("detector", "recognizer"):
            path = root / f"tiny-{component}.pt"
            torch.save(
                {
                    "protocol": CHECKPOINT_PROTOCOL,
                    "status": "complete",
                    "mode": "formal",
                    "component": component,
                    "seed": 7,
                    "corpus": identity,
                    "state_dict": {},
                },
                path,
            )
            tiny_artifacts[component] = str(path)
            tiny_hashes[component] = sha256_file(path)
        tiny_summary = root / "tiny-summary.json"
        tiny_summary.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mode": "formal",
                    "component_selection": "both",
                    "seed": 7,
                    "corpus": identity,
                    "artifacts": tiny_artifacts,
                    "artifact_sha256": tiny_hashes,
                }
            ),
            encoding="utf-8",
        )
        strong_identity = dict(identity)
        if strong_corpus_drift:
            strong_identity["summary_sha256"] = "9" * 64
        strong_artifact = root / "strong.pt"
        torch.save(
            {
                "protocol": STRONG_CHECKPOINT_PROTOCOL,
                "status": "complete",
                "mode": "formal",
                "component": "recognizer",
                "seed": 8,
                "corpus": strong_identity,
                "state_dict": {},
            },
            strong_artifact,
        )
        strong_summary = root / "strong-summary.json"
        strong_summary.write_text(
            json.dumps(
                {
                    "protocol": STRONG_CHECKPOINT_PROTOCOL,
                    "status": "complete",
                    "mode": "formal",
                    "seed": 8,
                    "corpus": identity,
                    "artifact": str(strong_artifact),
                    "artifact_sha256": sha256_file(strong_artifact),
                }
            ),
            encoding="utf-8",
        )
        return tiny_summary, strong_summary

    def test_training_verifier_requires_one_shared_aligned_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny, strong = self._fixture(root)
            identity = {
                "root": "C:\\pointer_read\\syncg_numeric_ocr_garc_aligned_v1",
                "summary_sha256": "1" * 64,
                "samples_sha256": "2" * 64,
                "tokens_sha256": "3" * 64,
            }
            verification = {
                "protocol": ALIGNMENT_PROTOCOL,
                "samples": 12_176,
                "groups": 551,
                "algorithm_fit_exact_coverage": True,
                "outer_group_overlap_zero": True,
                "outer_sample_overlap_zero": True,
                "seal_sha256": "4" * 64,
            }
            with patch(
                "experiments.verify_garc_ocr_training_evidence.verify_corpus",
                return_value=verification,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.corpus_identity",
                return_value=identity,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.verify_frozen_inner_assignment",
                return_value={"verified": True},
            ):
                result = verify_training_evidence(
                    corpus_root=root,
                    tiny_summary_path=tiny,
                    strong_summary_path=strong,
                )
            self.assertTrue(result["audit"]["all_checkpoints_bind_same_corpus"])

    def test_training_verifier_rejects_strong_corpus_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny, strong = self._fixture(root, strong_corpus_drift=True)
            identity = {
                "root": "C:\\pointer_read\\syncg_numeric_ocr_garc_aligned_v1",
                "summary_sha256": "1" * 64,
                "samples_sha256": "2" * 64,
                "tokens_sha256": "3" * 64,
            }
            verification = {
                "protocol": ALIGNMENT_PROTOCOL,
                "samples": 12_176,
                "groups": 551,
                "algorithm_fit_exact_coverage": True,
                "outer_group_overlap_zero": True,
                "outer_sample_overlap_zero": True,
                "seal_sha256": "4" * 64,
            }
            with patch(
                "experiments.verify_garc_ocr_training_evidence.verify_corpus",
                return_value=verification,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.corpus_identity",
                return_value=identity,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.verify_frozen_inner_assignment",
                return_value={"verified": True},
            ):
                with self.assertRaisesRegex(ValueError, "strong_recognizer corpus binding drift"):
                    verify_training_evidence(
                        corpus_root=root,
                        tiny_summary_path=tiny,
                        strong_summary_path=strong,
                    )

    def test_training_verifier_accepts_authenticated_tiny_without_strong(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny, _ = self._fixture(root)
            identity = {
                "root": "C:\\pointer_read\\syncg_numeric_ocr_garc_aligned_v1",
                "summary_sha256": "1" * 64,
                "samples_sha256": "2" * 64,
                "tokens_sha256": "3" * 64,
            }
            verification = {
                "protocol": ALIGNMENT_PROTOCOL,
                "samples": 12_176,
                "groups": 551,
                "algorithm_fit_exact_coverage": True,
                "outer_group_overlap_zero": True,
                "outer_sample_overlap_zero": True,
                "seal_sha256": "4" * 64,
            }
            with patch(
                "experiments.verify_garc_ocr_training_evidence.verify_corpus",
                return_value=verification,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.corpus_identity",
                return_value=identity,
            ), patch(
                "experiments.verify_garc_ocr_training_evidence.verify_frozen_inner_assignment",
                return_value={"verified": True},
            ):
                result = verify_training_evidence(
                    corpus_root=root,
                    tiny_summary_path=tiny,
                )
            self.assertFalse(result["strong"]["available"])
            self.assertTrue(result["audit"]["all_checkpoints_bind_same_corpus"])


if __name__ == "__main__":
    unittest.main()
