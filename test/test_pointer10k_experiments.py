"""Deterministic tests for the Pointer-10K auxiliary benchmark."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from experiments.datasets import (
    build_pointer10k_manifest,
    select_pointer10k_single_pointer_rows,
)
from experiments.evaluate_pointer10k_direction import (
    EVALUATION_PROTOCOL,
    summarize_direction_rows,
)
from experiments.extract_pointer10k_test import (
    POINTER10K_ARCHIVE_SHA256,
    _extract_member,
)
from experiments.summarize_pointer10k_direction import (
    _load_run,
    aggregate_runs,
    build_pairwise_comparisons,
    paired_bootstrap_comparison,
)


def _image(image_id: int) -> dict:
    return {
        "id": image_id,
        "file_name": f"{image_id:012d}.jpg",
        "width": 96,
        "height": 80,
    }


def _annotation(annotation_id: int, image_id: int, angle: str = "right") -> dict:
    if angle == "right":
        keypoints = [70, 40, 2, 60, 40, 0, 50, 40, 0]
    else:
        keypoints = [50, 20, 2, 50, 30, 0, 50, 40, 0]
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "area": 2400,
        "bbox": [20, 10, 60, 50],
        "num_keypoints": 3,
        "keypoints": keypoints,
    }


class Pointer10KExperimentTest(unittest.TestCase):
    def test_single_pointer_selection_uses_only_annotation_cardinality(self):
        coco = {
            "images": [_image(1), _image(2), _image(3)],
            "annotations": [
                _annotation(1, 1),
                _annotation(2, 2),
                _annotation(3, 2, "up"),
                _annotation(4, 3, "up"),
            ],
        }
        selected, audit = select_pointer10k_single_pointer_rows(coco)
        self.assertEqual([item[0]["id"] for item in selected], [1, 3])
        self.assertFalse(audit["selection_uses_predictions"])
        self.assertEqual(audit["pointer_count_distribution"], {"1": 2, "2": 1})
        self.assertEqual(audit["excluded_multi_pointer_images"], 1)

    def test_manifest_records_test_only_quality_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "test_pointer"
            annotation_dir = root / "annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            for image_id, intensity in ((1, 40), (2, 120), (3, 220)):
                image = np.full((80, 96, 3), intensity, dtype=np.uint8)
                image[:, ::4] = 255 - intensity
                self.assertTrue(
                    cv2.imwrite(str(image_dir / f"{image_id:012d}.jpg"), image)
                )
            coco = {
                "images": [_image(1), _image(2), _image(3)],
                "annotations": [
                    _annotation(1, 1),
                    _annotation(2, 2),
                    _annotation(3, 2, "up"),
                    _annotation(4, 3, "up"),
                ],
            }
            (annotation_dir / "ann_test_pointer.json").write_text(
                json.dumps(coco),
                encoding="utf-8",
            )
            output = root / "manifest.jsonl"
            count = build_pointer10k_manifest(
                root,
                output,
                strict_release=False,
            )
            self.assertEqual(count, 2)
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sample_id"] for row in rows], [
                "000000000001",
                "000000000003",
            ])
            for row in rows:
                self.assertEqual(row["dataset"], "Pointer-10K")
                self.assertEqual(row["metadata"]["pointer_count"], 1)
                self.assertIn("quality_groups", row["metadata"])
                self.assertEqual(len(row["metadata"]["dial_bbox"]), 4)
            protocol = json.loads(
                output.with_name(output.name + ".protocol.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(protocol["release_identity_verified"])
            self.assertFalse(protocol["training_or_fine_tuning_allowed"])
            self.assertEqual(protocol["pointer10k_training_images_used"], 0)
            self.assertFalse(protocol["selection_uses_predictions"])

    def test_direction_summary_penalizes_failures_and_keeps_denominator(self):
        rows = [
            {"status": True, "angle_error_degrees": 1.0},
            {"status": False, "angle_error_degrees": None},
        ]
        summary = summarize_direction_rows(
            rows,
            bootstrap_iterations=20,
            seed=5,
        )
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["mean_angle_error_degrees"], 90.5)
        self.assertAlmostEqual(summary["acc_2deg"], 0.5)
        self.assertIsNotNone(summary["mean_angle_error_bootstrap_95ci"])

    def test_aggregate_and_paired_bootstrap_use_matching_samples(self):
        baseline_rows = [
            {
                "sample_id": "a",
                "status": True,
                "angle_error_degrees": 20.0,
                "quality_groups": ["natural_low_quality_2of4"],
            },
            {
                "sample_id": "b",
                "status": True,
                "angle_error_degrees": 30.0,
                "quality_groups": [],
            },
        ]
        first = [
            {
                "sample_id": "a",
                "status": True,
                "angle_error_degrees": 8.0,
                "quality_groups": ["natural_low_quality_2of4"],
            },
            {
                "sample_id": "b",
                "status": True,
                "angle_error_degrees": 18.0,
                "quality_groups": [],
            },
        ]
        second = [
            {
                "sample_id": "a",
                "status": True,
                "angle_error_degrees": 12.0,
                "quality_groups": ["natural_low_quality_2of4"],
            },
            {
                "sample_id": "b",
                "status": True,
                "angle_error_degrees": 22.0,
                "quality_groups": [],
            },
        ]
        baseline = aggregate_runs("VDN", [baseline_rows])
        ours = aggregate_runs("Ours", [first, second])
        self.assertEqual(ours["runs"], 2)
        self.assertAlmostEqual(
            ours["metrics"]["mean_angle_error_degrees"]["mean"],
            15.0,
        )
        comparison = paired_bootstrap_comparison(
            ours,
            baseline,
            iterations=50,
            seed=7,
        )
        self.assertAlmostEqual(
            comparison["delta_mean_angle_degrees_method_minus_baseline"],
            -10.0,
        )
        self.assertEqual(comparison["method_win_rate"], 1.0)
        self.assertIsNotNone(comparison["paired_bootstrap_95ci"])
        pairwise = build_pairwise_comparisons(
            [baseline, ours],
            iterations=50,
            seed=7,
        )
        self.assertIn("Ours vs VDN", pairwise)
        self.assertEqual(pairwise["Ours vs VDN"]["method"], "Ours")
        self.assertEqual(pairwise["Ours vs VDN"]["baseline"], "VDN")
        self.assertAlmostEqual(
            pairwise["Ours vs VDN"][
                "delta_mean_angle_degrees_method_minus_baseline"
            ],
            -10.0,
        )

    def test_pinned_archive_hash_is_sha256(self):
        self.assertEqual(len(POINTER10K_ARCHIVE_SHA256), 64)
        int(POINTER10K_ARCHIVE_SHA256, 16)

    def test_extraction_reuses_only_crc_matching_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sample.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("image.jpg", b"official")
            destination = root / "output" / "image.jpg"
            destination.parent.mkdir()
            destination.write_bytes(b"corrupt!")
            with ZipFile(archive_path) as archive:
                info = archive.getinfo("image.jpg")
                self.assertEqual(
                    _extract_member(archive, info, destination),
                    "extracted",
                )
                self.assertEqual(destination.read_bytes(), b"official")
                self.assertEqual(
                    _extract_member(archive, info, destination),
                    "skipped",
                )

    def test_result_loader_rejects_mutated_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.jsonl"
            payload = '{"sample_id":"a"}\n'
            path.write_text(payload, encoding="utf-8")
            metadata = {
                "signature": {
                    "protocol": EVALUATION_PROTOCOL,
                    "diagnostic_limit": None,
                    "zero_shot_external_test": True,
                    "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            }
            path.with_name(path.name + ".meta.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            rows, _ = _load_run(path)
            self.assertEqual(rows[0]["sample_id"], "a")
            path.write_text('{"sample_id":"b"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "result SHA-256"):
                _load_run(path)


if __name__ == "__main__":
    unittest.main()
