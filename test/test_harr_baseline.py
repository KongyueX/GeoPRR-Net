"""Deterministic tests for the released HARR pointer-branch adapter."""
from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.harr_baseline import (
    HARR_PINNED_COMMIT,
    HARR_RELEASED_CHECKPOINT_SHA256,
    build_harr_pointer_model,
    decode_harr_pointer_direction,
    harr_tensor_from_bbox,
    verify_harr_source,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
HARR_SOURCE = PROJECT_DIR / "artifacts" / "vendor" / "Detect-and-read-meters"
HARR_CHECKPOINT = (
    HARR_SOURCE / "model" / "meter_data" / "textgraph_vgg_100.pth"
)


class HARRBaselineTest(unittest.TestCase):
    def test_decoder_orients_pointer_from_centre_towards_tip(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.line(mask, (31, 32), (56, 32), 1, thickness=3)
        logits = torch.full((1, 1, 64, 64), -10.0)
        logits[0, 0][torch.from_numpy(mask.astype(bool))] = 10.0
        prediction = decode_harr_pointer_direction(logits)
        self.assertEqual(prediction.valid.tolist(), [True])
        self.assertEqual(prediction.failure_reason, (None,))
        self.assertGreater(float(prediction.direction[0, 0]), 0.98)
        self.assertLess(abs(float(prediction.direction[0, 1])), 0.05)

    def test_decoder_reports_empty_mask_as_failure(self):
        prediction = decode_harr_pointer_direction(
            torch.full((2, 1, 32, 32), -20.0)
        )
        self.assertEqual(prediction.valid.tolist(), [False, False])
        self.assertEqual(
            prediction.failure_reason,
            ("empty_pointer_mask", "empty_pointer_mask"),
        )

    def test_released_checkpoint_identity_is_sha256(self):
        self.assertEqual(len(HARR_RELEASED_CHECKPOINT_SHA256), 64)
        int(HARR_RELEASED_CHECKPOINT_SHA256, 16)

    def test_pinned_source_and_released_checkpoint_load_when_available(self):
        if not HARR_SOURCE.is_dir() or not HARR_CHECKPOINT.is_file():
            self.skipTest("ignored HARR source/checkpoint is unavailable")
        self.assertEqual(verify_harr_source(HARR_SOURCE), HARR_PINNED_COMMIT)
        model, metadata = build_harr_pointer_model(
            HARR_SOURCE,
            HARR_CHECKPOINT,
        )
        self.assertEqual(metadata["source_commit"], HARR_PINNED_COMMIT)
        self.assertEqual(metadata["released_checkpoint_epoch"], 100)
        self.assertGreater(metadata["active_pointer_branch_parameters"], 1_000_000)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        self.assertTrue(bool(torch.isfinite(output).all()))

        demo = cv2.imread(str(HARR_SOURCE / "demo" / "777.jpg"))
        if demo is not None:
            demo_tensor = harr_tensor_from_bbox(
                demo,
                (260.0, 70.0, 990.0, 800.0),
                expansion=1.0,
            ).unsqueeze(0)
            with torch.inference_mode():
                demo_probability = torch.sigmoid(model(demo_tensor))
            self.assertGreater(float(demo_probability.max()), 0.99)
            self.assertGreater(int((demo_probability > 0.5).sum()), 100)


if __name__ == "__main__":
    unittest.main()
