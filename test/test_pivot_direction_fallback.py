import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from experiments.pivot_direction_fallback import (
    SyncGPivotDirectionDataset,
    build_pivot_direction_model,
    decode_pivot_direction,
    pivot_direction_loss,
)
from experiments.summarize_dual_route import route_prediction


class PivotDirectionFallbackTest(unittest.TestCase):
    def test_dataset_generates_pivot_and_direction_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "dial.png"
            image = np.full((100, 100, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            sample = SimpleNamespace(
                sample_id="sample",
                image_path=str(image_path),
                dial_bbox=(10.0, 10.0, 90.0, 90.0),
                pointer_tail=(50.0, 50.0),
                pointer_tip=(70.0, 50.0),
            )
            dataset = SyncGPivotDirectionDataset(
                [sample],
                image_size=64,
                heatmap_size=16,
                training=False,
                expansion=1.0,
            )
            tensor, heatmap, direction, pivot, sample_id = dataset[0]
            self.assertEqual(tuple(tensor.shape), (3, 64, 64))
            self.assertEqual(tuple(heatmap.shape), (1, 16, 16))
            np.testing.assert_allclose(direction.numpy(), [1.0, 0.0], atol=1e-6)
            np.testing.assert_allclose(pivot.numpy(), [8.0, 8.0], atol=1e-6)
            self.assertEqual(sample_id, "sample")
            peak_y, peak_x = np.unravel_index(int(heatmap.argmax()), heatmap.shape[1:])
            self.assertEqual((peak_x, peak_y), (8, 8))

    def test_model_output_and_decoder_shapes(self):
        model = build_pivot_direction_model(imagenet_pretrained=False).eval()
        with torch.inference_mode():
            heatmap, direction = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(heatmap.shape), (2, 1, 16, 16))
        self.assertEqual(tuple(direction.shape), (2, 2))
        pivot, vector, confidence, valid = decode_pivot_direction(
            torch.zeros(2, 1, 16, 16),
            torch.tensor([[3.0, 4.0], [0.0, -2.0]]),
        )
        self.assertEqual(tuple(pivot.shape), (2, 2))
        np.testing.assert_allclose(vector[0].numpy(), [0.6, 0.8], atol=1e-6)
        np.testing.assert_allclose(vector[1].numpy(), [0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(confidence.numpy(), [0.5, 0.5], atol=1e-6)
        self.assertTrue(bool(valid.all()))

    def test_loss_is_near_zero_for_matching_targets(self):
        target_heatmap = torch.zeros(1, 1, 8, 8)
        target_heatmap[0, 0, 3, 4] = 1.0
        logits = torch.where(
            target_heatmap > 0.5,
            torch.full_like(target_heatmap, 12.0),
            torch.full_like(target_heatmap, -12.0),
        )
        target_direction = torch.tensor([[0.0, 1.0]])
        loss, components = pivot_direction_loss(
            logits,
            target_direction.clone(),
            target_heatmap,
            target_direction,
            pivot_weight=1.0,
        )
        self.assertLess(float(loss), 1e-8)
        self.assertLess(float(components["direction_loss"]), 1e-8)

    def test_route_uses_fallback_only_on_hard_failure(self):
        self.assertEqual(route_prediction(1.25, 9.0), (1.25, "base"))
        self.assertEqual(route_prediction(None, 2.5), (2.5, "fallback"))
        self.assertEqual(route_prediction(None, None), (None, "unresolved"))


if __name__ == "__main__":
    unittest.main()
