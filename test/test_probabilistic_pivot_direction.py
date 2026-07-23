import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from experiments.evaluate_probabilistic_pivot_direction import (
    _select_direction_decoder,
)

from experiments.probabilistic_pivot_direction import (
    SyncGProbabilisticDirectionDataset,
    build_probabilistic_pivot_direction_model,
    decode_probabilistic_pivot_direction,
    equivariance_loss,
    probabilistic_direction_loss,
    transform_pivot_direction,
)


class ProbabilisticPivotDirectionTest(unittest.TestCase):
    def test_decoder_ablation_selects_expected_direction(self):
        pivot = torch.zeros(1, 1, 8, 8)
        direct = torch.tensor([[0.0, 1.0]])
        circular = torch.full((1, 36), -10.0)
        circular[0, 0] = 10.0
        log_variance = torch.zeros(1, 1)
        outputs = (pivot, direct, circular, log_variance)
        prediction = decode_probabilistic_pivot_direction(*outputs)
        direct_value, direct_valid = _select_direction_decoder(
            outputs, prediction, "direct"
        )
        circular_value, circular_valid = _select_direction_decoder(
            outputs, prediction, "circular"
        )
        np.testing.assert_allclose(direct_value.numpy(), [[0.0, 1.0]], atol=1e-6)
        np.testing.assert_allclose(circular_value.numpy(), [[1.0, 0.0]], atol=1e-5)
        self.assertTrue(bool(direct_valid[0] and circular_valid[0]))

    def test_dataset_returns_exact_projective_pair(self):
        random.seed(7)
        np.random.seed(7)
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
            dataset = SyncGProbabilisticDirectionDataset(
                [sample],
                image_size=64,
                heatmap_size=16,
                training=True,
                expansion=1.0,
                scale_factor=0.0,
                rotation_factor=0.0,
                translation_factor=0.0,
                perspective_probability=1.0,
                max_perspective_degrees=30.0,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["image"].shape), (3, 64, 64))
            self.assertEqual(tuple(item["paired_image"].shape), (3, 64, 64))
            self.assertEqual(tuple(item["homography"].shape), (3, 3))
            base_pivot = item["pivot"][None] * 4.0
            expected_pivot, expected_direction, valid = transform_pivot_direction(
                base_pivot,
                item["direction"][None],
                item["homography"][None],
                ray_length=16.0,
            )
            self.assertTrue(bool(valid[0]))
            np.testing.assert_allclose(
                expected_pivot[0].numpy(),
                (item["paired_pivot"] * 4.0).numpy(),
                atol=1e-4,
            )
            np.testing.assert_allclose(
                expected_direction[0].numpy(),
                item["paired_direction"].numpy(),
                atol=1e-4,
            )

    def test_model_and_probabilistic_decoder_shapes(self):
        model = build_probabilistic_pivot_direction_model(
            angle_bins=36,
            imagenet_pretrained=False,
        ).eval()
        with torch.inference_mode():
            outputs = model(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(outputs[0].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(outputs[1].shape), (2, 2))
        self.assertEqual(tuple(outputs[2].shape), (2, 36))
        self.assertEqual(tuple(outputs[3].shape), (2, 1))

        logits = torch.full((2, 36), -8.0)
        logits[0, 0] = 8.0
        logits[1, 9] = 8.0
        prediction = decode_probabilistic_pivot_direction(
            torch.zeros(2, 1, 16, 16),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            logits,
            torch.tensor([[-4.0], [-2.0]]),
        )
        np.testing.assert_allclose(
            prediction.direction.numpy(),
            [[1.0, 0.0], [0.0, 1.0]],
            atol=1e-4,
        )
        self.assertTrue(bool(prediction.valid.all()))
        self.assertLess(
            float(prediction.angle_std_degrees[0]),
            float(prediction.angle_std_degrees[1]),
        )
        self.assertTrue(bool((prediction.angle_entropy < 0.01).all()))

    def test_probabilistic_loss_is_finite_and_backpropagates(self):
        pivot_logits = torch.randn(3, 1, 8, 8, requires_grad=True)
        direction_raw = torch.randn(3, 2, requires_grad=True)
        angle_logits = torch.randn(3, 36, requires_grad=True)
        log_variance = torch.full((3, 1), -3.0, requires_grad=True)
        target_heatmap = torch.zeros(3, 1, 8, 8)
        target_heatmap[:, :, 4, 4] = 1.0
        target_direction = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        loss, components = probabilistic_direction_loss(
            pivot_logits,
            direction_raw,
            angle_logits,
            log_variance,
            target_heatmap,
            target_direction,
            pivot_weight=1.0,
            bin_weight=0.2,
            vector_weight=0.5,
            soft_target_sigma_bins=1.25,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(components["angular_nll_loss"])))
        loss.backward()
        for tensor in (pivot_logits, direction_raw, angle_logits, log_variance):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_identity_equivariance_loss_is_zero(self):
        pivot_logits = torch.full((2, 1, 8, 8), -8.0)
        pivot_logits[:, :, 3, 4] = 8.0
        direction_raw = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        angle_logits = torch.zeros(2, 36)
        angle_logits[0, 0] = 8.0
        angle_logits[1, 27] = 8.0
        log_variance = torch.full((2, 1), -3.0)
        outputs = (pivot_logits, direction_raw, angle_logits, log_variance)
        loss, components = equivariance_loss(
            outputs,
            outputs,
            torch.eye(3)[None].repeat(2, 1, 1),
            image_size=32,
            heatmap_size=8,
            pivot_weight=1.0,
        )
        self.assertLess(float(loss), 1e-7)
        self.assertEqual(float(components["equivariance_valid_fraction"]), 1.0)


if __name__ == "__main__":
    unittest.main()
