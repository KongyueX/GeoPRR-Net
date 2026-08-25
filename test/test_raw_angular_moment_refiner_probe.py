from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.raw_angular_moment_refiner_probe import (
    METHOD_ANGULAR_DIRECTION_AUX,
    METHOD_ANGULAR_NO_AUX,
    METHOD_BASE,
    AngularPredictionRecord,
    AngularProgressDataset,
    _polar_sampling_grid,
    angular_parameter_counts,
    build_angular_probe_from_foundation,
    summarize_records,
)
from experiments.raw_context_endpoint_control_probe import (
    METHOD_CONTEXT_MLP,
    METHOD_LINEAR_REFRESH,
)
from experiments.resnet18_direct_progress import DirectProgressDataset, DirectSample
from experiments.syncg_lightweight_regression_baselines import (
    LightweightProgressRegressor,
)


class RawAngularMomentRefinerProbeTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(31)
        foundation = LightweightProgressRegressor(
            "efficientnet_b0", imagenet_pretrained=False
        ).eval()
        return foundation, build_angular_probe_from_foundation(foundation)

    def test_polar_grid_uses_clockwise_zero_up_convention(self) -> None:
        grid, radii, directions = _polar_sampling_grid(
            radial_bins=3, angular_bins=4
        )
        self.assertEqual(tuple(grid.shape), (1, 3, 4, 2))
        torch.testing.assert_close(
            grid[0, :, 0, 0], torch.zeros(3), rtol=0.0, atol=1e-7
        )
        torch.testing.assert_close(
            grid[0, :, 0, 1], -radii, rtol=0.0, atol=1e-7
        )
        torch.testing.assert_close(
            grid[0, :, 1, 0], radii, rtol=0.0, atol=1e-6
        )
        torch.testing.assert_close(
            directions,
            torch.tensor(
                [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
            ),
            rtol=0.0,
            atol=1e-6,
        )

    def test_angular_arms_are_matched_and_start_at_base(self) -> None:
        foundation, model = self._model()
        model.eval()
        images = torch.rand(2, 3, 64, 64)
        with torch.inference_mode():
            expected = foundation(images)
            outputs = model(images)
        torch.testing.assert_close(outputs[METHOD_BASE], expected, rtol=1e-6, atol=1e-7)
        for method in (
            METHOD_LINEAR_REFRESH,
            METHOD_CONTEXT_MLP,
            METHOD_ANGULAR_NO_AUX,
            METHOD_ANGULAR_DIRECTION_AUX,
        ):
            torch.testing.assert_close(
                outputs[method], outputs[METHOD_BASE], rtol=0.0, atol=0.0
            )
        no_aux_state = model.angular_no_aux.state_dict()
        aux_state = model.angular_direction_aux.state_dict()
        self.assertEqual(set(no_aux_state), set(aux_state))
        self.assertTrue(
            all(torch.equal(no_aux_state[name], aux_state[name]) for name in no_aux_state)
        )
        counts = angular_parameter_counts(model)
        self.assertEqual(counts[METHOD_ANGULAR_NO_AUX], 105_944)
        self.assertEqual(
            counts[METHOD_ANGULAR_NO_AUX],
            counts[METHOD_ANGULAR_DIRECTION_AUX],
        )
        self.assertLessEqual(
            abs(counts[METHOD_ANGULAR_NO_AUX] - counts[METHOD_CONTEXT_MLP])
            / max(counts[METHOD_ANGULAR_NO_AUX], counts[METHOD_CONTEXT_MLP]),
            0.01,
        )
        self.assertEqual(
            counts["total_trainable"],
            counts[METHOD_ANGULAR_NO_AUX]
            + counts[METHOD_ANGULAR_DIRECTION_AUX],
        )

    def test_only_angular_arms_receive_gradients(self) -> None:
        _foundation, model = self._model()
        outputs = model(torch.rand(2, 3, 64, 64))
        targets = torch.tensor([0.25, 0.75])
        direction_targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        loss = (
            torch.nn.functional.l1_loss(outputs[METHOD_ANGULAR_NO_AUX], targets)
            + torch.nn.functional.l1_loss(
                outputs[METHOD_ANGULAR_DIRECTION_AUX], targets
            )
            + 0.01
            * torch.nn.functional.smooth_l1_loss(
                outputs[f"{METHOD_ANGULAR_DIRECTION_AUX}_direction"],
                direction_targets,
                beta=0.1,
            )
        )
        loss.backward()
        for module in (model.angular_no_aux, model.angular_direction_aux):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    for parameter in module.parameters()
                )
            )
        for module in (
            model.encoder,
            model.base_projection,
            model.linear_refresh,
            model.context_mlp,
        ):
            self.assertTrue(
                all(parameter.grad is None for parameter in module.parameters())
            )

    def test_training_dataset_is_image_equivalent_to_direct_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            image = np.zeros((96, 96, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(96, dtype=np.uint8)[None, :]
            image[:, :, 1] = np.arange(96, dtype=np.uint8)[:, None]
            image[:, :, 2] = 127
            self.assertTrue(cv2.imwrite(str(path), image))
            marks = tuple(
                (48.0 + 30.0 * math_sin, 48.0 - 30.0 * math_cos)
                for math_sin, math_cos in (
                    (0.0, 1.0),
                    (0.7, 0.7),
                    (1.0, 0.0),
                    (0.7, -0.7),
                    (0.0, -1.0),
                    (-0.7, -0.7),
                    (-1.0, 0.0),
                )
            )
            sample = DirectSample(
                sample_id="synthetic",
                group_id="group",
                scene_stem="scene",
                image_path=path,
                dial_bbox=(8.0, 8.0, 88.0, 88.0),
                normalized_target=0.375,
                protected_points_xy=(*marks, (48.0, 48.0), (72.0, 48.0)),
            )
            direct = DirectProgressDataset((sample,), training=True, seed=73)
            angular = AngularProgressDataset((sample,), training=True, seed=73)
            direct.set_epoch(30)
            angular.set_epoch(30)
            direct_image, direct_progress = direct[0]
            angular_row = angular[0]
            torch.testing.assert_close(
                angular_row["image"], direct_image, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                angular_row["progress"], direct_progress, rtol=0.0, atol=0.0
            )
            self.assertEqual(tuple(angular_row["direction_sin_cos"].shape), (2,))

    def test_progress_and_direction_bootstraps_are_reproducible(self) -> None:
        records = tuple(
            AngularPredictionRecord(
                sample_id=f"sample_{index}",
                scene_stem=f"scene_{index // 2}",
                condition="clean",
                target=0.5,
                predictions={
                    METHOD_BASE: 0.60,
                    METHOD_LINEAR_REFRESH: 0.59,
                    METHOD_CONTEXT_MLP: 0.58,
                    METHOD_ANGULAR_NO_AUX: 0.57,
                    METHOD_ANGULAR_DIRECTION_AUX: 0.55,
                },
                direction_available=True,
                target_direction=(0.0, 1.0),
                directions={
                    METHOD_ANGULAR_NO_AUX: (0.5, 0.5),
                    METHOD_ANGULAR_DIRECTION_AUX: (0.1, 0.9),
                },
                concentrations={
                    METHOD_ANGULAR_NO_AUX: 0.7,
                    METHOD_ANGULAR_DIRECTION_AUX: 0.9,
                },
            )
            for index in range(8)
        )
        first = summarize_records(records, bootstrap_replicates=25, bootstrap_seed=11)
        second = summarize_records(records, bootstrap_replicates=25, bootstrap_seed=11)
        self.assertEqual(first, second)
        comparison = first["comparisons"][
            "angular_direction_aux_minus_linear_refresh"
        ]
        self.assertAlmostEqual(comparison["mean_nmae_delta"], -0.04)
        self.assertEqual(
            comparison["scene_bootstrap_probability_delta_below_zero"], 1.0
        )
        self.assertLess(
            first["direction"][
                "direction_aux_minus_no_aux_mean_angular_error_degrees"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
