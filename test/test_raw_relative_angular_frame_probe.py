from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.raw_context_endpoint_control_probe import (
    METHOD_CONTEXT_MLP,
    METHOD_LINEAR_REFRESH,
)
from experiments.raw_relative_angular_frame_probe import (
    METHOD_BASE,
    METHOD_PRIOR_ANGULAR,
    METHOD_RELATIVE_FRAME_AUX,
    METHOD_RELATIVE_FRAME_NO_AUX,
    RelativeAngularFrameDataset,
    RelativeFramePredictionRecord,
    angular_frame_targets,
    build_relative_frame_probe_from_foundation,
    relative_frame_parameter_counts,
    summarize_records,
)
from experiments.resnet18_direct_progress import DirectProgressDataset, DirectSample
from experiments.syncg_lightweight_regression_baselines import (
    LightweightProgressRegressor,
)


class RawRelativeAngularFrameProbeTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(41)
        foundation = LightweightProgressRegressor(
            "efficientnet_b0", imagenet_pretrained=False
        ).eval()
        return foundation, build_relative_frame_probe_from_foundation(foundation)

    def test_clockwise_frame_recovers_half_progress(self) -> None:
        pivot = np.asarray([0.5, 0.5], dtype=np.float32)

        def point(angle_degrees: float, radius: float = 0.4) -> np.ndarray:
            angle = np.deg2rad(angle_degrees)
            return pivot + radius * np.asarray(
                [np.sin(angle), -np.cos(angle)], dtype=np.float32
            )

        marks = [
            point(300.0),
            point(340.0),
            point(20.0),
            point(60.0),
            point(100.0),
            point(140.0),
            point(180.0),
        ]
        points = np.stack((*marks, pivot, point(60.0, radius=0.3)), axis=0)
        labels = angular_frame_targets(points)
        torch.testing.assert_close(
            labels["oracle_phase_progress"],
            torch.tensor(0.5),
            rtol=0.0,
            atol=1e-6,
        )
        expected = torch.tensor(
            [
                [np.sin(np.deg2rad(60.0)), np.cos(np.deg2rad(60.0))],
                [np.sin(np.deg2rad(300.0)), np.cos(np.deg2rad(300.0))],
                [np.sin(np.deg2rad(180.0)), np.cos(np.deg2rad(180.0))],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            labels["frame_sin_cos"], expected, rtol=0.0, atol=1e-6
        )

    def test_relative_arms_are_matched_and_start_at_base(self) -> None:
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
            METHOD_PRIOR_ANGULAR,
            METHOD_RELATIVE_FRAME_NO_AUX,
            METHOD_RELATIVE_FRAME_AUX,
        ):
            torch.testing.assert_close(
                outputs[method], outputs[METHOD_BASE], rtol=0.0, atol=0.0
            )
        left = model.relative_frame_no_aux.state_dict()
        right = model.relative_frame_aux.state_dict()
        self.assertEqual(set(left), set(right))
        self.assertTrue(all(torch.equal(left[name], right[name]) for name in left))
        counts = relative_frame_parameter_counts(model)
        self.assertEqual(counts[METHOD_RELATIVE_FRAME_NO_AUX], 107_238)
        self.assertEqual(
            counts[METHOD_RELATIVE_FRAME_NO_AUX],
            counts[METHOD_RELATIVE_FRAME_AUX],
        )
        self.assertLessEqual(
            abs(counts[METHOD_RELATIVE_FRAME_NO_AUX] - counts[METHOD_CONTEXT_MLP])
            / max(counts[METHOD_RELATIVE_FRAME_NO_AUX], counts[METHOD_CONTEXT_MLP]),
            0.01,
        )

    def test_only_relative_frame_arms_receive_gradients(self) -> None:
        _foundation, model = self._model()
        outputs = model(torch.rand(2, 3, 64, 64))
        targets = torch.tensor([0.25, 0.75])
        frame_targets = torch.tensor(
            [
                [[0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            ]
        )
        loss = (
            torch.nn.functional.l1_loss(
                outputs[METHOD_RELATIVE_FRAME_NO_AUX], targets
            )
            + torch.nn.functional.l1_loss(
                outputs[METHOD_RELATIVE_FRAME_AUX], targets
            )
            + 0.01
            * torch.nn.functional.smooth_l1_loss(
                outputs[f"{METHOD_RELATIVE_FRAME_AUX}_frame"],
                frame_targets,
                beta=0.1,
            )
        )
        loss.backward()
        for module in (model.relative_frame_no_aux, model.relative_frame_aux):
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
            model.prior_angular_no_aux,
        ):
            self.assertTrue(
                all(parameter.grad is None for parameter in module.parameters())
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP is unavailable")
    def test_uniform_posterior_phase_backward_is_finite_under_amp(self) -> None:
        from experiments.raw_relative_angular_frame_probe import (
            RelativeAngularFrameLogitRefiner,
        )

        head = RelativeAngularFrameLogitRefiner().to("cuda:0")
        torch.nn.init.zeros_(head.direction_projection.weight)
        torch.nn.init.zeros_(head.direction_projection.bias)
        angular = torch.zeros(
            2, 16, 36, device="cuda:0", dtype=torch.float32, requires_grad=True
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            statistics, _directions, _concentrations, phase = (
                head._frame_statistics(angular)
            )
            loss = statistics.sum() * 0.0 + phase.sum() * 0.0
        loss.backward()
        self.assertIsNotNone(angular.grad)
        self.assertTrue(bool(torch.isfinite(angular.grad).all()))
        self.assertTrue(
            all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in head.parameters()
            )
        )

    def test_dataset_remains_image_equivalent_to_direct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            image = np.zeros((96, 96, 3), dtype=np.uint8)
            image[:, :, 0] = np.arange(96, dtype=np.uint8)[None, :]
            image[:, :, 1] = np.arange(96, dtype=np.uint8)[:, None]
            image[:, :, 2] = 127
            self.assertTrue(cv2.imwrite(str(path), image))
            pivot = np.asarray([48.0, 48.0], dtype=np.float32)

            def point(angle: float, radius: float = 30.0) -> tuple[float, float]:
                radians = np.deg2rad(angle)
                value = pivot + radius * np.asarray(
                    [np.sin(radians), -np.cos(radians)], dtype=np.float32
                )
                return float(value[0]), float(value[1])

            marks = tuple(point(angle) for angle in (300, 340, 20, 60, 100, 140, 180))
            sample = DirectSample(
                sample_id="synthetic",
                group_id="group",
                scene_stem="scene",
                image_path=path,
                dial_bbox=(8.0, 8.0, 88.0, 88.0),
                normalized_target=0.5,
                protected_points_xy=(*marks, (48.0, 48.0), point(60, 24.0)),
            )
            direct = DirectProgressDataset((sample,), training=True, seed=83)
            relative = RelativeAngularFrameDataset(
                (sample,), training=True, seed=83
            )
            direct.set_epoch(30)
            relative.set_epoch(30)
            direct_image, direct_progress = direct[0]
            relative_row = relative[0]
            torch.testing.assert_close(
                relative_row["image"], direct_image, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                relative_row["progress"], direct_progress, rtol=0.0, atol=0.0
            )
            self.assertEqual(tuple(relative_row["frame_sin_cos"].shape), (3, 2))

    def test_bootstrap_is_reproducible_for_progress_and_geometry(self) -> None:
        methods = (
            METHOD_BASE,
            METHOD_LINEAR_REFRESH,
            METHOD_CONTEXT_MLP,
            METHOD_PRIOR_ANGULAR,
            METHOD_RELATIVE_FRAME_NO_AUX,
            METHOD_RELATIVE_FRAME_AUX,
        )
        records = tuple(
            RelativeFramePredictionRecord(
                sample_id=f"sample_{index}",
                scene_stem=f"scene_{index // 2}",
                condition="clean",
                target=0.5,
                predictions=dict(
                    zip(methods, (0.60, 0.59, 0.58, 0.57, 0.56, 0.55), strict=True)
                ),
                frame_available=True,
                target_frame=((0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)),
                oracle_phase_progress=0.5,
                frames={
                    METHOD_RELATIVE_FRAME_NO_AUX: (
                        (0.5, 0.5),
                        (-0.5, 0.5),
                        (0.5, -0.5),
                    ),
                    METHOD_RELATIVE_FRAME_AUX: (
                        (0.1, 0.9),
                        (-0.9, 0.1),
                        (0.9, -0.1),
                    ),
                },
                concentrations={
                    METHOD_RELATIVE_FRAME_NO_AUX: (0.7, 0.7, 0.7),
                    METHOD_RELATIVE_FRAME_AUX: (0.9, 0.9, 0.9),
                },
                phase_predictions={
                    METHOD_RELATIVE_FRAME_NO_AUX: 0.56,
                    METHOD_RELATIVE_FRAME_AUX: 0.52,
                },
            )
            for index in range(8)
        )
        first = summarize_records(records, bootstrap_replicates=25, bootstrap_seed=13)
        second = summarize_records(records, bootstrap_replicates=25, bootstrap_seed=13)
        self.assertEqual(first, second)
        comparison = first["comparisons"][
            "relative_frame_aux_minus_linear_refresh"
        ]
        self.assertAlmostEqual(comparison["mean_nmae_delta"], -0.04)
        self.assertEqual(
            comparison["scene_bootstrap_probability_delta_below_zero"], 1.0
        )
        self.assertAlmostEqual(
            first["geometry"]["phase_progress_nmae"]["oracle_geometry"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
