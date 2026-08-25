from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.a15_2_mett import MomentExactPosteriorHead
from experiments.a15_fteb import frozen_twin_endpoint_forward
from experiments.evaluate_remst_resnet18_syncg import (
    validate_remst_resnet18_metadata,
)
from experiments.refresh_resnet18_endpoint import (
    DEFAULT_REFRESH_EPOCHS,
    PROTOCOL as ENDPOINT_REFRESH_PROTOCOL,
    load_refreshed_moment_exact_resnet18_anchor,
)
from experiments.remst_resnet18 import (
    DIRECT_RESNET18_ARCHITECTURE,
    RESNET18_STRIDE16_CHANNELS,
    RESNET18_STRIDE8_CHANNELS,
    ReMSTResNet18Correction,
    load_moment_exact_resnet18_anchor,
)
from experiments.resnet18_direct_progress import (
    DEFAULT_EPOCHS,
    IMAGENET_INITIALIZATION,
    PROTOCOL,
    SCENE_SPLIT_PROTOCOL,
    ResNet18DirectProgress,
)
from experiments.run_a15_fteb_inner_scene_probe import forward_a15_correction
from experiments.shrink_resnet18_endpoint import build_shrunk_endpoint


def _direct_checkpoint(path: Path) -> ResNet18DirectProgress:
    torch.manual_seed(17)
    model = ResNet18DirectProgress(imagenet_pretrained=False).eval()
    torch.save(
        {
            "protocol": PROTOCOL,
            "architecture": DIRECT_RESNET18_ARCHITECTURE,
            "pretrained_weights": IMAGENET_INITIALIZATION,
            "epochs": DEFAULT_EPOCHS,
            "checkpoint_selection": "terminal_fixed_epoch",
            "scene_disjoint": True,
            "split_protocol": SCENE_SPLIT_PROTOCOL,
            "seed": 17,
            "model_state": {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            },
        },
        path,
    )
    return model


class ReMSTResNet18Tests(unittest.TestCase):
    def test_midpoint_endpoint_is_one_exact_interpolated_linear_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct_path = root / "direct.pt"
            direct = _direct_checkpoint(direct_path)
            parent_state = {
                name: value.detach().clone()
                for name, value in direct.state_dict().items()
            }
            parent_state["backbone.fc.weight"] += 0.2
            parent_state["backbone.fc.bias"] -= 0.1
            parent_path = root / "endpoint_refresh.pt"
            torch.save(
                {
                    "protocol": ENDPOINT_REFRESH_PROTOCOL,
                    "epochs": DEFAULT_REFRESH_EPOCHS,
                    "encoder_state_unchanged": True,
                    "source_direct_checkpoint": str(direct_path),
                    "source": {"seed": 17},
                    "loss": "exact_l1",
                    "model_state": parent_state,
                    "calibration": {
                        "fit_population": "test",
                        "fit_rows": 3,
                        "conditions": ["clean"],
                        "knots_x": [0.0, 0.5, 1.0],
                        "knots_y": [0.0, 0.5, 1.0],
                    },
                },
                parent_path,
            )
            midpoint_path = root / "endpoint_midpoint.pt"
            result = build_shrunk_endpoint(
                parent_path, output_path=midpoint_path
            )
            anchor, metadata = load_refreshed_moment_exact_resnet18_anchor(
                midpoint_path
            )
            expected_weight = (
                direct.state_dict()["backbone.fc.weight"]
                + parent_state["backbone.fc.weight"]
            ) / 2.0
            expected_bias = (
                direct.state_dict()["backbone.fc.bias"]
                + parent_state["backbone.fc.bias"]
            ) / 2.0
            self.assertTrue(
                torch.equal(
                    anchor.raw_posterior_head.point_projection.weight,
                    expected_weight,
                )
            )
            self.assertTrue(
                torch.equal(
                    anchor.raw_posterior_head.point_projection.bias,
                    expected_bias,
                )
            )
            self.assertEqual(result["additional_parameters"], 0)
            self.assertEqual(
                metadata["endpoint_shrinkage"]["inference_linear_heads"], 1
            )

    def test_final_evaluator_requires_refreshed_terminal_single_backbone(self) -> None:
        evidence = {
            "image_encoder_modules": 1,
            "raw_and_sarn_share_parameter_objects": True,
            "anchor_frozen": True,
            "anchor_state_unchanged": True,
        }
        identity = validate_remst_resnet18_metadata(
            {
                "epochs": 5,
                "single_backbone_evidence": evidence,
                "source_endpoint_refresh_checkpoint": "endpoint_refresh.pt",
            },
            expected_variant="direct_scalar",
        )
        self.assertTrue(identity["endpoint_refreshed"])
        with self.assertRaisesRegex(ValueError, "lacks the endpoint refresh"):
            validate_remst_resnet18_metadata(
                {
                    "epochs": 5,
                    "single_backbone_evidence": evidence,
                    "source_endpoint_refresh_checkpoint": None,
                },
                expected_variant="direct_scalar",
            )

    def test_fixed_monotone_calibration_is_piecewise_linear_and_not_persistent(
        self,
    ) -> None:
        plain = MomentExactPosteriorHead(feature_dim=1)
        calibrated = MomentExactPosteriorHead(feature_dim=1)
        calibrated.set_monotone_calibration(
            (0.0, 0.5, 1.0), (0.0, 0.4, 1.0)
        )
        values = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0))
        expected = torch.tensor((0.0, 0.2, 0.4, 0.7, 1.0))
        self.assertTrue(
            torch.allclose(calibrated._calibrate(values), expected, atol=1.0e-7)
        )
        self.assertEqual(tuple(plain.state_dict()), tuple(calibrated.state_dict()))

    def test_imported_anchor_is_exact_direct_scalar_and_has_expected_scales(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "direct.pt"
            direct = _direct_checkpoint(checkpoint)
            anchor, metadata = load_moment_exact_resnet18_anchor(checkpoint)
            images = torch.randn(2, 3, 64, 64)
            with torch.inference_mode():
                expected = direct(images)
                observed = anchor(images)
            self.assertTrue(torch.equal(observed["point_progress"], expected))
            self.assertLess(
                float(observed["absolute_moment_error"].max()), 2.0e-6
            )
            features = observed["raw_encoder_features"]
            self.assertEqual(features["stride8"].shape, (2, 128, 8, 8))
            self.assertEqual(features["stride16"].shape, (2, 256, 4, 4))
            self.assertEqual(features["representation"].shape, (2, 512))
            self.assertTrue(metadata["single_backbone_parameter_set"])

    def test_raw_and_sarn_use_one_shared_encoder_parameter_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "direct.pt"
            _direct_checkpoint(checkpoint)
            anchor, _ = load_moment_exact_resnet18_anchor(checkpoint)
            calls = 0

            def count_call(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
                nonlocal calls
                calls += 1

            handle = anchor.raw_encoder.register_forward_hook(count_call)
            try:
                with torch.inference_mode():
                    endpoints = frozen_twin_endpoint_forward(
                        anchor,
                        torch.randn(2, 3, 64, 64),
                        torch.randn(2, 3, 64, 64),
                    )
            finally:
                handle.remove()
            self.assertEqual(calls, 2)
            self.assertEqual(
                endpoints["raw_features"]["stride8"].shape[1],
                RESNET18_STRIDE8_CHANNELS,
            )
            self.assertEqual(
                endpoints["sarn_features"]["stride16"].shape[1],
                RESNET18_STRIDE16_CHANNELS,
            )
            encoder_ids = {id(parameter) for parameter in anchor.raw_encoder.parameters()}
            self.assertEqual(
                len(encoder_ids),
                sum(1 for _ in anchor.raw_encoder.parameters()),
            )

    def test_correction_fallback_is_exact_raw_and_anchor_is_gradient_isolated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "direct.pt"
            _direct_checkpoint(checkpoint)
            anchor, _ = load_moment_exact_resnet18_anchor(checkpoint)
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            correction = ReMSTResNet18Correction()
            original = torch.randn(2, 3, 64, 64)
            sarn = torch.randn(2, 3, 64, 64)
            endpoints = frozen_twin_endpoint_forward(anchor, original, sarn)
            common = {
                "raw_posterior": endpoints["raw_posterior"],
                "sarn_posterior": endpoints["sarn_posterior"],
                "raw_mean": endpoints["raw_mean"],
                "sarn_mean": endpoints["sarn_mean"],
                "raw_features": endpoints["raw_features"],
                "sarn_features": endpoints["sarn_features"],
                "sarn_support_mask": torch.ones(2, 1, 64, 64),
                "raw_to_sarn_homography": torch.eye(3)[None].repeat(2, 1, 1),
            }
            inactive = dict(common)
            inactive["sarn_active"] = torch.zeros(2, dtype=torch.bool)
            output = forward_a15_correction(
                correction, inactive, endpoint_null=False
            )
            self.assertFalse(bool(output["relation_available"].any()))
            self.assertTrue(
                torch.equal(
                    output["progress_posterior"],
                    endpoints["raw_posterior"],
                )
            )
            self.assertTrue(torch.equal(output["mean"], endpoints["raw_mean"]))

            active = dict(common)
            active["sarn_active"] = torch.ones(2, dtype=torch.bool)
            trained = forward_a15_correction(
                correction, active, endpoint_null=False
            )
            self.assertTrue(bool(trained["relation_available"].all()))
            trained["mean"].sum().backward()
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and bool((parameter.grad != 0).any())
                    for parameter in correction.parameters()
                )
            )
            self.assertTrue(
                all(parameter.grad is None for parameter in anchor.parameters())
            )


if __name__ == "__main__":
    unittest.main()
