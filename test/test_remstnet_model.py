from __future__ import annotations

import unittest

import torch

from remstnet.model import (
    ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
    ADAPTIVE_REMST_NET_ARCHITECTURE,
    COORDINATED_REMST_NET_ARCHITECTURE,
    PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
    REMST_BLOCK_NET_ARCHITECTURE,
    CoordinatedReMSTNet,
    ReMSTBlockNet,
    remstnet_parameter_counts,
)


class ReMSTBlockNetTests(unittest.TestCase):
    @staticmethod
    def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
        return {
            "raw": torch.randn(batch, 3, 64, 64),
            "sarn": torch.randn(batch, 3, 64, 64),
            "support": torch.ones(batch, 1, 64, 64),
            "active": torch.tensor([True, False][:batch]),
            "homography": torch.eye(3).repeat(batch, 1, 1),
        }

    @staticmethod
    def _forward(
        model: ReMSTBlockNet, inputs: dict[str, torch.Tensor]
    ) -> dict[str, object]:
        return model(
            inputs["raw"],
            inputs["sarn"],
            inputs["support"],
            sarn_active=inputs["active"],
            raw_to_sarn_homography=inputs["homography"],
        )

    def test_architecture_is_native_and_parameter_decomposition_is_exact(self) -> None:
        model = ReMSTBlockNet()
        counts = remstnet_parameter_counts(model)

        self.assertNotIn("EfficientNet", REMST_BLOCK_NET_ARCHITECTURE)
        self.assertEqual(counts["shared_foundation"], 4_010_110)
        self.assertEqual(counts["scale8_remst_block"], 25_337)
        self.assertEqual(counts["scale16_remst_block"], 33_473)
        self.assertEqual(counts["trainable"], 58_810)
        self.assertEqual(counts["total_unique"], 4_068_920)
        self.assertEqual(counts["component_sum"], counts["total_unique"])
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in model._foundation_modules()
                for parameter in module.parameters()
            )
        )

    def test_zero_initialization_reaches_sarn_and_has_exact_raw_fallback(self) -> None:
        torch.manual_seed(20260818)
        model = ReMSTBlockNet().eval()
        output = self._forward(model, self._inputs())

        self.assertEqual(output["architecture"], REMST_BLOCK_NET_ARCHITECTURE)
        self.assertEqual(output["relation_available"].tolist(), [True, False])
        self.assertTrue(
            torch.equal(
                output["progress_posterior"][0],
                output["sarn_endpoint_posterior"][0],
            )
        )
        self.assertTrue(
            torch.equal(
                output["progress_posterior"][1],
                output["raw_anchor_posterior"][1],
            )
        )
        self.assertLessEqual(
            float(output["moment_absolute_error"].detach().max()), 2.0e-6
        )

    def test_both_feature_rewrite_and_moment_heads_receive_gradients(self) -> None:
        torch.manual_seed(20260819)
        model = ReMSTBlockNet().train()
        with torch.no_grad():
            model.scale8_block.feature_projection.weight.fill_(1.0e-4)
            model.scale16_block.feature_projection.weight.fill_(1.0e-4)
            model.scale8_block.moment_projection.bias.fill_(0.1)
            model.scale16_block.moment_projection.bias.fill_(0.1)
        inputs = self._inputs()
        inputs["active"] = torch.ones(2, dtype=torch.bool)
        output = self._forward(model, inputs)
        output["mean"].sum().backward()

        for block in (model.scale8_block, model.scale16_block):
            self.assertIsNotNone(block.feature_projection.weight.grad)
            self.assertIsNotNone(block.moment_projection.weight.grad)
            self.assertGreater(
                float(block.feature_projection.weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(block.moment_projection.weight.grad.abs().sum()), 0.0
            )
        self.assertTrue(
            all(
                parameter.grad is None
                for module in model._foundation_modules()
                for parameter in module.parameters()
            )
        )

    def test_state_dict_round_trips_strictly(self) -> None:
        source = ReMSTBlockNet()
        restored = ReMSTBlockNet()
        incompatibility = restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatibility.missing_keys, [])
        self.assertEqual(incompatibility.unexpected_keys, [])

    def test_coordinated_architecture_has_exact_parameter_decomposition(self) -> None:
        model = CoordinatedReMSTNet()
        counts = remstnet_parameter_counts(model)

        self.assertNotIn("EfficientNet", COORDINATED_REMST_NET_ARCHITECTURE)
        self.assertEqual(counts["shared_foundation"], 4_010_110)
        self.assertEqual(counts["scale8_remst_block"], 25_337)
        self.assertEqual(counts["scale16_remst_block"], 33_473)
        self.assertEqual(counts["cross_scale_moment_coordinator"], 193_373)
        self.assertEqual(counts["trainable"], 252_183)
        self.assertEqual(counts["total_unique"], 4_262_293)
        self.assertEqual(counts["component_sum"], counts["total_unique"])

    def test_coordinated_zero_init_and_non_cancelling_stage_shifts(self) -> None:
        torch.manual_seed(20260820)
        model = CoordinatedReMSTNet().eval()
        inputs = self._inputs()
        output = self._forward(model, inputs)

        self.assertEqual(
            output["architecture"], COORDINATED_REMST_NET_ARCHITECTURE
        )
        self.assertTrue(
            torch.equal(
                output["progress_posterior"][0],
                output["sarn_endpoint_posterior"][0],
            )
        )
        self.assertTrue(
            torch.equal(
                output["progress_posterior"][1],
                output["raw_anchor_posterior"][1],
            )
        )
        self.assertTrue(
            torch.allclose(
                output["stage_moment_allocation"].sum(dim=1),
                torch.ones(2),
                atol=1.0e-7,
                rtol=0.0,
            )
        )

        with torch.no_grad():
            model.moment_coordinator.total_output_a.bias.fill_(0.2)
            model.moment_coordinator.total_output_b.bias.fill_(0.2)
        shifted = self._forward(model, inputs)
        active_shifts = torch.stack(
            (
                shifted["scale8_moment_shift"],
                shifted["scale16_moment_shift"],
                shifted["context_moment_shift"],
            ),
            dim=1,
        )[0]
        self.assertTrue(bool((active_shifts > 0.0).all()))
        self.assertTrue(
            torch.allclose(
                active_shifts.sum(),
                shifted["total_moment_shift"][0],
                atol=1.0e-7,
                rtol=0.0,
            )
        )

    def test_coordinator_and_both_feature_rewrites_receive_gradients(self) -> None:
        torch.manual_seed(20260821)
        model = CoordinatedReMSTNet().train()
        with torch.no_grad():
            for block in (model.scale8_block, model.scale16_block):
                block.feature_projection.weight.fill_(1.0e-4)
                block.moment_projection.weight.fill_(1.0e-3)
            model.moment_coordinator.total_output_a.weight.fill_(1.0e-3)
            model.moment_coordinator.total_output_b.weight.fill_(1.0e-3)
            model.moment_coordinator.context_responsibility[-1].weight.fill_(
                1.0e-3
            )
        inputs = self._inputs()
        inputs["active"] = torch.ones(2, dtype=torch.bool)
        output = self._forward(model, inputs)
        output["mean"].sum().backward()

        for block in (model.scale8_block, model.scale16_block):
            self.assertIsNotNone(block.feature_projection.weight.grad)
            self.assertIsNotNone(block.moment_projection.weight.grad)
            self.assertGreater(
                float(block.feature_projection.weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(block.moment_projection.weight.grad.abs().sum()), 0.0
            )
        self.assertGreater(
            float(
                model.moment_coordinator.total_output_a.weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for module in model._foundation_modules()
                for parameter in module.parameters()
            )
        )

    def test_coordinated_state_dict_round_trips_strictly(self) -> None:
        source = CoordinatedReMSTNet()
        restored = CoordinatedReMSTNet()
        incompatibility = restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatibility.missing_keys, [])
        self.assertEqual(incompatibility.unexpected_keys, [])

    def test_adaptive_progress_mixing_has_bounded_trainable_budget(self) -> None:
        torch.manual_seed(20260822)
        model = CoordinatedReMSTNet(
            use_progress_mixing=True,
            learnable_budget_gain=True,
        ).train()
        counts = remstnet_parameter_counts(model)
        self.assertNotIn("EfficientNet", ADAPTIVE_REMST_NET_ARCHITECTURE)
        self.assertEqual(counts["cross_scale_moment_coordinator"], 202_718)
        self.assertEqual(counts["trainable"], 261_528)
        self.assertEqual(counts["total_unique"], 4_271_638)

        inputs = self._inputs()
        inputs["active"] = torch.ones(2, dtype=torch.bool)
        initial = self._forward(model, inputs)
        self.assertEqual(initial["architecture"], ADAPTIVE_REMST_NET_ARCHITECTURE)
        self.assertTrue(
            torch.equal(
                initial["progress_posterior"],
                initial["sarn_endpoint_posterior"],
            )
        )
        self.assertTrue(
            torch.equal(initial["moment_budget_gain"], torch.ones(2))
        )

        with torch.no_grad():
            model.moment_coordinator.total_output_a.weight.fill_(1.0e-3)
            model.moment_coordinator.total_output_b.weight.fill_(1.0e-3)
            model.moment_coordinator.budget_gain_parameter.fill_(0.1)
        shifted = self._forward(model, inputs)
        shifted["mean"].sum().backward()
        gain = float(shifted["moment_budget_gain"][0].detach())
        self.assertGreater(gain, 1.0)
        self.assertLessEqual(gain, 1.5)
        self.assertGreater(
            float(model.moment_coordinator.budget_gain_parameter.grad.abs()),
            0.0,
        )
        first_layer = model.moment_coordinator.bin_decoder.layers[0]
        self.assertIsNotNone(first_layer.progress_depthwise)
        self.assertGreater(
            float(first_layer.progress_depthwise.weight.grad.abs().sum()),
            0.0,
        )

    def test_factorial_ablations_independently_toggle_both_mechanisms(self) -> None:
        expected = {
            (False, False): (
                COORDINATED_REMST_NET_ARCHITECTURE,
                193_373,
                252_183,
                4_262_293,
            ),
            (True, False): (
                PROGRESS_MIXING_ONLY_REMST_NET_ARCHITECTURE,
                202_717,
                261_527,
                4_271_637,
            ),
            (False, True): (
                ADAPTIVE_BUDGET_ONLY_REMST_NET_ARCHITECTURE,
                193_374,
                252_184,
                4_262_294,
            ),
            (True, True): (
                ADAPTIVE_REMST_NET_ARCHITECTURE,
                202_718,
                261_528,
                4_271_638,
            ),
        }
        for flags, (architecture, coordinator, trainable, total) in expected.items():
            with self.subTest(flags=flags):
                model = CoordinatedReMSTNet(
                    use_progress_mixing=flags[0],
                    learnable_budget_gain=flags[1],
                ).eval()
                counts = remstnet_parameter_counts(model)
                self.assertEqual(model.architecture, architecture)
                self.assertEqual(
                    model.construction["use_progress_mixing"], flags[0]
                )
                self.assertEqual(
                    model.construction["learnable_budget_gain"], flags[1]
                )
                self.assertEqual(
                    counts["cross_scale_moment_coordinator"], coordinator
                )
                self.assertEqual(counts["trainable"], trainable)
                self.assertEqual(counts["total_unique"], total)
                layer = model.moment_coordinator.bin_decoder.layers[0]
                self.assertEqual(layer.progress_depthwise is not None, flags[0])
                self.assertEqual(
                    model.moment_coordinator.budget_gain_parameter is not None,
                    flags[1],
                )
                restored = CoordinatedReMSTNet(
                    use_progress_mixing=flags[0],
                    learnable_budget_gain=flags[1],
                )
                incompatibility = restored.load_state_dict(
                    model.state_dict(), strict=True
                )
                self.assertEqual(incompatibility.missing_keys, [])
                self.assertEqual(incompatibility.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
