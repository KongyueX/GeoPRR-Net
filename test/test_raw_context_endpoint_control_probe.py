from __future__ import annotations

import unittest

import torch

from experiments.raw_context_endpoint_control_probe import (
    METHOD_ACTIVATION_NULL,
    METHOD_BASE,
    METHOD_CONTEXT_MLP,
    METHOD_LINEAR_REFRESH,
    build_control_from_foundation,
    control_parameter_counts,
)
from experiments.syncg_lightweight_regression_baselines import (
    LightweightProgressRegressor,
)


class RawContextEndpointControlProbeTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(19)
        foundation = LightweightProgressRegressor(
            "efficientnet_b0", imagenet_pretrained=False
        ).eval()
        return foundation, build_control_from_foundation(foundation)

    def test_all_arms_start_at_the_exact_base_endpoint(self) -> None:
        foundation, model = self._model()
        images = torch.rand(2, 3, 64, 64)
        foundation.eval()
        model.eval()
        with torch.inference_mode():
            expected = foundation(images)
            outputs = model(images)
        torch.testing.assert_close(outputs[METHOD_BASE], expected, rtol=1e-6, atol=1e-7)
        for method in (
            METHOD_LINEAR_REFRESH,
            METHOD_ACTIVATION_NULL,
            METHOD_CONTEXT_MLP,
        ):
            torch.testing.assert_close(
                outputs[method], outputs[METHOD_BASE], rtol=0.0, atol=0.0
            )

    def test_activation_treatment_has_matched_initial_state_and_capacity(self) -> None:
        _foundation, model = self._model()
        counts = control_parameter_counts(model)
        self.assertEqual(
            counts[METHOD_ACTIVATION_NULL], counts[METHOD_CONTEXT_MLP]
        )
        null_state = model.activation_null.state_dict()
        mlp_state = model.context_mlp.state_dict()
        self.assertEqual(set(null_state), set(mlp_state))
        self.assertTrue(
            all(torch.equal(null_state[name], mlp_state[name]) for name in null_state)
        )
        self.assertEqual(counts[METHOD_LINEAR_REFRESH], 1_281)
        self.assertEqual(
            counts["total_trainable"],
            counts[METHOD_LINEAR_REFRESH]
            + counts[METHOD_ACTIVATION_NULL]
            + counts[METHOD_CONTEXT_MLP],
        )

    def test_trainable_arms_receive_gradients_while_foundation_stays_frozen(self) -> None:
        _foundation, model = self._model()
        model.train(True)
        outputs = model(torch.rand(2, 3, 64, 64))
        target = torch.tensor([0.25, 0.75])
        loss = sum(
            torch.nn.functional.l1_loss(outputs[method], target)
            for method in (
                METHOD_LINEAR_REFRESH,
                METHOD_ACTIVATION_NULL,
                METHOD_CONTEXT_MLP,
            )
        )
        loss.backward()
        for module in (
            model.linear_refresh,
            model.activation_null,
            model.context_mlp,
        ):
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    for parameter in module.parameters()
                )
            )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.encoder.parameters())
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.base_projection.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
