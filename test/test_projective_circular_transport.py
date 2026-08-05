import math
import unittest

import torch

from experiments.projective_circular_transport import (
    bidirectional_projective_posterior_consistency,
    circular_delta,
    fuse_circular_experts,
    periodic_linear_discrete_nll,
    projective_circular_pushforward,
    projective_circular_supervised_loss_v3,
)


def _four_heads(
    *,
    batch: int = 1,
    bins: int = 72,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pivot = torch.zeros(batch, 1, 8, 8)
    pivot[:, :, 4, 4] = 6.0
    direction = torch.tensor([[1.0, 0.0]]).repeat(batch, 1)
    angle_logits = torch.zeros(batch, bins)
    log_variance = torch.zeros(batch, 1)
    return pivot, direction, angle_logits, log_variance


class ProjectiveCircularTransportTest(unittest.TestCase):
    def test_periodic_linear_nll_interpolates_mass_not_log_mass(self):
        probability = torch.tensor(
            [[0.2, 0.6, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]],
            dtype=torch.float64,
        )
        half_bin_angle = math.pi / 4.0
        target = torch.tensor(
            [
                [math.cos(half_bin_angle), math.sin(half_bin_angle)],
                [math.cos(-math.pi / 4.0), math.sin(-math.pi / 4.0)],
            ],
            dtype=torch.float64,
        )
        score = periodic_linear_discrete_nll(torch.log(probability), target)
        expected = torch.tensor(
            [-math.log(0.4), -math.log(0.4)],
            dtype=torch.float64,
        )
        torch.testing.assert_close(score, expected, atol=1e-12, rtol=1e-12)

    def test_poe_returns_continuous_summary_and_bounded_precision(self):
        pivot, direction, logits, log_variance = _four_heads(batch=3)
        half_bin = math.pi / 72.0
        direction[0] = torch.tensor([math.cos(half_bin), math.sin(half_bin)])
        logits[0, 0:2] = 4.0
        log_variance[:, 0] = torch.tensor([-100.0, 100.0, -2.0])
        prediction = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
            min_direct_precision=0.05,
            direct_precision_cap=400.0,
        )
        self.assertEqual(tuple(prediction.probabilities.shape), (3, 360))
        torch.testing.assert_close(
            prediction.probabilities.sum(dim=1),
            torch.ones(3),
            atol=1e-6,
            rtol=1e-6,
        )
        self.assertTrue(bool(prediction.valid.all()))
        self.assertAlmostEqual(float(prediction.direct_precision[0]), 400.0, places=3)
        self.assertAlmostEqual(float(prediction.direct_precision[1]), 0.05, places=5)
        self.assertTrue(
            bool(
                torch.eq(
                    prediction.bin_expert_power
                    + prediction.direct_expert_power,
                    1.0,
                ).all()
            )
        )
        self.assertLess(
            abs(float(circular_delta(prediction.mean_angle[0], torch.tensor(half_bin)))),
            2e-3,
        )
        for value in (
            prediction.resultant_length,
            prediction.entropy,
            prediction.normalized_entropy,
            prediction.angle_std_degrees,
        ):
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_poe_confidence_controls_conflicting_expert(self):
        pivot, direction, logits, log_variance = _four_heads(batch=2)
        # The categorical expert favours 90 degrees; the direct expert favours 0.
        logits[:, 18] = 8.0
        log_variance[:, 0] = torch.tensor(
            [-math.log(0.05), -math.log(100.0)]
        )
        prediction = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
        )
        low_precision_error_to_bin = abs(
            float(
                circular_delta(
                    prediction.mean_angle[0],
                    torch.tensor(math.pi / 2.0),
                )
            )
        )
        high_precision_error_to_direct = abs(
            float(circular_delta(prediction.mean_angle[1], torch.tensor(0.0)))
        )
        self.assertLess(low_precision_error_to_bin, math.radians(5.0))
        self.assertLess(high_precision_error_to_direct, math.radians(3.0))
        self.assertGreater(
            float(prediction.resultant_length[1]),
            float(prediction.resultant_length[0]),
        )

    def test_direct_temperature_softens_correlated_expert(self):
        pivot, direction, logits, log_variance = _four_heads(batch=2)
        logits[:, 18] = 6.0
        log_variance[:, 0] = -math.log(100.0)
        cold = fuse_circular_experts(
            pivot[:1],
            direction[:1],
            logits[:1],
            log_variance[:1],
            direct_precision_temperature=1.0,
        )
        warm = fuse_circular_experts(
            pivot[1:],
            direction[1:],
            logits[1:],
            log_variance[1:],
            direct_precision_temperature=20.0,
        )
        self.assertAlmostEqual(
            float(cold.effective_direct_precision[0]),
            100.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(warm.effective_direct_precision[0]),
            5.0,
            places=4,
        )
        cold_error_to_direct = abs(
            float(circular_delta(cold.mean_angle[0], torch.tensor(0.0)))
        )
        warm_error_to_bin = abs(
            float(
                circular_delta(
                    warm.mean_angle[0],
                    torch.tensor(math.pi / 2.0),
                )
            )
        )
        self.assertLess(cold_error_to_direct, warm_error_to_bin)

    def test_classical_untempered_poe_is_explicit_sharper_ablation(self):
        pivot, direction, logits, log_variance = _four_heads()
        logits[0, 0] = 5.0
        log_variance[0, 0] = -math.log(20.0)
        default_pool = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
        )
        classical_poe = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
            bin_expert_power=1.0,
            direct_expert_power=1.0,
        )
        self.assertGreater(
            float(classical_poe.resultant_length[0]),
            float(default_pool.resultant_length[0]),
        )
        self.assertLess(
            float(classical_poe.normalized_entropy[0]),
            float(default_pool.normalized_entropy[0]),
        )

    def test_invalid_poe_row_is_finite_uniform_and_marked_invalid(self):
        pivot, direction, logits, log_variance = _four_heads()
        direction[0, 0] = float("nan")
        prediction = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
        )
        self.assertFalse(bool(prediction.valid[0]))
        self.assertTrue(bool(torch.isfinite(prediction.probabilities).all()))
        torch.testing.assert_close(
            prediction.probabilities,
            torch.full((1, 360), 1.0 / 360.0),
            atol=1e-7,
            rtol=1e-7,
        )

    def test_72_bin_expert_periodically_upsamples_to_normalized_360_support(self):
        pivot, direction, logits, log_variance = _four_heads()
        logits[0, 71] = 10.0
        prediction = fuse_circular_experts(
            pivot,
            direction,
            logits,
            log_variance,
            bin_expert_power=1.0,
            direct_expert_power=0.0,
            posterior_bins=360,
        )
        self.assertEqual(
            int(torch.argmax(prediction.probabilities[0])),
            355,
        )
        self.assertAlmostEqual(
            float(prediction.probabilities.sum()),
            1.0,
            places=6,
        )
        # The last source interval (355..360 degrees) interpolates periodically
        # from source bin 71 back to source bin 0.
        self.assertGreater(
            float(prediction.probabilities[0, 359]),
            float(prediction.probabilities[0, 0]),
        )
        identity = torch.eye(3)[None]
        transported = projective_circular_pushforward(
            prediction.probabilities,
            prediction.pivot_xy,
            identity,
        )
        self.assertTrue(
            torch.equal(
                transported.probabilities,
                prediction.probabilities,
            )
        )

    def test_identity_pushforward_is_bitwise_identical(self):
        generator = torch.Generator().manual_seed(17)
        probabilities = torch.rand(3, 72, generator=generator, dtype=torch.float64)
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        pivot = torch.tensor(
            [[10.0, 20.0], [0.0, 0.0], [150.5, 42.25]],
            dtype=torch.float64,
        )
        identity = torch.eye(3, dtype=torch.float64)[None].repeat(3, 1, 1)
        transported = projective_circular_pushforward(
            probabilities,
            pivot,
            identity,
        )
        self.assertTrue(torch.equal(transported.probabilities, probabilities))
        self.assertTrue(bool(transported.valid.all()))
        self.assertTrue(
            torch.equal(
                transported.mapped_bin_coordinate,
                torch.arange(72, dtype=torch.float64)[None].repeat(3, 1),
            )
        )

    def test_rotation_wraparound_uses_periodic_linear_splat(self):
        bins = 72
        probabilities = torch.zeros(1, bins, dtype=torch.float64)
        probabilities[0, bins - 1] = 1.0
        rotation = 1.5 * (2.0 * math.pi / bins)
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        homography = torch.tensor(
            [[[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        )
        transported = projective_circular_pushforward(
            probabilities,
            torch.tensor([[12.0, 7.0]], dtype=torch.float64),
            homography,
        )
        self.assertTrue(bool(transported.valid[0]))
        self.assertAlmostEqual(float(transported.probabilities[0, 0]), 0.5, places=10)
        self.assertAlmostEqual(float(transported.probabilities[0, 1]), 0.5, places=10)
        self.assertAlmostEqual(float(transported.probabilities.sum()), 1.0, places=12)

    def test_perspective_pushforward_conserves_arbitrary_mass(self):
        generator = torch.Generator().manual_seed(9)
        probabilities = torch.rand(2, 72, generator=generator, dtype=torch.float64)
        probabilities[0] *= 2.3 / probabilities[0].sum()
        probabilities[1] *= 0.7 / probabilities[1].sum()
        pivot = torch.tensor([[64.0, 64.0], [80.0, 35.0]], dtype=torch.float64)
        homography = torch.tensor(
            [
                [[1.0, 0.08, 3.0], [-0.04, 0.95, 2.0], [0.0008, -0.0004, 1.0]],
                [[0.9, -0.12, 5.0], [0.05, 1.1, -3.0], [-0.0005, 0.0007, 1.0]],
            ],
            dtype=torch.float64,
        )
        transported = projective_circular_pushforward(
            probabilities,
            pivot,
            homography,
        )
        self.assertTrue(bool(transported.valid.all()))
        torch.testing.assert_close(
            transported.target_mass,
            transported.source_mass,
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertTrue(bool((transported.probabilities >= 0.0).all()))

    def test_pushforward_is_invariant_to_projective_matrix_scale(self):
        generator = torch.Generator().manual_seed(17)
        probability = torch.rand(
            1, 360, generator=generator, dtype=torch.float64
        )
        probability /= probability.sum(dim=1, keepdim=True)
        pivot = torch.tensor([[0.31, -0.27]], dtype=torch.float64)
        base = torch.tensor(
            [
                [
                    [1.1, 0.17, 0.3],
                    [-0.08, 0.93, -0.2],
                    [0.021, -0.014, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        reference = projective_circular_pushforward(probability, pivot, base)
        self.assertTrue(bool(reference.valid.all()))
        for scale in (1e-12, -3.0, 1.0, 1e12):
            candidate = projective_circular_pushforward(
                probability,
                pivot,
                base * scale,
            )
            self.assertTrue(bool(candidate.valid.all()))
            torch.testing.assert_close(
                candidate.probabilities,
                reference.probabilities,
                atol=1e-11,
                rtol=1e-11,
            )
            torch.testing.assert_close(
                candidate.transformed_pivot_xy,
                reference.transformed_pivot_xy,
                atol=1e-12,
                rtol=1e-12,
            )
            torch.testing.assert_close(
                candidate.mapped_bin_coordinate,
                reference.mapped_bin_coordinate,
                atol=1e-11,
                rtol=1e-11,
            )
            torch.testing.assert_close(
                candidate.target_mass,
                reference.target_mass,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_scaled_projective_identity_is_exact_bypass(self):
        probability = torch.rand(4, 72, dtype=torch.float64)
        pivot = torch.tensor([[3.0, 4.0]]).repeat(4, 1).to(torch.float64)
        scales = torch.tensor([1e-12, -3.0, 1.0, 1e12], dtype=torch.float64)
        homography = (
            torch.eye(3, dtype=torch.float64)[None] * scales[:, None, None]
        )
        result = projective_circular_pushforward(
            probability,
            pivot,
            homography,
        )
        self.assertTrue(bool(result.valid.all()))
        self.assertTrue(bool(torch.equal(result.probabilities, probability)))
        self.assertTrue(bool(torch.equal(result.transformed_pivot_xy, pivot)))

    def test_transport_has_finite_probability_pivot_and_homography_gradients(self):
        generator = torch.Generator().manual_seed(91)
        probabilities = torch.rand(
            2,
            360,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        pivot = torch.tensor(
            [[40.0, 52.0], [75.0, 63.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        homography = torch.tensor(
            [
                [[1.0, 0.05, 2.0], [-0.03, 0.97, 1.0], [0.001, -0.0004, 1.0]],
                [[0.92, -0.08, 4.0], [0.04, 1.06, -2.0], [-0.0003, 0.0008, 1.0]],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        transported = projective_circular_pushforward(
            probabilities,
            pivot,
            homography,
        )
        coordinate_weight = torch.linspace(
            -1.0,
            1.0,
            360,
            dtype=torch.float64,
        )
        objective = torch.sum(
            transported.probabilities * coordinate_weight[None, :]
        )
        objective.backward()
        for tensor in (probabilities, pivot, homography):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(torch.linalg.vector_norm(tensor.grad)), 0.0)

    def test_invalid_homography_is_finite_mass_preserving_fallback(self):
        probabilities = torch.full((2, 72), 1.0 / 72.0)
        pivot = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
        homography = torch.zeros(2, 3, 3)
        homography[1, 0, 0] = float("nan")
        transported = projective_circular_pushforward(
            probabilities,
            pivot,
            homography,
        )
        self.assertFalse(bool(transported.valid.any()))
        self.assertTrue(bool(torch.isfinite(transported.probabilities).all()))
        torch.testing.assert_close(transported.probabilities, probabilities)
        torch.testing.assert_close(
            transported.target_mass,
            transported.source_mass,
        )

    def test_bidirectional_rotation_consistency_and_invalid_fraction(self):
        bins = 72
        first = torch.softmax(torch.linspace(-2.0, 3.0, bins)[None], dim=1)
        angle = math.pi / 2.0
        homography = torch.tensor(
            [
                [
                    [math.cos(angle), -math.sin(angle), 0.0],
                    [math.sin(angle), math.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        first_pivot = torch.tensor([[10.0, 20.0]])
        forward = projective_circular_pushforward(
            first,
            first_pivot,
            homography,
        )
        result = bidirectional_projective_posterior_consistency(
            first,
            forward.probabilities,
            first_pivot,
            forward.transformed_pivot_xy,
            homography,
        )
        self.assertTrue(bool(result.valid[0]))
        self.assertEqual(float(result.valid_fraction), 1.0)
        self.assertLess(abs(float(result.loss)), 1e-6)

        invalid = bidirectional_projective_posterior_consistency(
            first,
            first,
            first_pivot,
            first_pivot,
            torch.zeros(1, 3, 3),
        )
        self.assertFalse(bool(invalid.valid[0]))
        self.assertEqual(float(invalid.valid_fraction), 0.0)
        self.assertTrue(bool(torch.isfinite(invalid.loss)))

    def test_consistency_detaches_pivot_by_default_and_allows_ablation_gradient(self):
        generator = torch.Generator().manual_seed(44)
        first_logits = torch.randn(
            1,
            72,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        second_logits = torch.randn(
            1,
            72,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        first = torch.softmax(first_logits, dim=1)
        second = torch.softmax(second_logits, dim=1)
        homography = torch.tensor(
            [[[1.0, 0.1, 2.0], [-0.04, 0.95, 3.0], [0.002, -0.001, 1.0]]],
            dtype=torch.float64,
        )
        first_pivot = torch.tensor(
            [[30.0, 40.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        second_pivot = torch.tensor(
            [[34.0, 41.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        detached = bidirectional_projective_posterior_consistency(
            first,
            second,
            first_pivot,
            second_pivot,
            homography,
        )
        detached.loss.backward(retain_graph=True)
        self.assertIsNone(first_pivot.grad)
        self.assertIsNone(second_pivot.grad)
        self.assertIsNotNone(first_logits.grad)

        joint = bidirectional_projective_posterior_consistency(
            first,
            second,
            first_pivot,
            second_pivot,
            homography,
            detach_pivot=False,
        )
        joint.loss.backward()
        for tensor in (first_pivot, second_pivot):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(torch.linalg.vector_norm(tensor.grad)), 0.0)

    def test_v3_supervised_loss_is_finite_and_backpropagates(self):
        generator = torch.Generator().manual_seed(31)
        pivot_logits = torch.randn(
            4, 1, 8, 8, generator=generator, requires_grad=True
        )
        direction_raw = torch.randn(
            4, 2, generator=generator, requires_grad=True
        )
        angle_logits = torch.randn(
            4, 72, generator=generator, requires_grad=True
        )
        log_variance = torch.full((4, 1), -2.0, requires_grad=True)
        target_heatmap = torch.zeros(4, 1, 8, 8)
        target_heatmap[:, :, 4, 4] = 1.0
        target_direction = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        )
        loss, components = projective_circular_supervised_loss_v3(
            pivot_logits,
            direction_raw,
            angle_logits,
            log_variance,
            target_heatmap,
            target_direction,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        for component in components.values():
            self.assertTrue(bool(torch.isfinite(component)))
        loss.backward()
        for tensor in (
            pivot_logits,
            direction_raw,
            angle_logits,
            log_variance,
        ):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))
            self.assertGreater(float(torch.linalg.vector_norm(tensor.grad)), 0.0)

    def test_wrong_overconfident_direct_expert_has_larger_nll(self):
        pivot, direction, logits, log_variance = _four_heads(batch=2)
        direction[:] = torch.tensor([[0.0, 1.0]])
        log_variance[:, 0] = torch.tensor(
            [-math.log(0.1), -math.log(100.0)]
        )
        target_heatmap = torch.zeros_like(pivot)
        target_heatmap[:, :, 4, 4] = 1.0
        target_direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        _, low = projective_circular_supervised_loss_v3(
            pivot[:1],
            direction[:1],
            logits[:1],
            log_variance[:1],
            target_heatmap[:1],
            target_direction[:1],
        )
        _, high = projective_circular_supervised_loss_v3(
            pivot[1:],
            direction[1:],
            logits[1:],
            log_variance[1:],
            target_heatmap[1:],
            target_direction[1:],
        )
        self.assertGreater(
            float(high["direct_circular_nll_loss"]),
            float(low["direct_circular_nll_loss"]),
        )

    def test_v3_precision_barrier_has_two_sided_recovery_gradient(self):
        pivot, direction, logits, _ = _four_heads(batch=2)
        log_variance = torch.tensor(
            [[100.0], [-100.0]],
            requires_grad=True,
        )
        target_heatmap = torch.zeros_like(pivot)
        target_heatmap[:, :, 4, 4] = 1.0
        target_direction = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        loss, components = projective_circular_supervised_loss_v3(
            pivot,
            direction,
            logits,
            log_variance,
            target_heatmap,
            target_direction,
            pivot_weight=0.0,
            direct_nll_weight=0.0,
            bin_ce_weight=0.0,
            fused_posterior_weight=0.0,
            fused_mean_weight=0.0,
            overconfidence_weight=0.0,
            precision_barrier_weight=1.0,
        )
        self.assertGreater(float(components["precision_barrier_loss"]), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(log_variance.grad).all()))
        self.assertGreater(float(log_variance.grad[0, 0]), 0.0)
        self.assertLess(float(log_variance.grad[1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
