"""Small PMF checks for the frozen-checkpoint diagnostic interventions."""
import math
import unittest

import torch

from experiments.diagnose_geoprr_v2_relative_state import angular_metrics, intervention_joints, target_pmfs
from experiments.geoprr_v2_relative_state import RelativeStateReadout
from experiments.train_geoprr_v2_relative_state import validation_rows


class RelativeDiagnosisTests(unittest.TestCase):
    def test_twohot_seam_and_ordered_endpoint(self):
        delta = torch.tensor([2 * math.pi - .01])
        span = torch.tensor([2 * math.pi - .01])
        a, b = target_pmfs(delta, span, 8)
        self.assertGreater(float(a[0, 0]), .98)
        self.assertGreater(float(a[0, 7]), 0)
        self.assertGreater(float(b[0, 7]), .98)
        self.assertEqual(float(b[0, 0]), 0)
        stats = angular_metrics(a, delta, circular=True)
        self.assertLess(float(stats["mean_error_degrees"]), .1)
        self.assertGreater(float(stats["resultant_length"]), .99)
        torch.testing.assert_close(stats["adjacent_mass"], torch.ones(1))

    def test_interventions_preserve_unreplaced_marginal_and_mass(self):
        q = torch.zeros(1, 8, 8)
        q[0, 1, 2], q[0, 2, 4] = .4, .6
        variants = intervention_joints(q, torch.tensor([.9]), torch.tensor([4.3]))
        for p in variants.values():
            torch.testing.assert_close(p.sum((1, 2)), torch.ones(1))
        torch.testing.assert_close(variants["gt_delta_pred_span"].sum(1), q.sum(1))
        torch.testing.assert_close(variants["pred_delta_gt_span"].sum(2), q.sum(2))
        torch.testing.assert_close(variants["product_marginals"].sum(1), q.sum(1))
        torch.testing.assert_close(variants["product_marginals"].sum(2), q.sum(2))

    def test_all_four_fixed_readouts_match_atom_enumeration(self):
        q = torch.zeros(1, 8, 8)
        q[0, 1, 2], q[0, 6, 7] = .3, .7
        variants = intervention_joints(q, torch.tensor([.4]), torch.tensor([2 * math.pi - .01]))
        for p in variants.values():
            expected = torch.zeros(8)
            for d in range(8):
                for l in range(8):
                    mass = p[0, d, l]
                    if l in (0, 7):
                        expected += mass / 8
                        continue
                    span = 2 * math.pi * l / 7
                    angle = (2 * math.pi * d / 8 - span / 2 + math.pi) % (2 * math.pi) - math.pi + span / 2
                    position = min(1., max(0., angle / span)) * 7
                    lower = math.floor(position)
                    expected[lower] += mass * (1 - position + lower)
                    expected[min(7, lower + 1)] += mass * (position - lower)
            actual = RelativeStateReadout(8)(p)
            torch.testing.assert_close(actual["distribution"][0], expected, atol=2e-7, rtol=2e-6)
            torch.testing.assert_close(actual["invalid_mass"], p[:, :, (0, 7)].sum((1, 2)))

    def test_input_failure_keeps_row_with_error_one(self):
        q = torch.zeros(1, 8, 8)
        q[0, 1, 3] = 1
        output = RelativeStateReadout(8)(q)
        row = validation_rows(output, {"y": torch.tensor([.2]), "sample_id": ["sync_test"],
                              "group_id": ["group_test"], "input_failure": ["unreadable"]})[0]
        self.assertFalse(row["valid"])
        self.assertEqual(row["absolute_error"], 1.)
        self.assertIsNone(row["prediction"])


if __name__ == "__main__":
    unittest.main()
