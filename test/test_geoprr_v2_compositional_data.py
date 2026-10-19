from __future__ import annotations

import unittest

import numpy as np

from experiments.prepare_geoprr_v2_compositional_data import edges_for_group, generation_plan


class CompositionalSourceTests(unittest.TestCase):
    def test_frozen_plan_spreads_continuous_states_and_layout_ids(self) -> None:
        plan = generation_plan()
        self.assertEqual(plan, generation_plan())
        self.assertEqual(len({group["layout_id"] for group in plan}), 1024)
        self.assertEqual(len({group["render_seed"] for group in plan}), 1024)
        states = np.asarray([group["states"] for group in plan])
        np.testing.assert_array_equal(np.floor(states * 4), np.tile(np.arange(4), (1024, 1)))
        self.assertTrue(np.all((states > 0) & (states < 1)))
        self.assertEqual(len(set(states[:, 0])), 1024)

    def test_heldout_reverse_edges_never_enter_training(self) -> None:
        rows = [{"sample_id": f"s{index}", "metadata": {"layout_id": "layout", "factors": {"r": index / 4}}}
                for index in range(4)]
        training, heldout = edges_for_group(rows, [2, 0, 3, 1])
        edge = lambda row: (row["sample_a"], row["sample_b"])
        trained, tested = set(map(edge, training)), set(map(edge, heldout))
        self.assertEqual((len(trained), len(tested)), (8, 4))
        self.assertFalse(trained & tested)
        self.assertEqual(trained, {(b, a) for a, b in trained})
        self.assertEqual(tested, {(b, a) for a, b in tested})
        self.assertEqual(trained | tested, {(f"s{a}", f"s{b}") for a in range(4) for b in range(4) if a != b})
        for donor in range(4):
            self.assertEqual(sum(a == f"s{donor}" for a, _ in trained), 2)
            self.assertEqual(sum(a == f"s{donor}" for a, _ in tested), 1)

    def test_invalid_vertex_partition_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            edges_for_group([{}] * 4, [0, 1, 1, 3])


if __name__ == "__main__":
    unittest.main()
