from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from experiments.evaluate_geoprr_perspective_scan import apply_perspective_angle
from experiments.robustness_degradations import apply_degradation
from experiments.run_cagh_v5_plain_paper_batch import ROBUSTNESS_SEED
from experiments.summarize_geoprr_figure3 import (
    CELL_BY_VARIANT,
    NO_GEOMETRY_FIXED_ROUTING,
    _cluster_bootstrap_mean_ci,
    _factorial_path,
)


class GeoPRRFigure3ExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.arange(73 * 91 * 3, dtype=np.uint32).reshape(73, 91, 3)
        self.image = np.asarray(values % 251, dtype=np.uint8)

    def test_formal_perspective_angles_reproduce_existing_pixels(self) -> None:
        for angle, condition in (
            (25, "perspective_moderate"),
            (45, "perspective_severe"),
        ):
            with self.subTest(angle=angle):
                actual, metadata = apply_perspective_angle(
                    self.image,
                    angle=angle,
                    sample_id="sync_10015",
                )
                expected, formal_metadata = apply_degradation(
                    self.image,
                    condition,
                    sample_id="sync_10015",
                    seed=ROBUSTNESS_SEED,
                )
                self.assertTrue(np.array_equal(actual, expected))
                self.assertEqual(
                    metadata["axis"],
                    formal_metadata["perspective"]["axis"],
                )
                self.assertEqual(
                    metadata["sign"],
                    formal_metadata["perspective"]["sign"],
                )

    def test_axis_and_sign_do_not_change_between_nonzero_angles(self) -> None:
        assignments = []
        support_fractions = []
        for angle in (15, 25, 35, 45, 60):
            _image, metadata = apply_perspective_angle(
                self.image,
                angle=angle,
                sample_id="sync_10015",
            )
            assignments.append((metadata["axis"], metadata["sign"]))
            support_fractions.append(metadata["valid_support_fraction"])
        self.assertEqual(len(set(assignments)), 1)
        self.assertTrue(
            all(
                left > right
                for left, right in zip(
                    support_fractions,
                    support_fractions[1:],
                )
            )
        )

    def test_zero_degrees_is_an_independent_identity_copy(self) -> None:
        output, metadata = apply_perspective_angle(
            self.image,
            angle=0,
            sample_id="sync_10015",
        )
        self.assertTrue(np.array_equal(output, self.image))
        self.assertIsNot(output, self.image)
        self.assertEqual(metadata["axis"], "none")
        self.assertEqual(metadata["sign"], 0)
        self.assertEqual(metadata["valid_support_fraction"], 1.0)

    def test_scene_cluster_bootstrap_keeps_constant_effect(self) -> None:
        rows = [
            (f"scene_{index:02d}", 0.125)
            for index in range(14)
            for _repeat in range(index + 1)
        ]
        result = _cluster_bootstrap_mean_ci(rows, replicates=100, seed=7)
        self.assertEqual(result["clusters"], 14)
        self.assertAlmostEqual(result["point_estimate"], 0.125)
        self.assertAlmostEqual(result["ci_95_lower"], 0.125)
        self.assertAlmostEqual(result["ci_95_upper"], 0.125)

    def test_joint_factorial_path_includes_the_variant_directory(self) -> None:
        cell = CELL_BY_VARIANT[NO_GEOMETRY_FIXED_ROUTING]
        actual = _factorial_path(
            Path("existing"),
            Path("joint"),
            20_262_020,
            cell,
        )
        self.assertEqual(
            actual,
            Path(
                "joint/seed_20262020/no_geometry_fixed_routing/syncg.json"
            ),
        )


if __name__ == "__main__":
    unittest.main()
