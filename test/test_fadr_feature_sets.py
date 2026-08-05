from __future__ import annotations

import unittest

from experiments.calibrated_progress_router import FEATURE_NAMES
from experiments.fadr_feature_sets import (
    CROSS_REPRESENTATION_DISAGREEMENT_FEATURES,
    FADR_ROUTER_FEATURE_SETS,
    REFERENCE_CONDITIONED_ROUTING_FEATURES,
)
from experiments.quality_router import RAW_QUALITY_FEATURES
from experiments.uncertainty_fusion import NATIVE_UNCERTAINTY_FEATURE_NAMES


class FadrFeatureSetsTest(unittest.TestCase):
    def test_full_schema_is_the_frozen_59_vector(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 59)
        self.assertEqual(FADR_ROUTER_FEATURE_SETS["full"], tuple(FEATURE_NAMES))

    def test_expected_feature_counts_are_stable(self) -> None:
        self.assertEqual(
            {
                name: len(features)
                for name, features in FADR_ROUTER_FEATURE_SETS.items()
            },
            {
                "full": 59,
                "disagreement_only": 12,
                "without_mask_quality": 35,
                "without_pepd_native_uncertainty": 52,
                "without_reference_conditioned_router_evidence": 49,
            },
        )

    def test_disagreement_only_has_no_quality_or_uncertainty_features(self) -> None:
        disagreement = set(CROSS_REPRESENTATION_DISAGREEMENT_FEATURES)
        self.assertTrue(disagreement.isdisjoint(RAW_QUALITY_FEATURES))
        self.assertTrue(disagreement.isdisjoint(NATIVE_UNCERTAINTY_FEATURE_NAMES))
        self.assertEqual(
            set(FADR_ROUTER_FEATURE_SETS["disagreement_only"]),
            disagreement,
        )

    def test_family_removals_are_exact(self) -> None:
        full = set(FEATURE_NAMES)
        self.assertEqual(
            set(FADR_ROUTER_FEATURE_SETS["without_mask_quality"]),
            full - set(RAW_QUALITY_FEATURES),
        )
        self.assertEqual(
            set(FADR_ROUTER_FEATURE_SETS["without_pepd_native_uncertainty"]),
            full - set(NATIVE_UNCERTAINTY_FEATURE_NAMES),
        )
        self.assertEqual(
            set(
                FADR_ROUTER_FEATURE_SETS[
                    "without_reference_conditioned_router_evidence"
                ]
            ),
            full - set(REFERENCE_CONDITIONED_ROUTING_FEATURES),
        )

    def test_every_variant_preserves_full_schema_order(self) -> None:
        for features in FADR_ROUTER_FEATURE_SETS.values():
            selected = set(features)
            self.assertEqual(
                features,
                tuple(name for name in FEATURE_NAMES if name in selected),
            )


if __name__ == "__main__":
    unittest.main()
