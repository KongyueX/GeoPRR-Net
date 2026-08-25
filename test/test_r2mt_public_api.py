from __future__ import annotations

import unittest
from pathlib import Path

from torch import nn

from r2mt import (
    DEFAULT_ADAPTIVE_STRENGTH,
    DEFAULT_FUSION_MODE,
    PUBLICATION_NAME,
    load_r2mt_net,
    publication_model_identity,
    r2mt_parameter_counts,
)


class R2MTPublicApiTests(unittest.TestCase):
    def test_publication_identity_is_r2mt_only(self) -> None:
        identity = publication_model_identity()
        self.assertEqual(PUBLICATION_NAME, "R²MT-Net")
        self.assertEqual(identity["machine_key"], "r2mt_net")
        self.assertEqual(identity["display_name"], "R²MT-Net")
        self.assertIn("checkpoint_architecture", identity)

    def test_unique_parameter_inventory_counts_each_module_once(self) -> None:
        anchor = nn.Linear(3, 2)
        correction = nn.Linear(2, 1)
        counts = r2mt_parameter_counts(anchor, correction)
        self.assertEqual(counts["shared_anchor"], 8)
        self.assertEqual(counts["relation_and_risk_transport"], 3)
        self.assertEqual(counts["total_unique"], 11)
        self.assertEqual(counts["additional_image_encoders"], 0)

    def test_invalid_public_configuration_fails_before_checkpoint_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "adaptive_strength"):
            load_r2mt_net(
                Path("checkpoint-is-not-opened.pt"),
                device="cpu",
                adaptive_strength=-0.1,
            )
        with self.assertRaisesRegex(ValueError, "fusion mode"):
            load_r2mt_net(
                Path("checkpoint-is-not-opened.pt"),
                device="cpu",
                adaptive_strength=DEFAULT_ADAPTIVE_STRENGTH,
                fusion_mode=DEFAULT_FUSION_MODE + "_invalid",
            )


if __name__ == "__main__":
    unittest.main()
