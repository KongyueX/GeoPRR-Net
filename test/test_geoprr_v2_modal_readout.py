"""Numerical contracts for the optional local-mode field readout."""
from __future__ import annotations

import unittest

import torch

from experiments.geoprr_v2_model import (
    apply_modal_field_readout,
    masked_field_readout,
)


class ModalFieldReadoutTests(unittest.TestCase):
    def test_local_mode_excludes_remote_mass_and_preserves_continuous_mean(self) -> None:
        probabilities = torch.tensor([[0.0, 0.05, 0.45, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25]])
        output = masked_field_readout(
            probabilities[:, :, None, None], torch.ones(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1), 0.15,
        )
        result = apply_modal_field_readout(output, radius=1)
        expected = probabilities.clone()
        expected[0, 8] = 0.0
        expected /= expected.sum(1, keepdim=True)
        torch.testing.assert_close(result["distribution"], expected)
        torch.testing.assert_close(result["reading"], torch.tensor([1.7 / 6.0]))
        torch.testing.assert_close(output["distribution"], probabilities)
        torch.testing.assert_close(result["field_distribution"], probabilities)
        self.assertNotEqual(float(result["reading"][0]), 2 / 8)
        for radius in (0, 9):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                apply_modal_field_readout(output, radius)

    def test_endpoint_windows_do_not_wrap(self) -> None:
        probabilities = torch.tensor([
            [0.6, 0.1, 0.0, 0.0, 0.3],
            [0.3, 0.0, 0.0, 0.1, 0.6],
        ])
        output = masked_field_readout(
            probabilities[:, :, None, None], torch.ones(2, 1, 1, 1),
            torch.ones(2, 1, 1, 1), 0.15,
        )
        result = apply_modal_field_readout(output, radius=1)
        self.assertEqual(float(result["distribution"][0, -1]), 0.0)
        self.assertEqual(float(result["distribution"][1, 0]), 0.0)
        torch.testing.assert_close(result["reading"], torch.tensor([1 / 28, 27 / 28]))

    def test_invalid_field_preserves_fallback_without_nan(self) -> None:
        field = torch.tensor([[0.0, 0.0, 1.0], [0.8, 0.0, 0.2]])
        appearance = torch.tensor([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]])
        output = masked_field_readout(
            field[:, :, None, None], torch.ones(2, 1, 1, 1),
            torch.tensor([0.0, 1.0]).view(2, 1, 1, 1), 0.15, appearance,
        )
        result = apply_modal_field_readout(output, radius=1)
        torch.testing.assert_close(result["distribution"][0], output["distribution"][0])
        torch.testing.assert_close(result["reading"][0], output["reading"][0])
        self.assertTrue(bool(torch.isfinite(result["distribution"]).all()))
        self.assertTrue(bool(torch.isfinite(result["reading"]).all()))
        output["field_valid"] = torch.zeros(2, dtype=torch.bool)
        all_invalid = apply_modal_field_readout(output, radius=1)
        torch.testing.assert_close(all_invalid["distribution"], output["distribution"])
        torch.testing.assert_close(all_invalid["reading"], output["reading"])


if __name__ == "__main__":
    unittest.main()
