from __future__ import annotations

import math
import unittest

import torch

from experiments.geometric_progress_decoder import decode_geometric_progress


class GeometricProgressDecoderTests(unittest.TestCase):
    def test_clockwise_quarter_arc_midpoint(self) -> None:
        pivot = torch.tensor([[0.5, 0.5]])
        direction = torch.tensor(
            [[math.sqrt(0.5), math.sqrt(0.5)]], dtype=torch.float32
        )
        references = torch.tensor([[[0.5, 0.1], [0.9, 0.5]]])
        result = decode_geometric_progress(
            pivot, direction, references, clockwise=True
        )
        torch.testing.assert_close(
            result["progress"], torch.tensor([0.5]), atol=1.0e-6, rtol=0.0
        )
        self.assertTrue(bool(result["valid"][0]))
        self.assertTrue(bool(result["inside_arc"][0]))

    def test_outside_arc_selects_nearest_endpoint(self) -> None:
        pivot = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        # Just counter-clockwise of the start, then just clockwise of the end.
        direction = torch.tensor([[-0.1, 0.995], [0.995, -0.1]])
        references = torch.tensor(
            [
                [[0.5, 0.1], [0.9, 0.5]],
                [[0.5, 0.1], [0.9, 0.5]],
            ]
        )
        result = decode_geometric_progress(
            pivot, direction, references, clockwise=True
        )
        torch.testing.assert_close(
            result["progress"], torch.tensor([0.0, 1.0]), atol=0.0, rtol=0.0
        )
        self.assertTrue(bool(result["valid"].all()))
        self.assertFalse(bool(result["inside_arc"].any()))


if __name__ == "__main__":
    unittest.main()
