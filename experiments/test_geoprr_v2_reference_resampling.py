"""Check that the diagnostic sampler preserves the frozen head's coordinates."""
import unittest

import torch

from experiments.diagnose_geoprr_v2_reference_relation import resample_state
from experiments.geoprr_v2_reference_relation import ReferenceRelationStateHead


class ReferenceResamplingTest(unittest.TestCase):
    def test_predicted_coordinates_reproduce_native_state(self):
        torch.set_num_threads(1)
        torch.manual_seed(20262020)
        head = ReferenceRelationStateHead().eval()
        features = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 192)
        with torch.inference_mode():
            native = head(features, context)
            calibration = native["calibration"]
            replay = resample_state(head, features, calibration["center"], calibration["spd"])
        for key in ("A", "P_rel", "B_rel", "J", "product"):
            torch.testing.assert_close(replay[key], native[key], atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
