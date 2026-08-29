from __future__ import annotations

import unittest

from geoprr import (
    CANDIDATE_NAMES,
    GeoPRRNet,
    PUBLICATION_NAME,
    PUBLICATION_PROTOCOL,
    UnifiedPointerReader,
    publication_model_identity,
)


class GeoPRRPublicApiTests(unittest.TestCase):
    def test_publication_identity_is_stable(self) -> None:
        identity = publication_model_identity()
        self.assertEqual(PUBLICATION_NAME, "GeoPRR-Net")
        self.assertEqual(PUBLICATION_PROTOCOL, "geoprr_net_inference_v1")
        self.assertEqual(identity["machine_key"], "geoprr_net")
        self.assertEqual(identity["display_name"], "GeoPRR-Net")
        self.assertEqual(tuple(identity["candidate_names"]), CANDIDATE_NAMES)

    def test_geoprr_is_the_primary_public_model_class(self) -> None:
        self.assertIs(GeoPRRNet, UnifiedPointerReader)


if __name__ == "__main__":
    unittest.main()
