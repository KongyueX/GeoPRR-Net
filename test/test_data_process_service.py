import base64
import unittest

import cv2
import numpy as np

from services.DataProcessService import DataProcessService


class DataProcessServiceTest(unittest.TestCase):
    def test_base64_decode_preserves_opencv_bgr_pixels(self):
        image = np.array(
            [
                [[0, 10, 255], [25, 50, 75]],
                [[100, 125, 150], [175, 200, 225]],
            ],
            dtype=np.uint8,
        )
        encoded, buffer = cv2.imencode(".png", image)
        self.assertTrue(encoded)

        service = DataProcessService()
        success, decoded = service.get_image_from_base64(
            base64.b64encode(buffer).decode("ascii")
        )

        self.assertTrue(success)
        np.testing.assert_array_equal(decoded, image)

    def test_base64_decode_rejects_invalid_payload(self):
        success, message = DataProcessService().get_image_from_base64("not-base64!")
        self.assertFalse(success)
        self.assertEqual(message, "Error: Invalid base64 encoding")


if __name__ == "__main__":
    unittest.main()
