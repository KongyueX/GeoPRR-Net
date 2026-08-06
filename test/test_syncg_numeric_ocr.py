from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.syncg_numeric_ocr import (
    CHAR_TO_INDEX,
    VOCABULARY,
    GaugeTextDetector,
    TinyCTCRecognizer,
    ctc_prefix_beam_search,
    detect_boxes_from_probability,
    encode_text,
    greedy_ctc_decode,
    grouped_three_way_split,
    normalize_bbox_to_roi,
    normalize_numeric_text,
    render_detection_target,
    resize_recognizer_crop,
    synthesize_decimal_word,
)


class NumericTextTest(unittest.TestCase):
    def test_signed_decimal_normalization_is_not_integer_classification(self) -> None:
        self.assertEqual(normalize_numeric_text("−12,50"), "-12.50")
        self.assertEqual(normalize_numeric_text("+0.125"), "+0.125")
        self.assertEqual(len(encode_text("-12.5")), 5)
        with self.assertRaises(ValueError):
            normalize_numeric_text("12 V")

    def test_greedy_and_beam_interfaces_retain_strings(self) -> None:
        text = "-12.5"
        path: list[int] = [0]
        for character in text:
            path.extend((CHAR_TO_INDEX[character], 0))
        logits = torch.full((len(path), 1, len(VOCABULARY)), -8.0)
        for step, index in enumerate(path):
            logits[step, 0, index] = 8.0
        decoded, confidence = greedy_ctc_decode(logits)
        self.assertEqual(decoded, [text])
        self.assertGreater(confidence[0], 0.99)
        beam = ctc_prefix_beam_search(logits, beam_width=5)[0]
        self.assertEqual(beam[0].text, text)
        self.assertAlmostEqual(sum(item.beam_probability for item in beam), 1.0, places=6)


class GroupSplitTest(unittest.TestCase):
    def test_three_way_split_is_deterministic_and_group_disjoint(self) -> None:
        rows = [
            {"sample_id": f"s{group}_{index}", "group_id": f"g{group}"}
            for group in range(12)
            for index in range(4)
        ]
        first = grouped_three_way_split(rows, seed=17)
        second = grouped_three_way_split(list(reversed(rows)), seed=17)
        self.assertEqual(first, second)
        group_partition: dict[str, set[str]] = {}
        for row in rows:
            group_partition.setdefault(row["group_id"], set()).add(first[row["sample_id"]])
        self.assertTrue(all(len(values) == 1 for values in group_partition.values()))
        self.assertEqual(set(first.values()), {"train", "calibration", "validation"})


class ImageContractTest(unittest.TestCase):
    def test_bbox_normalization_and_detection_target(self) -> None:
        normalized = normalize_bbox_to_roi((20, 30, 40, 50), (10, 20, 110, 120))
        self.assertEqual(normalized, (0.1, 0.1, 0.3, 0.3))
        target = render_detection_target([normalized], size=64)
        self.assertGreater(float(target.sum()), 0.0)
        boxes = detect_boxes_from_probability(target, threshold=0.4)
        self.assertEqual(len(boxes), 1)

    def test_train_only_decimal_synthesis_changes_pixels_and_label(self) -> None:
        image = np.full((20, 50), 230, dtype=np.uint8)
        cv = np.random.default_rng(3)
        output, label, synthesized = synthesize_decimal_word(image, "-125", cv)
        self.assertTrue(synthesized)
        self.assertIn(".", label)
        self.assertGreater(output.shape[1], image.shape[1])
        tensor = resize_recognizer_crop(output)
        self.assertEqual(tuple(tensor.shape), (1, 32, 160))


class ModelContractTest(unittest.TestCase):
    def test_recognizer_emits_ctc_logits(self) -> None:
        model = TinyCTCRecognizer().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 1, 32, 160))
        self.assertEqual(logits.shape[1:], (2, len(VOCABULARY)))
        self.assertGreater(logits.shape[0], 8)

    def test_detector_baseline_and_geometry_extension_shapes(self) -> None:
        baseline = GaugeTextDetector(pretrained=False).eval()
        with torch.inference_mode():
            output = baseline(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))
        with self.assertRaises(ValueError):
            baseline(torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 64, 64))

        extended = GaugeTextDetector(
            pretrained=False, annular_geometry_channels=2
        ).eval()
        with torch.inference_mode():
            output = extended(
                torch.zeros(1, 3, 64, 64), torch.zeros(1, 2, 64, 64)
            )
        self.assertEqual(tuple(output.shape), (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
