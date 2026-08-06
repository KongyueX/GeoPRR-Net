from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments.syncg_numeric_ocr import VOCABULARY, ctc_prefix_beam_search
from experiments.syncg_strong_numeric_ocr import (
    DEFAULT_OFFICIAL_WEIGHTS,
    OFFICIAL_MOBILENET_V3_SMALL_SHA256,
    STRONG_ARCHITECTURE,
    STRONG_CHECKPOINT_PROTOCOL,
    MobileSVTRCTCRecognizer,
    load_strong_recognizer_checkpoint,
    model_inventory,
    verify_official_backbone,
    sha256_file,
)
from experiments.train_syncg_strong_numeric_ocr import ACTIVATION_GATE


class StrongRecognizerContractTest(unittest.TestCase):
    def test_model_is_larger_than_tiny_and_retains_ctc_top_k_contract(self) -> None:
        model = MobileSVTRCTCRecognizer().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 1, 32, 160))
        self.assertEqual(tuple(logits.shape), (16, 2, len(VOCABULARY)))
        self.assertGreater(model_inventory(model)["parameters"], 2_000_000)
        beams = ctc_prefix_beam_search(logits, beam_width=5)
        self.assertEqual(len(beams), 2)
        self.assertTrue(all(1 <= len(values) <= 5 for values in beams))

    def test_official_weight_hash_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary) / "weights.pth"
            wrong.write_bytes(b"not official weights")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_official_backbone(wrong)

        if DEFAULT_OFFICIAL_WEIGHTS.exists():
            identity = verify_official_backbone(DEFAULT_OFFICIAL_WEIGHTS)
            self.assertEqual(identity["sha256"], OFFICIAL_MOBILENET_V3_SMALL_SHA256)

    def test_formal_checkpoint_round_trip_preserves_provenance(self) -> None:
        model = MobileSVTRCTCRecognizer()
        checkpoint = {
            "protocol": STRONG_CHECKPOINT_PROTOCOL,
            "status": "complete",
            "component": "recognizer",
            "vocabulary": list(VOCABULARY),
            "model_config": {
                "architecture": STRONG_ARCHITECTURE,
                "embedding_dim": 256,
                "attention_heads": 8,
                "attention_layers": 4,
                "feedforward_dim": 768,
                "dropout": 0.10,
            },
            "initialization": {"sha256": OFFICIAL_MOBILENET_V3_SMALL_SHA256},
            "code": {"implementation_sha256": sha256_file(
                Path("experiments/syncg_strong_numeric_ocr.py")
            )},
            "state_dict": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strong.pt"
            torch.save(checkpoint, path)
            restored, value = load_strong_recognizer_checkpoint(path)
        self.assertEqual(value["model_config"]["architecture"], STRONG_ARCHITECTURE)
        self.assertEqual(model_inventory(restored), model_inventory(model))


class FrozenSwitchGateTest(unittest.TestCase):
    def test_gate_separates_recognition_and_detection_failures(self) -> None:
        recognizer = ACTIVATION_GATE["activate_strong_recognizer_if_any"]
        detector = ACTIVATION_GATE["activate_dbnet_plus_plus_detector_if_any"]
        self.assertEqual(recognizer["tiny_validation_exact_accuracy_below"], 0.92)
        self.assertEqual(detector["garc_full_denominator_coverage_below"], 0.90)
        self.assertNotIn("garc_full_denominator_coverage_below", recognizer)
        self.assertEqual(ACTIVATION_GATE["selection_data"], "calibration physical groups only")


if __name__ == "__main__":
    unittest.main()
