from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.train_geoprr_v2 import (
    controlled_pair_losses,
    make_loader,
    prediction_rows,
    reading_losses,
    save_checkpoint,
    source_split_check,
    structure_losses,
    summarize_rows,
    two_hot,
    warp_to_partner,
)
from experiments.evaluate_geoprr_v2 import FullPopulationDataset


class GeoPRRV2TrainingTests(unittest.TestCase):
    def test_interrupted_checkpoint_write_preserves_previous_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save({"epoch": 1}, path)
            def interrupted_save(value, temporary):
                Path(temporary).write_bytes(b"partial")
                raise OSError("interrupted checkpoint write")
            with patch("experiments.train_geoprr_v2.torch.save", side_effect=interrupted_save):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"epoch": 2})
            self.assertEqual(torch.load(path, weights_only=False)["epoch"], 1)

    def test_cross_entropy_recovers_gradient_from_saturated_wrong_direct_bin(self) -> None:
        logits = torch.tensor([[22.0, 0.0, 0.0]], requires_grad=True)
        probabilities = logits.softmax(1)
        reading = (probabilities * torch.linspace(0, 1, 3)).sum(1)
        output = {"appearance_logits": logits, "appearance_probs": probabilities,
                  "appearance_reading": reading, "reading": reading, "distribution": probabilities}
        target = torch.ones(1)
        original = reading_losses(output, target, structural_warmup=True, appearance_auxiliary=True)
        self.assertNotIn("appearance_ce", original)
        losses = reading_losses(output, target, structural_warmup=True,
                                appearance_auxiliary=True, cross_entropy=True)
        losses["appearance_ce"].backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.9)
        self.assertLess(float(logits.grad[0, 2]), -0.9)

    def test_direct_control_receives_same_stable_distribution_supervision(self) -> None:
        logits = torch.zeros(2, 5, requires_grad=True)
        probabilities = logits.softmax(1)
        output = {"appearance_logits": logits, "appearance_probs": probabilities,
                  "reading": (probabilities * torch.linspace(0, 1, 5)).sum(1), "distribution": probabilities}
        losses = reading_losses(output, torch.tensor([0.1, 0.9]), structural_warmup=False, cross_entropy=True)
        losses["reading_ce"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_epoch_seed_reproduces_resume_shuffle_without_restarting_epoch_zero(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(32))
        def order(epoch: int) -> torch.Tensor:
            loader = make_loader(dataset, batch_size=8, workers=0, shuffle=True, seed=20261019 + epoch, device=torch.device("cpu"))
            return torch.cat([batch[0] for batch in loader])
        epoch_zero, epoch_five = order(0), order(5)
        resumed_epoch_five = order(5)
        self.assertFalse(torch.equal(epoch_zero, epoch_five))
        self.assertTrue(torch.equal(epoch_five, resumed_epoch_five))

    def test_two_hot_preserves_continuous_targets_and_endpoints(self) -> None:
        target = torch.tensor([0.0, 0.0001, 0.12345, 0.5, 0.9999, 1.0])
        probability = two_hot(target, 256)
        self.assertTrue(torch.allclose(probability.sum(1), torch.ones_like(target)))
        self.assertTrue(torch.allclose((probability * torch.linspace(0, 1, 256)).sum(1), target, atol=1e-7))
        self.assertEqual(float(probability[0, 0]), 1.0)
        self.assertEqual(float(probability[-1, -1]), 1.0)

    def test_two_hot_rejects_invalid_reading_labels(self) -> None:
        for value in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                two_hot(torch.tensor([value]), 8)

    def test_field_supervision_excludes_undefined_pixels(self) -> None:
        field = torch.tensor([[[[0.9, 0.2]], [[0.1, 0.8]]]], requires_grad=True)
        output = {"field_probs": field, "field_logits": field.log(), "point_logits": torch.zeros(1, 1, 1, 2),
                  "support_logits": torch.zeros(1, 1, 1, 2)}
        batch = {"field_target": torch.tensor([[[0.0, 1.0]]]), "field_mask": torch.tensor([[[[1.0, 0.0]]]]),
                 "point_target": torch.tensor([[[[1.0, 0.0]]]]), "support_target": torch.tensor([[[[1.0, 0.0]]]])}
        losses = structure_losses(output, batch)
        losses["field_ce"].backward()
        self.assertEqual(float(field.grad[..., 1].abs().sum()), 0.0)
        self.assertAlmostEqual(float(losses["field_l1"]), 0.1, places=6)

    def test_identity_and_translation_follow_roi_pixel_centers(self) -> None:
        source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        identity = torch.eye(3)[None]
        warped, valid, jacobian = warp_to_partner(source, identity, identity, image_size=16)
        self.assertTrue(torch.allclose(warped, source, atol=1e-6))
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(jacobian, torch.ones_like(jacobian)))
        translated = identity.clone()
        translated[:, 0, 2] = 4
        warped, valid, _ = warp_to_partner(source, identity, translated, image_size=16)
        self.assertTrue(torch.allclose(warped[..., :-1], source[..., 1:], atol=1e-6))
        self.assertFalse(bool(valid[..., -1].any()))

    def test_cross_state_composition_reads_partner_state(self) -> None:
        field_target = torch.linspace(0, 1, 5).reshape(1, 1, 5).expand(1, 5, 5)
        field = two_hot(field_target, 5, dim=1)
        point_a, point_b = torch.zeros(1, 1, 5, 5), torch.zeros(1, 1, 5, 5)
        point_a[0, 0, 2, 1], point_b[0, 0, 2, 3] = 1, 1
        support = torch.ones(1, 1, 5, 5)
        source = {"field_probs": field, "point_probs": point_a, "support_probs": support}
        partner = {"field_probs": field, "point_probs": point_b, "support_probs": support}
        batch = {"pair_valid": torch.tensor([True]), "pair_same_state": torch.tensor([False]),
                 "roi_to_canonical": torch.eye(3)[None], "pair_roi_to_canonical": torch.eye(3)[None],
                 "field_mask": support, "pair_field_mask": support, "pair_y": torch.tensor([0.75])}
        losses, counts = controlled_pair_losses(source, partner, batch, image_size=20, support_threshold=0.15)
        self.assertAlmostEqual(float(losses["pair_reading_l1"]), 0.0, places=6)
        self.assertEqual(counts, {"eligible": 1, "field_valid": 1})
        source["support_probs"] = torch.zeros_like(support)
        losses, counts = controlled_pair_losses(source, partner, batch, image_size=20, support_threshold=0.15)
        self.assertNotIn("pair_reading_l1", losses)
        self.assertAlmostEqual(float(losses["pair_support_mass"]), 0.15, places=6)
        self.assertEqual(counts, {"eligible": 1, "field_valid": 0})

    def test_invalid_predictions_remain_in_denominator(self) -> None:
        output = {"reading": torch.tensor([0.2, math.nan]), "field_valid": torch.tensor([True, False]),
                  "support_mass": torch.tensor([0.8, 0.0])}
        batch = {"y": torch.tensor([0.2, 0.4]), "sample_id": ["a", "b"], "group_id": ["g1", "g2"]}
        rows = prediction_rows(output, batch)
        result = summarize_rows(rows)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertAlmostEqual(result["nmae"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["acc_at_1"], 0.5)
        self.assertEqual(result["tail10_nmae"], 1.0)

    def test_failed_roi_keeps_real_target_and_missing_target_stops_scoring(self) -> None:
        class MissingImageDataset:
            rows = [{"sample_id": "missing", "group_id": "group", "y": 0.3}]
            image_size = 32
            def __len__(self) -> int:
                return len(self.rows)
            def __getitem__(self, index: int) -> dict:
                raise FileNotFoundError("missing ROI")
        dataset = MissingImageDataset()
        sample = FullPopulationDataset(dataset)[0]
        self.assertFalse(sample["input_valid"])
        self.assertAlmostEqual(float(sample["y"]), 0.3, places=6)
        self.assertEqual(sample["sample_id"], "missing")
        dataset.rows = [{"sample_id": "unlabeled", "group_id": "group"}]
        with self.assertRaises(ValueError):
            FullPopulationDataset(dataset)

    def test_model_selection_rejects_target_data_and_group_overlap(self) -> None:
        a = {"sample_id": "a", "group_id": "layout_a", "dataset": "SyncG", "split": "train", "metadata": {"scene_name": "room_a.blend"}}
        b = {"sample_id": "b", "group_id": "layout_b", "dataset": "SyncG", "split": "train", "metadata": {"scene_name": "room_b.blend"}}
        source_split_check([a], [b])
        for invalid in (dict(b, dataset="Industrial"), dict(b, split="test"), dict(b, split="validation"),
                        {key: value for key, value in b.items() if key != "split"}, dict(b, group_id="layout_a"),
                        dict(b, metadata={"scene_name": "room_a.blend"})):
            with self.assertRaises(ValueError):
                source_split_check([a], [invalid])


if __name__ == "__main__":
    unittest.main()
