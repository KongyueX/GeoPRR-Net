"""Focused CPU checks for C4 view grouping, scalar accounting and attempt scheduling."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments import geoprr_v2_semantic_direct as direct
from experiments.geoprr_v2_c4_orbit import (
    C4OrbitReader, TOTAL_ATTEMPTS, evaluate_population, learning_rate, orbit_views, virtual_epoch,
)


class CornerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_head = nn.Conv2d(3, 5, 1)

    def forward(self, image):
        channel = image[:, 0]
        return torch.stack((channel[:, 0, 0], channel[:, 0, -1], channel[:, -1, -1],
                            channel[:, -1, 0], channel.mean((-2, -1))), dim=1)


def reader():
    model = C4OrbitReader(CornerEncoder())
    with torch.no_grad():
        model.point.weight.zero_()
        model.point.weight[0, 0] = 1.
        model.point.bias.fill_(.3)
    return model


class Population(Dataset):
    def __init__(self, count, failed=None):
        self.rows = [{"sample_id": str(index), "group_id": "g"} for index in range(count)]
        self.failed = failed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image = torch.zeros(3, 4, 4) if index == self.failed else torch.full((3, 4, 4), index / max(1, len(self.rows)))
        return {**self.rows[index], "image": image,
                "y": torch.tensor(.2), "input_failure": "unreadable RGB" if index == self.failed else ""}


def cpu_loader(dataset, *, batch_size, **_kwargs):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class C4OrbitTests(unittest.TestCase):
    def test_k_major_grouping_and_mean_semantic_before_sigmoid_without_new_state(self):
        model = reader()
        image = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3).expand(-1, 3, -1, -1) - 9
        views = orbit_views(image)
        for k in range(4):
            self.assertTrue(torch.equal(views[2 * k:2 * (k + 1)], torch.rot90(image, k, (-2, -1))))
        semantic = model.encoder(views).reshape(4, 2, 5)
        expected = model.point(semantic.float().mean(0)).squeeze(-1).sigmoid()
        wrong = model.point(semantic).squeeze(-1).sigmoid().mean(0)
        self.assertTrue(torch.equal(model(image), expected))
        self.assertGreater(float((expected - wrong).abs().max()), .01)
        base = direct.SemanticDirectReader(CornerEncoder())
        wrapped = C4OrbitReader(base.encoder, base.point)
        self.assertIs(wrapped.point, base.point)
        self.assertEqual(set(wrapped.state_dict()), set(base.state_dict()))
        wrapped.load_state_dict(base.state_dict(), strict=True)

    def test_input_orbit_pooling_is_invariant_for_a_noninvariant_encoder(self):
        model = reader().eval()
        image = torch.tensor([[[[0., 1., 2.], [3., 4., 5.], [6., 7., 9.]]]]).expand(-1, 3, -1, -1)
        unpooled = [model.point(model.encoder(torch.rot90(image, k, (-2, -1)))) for k in range(4)]
        self.assertGreater(float((unpooled[0] - unpooled[1]).detach().abs().max()), 1.)
        original = model(image)
        for k in range(1, 4):
            self.assertTrue(torch.allclose(model(torch.rot90(image, k, (-2, -1))), original, atol=1e-7, rtol=1e-6))
        feature = image[:, :1]
        self.assertTrue(torch.equal(feature.mean((-2, -1)), torch.rot90(feature, 1, (-2, -1)).mean((-2, -1))))

    def test_final_partial_original_batch_retains_all_four_cnn_views(self):
        population = Population(18)
        with patch("experiments.geoprr_v2_c4_orbit.make_loader", cpu_loader):
            rows, cost = evaluate_population(reader(), population, torch.device("cpu"))
        self.assertEqual([row["sample_id"] for row in rows], [str(index) for index in range(18)])
        self.assertEqual(cost["counts"]["cnn_forward_calls"], 2)
        self.assertEqual(cost["counts"]["cnn_image_forwards"], 72)
        self.assertEqual(cost["counts"]["original_image_presentations"], 18)

    def test_failed_original_retains_four_placeholder_views_and_one_error_one_row(self):
        population = Population(3, failed=1)
        with patch("experiments.geoprr_v2_c4_orbit.make_loader", cpu_loader):
            rows, cost = evaluate_population(reader(), population, torch.device("cpu"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["absolute_error"], 1.)
        self.assertIsNone(rows[1]["prediction"])
        self.assertEqual(sum(not row["valid"] for row in rows), 1)
        self.assertEqual(cost["counts"]["cnn_image_forwards"], 12)
        self.assertEqual(cost["counts"]["failed_original_inputs"], 1)
        self.assertEqual(cost["counts"]["failed_placeholder_CNN_views"], 4)

    def test_virtual_schedule_uses_all_attempts_including_amp_skips_and_resume(self):
        for index, epoch in ((0, 1), (8714, 30), (8715, 31), (17429, 60)):
            self.assertEqual(virtual_epoch(index), epoch)
            self.assertEqual(learning_rate(index), direct.learning_rate(epoch))
        histogram = [0] * 60
        for index in range(TOTAL_ATTEMPTS):
            histogram[virtual_epoch(index) - 1] += 1
        self.assertEqual(sum(histogram), TOTAL_ATTEMPTS)
        self.assertEqual(set(histogram), {290, 291})
        effective_updates, amp_skips = 1161, 1
        saved_next_attempt = effective_updates + amp_skips
        self.assertEqual(saved_next_attempt, 1162)
        self.assertEqual(virtual_epoch(saved_next_attempt), 5)
        self.assertEqual(virtual_epoch(effective_updates), 4)
        self.assertNotEqual(learning_rate(saved_next_attempt), learning_rate(effective_updates))
        self.assertEqual(virtual_epoch(7 * 1162 + 580), 30)
        self.assertEqual(virtual_epoch(7 * 1162 + 581), 31)
        self.assertEqual(virtual_epoch(8 * 1162 - 1), 32)
        with self.assertRaises(ValueError):
            virtual_epoch(-1)
        with self.assertRaises(ValueError):
            learning_rate(TOTAL_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
