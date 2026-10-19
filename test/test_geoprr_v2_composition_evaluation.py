"""Prediction-only composition and full edge-denominator contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.evaluate_geoprr_v2_composition import (
    FullCompositionPopulation, audit_heldout_edges, compose_predictions, composition_rows,
)
from experiments.train_geoprr_v2 import summarize_rows


class CompositionEvaluationTests(unittest.TestCase):
    def test_readout_uses_both_predicted_masks_and_ignores_gt_fields(self):
        donor = {"field_probs": torch.tensor([[[[1., 0.]], [[0., 0.]], [[0., 1.]]]]),
                 "support_probs": torch.tensor([[[[1., .2]]]]), "field_mask": torch.zeros(1, 1, 1, 2)}
        recipient = {"point_probs": torch.tensor([[[[.25, .75]]]]),
                     "support_probs": torch.tensor([[[[.4, 1.]]]]), "point_target": torch.tensor([[[[1., 0.]]]])}
        output = compose_predictions(donor, recipient)
        self.assertAlmostEqual(float(output["support_mass"]), .25, places=6)
        self.assertAlmostEqual(float(output["reading"]), .6, places=6)
        self.assertTrue(bool(output["field_valid"][0]))

    def test_all_edge_failures_remain_in_summary_denominator_without_fallback(self):
        output = {"reading": torch.tensor([.75, 0., 0., 1.2, .5]),
                  "support_mass": torch.tensor([1., .1, float('nan'), 1., 1.]),
                  "field_valid": torch.tensor([True, False, False, True, True])}
        batch = {"target": torch.full((5,), .5), "input_valid": [True, True, True, True, False],
                 "input_failure": ['', '', '', '', 'missing donor'],
                 "donor_id": ['a'] * 5, "recipient_id": ['b', 'c', 'd', 'e', 'f'], "layout_id": ['group'] * 5}
        rows = composition_rows(output, batch)
        summary = summarize_rows(rows)
        self.assertEqual(summary["samples"], 5)
        self.assertEqual(summary["failed"], 4)
        self.assertAlmostEqual(summary["nmae"], .85)
        self.assertEqual([row['prediction'] for row in rows[1:]], [None] * 4)
        self.assertEqual(rows[1]['failure'], 'insufficient_predicted_support')
        self.assertEqual(rows[2]['failure'], 'nonfinite_composition')
        self.assertEqual(rows[3]['failure'], 'out_of_range_composition')
        self.assertEqual(rows[4]['failure'], 'missing donor')

    def test_input_failure_retains_declared_edge_and_recipient_target(self):
        class BrokenPairDataset:
            image_size = 32
            rows = [{"sample_id": "a", "y": .2}, {"sample_id": "b", "y": .7}]
            by_id = {"a": 0, "b": 1}
            items = [(0, {"sample_a": "a", "sample_b": "b", "layout_id": "group", "kind": "state"})]

            def __getitem__(self, index):
                raise FileNotFoundError('missing a')

        population = FullCompositionPopulation(BrokenPairDataset())
        self.assertEqual(len(population), 1)
        sample = population[0]
        self.assertFalse(sample['input_valid'])
        self.assertAlmostEqual(float(sample['target']), .7, places=6)
        self.assertEqual(sample['donor_id'], 'a')
        self.assertEqual(sample['recipient_id'], 'b')
        self.assertEqual(tuple(sample['image'].shape), (3, 32, 32))

    def test_training_edges_are_rejected_while_images_must_be_seen(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'training_pairs.jsonl'
            path.write_text(json.dumps({'sample_a': 'a', 'sample_b': 'b'}) + '\n')
            checkpoint = {'manifests': {'pairs': str(path)},
                          'run_metadata': {'sample_ids': {'train': ['a', 'b', 'c']}}}
            audit = audit_heldout_edges(checkpoint, [{'sample_a': 'a', 'sample_b': 'c'}])
            self.assertEqual(audit['training_edges'], 1)
            self.assertEqual(audit['heldout_edges'], 1)
            self.assertEqual(audit['overlap_edges'], 0)
            self.assertEqual(audit['images_seen_in_training'], 2)
            with self.assertRaisesRegex(ValueError, 'training edges'):
                audit_heldout_edges(checkpoint, [{'sample_a': 'a', 'sample_b': 'b'}])
            with self.assertRaisesRegex(ValueError, 'already present'):
                audit_heldout_edges(checkpoint, [{'sample_a': 'a', 'sample_b': 'unseen'}])


if __name__ == "__main__":
    unittest.main()
