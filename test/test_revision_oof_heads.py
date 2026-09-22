"""Group-isolation tests on constructed CPU features, not study results."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.revision_oof_heads import (
    CONDITIONS,
    fold_partition,
    load_features,
    load_folds,
    selection_key,
    train_oof,
    training_normalization,
)


def fixture() -> tuple[dict, dict]:
    """Small test roster; production counts are patched only by the tests."""
    rows = 30
    payload = {
        "features": torch.randn(rows, 4, generator=torch.Generator().manual_seed(71)),
        "target": torch.full((rows,), .25),
        "sample_ids": [f"test-image-{i // 6}" for i in range(rows)],
        "group_ids": [f"test-group-{i // 6}" for i in range(rows)],
        "condition": [CONDITIONS[i % 6] for i in range(rows)],
        "seed": 123, "method": "constructed-test-reader",
        "feature_definition": "four synthetic values used only for a CPU unit test",
        "dataset": "industrial", "scope": "target_adaptation", "availability_policy": "test_fixture",
    }
    return payload, {"group_to_fold": {f"test-group-{i}": i for i in range(5)}}


class RevisionOOFTests(unittest.TestCase):
    def setUp(self) -> None:
        counts = mock.patch.multiple("experiments.revision_oof_heads", EXPECTED_IMAGES=5, EXPECTED_GROUPS=5)
        counts.start()
        self.addCleanup(counts.stop)

    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_partitions_keep_entire_groups_and_conditions_together(self) -> None:
        payload, folds = fixture()
        test_memberships = torch.zeros(30, dtype=torch.int64)
        for k in range(5):
            masks, groups = fold_partition(payload["group_ids"], folds["group_to_fold"], k)
            self.assertEqual(groups["test"], [f"test-group-{k}"])
            self.assertEqual(groups["validation"], [f"test-group-{(k + 1) % 5}"])
            self.assertFalse(set(groups["train"]) & set(groups["validation"]))
            self.assertFalse(set(groups["train"]) & set(groups["test"]))
            self.assertFalse(set(groups["validation"]) & set(groups["test"]))
            self.assertEqual(int(masks["train"].sum()), 18)
            self.assertEqual(int(masks["validation"].sum()), 6)
            test_memberships += masks["test"].long()
        self.assertTrue(torch.equal(test_memberships, torch.ones(30, dtype=torch.int64)))

    def test_standardization_uses_only_training_groups(self) -> None:
        payload, folds = fixture()
        masks, _ = fold_partition(payload["group_ids"], folds["group_to_fold"], 0)
        features = payload["features"]
        mean, scale = training_normalization(features, masks["train"], True)
        modified = features.clone()
        modified[~masks["train"]] = 1.0e6
        other_mean, other_scale = training_normalization(modified, masks["train"], True)
        self.assertTrue(torch.equal(mean, other_mean))
        self.assertTrue(torch.equal(scale, other_scale))
        self.assertTrue(torch.equal(mean, features[masks["train"]].mean(0)))
        identity_mean, identity_scale = training_normalization(features, masks["train"], False)
        self.assertTrue(torch.equal(identity_mean, torch.zeros(4)))
        self.assertTrue(torch.equal(identity_scale, torch.ones(4)))

    def test_lexicographic_order_prefers_acc5_then_acc2_then_nmae(self) -> None:
        target = torch.tensor([.2, .7])
        accurate_at_five = selection_key(torch.tensor([.245, .745]), target)
        one_exact_one_bad = selection_key(torch.tensor([.2, .9]), target)
        self.assertGreater(accurate_at_five, one_exact_one_bad)
        self.assertGreater(selection_key(target + .01, target), accurate_at_five)
        self.assertGreater(selection_key(target + .005, target), selection_key(target + .01, target))

    def test_requires_explicit_feature_definition_and_exact_fold_roster(self) -> None:
        payload, folds = fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            del payload["feature_definition"]
            torch.save(payload, directory / "cache.pt")
            with self.assertRaisesRegex(ValueError, "feature_definition"):
                load_features(directory / "cache.pt", method=payload["method"], seed=123)
            folds["group_to_fold"]["unexpected-test-group"] = 0
            (directory / "folds.json").write_text(json.dumps(folds))
            with self.assertRaisesRegex(ValueError, "exactly"):
                load_folds(directory / "folds.json", payload["group_ids"])

    def test_rejects_wrong_dataset_or_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            for field, value in [("dataset", "syncg"), ("scope", "evaluation")]:
                with self.subTest(field=field):
                    payload, _ = fixture()
                    payload[field] = value
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, field):
                        load_features(path, method=payload["method"], seed=123)

    def test_rejects_invented_missing_or_duplicate_conditions_and_inconsistent_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            for case, message in [("invented", "unknown evaluation condition"),
                                  ("missing", "all six formal conditions"),
                                  ("duplicate", "duplicate image-condition"),
                                  ("target", "different targets"),
                                  ("group", "different acquisition groups")]:
                with self.subTest(case=case):
                    payload, _ = fixture()
                    if case == "invented":
                        payload["condition"][1] = "invented"
                    elif case == "missing":
                        for key in ("features", "target", "sample_ids", "group_ids", "condition"):
                            payload[key] = payload[key][1:]
                    elif case == "duplicate":
                        payload["condition"][1] = payload["condition"][0]
                    elif case == "target":
                        payload["target"][1] = .9
                    else:
                        payload["group_ids"][1] = "different-test-group"
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_features(path, method=payload["method"], seed=123)

    def test_requires_the_complete_image_and_group_counts(self) -> None:
        payload, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(payload, path)
            for field, message in [("EXPECTED_IMAGES", "6 unique images"),
                                   ("EXPECTED_GROUPS", "6 acquisition groups")]:
                with self.subTest(field=field), mock.patch(f"experiments.revision_oof_heads.{field}", 6):
                    with self.assertRaisesRegex(ValueError, message):
                        load_features(path, method=payload["method"], seed=123)

    def test_oof_saves_full_roster_and_test_labels_do_not_select_checkpoint(self) -> None:
        payload, folds = fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache_path = directory / "features.pt"
            folds_path = directory / "folds.json"
            torch.save(payload, cache_path)
            folds_path.write_text(json.dumps(folds))
            args = dict(method=payload["method"], seed=123, learning_rate=1.0e-12,
                        weight_decay=0.0, batch_size=64, loss="l1", activation="gelu", standardize=True)
            first = train_oof(cache_path, folds_path, directory / "first", **args)
            self.assertEqual(first["rows"], 30)
            self.assertEqual(first["groups"], 5)
            self.assertEqual(first["failures"], 0)
            with (directory / "first/predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["sample_id"] for r in rows], payload["sample_ids"])
            self.assertEqual(len({(r["sample_id"], r["condition"]) for r in rows}), 30)
            for row in rows:
                self.assertEqual(int(row["outer_fold"]), folds["group_to_fold"][row["group_id"]])
            checkpoint = torch.load(directory / "first/fold_0.pt", weights_only=False)
            self.assertEqual(checkpoint["selected_epoch"], 1)
            self.assertEqual(len(checkpoint["history"]), 21)
            self.assertEqual(checkpoint["groups"]["test"], ["test-group-0"])
            payload["target"][:6] = .9
            torch.save(payload, cache_path)
            train_oof(cache_path, folds_path, directory / "second", **args)
            other = torch.load(directory / "second/fold_0.pt", weights_only=False)
            self.assertEqual(checkpoint["selected_epoch"], other["selected_epoch"])
            self.assertEqual(checkpoint["selection_key"], other["selection_key"])
            self.assertTrue(all(torch.equal(value, other["state_dict"][key])
                                for key, value in checkpoint["state_dict"].items()))


if __name__ == "__main__":
    unittest.main()
