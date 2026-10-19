"""Source fold contracts for the fixed MobileNet comparison."""
import unittest

from experiments.run_geoprr_v2_optimization import source_folds


class OptimizationFoldTests(unittest.TestCase):
    def test_folds_partition_images_without_splitting_scenes(self):
        rows = [{"sample_id": f"{scene}_{index}", "dataset": "SyncG", "split": "train",
                 "metadata": {"scene_name": f"{scene}.blend"}}
                for scene, count in (("a", 7), ("b", 5), ("c", 4), ("d", 3), ("e", 2), ("f", 1))
                for index in range(count)]
        folds = source_folds(rows)
        self.assertEqual(sum(map(len, folds)), len(rows))
        self.assertEqual({r["sample_id"] for fold in folds for r in fold}, {r["sample_id"] for r in rows})
        scenes = [{r["metadata"]["scene_name"] for r in fold} for fold in folds]
        self.assertFalse(scenes[0] & scenes[1] or scenes[0] & scenes[2] or scenes[1] & scenes[2])
        self.assertEqual(source_folds(rows), folds)

    def test_test_images_cannot_enter_router_folds(self):
        with self.assertRaises(ValueError):
            source_folds([{"dataset": "SyncG", "split": "test", "metadata": {"scene_name": "a"}}])


if __name__ == "__main__":
    unittest.main()
