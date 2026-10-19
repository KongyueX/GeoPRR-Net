from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.geoprr_v2_router import FEATURE_PROTOCOL, ReliabilityRouter, apply_expert_mode, build_router_features
from experiments.train_geoprr_v2_router import load_fit_data


def expert_output() -> dict[str, torch.Tensor]:
    return {"field_distribution": torch.tensor([[1., 0., 0.], [1., 0., 0.]]),
            "appearance_probs": torch.tensor([[0., 0., 1.], [0., 0., 1.]]),
            "field_reading": torch.zeros(2), "appearance_reading": torch.ones(2),
            "field_valid": torch.tensor([True, False]), "support_mass": torch.tensor([.9, .1]),
            "point_probs": torch.full((2, 1, 2, 2), .25), "global_features": torch.zeros(2, 4)}


class FixedRouter(ReliabilityRouter):
    def __init__(self, choose_field: bool) -> None:
        super().__init__(20)
        self.choose_field = choose_field

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.new_tensor([[0., 1.] if self.choose_field else [1., 0.]]).expand(len(features), -1)


class GeoPRRV2RouterTests(unittest.TestCase):
    def test_ground_truth_is_not_a_router_feature(self) -> None:
        output = expert_output()
        original = build_router_features(output)
        output.update(y=torch.ones(2), target=torch.zeros(2), ground_truth=torch.rand(2))
        torch.testing.assert_close(original, build_router_features(output))
        output["y"].zero_()
        torch.testing.assert_close(original, build_router_features(output))

    def test_invalid_field_forces_direct_even_if_router_prefers_field(self) -> None:
        output = apply_expert_mode(expert_output(), router=FixedRouter(True))
        torch.testing.assert_close(output["expert_index"], torch.tensor([0, 1]))
        torch.testing.assert_close(output["reading"], torch.tensor([0., 1.]))

    def test_routing_selects_whole_expert_without_averaging(self) -> None:
        original = expert_output()
        output = apply_expert_mode(original, router=FixedRouter(False))
        torch.testing.assert_close(output["reading"], torch.ones(2))
        torch.testing.assert_close(output["distribution"], original["appearance_probs"])
        self.assertNotEqual(float(output["reading"][0]), .5)

    @staticmethod
    def cache(path: Path, *, role: str, scene: str, sample: str, split: str = "train") -> None:
        metadata = {"feature_protocol": FEATURE_PROTOCOL, "role": role, "fold_id": "fold0",
                    "dataset": "SyncG", "split": split, "heldout_verified": True, "reader_config": {"bins": 3},
                    "reader_seed": 20262020, "reader_train_sample_ids": ["reader_fit"],
                    "reader_training": {"training": {"seed": 20262020}, "loss_weights": {"appearance_ce": 1.0}},
                    "reader_train_scene_ids": ["reader_fit"], "heldout_sample_ids": [sample]}
        np.savez(path, features=np.zeros((1, 20), np.float32), expert_predictions=np.zeros((1, 2), np.float32),
                 field_valid=np.ones(1, bool), target=np.zeros(1, np.float32), sample_ids=np.asarray([sample]),
                 group_ids=np.asarray([sample]), scene_ids=np.asarray([scene]), metadata=np.asarray(json.dumps(metadata)))

    def test_fit_rejects_test_data_duplicate_ids_and_shared_validation_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train, val = Path(directory) / "train.npz", Path(directory) / "val.npz"
            self.cache(val, role="validation", scene="val", sample="val")
            self.cache(train, role="oof_train", scene="train", sample="train", split="test")
            with self.assertRaisesRegex(ValueError, "test caches are forbidden"):
                load_fit_data([train], val)
            self.cache(train, role="oof_train", scene="val", sample="train")
            with self.assertRaisesRegex(ValueError, "scene_ids"):
                load_fit_data([train], val)
            self.cache(train, role="oof_train", scene="train", sample="train")
            with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                load_fit_data([train, train], val)
            data, validation, _ = load_fit_data([train], val)
            self.assertEqual(len(data["target"]), len(validation["target"]))


if __name__ == "__main__":
    unittest.main()
