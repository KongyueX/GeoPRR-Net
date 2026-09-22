"""C4 protocol checks use temporary images and a tiny local model."""
import csv
import json

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from experiments import revision_raw_baseline as raw
from experiments.resnet18_direct_progress import DirectSample
from experiments.train_pcrt_meter import PCRTConditionedMultiViewDataset


@pytest.fixture
def samples(tmp_path):
    image = np.arange(40 * 56 * 3, dtype=np.uint8).reshape(40, 56, 3)
    path = tmp_path / "source.png"
    assert cv2.imwrite(str(path), image)
    return tuple(DirectSample(f"image_{i}", f"group_{i}", f"scene_{i}", path,
                              (1, 2, 54, 39), target)
                 for i, target in enumerate((0.2, 0.8)))


def test_raw_pixels_match_existing_conditioned_dataset(samples):
    for condition in raw.CONDITIONS:
        expected = PCRTConditionedMultiViewDataset(samples, condition=condition)
        actual = raw.RawConditionDataset(samples, seed=raw.SEEDS[0], condition=condition)
        for index in range(len(samples)):
            image, target = actual[index]
            assert torch.equal(image, expected[index]["original_view"])
            assert torch.equal(target, expected[index]["target"])


def test_epoch_conditions_are_reproducible_across_worker_counts(samples):
    dataset = raw.RawConditionDataset(samples, seed=raw.SEEDS[0])
    dataset.set_epoch(7)
    selected = dataset.condition_indices.copy()
    expected = list(DataLoader(dataset, batch_size=1, num_workers=0))
    actual = list(DataLoader(dataset, batch_size=1, num_workers=2, multiprocessing_context="spawn"))
    for (image_a, target_a), (image_b, target_b) in zip(expected, actual):
        assert torch.equal(image_a, image_b)
        assert torch.equal(target_a, target_b)
    dataset.set_epoch(7)
    assert np.array_equal(selected, dataset.condition_indices)
    assert len(dataset) == len(samples)
    assert np.all((dataset.condition_indices >= 0) & (dataset.condition_indices < 6))


def test_fixed_learning_rate_schedule():
    assert raw.learning_rate(0) == pytest.approx(3e-4)
    assert raw.learning_rate(30) == pytest.approx(3e-5)
    assert raw.learning_rate(29) < raw.learning_rate(30)
    for start in (0, 30):
        values = [raw.learning_rate(epoch) for epoch in range(start, start + 30)]
        assert all(first > second for first, second in zip(values, values[1:]))
    with pytest.raises(ValueError):
        raw.learning_rate(60)


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, image):
        return torch.sigmoid(self.linear(image.mean(dim=(2, 3))).squeeze(1))


def test_terminal_training_budget_initialization_and_checkpoint(samples, tmp_path, monkeypatch):
    builds = []

    def build(architecture, *, imagenet_pretrained):
        assert architecture == "efficientnet_b0"
        model = TinyRegressor()
        builds.append((imagenet_pretrained, model.linear.weight.detach().clone()))
        return model

    def tiny_image(self, index):
        return torch.full((3, 8, 8), index + 0.25), torch.tensor(self.samples[index].normalized_target)

    monkeypatch.setattr(raw, "LightweightProgressRegressor", build)
    monkeypatch.setattr(raw.RawConditionDataset, "__getitem__", tiny_image)
    output = tmp_path / "terminal.pt"
    metadata = raw.train_checkpoint(samples, manifest=tmp_path / "train.jsonl", output=output,
                                    seed=raw.SEEDS[0], device=torch.device("cpu"), workers=0)
    assert builds[0][0] is True
    assert metadata["epochs"] == 60
    assert metadata["batch_size"] == 64
    assert metadata["checkpoint_selection"] == "terminal_epoch_60"
    assert len(metadata["history"]) == 60
    assert sum(row["train"]["samples"] for row in metadata["history"]) == 60 * len(samples)
    assert all(sum(row["condition_counts"].values()) == len(samples) for row in metadata["history"])
    assert metadata["history"][30]["learning_rate"] == pytest.approx(3e-5)
    assert metadata["augmentation"]["additional_random_augmentation"] is False
    assert metadata["training_scope"]["target_domain_access"] is False
    loaded, payload = raw.load_model(output, torch.device("cpu"))
    assert builds[-1][0] is False
    assert not torch.equal(loaded.linear.weight, builds[0][1])
    assert payload["seed"] == raw.SEEDS[0]
    assert json.loads(output.with_suffix(".json").read_text())["availability_policy"] == "raw_only"
    with pytest.raises(FileExistsError):
        raw.train_checkpoint(samples, manifest=tmp_path / "train.jsonl", output=output,
                             seed=raw.SEEDS[0], device=torch.device("cpu"), workers=0)


def test_evaluation_keeps_all_conditions_and_failed_predictions(samples, tmp_path):
    class WithFailure(torch.nn.Module):
        def forward(self, images):
            values = torch.full((len(images),), 0.5, device=images.device)
            values[0] = torch.nan
            return values

    output = tmp_path / "industrial.csv"
    metadata = raw.evaluate(WithFailure(), samples, output=output, seed=raw.SEEDS[1],
                            cohort="industrial", checkpoint=tmp_path / "terminal.pt",
                            manifest=tmp_path / "industrial.jsonl", device=torch.device("cpu"),
                            batch_size=2, workers=0)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 * len(samples)
    assert set(row["condition"] for row in rows) == set(raw.CONDITIONS)
    assert len({(row["sample_id"], row["condition"]) for row in rows}) == len(rows)
    failures = [row for row in rows if row["status"] == "failure"]
    assert len(failures) == 6
    assert all(float(row["error"]) == 1.0 and row["prediction"] == "" for row in failures)
    assert float(rows[0]["target"]) == float(torch.tensor(samples[0].normalized_target))
    assert metadata["dataset"] == "industrial"
    assert metadata["availability_policy"] == "raw_only"
    assert metadata["failures"] == 6


def test_train_cli_only_loads_official_source_pool(tmp_path, monkeypatch):
    calls = []

    def load(path, cohort):
        calls.append((path, cohort))
        return ()

    monkeypatch.setattr(raw, "load_revision_samples", load)
    monkeypatch.setattr(raw, "train_checkpoint", lambda *args, **kwargs: {})
    manifest = tmp_path / "train.jsonl"
    assert raw.main(["train", "--manifest", str(manifest), "--seed", str(raw.SEEDS[0]),
                     "--output", str(tmp_path / "model.pt"), "--device", "cpu"]) == 0
    assert calls == [(manifest, "syncg_train")]
