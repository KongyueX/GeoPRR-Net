from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments import revision_full_training as full
from experiments import train_unified_pointer_reader as trainer


def write_manifest(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def source_row(identifier, split="train"):
    return {"dataset": "SyncG", "split": split, "sample_id": identifier}


def test_source_manifest_rejects_test_rows_duplicates_and_wrong_population(tmp_path):
    path = write_manifest(tmp_path / "source.jsonl", [source_row("a", "test")])
    with pytest.raises(ValueError, match="split=train"):
        full.read_source_rows(path, expected_samples=1)
    write_manifest(path, [source_row("a"), source_row("a")])
    with pytest.raises(ValueError, match="unique"):
        full.read_source_rows(path, expected_samples=2)
    with pytest.raises(ValueError, match="expected 16000"):
        full.read_source_rows(path, expected_samples=16_000)


def test_warm_selection_uses_official_targets_and_rejects_foreign_ids():
    official = [dict(source_row("a"), ground_truth=0.2), dict(source_row("b"), ground_truth=0.5)]
    selected = full.select_warm_rows(official, [dict(source_row("b"), ground_truth=99)])
    assert selected == (official[1],)
    with pytest.raises(ValueError, match="subset"):
        full.select_warm_rows(official, [source_row("test-only")])


def test_independent_warm_requires_matching_seed_and_full_foundation():
    warm = {"protocol": full.WARM_PROTOCOL, "seed": 7, "foundation": {"seed": 7, "train_samples": 16_000}}
    bank = {"seed": 7, "source_metadata": {"source_foundation": {"source_seed": 7}}}
    full.validate_seed_chain(warm, bank, 7)
    with pytest.raises(ValueError, match="independently fitted"):
        full.validate_seed_chain(dict(warm, seed=8), bank, 7)
    with pytest.raises(ValueError, match="foundation seed"):
        full.validate_seed_chain(warm, {"seed": 7, "source_metadata": {"source_foundation": {"source_seed": 8}}}, 7)
    with pytest.raises(ValueError, match="16000|16,000"):
        full.validate_seed_chain(dict(warm, foundation={"seed": 7, "train_samples": 14_442}), bank, 7)


def test_foundation_check_compares_tensors_not_just_seed():
    stages = [torch.nn.Sequential(torch.nn.Linear(2, 2)) for _ in range(3)]
    point = torch.nn.Linear(2, 1)
    bank = SimpleNamespace(base_model=SimpleNamespace(
        shared_scale8_encoder=stages[0], shared_scale16_encoder=stages[1],
        shared_context_encoder=stages[2], moment_readout=SimpleNamespace(point_projection=point)))
    state = {}
    for i, stage in enumerate(stages):
        state.update({f"backbone.features.{i}.{name}": value.clone() for name, value in stage[0].state_dict().items()})
    state.update({f"backbone.classifier.1.{name}": value.clone() for name, value in point.state_dict().items()})
    full._verify_bank_foundation(bank, state)
    state["backbone.features.0.weight"].add_(1)
    with pytest.raises(ValueError, match="different foundation tensors"):
        full._verify_bank_foundation(bank, state)


def suite_arguments(output):
    return dict(source_r2mt_checkpoint=Path("bank.pt"), warm_polar_checkpoint=Path("warm.pt"),
                fit_manifest_path=Path("legacy-fit.jsonl"), outer_split_path=Path("legacy-split.json"),
                output_root=output, device_name="cpu", seed=7, polar_epochs=10,
                router_stage_one_epochs=1, router_stage_two_epochs=1, polar_batch_size=64,
                router_batch_size=512, workers=0, ema_decay=.995, learning_rate=3e-4,
                smooth_max_temperature=2e-4, population_mode="official_full",
                official_train_manifest=Path("official-train.jsonl"), availability_policy="image_only")


def test_legacy_defaults_remain_and_full_policy_and_polar_budget_are_explicit(tmp_path):
    args = trainer.build_argument_parser().parse_args(["--source-r2mt", "bank.pt"])
    assert args.population_mode == "inner_scene" and args.polar_epochs == 5
    assert args.availability_policy is None and args.official_train_manifest is None
    with patch.object(trainer, "train_suite", return_value={"status": "complete", "seed": 7, "elapsed_seconds": 0.0}) as train:
        trainer.main(["--source-r2mt", "bank.pt"])
    assert train.call_args.kwargs["smooth_max_temperature"] == 2e-4
    arguments = suite_arguments(tmp_path / "out")
    with pytest.raises(ValueError, match="polar-epochs 10"):
        trainer.train_suite(**dict(arguments, polar_epochs=5))
    with pytest.raises(ValueError, match="availability-policy"):
        trainer.train_suite(**dict(arguments, availability_policy=None))
    with pytest.raises(ValueError, match="smooth-max-temperature"):
        trainer.train_suite(**dict(arguments, smooth_max_temperature=None))
    assert not (tmp_path / "out").exists()


def test_full_orchestration_uses_same_pool_and_writes_truthful_scopes(tmp_path):
    samples = tuple(SimpleNamespace(sample_id=str(i), scene_stem=str(i % 145)) for i in range(16_000))
    polar = torch.nn.Linear(1, 1)
    polar.radial_bins, polar.angular_bins = 8, 36
    seen = {}

    def train_polar(model, population, **kwargs):
        seen["polar_samples"] = population
        state = trainer._clone_state(model.polar_expert)
        return [], state, state, 1

    def collect(model, population, **kwargs):
        assert population is seen["polar_samples"]
        assert kwargs["scope"] == "source_train" and kwargs["availability_policy"] == "image_only"
        rows = len(population) * 6
        cache = {name: torch.zeros(rows) for name in trainer.TRAIN_FIELDS}
        cache["polar_posterior"] = torch.ones(rows, 1)
        cache["condition_index"] = torch.arange(6).repeat_interleave(len(population))
        torch.save({"scope": "source_train", "cache": cache, "rows": rows}, kwargs["cache_path"])
        return cache, 0.0

    def train_router(router, cache, **kwargs):
        kwargs["ema"].update(router)
        return []

    def route(router, arguments, active):
        weights = active.float()[None].expand(arguments[0].shape[0], -1)
        return weights / weights.sum(1, keepdim=True), None, None

    with (
        patch.object(full, "load_official_train_population", return_value=samples),
        patch.object(full, "load_independent_warm", return_value=(polar, {"role": "independent_same_seed_foundation"})),
        patch.object(trainer, "load_inner_scene_population", side_effect=AssertionError("legacy split must stay closed")),
        patch.object(trainer, "load_remst_r2mt_expert_bank", return_value=(torch.nn.Linear(1, 1), {"seed": 7})),
        patch.object(trainer, "_train_polar_with_ema", side_effect=train_polar),
        patch("experiments.revision_cache.collect_revision_cache", side_effect=collect),
        patch.object(trainer, "_collect_router_cache", side_effect=AssertionError("legacy cache must not be used")),
        patch.object(trainer, "_train_router_stage_one", side_effect=train_router),
        patch.object(trainer, "_train_router_stage_two", side_effect=train_router),
        patch.object(trainer, "_router_forward", side_effect=route),
        patch.object(trainer, "parameter_inventory", return_value={}),
    ):
        result = trainer.train_suite(**suite_arguments(tmp_path / "full"))
    scope = result["training_scope"]
    assert scope["source_unique_images"] == scope["polar_router_image_overlap"] == 16_000
    assert scope["router_train_rows"] == 96_000 and not scope["inner_scene_disjoint"]
    assert "inner_dev" not in json.dumps(result)
    cache = torch.load(tmp_path / "full/source_train_cache_ema_polar.pt", weights_only=False)
    assert cache["scope"] == "source_train" and cache["rows"] == 96_000
    checkpoint = torch.load(result["variants"][trainer.FULL]["ema_checkpoint"], weights_only=False)
    assert checkpoint["training"]["cache_scope"] == "source_train"
    assert "ema_source_train" in checkpoint["training"]
    assert trainer.NO_GEOMETRY_FIXED_ROUTING in result["variants"]


def test_official_samples_use_scene_groups(tmp_path):
    from experiments.resnet18_direct_progress import DirectSample

    sample = DirectSample("a", "type::scene", "scene", tmp_path / "a.png", (0, 0, 32, 32), .2)
    with patch("experiments.resnet18_direct_progress._direct_sample", return_value=sample):
        actual = full._samples([source_row("a")], tmp_path / "source.jsonl")
    assert actual[0].group_id == "scene"


def test_warm_training_saves_a_loadable_single_arm_without_evaluation(tmp_path):
    from experiments.raw_angular_moment_refiner_probe import AngularMomentLogitRefiner
    from experiments.syncg_lightweight_regression_baselines import LightweightProgressRegressor
    from experiments.train_cagh_scalemark_reference_probe_v5 import PhotoAugmentation

    foundation = LightweightProgressRegressor("efficientnet_b0", imagenet_pretrained=False)
    foundation_path = tmp_path / "foundation.pt"
    torch.save({"architecture": "efficientnet_b0", "seed": 7, "train_samples": 16_000,
                "epochs": 60, "model_state": foundation.state_dict()}, foundation_path)
    official = tuple(dict(source_row(str(i)), metadata={"scene_name": "scene"}) for i in range(16_000))
    sample = SimpleNamespace(sample_id="0", scene_stem="scene")

    class ToyDataset(torch.utils.data.Dataset):
        augmentation = PhotoAugmentation.disabled()

        def __init__(self, samples, **kwargs):
            assert samples == (sample,)

        def set_epoch(self, epoch):
            assert epoch == 60

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"image": torch.zeros(3, 32, 32), "progress": torch.tensor(.2)}

    with (
        patch.object(full, "EXPECTED_SCENES", 1),
        patch.object(full, "read_source_rows", side_effect=[official, (official[0],)]),
        patch.object(full, "_samples", return_value=(sample,)),
        patch("experiments.raw_angular_moment_refiner_probe.AngularProgressDataset", ToyDataset),
    ):
        result = full.train_warm_polar(
            foundation_checkpoint=foundation_path, official_train_manifest=tmp_path / "official.jsonl",
            warm_manifest=tmp_path / "warm.jsonl", expected_warm_images=1, output=tmp_path / "warm.pt",
            seed=7, epochs=1, batch_size=1, workers=0, learning_rate=3e-4, weight_decay=1e-4,
            device_name="cpu")
    payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    model = AngularMomentLogitRefiner()
    model.load_state_dict(payload["angular_no_aux_state"], strict=True)
    assert payload["foundation"]["seed"] == 7
    assert payload["training_scope"]["warm_sample_ids"] == ["0"]
    assert payload["history"][0]["samples"] == 1
    assert not payload["training_scope"]["official_syncg_test_access"]
    assert torch.count_nonzero(model.fusion[-1].bias).item() == 1


def test_official_population_and_scope_reject_144_scenes(tmp_path):
    samples = tuple(SimpleNamespace(scene_stem=str(i % 144)) for i in range(16_000))
    with (
        patch.object(full, "read_source_rows", return_value=()),
        patch.object(full, "_samples", return_value=samples),
    ):
        with pytest.raises(ValueError, match="144 scenes; expected 145"):
            full.load_official_train_population(tmp_path / "train.jsonl")
    with pytest.raises(ValueError, match="144 scenes; expected 145"):
        full.official_training_scope(samples, tmp_path / "train.jsonl", "image_only")


def test_warm_rejects_incomplete_source_scenes_before_loading_foundation(tmp_path):
    rows = tuple(dict(source_row(str(i)), metadata={"scene_name": f"scene_{i % 144}"})
                 for i in range(16_000))
    with patch.object(full, "read_source_rows", return_value=rows):
        with pytest.raises(ValueError, match="144 scenes; expected 145"):
            full.train_warm_polar(
                foundation_checkpoint=tmp_path / "unopened.pt", official_train_manifest=tmp_path / "train.jsonl",
                warm_manifest=tmp_path / "warm.jsonl", expected_warm_images=1,
                output=tmp_path / "warm.pt", seed=7, epochs=1, batch_size=1, workers=0,
                learning_rate=3e-4, weight_decay=1e-4, device_name="cpu")
    assert not (tmp_path / "warm.pt").exists()
