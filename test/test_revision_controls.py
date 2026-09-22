"""CPU checks on constructed test tensors, not experimental gauge results."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.revision_controls import (
    CONDITIONS,
    ROUTER_ARGUMENT_NAMES,
    DualViewHead,
    LinearGainRouter,
    ScalarGainRouter,
    evaluate_control,
    load_cache,
    predict_batch,
    train_control,
)


SEED = 20262020


def make_payload(scope: str = "source_train") -> dict:
    """Two synthetic test images with all six condition rows each."""
    generator = torch.Generator().manual_seed(17)
    rows = 12
    base = torch.linspace(.1, .8, rows)
    return {
        "scope": scope, "candidate_lineage": "fixture/source-run",
        "candidate_variant": "full", "availability_policy": "image_only", "seed": SEED, "conditions": list(CONDITIONS),
        "dataset": "constructed_test_tensors", "availability_policy": "test_fixture",
        "sample_ids": [f"test-image-{i // 6}" for i in range(rows)],
        "group_ids": [f"test-group-{i // 6}" for i in range(rows)],
        "cache": {
            "raw_representation": torch.randn(rows, 1280, generator=generator),
            "sarn_representation": torch.randn(rows, 1280, generator=generator),
            "geometry_features": torch.randn(rows, 13, generator=generator),
            "base_mean": base,
            "polar_mean": base + .05,
            "r2mt_mean": base - .05,
            "raw_mean": base + .02,
            "sarn_mean": base - .02,
            "relation_available": torch.arange(rows) % 2 == 0,
            "polar_posterior": torch.full((rows, 36), 1 / 36),
            "polar_concentration": torch.full((rows,), .5),
            "polar_entropy": torch.full((rows,), .8),
            "target": base + .01,
            "condition_index": torch.arange(6).repeat(2),
            "full_prediction": base + .008,
        },
    }


class RevisionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_gate_inputs_and_zero_initialization_match_the_prior(self) -> None:
        cache = make_payload()["cache"]
        arguments = tuple(cache[name] for name in ROUTER_ARGUMENT_NAMES)
        for router, expected_parameters in [(LinearGainRouter(), 44434), (ScalarGainRouter(), 1368)]:
            with self.subTest(router=type(router).__name__):
                weights, gains, _ = router(*arguments)
                self.assertEqual(sum(p.numel() for p in router.parameters()), expected_parameters)
                self.assertTrue(torch.equal(gains, torch.zeros(12, 2)))
                self.assertTrue(torch.allclose(weights, torch.tensor([.5, .25, .25]).expand(12, 3)))
        linear = LinearGainRouter()
        self.assertEqual(linear.shared_projection[1].in_features, 1280)
        self.assertEqual(linear.regret_network[-1].in_features, 220)
        self.assertFalse(hasattr(ScalarGainRouter(), "shared_projection"))

    def test_scalar_gate_ignores_latent_geometry_and_posterior_inputs(self) -> None:
        router = ScalarGainRouter()
        with torch.no_grad():
            router.regret_network[-1].weight.fill_(.03)
        cache = make_payload()["cache"]
        changed = {name: value.clone() for name, value in cache.items()}
        for name in ("raw_representation", "sarn_representation", "geometry_features", "polar_posterior"):
            changed[name].fill_(float("nan"))
        first = router(*(cache[name] for name in ROUTER_ARGUMENT_NAMES))
        second = router(*(changed[name] for name in ROUTER_ARGUMENT_NAMES))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))

    def test_train_cache_rejects_evaluation_scope_seed_and_missing_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(make_payload("evaluation"), path)
            with self.assertRaisesRegex(ValueError, "scope"):
                load_cache(path, scope="source_train", seed=SEED)
            payload = make_payload()
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "seed"):
                load_cache(path, scope="source_train", seed=SEED + 1)
            payload["sample_ids"][0] = "incomplete-test-image"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "all six"):
                load_cache(path, scope="source_train", seed=SEED)

    def test_cache_detaches_features_and_static_controls_have_exact_means(self) -> None:
        payload = make_payload()
        payload["cache"]["raw_representation"].requires_grad_()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(payload, path)
            cache = load_cache(path, scope="source_train", seed=SEED)["cache"]
        self.assertFalse(cache["raw_representation"].requires_grad)
        candidates = torch.stack([cache[k] for k in ("base_mean", "polar_mean", "r2mt_mean")], dim=1)
        self.assertTrue(torch.allclose(predict_batch("uniform", cache), candidates.mean(1)))
        self.assertTrue(torch.allclose(predict_batch("fixed_prior", cache), candidates @ torch.tensor([.5, .25, .25])))
        self.assertTrue(torch.equal(predict_batch("base_only", cache), cache["base_mean"]))
        self.assertTrue(torch.equal(predict_batch("full", cache), cache["full_prediction"]))

    def test_failed_readings_remain_in_csv_and_metadata_retains_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            cache_path = directory / "evaluation.pt"
            payload = make_payload("evaluation")
            payload["cache"]["full_prediction"][2] = torch.nan
            payload["cache"]["full_prediction"][5] = torch.inf
            torch.save(payload, cache_path)
            result = evaluate_control(cache_path, directory / "full.csv", method="full", seed=SEED)
            with (directory / "full.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12)
            self.assertEqual(result["failures"], 2)
            for i in (2, 5):
                self.assertEqual(rows[i]["status"], "failure")
                self.assertEqual(rows[i]["prediction"], "")
                self.assertEqual(float(rows[i]["error"]), 1.0)
            metadata = json.loads(Path(result["metadata"]).read_text())
            self.assertEqual(metadata["dataset"], "constructed_test_tensors")
            self.assertEqual(metadata["availability_policy"], "test_fixture")
            self.assertEqual(rows[0]["sample_id"], payload["sample_ids"][0])
            self.assertEqual(rows[-1]["condition"], CONDITIONS[-1])
            payload["cache"]["polar_mean"][0] = torch.inf
            self.assertFalse(torch.isfinite(predict_batch("fixed_prior", payload["cache"])[0]))

    def test_gate_training_requires_the_actual_smooth_max_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "smooth_max_temperature"):
            train_control(Path("unused.pt"), Path("unused"), method="linear_gate", seed=SEED)

    def test_cpu_training_and_checkpoint_evaluation_use_no_test_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            train_cache = directory / "source.pt"
            torch.save(make_payload(), train_cache)
            eval_cache = directory / "evaluation.pt"
            torch.save(make_payload("evaluation"), eval_cache)
            for method in ("linear_gate", "scalar_gate", "dual_view"):
                with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                    outputs = train_control(train_cache, directory / method, method=method, seed=SEED,
                                            smooth_max_temperature=.01)
                    fitted = torch.load(outputs["terminal_checkpoint"], weights_only=False)
                    self.assertEqual(fitted["method"], method)
                    self.assertFalse(fitted["training_config"]["evaluation_used_for_selection"])
                    if method == "dual_view":
                        self.assertNotIn("ema_checkpoint", outputs)
                        self.assertEqual(len(fitted["history"]), 40)
                    else:
                        self.assertIn("ema_checkpoint", outputs)
                        self.assertEqual(fitted["training_config"]["ema_updates"], 160)
                    result = evaluate_control(eval_cache, directory / f"{method}.csv", method=method,
                                              seed=SEED, checkpoint=Path(outputs["terminal_checkpoint"]))
                    self.assertEqual(result["rows"], 12)
                    self.assertEqual(result["failures"], 0)
                    with self.assertRaisesRegex(ValueError, "scope"):
                        evaluate_control(train_cache, directory / "invalid.csv", method=method,
                                         seed=SEED, checkpoint=Path(outputs["terminal_checkpoint"]))
                    if method != "dual_view":
                        other = "scalar_gate" if method == "linear_gate" else "linear_gate"
                        with self.assertRaisesRegex(ValueError, "method or protocol"):
                            evaluate_control(eval_cache, directory / "mismatch.csv", method=other,
                                             seed=SEED, checkpoint=Path(outputs["terminal_checkpoint"]))
            self.assertEqual(sum(p.numel() for p in DualViewHead().parameters()), 824961)


if __name__ == "__main__":
    unittest.main()


def test_each_expert_only_readout_uses_its_own_candidate():
    cache=make_payload()['cache']
    for method,key in [('base_only','base_mean'),('polar_only','polar_mean'),('relational_only','r2mt_mean')]:
        assert torch.equal(predict_batch(method,cache),cache[key].float())


def test_control_rejects_same_seed_from_a_different_candidate_run(tmp_path):
    from experiments.revision_controls import PROTOCOL, candidate_lineage
    payload=make_payload('evaluation')
    checkpoint=tmp_path/'scalar.pt'
    torch.save({'protocol':PROTOCOL,'method':'scalar_gate','seed':SEED,
                'state_dict':ScalarGainRouter().state_dict(),
                'training_config':{'candidate_lineage':candidate_lineage(payload)}},checkpoint)
    payload['candidate_lineage']='fixture/another-source-run'
    cache_path=tmp_path/'evaluation.pt';torch.save(payload,cache_path)
    with unittest.TestCase().assertRaisesRegex(ValueError,'lineage or input policy differ'):
        evaluate_control(cache_path,tmp_path/'bad.csv',method='scalar_gate',seed=SEED,checkpoint=checkpoint)


def test_variant_prediction_keeps_its_checkpoint_identity(tmp_path):
    payload=make_payload('evaluation')
    payload['model_metadata']={'variant':'no_geometry_fusion'}
    payload['cache']['full_prediction']=payload['cache']['base_mean']
    cache_path=tmp_path/'variant.pt';torch.save(payload,cache_path)
    out=tmp_path/'variant.csv'
    evaluate_control(cache_path,out,method='full',seed=SEED)
    with out.open() as stream:
        rows=list(csv.DictReader(stream))
    assert {row['method'] for row in rows}=={'no_geometry_fusion'}
