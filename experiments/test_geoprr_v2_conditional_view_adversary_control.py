"""Necessary NEWCPU witnesses using synthetic tensors, metadata and isolated toy states."""
from copy import deepcopy
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_conditional_view_adversary_control as control
from experiments import run_geoprr_v2_conditional_view_adversary_control as runner
from experiments.geoprr_v2_semantic_direct import SemanticDirectReader


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_head = SimpleNamespace(out_channels=1280)
        self.projection = nn.Linear(3, 1280)

    def forward(self, value):
        return self.projection(value)


def toy_reader():
    return control.ConditionalViewReader(SemanticDirectReader(ToyEncoder()))


def synthetic_metadata():
    rows, scalars = [], {}
    for view in ("original", "roll"):
        for parent in range(687):
            alias = control.SEEN[0] if parent < 338 else control.SEEN[1]
            for slot in range(4):
                sample_id = f"{view}_{parent}_{slot}"
                layout = f"{view}_{parent}"
                saved = {"view": view, "pointer": alias, "layout": layout, "slot": slot, "state": f"state{slot}"}
                if view == "roll":
                    saved.update(parent_layout=f"original_{parent}", parent_sample_id=f"original_{parent}_{slot}", parent_equal=True)
                rows.append({"sample_id": sample_id, "group_id": layout, "metadata": {"needle_control": saved}})
                scalars[sample_id] = (.1, .25, .55, .9)[slot]
    return rows, scalars


def complete_epoch_checkpoint():
    contract = {"prepared": {"source_contract": {"synthetic": True}}, "bindings": [], "train_rows": []}
    totals = control.initial_totals()
    for key in ("train_batches", "train_CNN_calls", "train_point_calls", "train_critic_calls", "combined_backward_attempts",
                "CNN_backward_calls", "point_backward_calls", "critic_backward_calls", "optimizer_updates"):
        totals[key] = 86
    totals.update(train_images=5496, train_critic_examples=5496, training_seconds=1.,
                  critic_FP32_and_BCE_cuda_seconds=.1, hook_registration_removal_host_seconds=.01)
    history = [{"epoch": 1, "learning_rate": .0003, "train_images": 5496, "train_batches": 86,
        "last_batch_size": 56, "Z_counts": [2748, 2748], "critic_total": 5496, "critic_correct": 3600,
        "critic_online_accuracy": 3600 / 5496, "validation_calls": 0, "optimizer_updates": 86, "amp_skipped_updates": 0,
        "primitive_smooth_l1": .1, "critic_BCE": .7, "forward_numerical_sum": .8, "cost": {"seconds": 1.},
        "critic_FP32_and_BCE_cuda_seconds": .1, "hook_registration_removal_host_seconds": .01,
        **dict.fromkeys(("CNN_calls", "point_calls", "critic_calls", "combined_backward_attempts", "CNN_backward_calls",
                        "point_backward_calls", "critic_backward_calls"), 86)}]
    model = {f"reader.synthetic{i}": torch.zeros(1) for i in range(464)}
    model.update({key: torch.zeros(1) for key in ("critic.0.weight", "critic.0.bias", "critic.2.weight", "critic.2.bias")})
    metadata = dict(zip(("parameters", "parameter_tensors", "state_tensors", "critic_parameters", "critic_parameter_tensors",
                         "critic_state_tensors", "BN_counter_tensors"), (8517906, 237, 468, 82113, 4, 4, 77)))
    saved = {"protocol": control.PROTOCOL, "config": deepcopy(control.CONFIG), "arm": "detached_control", "alpha": 0.,
        "source_contract": contract["prepared"]["source_contract"], "source_metadata_contract": contract,
        "model": model, "model_metadata": metadata, "initialization": {"synthetic": True},
        "optimizer": {"param_groups": [{"params": [0], "lr": .0003, "weight_decay": .0001}]},
        "scaler": {"scale": 65536., "growth_factor": 2., "backoff_factor": .5, "growth_interval": 2000, "_growth_tracker": 0},
        "rng": dict.fromkeys(("python", "numpy", "torch", "cuda")), "history": history, "totals": totals,
        "epoch": 1, "phase": "training", "phases": {}, "checkpointed_module_seconds": 1.}
    return saved, contract


class GradientAndHookTests(unittest.TestCase):
    def test_GRL_identity_and_only_h_gradient_reversal_critic_descent_and_y_stop(self):
        critic = control.make_critic()
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.fill_(.01)
        source_h = torch.linspace(.01, .02, 2 * 1280).reshape(2, 1280)
        source_y, Z = torch.tensor([.2, .8]), torch.tensor([0., 1.])
        direct_h, direct_y = source_h.clone().requires_grad_(), source_y.clone().requires_grad_()
        direct_critic = deepcopy(critic)
        baseline = F.binary_cross_entropy_with_logits(direct_critic(torch.cat((direct_h, direct_y[:, None].detach()), 1)).squeeze(-1), Z)
        baseline.backward()
        self.assertIsNone(direct_y.grad)
        self.assertGreater(float(direct_h.grad.abs().sum()), 0.)
        for alpha in (0., .05):
            with self.subTest(alpha=alpha):
                h, y = source_h.clone().requires_grad_(), source_y.clone().requires_grad_()
                arm_critic = deepcopy(critic)
                reversed_h = control.reverse_gradient(h, alpha)
                self.assertTrue(torch.equal(h, reversed_h))
                loss = F.binary_cross_entropy_with_logits(arm_critic(torch.cat((reversed_h, y[:, None].detach()), 1)).squeeze(-1), Z)
                loss.backward()
                self.assertTrue(torch.allclose(h.grad, -alpha * direct_h.grad))
                self.assertIsNone(y.grad)
                for actual, expected in zip(arm_critic.parameters(), direct_critic.parameters(), strict=True):
                    self.assertTrue(torch.equal(actual.grad, expected.grad))
                self.assertFalse(any("alpha" in key for key in arm_critic.state_dict()))

    def test_fork_rng_preserves_main_CPU_stream_and_standard_critic_counts(self):
        before = torch.get_rng_state().clone()
        first = control.make_critic()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        second = control.make_critic()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(parameter.numel() for parameter in first.parameters()), 82113)
        self.assertEqual(len(first.state_dict()), 4)
        self.assertTrue(all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items()))
        self.assertFalse(any(isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for module in first.modules()))

    def test_original_scalar_forward_local_hook_F32_under_autocast_eval_zero_critic(self):
        model = toy_reader()
        image, y, Z = torch.ones(2, 3), torch.tensor([.2, .8], requires_grad=True), torch.tensor([0., 1.])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            expected = model.reader(image)
            result = control.training_forward(model, image, y, Z, .05)
        self.assertTrue(torch.equal(result["prediction"], expected))
        self.assertEqual(result["prediction"].dtype, torch.float32)
        self.assertEqual(result["critic_input_dtype"], "torch.float32")
        self.assertEqual(result["critic_input_shape"], [2, 1281])
        self.assertEqual(result["logits"].dtype, torch.float32)
        self.assertEqual(result["critic_BCE"].dtype, torch.float32)
        self.assertFalse(result["conditioning_requires_grad"])
        result["critic_BCE"].backward()
        self.assertIsNone(y.grad)
        self.assertEqual(len(model.point._forward_pre_hooks), 0)
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["image"])
        counts, hook = control.critic_counter(model)
        try:
            model.eval()
            with torch.inference_mode():
                model(image)
            self.assertEqual(counts, {"critic_calls": 0, "critic_examples": 0})
            with self.assertRaises(TypeError):
                model(image, y)
        finally:
            hook.remove()
        with patch.object(model.reader, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                control.training_forward(model, image, y, Z, .05)
        self.assertEqual(len(model.point._forward_pre_hooks), 0)

    def test_only_new_toy_state_serializes_and_strictly_reloads(self):
        model = toy_reader()
        with TemporaryDirectory(prefix="conditional_view_NEWCPU_toy_") as temporary:
            path = Path(temporary) / "new_toy_state.pt"
            torch.save(model.state_dict(), path)
            restored = toy_reader()
            result = restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.assertFalse(result.missing_keys or result.unexpected_keys)
            self.assertTrue(all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()))


class MetadataConsumerTests(unittest.TestCase):
    def test_saved_Z_parent_pair_and_production_target_consumer(self):
        rows, values = synthetic_metadata()
        # Mock consumed scalars, never a replacement GT formula or invented manifest y field.
        with patch("experiments.geoprr_v2_data.normalized_target", side_effect=lambda row: values[row["sample_id"]]) as reader:
            bindings, summary = control.validate_metadata_rows(rows, deepcopy(rows))
        self.assertEqual(reader.call_count, 5496)
        self.assertEqual(summary["Z_counts"], [2748, 2748])
        self.assertEqual(summary["paired_examples"], 2748)
        self.assertEqual(summary["view_quartets"], 1374)
        self.assertEqual(summary["max_paired_consumed_target_difference"], 0.)
        self.assertEqual(bindings[0]["y"], float(torch.tensor(.1, dtype=torch.float32)))
        self.assertTrue(all("y" not in row and "normalized_progress" not in row for row in rows))
        for failure in ("view", "binding", "target", "pair", "order"):
            changed, scalars = deepcopy(rows), dict(values)
            if failure == "view":
                changed[0]["metadata"]["needle_control"]["view"] = "unknown"
            elif failure == "binding":
                changed[2748]["metadata"]["needle_control"]["parent_sample_id"] = "original_0_1"
            elif failure == "target":
                scalars[changed[0]["sample_id"]] = float("nan")
            elif failure == "pair":
                scalars[changed[2748]["sample_id"]] += .001
            expected = deepcopy(changed)
            if failure == "order":
                expected = list(reversed(expected))
            with self.subTest(failure=failure), self.assertRaises(control.MetadataNotAdmitted):
                control.validate_metadata_rows(changed, expected, target_reader=lambda row: scalars[row["sample_id"]])


class ConditionalStateTests(unittest.TestCase):
    def test_new_conditional_counts_and_joint_thresholds(self):
        report = {"totals": control.initial_totals()}
        totals = report["totals"]
        for key in ("train_CNN_calls", "train_point_calls", "train_critic_calls", "combined_backward_attempts", "optimizer_updates"):
            totals[key] = 1290
        totals.update(train_images=82440, train_critic_examples=82440)
        branches = [([(report, None)], (82732, 1295, 82440, 1290)),
            ([(report, {})], (83024, 1300, 82440, 1290)),
            ([(report, {}), (report, None)], (165756, 2595, 164880, 2580)),
            ([(report, {}), (report, {})], (166048, 2600, 164880, 2580))]
        for arms, expected in branches:
            actual = control.combined_counts(arms)
            self.assertEqual(tuple(actual[key] for key in ("CNN_images", "CNN_calls", "critic_examples", "critic_calls")), expected)
        summary = {"nmae": .02, "tail10_nmae": .1, "maximum_error": .4, "count_at_5_percent_fs": 280}
        limits = {"nmae": .02, "tail10_nmae": .1, "maximum_error": .4, "over5": 12}
        pair = {"new_over5_on_protected": 0}
        self.assertFalse(control.view_gate(summary, limits, pair, strict_tail=True)["passed"])
        self.assertTrue(control.view_gate(summary, limits, pair, strict_tail=False)["passed"])
        pair["new_over5_on_protected"] = 1
        self.assertFalse(control.view_gate(summary, limits, pair, strict_tail=False)["passed"])
        self.assertTrue(control.critic_effect(.8, .5)["passed"])
        self.assertFalse(control.critic_effect(.65, .6)["passed"])
        # Critic effect is absent from the runner native routing predicate, so reading-pass still observes CF.
        with TemporaryDirectory(prefix="conditional_view_NEWCPU_stages_") as temporary:
            root = Path(temporary)
            self.assertEqual(runner.stage_action(root, root, "reversal_cf"), "run")

    def test_strict_new_resume_and_partial_observation_guards(self):
        saved, contract = complete_epoch_checkpoint()
        control.validate_resume(saved, contract, "detached_control", saved["optimizer"], saved["scaler"])
        for field in ("protocol", "config", "arm", "alpha", "source_metadata_contract", "states", "history", "totals"):
            changed = deepcopy(saved)
            if field == "states":
                changed["model"].pop("critic.2.bias")
            elif field == "history":
                changed["history"][0]["critic_total"] = 5495
            elif field == "totals":
                changed["totals"]["train_critic_examples"] -= 1
            elif field == "alpha":
                changed[field] = .05
            elif field == "config":
                changed[field]["training"]["epochs"] = 16
            elif field == "source_metadata_contract":
                changed[field]["bindings"] = ["changed"]
            else:
                changed[field] = "old_or_other_arm"
            with self.subTest(field=field), self.assertRaises(ValueError):
                control.validate_resume(changed, contract, "detached_control")
        with TemporaryDirectory(prefix="conditional_view_NEWCPU_partial_") as temporary:
            output = Path(temporary) / "detached_control"
            self.assertFalse(control.guard_arm_output(output))
            output.mkdir()
            with self.assertRaises(FileExistsError):
                control.guard_arm_output(output)
            (output / "latest.pt").write_text("synthetic marker only; never read as a weight")
            self.assertTrue(control.guard_arm_output(output))
            (output / "active_epoch.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                control.guard_arm_output(output)
            (output / "active_epoch.json").unlink()
            (output / "source_id_predictions.csv").write_text("synthetic partial CSV")
            with self.assertRaises(FileExistsError):
                runner.stage_action(Path(temporary), Path(temporary), "detached_control")
            with self.assertRaises(FileExistsError):
                control.load_evaluation(output, "source_id", contract, "detached_control", control.NATIVE_ORDER)
            (output / "source_id_active.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                control.guard_arm_output(output)


if __name__ == "__main__":
    unittest.main()
