"""Focused CPU checks for generic multiscale token reading and source risk selection."""
from copy import deepcopy
import inspect
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments.geoprr_v2_semantic_tokens import (
    CONFIG, PROTOCOL, SemanticTokenReader, SemanticTokenReadout, gradient_evidence,
    position_encoding, selection_score, validate_external_checkpoint, validate_resume,
)
from experiments.geoprr_v2_semantic_direct import scalar_rows
from experiments.evaluate_geoprr_v2_role_external import summarize


class ToyEncoder(nn.Module):
    """Expose the declared native shapes with a small RGB encoder and semantic neck."""

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.conv_stem = nn.Conv2d(3, 3, 1)
        self.blocks = nn.ModuleList((nn.Conv2d(3, 48, 4, stride=4), nn.Conv2d(48, 80, 1, stride=2),
                                     nn.Conv2d(80, 160, 1, stride=2), nn.GELU(),
                                     nn.Conv2d(160, 960, 1, stride=2)))
        self.conv_head = nn.Conv2d(960, 1280, 1)
        self.norm_head = nn.BatchNorm2d(1280)
        self.num_features = 960

    def forward(self, image):
        self.calls += 1
        feature = self.conv_stem(image)
        for block in self.blocks:
            feature = block(feature)
        return self.norm_head(self.conv_head(feature)).mean((2, 3))


def feature_batch(batch):
    return [torch.randn(batch, *shape) for shape in ((48, 64, 64), (80, 32, 32), (160, 16, 16), (960, 8, 8))]


def source_summary(mean_original=.1, mean_roll=.2, tail_original=.5, tail_roll=.6):
    return {"original": {"samples": 1506, "nmae": mean_original, "tail10_nmae": tail_original},
            "roll": {"samples": 512, "nmae": mean_roll, "tail10_nmae": tail_roll}}


def resume_state():
    populations = [source_summary(.1, .2, .5, .6), source_summary(.1, .2, .3, .4), source_summary(.1, .2, .3, .4)]
    scores = [selection_score(population) for population in populations]
    return {"protocol": PROTOCOL, "config": deepcopy(CONFIG), "data_contract": {"source": "fixed"},
            "model": {}, "optimizer": {}, "scheduler": {}, "scaler": {}, "rng": {}, "totals": {},
            "initialization": {}, "checkpointed_process_seconds": 1., "epoch": 3,
            "best_score": scores[1], "best_epoch": 2,
            "history": [{"epoch": i+1, "validation": population, "selection_score": scores[i]}
                        for i, population in enumerate(populations)],
            "validation_rows": {"original": [{}]*1506, "roll": [{}]*512}}


class SemanticTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_fine_projection_receives_ordered_pixel_unshuffle_corners(self):
        readout = SemanticTokenReadout()
        features = [torch.zeros(1, *shape) for shape in ((48, 64, 64), (80, 32, 32), (160, 16, 16), (960, 8, 8))]
        features[0][0, 0, :2, :2] = torch.tensor([[1., 2.], [3., 4.]])
        features[0][0, 1, :2, :2] = torch.tensor([[5., 6.], [7., 8.]])
        captured = []
        hook = readout.projections[0].register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
        try:
            with torch.inference_mode():
                memory = readout.spatial_memory(features)
        finally:
            hook.remove()
        self.assertEqual(tuple(captured[0].shape), (1, 192, 32, 32))
        torch.testing.assert_close(captured[0][0, :8, 0, 0], torch.arange(1., 9.))
        self.assertEqual(tuple(memory.shape), (1, 2368, 128))

    def test_regular_positions_and_batch_partial_batch_independence(self):
        position = position_encoding(3, 4).reshape(3, 4, 128)
        torch.testing.assert_close(position[0, 0, :64], position[1, 0, :64])
        torch.testing.assert_close(position[0, 0, 64:], position[0, 1, 64:])
        self.assertFalse(torch.equal(position[0, 0], position[0, 1]))
        self.assertFalse(torch.equal(position[0, 0], position[1, 0]))
        readout = SemanticTokenReadout().eval()
        features, semantic = feature_batch(2), torch.randn(2, 1280)
        shapes = []
        hook = readout.decoders[0].register_forward_pre_hook(
            lambda _module, inputs: shapes.append((tuple(inputs[0].shape), tuple(inputs[1].shape))))
        try:
            with torch.inference_mode():
                full = readout(features, semantic)
                partial = readout([feature[1:] for feature in features], semantic[1:])
        finally:
            hook.remove()
        self.assertEqual(shapes, [((2, 4, 128), (2, 2368, 128)), ((1, 4, 128), (1, 2368, 128))])
        torch.testing.assert_close(full[1:], partial, rtol=1e-5, atol=1e-7)
        self.assertFalse(torch.equal(readout.decoders[0].self_attn.in_proj_weight,
                                     readout.decoders[1].self_attn.in_proj_weight))

    def test_output_uses_spatial_content_position_and_semantic_context(self):
        readout = SemanticTokenReadout().eval()
        features, semantic = feature_batch(1), torch.randn(1, 1280)
        rearranged_features = [F.pixel_shuffle(F.pixel_unshuffle(features[0], 2).flip(-1), 2),
                               *(feature.flip(-1) for feature in features[1:])]
        with torch.inference_mode():
            reference = readout(features, semantic)
            rearranged = readout(rearranged_features, semantic)
            changed_context = readout(features, semantic + torch.randn_like(semantic))
            changed_fine = readout([features[0] + torch.randn_like(features[0]), *features[1:]], semantic)
        self.assertGreater(float((reference-rearranged).abs().max()), 1e-8)
        self.assertGreater(float((reference-changed_context).abs().max()), 1e-8)
        self.assertGreater(float((reference-changed_fine).abs().max()), 1e-8)
        for level in range(4):
            getattr(readout, f"position_{level}").zero_()
        with torch.inference_mode():
            no_position = readout(features, semantic)
            no_position_rearranged = readout(rearranged_features, semantic)
        torch.testing.assert_close(no_position, no_position_rearranged, rtol=1e-5, atol=1e-7)

    def test_full_rgb_path_gradients_cover_semantic_neck_fine_path_and_decoders(self):
        model = SemanticTokenReader(ToyEncoder()).train()
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        prediction = model(torch.randn(2, 3, 256, 256))
        F.smooth_l1_loss(prediction, torch.tensor([0., 1.]), beta=.05).backward()
        finite, nonzero = gradient_evidence(model)
        self.assertTrue(finite)
        self.assertTrue(all(nonzero.values()), nonzero)
        self.assertEqual(model.encoder.calls, 1)
        self.assertEqual(model.last_feature_shapes, [[2, 48, 64, 64], [2, 80, 32, 32], [2, 160, 16, 16], [2, 960, 8, 8]])
        self.assertTrue(all(torch.equal(parameter, original[name]) for name, parameter in model.named_parameters()))
        self.assertTrue(bool(torch.isfinite(prediction).all()))
        self.assertTrue(bool(((prediction >= 0) & (prediction <= 1)).all()))

    def test_rgb_only_interface_rejects_gt_before_encoder_and_state_reload_is_strict(self):
        model = SemanticTokenReader(ToyEncoder()).eval()
        self.assertEqual(list(inspect.signature(model.forward).parameters), ["image"])
        image = torch.randn(1, 3, 256, 256)
        with self.assertRaises(TypeError):
            model(image, ground_truth=torch.tensor([.5]))
        self.assertEqual(model.encoder.calls, 0)
        state = deepcopy(model.state_dict())
        self.assertFalse(any("point" in key or "parent" in key for key in state))
        restored = SemanticTokenReader(ToyEncoder()).eval()
        restored.load_state_dict(state, strict=True)
        with torch.inference_mode():
            torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)
        missing = dict(state); del missing["readout.semantic_context.weight"]
        with self.assertRaises(RuntimeError): restored.load_state_dict(missing, strict=True)
        old_scalar = dict(state); old_scalar["point.weight"] = torch.zeros(1, 1280)
        with self.assertRaises(RuntimeError): restored.load_state_dict(old_scalar, strict=True)

    def test_source_risk_uses_actual_mean_and_ceil_worst10_with_failures(self):
        def rows(count, high_count, high_error):
            batch = {"sample_id": [str(i) for i in range(count)], "group_id": ["source"]*count,
                     "y": torch.zeros(count), "input_failure": [""]*count}
            prediction = torch.zeros(count)
            prediction[:high_count] = high_error
            return scalar_rows(prediction, batch)
        original = rows(1506, math.ceil(.1*1506), .8)
        roll = rows(512, math.ceil(.1*512), .4)
        original[0].update(valid=False, input_valid=False, absolute_error=1., failure="missing_RGB")
        populations = {"original": summarize(original), "roll": summarize(roll)}
        self.assertAlmostEqual(populations["original"]["tail10_nmae"], (1+150*.8)/151, places=6)
        self.assertAlmostEqual(populations["roll"]["tail10_nmae"], .4, places=6)
        expected = (sum(row["absolute_error"] for row in original)/1506
                    + sum(row["absolute_error"] for row in roll)/512
                    + (1+150*float(torch.tensor(.8)))/151 + float(torch.tensor(.4))) / 4
        self.assertAlmostEqual(selection_score(populations), expected, places=12)
        self.assertAlmostEqual(selection_score(source_summary()), (.1+.2+.5+.6)/4, places=15)
        with self.assertRaises(ValueError):
            selection_score({**populations, "industrial": {"samples": 1395, "nmae": 0., "tail10_nmae": 0.}})
        partial = deepcopy(populations); partial["roll"]["samples"] = 511
        with self.assertRaises(ValueError): selection_score(partial)
        invalid = deepcopy(populations); invalid["original"]["tail10_nmae"] = float("nan")
        with self.assertRaises(ValueError): selection_score(invalid)

    def test_new_risk_resume_preserves_first_tie_and_external_requires_completed_best(self):
        state = resume_state()
        validate_resume(state, {"source": "fixed"})
        tied = deepcopy(state); tied["best_epoch"] = 3
        with self.assertRaises(ValueError): validate_resume(tied, {"source": "fixed"})
        mean_only = deepcopy(state); mean_only["best_score"] = .15
        with self.assertRaises(ValueError): validate_resume(mean_only, {"source": "fixed"})
        missing = deepcopy(state); del missing["scheduler"]
        with self.assertRaises(ValueError): validate_resume(missing, {"source": "fixed"})
        checkpoint = {"protocol": PROTOCOL, "config": CONFIG, "epoch": 20, "best_epoch": 20, "best_score": .3}
        completed = {"protocol": PROTOCOL, "config": CONFIG, "status": "complete", "epochs": 60,
                     "best_epoch": 20, "best_score": .3, "completion_reason": "source_score_patience"}
        validate_external_checkpoint(checkpoint, completed)
        with self.assertRaises(ValueError): validate_external_checkpoint({**checkpoint, "epoch": 60}, completed)
        with self.assertRaises(ValueError): validate_external_checkpoint(checkpoint, {**completed, "epochs": 59})
        with self.assertRaises(ValueError): validate_external_checkpoint({**checkpoint, "protocol": "old_v22"}, completed)


if __name__ == "__main__":
    unittest.main()
