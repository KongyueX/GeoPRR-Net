"""CPU toy checks for fixed BN statistics, trainable parameters and exact resume."""
from copy import deepcopy
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiments import geoprr_v2_fixed_bn_semantic as fixed


class ToyEncoder(nn.Module):
    def __init__(self, *, bn_count=2):
        super().__init__()
        self.num_features = 4
        self.conv_stem = nn.Conv2d(3, 4, 1)
        self.norm_stem = nn.BatchNorm2d(4)
        self.activation = nn.Tanh()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(4, 6, 1)
        self.norm_head = nn.BatchNorm2d(6)
        self.extra_norms = nn.ModuleList(nn.BatchNorm2d(1) for _ in range(bn_count - 2))
        for index, module in enumerate(m for m in self.modules() if isinstance(m, nn.BatchNorm2d)):
            module.running_mean.fill_(.1 * (index + 1))
            module.running_var.fill_(1.5 + .1 * index)
            module.num_batches_tracked.fill_(13 + index)

    def forward(self, image):
        feature = self.global_pool(self.activation(self.norm_stem(self.conv_stem(image))))
        return self.norm_head(self.conv_head(feature)).flatten(1)


def reader(*, capture=True, bn_count=2):
    return fixed.FixedBNSemanticReader(ToyEncoder(bn_count=bn_count), capture_reference=capture)


def resume_fixture():
    model = reader(bn_count=77)
    validation = {name: {"samples": count, "nmae": .2, "tail10_nmae": .4}
                  for name, count in (("original", 1506), ("roll", 512))}
    score = fixed.selection_score(validation)
    contract = {"RGB_input_contract": deepcopy(fixed.RGB_INPUT_CONTRACT)}
    saved = {"protocol": fixed.PROTOCOL, "config": deepcopy(fixed.CONFIG), "data_contract": contract,
             "model": deepcopy(model.state_dict()), "fixed_bn_reference": model.fixed_bn_checkpoint(),
             "initialization": {"fixed_BN_reference_origin": "fresh_cached_ImageNet_encoder_before_source",
                                "fixed_BN_reference_tensors": 231},
             "optimizer": {}, "scheduler": {}, "scaler": {}, "rng": {}, "totals": {},
             "checkpointed_process_seconds": 0., "epoch": 1, "best_score": score, "best_epoch": 1,
             "history": [{"epoch": 1, "validation": validation, "selection_score": score}],
             "validation_rows": {"original": [{}] * 1506, "roll": [{}] * 512}}
    return saved, contract


class FixedBNSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(20262020)

    def test_train_override_keeps_only_BN_in_eval_and_all_parameters_trainable(self):
        model = reader()
        for mode in (True, False, True):
            self.assertIs(model.train(mode), model)
            self.assertEqual(model.training, mode)
            for module in model.encoder.modules():
                self.assertEqual(module.training, False if isinstance(module, nn.BatchNorm2d) else mode)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(model.point.in_features, 6)
        self.assertEqual(model.point.out_features, 1)
        self.assertTrue(bool((model.point.bias == 0).all()))

    def test_train_and_eval_forwards_keep_all_BN_statistics_and_parameters_exact(self):
        model = reader()
        original = deepcopy(model.state_dict())
        for training in (True, True, False):
            model.train(training)
            prediction = model(torch.randn(4, 3, 8, 8))
            self.assertEqual(prediction.shape, (4,))
            self.assertEqual(prediction.dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            self.assertEqual(model.assert_fixed_bn(), {"buffer_tensors_checked": 6,
                             "all_equal_original_cached_ImageNet": True})
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))

    def test_encoder_BN_affine_and_point_receive_finite_nonzero_gradients_and_update(self):
        model = reader().train()
        original = deepcopy(model.state_dict())
        prediction = model(torch.randn(4, 3, 8, 8))
        loss = F.smooth_l1_loss(prediction, torch.tensor([.02, .2, .3, .9]), beta=.05)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
            self.assertTrue(bool(parameter.grad.abs().sum() > 0), name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        optimizer.step()
        for name, parameter in model.named_parameters():
            self.assertFalse(torch.equal(parameter, original[name]), name)
        model.assert_fixed_bn()

    def test_missing_reference_on_uninitialized_restore_is_rejected(self):
        model = reader(capture=False)
        with self.assertRaisesRegex(ValueError, "reference is missing"):
            model.assert_fixed_bn()
        with self.assertRaisesRegex(ValueError, "reference is missing"):
            model.fixed_bn_checkpoint()

    def test_reference_moves_with_model_and_stays_outside_model_state(self):
        model = reader()
        keys = set(model.state_dict())
        model.to(dtype=torch.float64)
        model.assert_fixed_bn()
        self.assertEqual(set(model.state_dict()), keys)
        self.assertTrue(all("reference" not in key for key in keys))
        for key, value in model.fixed_bn_buffers().items():
            reference = model._fixed_bn_reference[key]
            self.assertEqual((reference.dtype, reference.device), (value.dtype, value.device))

    def test_resume_restores_original_reference_instead_of_new_encoder_statistics(self):
        model = reader()
        checkpoint = {"model": deepcopy(model.state_dict()), "fixed_bn_reference": model.fixed_bn_checkpoint()}
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        restored = reader(capture=False)
        restored.encoder.norm_head.running_mean.fill_(99.)
        restored.load_state_dict(saved["model"], strict=True)
        restored.restore_fixed_bn_reference(saved["fixed_bn_reference"])
        self.assertTrue(all(torch.equal(value, saved["fixed_bn_reference"][key])
                            for key, value in restored.fixed_bn_buffers().items()))
        restored.train()
        restored(torch.randn(2, 3, 8, 8))
        restored.assert_fixed_bn()
        missing = dict(saved["model"])
        del missing["point.weight"]
        with self.assertRaises(RuntimeError):
            restored.load_state_dict(missing, strict=True)

    def test_reference_and_model_mismatch_or_missing_buffer_is_rejected(self):
        model = reader()
        reference = model.fixed_bn_checkpoint()
        missing = dict(reference)
        del missing[next(iter(missing))]
        with self.assertRaisesRegex(ValueError, "reference is incomplete"):
            model.restore_fixed_bn_reference(missing)
        changed = deepcopy(reference)
        changed["encoder.norm_head.running_var"].add_(.1)
        with self.assertRaisesRegex(ValueError, "differs from its original"):
            model.restore_fixed_bn_reference(changed)
        model.encoder.norm_head.num_batches_tracked.add_(1)
        with self.assertRaisesRegex(ValueError, "buffer changed"):
            model.assert_fixed_bn()

    def test_resume_accepts_fixed_BN_state_and_rejects_V26_configuration(self):
        saved, contract = resume_fixture()
        fixed.validate_resume(saved, contract)
        for field, value in (("protocol", "geoprr_v2_high_resolution_direct_source_only_v1"),):
            wrong = deepcopy(saved)
            wrong[field] = value
            with self.assertRaisesRegex(ValueError, "resume protocol"):
                fixed.validate_resume(wrong, contract)
        wrong = deepcopy(saved)
        wrong["config"]["model"]["source_training_BN_updates"] = True
        with self.assertRaisesRegex(ValueError, "resume protocol"):
            fixed.validate_resume(wrong, contract)

    def test_resume_rejects_missing_reference_and_reference_or_saved_buffer_change(self):
        saved, contract = resume_fixture()
        wrong = deepcopy(saved)
        del wrong["fixed_bn_reference"]
        with self.assertRaisesRegex(ValueError, "resume requires"):
            fixed.validate_resume(wrong, contract)
        for section in ("model", "fixed_bn_reference"):
            wrong = deepcopy(saved)
            wrong[section]["encoder.norm_head.running_var"].add_(.1)
            with self.assertRaisesRegex(ValueError, "BN buffers differ"):
                fixed.validate_resume(wrong, contract)
        wrong = deepcopy(saved)
        wrong["initialization"]["fixed_BN_reference_origin"] = "fresh_random_model"
        with self.assertRaisesRegex(ValueError, "original cached ImageNet BN reference"):
            fixed.validate_resume(wrong, contract)


if __name__ == "__main__":
    unittest.main()
