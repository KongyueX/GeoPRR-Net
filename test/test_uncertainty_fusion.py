import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from experiments.train_uncertainty_fusion import _fit_model, _predict

from experiments.uncertainty_fusion import (
    FEATURE_NAMES,
    DualVarianceMLP,
    extract_uncertainty_features,
    fit_robust_preprocessor,
    inverse_variance_weights,
    soft_fusion_prediction,
    transform_feature_matrix,
)


class UncertaintyFusionTest(unittest.TestCase):
    def test_small_variance_training_loop(self):
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(32, len(FEATURE_NAMES)))
        matrix[0, :3] = math.nan
        truth = rng.uniform(0.1, 0.9, size=32)
        base_progress = truth + rng.normal(0.0, 0.04, size=32)
        vector_progress = truth + rng.normal(0.0, 0.07, size=32)
        args = SimpleNamespace(
            device="cpu",
            batch_size=8,
            hidden_features=16,
            learning_rate=1e-3,
            weight_decay=1e-4,
            epochs=2,
            patience=2,
            fusion_loss_weight=2.0,
            temperature=1.0,
        )
        model, medians, scales, summary = _fit_model(
            train_matrix=matrix[:24],
            train_base_error=base_progress[:24] - truth[:24],
            train_vector_error=vector_progress[:24] - truth[:24],
            train_base_progress=base_progress[:24],
            train_vector_progress=vector_progress[:24],
            train_truth_progress=truth[:24],
            validation_matrix=matrix[24:],
            validation_base_progress=base_progress[24:],
            validation_vector_progress=vector_progress[24:],
            validation_truth_progress=truth[24:],
            args=args,
            seed=11,
            fixed_epochs=2,
        )
        prediction = _predict(
            model,
            matrix[24:],
            medians=medians,
            scales=scales,
            device=torch.device("cpu"),
        )
        self.assertEqual(prediction.shape, (8, 2))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(summary["epochs_ran"], 2)

    def test_feature_extraction_does_not_read_labels_or_identifiers(self):
        raw = {
            "sample_id": "first",
            "ground_truth": 1.0,
            "scale_start": 0.0,
            "scale_end": 10.0,
            "features": {"p_geom": 0.6},
            "methods": {},
            "status": True,
        }
        base = {
            "sample_id": "first",
            "ground_truth": 1.0,
            "scale_start": 0.0,
            "scale_end": 10.0,
            "predictions": {"ours": 4.0},
        }
        vector = {
            "sample_id": "first",
            "ground_truth": 1.0,
            "status": True,
            "prediction": 5.0,
            "progress": 0.5,
            "angle_std_degrees": 3.0,
            "angle_bin_entropy": 0.2,
            "angle_bin_resultant_length": 0.9,
            "pivot_input_xy": [127.5, 127.5],
            "pivot_peak": 0.8,
        }
        first = extract_uncertainty_features(
            raw_row=raw,
            base_row=base,
            vector_row=vector,
            reference_row={"meter_confidence": 0.9},
        )
        raw["sample_id"], raw["ground_truth"] = "second", 999.0
        base["sample_id"], base["ground_truth"] = "second", -999.0
        vector["sample_id"], vector["ground_truth"] = "second", 123.0
        second = extract_uncertainty_features(
            raw_row=raw,
            base_row=base,
            vector_row=vector,
            reference_row={"meter_confidence": 0.9},
        )
        for name in FEATURE_NAMES:
            if math.isnan(first[name]):
                self.assertTrue(math.isnan(second[name]))
            else:
                self.assertEqual(first[name], second[name])

    def test_robust_preprocessor_imputes_and_scales(self):
        matrix = np.tile(np.arange(len(FEATURE_NAMES), dtype=np.float64), (4, 1))
        matrix[0, 0] = math.nan
        matrix[:, 1] = 3.0
        medians, scales = fit_robust_preprocessor(matrix)
        transformed = transform_feature_matrix(
            matrix,
            medians=medians,
            scales=scales,
        )
        self.assertEqual(transformed.shape, matrix.shape)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(bool((scales > 0.0).all()))

    def test_inverse_variance_prefers_more_certain_expert(self):
        weights = inverse_variance_weights(torch.tensor([[-6.0, -2.0]]))
        self.assertGreater(float(weights[0, 0]), 0.98)
        self.assertLess(float(weights[0, 1]), 0.02)

    def test_soft_fusion_and_hard_failure_policy(self):
        prediction, route, mask_weight, _ = soft_fusion_prediction(
            base_prediction=2.0,
            vector_prediction=8.0,
            scale_start=0.0,
            scale_end=10.0,
            mask_log_variance=-6.0,
            vector_log_variance=-2.0,
        )
        self.assertEqual(route, "uncertainty_soft_fusion")
        self.assertGreater(mask_weight, 0.98)
        self.assertLess(prediction, 2.2)
        self.assertEqual(
            soft_fusion_prediction(
                base_prediction=None,
                vector_prediction=7.0,
                scale_start=0.0,
                scale_end=10.0,
                mask_log_variance=None,
                vector_log_variance=-2.0,
            )[:3],
            (7.0, "vector_hard_fallback", 0.0),
        )

    def test_variance_model_output_is_bounded(self):
        model = DualVarianceMLP(hidden_features=16)
        output = model(torch.zeros(5, len(FEATURE_NAMES)))
        self.assertEqual(tuple(output.shape), (5, 2))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output >= -12.0).all()))
        self.assertTrue(bool((output <= 2.0).all()))


if __name__ == "__main__":
    unittest.main()
