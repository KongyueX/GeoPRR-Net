"""Small deterministic tests for the paper-facing algorithm components."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.datasets import (
    SYNCG_SAMPLE_IDS_SHA256,
    select_rpm10k_single_pointer_rows,
    syncg_sample_ids_sha256,
)
from experiments.make_frontend_transfer_table import (
    summarize_frontend_transfer,
)
from experiments.make_failure_table import summarize_cache
from experiments.plot_risk_coverage import plot_risk_coverage
from experiments.selective_experiment import (
    _evaluate,
    _fit,
    _method_metrics,
    _paired_comparison,
)
from experiments.train_syncg_segmentation import (
    SyncGPointerSegDataset,
    SyncGSegSample,
    _single_output_loss,
    build_letterbox_content_mask,
    grouped_train_val_split,
    retain_largest_components,
)
from experiments.verify_segmentation_run import (
    _sha256 as file_sha256,
    verify_segmentation_run,
)
from utils.angleDetect.pointerSeg.detectSeg import (
    load_u2net_state_dict,
    u2netpSeg,
)
from utils.angleDetect.residual_calibrator import SELECTIVE_FEATURE_COLUMNS
from utils.angleDetect.residual_calibrator import apply_calibrator
from utils.angleDetect.residual_calibrator import make_calibrator_row
from utils.angleDetect.zeroShotMeter import meterZeroShot


class _ConstantResidualModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value, dtype=np.float64)


class SelectiveGeometryTest(unittest.TestCase):
    def test_syncg_release_id_hashes_are_pinned(self):
        self.assertEqual(
            syncg_sample_ids_sha256(f"sync_{index}" for index in range(16_000)),
            SYNCG_SAMPLE_IDS_SHA256["train"],
        )
        self.assertEqual(
            syncg_sample_ids_sha256(
                f"sync_{index}" for index in range(16_000, 20_000)
            ),
            SYNCG_SAMPLE_IDS_SHA256["test"],
        )

    def test_segmentation_run_verifier_rejects_changed_released_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            initial_path = run_dir / "initial.pt"
            best_path = run_dir / "best.pt"
            released_path = run_dir / "released_calibrated.pt"
            initial_state = {"weight": torch.tensor([1.0, 2.0])}
            torch.save(initial_state, initial_path)
            project_root = Path(__file__).resolve().parents[1]
            config = {
                "signature": "run-signature",
                "protocol": "syncg_train_only_u2netp_finetune_v1",
                "epochs": 1,
                "batch_size": 2,
                "seed": 7,
                "limit": None,
                "syncg_reference_commit": (
                    "14204c3f5b35d160fafa39ad195cd5a63e6e9c12"
                ),
                "syncg_release_identity_verified": True,
                "syncg_sample_ids_sha256": SYNCG_SAMPLE_IDS_SHA256["train"],
                "initial_weights": str(initial_path),
                "initial_weights_sha256": file_sha256(initial_path),
                "source_sha256": {
                    "training": file_sha256(
                        project_root
                        / "experiments"
                        / "train_syncg_segmentation.py"
                    ),
                    "dataset_protocol": file_sha256(
                        project_root / "experiments" / "datasets.py"
                    ),
                    "u2netp": file_sha256(
                        project_root
                        / "utils"
                        / "angleDetect"
                        / "pointerSeg"
                        / "u2netp.py"
                    ),
                    "checkpoint_loader": file_sha256(
                        project_root
                        / "utils"
                        / "angleDetect"
                        / "pointerSeg"
                        / "detectSeg.py"
                    ),
                },
            }
            (run_dir / "run_config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            payload = {
                "state_dict": initial_state,
                "probability_threshold": 0.5,
                "training_protocol": {"signature": "run-signature"},
            }
            torch.save(payload, best_path)
            torch.save(payload, released_path)

            def write_summary():
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "run_signature": "run-signature",
                            "best_epoch": 1,
                            "best_checkpoint_sha256": file_sha256(best_path),
                            "released_calibrated_checkpoint_sha256": (
                                file_sha256(released_path)
                            ),
                        }
                    ),
                    encoding="utf-8",
                )

            write_summary()
            verified = verify_segmentation_run(
                run_dir,
                initial_weights=initial_path,
                require_formal_syncg=True,
                expected_epochs=1,
                expected_batch_size=2,
                expected_seed=7,
            )
            self.assertTrue(verified["released_weights_unchanged"])

            changed_payload = dict(payload)
            changed_payload["state_dict"] = {
                "weight": torch.tensor([1.0, 3.0])
            }
            torch.save(changed_payload, released_path)
            write_summary()
            with self.assertRaisesRegex(ValueError, "changed released model"):
                verify_segmentation_run(
                    run_dir,
                    initial_weights=initial_path,
                    require_formal_syncg=True,
                )

    def test_paired_comparison_penalizes_one_sided_failure(self):
        rows = [
            {
                "ground_truth": 0.0,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "a",
            },
            {
                "ground_truth": 0.0,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "b",
            },
        ]
        comparison = _paired_comparison(
            rows,
            [0.0, None],
            [0.1, 0.5],
            seed=7,
            bootstrap_iterations=0,
        )

        self.assertEqual(comparison["paired_samples"], 2)
        self.assertEqual(comparison["common_successes"], 1)
        self.assertAlmostEqual(comparison["delta_nmae"], 0.2)

    def test_failure_table_keeps_failed_rows_in_denominator(self):
        rows = [
            {
                "methods": {
                    "weighted_fusion": {"status": True, "prediction": 0.5},
                    "geometry_v1": {"status": True, "prediction": 0.5},
                    "geometry_v2": {"status": True, "prediction": 0.5},
                }
            },
            {
                "error_code": "pointer_not_found",
                "methods": {},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = summarize_cache("test", path)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["weighted_success"], 1)
        self.assertAlmostEqual(summary["weighted_coverage"], 0.5)
        self.assertEqual(summary["failures"]["pointer_not_found"], 1)

    def test_frontend_transfer_table_allows_only_segmentation_difference(self):
        rows = [
            {
                "dataset": "RPM-10K",
                "split": "test",
                "sample_id": "one",
                "group_id": "g1",
                "meter_id": "biao1",
                "ground_truth": 0.5,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "methods": {
                    "transformer": {"status": True, "prediction": 0.5},
                    "weighted_fusion": {"status": True, "prediction": 0.5},
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            released = root / "released.jsonl"
            finetuned = root / "finetuned.jsonl"

            def write_cache(path, segmentation_hash):
                path.write_text(
                    json.dumps(rows[0]) + "\n",
                    encoding="utf-8",
                )
                path.with_name(path.name + ".meta.json").write_text(
                    json.dumps(
                        {
                            "signature": {
                                "manifest_sha256": "same-manifest",
                                "manifest_protocol_sha256": "same-protocol",
                                "device": "cuda",
                                "weights_sha256": {
                                    "segmentation": segmentation_hash,
                                    "meter_detector": "same-detector",
                                    "meter_transformer": "same-transformer",
                                    "keypoint_detector": "same-keypoints",
                                },
                                "source_sha256": {"collector": "same"},
                                "correction_mode": "off",
                            }
                        }
                    ),
                    encoding="utf-8",
                )

            write_cache(released, "released-segmentation")
            write_cache(finetuned, "finetuned-segmentation")
            summary = summarize_frontend_transfer(
                released,
                finetuned,
                seed=7,
                bootstrap_iterations=0,
            )

        self.assertTrue(summary["only_segmentation_checkpoint_differs"])
        self.assertEqual(summary["samples"], 1)

    def test_risk_coverage_plot_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "risk.csv"
            csv_path.write_text(
                "threshold,correction_coverage,selected_nmae,"
                "system_nmae,negative_transfer_rate\n"
                "0.0,1.0,0.1,0.1,0.2\n"
                "0.5,0.5,0.05,0.08,0.1\n",
                encoding="utf-8",
            )
            output = root / "risk.png"

            plot_risk_coverage([("test", csv_path)], output)

            self.assertGreater(output.stat().st_size, 0)
            self.assertGreater(output.with_suffix(".pdf").stat().st_size, 0)

    def test_logit_segmentation_loss_is_fp16_gradient_safe(self):
        logits = torch.tensor(
            [[[[-1000.0, 0.0], [12.0, -12.0]]]],
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.tensor(
            [[[[1.0, 0.0], [1.0, 0.0]]]],
            dtype=torch.float16,
        )

        loss = _single_output_loss(
            logits,
            target,
            positive_weight=8.0,
            dice_weight=1.0,
            from_logits=True,
        )
        self.assertEqual(loss.dtype, torch.float32)
        (loss * 512.0).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_segmentation_loss_ignores_letterbox_padding(self):
        geometry = torch.tensor([[0, 1, 4, 2]], dtype=torch.int64)
        valid_mask = build_letterbox_content_mask(
            geometry,
            (4, 4),
            device=torch.device("cpu"),
        )
        target = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        target[:, :, 1, 1:3] = 1.0
        reference = torch.zeros(
            (1, 1, 4, 4),
            dtype=torch.float32,
            requires_grad=True,
        )
        perturbed = reference.detach().clone()
        perturbed[:, :, 0, :] = 1000.0
        perturbed[:, :, 3, :] = -1000.0
        perturbed.requires_grad_(True)

        reference_loss = _single_output_loss(
            reference,
            target,
            positive_weight=8.0,
            dice_weight=1.0,
            from_logits=True,
            valid_mask=valid_mask,
        )
        perturbed_loss = _single_output_loss(
            perturbed,
            target,
            positive_weight=8.0,
            dice_weight=1.0,
            from_logits=True,
            valid_mask=valid_mask,
        )
        perturbed_loss.backward()

        self.assertAlmostEqual(
            float(reference_loss.detach()),
            float(perturbed_loss.detach()),
            places=6,
        )
        self.assertEqual(float(perturbed.grad[:, :, 0, :].abs().sum()), 0.0)
        self.assertEqual(float(perturbed.grad[:, :, 3, :].abs().sum()), 0.0)

    def test_headline_metrics_penalize_failed_predictions(self):
        rows = [
            {
                "sample_id": "ok",
                "group_id": "group_a",
                "meter_id": "meter",
                "ground_truth": 50.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
            {
                "sample_id": "failed",
                "group_id": "group_b",
                "meter_id": "meter",
                "ground_truth": 50.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
        ]

        metrics = _method_metrics(
            rows,
            [50.0, None],
            seed=7,
            bootstrap_iterations=0,
        )

        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["successful_nmae"], 0.0)
        self.assertAlmostEqual(metrics["nmae"], 0.5)
        self.assertAlmostEqual(metrics["acc_1pct"], 0.5)
        self.assertAlmostEqual(metrics["range_nmae_capped_10"], 5.0)
        self.assertAlmostEqual(metrics["relative_mae_capped_100"], 50.0)
        self.assertAlmostEqual(metrics["acc_relative_5pct"], 0.5)
        self.assertAlmostEqual(metrics["dialbench_ref_successful"], 0.0)
        self.assertAlmostEqual(metrics["dialbench_rel_successful"], 0.0)
        self.assertAlmostEqual(metrics["dialbench_acc_epsilon_e2e"], 0.5)
        self.assertAlmostEqual(metrics["dialbench_acc_theta_e2e"], 0.5)

    def test_selective_mask_features_are_resolution_invariant(self):
        def feature_row(size, point_count, distance):
            v1 = {
                "progress_ratio": 0.4,
                "resultNum": 4.0,
                "confidence": 0.8,
                "tip_info": {
                    "point_count": point_count,
                    "candidate_count": point_count // 2,
                    "center_distance": distance,
                    "threshold": distance,
                    "axis_score": 0.9,
                    "support_ratio": 0.5,
                    "direction_consistency": 0.8,
                    "tip_support_ratio": 0.1,
                },
            }
            v2 = {
                "progress_ratio": 0.42,
                "resultNum": 4.2,
                "confidence": 0.7,
                "tip_info": {
                    "axis_score": 0.85,
                    "support_ratio": 0.3,
                    "vote_concentration": 0.75,
                    "side_separation": 0.6,
                },
            }
            fusion = {
                "progress_ratio": 0.41,
                "fusion_source_weights": {
                    "geometry_direct": 0.53,
                    "geometry_direct_v2": 0.47,
                },
            }
            return make_calibrator_row(
                v1,
                v2,
                fusion,
                {
                    "pointer_mask": np.zeros((size, size), dtype=np.uint8),
                    "startAngle": 45.0,
                    "endAngle": 315.0,
                    "disAngle": 270.0,
                },
                scale_start=0.0,
                scale_end=10.0,
            )

        small = feature_row(100, 100, 10.0)
        large = feature_row(200, 400, 20.0)

        for key in (
            "mask_component_area_ratio",
            "mask_candidate_ratio",
            "mask_center_distance_ratio",
            "mask_axis_threshold_ratio",
        ):
            self.assertAlmostEqual(small[key], large[key])

    def test_packaged_segmentation_checkpoint_exposes_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "seg.pt"
            torch.save(
                {
                    "format_version": "u2netp_pointer_seg_v1",
                    "state_dict": {"example": torch.tensor([1.0])},
                    "probability_threshold": 0.3,
                },
                checkpoint_path,
            )

            state_dict, metadata = load_u2net_state_dict(
                checkpoint_path,
                "cpu",
            )

        self.assertEqual(float(state_dict["example"][0]), 1.0)
        self.assertAlmostEqual(metadata["probability_threshold"], 0.3)

    def test_syncg_validation_split_has_no_scene_group_leakage(self):
        samples = [
            SyncGSegSample(
                sample_id=f"sample_{index}",
                group_id=f"type_{index % 2}::scene_{index % 5}",
                gauge_type=f"type_{index % 2}",
                scene_name=f"scene_{index % 5}",
                image_path="unused.jpg",
                mask_path="unused.png",
                annotation_path="unused.json",
                dial_bbox=(0.0, 0.0, 10.0, 10.0),
            )
            for index in range(40)
        ]

        train, validation = grouped_train_val_split(
            samples,
            val_fraction=0.2,
            seed=20260720,
        )

        train_groups = {sample.group_id for sample in train}
        validation_groups = {sample.group_id for sample in validation}
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertEqual(
            {sample.sample_id for sample in train + validation},
            {sample.sample_id for sample in samples},
        )

    def test_segmentation_mask_is_unletterboxed_before_resize(self):
        letterboxed = np.ones((256, 256), dtype=np.float32)
        # A 400x200 image occupies rows [64, 192) after 256px letterboxing.
        letterboxed[64:192, :] = 0.25

        restored = u2netpSeg._restore_letterbox_mask(
            letterboxed,
            (400, 200),
        )

        self.assertEqual(restored.shape, (200, 400))
        self.assertTrue(np.allclose(restored, 0.25))

    def test_syncg_dataset_reports_exact_letterbox_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (400, 200), (128, 64, 32)).save(image_path)
            mask = np.zeros((200, 400), dtype=np.uint8)
            mask[90:110, 40:360] = 255
            Image.fromarray(mask).save(mask_path)
            sample = SyncGSegSample(
                sample_id="wide",
                group_id="type::scene",
                gauge_type="type",
                scene_name="scene",
                image_path=image_path,
                mask_path=mask_path,
                annotation_path=root / "annotation.json",
                dial_bbox=(0.0, 0.0, 400.0, 200.0),
            )
            dataset = SyncGPointerSegDataset(
                [sample],
                train=False,
                crop_padding=0.0,
                input_size=256,
            )

            _image, target, content = dataset[0]

        self.assertEqual(content.tolist(), [0, 64, 256, 128])
        self.assertEqual(int(target[:, :64, :].sum()), 0)
        self.assertEqual(int(target[:, 192:, :].sum()), 0)
        self.assertGreater(int(target[:, 64:192, :].sum()), 0)

    def test_segmentation_probability_summary_is_runtime_only(self):
        probability = np.asarray(
            [[0.0, 0.2], [0.8, 1.0]],
            dtype=np.float32,
        )

        summary = u2netpSeg._summarize_probability(probability, 0.5)

        self.assertAlmostEqual(summary["probability_max"], 1.0)
        self.assertAlmostEqual(summary["probability_mean"], 0.5)
        self.assertAlmostEqual(summary["foreground_ratio"], 0.5)
        self.assertAlmostEqual(summary["threshold"], 0.5)

    def test_segmentation_postprocess_keeps_only_largest_component(self):
        masks = np.zeros((1, 12, 12), dtype=bool)
        masks[0, 1:3, 1:3] = True
        masks[0, 6:11, 5:10] = True

        retained = retain_largest_components(masks)

        self.assertEqual(int(retained.sum()), 25)
        self.assertFalse(retained[0, 1, 1])
        self.assertTrue(retained[0, 8, 7])

    def test_rpm10k_protocol_filters_only_from_label_schema(self):
        rows = [
            {
                "image": "a.jpg",
                "reading": "4.2",
                "range": 6,
                "meter_type": "biao2",
                "environment_conditions": "blur,tilted",
            },
            {
                "image": "b.jpg",
                "reading": "1.0",
                "range": 3,
                "meter_type": "biao5",
                "environment_conditions": "normal",
            },
            {
                "image": "c.jpg",
                "reading": "2.0",
                "range": 3,
                "meter_type": "others",
            },
            {
                "image": "d.jpg",
                "reading": "upper: 2; lower: 4",
                "range": 6,
                "meter_type": "biao6",
            },
            {
                "image": "e.jpg",
                "reading": "8.2",
                "range": 6,
                "meter_type": "biao1",
            },
        ]

        selected, audit = select_rpm10k_single_pointer_rows(rows)

        self.assertEqual([row["image"] for row in selected], ["a.jpg", "b.jpg"])
        self.assertTrue(audit["derived_subset"])
        self.assertFalse(audit["selection_uses_predictions"])
        self.assertEqual(
            audit["exclusions"],
            {
                "non_scalar_reading": 1,
                "outside_zero_based_range": 1,
                "unsupported_meter_type": 1,
            },
        )

    def test_quality_weighted_fusion_uses_confidence(self):
        v1 = {
            "status": True,
            "backend": "geometry_direct",
            "resultNum": 0.0,
            "progress_ratio": 0.0,
            "endNum_float": 0.0,
            "pointer_angle": 0.0,
            "pointer_relative_angle": 0.0,
            "tip_info": {"confidence": 0.9},
        }
        v2 = {
            "status": True,
            "backend": "geometry_direct_v2",
            "resultNum": 10.0,
            "progress_ratio": 1.0,
            "endNum_float": 100.0,
            "pointer_angle": 90.0,
            "pointer_relative_angle": 90.0,
            "tip_info": {"confidence": 0.1},
        }

        reading = meterZeroShot._build_geometry_weighted_fusion_reading(v1, v2)

        self.assertTrue(reading["status"])
        self.assertAlmostEqual(reading["resultNum"], 1.0)
        self.assertAlmostEqual(
            reading["fusion_source_weights"]["geometry_direct"],
            0.9,
        )

    def test_normalized_residual_scales_to_meter_range(self):
        package = {
            "model": _ConstantResidualModel(0.1),
            "feature_columns": ["p_fusion"],
            "residual_unit": "normalized_range",
            "residual_clip": 0.2,
            "backend_name": "geometry_fusion_weighted_calibrated",
        }
        feature_row = {
            "p_fusion": 0.5,
            "scaleStart": 10.0,
            "scaleEnd": 20.0,
        }
        base = {
            "status": True,
            "backend": "geometry_fusion_weighted",
            "resultNum": 15.0,
            "progress_ratio": 0.5,
        }

        reading = apply_calibrator(package, feature_row, base)

        self.assertTrue(reading["status"])
        self.assertEqual(
            reading["backend"],
            "geometry_fusion_weighted_calibrated",
        )
        self.assertAlmostEqual(reading["resultNum"], 16.0)
        self.assertAlmostEqual(reading["calibration"]["residual"], 1.0)

    def test_residual_fallback_does_not_reuse_previous_gate_state(self):
        package = {
            "model": _ConstantResidualModel(0.1),
            "feature_columns": ["p_fusion"],
            "residual_unit": "normalized_range",
            "max_abs_residual": 0.05,
            "_last_learned_gate": {
                "probability": 0.99,
                "threshold": 0.5,
            },
        }
        reading = apply_calibrator(
            package,
            {
                "p_fusion": 0.5,
                "scaleStart": 0.0,
                "scaleEnd": 10.0,
            },
            {
                "status": True,
                "backend": "geometry_fusion_weighted",
                "resultNum": 5.0,
                "progress_ratio": 0.5,
            },
        )

        self.assertFalse(reading["calibration"]["applied"])
        self.assertEqual(reading["calibration"]["reason"], "residual_abs_gate")
        self.assertIsNone(
            reading["calibration"]["learned_gate_probability"]
        )

    def test_grouped_fit_and_frozen_evaluation_smoke(self):
        def make_rows(split):
            rows = []
            for index in range(36):
                progress = 0.08 + 0.84 * index / 35
                ground_truth = 100.0 * progress
                structured_bias = 3.0 * np.sin(progress * np.pi * 2.0)
                weighted = ground_truth + structured_bias
                v1 = weighted + 1.2
                v2 = weighted - 0.8
                features = {
                    column: float((position + 1) * 0.001)
                    for position, column in enumerate(SELECTIVE_FEATURE_COLUMNS)
                }
                features.update(
                    {
                        "p_geom": v1 / 100.0,
                        "p_geom_v2": v2 / 100.0,
                        "p_fusion": weighted / 100.0,
                        "v1_v2_progress_delta": abs(v1 - v2) / 100.0,
                        "endNum": weighted,
                        "v1_confidence": 0.8,
                        "v2_confidence": 0.7,
                        "fusion_weight_v1": 0.8 / 1.5,
                        "fusion_weight_v2": 0.7 / 1.5,
                    }
                )
                rows.append(
                    {
                        "dataset": "SyncG",
                        "split": split,
                        "sample_id": f"{split}_{index:03d}",
                        "group_id": f"group_{index % 6}",
                        "meter_id": f"meter_{index % 3}",
                        "ground_truth": ground_truth,
                        "scale_start": 0.0,
                        "scale_end": 100.0,
                        "methods": {
                            "geometry_v1": {"status": True, "prediction": v1},
                            "geometry_v2": {"status": True, "prediction": v2},
                            "mean_fusion": {
                                "status": True,
                                "prediction": (v1 + v2) / 2.0,
                            },
                            "weighted_fusion": {
                                "status": True,
                                "prediction": weighted,
                            },
                        },
                        "features": features,
                    }
                )
            return rows

        def write_cache(path, rows, manifest_hash):
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            metadata = {
                "signature": {
                    "manifest_sha256": manifest_hash,
                    "device": "cuda",
                    "weights_sha256": {"front_end": "same"},
                    "source_sha256": {"collector": "same"},
                    "inference": {"correction_mode": "off"},
                }
            }
            path.with_name(path.name + ".meta.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = root / "train.jsonl"
            test_path = root / "test.jsonl"
            fit_dir = root / "fit"
            eval_dir = root / "eval"
            write_cache(train_path, make_rows("train"), "train-manifest")
            write_cache(test_path, make_rows("test"), "test-manifest")

            _fit(
                argparse.Namespace(
                    train_predictions=train_path,
                    output_dir=fit_dir,
                    feature_set="full",
                    folds=3,
                    trees=20,
                    gate_trees=20,
                    residual_min_samples_leaf=2,
                    gate_min_samples_leaf=2,
                    seed=7,
                    improvement_margin=0.0005,
                    min_correction_coverage=0.10,
                    std_quantile=0.95,
                    residual_clip_quantile=0.995,
                    max_residual_clip=0.15,
                    bootstrap_iterations=10,
                    allow_other_training_data=True,
                )
            )
            calibrator = fit_dir / "calibrator.joblib"
            self.assertTrue(calibrator.is_file())
            training_summary = json.loads(
                (fit_dir / "training_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                training_summary["gate_oof_protocol"],
                "nested_grouped_cross_fit",
            )
            self.assertEqual(
                training_summary["oof_residual_clip_protocol"],
                "fit_within_each_training_partition",
            )

            _evaluate(
                argparse.Namespace(
                    predictions=test_path,
                    calibrator=calibrator,
                    output_dir=eval_dir,
                    seed=7,
                    bootstrap_iterations=10,
                    allow_front_end_mismatch=False,
                )
            )
            metrics = json.loads(
                (eval_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["protocol"], "frozen_model_evaluation")
            self.assertTrue(metrics["front_end_signature_verified"])
            self.assertEqual(metrics["samples_used"], 36)
            self.assertIn(
                "acc_relative_5pct",
                metrics["metrics"]["Ours"],
            )


if __name__ == "__main__":
    unittest.main()
