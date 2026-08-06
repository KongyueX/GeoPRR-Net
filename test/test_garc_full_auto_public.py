from __future__ import annotations

import hashlib
import unittest

import numpy as np

from experiments.automatic_numeric_range import NumericRangePrediction, image_sha256
from experiments.evaluate_garc_full_auto_public import (
    PublicTruth,
    detector_box_metrics,
    range_variant_sha256,
    score_full_auto_rows,
    score_range_rows,
    select_threshold,
    threshold_table,
)
from experiments.garc_full_auto_public import (
    EXPECTED_JOINT_OOF,
    EXPECTED_JOINT_OOF_BY_SEED,
    EXPECTED_JOINT_OOF_GROUPS_BY_SEED,
    EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY,
    EXPECTED_RANGE_INDEPENDENT,
    GARCFullAutoRangePipeline,
    _ConsensusModeDecoder,
    validate_prediction_row,
)
from experiments.garc_posterior_consensus import GARCConsensusConfig


class _Decoder:
    def __init__(self) -> None:
        self.config = GARCConsensusConfig()
        self.calls: list[str] = []

    def predict(self, tokens):
        self.calls.append("topk")
        return "topk-result"

    def top1_control(self, tokens):
        self.calls.append("top1")
        return "top1-result"


class _Identity:
    def __init__(self, name: str) -> None:
        self.identity = {"protocol": name, "checkpoint_sha256": hashlib.sha256(name.encode()).hexdigest()}


class _GARCProvider:
    def __init__(self) -> None:
        self.geometry_provider = _Identity("geometry")
        self.posterior_ocr_backend = _Identity("ocr")
        self.tick_proximity_provider = None
        self.input_size = 512
        self.decoder = _ConsensusModeDecoder(_Decoder(), "topk")

    def predict(self, image: np.ndarray) -> NumericRangePrediction:
        digest = image_sha256(image)
        return NumericRangePrediction(
            protocol="fake_garc",
            status=True,
            prediction_space="real_numeric_scale_start_end",
            pred_start=-10.0,
            pred_end=90.0,
            confidence=0.8,
            failure_reason=None,
            telemetry={
                "primary_adapter": {
                    "input_image_sha256": digest,
                    "geometry_and_ocr_same_source_image_sha256": digest,
                    "accepts_manual_geometry": False,
                    "accepts_physical_scale_values": False,
                }
            },
        )


def _truth() -> dict[str, PublicTruth]:
    return {
        "a": PublicTruth("g1", 0.0, 100.0, 40.0, ((0.1, 0.1, 0.2, 0.2),)),
        "b": PublicTruth("g2", 0.0, 100.0, 50.0, ((0.6, 0.6, 0.7, 0.7),)),
        "c": PublicTruth("g3", 0.0, 100.0, 60.0, ((0.3, 0.3, 0.4, 0.4),)),
    }


def _row(
    sample_id: str,
    *,
    range_status: bool = True,
    full_status: bool = True,
    start: float | None = 0.0,
    end: float | None = 100.0,
    reading: float | None = 40.0,
    progress: float | None = 0.4,
    confidence: float = 0.8,
    joint: bool = True,
    range_failure: str | None = None,
    full_failure: str | None = None,
    range_prediction=None,
):
    return {
        "sample_id": sample_id,
        "group_id": _truth()[sample_id].group_id,
        "range_status": range_status,
        "full_status": full_status,
        "predicted_scale_start": start,
        "predicted_scale_end": end,
        "predicted_reading": reading,
        "prediction_progress": progress,
        "range_confidence": confidence,
        "joint_oof_eligible": joint,
        "range_failure_reason": range_failure,
        "failure_reason": full_failure,
        "range_prediction": range_prediction,
    }


class DecoderAndBridgeTests(unittest.TestCase):
    def test_top1_and_topk_are_literal_decoder_ablations(self) -> None:
        for mode, expected in (("top1", "top1-result"), ("topk", "topk-result")):
            decoder = _Decoder()
            proxy = _ConsensusModeDecoder(decoder, mode)
            self.assertEqual(proxy.predict([1, 2, 3]), expected)
            self.assertEqual(decoder.calls, [mode])
            self.assertEqual(proxy.identity["ablation_mode"], mode)

    def test_garc_wrapper_adds_strict_same_image_attestation(self) -> None:
        provider = _GARCProvider()
        pipeline = GARCFullAutoRangePipeline(
            provider, recognizer_kind="tiny", consensus_mode="topk"
        )
        image = np.full((64, 72, 3), 127, dtype=np.uint8)
        result = pipeline.predict(image)
        primary = result.telemetry["primary_adapter"]
        self.assertEqual(
            primary["geometry_and_ocr_same_image_sha256"], image_sha256(image)
        )
        self.assertFalse(primary["accepts_reference_packet"])
        self.assertIs(pipeline.last_prediction, result)


class MetricTests(unittest.TestCase):
    def test_range_metrics_keep_failures_in_full_denominator(self) -> None:
        rows = [
            _row("a"),
            _row("b", start=10.0, end=90.0, reading=50.0, progress=0.5),
            _row(
                "c",
                range_status=False,
                full_status=False,
                start=None,
                end=None,
                reading=None,
                progress=None,
                range_failure="no_boxes",
                full_failure="automatic_numeric_range:no_boxes",
            ),
        ]
        metrics = score_range_rows(rows, _truth(), threshold=0.5)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["pair_rounded_exact_full_denominator"], 1 / 3)
        self.assertAlmostEqual(metrics["pair_rounded_exact_conditional"], 0.5)
        self.assertAlmostEqual(metrics["range_endpoint_mae_conditional"], 5.0)
        self.assertEqual(metrics["failure_breakdown"]["pipeline:no_boxes"], 1)

    def test_full_auto_nmae_uses_failure_penalty_one(self) -> None:
        rows = [
            _row("a", reading=50.0, progress=0.5),
            _row(
                "b",
                full_status=False,
                reading=None,
                progress=None,
                full_failure="progress:failed",
            ),
        ]
        metrics = score_full_auto_rows(rows, _truth(), threshold=0.5)
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertAlmostEqual(metrics["reading_nmae_conditional"], 0.1)
        self.assertAlmostEqual(
            metrics["reading_nmae_full_denominator_failure_penalty_1"], 0.55
        )
        self.assertAlmostEqual(
            metrics["reading_mae_full_denominator_failure_penalty_one_range"], 55.0
        )

    def test_threshold_selection_prefers_target_precision_then_coverage(self) -> None:
        rows = [
            _row("a", confidence=0.9),
            _row("b", start=10.0, end=90.0, confidence=0.4),
            _row("c", confidence=0.8, reading=60.0, progress=0.6),
        ]
        table = threshold_table(rows, _truth(), [0.0, 0.5, 0.85])
        selected, reason = select_threshold(
            table, minimum_accepted=1, target_precision=0.9
        )
        self.assertEqual(selected["threshold"], 0.5)
        self.assertEqual(reason, "target_precision_met_maximum_coverage")

    def test_detector_metrics_report_precision_recall_and_dbnet_gate(self) -> None:
        prediction = {
            "telemetry": {
                "ocr_input_shape": [100, 100, 3],
                "bridge": {
                    "box_trace": [
                        {"box": [[10, 10], [20, 10], [20, 20], [10, 20]]},
                        {"box": [[75, 75], [85, 75], [85, 85], [75, 85]]},
                    ]
                },
            }
        }
        metrics = detector_box_metrics(
            [_row("a", range_prediction=prediction)],
            {"a": _truth()["a"]},
        )
        self.assertAlmostEqual(metrics["iou_0_5"]["precision"], 0.5)
        self.assertAlmostEqual(metrics["iou_0_5"]["recall"], 1.0)
        self.assertFalse(metrics["dbnet_plus_plus_gate"]["triggered"])


class IntegrityBoundaryTests(unittest.TestCase):
    @staticmethod
    def _range_plan(*, checkpoint: str, verification: str, provider: str = "pepd"):
        digest = "a" * 64
        provider_identity = {
            "protocol": "progress-provider-v1",
            "provider": provider,
            "checkpoint_sha256": checkpoint,
            "verification_sha256": verification,
            "verification_protocol": "verified-group-oof-v1",
            "automatic_reference": {"detector_sha256": "b" * 64},
            "reference_detector_sha256": "b" * 64,
            "reference_is_internal": True,
            "amp_enabled": True,
            "native_input_size": 256,
        }
        return {
            "progress_component": {
                "binding": {
                    "provider_protocol": "progress-provider-v1",
                    "provider_identity": provider_identity,
                    "artifact_sha256": {
                        "checkpoint": checkpoint,
                        "verification": verification,
                        "reference_detector": "b" * 64,
                    },
                    "source_sha256": {
                        "factory": "c" * 64,
                        "authoritative_handoff": "d" * 64,
                    },
                }
            },
            "progress_factory": {"sha256": "e" * 64, "function": "build"},
            "reference": {"mode": "automatic", "detector_sha256": "b" * 64},
            "garc": {
                "recognizer_kind": "tiny",
                "consensus_mode": "topk",
                "geometry_mode": "v5",
                "geometry_provider": "enhanced_v5_oof_fold",
                "geometry_fold": {
                    "role": "joint_oof_fold_routed_pepd_and_enhanced_v5_head",
                    "oof_summary": {"sha256": "f" * 64},
                    "joint_assignment": {
                        "sample_assignment_sha256": "1" * 64,
                        "group_assignment_sha256": "2" * 64,
                    },
                },
                "input_size": 768,
                "detector_threshold": 0.4,
                "posterior_top_k": 5,
                "consensus_config": {"key": "value"},
                "artifacts": {
                    "detector": {"sha256": "3" * 64},
                    "recognizer": {"sha256": "4" * 64},
                    "geometry": {"sha256": "5" * 64},
                    "geometry_backbone": {"sha256": checkpoint},
                },
            },
            "code_bindings": {
                name: {"sha256": digest}
                for name in (
                    "garc_bridge",
                    "garc_consensus",
                    "garc_geometry_fusion",
                    "tiny_ocr",
                    "strong_ocr",
                    "geometry_provider",
                    "enhanced_v5_oof_geometry_provider",
                    "enhanced_v5_head",
                )
            },
        }

    def test_range_variant_is_fold_invariant_but_progress_family_bound(self) -> None:
        seed20 = self._range_plan(checkpoint="6" * 64, verification="7" * 64)
        seed21 = self._range_plan(checkpoint="8" * 64, verification="9" * 64)
        self.assertEqual(
            range_variant_sha256(seed20), range_variant_sha256(seed21)
        )
        vdn = self._range_plan(
            checkpoint="8" * 64,
            verification="9" * 64,
            provider="vdn_official200",
        )
        self.assertNotEqual(range_variant_sha256(seed20), range_variant_sha256(vdn))

    def test_paper_cohort_boundaries_are_hard_coded(self) -> None:
        self.assertEqual(EXPECTED_RANGE_INDEPENDENT, (1080, 50))
        self.assertEqual(EXPECTED_JOINT_OOF, (412, 19))
        self.assertEqual(
            EXPECTED_JOINT_OOF_BY_SEED,
            {20260720: 168, 20260721: 116, 20260722: 128},
        )
        self.assertEqual(
            EXPECTED_JOINT_OOF_GROUPS_BY_SEED,
            {20260720: 8, 20260721: 5, 20260722: 6},
        )
        self.assertEqual(
            EXPECTED_PRIMARY_FIXED_FOLD_SENSITIVITY,
            {
                "geometry_unseen_samples": 168,
                "geometry_unseen_groups": 8,
                "geometry_fit_overlap_samples": 912,
                "geometry_fit_overlap_groups": 42,
            },
        )

    def test_prediction_validator_rejects_embedded_truth(self) -> None:
        roster = {"sample_id": "a", "group_id": "g1"}
        row = {
            "schema_version": 1,
            "protocol": "garc_full_auto_public_predictions_v1",
            "partition": "independent_validation",
            "sample_id": "a",
            "group_id": "g1",
            "canonical_roi_sha256": "1" * 64,
            "full_status": False,
            "range_status": False,
            "prediction_progress": None,
            "predicted_scale_start": None,
            "predicted_scale_end": None,
            "predicted_reading": None,
            "range_confidence": 0.0,
            "failure_reason": "failed",
            "range_failure_reason": "failed",
            "progress_checkpoint_sha256": None,
            "geometry_head_checkpoint_sha256": "2" * 64,
            "geometry_backbone_checkpoint_sha256": None,
            "geometry_oof_seed": None,
            "progress_group_unseen": False,
            "geometry_group_unseen": False,
            "joint_oof_eligible": False,
            "method_record": {"status": False},
            "range_prediction": {"ground_truth": 12.0},
            "sample_seconds": 0.1,
        }
        with self.assertRaisesRegex(ValueError, "label-derived key"):
            validate_prediction_row(
                row, partition="independent_validation", roster_row=roster
            )


if __name__ == "__main__":
    unittest.main()
