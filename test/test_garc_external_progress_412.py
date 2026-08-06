from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from experiments.garc_external_progress_412 import (
    HANDOFF_PROTOCOL,
    PREDICTION_PROTOCOL,
    TRANSFORMER_METHOD,
    VDN_METHOD,
    _normalize_provider_record,
    load_protocol,
    paired_group_bootstrap,
    score_method,
    validate_external_prediction_row,
    validate_handoff_row,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _roster() -> dict:
    return {
        "sample_id": "sync_1",
        "group_id": "pressure::scene",
        "image_relpath": "sync_1.jpg",
        "dial_bbox": [1.0, 2.0, 30.0, 40.0],
    }


def _route() -> dict:
    return {
        "sample_id": "sync_1",
        "group_id": "pressure::scene",
        "canonical_roi_sha256": _SHA_A,
        "pepd_seed": 20260720,
        "pepd_checkpoint_sha256": _SHA_B,
    }


def _handoff() -> dict:
    return {
        "schema_version": 1,
        "protocol": HANDOFF_PROTOCOL,
        "sample_id": "sync_1",
        "group_id": "pressure::scene",
        "image_relpath": "sync_1.jpg",
        "dial_bbox": [1.0, 2.0, 30.0, 40.0],
        "canonical_roi_sha256": _SHA_A,
        "oof_seed": 20260720,
        "vdn_checkpoint_sha256": _SHA_C,
        "range_status": True,
        "predicted_scale_start": 0.0,
        "predicted_scale_end": 100.0,
        "range_confidence": 0.9,
        "range_accepted": True,
        "garc_full_status": True,
        "garc_prediction_progress": 0.5,
        "garc_predicted_reading": 50.0,
        "garc_failure_reason": None,
        "garc_progress_checkpoint_sha256": _SHA_B,
        "garc_geometry_head_checkpoint_sha256": _SHA_D,
        "garc_geometry_backbone_checkpoint_sha256": _SHA_B,
        "garc_all_components_group_unseen": True,
    }


def _external(method: str = VDN_METHOD) -> dict:
    return {
        "schema_version": 1,
        "protocol": PREDICTION_PROTOCOL,
        "method": method,
        "sample_id": "sync_1",
        "group_id": "pressure::scene",
        "canonical_roi_sha256": _SHA_A,
        "oof_seed": 20260720,
        "status": True,
        "prediction_progress": 0.5,
        "failure_reason": None,
        "checkpoint_sha256": _SHA_C if method == VDN_METHOD else _SHA_E,
        "provider_identity_sha256": _SHA_D,
        "provider_record_sha256": _SHA_E,
        "strict_oof_eligible": method == VDN_METHOD,
        "sample_seconds": 0.1,
    }


class ProtocolTests(unittest.TestCase):
    def test_frozen_roles_do_not_promote_transformer(self) -> None:
        _, protocol = load_protocol()
        self.assertEqual(
            protocol["scoring"]["strict_formal_methods"],
            ["garc", VDN_METHOD],
        )
        self.assertFalse(protocol[TRANSFORMER_METHOD]["strict_412_table_eligible"])
        self.assertFalse(
            protocol[TRANSFORMER_METHOD]["historical_oof"][
                "formal_same_input_reuse_allowed"
            ]
        )

    def test_handoff_row_requires_all_component_unseen(self) -> None:
        row = _handoff()
        validate_handoff_row(
            row,
            roster_row=_roster(),
            route=_route(),
            vdn_checkpoint_sha256=_SHA_C,
        )
        row["garc_all_components_group_unseen"] = False
        with self.assertRaisesRegex(ValueError, "non-jointly-unseen"):
            validate_handoff_row(
                row,
                roster_row=_roster(),
                route=_route(),
                vdn_checkpoint_sha256=_SHA_C,
            )

    def test_vdn_prediction_is_strict_but_transformer_is_not(self) -> None:
        validate_external_prediction_row(
            _external(VDN_METHOD),
            handoff_row=_handoff(),
            transformer_checkpoint_sha256=_SHA_E,
        )
        transformer = _external(TRANSFORMER_METHOD)
        validate_external_prediction_row(
            transformer,
            handoff_row=_handoff(),
            transformer_checkpoint_sha256=_SHA_E,
        )
        transformer["strict_oof_eligible"] = True
        with self.assertRaisesRegex(ValueError, "claim role"):
            validate_external_prediction_row(
                transformer,
                handoff_row=_handoff(),
                transformer_checkpoint_sha256=_SHA_E,
            )

    def test_successful_progress_must_be_in_unit_interval(self) -> None:
        row = _external()
        row["prediction_progress"] = 1.1
        with self.assertRaisesRegex(ValueError, "progress is invalid"):
            validate_external_prediction_row(
                row,
                handoff_row=_handoff(),
                transformer_checkpoint_sha256=_SHA_E,
            )

    def test_provider_record_normalization(self) -> None:
        self.assertEqual(
            _normalize_provider_record(
                {"status": True, "prediction_progress": 0.25}
            ),
            (True, 0.25, None),
        )
        status, progress, reason = _normalize_provider_record(
            {"status": "ok", "progress": 1.5}
        )
        self.assertFalse(status)
        self.assertIsNone(progress)
        self.assertEqual(reason, "progress_out_of_range")


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        first = _handoff()
        second = dict(_handoff())
        second.update(
            {
                "sample_id": "sync_2",
                "group_id": "oil::scene",
                "range_accepted": True,
                "garc_full_status": False,
                "garc_prediction_progress": None,
                "garc_predicted_reading": None,
                "garc_failure_reason": "progress_unavailable",
            }
        )
        self.handoff = [first, second]
        self.truth = {
            "sync_1": SimpleNamespace(
                scale_start=0.0, scale_end=100.0, reading=50.0
            ),
            "sync_2": SimpleNamespace(
                scale_start=0.0, scale_end=100.0, reading=80.0
            ),
        }

    def test_failure_receives_nmae_one(self) -> None:
        metrics, errors = score_method(
            method="garc", handoff_rows=self.handoff, truth=self.truth
        )
        np.testing.assert_allclose(errors, [0.0, 1.0])
        self.assertEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["full_denominator_nmae_failure_penalty_1"], 0.5)

    def test_external_uses_same_predicted_range(self) -> None:
        first = _external()
        second = dict(_external())
        second.update(
            {
                "sample_id": "sync_2",
                "group_id": "oil::scene",
                "prediction_progress": 0.5,
            }
        )
        metrics, errors = score_method(
            method=VDN_METHOD,
            handoff_rows=self.handoff,
            truth=self.truth,
            external_rows=[first, second],
        )
        np.testing.assert_allclose(errors, [0.0, 0.3])
        self.assertEqual(metrics["coverage"], 1.0)
        self.assertAlmostEqual(
            metrics["full_denominator_nmae_failure_penalty_1"], 0.15
        )

    def test_group_bootstrap_is_deterministic(self) -> None:
        kwargs = dict(
            comparator_errors=np.asarray([0.1, 0.2, 0.4]),
            garc_errors=np.asarray([0.2, 0.2, 0.1]),
            groups=["a", "a", "b"],
            comparator_name=VDN_METHOD,
            iterations=100,
            seed=17,
        )
        self.assertEqual(
            paired_group_bootstrap(**kwargs), paired_group_bootstrap(**kwargs)
        )


if __name__ == "__main__":
    unittest.main()
