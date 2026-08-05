from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from experiments.verify_reference_conditioned_training import (
    _assert_group_folds,
    _json_no_clobber,
    _validate_diagnostic_identities,
    _validated_training_seed,
)


class ReferenceConditionedTrainingSeedAuditTest(unittest.TestCase):
    def test_accepts_the_same_explicit_integer_seed(self) -> None:
        self.assertEqual(
            _validated_training_seed(
                {"seed": 20260722},
                {"seed": 20260722},
                label="fixture",
            ),
            20260722,
        )

    def test_rejects_missing_or_mismatched_seed(self) -> None:
        for summary, artifact in (
            ({}, {"seed": 20260722}),
            ({"seed": 20260722}, {}),
            ({"seed": 20260722}, {"seed": 20260723}),
        ):
            with self.subTest(summary=summary, artifact=artifact):
                with self.assertRaisesRegex(ValueError, "seed audit failed"):
                    _validated_training_seed(summary, artifact, label="fixture")

    def test_rejects_boolean_or_string_seed(self) -> None:
        for value in (True, "20260722", 20260722.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "seed audit failed"):
                    _validated_training_seed(
                        {"seed": value},
                        {"seed": value},
                        label="fixture",
                    )

    def test_diagnostic_group_identity_must_match_input(self) -> None:
        rows = [{"sample_id": "sample-a", "group_id": "physical-a"}]
        diagnostics = [{"sample_id": "sample-a", "group_id": "physical-b"}]
        with self.assertRaisesRegex(ValueError, "group identity mismatch"):
            _validate_diagnostic_identities(
                rows,
                diagnostics,
                label="fixture",
            )

    def test_fold_audit_rejects_boolean_and_cross_fold_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid fold"):
            _assert_group_folds(
                [{"group_id": "physical-a", "fold": True}],
                fold_name="fold",
                label="fixture",
            )
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            _assert_group_folds(
                [
                    {"group_id": "physical-a", "fold": 1},
                    {"group_id": "physical-a", "fold": 2},
                ],
                fold_name="fold",
                label="fixture",
            )

    def test_formal_verification_writer_is_no_clobber_and_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "verification.json"
            _json_no_clobber(path, {"status": "verified"})
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _json_no_clobber(path, {"status": "replacement"})
            with self.assertRaises(ValueError):
                _json_no_clobber(
                    Path(temporary) / "nonfinite.json",
                    {"value": math.nan},
                )


if __name__ == "__main__":
    unittest.main()
