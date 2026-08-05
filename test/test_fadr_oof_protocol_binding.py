from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.fadr_multiseed_protocol import EXPECTED_OOF_PROTOCOL
from experiments.train_reference_conditioned_progress_calibrator import (
    _validate_input_oof_contract as validate_calibrator_input,
)
from experiments.train_reference_conditioned_router import (
    _validate_input_oof_contract as validate_router_input,
)
from experiments.uncertainty_fusion import UNCERTAINTY_FUSION_OOF_PROTOCOL
from experiments.vdn_baseline import sha256_file
from experiments.verify_reference_conditioned_training import (
    _declared_oof_protocol_matches,
)


class FadrOofProtocolBindingTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        metadata_protocol: str = EXPECTED_OOF_PROTOCOL,
        summary_protocol: str | None = None,
    ) -> Path:
        oof = root / "authoritative_oof.jsonl"
        oof.write_text("{}\n", encoding="utf-8")
        oof.with_name(oof.name + ".meta.json").write_text(
            json.dumps(
                {
                    "signature": {
                        "protocol": metadata_protocol,
                        "split": "SyncG/train only",
                        "test_sets_used": [],
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        oof.with_name(oof.stem + ".summary.json").write_text(
            json.dumps(
                {
                    "protocol": summary_protocol or metadata_protocol,
                    "status": "complete",
                    "output_sha256": sha256_file(oof),
                    "group_leakage_count": 0,
                    "test_samples_used": 0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return oof

    def test_both_formal_consumers_accept_the_exact_v2_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oof = self._fixture(Path(temporary))
            calibrator_metadata, calibrator_summary = validate_calibrator_input(
                oof,
                expected_protocol=EXPECTED_OOF_PROTOCOL,
            )
            router_metadata, router_summary = validate_router_input(
                oof,
                expected_protocol=EXPECTED_OOF_PROTOCOL,
            )
        self.assertEqual(
            calibrator_metadata["signature"]["protocol"],
            EXPECTED_OOF_PROTOCOL,
        )
        self.assertEqual(calibrator_summary["protocol"], EXPECTED_OOF_PROTOCOL)
        self.assertEqual(router_metadata, calibrator_metadata)
        self.assertEqual(router_summary, calibrator_summary)

    def test_v2_metadata_is_rejected_when_a_consumer_still_expects_v1(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oof = self._fixture(Path(temporary))
            for validator in (validate_calibrator_input, validate_router_input):
                with self.subTest(validator=validator.__module__):
                    with self.assertRaisesRegex(ValueError, "wrong.*OOF protocol"):
                        validator(
                            oof,
                            expected_protocol=UNCERTAINTY_FUSION_OOF_PROTOCOL,
                        )

    def test_legacy_v1_contract_remains_supported_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oof = self._fixture(
                Path(temporary),
                metadata_protocol=UNCERTAINTY_FUSION_OOF_PROTOCOL,
            )
            for validator in (validate_calibrator_input, validate_router_input):
                with self.subTest(validator=validator.__module__):
                    metadata, summary = validator(
                        oof,
                        expected_protocol=UNCERTAINTY_FUSION_OOF_PROTOCOL,
                    )
                    self.assertEqual(
                        metadata["signature"]["protocol"],
                        UNCERTAINTY_FUSION_OOF_PROTOCOL,
                    )
                    self.assertEqual(
                        summary["protocol"],
                        UNCERTAINTY_FUSION_OOF_PROTOCOL,
                    )

    def test_metadata_and_summary_must_declare_the_same_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            oof = self._fixture(
                Path(temporary),
                summary_protocol=UNCERTAINTY_FUSION_OOF_PROTOCOL,
            )
            for validator in (validate_calibrator_input, validate_router_input):
                with self.subTest(validator=validator.__module__):
                    with self.assertRaisesRegex(ValueError, "train-only audit"):
                        validator(oof, expected_protocol=EXPECTED_OOF_PROTOCOL)

    def test_missing_artifact_declaration_is_legacy_v1_only(self) -> None:
        self.assertTrue(
            _declared_oof_protocol_matches(
                None,
                expected=UNCERTAINTY_FUSION_OOF_PROTOCOL,
            )
        )
        self.assertFalse(
            _declared_oof_protocol_matches(
                None,
                expected=EXPECTED_OOF_PROTOCOL,
            )
        )


if __name__ == "__main__":
    unittest.main()
