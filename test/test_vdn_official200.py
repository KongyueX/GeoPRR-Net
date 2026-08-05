from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.preflight_vdn_official200 import (
    _require_output_dirs_absent,
    _strict_json,
    write_json_no_clobber,
)
from experiments.train_vdn_official200 import (
    exclusive_writer_lock,
    prepare_output_dir,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_DETERMINISM_POLICY,
    OFFICIAL200_EPOCHS,
    OFFICIAL200_FORMAL_SEEDS,
    OFFICIAL200_STOPPING_POLICY,
    assert_syncg_train_manifest_path,
    assert_train_only_path,
    expected_sample_order_sha256,
    official200_epoch_seed,
    official200_initial_scaler_state,
    official200_learning_rate,
    official200_vector_weight,
    tail_diagnostics,
    validate_official200_history,
)


def _validation(*, angle: float, loss: float = 1.0) -> dict:
    return {
        "loss": loss,
        "heatmap_loss": loss * 0.75,
        "vector_loss": loss * 0.25,
        "samples": 1,
        "valid_directions": 1,
        "direction_coverage": 1.0,
        "angle_mae_degrees": angle,
        "angle_median_degrees": angle,
        "angle_acc_1deg": 0.0,
        "angle_acc_3deg": 0.5,
        "angle_acc_5deg": 1.0,
        "mean_heatmap_peak": 0.8,
    }


def _tail_history(*, improving: bool) -> list[dict]:
    rows = []
    for epoch in range(1, OFFICIAL200_EPOCHS + 1):
        if improving and epoch > 180:
            angle = 1.0 - 0.01 * (epoch - 180)
            loss = 1.0 - 0.008 * (epoch - 180)
        else:
            angle = 1.0
            loss = 1.0
        rows.append(
            {
                "epoch": epoch,
                "validation": _validation(angle=angle, loss=loss),
            }
        )
    return rows


class VDNOfficial200ProtocolTests(unittest.TestCase):
    def test_schedule_has_exact_140_190_boundaries(self):
        self.assertEqual(official200_learning_rate(1), 1e-3)
        self.assertEqual(official200_learning_rate(140), 1e-3)
        self.assertEqual(official200_learning_rate(141), 1e-4)
        self.assertEqual(official200_learning_rate(190), 1e-4)
        self.assertEqual(official200_learning_rate(191), 1e-5)
        self.assertEqual(official200_learning_rate(200), 1e-5)
        for invalid in (0, 201):
            with self.assertRaises(ValueError):
                official200_learning_rate(invalid)

    def test_vector_weight_is_official_linear_0_to_1(self):
        self.assertEqual(official200_vector_weight(1), 0.0)
        self.assertEqual(official200_vector_weight(200), 1.0)
        self.assertAlmostEqual(
            official200_vector_weight(140),
            139.0 / 199.0,
        )
        self.assertAlmostEqual(
            official200_vector_weight(190),
            189.0 / 199.0,
        )

    def test_epoch_seed_is_seed_specific_and_resume_stable(self):
        first = official200_epoch_seed(20260720, 1)
        second = official200_epoch_seed(20260720, 2)
        other = official200_epoch_seed(20260721, 1)
        self.assertEqual(first, 20260720)
        self.assertEqual(second - first, 1009)
        self.assertNotEqual(first, other)
        with self.assertRaises(ValueError):
            official200_epoch_seed(7, 1)

    def test_sample_order_is_reconstructible(self):
        samples = [
            SimpleNamespace(sample_id=f"sample-{index}")
            for index in range(20)
        ]
        first = expected_sample_order_sha256(
            samples,
            epoch_seed=official200_epoch_seed(20260720, 1),
        )
        repeated = expected_sample_order_sha256(
            samples,
            epoch_seed=official200_epoch_seed(20260720, 1),
        )
        second = expected_sample_order_sha256(
            samples,
            epoch_seed=official200_epoch_seed(20260720, 2),
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)

    def test_tail_plateau_is_diagnostic_not_extension_gate(self):
        diagnostic = tail_diagnostics(_tail_history(improving=False))
        self.assertTrue(diagnostic["plateau_observed"])
        self.assertFalse(diagnostic["authorization_gate"])
        self.assertFalse(diagnostic["additional_training_authorized"])
        self.assertFalse(diagnostic["phase4_authorized"])
        self.assertTrue(diagnostic["hard_stopping_boundary_reached"])

    def test_improving_tail_requires_limitation_but_never_more_epochs(self):
        diagnostic = tail_diagnostics(_tail_history(improving=True))
        self.assertFalse(diagnostic["plateau_observed"])
        self.assertTrue(diagnostic["manuscript_limitation_required"])
        self.assertFalse(diagnostic["additional_training_authorized"])
        self.assertFalse(diagnostic["phase4_authorized"])
        self.assertFalse(
            OFFICIAL200_STOPPING_POLICY["adaptive_extension_allowed"]
        )

    def test_complete_synthetic_history_validates_all_200_epochs(self):
        samples = [
            SimpleNamespace(sample_id="sample-a"),
            SimpleNamespace(sample_id="sample-b"),
        ]
        signature = {
            "seed": 20260720,
            "train_samples": 2,
            "validation_samples": 1,
            "batch_size": 8,
            "full_run_max_skipped_optimizer_steps": 0,
            "initial_scaler_state": official200_initial_scaler_state(),
            "preflight": {
                "protocol": "vdn_official200_training_preflight_v1",
                "report_path": "preflight.json",
                "report_sha256": "a" * 64,
                "canonical_payload_sha256": "b" * 64,
            },
            "determinism": OFFICIAL200_DETERMINISM_POLICY,
            "determinism_authorization": {
                "protocol": (
                    "vdn_official200_full_epoch_determinism_probe_v1"
                ),
                "report_path": "determinism.json",
                "report_sha256": "c" * 64,
                "canonical_payload_sha256": "d" * 64,
                "semantic_payload_sha256": "e" * 64,
            },
        }
        history = []
        for epoch in range(1, OFFICIAL200_EPOCHS + 1):
            start = official200_initial_scaler_state()
            start["_growth_tracker"] = epoch - 1
            end = official200_initial_scaler_state()
            end["_growth_tracker"] = epoch
            angle = 2.0 - epoch / 1000.0
            history.append(
                {
                    "epoch": epoch,
                    "epoch_seed": official200_epoch_seed(
                        20260720,
                        epoch,
                    ),
                    "learning_rate": official200_learning_rate(epoch),
                    "train": {
                        "loss": 1.0,
                        "heatmap_loss": 0.75,
                        "vector_loss": 0.25,
                        "samples": 2,
                        "vector_weight": official200_vector_weight(epoch),
                        "optimizer_steps": 1,
                        "skipped_optimizer_steps": 0,
                        "sample_order_sha256": (
                            expected_sample_order_sha256(
                                samples,
                                epoch_seed=official200_epoch_seed(
                                    20260720,
                                    epoch,
                                ),
                            )
                        ),
                        "scaler_start_state": start,
                        "scaler_skipped_batch_indices": [],
                        "scaler_end_state": end,
                    },
                    "validation": _validation(angle=angle),
                    "best": True,
                    "training_elapsed_seconds": float(epoch),
                    "preflight_journal": signature["preflight"],
                    "determinism_authorization": signature[
                        "determinism_authorization"
                    ],
                    "determinism_policy": OFFICIAL200_DETERMINISM_POLICY,
                }
            )
        health = validate_official200_history(
            samples,
            history,
            through_epoch=OFFICIAL200_EPOCHS,
            signature=signature,
        )
        self.assertEqual(health["epochs"], 200)
        self.assertEqual(health["cumulative_optimizer_steps"], 200)
        self.assertEqual(health["sample_orders_verified"], 200)

        history[140]["learning_rate"] = 1e-3
        with self.assertRaisesRegex(ValueError, "learning rate drifted"):
            validate_official200_history(
                samples,
                history,
                through_epoch=OFFICIAL200_EPOCHS,
                signature=signature,
            )


class VDNOfficial200NoClobberTests(unittest.TestCase):
    def test_new_run_requires_absent_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            prepare_output_dir(path, resume=False)
            self.assertTrue(path.is_dir())
            with self.assertRaisesRegex(FileExistsError, "refusing to reuse"):
                prepare_output_dir(path, resume=False)

    def test_resume_requires_authoritative_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            path.mkdir()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "authoritative checkpoint",
            ):
                prepare_output_dir(path, resume=True)

    def test_verified_run_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run"
            path.mkdir()
            (path / "last.pt").write_bytes(b"checkpoint")
            (path / "verification_v1.json").write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "immutable"):
                prepare_output_dir(path, resume=True)

    def test_writer_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with exclusive_writer_lock(path):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    with exclusive_writer_lock(path):
                        pass
            self.assertFalse((path / "writer.lock").exists())

    def test_preflight_requires_all_three_run_dirs_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            records = _require_output_dirs_absent(root)
            self.assertEqual(
                [record["seed"] for record in records],
                list(OFFICIAL200_FORMAL_SEEDS),
            )
            (root / "seed_20260721").mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                _require_output_dirs_absent(root)

    def test_preflight_writer_never_clobbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight.json"
            write_json_no_clobber({"value": 1}, output)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                write_json_no_clobber({"value": 2}, output)
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                digest,
            )

    def test_strict_json_rejects_overflow_to_infinity_recursively(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            path.write_text(
                '{"nested": {"value": 1e999}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite numeric"):
                _strict_json(path)

    def test_strict_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            path.write_text('{"value": 1, "value": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                _strict_json(path)

    def test_train_only_path_guard_rejects_forbidden_namespaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for namespace in (
                "test",
                "public",
                "field",
                "sealed",
                "confirmatory",
                "RPM-10K",
                "Pointer-10K",
                "syncg_test",
            ):
                with self.subTest(namespace=namespace):
                    with self.assertRaisesRegex(
                        ValueError,
                        "forbidden evaluation namespace",
                    ):
                        assert_train_only_path(
                            root / namespace / "artifact.json",
                            label="synthetic",
                        )

    def test_manifest_guard_requires_exact_syncg_train_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            accepted = assert_syncg_train_manifest_path(
                root / "syncg_train.jsonl"
            )
            self.assertEqual(accepted.name, "syncg_train.jsonl")
            with self.assertRaisesRegex(ValueError, "pinned syncg_train"):
                assert_syncg_train_manifest_path(
                    root / "another_train.jsonl"
                )


if __name__ == "__main__":
    unittest.main()
