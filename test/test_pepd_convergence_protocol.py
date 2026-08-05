from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from experiments.build_pepd_oof_handoff import (
    VERSIONED_COLLECTOR_CONTRACT_PROTOCOL,
    _string_set_hash,
    _versioned_collector_contract,
    _write_or_validate as _write_oof_handoff_or_validate,
)
from experiments.strict_json import (
    strict_json_load,
    strict_json_source_sha256,
)
from experiments.evaluate_pepd_grouped_validation import (
    GroupedValidationPerspectiveDataset,
    _decode_direction_views,
    _group_bootstrap,
    _group_bootstrap_decoder_contrast,
    _rows_for_decoder_view,
    _summarize,
    _write_or_validate as _write_grouped_or_validate,
)
from experiments.evaluate_pepd_uncertainty_grouped_validation import (
    _direction_summary,
)
from experiments.pepd_convergence_protocol import (
    CONTINUATION_LEARNING_RATE,
    DECODER_VIEWS,
    FORMAL_MODEL_SOURCE_SHA256,
    FORMAL_SEEDS,
    PARENT_EPOCH,
    PEPD_CONTINUATION_PROTOCOL,
    PEPD_COHORT_PROTOCOL,
    MECHANISM_ARMS,
    PRIMARY_MECHANISM_ARMS,
    UNCERTAINTY_MECHANISM_ARMS,
    PEPD_TRAINING_PROTOCOL,
    TERMINAL_EPOCH,
    build_continuation_signature,
    audit_main_convergence_cohort,
    convergence_audit,
    expected_full_configuration,
    formal_manifest_path,
    mechanism_parent_dir,
    formal_parent_pin,
    validate_combined_history,
    validate_parent_summary,
)
from experiments.pepd_uncertainty_objectives import (
    make_global_log_variance,
    probabilistic_direction_loss_with_uncertainty_mode,
    uncertainty_semantics,
)
from experiments.probabilistic_pivot_direction import (
    decode_probabilistic_pivot_direction,
)
from experiments.pepd_uncertainty_metrics import (
    grouped_uncertainty_diagnostics,
)
from experiments.preflight_pepd_mechanism import (
    expected_mechanism_configuration,
)
from experiments.train_pepd_mechanism_continuation_syncg import (
    mechanism_continuation_signature,
)
from experiments.train_pepd_uncertainty_ablation_syncg import (
    model_state_sha256,
)
from experiments.preflight_pepd_convergence import _audit_parent_checkpoint
from experiments.train_pepd_convergence_syncg import (
    _restore_rng_state,
    _rng_state,
    _training_namespace,
)
from experiments.verify_pepd_convergence_run import (
    _validate_phase2_checkpoint,
)
from experiments.verify_pepd_uncertainty_ablation import (
    _global_variance_changed,
    _variance_head_changed,
)


def _record(
    epoch: int,
    mae: float,
    *,
    train_samples: int,
    validation_samples: int,
    learning_rate: float,
) -> dict:
    expected_batches = math.ceil(train_samples / 24)
    return {
        "epoch": epoch,
        "learning_rate": learning_rate,
        "train": {
            "samples": train_samples,
            "optimizer_steps": expected_batches,
            "skipped_optimizer_steps": 0,
            "loss": -1.0,
        },
        "validation": {
            "samples": validation_samples,
            "loss": -1.0,
            "angle_mae_degrees": mae,
            "angular_calibration_nll": -3.0,
            "pivot_mean_error_fraction": 0.01,
            "direction_coverage": 1.0,
        },
    }


def _history(
    *,
    train_samples: int,
    validation_samples: int,
    trailing: list[float] | None = None,
) -> list[dict]:
    values = [1.0 - 0.01 * min(epoch, 20) for epoch in range(1, 61)]
    values[20:50] = [0.80] * 30
    values[50:] = trailing or [0.80] * 10
    return [
        _record(
            epoch,
            values[epoch - 1],
            train_samples=train_samples,
            validation_samples=validation_samples,
            learning_rate=3e-4 if epoch <= 30 else CONTINUATION_LEARNING_RATE,
        )
        for epoch in range(1, 61)
    ]


class PEPDConvergenceProtocolTests(unittest.TestCase):
    def test_manifest_gate_rejects_non_train_manifest_without_reading(self) -> None:
        with self.assertRaisesRegex(ValueError, "SyncG train manifest"):
            formal_manifest_path(Path("artifacts/manifests/syncg_test.jsonl"))

    def test_frozen_parent_summary_contract(self) -> None:
        pin = formal_parent_pin(20260720)
        signature = expected_full_configuration() | {
            "seed": pin.seed,
            "manifest_sha256": (
                "429e4bc24515b4bf7a1d6fb638210e653872d287ba1edc0c1d6e193cb99d28ca"
            ),
            "manifest_protocol_sha256": (
                "315e17ac8aba46d00f84a0060dba145d7e036fa096bd26423dbd34a170600c59"
            ),
            "model_source_sha256": FORMAL_MODEL_SOURCE_SHA256,
            "trainer_source_sha256": (
                "a992be129084e3ab19a230d95f333b91e37faad10882b50e1c4b7682c63cf284"
            ),
            "train_samples": pin.train_samples,
            "validation_samples": pin.validation_samples,
            "train_sample_ids_sha256": pin.train_ids_sha256,
            "validation_sample_ids_sha256": pin.validation_ids_sha256,
        }
        summary = {
            "protocol": PEPD_TRAINING_PROTOCOL,
            "status": "complete",
            "signature": signature,
            "history": [
                _record(
                    epoch,
                    1.0,
                    train_samples=pin.train_samples,
                    validation_samples=pin.validation_samples,
                    learning_rate=(
                        CONTINUATION_LEARNING_RATE
                        + 0.5
                        * (3e-4 - CONTINUATION_LEARNING_RATE)
                        * (
                            1.0
                            + math.cos(
                                math.pi * float(epoch - 1) / float(PARENT_EPOCH)
                            )
                        )
                    ),
                )
                for epoch in range(1, 31)
            ],
            "best_epoch": pin.best_epoch,
            "best_validation_angle_mae_degrees": pin.best_angle_mae_degrees,
        }
        validate_parent_summary(summary, seed=pin.seed)
        summary["signature"]["epochs"] = 31
        with self.assertRaisesRegex(ValueError, "epochs mismatch"):
            validate_parent_summary(summary, seed=pin.seed)

    def test_plateau_converges_only_at_fixed_terminal_epoch(self) -> None:
        history = _history(train_samples=100, validation_samples=20)
        audit = convergence_audit(history, best_epoch=40)
        self.assertTrue(audit["converged"])
        self.assertEqual(audit["trailing_epochs"], [51, 60])

    def test_material_trend_fails_convergence(self) -> None:
        trailing = [0.80 - 0.01 * index for index in range(10)]
        history = _history(
            train_samples=100,
            validation_samples=20,
            trailing=trailing,
        )
        audit = convergence_audit(history, best_epoch=60)
        self.assertFalse(audit["converged"])
        self.assertFalse(audit["checks"]["absolute_trailing_slope"])
        self.assertIn("do not extend", audit["failure_action"])

    def test_boundary_gain_fails_even_with_small_global_slope(self) -> None:
        trailing = [0.80] * 7 + [0.77, 0.77, 0.77]
        history = _history(
            train_samples=100,
            validation_samples=20,
            trailing=trailing,
        )
        audit = convergence_audit(history, best_epoch=60)
        self.assertFalse(audit["converged"])
        self.assertFalse(audit["checks"]["boundary_not_materially_better"])

    def test_history_gate_rejects_lr_and_accounting_drift(self) -> None:
        history = _history(train_samples=100, validation_samples=20)
        validate_combined_history(
            history,
            train_samples=100,
            validation_samples=20,
        )
        history[30]["learning_rate"] = 4e-6
        with self.assertRaisesRegex(ValueError, "learning rate drifted"):
            validate_combined_history(
                history,
                train_samples=100,
                validation_samples=20,
            )

    def test_continuation_signature_is_fixed_and_train_val_only(self) -> None:
        signature = build_continuation_signature(
            seed=20260720,
            parent_summary_sha256=formal_parent_pin(20260720).summary_sha256,
            parent_best_sha256=formal_parent_pin(20260720).best_sha256,
            parent_last_sha256=formal_parent_pin(20260720).last_sha256,
            continuation_source_sha256="c" * 64,
            imported_trainer_source_sha256="t" * 64,
            model_source_sha256=FORMAL_MODEL_SOURCE_SHA256,
        )
        self.assertEqual(signature["protocol"], PEPD_CONTINUATION_PROTOCOL)
        self.assertEqual(signature["terminal_epoch"], TERMINAL_EPOCH)
        self.assertFalse(signature["early_stopping"])
        self.assertEqual(
            signature["strict_json_source_sha256"],
            strict_json_source_sha256(),
        )
        self.assertEqual(
            signature["scope"],
            "SyncG official train grouped validation only",
        )

    def test_main_cohort_gate_requires_exact_verified_seed_membership(self) -> None:
        cohort = {
            "protocol": PEPD_COHORT_PROTOCOL,
            "status": "converged",
            "all_runs_verified": True,
            "all_runs_converged": True,
            "seeds": list(FORMAL_SEEDS),
            "runs": [
                {
                    "seed": seed,
                    "verified": True,
                    "converged": True,
                    "best_checkpoint_sha256": "a" * 64,
                    "summary_sha256": "b" * 64,
                    "verification_sha256": "c" * 64,
                }
                for seed in FORMAL_SEEDS
            ],
            "grouped_validation_controlled_perspective_authorized": True,
            "grouped_validation_controlled_robustness_authorized": True,
            "public_test_field_evaluation_authorized": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cohort.json"
            path.write_text(json.dumps(cohort), encoding="utf-8")
            audit = audit_main_convergence_cohort(path)
            self.assertTrue(audit["verified"])
            cohort["runs"][0]["converged"] = False
            path.write_text(json.dumps(cohort), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not verified/converged"):
                audit_main_convergence_cohort(path)

    def test_parent_checkpoint_audit_requires_terminal_scheduler(self) -> None:
        signature = {"protocol": PEPD_TRAINING_PROTOCOL}
        state = {"weight": torch.ones(2)}
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        optimizer.zero_grad()
        optimizer.param_groups[0]["params"][0].grad = torch.ones(1)
        optimizer.step()
        optimizer_state = optimizer.state_dict()
        optimizer_state["param_groups"][0]["lr"] = CONTINUATION_LEARNING_RATE
        checkpoint = {
            "protocol": PEPD_TRAINING_PROTOCOL,
            "signature": signature,
            "epoch": PARENT_EPOCH,
            "history": [{}] * PARENT_EPOCH,
            "model_state": state,
            "optimizer_state": optimizer_state,
            "scheduler_state": {
                "T_max": PARENT_EPOCH,
                "last_epoch": PARENT_EPOCH,
                "eta_min": CONTINUATION_LEARNING_RATE,
                "_last_lr": [CONTINUATION_LEARNING_RATE],
            },
            "scaler_state": {"scale": 256.0},
        }
        health = _audit_parent_checkpoint(
            checkpoint,
            summary_signature=signature,
            expected_epoch=PARENT_EPOCH,
            expect_optimizer_state=True,
        )
        self.assertEqual(health["nonfinite_count"], 0)
        checkpoint["scheduler_state"]["last_epoch"] = 29
        with self.assertRaisesRegex(ValueError, "last_epoch mismatch"):
            _audit_parent_checkpoint(
                checkpoint,
                summary_signature=signature,
                expected_epoch=PARENT_EPOCH,
                expect_optimizer_state=True,
            )

    def test_phase2_checkpoint_requires_rng_and_has_no_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=CONTINUATION_LEARNING_RATE)
        optimizer.zero_grad()
        parameter.grad = torch.ones(1)
        optimizer.step()
        generator = torch.Generator().manual_seed(7)
        checkpoint = {
            "protocol": PEPD_TRAINING_PROTOCOL,
            "signature": {"parent": True},
            "continuation_signature": {"phase2": True},
            "epoch": TERMINAL_EPOCH,
            "history": [{}] * TERMINAL_EPOCH,
            "model_state": {"weight": torch.ones(1)},
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": {"scale": 128.0},
        } | _rng_state(generator)
        health = _validate_phase2_checkpoint(
            checkpoint,
            expected_epoch=TERMINAL_EPOCH,
            parent_signature={"parent": True},
            continuation_signature={"phase2": True},
        )
        self.assertGreater(health["parameter_count"], 0)
        checkpoint["scheduler_state"] = {}
        with self.assertRaisesRegex(ValueError, "scheduler"):
            _validate_phase2_checkpoint(
                checkpoint,
                expected_epoch=TERMINAL_EPOCH,
                parent_signature={"parent": True},
                continuation_signature={"phase2": True},
            )

    def test_rng_roundtrip_restores_loader_generator(self) -> None:
        generator = torch.Generator().manual_seed(11)
        snapshot = _rng_state(generator)
        expected = torch.rand(4, generator=generator)
        _restore_rng_state(snapshot, generator)
        actual = torch.rand(4, generator=generator)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_training_namespace_preserves_parent_loss_and_geometry(self) -> None:
        signature = expected_full_configuration()
        args = _training_namespace(signature)
        self.assertEqual(args.equivariance_weight, 0.5)
        self.assertEqual(args.paired_supervision_weight, 1.0)
        self.assertEqual(args.perspective_probability, 0.8)
        self.assertEqual(args.learning_rate, CONTINUATION_LEARNING_RATE)
        self.assertEqual(args.epochs, TERMINAL_EPOCH)
        self.assertEqual(args.workers, 0)

    def test_mechanism_aliases_do_not_create_a_fourth_geometry_arm(self) -> None:
        self.assertEqual(
            PRIMARY_MECHANISM_ARMS,
            ("full", "paired_supervision_only", "no_projective_pair"),
        )
        paired = MECHANISM_ARMS["paired_supervision_only"]
        self.assertEqual(paired.equivariance_weight, 0.0)
        self.assertEqual(paired.perspective_probability, 0.8)
        self.assertIn("no-equiv", paired.interpretation)

    def test_mechanism_phase1_paths_are_predeclared(self) -> None:
        legacy = mechanism_parent_dir("paired_supervision_only", 20260722)
        future = mechanism_parent_dir("paired_supervision_only", 20260720)
        self.assertIn("no_equivariance_loss", str(legacy))
        self.assertIn("pepd_mechanism_phase1", str(future))
        self.assertTrue(str(future).endswith("seed_20260720"))

    def test_mechanism_configuration_and_phase2_signature_are_fixed(self) -> None:
        paired = expected_mechanism_configuration("paired_supervision_only")
        no_pair = expected_mechanism_configuration("no_projective_pair")
        self.assertEqual(paired["equivariance_weight"], 0.0)
        self.assertEqual(paired["perspective_probability"], 0.8)
        self.assertEqual(no_pair["perspective_probability"], 0.0)
        signature = mechanism_continuation_signature(
            arm="no_projective_pair",
            seed=20260720,
            parent_summary_sha256="a" * 64,
            parent_best_sha256="b" * 64,
            parent_last_sha256="c" * 64,
            authoritative_pepd_v2_gate={
                "checked": True,
                "cohort_sha256": "d" * 64,
            },
        )
        self.assertEqual(signature["terminal_epoch"], 60)
        self.assertFalse(signature["early_stopping"])
        self.assertEqual(signature["equivariance_weight"], 0.0)
        self.assertEqual(signature["perspective_probability"], 0.0)
        self.assertEqual(
            signature["authoritative_pepd_v2_gate"],
            {"checked": True, "cohort_sha256": "d" * 64},
        )
        self.assertIn("not algorithm selection", signature["role"])

    def test_uncertainty_ablations_are_frozen_secondary_arms(self) -> None:
        self.assertEqual(
            UNCERTAINTY_MECHANISM_ARMS,
            (
                "learned_heteroscedastic",
                "global_homoscedastic",
                "no_angular_nll",
            ),
        )
        self.assertEqual(
            MECHANISM_ARMS["global_homoscedastic"].uncertainty_objective,
            "learned_global_homoscedastic_angular_nll",
        )
        self.assertEqual(
            MECHANISM_ARMS["no_angular_nll"].formal_priority,
            "secondary",
        )

    def test_uncertainty_objectives_only_change_angular_term(self) -> None:
        def make_inputs():
            return (
                torch.zeros((2, 1, 4, 4), requires_grad=True),
                torch.tensor(
                    [[1.0, 0.2], [0.1, 1.0]],
                    requires_grad=True,
                ),
                torch.zeros((2, 8), requires_grad=True),
                torch.tensor([[0.2], [-0.3]], requires_grad=True),
                torch.zeros((2, 1, 4, 4)),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            )

        logvar_grads = {}
        global_grads = {}
        losses = {}
        for mode in UNCERTAINTY_MECHANISM_ARMS:
            inputs = make_inputs()
            global_log_variance = torch.tensor(
                0.0,
                requires_grad=True,
            )
            loss, components = probabilistic_direction_loss_with_uncertainty_mode(
                *inputs,
                pivot_weight=1.0,
                bin_weight=0.2,
                vector_weight=0.5,
                soft_target_sigma_bins=1.25,
                uncertainty_mode=mode,
                global_log_variance_raw=global_log_variance,
            )
            loss.backward()
            gradient = inputs[3].grad
            logvar_grads[mode] = (
                None if gradient is None else float(torch.linalg.vector_norm(gradient))
            )
            global_gradient = global_log_variance.grad
            global_grads[mode] = (
                None
                if global_gradient is None
                else float(torch.abs(global_gradient))
            )
            losses[mode] = float(loss.detach())
            self.assertAlmostEqual(
                float(components["pivot_loss"]),
                float(
                    probabilistic_direction_loss_with_uncertainty_mode(
                        *make_inputs(),
                        pivot_weight=1.0,
                        bin_weight=0.2,
                        vector_weight=0.5,
                        soft_target_sigma_bins=1.25,
                        uncertainty_mode="learned_heteroscedastic",
                        global_log_variance_raw=torch.tensor(0.0),
                    )[1]["pivot_loss"]
                ),
            )
        self.assertGreater(logvar_grads["learned_heteroscedastic"], 0.0)
        self.assertIsNone(logvar_grads["global_homoscedastic"])
        self.assertIsNone(logvar_grads["no_angular_nll"])
        self.assertIsNone(global_grads["learned_heteroscedastic"])
        self.assertGreater(global_grads["global_homoscedastic"], 0.0)
        self.assertIsNone(global_grads["no_angular_nll"])
        self.assertNotEqual(
            losses["learned_heteroscedastic"],
            losses["global_homoscedastic"],
        )

    def test_global_homoscedastic_is_learned_but_has_no_ranking(self) -> None:
        raw = torch.tensor([[1.0], [-1.0]])
        learned = uncertainty_semantics(raw, mode="learned_heteroscedastic")
        global_value = torch.tensor(-2.0, requires_grad=True)
        homoscedastic = uncertainty_semantics(
            raw,
            mode="global_homoscedastic",
            global_log_variance_raw=global_value,
        )
        no_nll = uncertainty_semantics(raw, mode="no_angular_nll")
        self.assertFalse(
            torch.equal(learned.log_variance, homoscedastic.log_variance)
        )
        torch.testing.assert_close(
            homoscedastic.log_variance,
            torch.full((2,), -2.0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(homoscedastic.sample_ranking_available)
        self.assertIsNone(no_nll.log_variance)
        self.assertIsNone(no_nll.angle_std_degrees)
        self.assertIn("unavailable", no_nll.calibration_semantics)

    def test_model_state_hash_binds_names_shapes_and_values(self) -> None:
        first = {
            "b": torch.tensor([2.0]),
            "a": torch.tensor([[1.0, 3.0]]),
        }
        reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
        changed = {"a": torch.tensor([[1.0, 4.0]]), "b": first["b"].clone()}
        self.assertEqual(model_state_sha256(first), model_state_sha256(reordered))
        self.assertNotEqual(model_state_sha256(first), model_state_sha256(changed))

    def test_variance_head_change_audit_ignores_unrelated_parameters(self) -> None:
        initial = {
            "encoder.weight": torch.tensor([1.0]),
            "log_variance_head.weight": torch.tensor([[2.0]]),
            "log_variance_head.bias": torch.tensor([0.0]),
        }
        unrelated_change = {
            name: value.clone() for name, value in initial.items()
        }
        unrelated_change["encoder.weight"] += 1.0
        self.assertFalse(
            _variance_head_changed(initial, unrelated_change)
        )
        variance_change = {
            name: value.clone() for name, value in initial.items()
        }
        variance_change["log_variance_head.bias"] += 0.25
        self.assertTrue(_variance_head_changed(initial, variance_change))

    def test_global_variance_change_audit_is_exact(self) -> None:
        parameter = make_global_log_variance(device=torch.device("cpu"))
        unchanged = {
            "global_log_variance_state": parameter.detach().clone()
        }
        changed = {
            "global_log_variance_state": torch.tensor(-0.25)
        }
        self.assertFalse(
            _global_variance_changed(unchanged, initial_value=0.0)
        )
        self.assertTrue(
            _global_variance_changed(changed, initial_value=0.0)
        )

    def test_grouped_uncertainty_aurc_handles_ties_and_failures(self) -> None:
        ranked = [
            {
                "group_id": "easy",
                "valid": True,
                "angle_error_degrees": 1.0,
                "angle_std_degrees": 1.0,
            },
            {
                "group_id": "hard",
                "valid": False,
                "angle_error_degrees": 0.0,
                "angle_std_degrees": 10.0,
            },
        ]
        heteroscedastic = grouped_uncertainty_diagnostics(
            ranked,
            mode="learned_heteroscedastic",
        )
        self.assertEqual(heteroscedastic["invalid_directions"], 1)
        self.assertGreater(
            heteroscedastic["risk_coverage"]["aurc_skill_degrees"],
            0.0,
        )
        tied = [dict(row, angle_std_degrees=2.0) for row in ranked]
        homoscedastic = grouped_uncertainty_diagnostics(
            tied,
            mode="global_homoscedastic",
        )
        self.assertEqual(
            homoscedastic["risk_coverage"][
                "distinct_uncertainty_scores"
            ],
            1,
        )
        self.assertAlmostEqual(
            homoscedastic["risk_coverage"]["aurc_skill_degrees"],
            0.0,
        )
        absent = grouped_uncertainty_diagnostics(
            [
                dict(row, angle_std_degrees=None)
                for row in ranked
            ],
            mode="no_angular_nll",
        )
        self.assertFalse(absent["risk_coverage"]["available"])

    def test_oof_group_identity_hash_is_order_independent_and_exact(self) -> None:
        self.assertEqual(
            _string_set_hash(["meter-b", "meter-a", "meter-a"]),
            _string_set_hash(["meter-a", "meter-b"]),
        )
        self.assertNotEqual(
            _string_set_hash(["meter-a", "meter-b"]),
            _string_set_hash(["meter-a", "meter-c"]),
        )

    def test_oof_handoff_authorizes_only_the_versioned_collector(self) -> None:
        contract = _versioned_collector_contract()
        self.assertEqual(
            set(contract),
            {
                "protocol",
                "authorized",
                "collector",
                "collector_source_sha256",
                "required_cli",
                "legacy_checkpoint_fallback_allowed",
            },
        )
        self.assertEqual(
            contract["protocol"],
            VERSIONED_COLLECTOR_CONTRACT_PROTOCOL,
        )
        self.assertEqual(
            contract["required_cli"],
            ["--pepd-oof-handoff", "--pepd-cohort"],
        )
        self.assertFalse(contract["legacy_checkpoint_fallback_allowed"])

    def test_authorization_json_rejects_duplicates_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "authorization.json"
            path.write_text('{"authorized":true,"authorized":false}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                strict_json_load(path)
            path.write_text('{"metric":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                strict_json_load(path)

    def test_authorization_writer_is_no_clobber_and_finite_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "handoff.json"
            _write_oof_handoff_or_validate(path, {"authorized": True})
            _write_oof_handoff_or_validate(path, {"authorized": True})
            with self.assertRaises(FileExistsError):
                _write_oof_handoff_or_validate(path, {"authorized": False})
            nonfinite = Path(temporary) / "nonfinite.json"
            with self.assertRaises(ValueError):
                _write_oof_handoff_or_validate(
                    nonfinite,
                    {"metric": float("nan")},
                )
            self.assertFalse(nonfinite.exists())


class _SyntheticSample:
    sample_id = "synthetic-1"
    group_id = "meter-a"
    dial_bbox = (8.0, 8.0, 56.0, 56.0)
    pointer_tail = np.asarray([32.0, 32.0], dtype=np.float32)
    pointer_tip = np.asarray([48.0, 32.0], dtype=np.float32)

    def __init__(self, image_path: str) -> None:
        self.image_path = image_path


class PEPDGroupedValidationTests(unittest.TestCase):
    def test_perspective_dataset_is_deterministic_on_synthetic_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dial.png"
            image = np.full((64, 64, 3), 127, dtype=np.uint8)
            cv2.line(image, (32, 32), (48, 32), (0, 0, 255), 2)
            self.assertTrue(cv2.imwrite(str(path), image))
            sample = _SyntheticSample(str(path))
            dataset = GroupedValidationPerspectiveDataset(
                [sample],
                image_size=64,
                expansion=1.0,
                condition="perspective_severe",
                degradation_seed=20260724,
            )
            first = dataset[0]
            second = dataset[0]
            torch.testing.assert_close(
                first["image"],
                second["image"],
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                first["target_direction"],
                second["target_direction"],
                rtol=0.0,
                atol=0.0,
            )
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(first["target_direction"])),
                1.0,
                places=6,
            )

    def test_combined_severe_dataset_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dial.png"
            image = np.full((64, 64, 3), 127, dtype=np.uint8)
            cv2.line(image, (32, 32), (48, 32), (0, 0, 255), 2)
            self.assertTrue(cv2.imwrite(str(path), image))
            dataset = GroupedValidationPerspectiveDataset(
                [_SyntheticSample(str(path))],
                image_size=64,
                expansion=1.0,
                condition="combined_severe",
                degradation_seed=20260724,
            )
            first = dataset[0]
            second = dataset[0]
            torch.testing.assert_close(first["image"], second["image"])
            torch.testing.assert_close(first["homography"], second["homography"])
            self.assertFalse(
                torch.equal(
                    first["homography"],
                    torch.eye(3, dtype=first["homography"].dtype),
                )
            )

    def test_fixed_decoder_views_share_outputs_and_have_fixed_order(self) -> None:
        pivot = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        direct = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        logits = torch.full((1, 72), -20.0, dtype=torch.float32)
        logits[0, 18] = 20.0
        log_variance = torch.zeros((1, 1), dtype=torch.float32)
        outputs = (pivot, direct, logits, log_variance)
        prediction = decode_probabilistic_pivot_direction(*outputs)
        views = _decode_direction_views(outputs, prediction)
        self.assertEqual(tuple(views), DECODER_VIEWS)
        torch.testing.assert_close(
            views["direct"][0],
            torch.tensor([[1.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            views["circular"][0],
            torch.tensor([[0.0, 1.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        expected_fused = torch.tensor(
            [[math.sqrt(0.5), math.sqrt(0.5)]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            views["fused"][0],
            expected_fused,
            atol=1e-6,
            rtol=0.0,
        )

    def test_decoder_contrast_uses_all_denominators_and_groups(self) -> None:
        rows = [
            {
                "sample_id": "a1",
                "group_id": "a",
                "angle_std_degrees": 2.0,
                "pivot_valid": True,
                "pivot_error_fraction": 0.0,
                "decoder_views": {
                    "direct": {
                        "valid": False,
                        "angle_error_degrees": 0.0,
                        "signed_angle_error_degrees": 0.0,
                    },
                    "circular": {
                        "valid": True,
                        "angle_error_degrees": 4.0,
                        "signed_angle_error_degrees": 4.0,
                    },
                    "fused": {
                        "valid": True,
                        "angle_error_degrees": 2.0,
                        "signed_angle_error_degrees": 2.0,
                    },
                },
            },
            {
                "sample_id": "b1",
                "group_id": "b",
                "angle_std_degrees": 2.0,
                "pivot_valid": True,
                "pivot_error_fraction": 0.0,
                "decoder_views": {
                    "direct": {
                        "valid": True,
                        "angle_error_degrees": 6.0,
                        "signed_angle_error_degrees": 6.0,
                    },
                    "circular": {
                        "valid": True,
                        "angle_error_degrees": 3.0,
                        "signed_angle_error_degrees": 3.0,
                    },
                    "fused": {
                        "valid": True,
                        "angle_error_degrees": 1.0,
                        "signed_angle_error_degrees": 1.0,
                    },
                },
            },
        ]
        direct_rows = _rows_for_decoder_view(rows, "direct")
        self.assertAlmostEqual(
            _summarize(direct_rows)["angle_mae_degrees"],
            93.0,
        )
        contrast = _group_bootstrap_decoder_contrast(
            rows,
            comparator="direct",
            reference="fused",
            iterations=100,
            seed=7,
        )
        self.assertEqual(contrast["groups"], 2)
        self.assertAlmostEqual(
            contrast["group_macro_effect_degrees"],
            91.5,
        )

    def test_grouped_evaluator_publish_is_idempotent_not_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            _write_grouped_or_validate(path, "same\n")
            _write_grouped_or_validate(path, "same\n")
            with self.assertRaises(FileExistsError):
                _write_grouped_or_validate(path, "different\n")

    def test_grouped_metrics_use_groups_not_samples(self) -> None:
        rows = [
            {
                "group_id": "a",
                "valid": True,
                "angle_error_degrees": 1.0,
                "signed_angle_error_degrees": 1.0,
                "angle_std_degrees": 2.0,
                "pivot_error_fraction": 0.01,
            },
            {
                "group_id": "a",
                "valid": True,
                "angle_error_degrees": 3.0,
                "signed_angle_error_degrees": -3.0,
                "angle_std_degrees": 4.0,
                "pivot_error_fraction": 0.02,
            },
            {
                "group_id": "b",
                "valid": True,
                "angle_error_degrees": 10.0,
                "signed_angle_error_degrees": 10.0,
                "angle_std_degrees": 8.0,
                "pivot_error_fraction": 0.03,
            },
        ]
        metrics = _summarize(rows)
        grouped = _group_bootstrap(rows, iterations=100, seed=3)
        self.assertAlmostEqual(metrics["angle_mae_degrees"], 14.0 / 3.0)
        self.assertAlmostEqual(grouped["macro_angle_mae_degrees"], 6.0)
        self.assertEqual(grouped["groups"], 2)

    def test_invalid_predictions_remain_in_primary_metric_denominators(self) -> None:
        rows = [
            {
                "group_id": "a",
                "valid": True,
                "pivot_valid": True,
                "angle_error_degrees": 0.0,
                "signed_angle_error_degrees": 0.0,
                "angle_std_degrees": 2.0,
                "pivot_error_fraction": 0.0,
            },
            {
                "group_id": "b",
                "valid": False,
                "pivot_valid": False,
                "angle_error_degrees": 0.0,
                "signed_angle_error_degrees": 0.0,
                "angle_std_degrees": 2.0,
                "pivot_error_fraction": 0.0,
            },
        ]
        main = _summarize(rows)
        uncertainty = _direction_summary(rows)
        grouped = _group_bootstrap(rows, iterations=50, seed=9)
        self.assertAlmostEqual(main["angle_mae_degrees"], 90.0)
        self.assertAlmostEqual(uncertainty["angle_mae_degrees"], 90.0)
        self.assertAlmostEqual(main["angle_acc_1deg"], 0.5)
        self.assertAlmostEqual(
            main["pivot_mean_error_fraction"],
            math.sqrt(2.0) / 2.0,
        )
        self.assertAlmostEqual(
            grouped["macro_angle_mae_degrees"],
            90.0,
        )


if __name__ == "__main__":
    unittest.main()
