"""CPU-only tests for the frozen VDN phase-2 convergence protocol."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.train_vdn_phase2 import (
    _bootstrap_authoritative_state,
    _exclusive_writer_lock,
    _live_optimizer_step_invariant,
    main as train_vdn_phase2_main,
    _prepare_output_dir,
    _recover_derived_artifacts,
)
from experiments.train_vdn_syncg import _train_epoch
from experiments.vdn_baseline import (
    VDN_PINNED_COMMIT,
    VDN_PROTOCOL,
    sample_ids_hash,
    sha256_file,
)
from experiments.vdn_phase2_protocol import (
    FORMAL_PHASE2_SEEDS,
    PHASE2_CHECKPOINT_PROTOCOL,
    PHASE2_END_EPOCH,
    PHASE2_PROTOCOL,
    PHASE2_SCHEMA_VERSION,
    PHASE2_SOURCE_HASH_PROTOCOL,
    PHASE2_START_EPOCH,
    build_phase2_adam_optimizer,
    convergence_diagnostics,
    expected_optimizer_steps,
    expected_phase2_sample_order_sha256,
    formal_phase2_seed,
    load_parent_lineage,
    normalize_content_inventory_identity,
    phase2_epoch_seed,
    phase2_learning_rate,
    validate_adam_optimizer_state,
    validate_phase_history,
    validate_scaler_state,
    validate_scaler_transition,
)
from experiments.verify_vdn_phase2 import (
    _exclusive_verification_lock,
    write_json_no_clobber,
)


def _history_row(epoch: int, angle: float, loss: float) -> dict:
    return {
        "epoch": epoch,
        "validation": {
            "angle_mae_degrees": angle,
            "loss": loss,
        },
    }


def _scaler_state(*, scale: float = 512.0, growth_tracker: int = 0) -> dict:
    return {
        "scale": scale,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": growth_tracker,
    }


class _TinyTrainingDataset(Dataset):
    def __init__(self, count: int):
        self.sample_ids = [f"tiny-{index}" for index in range(count)]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        image = torch.tensor([float(index + 1)], dtype=torch.float32)
        target_heatmap = torch.tensor([0.0], dtype=torch.float32)
        target_vector = torch.tensor([0.5], dtype=torch.float32)
        direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
        return (
            image,
            target_heatmap,
            target_vector,
            direction,
            self.sample_ids[index],
        )


class _TinyTwoHeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25]))

    def forward(self, images):
        output = images * self.weight
        return output, output


class VDNPhase2ProtocolTests(unittest.TestCase):
    def test_formal_seed_and_schedule_are_frozen(self):
        self.assertEqual(
            FORMAL_PHASE2_SEEDS,
            {
                20260720: 21260723,
                20260721: 21260724,
                20260722: 21260725,
            },
        )
        self.assertEqual(formal_phase2_seed(20260720), 21260723)
        self.assertEqual(phase2_learning_rate(101), 1e-5)
        self.assertEqual(phase2_learning_rate(120), 1e-5)
        self.assertEqual(phase2_learning_rate(121), 1e-6)
        self.assertEqual(phase2_learning_rate(150), 1e-6)
        self.assertNotEqual(
            phase2_epoch_seed(20260720, 101),
            phase2_epoch_seed(20260720, 102),
        )
        self.assertEqual(
            phase2_epoch_seed(20260720, 101),
            phase2_epoch_seed(20260720, 101),
        )
        self.assertEqual(PHASE2_SCHEMA_VERSION, 2)
        self.assertTrue(PHASE2_PROTOCOL.endswith("_v2"))
        self.assertEqual(
            PHASE2_SOURCE_HASH_PROTOCOL,
            "utf8_source_newlines_lf_v1",
        )
        with self.assertRaises(ValueError):
            formal_phase2_seed(1)
        with self.assertRaises(ValueError):
            phase2_learning_rate(100)
        with self.assertRaises(ValueError):
            phase2_learning_rate(151)

    def test_convergence_gate_passes_a_plateau_away_from_boundary(self):
        parent = [
            _history_row(epoch, 2.0 - epoch * 0.01, 0.02 - epoch * 0.00005)
            for epoch in range(1, 101)
        ]
        phase = []
        for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1):
            if epoch <= 130:
                angle = 1.0 - (epoch - 100) * 0.01
                loss = 0.015 - (epoch - 100) * 0.0001
            else:
                angle = 0.705 + ((epoch % 3) - 1) * 0.002
                loss = 0.01205 + ((epoch % 2) * 0.00001)
            phase.append(_history_row(epoch, angle, loss))
        result = convergence_diagnostics(parent, phase)
        self.assertTrue(result["passed"])
        self.assertLess(result["best_epoch"], 146)
        self.assertTrue(all(result["checks"].values()))

    def test_convergence_gate_rejects_boundary_improvement(self):
        parent = [
            _history_row(epoch, 2.0 - epoch * 0.005, 0.03 - epoch * 0.00005)
            for epoch in range(1, 101)
        ]
        phase = [
            _history_row(
                epoch,
                1.5 - (epoch - 100) * 0.01,
                0.025 - (epoch - 100) * 0.0002,
            )
            for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1)
        ]
        result = convergence_diagnostics(parent, phase)
        self.assertFalse(result["passed"])
        self.assertEqual(result["best_epoch"], PHASE2_END_EPOCH)
        self.assertFalse(
            result["checks"]["best_not_in_final_boundary_window"]
        )

    def test_convergence_gate_rejects_catastrophic_regression(self):
        parent = [_history_row(epoch, 1.0, 1.0) for epoch in range(1, 101)]
        phase = []
        for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1):
            if epoch <= 130:
                angle, loss = 0.5, 0.5
            elif epoch <= 140:
                angle, loss = 1.0, 1.0
            else:
                angle, loss = 2.0, 2.0
            phase.append(_history_row(epoch, angle, loss))
        result = convergence_diagnostics(parent, phase)
        self.assertFalse(result["passed"])
        self.assertEqual(
            result["angle_mae"]["absolute_relative_change"],
            1.0,
        )
        self.assertFalse(
            result["checks"][
                "angle_absolute_relative_change_below_limit"
            ]
        )
        self.assertFalse(
            result["checks"][
                "loss_absolute_relative_change_below_limit"
            ]
        )

    def test_convergence_gate_rejects_stable_phase_regression_from_parent(self):
        parent = [
            _history_row(
                epoch,
                0.5 if epoch >= 91 else 1.0,
                0.5 if epoch >= 91 else 1.0,
            )
            for epoch in range(1, 101)
        ]
        phase = [
            _history_row(epoch, 2.0, 2.0)
            for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1)
        ]
        result = convergence_diagnostics(parent, phase)
        self.assertFalse(result["passed"])
        self.assertFalse(result["phase2_improved_over_parent"])
        self.assertFalse(
            result["checks"][
                "angle_parent_terminal_retention_within_limit"
            ]
        )
        self.assertFalse(
            result["checks"][
                "loss_parent_terminal_retention_within_limit"
            ]
        )
        self.assertEqual(
            result["angle_mae"][
                "final_vs_parent_terminal_relative_change"
            ],
            3.0,
        )

    def test_parent_retention_allows_improvement_and_inclusive_one_percent(self):
        parent = [
            _history_row(epoch, 1.0, 1.0)
            for epoch in range(1, 101)
        ]
        phase = [
            _history_row(epoch, 1.01, 1.01)
            for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1)
        ]
        result = convergence_diagnostics(parent, phase)
        self.assertTrue(
            result["checks"][
                "angle_parent_terminal_retention_within_limit"
            ]
        )
        self.assertTrue(
            result["checks"][
                "loss_parent_terminal_retention_within_limit"
            ]
        )
        improved = [
            _history_row(epoch, 0.5, 0.5)
            for epoch in range(PHASE2_START_EPOCH, PHASE2_END_EPOCH + 1)
        ]
        improved_result = convergence_diagnostics(parent, improved)
        self.assertTrue(
            improved_result["checks"][
                "angle_parent_terminal_retention_within_limit"
            ]
        )
        self.assertTrue(improved_result["phase2_improved_over_parent"])

    def test_actual_batch_order_hash_and_step_budget_are_exact(self):
        dataset = _TinyTrainingDataset(5)
        seed = 12345
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        model = _TinyTwoHeadModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        metrics = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device=torch.device("cpu"),
            amp_enabled=False,
            vector_weight=1.0,
            epoch=101,
        )
        metadata_samples = [
            SimpleNamespace(sample_id=sample_id)
            for sample_id in dataset.sample_ids
        ]
        expected_order = expected_phase2_sample_order_sha256(
            metadata_samples,
            batch_size=2,
            epoch_seed=seed,
        )
        self.assertEqual(metrics["sample_order_sha256"], expected_order)
        self.assertEqual(metrics["optimizer_steps"], 3)
        self.assertEqual(metrics["skipped_optimizer_steps"], 0)
        self.assertEqual(
            metrics["optimizer_steps"],
            expected_optimizer_steps(len(dataset), 2),
        )

    def test_phase_history_audits_success_skip_budget_and_scaler_trace(self):
        train_samples = [
            SimpleNamespace(sample_id=f"train-{index}")
            for index in range(80)
        ]
        parent_scaler = _scaler_state()
        lineage = SimpleNamespace(
            seed=20260720,
            train_samples=train_samples,
            last_checkpoint={"scaler_state": parent_scaler},
            summary={
                "best_validation_angle_mae_degrees": 2.0,
                "signature": {
                    "batch_size": 2,
                    "train_samples": 80,
                    "validation_samples": 1,
                }
            },
        )
        epoch_seed = phase2_epoch_seed(lineage.seed, 101)
        expected_signature = {
            "determinism": {
                "cuda_matmul_allow_fp16_reduced_precision_reduction": False,
                "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
            },
            "determinism_authorization_journal": {
                "protocol": "fixture",
                "report_sha256": "a" * 64,
            }
        }
        row = {
            "epoch": 101,
            "epoch_seed": epoch_seed,
            "learning_rate": phase2_learning_rate(101),
            "phase_elapsed_seconds": 1.0,
            "best": True,
            "determinism_authorization": expected_signature[
                "determinism_authorization_journal"
            ],
            "determinism_policy": expected_signature["determinism"],
            "train": {
                "loss": 1.0,
                "heatmap_loss": 0.5,
                "vector_loss": 0.5,
                "samples": 80,
                "vector_weight": 1.0,
                "optimizer_steps": 40,
                "skipped_optimizer_steps": 0,
                "sample_order_sha256": (
                    expected_phase2_sample_order_sha256(
                        train_samples,
                        batch_size=2,
                        epoch_seed=epoch_seed,
                    )
                ),
                "scaler_start_state": parent_scaler,
                "scaler_skipped_batch_indices": [],
                "scaler_end_state": _scaler_state(growth_tracker=40),
            },
            "validation": {
                "loss": 1.0,
                "heatmap_loss": 0.5,
                "vector_loss": 0.5,
                "angle_mae_degrees": 1.0,
                "angle_median_degrees": 1.0,
                "angle_acc_1deg": 1.0,
                "angle_acc_3deg": 1.0,
                "angle_acc_5deg": 1.0,
                "mean_heatmap_peak": 0.5,
                "samples": 1,
                "valid_directions": 1,
                "direction_coverage": 1.0,
            },
        }
        health = validate_phase_history(
            lineage,
            [row],
            through_epoch=101,
            expected_signature=expected_signature,
        )
        self.assertEqual(health["cumulative_optimizer_steps"], 40)
        self.assertEqual(health["cumulative_skipped_optimizer_steps"], 0)
        tampered = copy.deepcopy(row)
        tampered["train"]["optimizer_steps"] = 39
        tampered["train"]["skipped_optimizer_steps"] = 1
        with self.assertRaisesRegex(ValueError, "successful optimizer steps"):
            validate_phase_history(
                lineage,
                [tampered],
                through_epoch=101,
                expected_signature=expected_signature,
            )
        one_skip = copy.deepcopy(row)
        one_skip["train"]["optimizer_steps"] = 39
        one_skip["train"]["skipped_optimizer_steps"] = 1
        one_skip["train"]["scaler_skipped_batch_indices"] = [0]
        one_skip["train"]["scaler_end_state"] = _scaler_state(
            scale=256.0,
            growth_tracker=39,
        )
        skip_health = validate_phase_history(
            lineage,
            [one_skip],
            through_epoch=101,
            expected_signature=expected_signature,
        )
        self.assertEqual(skip_health["cumulative_optimizer_steps"], 39)
        self.assertEqual(
            skip_health["cumulative_skipped_optimizer_steps"],
            1,
        )
        over_budget = copy.deepcopy(one_skip)
        over_budget["train"]["optimizer_steps"] = 38
        over_budget["train"]["skipped_optimizer_steps"] = 2
        over_budget["train"]["scaler_skipped_batch_indices"] = [0, 1]
        over_budget["train"]["scaler_end_state"] = _scaler_state(
            scale=128.0,
            growth_tracker=38,
        )
        with self.assertRaisesRegex(ValueError, "full-run frozen budget"):
            validate_phase_history(
                lineage,
                [over_budget],
                through_epoch=101,
                expected_signature=expected_signature,
            )

    def test_adam_and_scaler_state_are_strictly_validated(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 1),
        )
        optimizer = build_phase2_adam_optimizer(
            model,
            learning_rate=1e-3,
        )
        loss = model(torch.ones(2, 2)).square().mean()
        loss.backward()
        optimizer.step()
        health = validate_adam_optimizer_state(
            model,
            optimizer.state_dict(),
            expected_step=1,
            expected_learning_rate=1e-3,
            expected_parameter_count=4,
            label="toy",
        )
        self.assertEqual(health["state_count"], 4)
        _live_optimizer_step_invariant(
            model,
            optimizer,
            expected_step=1,
            expected_learning_rate=1e-3,
            expected_parameter_count=4,
        )
        tampered = copy.deepcopy(optimizer.state_dict())
        first_state = tampered["state"][next(iter(tampered["state"]))]
        first_state["step"] = torch.tensor(2.0)
        with self.assertRaisesRegex(ValueError, "Adam steps"):
            validate_adam_optimizer_state(
                model,
                tampered,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )
        wrong_step_dtype = copy.deepcopy(optimizer.state_dict())
        first_state_id = next(iter(wrong_step_dtype["state"]))
        wrong_step_dtype["state"][first_state_id]["step"] = (
            wrong_step_dtype["state"][first_state_id]["step"].double()
        )
        with self.assertRaisesRegex(ValueError, "step tensor policy"):
            validate_adam_optimizer_state(
                model,
                wrong_step_dtype,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )

        group_mutations = {
            "maximize": True,
            "initial_lr": 2e-3,
            "foreach": False,
            "capturable": True,
            "differentiable": True,
            "fused": False,
            "decoupled_weight_decay": True,
        }
        for field, value in group_mutations.items():
            with self.subTest(group_field=field):
                tampered = copy.deepcopy(optimizer.state_dict())
                tampered["param_groups"][0][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_adam_optimizer_state(
                        model,
                        tampered,
                        expected_step=1,
                        expected_learning_rate=1e-3,
                        expected_parameter_count=4,
                        label="toy",
                    )

        reordered = copy.deepcopy(optimizer.state_dict())
        reordered["param_groups"][0]["params"].reverse()
        with self.assertRaisesRegex(ValueError, "layout/order"):
            validate_adam_optimizer_state(
                model,
                reordered,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )
        missing = copy.deepcopy(optimizer.state_dict())
        del missing["param_groups"][0]["foreach"]
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_adam_optimizer_state(
                model,
                missing,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )
        unknown = copy.deepcopy(optimizer.state_dict())
        unknown["param_groups"][0]["unknown_option"] = False
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_adam_optimizer_state(
                model,
                unknown,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )
        negative_second_moment = copy.deepcopy(optimizer.state_dict())
        first_state_id = next(iter(negative_second_moment["state"]))
        negative_second_moment["state"][first_state_id][
            "exp_avg_sq"
        ].fill_(-1.0)
        with self.assertRaisesRegex(ValueError, "exp_avg_sq is negative"):
            validate_adam_optimizer_state(
                model,
                negative_second_moment,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )
        wrong_moment_dtype = copy.deepcopy(optimizer.state_dict())
        first_state_id = next(iter(wrong_moment_dtype["state"]))
        wrong_moment_dtype["state"][first_state_id]["exp_avg"] = (
            wrong_moment_dtype["state"][first_state_id]["exp_avg"].double()
        )
        with self.assertRaisesRegex(ValueError, "exp_avg dtype mismatch"):
            validate_adam_optimizer_state(
                model,
                wrong_moment_dtype,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
                label="toy",
            )

        optimizer.param_groups[0]["maximize"] = True
        with self.assertRaisesRegex(RuntimeError, "maximize"):
            _live_optimizer_step_invariant(
                model,
                optimizer,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
            )
        optimizer.param_groups[0]["maximize"] = False
        live_parameters = optimizer.param_groups[0]["params"]
        live_parameters[0], live_parameters[1] = (
            live_parameters[1],
            live_parameters[0],
        )
        with self.assertRaisesRegex(RuntimeError, "layout/order"):
            _live_optimizer_step_invariant(
                model,
                optimizer,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
            )
        live_parameters[0], live_parameters[1] = (
            live_parameters[1],
            live_parameters[0],
        )
        first_parameter = next(iter(model.parameters()))
        live_moment = optimizer.state[first_parameter]["exp_avg"]
        live_moment.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "exp_avg is non-finite"):
            _live_optimizer_step_invariant(
                model,
                optimizer,
                expected_step=1,
                expected_learning_rate=1e-3,
                expected_parameter_count=4,
            )

        scaler_state = _scaler_state(growth_tracker=7)
        self.assertEqual(
            validate_scaler_state(scaler_state, label="toy")["scale"],
            512.0,
        )
        scaler_mutations = {
            "growth_factor": 3.0,
            "backoff_factor": 0.25,
            "growth_interval": 1000,
        }
        for field, value in scaler_mutations.items():
            with self.subTest(scaler_field=field):
                invalid_scaler = dict(scaler_state)
                invalid_scaler[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    validate_scaler_state(invalid_scaler, label="toy")
        invalid_scaler = dict(scaler_state)
        invalid_scaler["scale"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid numeric"):
            validate_scaler_state(invalid_scaler, label="toy")
        invalid_tracker = dict(scaler_state)
        invalid_tracker["_growth_tracker"] = 2000
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_scaler_state(invalid_tracker, label="toy")

        transition = validate_scaler_transition(
            scaler_state,
            _scaler_state(scale=256.0, growth_tracker=1),
            [1],
            attempted_steps=3,
            label="toy",
        )
        self.assertEqual(transition["successful_steps"], 2)
        self.assertEqual(transition["skipped_steps"], 1)
        for indices, message in (
            ([1, 1], "strictly increasing"),
            ([3], "out of range"),
        ):
            with self.subTest(indices=indices):
                with self.assertRaisesRegex(ValueError, message):
                    validate_scaler_transition(
                        scaler_state,
                        _scaler_state(scale=256.0, growth_tracker=1),
                        indices,
                        attempted_steps=3,
                        label="toy",
                    )
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            validate_scaler_transition(
                scaler_state,
                _scaler_state(scale=512.0, growth_tracker=1),
                [1],
                attempted_steps=3,
                label="toy",
            )

    def test_content_inventory_identity_binds_all_required_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "syncg_train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            protocol = manifest.with_name(manifest.name + ".protocol.json")
            protocol.write_text("{}\n", encoding="utf-8")
            tool_source = root / "inventory_tool.py"
            tool_source.write_text("# fixture\n", encoding="utf-8")
            verification = {
                "protocol": (
                    "vdn_phase2_syncg_train_content_inventory_verification_v1"
                ),
                "verified": True,
                "content_rehashed": True,
                "report_path": (
                    "artifacts/protocols/"
                    "vdn_phase2_syncg_train_content_inventory_v1.json"
                ),
                "inventory_report_sha256": "a" * 64,
                "canonical_inventory_sha256": "b" * 64,
                "canonical_report_payload_sha256": "c" * 64,
                "manifest_sha256": sha256_file(manifest),
                "manifest_protocol_sha256": sha256_file(protocol),
                "vdn_protocol_document_sha256": "d" * 64,
                "vdn_protocol_source_sha256": "e" * 64,
                "rows": 16_000,
            }
            identity = normalize_content_inventory_identity(
                verification,
                inventory_tool_source=tool_source,
                manifest=manifest,
            )
            self.assertEqual(identity["report_sha256"], "a" * 64)
            self.assertEqual(
                identity["canonical_inventory_sha256"],
                "b" * 64,
            )
            self.assertEqual(len(identity["inventory_tool_source_sha256"]), 64)
            self.assertTrue(identity["fresh_content_rehashed"])
            tampered = dict(verification)
            tampered["rows"] = 15_999
            with self.assertRaisesRegex(ValueError, "rows"):
                normalize_content_inventory_identity(
                    tampered,
                    inventory_tool_source=tool_source,
                    manifest=manifest,
                )

    def test_bootstrap_last_is_authoritative_and_recovers_derivatives(self):
        lineage = SimpleNamespace(
            summary={
                "best_epoch": 91,
                "best_validation_angle_mae_degrees": 0.7,
                "history": [],
            },
            last_checkpoint={
                "model_state": {"weight": torch.tensor([1.0])},
                "optimizer_state": {"state": {}, "param_groups": []},
                "scaler_state": {
                    "scale": 512.0,
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "_growth_tracker": 0,
                },
            },
            best_checkpoint={
                "model_state": {"weight": torch.tensor([0.7])},
            },
        )
        signature = {"protocol": PHASE2_PROTOCOL, "fixture": True}
        environment = {"fixture": "cpu-only"}
        state = _bootstrap_authoritative_state(
            lineage,
            signature=signature,
            environment=environment,
        )
        self.assertEqual(state["epoch"], 100)
        self.assertEqual(state["status"], "bootstrap")
        self.assertEqual(
            state["checkpoint_protocol"],
            PHASE2_CHECKPOINT_PROTOCOL,
        )
        self.assertTrue(
            torch.equal(
                state["current_model_state"]["weight"],
                torch.tensor([1.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                state["best_model_state"]["weight"],
                torch.tensor([0.7]),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase2"
            output.mkdir()
            torch.save(state, output / "last.pt")
            _prepare_output_dir(output, resume=True)
            _recover_derived_artifacts(output, state, lineage)
            (output / "best.pt").unlink()
            (output / "summary.json").unlink()
            recovered = torch.load(
                output / "last.pt",
                map_location="cpu",
                weights_only=False,
            )
            _recover_derived_artifacts(output, recovered, lineage)
            best = torch.load(
                output / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(best["authoritative_epoch"], 100)
            self.assertEqual(summary["authoritative_epoch"], 100)
            self.assertEqual(summary["status"], "running")

    def test_exclusive_writer_lock_covers_the_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase2"
            output.mkdir()
            with _exclusive_writer_lock(output):
                self.assertTrue((output / "writer.lock").is_file())
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    with _exclusive_writer_lock(output):
                        self.fail("a second writer unexpectedly acquired the lock")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "writer/verification lock",
                ):
                    with _exclusive_verification_lock(output):
                        self.fail("a verifier raced an active trainer")
            self.assertFalse((output / "writer.lock").exists())
            with _exclusive_verification_lock(output):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    with _exclusive_writer_lock(output):
                        self.fail("a trainer raced an active verifier")
            self.assertFalse((output / "writer.lock").exists())

    def test_new_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "phase2"
            _prepare_output_dir(root, resume=False)
            (root / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _prepare_output_dir(root, resume=False)
            with self.assertRaises(FileNotFoundError):
                _prepare_output_dir(root, resume=True)
            (root / "last.pt").write_bytes(b"checkpoint")
            _prepare_output_dir(root, resume=True)
            self.assertEqual(
                (root / "sentinel.txt").read_text(encoding="utf-8"),
                "keep",
            )

    def test_missing_prestart_policy_has_zero_output_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "phase2-output"
            args = SimpleNamespace(
                manifest=root / "manifest.jsonl",
                vdn_source=root / "vdn",
                content_inventory=root / "inventory.json",
                determinism_report=root / "determinism.json",
                output_dir=output,
                parent_run=root / "parent",
                resume=False,
            )
            authorization = {
                "scientific_identity": {
                    "environment": {
                        "policy": {
                            "cublas_workspace_config": ":4096:8",
                            "deterministic_algorithms": True,
                            "deterministic_warn_only": False,
                            "cudnn_deterministic": True,
                            "cudnn_benchmark": False,
                            "cuda_matmul_allow_tf32": False,
                            "cudnn_allow_tf32": False,
                            "float32_matmul_precision": "highest",
                            "pythonhashseed": "21260723",
                            (
                                "cuda_matmul_allow_fp16_"
                                "reduced_precision_reduction"
                            ): False,
                            (
                                "cuda_matmul_allow_bf16_"
                                "reduced_precision_reduction"
                            ): False,
                        }
                    }
                }
            }
            with (
                mock.patch(
                    "experiments.train_vdn_phase2.parse_args",
                    return_value=args,
                ),
                mock.patch(
                    "experiments.train_vdn_phase2."
                    "validate_determinism_authorization_report",
                    return_value=authorization,
                ),
                mock.patch(
                    "experiments.train_vdn_phase2._prepare_output_dir"
                ) as prepare,
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "PYTHONHASHSEED"),
            ):
                train_vdn_phase2_main()
            prepare.assert_not_called()
            self.assertFalse(output.exists())

    def test_formal_commands_set_prestart_policy_and_pass_report(self):
        project_root = Path(__file__).resolve().parents[1]
        document = (
            project_root / "docs" / "VDN_CONVERGENCE_PROTOCOL_CN.md"
        ).read_text(encoding="utf-8")
        supervisor = (
            project_root / "experiments" / "run_vdn_phase2.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$env:PYTHONHASHSEED = '21260723'", document)
        self.assertIn(
            "$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'",
            document,
        )
        self.assertGreaterEqual(document.count("--determinism-report"), 3)
        self.assertGreaterEqual(document.count("--content-inventory"), 4)
        self.assertIn(
            "vdn_phase2_determinism_probe_v2.json",
            document,
        )
        hash_index = supervisor.index("$env:PYTHONHASHSEED")
        cublas_index = supervisor.index("$env:CUBLAS_WORKSPACE_CONFIG")
        launch_index = supervisor.index("& $PythonPath")
        self.assertLess(hash_index, launch_index)
        self.assertLess(cublas_index, launch_index)
        self.assertIn("'--determinism-report'", supervisor)

    def test_verification_report_is_atomic_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "verification-v2.json"
            write_json_no_clobber({"verified": True}, output)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json_no_clobber({"verified": False}, output)
            self.assertEqual(output.read_bytes(), original)

            failed_output = root / "failed.json"
            with self.assertRaises(ValueError):
                write_json_no_clobber({"metric": float("nan")}, failed_output)
            self.assertFalse(failed_output.exists())

    def test_parent_lineage_binds_summary_and_checkpoint_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "parent"
            run.mkdir()
            manifest = root / "syncg_train.jsonl"
            manifest.write_text(
                '{"sample_id":"placeholder"}\n',
                encoding="utf-8",
            )
            manifest_protocol_path = manifest.with_name(
                manifest.name + ".protocol.json"
            )
            manifest_protocol_path.write_text(
                '{"dataset":"SyncG","split":"train"}\n',
                encoding="utf-8",
            )
            train = [
                SimpleNamespace(sample_id="train-a", group_id="group-a"),
            ]
            validation = [
                SimpleNamespace(sample_id="val-a", group_id="group-b"),
            ]
            signature = {
                "protocol": VDN_PROTOCOL,
                "vdn_source_commit": VDN_PINNED_COMMIT,
                "epochs": 100,
                "seed": 20260720,
                "weight_decay": 0.0,
                "batch_size": 8,
                "mixed_precision": True,
                "validation_fraction": 0.5,
                "manifest_sha256": sha256_file(manifest),
                "manifest_protocol_sha256": sha256_file(
                    manifest_protocol_path
                ),
                "train_samples": len(train),
                "validation_samples": len(validation),
                "train_sample_ids_sha256": sample_ids_hash(train),
                "validation_sample_ids_sha256": sample_ids_hash(validation),
            }
            summary = {
                "status": "complete",
                "signature": signature,
                "best_epoch": 99,
                "best_validation_angle_mae_degrees": 0.7,
                "history": [],
            }
            summary_path = run / "summary.json"
            summary_path.write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            last_path = run / "last.pt"
            best_path = run / "best.pt"
            torch.save(
                {
                    "signature": signature,
                    "epoch": 100,
                    "model_state": {},
                },
                last_path,
            )
            torch.save(
                {
                    "signature": signature,
                    "epoch": 99,
                    "model_state": {},
                },
                best_path,
            )
            verification = {
                "verified": True,
                "epochs": 100,
                "best_epoch": 99,
                "summary_sha256": sha256_file(summary_path),
                "last_checkpoint_sha256": sha256_file(last_path),
                "best_checkpoint_sha256": sha256_file(best_path),
            }
            verification_path = run / "verification.json"
            verification_path.write_text(
                json.dumps(verification),
                encoding="utf-8",
            )
            manifest_protocol = {
                "dataset": "SyncG",
                "split": "train",
            }
            with (
                mock.patch(
                    "experiments.vdn_phase2_protocol.load_syncg_manifest",
                    return_value=([*train, *validation], manifest_protocol),
                ) as loader,
                mock.patch(
                    "experiments.vdn_phase2_protocol.grouped_train_val_split",
                    return_value=(train, validation),
                ),
            ):
                lineage = load_parent_lineage(
                    run,
                    manifest=manifest,
                    load_checkpoints=False,
                )
                self.assertEqual(lineage.seed, 20260720)
                loader.assert_called_once_with(
                    manifest.resolve(),
                    expected_split="train",
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "continuation state",
                ):
                    load_parent_lineage(
                        run,
                        manifest=manifest,
                        load_checkpoints=True,
                    )

                last_path.write_bytes(b"mutated")
                with self.assertRaisesRegex(
                    ValueError,
                    "last checkpoint SHA-256",
                ):
                    load_parent_lineage(
                        run,
                        manifest=manifest,
                        load_checkpoints=False,
                    )


if __name__ == "__main__":
    unittest.main()
