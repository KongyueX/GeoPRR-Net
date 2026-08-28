"""Deterministic tests for the external VDN baseline adapter."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments import evaluate_vdn_baseline as evaluate_vdn
from experiments.evaluate_vdn_baseline import (
    _authorize_checkpoint_for_evaluation,
    _authorize_public_image_inventory,
    _dialbench_summary,
    _exclusive_output_lock,
    _load_shared_predictions,
    _PredictionJournal,
    _preauthorize_checkpoint,
    _safe_load_authorized_checkpoint,
    _snapshot_phase2_inputs,
    _validate_output_does_not_alias_inputs,
)
from experiments.robustness_degradations import ROBUSTNESS_PROTOCOL
from experiments.vdn_baseline import (
    PROJECT_DIR,
    SOURCE_TEXT_SHA256_PROTOCOL,
    VDNSample,
    affine_for_dial,
    angular_error_degrees,
    generate_vdn_targets,
    grouped_train_val_split,
    image_angle_from_direction,
    normalize_reference_points,
    predict_directions,
    reference_angles,
    reading_from_pointer_angle,
    sha256_file,
    sha256_source_file,
    summarize_scalar_predictions,
    transform_point,
)
from experiments.summarize_vdn_comparison import _paired_comparison
from experiments.vdn_phase2_protocol import (
    PHASE2_CHECKPOINT_PROTOCOL,
    PHASE2_PROTOCOL,
)
from experiments.verify_vdn_run import _state_health


def _write_unsafe_sentinel(path: str) -> dict:
    Path(path).write_text("executed", encoding="utf-8")
    return {}


class _UnsafeCheckpoint:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return _write_unsafe_sentinel, (str(self.sentinel),)


def _sample(sample_id: str, group_id: str) -> VDNSample:
    return VDNSample(
        sample_id=sample_id,
        group_id=group_id,
        dataset="SyncG",
        split="train",
        image_path="unused.jpg",
        dial_bbox=(10.0, 20.0, 110.0, 100.0),
        pointer_tip=(70.0, 35.0),
        pointer_tail=(60.0, 60.0),
        ground_truth=0.5,
        scale_start=0.0,
        scale_end=1.0,
        metadata={},
    )


class VDNBaselineTest(unittest.TestCase):
    def test_public_image_inventory_binds_bytes_and_rejects_forbidden_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "b.jpg"
            first.write_bytes(b"first-image")
            second.write_bytes(b"second-image")
            rows = [
                {
                    "sample_id": "b",
                    "image_path": str(second),
                    "ground_truth": 0.2,
                },
                {
                    "sample_id": "a",
                    "image_path": str(first),
                    "ground_truth": 0.1,
                },
            ]
            records = [
                {
                    "sample_id": "a",
                    "image_name": "a.jpg",
                    "image_sha256": hashlib.sha256(
                        b"first-image"
                    ).hexdigest(),
                    "portable_manifest_row": {
                        "ground_truth": 0.1,
                        "sample_id": "a",
                    },
                },
                {
                    "sample_id": "b",
                    "image_name": "b.jpg",
                    "image_sha256": hashlib.sha256(
                        b"second-image"
                    ).hexdigest(),
                    "portable_manifest_row": {
                        "ground_truth": 0.2,
                        "sample_id": "b",
                    },
                },
            ]
            expected = hashlib.sha256(
                json.dumps(
                    records,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            with mock.patch.dict(
                evaluate_vdn.PINNED_PUBLIC_RELEASE_INVENTORY_SHA256,
                {"fixture": expected},
                clear=True,
            ):
                digest, image_hashes = _authorize_public_image_inventory(
                    rows,
                    public_scope="fixture",
                )
                self.assertEqual(digest, expected)
                self.assertEqual(
                    image_hashes["a"],
                    records[0]["image_sha256"],
                )
                first.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "differs"):
                    _authorize_public_image_inventory(
                        rows,
                        public_scope="fixture",
                    )

            forbidden = root / "field" / "image.jpg"
            forbidden.parent.mkdir()
            forbidden.write_bytes(b"must-not-open")
            with (
                mock.patch.dict(
                    evaluate_vdn.PINNED_PUBLIC_RELEASE_INVENTORY_SHA256,
                    {"fixture": "0" * 64},
                    clear=True,
                ),
                mock.patch.object(
                    evaluate_vdn,
                    "sha256_file",
                ) as hasher,
            ):
                with self.assertRaises(PermissionError):
                    _authorize_public_image_inventory(
                        [{"sample_id": "x", "image_path": str(forbidden)}],
                        public_scope="fixture",
                    )
            hasher.assert_not_called()

    def test_formal_legacy_root_is_project_anchored_across_cwd(self):
        expected = evaluate_vdn.DEFAULT_LEGACY_RUN_ROOT
        self.assertTrue(expected.is_absolute())
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(
                    evaluate_vdn.DEFAULT_LEGACY_RUN_ROOT.resolve(),
                    expected,
                )
            finally:
                os.chdir(previous)

    def test_evaluator_import_cannot_execute_cwd_pointget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            (root / "pointGet.py").write_text(
                (
                    "from pathlib import Path\n"
                    f"Path({str(sentinel)!r}).write_text('executed')\n"
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                [
                    str(PROJECT_DIR),
                    environment.get("PYTHONPATH", ""),
                ]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import experiments.evaluate_vdn_baseline as module; "
                        "assert 'utils.angleDetect.yoloDetection."
                        "yoloDectect' not in __import__('sys').modules"
                    ),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + result.stderr,
            )
            self.assertFalse(sentinel.exists())

    def test_detector_loader_rejects_non_distribution_ultralytics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            package = root / "ultralytics"
            package.mkdir()
            (package / "__init__.py").write_text(
                (
                    "from pathlib import Path\n"
                    f"Path({str(sentinel)!r}).write_text('executed')\n"
                    "class YOLO: pass\n"
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(PROJECT_DIR)
            script = (
                "from experiments.evaluate_vdn_baseline import "
                "_load_target_detector_class\n"
                "try:\n"
                "    _load_target_detector_class()\n"
                "except ImportError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('unexpected detector import')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=result.stdout + result.stderr,
            )
            self.assertFalse(sentinel.exists())

    def test_output_cannot_alias_any_read_only_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (root / "best.pt").resolve()
            args = Namespace(
                output=checkpoint,
                manifest=(root / "manifest.jsonl").resolve(),
                checkpoint=checkpoint,
                meter_detector_weights=(root / "meter.pt").resolve(),
                keypoint_detector_weights=(root / "point.pt").resolve(),
                shared_predictions=None,
                phase2_cohort_authorization=None,
                legacy_training_verification=(root / "verification.json").resolve(),
            )
            with self.assertRaisesRegex(PermissionError, "aliases"):
                _validate_output_does_not_alias_inputs(args)

    def test_main_authorizes_plan_before_creating_writer_lock(self):
        args = Namespace(output=Path("field-output.jsonl"))
        with (
            mock.patch.object(evaluate_vdn, "parse_args", return_value=args),
            mock.patch.object(
                evaluate_vdn,
                "_prepare_evaluation",
                side_effect=PermissionError("plan rejected"),
            ),
            mock.patch.object(
                evaluate_vdn,
                "_exclusive_output_lock",
            ) as writer_lock,
        ):
            with self.assertRaisesRegex(PermissionError, "plan rejected"):
                evaluate_vdn.main()
        writer_lock.assert_not_called()

    def test_phase2_checkpoint_requires_cohort_authorization(self):
        with self.assertRaisesRegex(PermissionError, "exactly one"):
            _preauthorize_checkpoint(
                checkpoint_path=Path("best.pt"),
                phase2_cohort_authorization=None,
                legacy_training_verification=None,
            )

    def test_external_authorization_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(PermissionError, "exactly one"):
            _preauthorize_checkpoint(
                checkpoint_path=Path("best.pt"),
                phase2_cohort_authorization=Path("cohort.json"),
                legacy_training_verification=Path("verification.json"),
            )

    def test_phase2_preauthorization_uses_external_cohort(self):
        expected = {
            "training_protocol": PHASE2_PROTOCOL,
            "checkpoint_sha256": "a" * 64,
        }
        with mock.patch.object(
            evaluate_vdn,
            "validate_cohort_evaluation_authorization",
            return_value=expected,
        ) as validator:
            result = _preauthorize_checkpoint(
                checkpoint_path=Path("best.pt"),
                phase2_cohort_authorization=Path("cohort.json"),
                legacy_training_verification=None,
            )
        self.assertEqual(result, expected)
        validator.assert_called_once_with(
            Path("cohort.json"),
            Path("best.pt"),
        )

    def test_phase2_checkpoint_consumes_exact_cohort_authorization(self):
        checkpoint = {
            "checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
            "signature": {"protocol": PHASE2_PROTOCOL},
        }
        expected = {
            "protocol": "fixture",
            "training_protocol": PHASE2_PROTOCOL,
        }
        protocol, authorization = _authorize_checkpoint_for_evaluation(
            checkpoint,
            external_authorization=expected,
        )
        self.assertEqual(protocol, PHASE2_PROTOCOL)
        self.assertEqual(authorization, expected)

    def test_phase2_missing_preflight_fails_before_public_plan_reads(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"fixture")
            cohort = root / "cohort.json"
            args = Namespace(
                manifest=root / "public.jsonl",
                checkpoint=checkpoint_path,
                phase2_cohort_authorization=cohort,
                phase2_public_preflight=None,
                legacy_training_verification=None,
                vdn_source=root / "vdn",
                shared_predictions=root / "shared.jsonl",
                output=root / "evaluations" / "clean.jsonl",
                device="cuda",
                batch_size=16,
                condition="clean",
                degradation_seed=20260720,
                bootstrap_iterations=2000,
                seed=20260724,
                limit=None,
                resume=False,
                overwrite=False,
                no_amp=False,
                meter_detector_weights=root / "meter.pt",
                keypoint_detector_weights=root / "point.pt",
            )
            authorization = {
                "training_protocol": PHASE2_PROTOCOL,
                "checkpoint_sha256": "a" * 64,
            }
            checkpoint = {
                "checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
                "signature": {
                    "protocol": PHASE2_PROTOCOL,
                    "vdn_source_commit": "fixture-commit",
                },
            }
            with (
                mock.patch.object(
                    evaluate_vdn,
                    "_preauthorize_checkpoint",
                    return_value=authorization,
                ),
                mock.patch.object(
                    evaluate_vdn,
                    "_safe_load_authorized_checkpoint",
                    return_value=checkpoint,
                ),
                mock.patch.object(
                    evaluate_vdn,
                    "verify_vdn_source",
                    return_value="fixture-commit",
                ),
                mock.patch.object(
                    evaluate_vdn,
                    "validate_phase2_evaluation_plan",
                ) as plan_validator,
                mock.patch.object(
                    evaluate_vdn,
                    "_snapshot_phase2_inputs",
                ) as snapshotter,
            ):
                with self.assertRaisesRegex(
                    PermissionError,
                    "public preflight",
                ):
                    evaluate_vdn._prepare_evaluation(args)
            plan_validator.assert_not_called()
            snapshotter.assert_not_called()

    def test_checkpoint_self_label_cannot_override_external_authorization(self):
        checkpoint = {
            "signature": {"protocol": evaluate_vdn.VDN_PROTOCOL}
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _authorize_checkpoint_for_evaluation(
                checkpoint,
                external_authorization={
                    "training_protocol": PHASE2_PROTOCOL,
                },
            )

    def test_phase2_main_fails_before_opening_evaluation_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "best.pt"
            checkpoint_path.write_bytes(b"fixture")
            args = Namespace(
                manifest=root / "test.jsonl",
                checkpoint=checkpoint_path,
                phase2_cohort_authorization=None,
                legacy_training_verification=None,
                vdn_source=root / "vdn",
                shared_predictions=None,
                output=root / "predictions.jsonl",
                device="cpu",
                batch_size=1,
                condition="clean",
                degradation_seed=20260720,
                bootstrap_iterations=0,
                seed=20260720,
                limit=None,
                resume=False,
                overwrite=False,
                no_amp=True,
                meter_detector_weights=root / "meter.pt",
                keypoint_detector_weights=root / "point.pt",
            )
            checkpoint = {
                "checkpoint_protocol": PHASE2_CHECKPOINT_PROTOCOL,
                "signature": {"protocol": PHASE2_PROTOCOL},
            }
            with (
                mock.patch.object(evaluate_vdn, "parse_args", return_value=args),
                mock.patch.object(
                    evaluate_vdn.torch,
                    "load",
                    return_value=checkpoint,
                ),
                mock.patch.object(evaluate_vdn, "_read_jsonl") as reader,
            ):
                with self.assertRaisesRegex(
                    PermissionError,
                    "exactly one",
                ):
                    evaluate_vdn.main()
            reader.assert_not_called()

    def test_authorized_checkpoint_loader_never_executes_pickle_globals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "unsafe.pt"
            sentinel = root / "sentinel.txt"
            torch.save(
                {"payload": _UnsafeCheckpoint(sentinel)},
                checkpoint,
            )
            authorization = {
                "checkpoint_sha256": evaluate_vdn.sha256_file(checkpoint),
            }
            with self.assertRaises(Exception):
                _safe_load_authorized_checkpoint(
                    checkpoint,
                    authorization,
                )
            self.assertFalse(sentinel.exists())

    def test_phase2_input_snapshot_consumes_only_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = {}
            expected = {}
            for name in (
                "manifest",
                "manifest_protocol",
                "shared_predictions",
                "shared_predictions_metadata",
                "meter_detector_weights",
                "keypoint_detector_weights",
            ):
                path = root / name
                payload = f"authorized-{name}".encode("utf-8")
                path.write_bytes(payload)
                expected[name] = payload
                identities[f"{name}_path"] = str(path)
                identities[f"{name}_sha256"] = sha256_file(path)
            snapshot = _snapshot_phase2_inputs(identities)
            self.assertEqual(snapshot, expected)
            (root / "shared_predictions").write_bytes(b"replaced")
            self.assertEqual(
                snapshot["shared_predictions"],
                expected["shared_predictions"],
            )
            with self.assertRaisesRegex(ValueError, "changed"):
                _snapshot_phase2_inputs(identities)

    def test_output_writer_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with _exclusive_output_lock(output):
                with self.assertRaisesRegex(RuntimeError, "owns the output lock"):
                    with _exclusive_output_lock(output):
                        pass

    def test_prediction_journal_detects_same_id_content_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            journal_path = root / "predictions.jsonl.journal.jsonl"
            output.touch()
            journal_path.touch()
            writer = _PredictionJournal(
                output,
                journal_path,
                resume=False,
            )
            writer.append(
                [
                    {"sample_id": "a", "prediction": 0.1},
                    {"sample_id": "b", "prediction": 0.2},
                ]
            )
            _PredictionJournal(output, journal_path, resume=True)
            original = output.read_bytes()
            output.write_bytes(original.replace(b"0.1", b"0.9"))
            with self.assertRaisesRegex(ValueError, "prefix hash"):
                _PredictionJournal(output, journal_path, resume=True)

    def test_prediction_journal_rejects_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            journal_path = root / "predictions.jsonl.journal.jsonl"
            output.touch()
            journal_path.touch()
            writer = _PredictionJournal(
                output,
                journal_path,
                resume=False,
            )
            writer.append([{"sample_id": "a", "prediction": 0.1}])
            with output.open("a", encoding="utf-8") as handle:
                handle.write('{"sample_id":"b","prediction":0.2}\n')
            with self.assertRaisesRegex(ValueError, "not committed"):
                _PredictionJournal(output, journal_path, resume=True)

    def test_prediction_journal_recovers_only_uncommitted_tails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "predictions.jsonl"
            journal_path = root / "predictions.jsonl.journal.jsonl"
            output.touch()
            journal_path.touch()
            writer = _PredictionJournal(
                output,
                journal_path,
                resume=False,
            )
            writer.append([{"sample_id": "a", "prediction": 0.1}])
            committed_output = output.read_bytes()
            committed_journal = journal_path.read_bytes()
            with output.open("ab") as handle:
                handle.write(b'{"sample_id":"b","prediction":0.2}\n')
            with journal_path.open("ab") as handle:
                handle.write(b'{"protocol":"partial')

            recovered = _PredictionJournal(
                output,
                journal_path,
                resume=True,
                recover_uncommitted_tail=True,
            )

            self.assertEqual(output.read_bytes(), committed_output)
            self.assertEqual(journal_path.read_bytes(), committed_journal)
            self.assertGreater(recovered.recovered_output_tail_bytes, 0)
            self.assertGreater(recovered.recovered_journal_tail_bytes, 0)
            self.assertEqual(recovered.total_rows, 1)

    def test_source_hash_canonicalizes_only_newline_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf_source = root / "lf.py"
            crlf_source = root / "crlf.py"
            data = root / "weights.pt"
            lf_source.write_bytes(b"first = 1\nsecond = 2\n")
            crlf_source.write_bytes(b"first = 1\r\nsecond = 2\r\n")
            data.write_bytes(b"first = 1\r\nsecond = 2\r\n")

            self.assertEqual(
                sha256_source_file(lf_source),
                sha256_source_file(crlf_source),
            )
            self.assertNotEqual(sha256_file(lf_source), sha256_file(crlf_source))
            with self.assertRaises(ValueError):
                sha256_source_file(data)
            self.assertEqual(
                SOURCE_TEXT_SHA256_PROTOCOL,
                "utf8_source_newlines_lf_v1",
            )

    def test_affine_centers_square_crop_and_rotates_points(self):
        matrix = affine_for_dial(
            (10.0, 20.0, 110.0, 100.0),
            output_size=200,
            expansion=1.0,
            rotation_degrees=90.0,
        )
        center = transform_point((60.0, 60.0), matrix)
        right = transform_point((110.0, 60.0), matrix)
        np.testing.assert_allclose(center, (100.0, 100.0), atol=1e-5)
        np.testing.assert_allclose(right, (100.0, 0.0), atol=1e-5)

    def test_targets_match_tip_peak_and_tail_to_tip_direction(self):
        heatmap, vector_map, direction = generate_vdn_targets(
            (256.0, 192.0),
            (192.0, 192.0),
            image_size=384,
            heatmap_size=96,
        )
        self.assertEqual(tuple(heatmap.shape), (1, 96, 96))
        self.assertEqual(tuple(vector_map.shape), (2, 96, 96))
        self.assertAlmostEqual(float(heatmap[0, 48, 64]), 1.0)
        torch.testing.assert_close(direction, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(
            vector_map[:, 48, 64],
            torch.tensor([1.0, 0.0]),
        )

    def test_targets_match_pinned_official_code_when_available(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vendor"
            / "VectorDetectionNetwork"
        )
        dataset_source = source / "libs" / "dataset" / "JointsDataset.py"
        if not dataset_source.is_file():
            self.skipTest("ignored external VDN checkout is unavailable")
        sys.path.insert(0, str(source))
        try:
            spec = importlib.util.spec_from_file_location(
                "external_vdn_joints_dataset_test",
                dataset_source,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(source))
        official = object.__new__(module.JointsDataset)
        official.num_joints = 1
        official.heatmap_size = np.asarray([96, 96])
        official.image_size = np.asarray([384, 384])
        official.sigma = 3
        official.target_type = "gaussian"
        tip = (12.0, 20.0)
        tail = (180.0, 180.0)
        joints = np.asarray(
            [[[*tip, *tail, 1.0]]],
            dtype=np.float32,
        )
        official_heatmap, official_vectors = official.generate_target(joints)
        heatmap, vectors, _ = generate_vdn_targets(
            tip,
            tail,
            image_size=384,
            heatmap_size=96,
        )
        np.testing.assert_array_equal(official_heatmap, heatmap.numpy())
        np.testing.assert_array_equal(official_vectors.squeeze(0), vectors.numpy())
        official_affine = module.get_affine_transform(
            np.asarray([60.0, 60.0], dtype=np.float32),
            np.asarray([0.625, 0.625], dtype=np.float32),
            30.0,
            np.asarray([384, 384]),
        )
        local_affine = affine_for_dial(
            (10.0, 20.0, 110.0, 100.0),
            output_size=384,
            rotation_degrees=30.0,
        )
        np.testing.assert_allclose(local_affine, official_affine, atol=1e-5)

    def test_pinned_official_adam_does_not_apply_yaml_weight_decay(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "vendor"
            / "VectorDetectionNetwork"
            / "libs"
            / "utils"
            / "utils.py"
        )
        if not source.is_file():
            self.skipTest("ignored external VDN checkout is unavailable")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        optimizer_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_optimizer"
        )
        adam_calls = [
            node
            for node in ast.walk(optimizer_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Adam"
        ]
        self.assertEqual(len(adam_calls), 1)
        self.assertNotIn(
            "weight_decay",
            {keyword.arg for keyword in adam_calls[0].keywords},
        )

    def test_prediction_samples_vector_at_heatmap_peak(self):
        heatmap = torch.zeros(2, 1, 4, 4)
        vectors = torch.zeros(2, 2, 4, 4)
        heatmap[0, 0, 1, 2] = 0.9
        vectors[0, :, 1, 2] = torch.tensor([3.0, 4.0])
        heatmap[1, 0, 3, 0] = 0.8
        directions, confidence, valid = predict_directions(heatmap, vectors)
        torch.testing.assert_close(directions[0], torch.tensor([0.6, 0.8]))
        torch.testing.assert_close(confidence, torch.tensor([0.9, 0.8]))
        self.assertEqual(valid.tolist(), [True, False])
        error = angular_error_degrees(
            directions[:1],
            torch.tensor([[0.6, 0.8]]),
        )
        self.assertAlmostEqual(float(error[0]), 0.0, places=4)

    def test_image_angle_and_reading_use_production_convention(self):
        self.assertAlmostEqual(image_angle_from_direction((0.0, 1.0)), 0.0)
        self.assertAlmostEqual(image_angle_from_direction((-1.0, 0.0)), 90.0)
        self.assertAlmostEqual(image_angle_from_direction((0.0, -1.0)), 180.0)
        self.assertAlmostEqual(image_angle_from_direction((1.0, 0.0)), 270.0)
        reading, progress = reading_from_pointer_angle(
            270.0,
            start_angle=180.0,
            range_angle=270.0,
            scale_start=0.0,
            scale_end=60.0,
        )
        self.assertAlmostEqual(progress, 1.0 / 3.0)
        self.assertAlmostEqual(reading, 20.0)

    def test_reference_points_follow_production_fallback_branches(self):
        start, end = normalize_reference_points(
            (20.0, 80.0),
            (22.0, 81.0),
            image_width=100,
        )
        self.assertEqual(start, (20.0, 80.0))
        self.assertIsNone(end)
        start_angle, range_angle, branch = reference_angles(
            (100, 100, 3),
            start,
            end,
        )
        self.assertEqual(branch, "start_only")
        self.assertAlmostEqual(range_angle, 270.0)
        self.assertTrue(0.0 <= start_angle < 360.0)

        start, end = normalize_reference_points(
            (80.0, 80.0),
            (20.0, 80.0),
            image_width=100,
        )
        self.assertEqual(start, (20.0, 80.0))
        self.assertEqual(end, (80.0, 80.0))
        _, range_angle, branch = reference_angles((100, 100, 3), start, end)
        self.assertEqual(branch, "start_and_end")
        self.assertAlmostEqual(range_angle, 270.0)

    def test_grouped_split_is_reproducible_without_leakage(self):
        samples = [
            _sample(f"sample-{group}-{index}", f"group-{group}")
            for group in range(10)
            for index in range(3)
        ]
        train_a, validation_a = grouped_train_val_split(
            samples,
            validation_fraction=0.2,
            seed=17,
        )
        train_b, validation_b = grouped_train_val_split(
            samples,
            validation_fraction=0.2,
            seed=17,
        )
        self.assertEqual(
            [sample.sample_id for sample in validation_a],
            [sample.sample_id for sample in validation_b],
        )
        self.assertEqual(len(train_a), 24)
        self.assertEqual(len(validation_a), 6)
        self.assertTrue(
            {sample.group_id for sample in train_a}.isdisjoint(
                {sample.group_id for sample in validation_a}
            )
        )

    def test_scalar_summary_penalizes_inference_failures(self):
        rows = [
            {
                "prediction": 0.51,
                "ground_truth": 0.50,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "a",
            },
            {
                "prediction": None,
                "ground_truth": 0.50,
                "scale_start": 0.0,
                "scale_end": 1.0,
                "group_id": "b",
            },
        ]
        summary = summarize_scalar_predictions(
            rows,
            bootstrap_iterations=20,
            seed=3,
        )
        self.assertAlmostEqual(summary["coverage"], 0.5)
        self.assertAlmostEqual(summary["nmae"], 0.505)
        self.assertAlmostEqual(summary["acc_2pct"], 0.5)
        self.assertIsNotNone(summary["nmae_group_bootstrap_95ci"])

    def test_dialbench_accuracy_keeps_failures_in_full_denominator(self):
        rows = [
            {
                "prediction": 10.4,
                "ground_truth": 10.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
            {
                "prediction": None,
                "ground_truth": 10.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
            {
                "prediction": 0.0,
                "ground_truth": 0.0,
                "scale_start": 0.0,
                "scale_end": 100.0,
            },
        ]
        summary = _dialbench_summary(rows)
        self.assertEqual(summary["eligible_samples"], 3)
        self.assertEqual(summary["relative_samples"], 2)
        self.assertAlmostEqual(summary["ref_successful"], 0.002)
        self.assertAlmostEqual(
            summary["acc_epsilon_ref_le_1pct_e2e"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(summary["acc_theta_rel_lt_5pct_e2e"], 0.5)

    def test_checkpoint_health_rejects_collapsed_backbone(self):
        state = {
            "conv1.weight": torch.randn(8, 3, 3, 3) * 0.1,
            "layer1.0.conv1.weight": torch.randn(8, 8, 3, 3) * 0.05,
            "deconv_layers.0.weight": torch.randn(8, 8, 3, 3) * 0.001,
            "final_layer_hm.weight": torch.randn(1, 8, 1, 1) * 0.001,
            "final_layer_v.weight": torch.randn(2, 8, 3, 3) * 0.001,
        }
        health = _state_health(state)
        self.assertEqual(health["non_finite_tensors"], 0)
        state["conv1.weight"].zero_()
        with self.assertRaisesRegex(ValueError, "conv1 collapsed"):
            _state_health(state)

    def test_external_paired_comparison_keeps_failures_in_denominator(self):
        base = {
            "ground_truth": 0.5,
            "scale_start": 0.0,
            "scale_end": 1.0,
        }
        vdn = [
            {**base, "sample_id": "a", "group_id": "g1", "prediction": 0.6},
            {**base, "sample_id": "b", "group_id": "g2", "prediction": None},
        ]
        ours = [
            {
                **base,
                "sample_id": "a",
                "group_id": "g1",
                "predictions": {"ours": 0.55},
            },
            {
                **base,
                "sample_id": "b",
                "group_id": "g2",
                "predictions": {"ours": 0.7},
            },
        ]
        paired = _paired_comparison(vdn, ours, iterations=20, seed=4)
        self.assertAlmostEqual(paired["raw_vdn_nmae"], 0.55)
        self.assertAlmostEqual(paired["raw_ours_nmae"], 0.125)
        self.assertAlmostEqual(paired["delta_nmae_ours_minus_vdn"], -0.425)
        self.assertEqual(paired["common_successes"], 1)

    def test_shared_cache_requires_identical_degradation_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            protocol = manifest.with_name(manifest.name + ".protocol.json")
            protocol.write_text('{"protocol":"test"}\n', encoding="utf-8")
            shared = root / "shared.jsonl"
            shared.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            meter_weights = root / "meter.pt"
            point_weights = root / "point.pt"
            meter_weights.write_bytes(b"meter")
            point_weights.write_bytes(b"point")
            signature = {
                "manifest_sha256": sha256_file(manifest),
                "manifest_protocol_sha256": sha256_file(protocol),
                "correction_mode": "off",
                "input_degradation": {
                    "condition": "clean",
                    "protocol": ROBUSTNESS_PROTOCOL,
                    "seed": 20260720,
                },
                "input_degradation_source_sha256": sha256_file(
                    PROJECT_DIR / "experiments" / "robustness_degradations.py"
                ),
                "weights_sha256": {
                    "meter_detector": sha256_file(meter_weights),
                    "keypoint_detector": sha256_file(point_weights),
                },
            }
            metadata = shared.with_name(shared.name + ".meta.json")
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            by_id, _ = _load_shared_predictions(
                shared,
                manifest=manifest,
                rows=[{"sample_id": "a"}],
                condition="clean",
                degradation_seed=20260720,
                meter_weights=meter_weights,
                point_weights=point_weights,
            )
            self.assertEqual(set(by_id), {"a"})

            authorized_inputs = {
                "shared_predictions": shared.read_bytes(),
                "shared_predictions_metadata": metadata.read_bytes(),
            }
            plan_identity = {
                "manifest_sha256": signature["manifest_sha256"],
                "manifest_protocol_sha256": signature[
                    "manifest_protocol_sha256"
                ],
                "meter_detector_weights_sha256": signature[
                    "weights_sha256"
                ]["meter_detector"],
                "keypoint_detector_weights_sha256": signature[
                    "weights_sha256"
                ]["keypoint_detector"],
            }
            shared.write_bytes(b"replaced after authorization")
            metadata.write_text("{}", encoding="utf-8")
            snapshotted, _ = _load_shared_predictions(
                shared,
                manifest=manifest,
                rows=[{"sample_id": "a"}],
                condition="clean",
                degradation_seed=20260720,
                meter_weights=meter_weights,
                point_weights=point_weights,
                authorized_inputs=authorized_inputs,
                plan_identity=plan_identity,
            )
            self.assertEqual(set(snapshotted), {"a"})

            shared.write_text('{"sample_id":"a"}\n', encoding="utf-8")

            signature["input_degradation"]["protocol"] = "different"
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "degradation protocol"):
                _load_shared_predictions(
                    shared,
                    manifest=manifest,
                    rows=[{"sample_id": "a"}],
                    condition="clean",
                    degradation_seed=20260720,
                    meter_weights=meter_weights,
                    point_weights=point_weights,
                )

            signature["input_degradation"] = {}
            signature.pop("input_degradation_source_sha256")
            metadata.write_text(
                json.dumps({"signature": signature}),
                encoding="utf-8",
            )
            legacy, _ = _load_shared_predictions(
                shared,
                manifest=manifest,
                rows=[{"sample_id": "a"}],
                condition="clean",
                degradation_seed=20260720,
                meter_weights=meter_weights,
                point_weights=point_weights,
            )
            self.assertEqual(set(legacy), {"a"})
            with self.assertRaisesRegex(ValueError, "clean legacy"):
                _load_shared_predictions(
                    shared,
                    manifest=manifest,
                    rows=[{"sample_id": "a"}],
                    condition="blur_severe",
                    degradation_seed=20260720,
                    meter_weights=meter_weights,
                    point_weights=point_weights,
                )


if __name__ == "__main__":
    unittest.main()
