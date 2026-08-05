from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.compare_pct_determinism_runs import semantic_sha256
from experiments.probe_vdn_official200_determinism import (
    DETERMINISM_PROTOCOL,
    WORKER_KEYS,
    WORKER_PROTOCOL,
    compare_workers,
)
from experiments.vdn_official200_protocol import (
    OFFICIAL200_DETERMINISM_REPORT_KEYS,
    canonical_json_sha256,
    validate_authorized_runtime_environment,
    validate_determinism_report_payload,
)


class VDNOfficial200DeterminismProbeTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "identity": {
                "preflight": {"report_sha256": "a" * 64},
                "content_inventory": {"report_sha256": "b" * 64},
                "runtime_environment": {
                    "gpu": "synthetic",
                    "pythonhashseed": "20260720",
                },
                "training_source_sha256": {"trainer": "c" * 64},
            },
            "train_metrics": {"loss": 1.0},
            "validation_metrics": {"angle_mae_degrees": 0.7},
            "optimizer_accounting": {
                "attempts": 10,
                "successful": 10,
                "skipped": 0,
            },
            "state_sha256": {
                "initial": {"model": "d" * 64},
                "final": {"model": "e" * 64},
            },
            "state_health": {"model": {"non_finite_tensors": 0}},
            "test_data_opened_or_read": False,
            "public_data_opened_or_read": False,
            "field_data_opened_or_read": False,
            "sealed_data_opened_or_read": False,
            "confirmatory_data_opened_or_read": False,
        }

    def _worker(self, replicate: str, pid: int) -> dict:
        from experiments.vdn_baseline import sha256_source_file

        source = Path(
            "experiments/probe_vdn_official200_determinism.py"
        ).resolve()
        payload = self._payload()
        worker = {
            "protocol": WORKER_PROTOCOL,
            "schema_version": 1,
            "replicate": replicate,
            "pid": pid,
            "process_token": f"token-{replicate}",
            "semantic_payload": payload,
            "semantic_payload_sha256": semantic_sha256(payload),
            "worker_source_sha256": sha256_source_file(source),
        }
        self.assertEqual(set(worker), WORKER_KEYS)
        return worker

    def test_two_exact_workers_authorize_training_only(self):
        report = compare_workers(
            self._worker("a", 101),
            self._worker("b", 202),
        )
        self.assertEqual(report["protocol"], DETERMINISM_PROTOCOL)
        self.assertTrue(report["official200_training_start_authorized"])
        self.assertFalse(report["supporting_test_evaluation_authorized"])
        self.assertFalse(report["field_confirmatory_evaluation_authorized"])
        self.assertTrue(
            all(
                component["exact"]
                for component in report["components"].values()
            )
        )

    def test_metric_difference_fails_closed(self):
        worker_a = self._worker("a", 101)
        worker_b = self._worker("b", 202)
        worker_b["semantic_payload"]["train_metrics"]["loss"] = 0.9
        worker_b["semantic_payload_sha256"] = semantic_sha256(
            worker_b["semantic_payload"]
        )
        with self.assertRaisesRegex(ValueError, "not exactly deterministic"):
            compare_workers(worker_a, worker_b)

    def test_runtime_authorization_ignores_only_seed_hash_and_journal(self):
        report = {
            "scientific_identity": {
                "runtime_environment": {
                    "gpu": "synthetic",
                    "torch": "2.11",
                    "pythonhashseed": "20260720",
                }
            }
        }
        current = {
            "gpu": "synthetic",
            "torch": "2.11",
            "pythonhashseed": "20260722",
            "determinism_authorization": {"exact": True},
        }
        result = validate_authorized_runtime_environment(report, current)
        self.assertTrue(
            result[
                "exact_authorized_runtime_except_seed_specific_pythonhashseed"
            ]
        )
        current["gpu"] = "different"
        with self.assertRaisesRegex(RuntimeError, "differs"):
            validate_authorized_runtime_environment(report, current)

    def test_static_authorization_rejects_tampered_canonical_payload(self):
        preflight = {
            "protocol": "vdn_official200_training_preflight_v1",
            "report_path": "preflight.json",
            "report_sha256": "a" * 64,
            "canonical_payload_sha256": "b" * 64,
        }
        content = {"identity": "train-only"}
        sources = {"trainer": "c" * 64}
        report = {key: None for key in OFFICIAL200_DETERMINISM_REPORT_KEYS}
        report.update(
            {
                "protocol": DETERMINISM_PROTOCOL,
                "schema_version": 1,
                "status": "passed",
                "official200_training_start_authorized": True,
                "supporting_test_evaluation_authorized": False,
                "field_confirmatory_evaluation_authorized": False,
                "fixed_probe": {
                    "seed": 20260720,
                    "epoch": 1,
                    "epoch_seed": 20260720,
                    "complete_train_and_validation_epoch": True,
                    "separate_python_processes": True,
                    "model_optimizer_scaler_dataloader_rebuilt_per_worker": (
                        True
                    ),
                },
                "preflight": preflight,
                "scientific_identity": {
                    "preflight": preflight,
                    "content_inventory": content,
                    "training_source_sha256": sources,
                    "runtime_environment": {"gpu": "synthetic"},
                },
                "components": {
                    "metrics": {
                        "exact": True,
                        "semantic_sha256": "d" * 64,
                    }
                },
                "worker_execution": {
                    "a_pid": 1,
                    "b_pid": 2,
                    "process_tokens_distinct": True,
                    "a_semantic_payload_sha256": "e" * 64,
                    "b_semantic_payload_sha256": "e" * 64,
                },
                "source_hash_protocol": "utf8_source_newlines_lf_v1",
                "probe_source_sha256": "f" * 64,
                "training_source_sha256": sources,
                "test_data_opened_or_read": False,
                "public_data_opened_or_read": False,
                "field_data_opened_or_read": False,
                "sealed_data_opened_or_read": False,
                "confirmatory_data_opened_or_read": False,
            }
        )
        payload = dict(report)
        payload.pop("canonical_probe_payload_sha256")
        report["canonical_probe_payload_sha256"] = canonical_json_sha256(
            payload
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "determinism.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with (
                mock.patch(
                    "experiments.vdn_official200_protocol."
                    "official200_source_hashes",
                    return_value=sources,
                ),
                mock.patch(
                    "experiments.vdn_official200_protocol."
                    "sha256_source_file",
                    return_value="f" * 64,
                ),
            ):
                authorization = validate_determinism_report_payload(
                    report,
                    report_path=path,
                    preflight_binding=preflight,
                    content_inventory_identity=content,
                    vdn_source=Path(temporary),
                )
                self.assertEqual(
                    authorization["semantic_payload_sha256"],
                    "e" * 64,
                )
                tampered = copy.deepcopy(report)
                tampered["status"] = "failed"
                with self.assertRaisesRegex(ValueError, "canonical digest"):
                    validate_determinism_report_payload(
                        tampered,
                        report_path=path,
                        preflight_binding=preflight,
                        content_inventory_identity=content,
                        vdn_source=Path(temporary),
                    )


if __name__ == "__main__":
    unittest.main()
