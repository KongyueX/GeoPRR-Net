from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from experiments.build_fadr_input_authorization import build_authorization
from experiments.fadr_multiseed_protocol import (
    EXPECTED_OOF_PROTOCOL,
    EXPECTED_PEPD_COHORT_PROTOCOL,
    EXPECTED_PEPD_COLLECTOR_CONTRACT_PROTOCOL,
    EXPECTED_PEPD_HANDOFF_PROTOCOL,
    EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED,
    EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED,
    FADR_INPUT_AUTHORIZATION_PROTOCOL,
    FADR_INPUT_PREFLIGHT_PROTOCOL,
    FADR_SEEDS,
    PEPD_DIRECTION_SEEDS,
    assert_train_only_path,
    sha256_file,
    strict_json_load,
)
from experiments.preflight_fadr_multiseed import validate_preflight_inputs
from experiments.strict_json import (
    strict_json_load as upstream_strict_json_load,
    strict_json_source_sha256,
)
from experiments.vdn_baseline import PROJECT_DIR, sha256_source_file


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mapping_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _valid_fixture(root: Path) -> dict[str, Path]:
    checkpoints = {
        str(seed): f"{offset:064x}"
        for offset, seed in enumerate(PEPD_DIRECTION_SEEDS, start=1)
    }
    cohort = root / "pepd_cohort.json"
    _write_json(
        cohort,
        {
            "schema_version": 2,
            "protocol": EXPECTED_PEPD_COHORT_PROTOCOL,
            "status": "converged",
            "seeds": list(PEPD_DIRECTION_SEEDS),
            "all_runs_verified": True,
            "all_runs_converged": True,
            "mixed_authority": {
                "20260720": "convergence_v1",
                "20260721": "bounded_extension_v2",
                "20260722": "convergence_v1",
            },
            "further_epoch_extension_authorized": False,
            "public_test_field_evaluation_authorized": False,
            "runs": [
                {
                    "seed": seed,
                    "verified": True,
                    "converged": True,
                    "source_phase": (
                        "bounded_extension_v2"
                        if seed == 20260721
                        else "convergence_v1"
                    ),
                    "authoritative_run_protocol": (
                        EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED[str(seed)]
                    ),
                    "verification_protocol": (
                        EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED[str(seed)]
                    ),
                    "best_checkpoint_sha256": checkpoints[str(seed)],
                }
                for seed in PEPD_DIRECTION_SEEDS
            ],
        },
    )
    handoff_runs = {}
    for seed in PEPD_DIRECTION_SEEDS:
        seed_key = str(seed)
        base_signature = {
            "protocol": EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED["20260720"],
            "seed": seed,
        }
        run_signature = {
            "protocol": EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED[seed_key],
            "seed": seed,
        }
        lineage_signature = (
            run_signature if seed == 20260721 else base_signature
        )
        handoff_runs[seed_key] = {
            "seed": seed,
            "source_phase": (
                "bounded_extension_v2"
                if seed == 20260721
                else "convergence_v1"
            ),
            "authoritative_run_protocol": (
                EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED[seed_key]
            ),
            "authoritative_run_signature": run_signature,
            "authoritative_run_signature_sha256": _mapping_sha256(
                run_signature
            ),
            "verification_protocol": (
                EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED[seed_key]
            ),
            "authoritative_best_checkpoint_sha256": checkpoints[seed_key],
            "authoritative_best_epoch": 80 if seed == 20260721 else 60,
            "checkpoint_changed_from_legacy_parent": True,
            "base_v1_continuation_signature": base_signature,
            "base_v1_continuation_signature_sha256": _mapping_sha256(
                base_signature
            ),
            "checkpoint_lineage_protocol": lineage_signature["protocol"],
            "checkpoint_lineage_signature": lineage_signature,
            "checkpoint_lineage_signature_sha256": _mapping_sha256(
                lineage_signature
            ),
            "summary_sha256": "e" * 64,
            "verification_sha256": "f" * 64,
            "grouped_split": {
                "split_seed": seed,
                "group_overlap": 0,
                "validation_samples": 1,
                "validation_groups": 1,
                "train_sample_ids_sha256": "a" * 64,
                "validation_sample_ids_sha256": "b" * 64,
                "train_group_ids_sha256": "c" * 64,
                "validation_group_ids_sha256": "d" * 64,
            },
        }
    handoff = root / "pepd_oof_handoff.json"
    _write_json(
        handoff,
        {
            "schema_version": 2,
            "protocol": EXPECTED_PEPD_HANDOFF_PROTOCOL,
            "status": "authorized",
            "formal_seeds": list(PEPD_DIRECTION_SEEDS),
            "mixed_authority": dict(EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED),
            "cohort": str(cohort.resolve()),
            "cohort_sha256": sha256_file(cohort),
            "further_epoch_extension_authorized": False,
            "public_test_field_evaluation_authorized": False,
            "versioned_collector_contract": {
                "protocol": EXPECTED_PEPD_COLLECTOR_CONTRACT_PROTOCOL,
                "authorized": True,
                "collector": "experiments/collect_uncertainty_fusion_oof.py",
                "collector_source_sha256": sha256_source_file(
                    PROJECT_DIR
                    / "experiments"
                    / "collect_uncertainty_fusion_oof.py"
                ),
                "required_cli": ["--pepd-oof-handoff", "--pepd-cohort"],
                "legacy_checkpoint_fallback_allowed": False,
            },
            "oof_assignment_contract": {
                "allowed_seeds": list(PEPD_DIRECTION_SEEDS),
                "fallback_policy": "none",
                "legacy_checkpoint_fallback_allowed": False,
                "test_sets_used": [],
            },
            "authoritative_runs": handoff_runs,
        },
    )

    oof = root / "train_oof.jsonl"
    rows = [
        {
            "dataset": "SyncG",
            "split": "train",
            "sample_id": f"sample-{index}",
            "group_id": f"group-{index}",
            "held_out_seed": seed,
        }
        for index, seed in enumerate(PEPD_DIRECTION_SEEDS)
    ]
    oof.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    signature = {
        "protocol": EXPECTED_OOF_PROTOCOL,
        "split": "SyncG/train only",
        "test_sets_used": [],
        "pepd_oof_handoff": str(handoff.resolve()),
        "pepd_oof_handoff_sha256": sha256_file(handoff),
        "pepd_oof_handoff_protocol": EXPECTED_PEPD_HANDOFF_PROTOCOL,
        "pepd_cohort": str(cohort.resolve()),
        "pepd_cohort_sha256": sha256_file(cohort),
        "pepd_cohort_protocol": EXPECTED_PEPD_COHORT_PROTOCOL,
        "legacy_checkpoint_fallback_allowed": False,
        "direction_runs": {
            str(seed): {
                "checkpoint_sha256": checkpoints[str(seed)],
                "summary_sha256": "e" * 64,
                "verification_sha256": "f" * 64,
                "authoritative_run_protocol": (
                    EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED[str(seed)]
                ),
                "verification_protocol": (
                    EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED[str(seed)]
                ),
                "checkpoint_lineage_protocol": handoff_runs[str(seed)][
                    "checkpoint_lineage_protocol"
                ],
                "validation_samples": 1,
                "validation_groups": 1,
            }
            for seed in PEPD_DIRECTION_SEEDS
        },
        "assigned_samples": len(rows),
        "assigned_groups": len(rows),
        "oof_coverage": {
            "design": (
                "union of the three authoritative grouped validation splits "
                "(nominally 10% each); not complete K-fold OOF"
            ),
            "manifest_samples": 6,
            "assigned_samples": len(rows),
            "manifest_groups": 4,
            "assigned_groups": len(rows),
            "sample_fraction": len(rows) / 6,
            "group_fraction": len(rows) / 4,
            "complete_manifest_oof": False,
        },
        "source_sha256": sha256_file(
            PROJECT_DIR / "experiments" / "collect_uncertainty_fusion_oof.py"
        ),
        "strict_json_protocol": "research_artifact_strict_json_v1",
        "strict_json_source_sha256": strict_json_source_sha256(),
    }
    metadata = root / "train_oof.jsonl.meta.json"
    _write_json(metadata, {"signature": signature})
    summary = root / "train_oof.summary.json"
    _write_json(
        summary,
        {
            "schema_version": 1,
            "protocol": EXPECTED_OOF_PROTOCOL,
            "status": "complete",
            "samples": len(rows),
            "groups": len(rows),
            "group_leakage_count": 0,
            "test_samples_used": 0,
            "public_samples_used": 0,
            "field_samples_used": 0,
            "oof_coverage": signature["oof_coverage"],
            "signature": signature,
            "output_sha256": sha256_file(oof),
        },
    )
    authorization = root / "fadr_input_authorization.json"
    inputs = {
        "oof_pairs": oof,
        "oof_metadata": metadata,
        "oof_summary": summary,
        "pepd_cohort": cohort,
        "pepd_oof_handoff": handoff,
    }
    _write_json(
        authorization,
        {
            "schema_version": 1,
            "protocol": FADR_INPUT_AUTHORIZATION_PROTOCOL,
            "status": "authorized",
            "scope": "SyncG/train strict grouped OOF -> FADR train-only",
            "dataset": "SyncG",
            "split": "train",
            "train_only_certified": True,
            "group_leakage_count": 0,
            "test_samples_used": 0,
            "public_samples_used": 0,
            "field_samples_used": 0,
            "public_test_field_evaluation_authorized": False,
            "direction_seeds": list(PEPD_DIRECTION_SEEDS),
            "fadr_seeds": list(FADR_SEEDS),
            "direction_checkpoint_sha256": checkpoints,
            "direction_run_protocol": dict(
                EXPECTED_PEPD_RUN_PROTOCOL_BY_SEED
            ),
            "direction_verification_protocol": dict(
                EXPECTED_PEPD_VERIFICATION_PROTOCOL_BY_SEED
            ),
            "inputs": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for name, path in inputs.items()
            },
            "source_identity": {
                "builder": sha256_file(
                    PROJECT_DIR
                    / "experiments"
                    / "build_fadr_input_authorization.py"
                ),
                "protocol": sha256_file(
                    PROJECT_DIR / "experiments" / "fadr_multiseed_protocol.py"
                ),
                "strict_json": strict_json_source_sha256(),
            },
        },
    )
    return {
        **inputs,
        "input_authorization": authorization,
    }


class FadrMultiseedProtocolTest(unittest.TestCase):
    def test_fadr_reexports_the_upstream_strict_json_reader(self) -> None:
        self.assertIs(strict_json_load, upstream_strict_json_load)

    def test_valid_exact_authorization_and_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _valid_fixture(Path(temporary))
            result = validate_preflight_inputs(**paths)
        self.assertEqual(result["protocol"], FADR_INPUT_PREFLIGHT_PROTOCOL)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["direction_seeds"], list(PEPD_DIRECTION_SEEDS))
        self.assertEqual(result["fadr_seeds"], list(FADR_SEEDS))
        self.assertEqual(result["primary_fadr_seed"], FADR_SEEDS[0])
        self.assertEqual(result["oof_protocol"], EXPECTED_OOF_PROTOCOL)
        self.assertEqual(result["rows"]["samples"], 3)
        self.assertEqual(result["group_leakage_count"], 0)
        self.assertFalse(result["rows"]["coverage"]["complete_manifest_oof"])

    def test_builder_emits_the_exact_authorization_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _valid_fixture(Path(temporary))
            built = build_authorization(
                oof_pairs=paths["oof_pairs"],
                oof_metadata=paths["oof_metadata"],
                oof_summary=paths["oof_summary"],
                pepd_cohort=paths["pepd_cohort"],
                pepd_oof_handoff=paths["pepd_oof_handoff"],
            )
            frozen = strict_json_load(paths["input_authorization"])
        self.assertEqual(built, frozen)

    def test_authorization_hash_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _valid_fixture(Path(temporary))
            paths["oof_pairs"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_preflight_inputs(**paths)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text('{"status":"a","status":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                strict_json_load(path)

    def test_nonfinite_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for payload in ('{"value": NaN}', '{"value": 1e999}'):
                path = Path(temporary) / "nonfinite.json"
                path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        strict_json_load(path)

    def test_public_test_field_and_sealed_paths_are_rejected(self) -> None:
        for name in (
            "public_original",
            "test",
            "field_development",
            "field_confirmatory",
            "sealed",
            "confirmatory_batch",
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "forbidden path"):
                    assert_train_only_path(
                        Path("artifacts") / "runs" / name / "input.json",
                        label="fixture",
                    )


if __name__ == "__main__":
    unittest.main()
