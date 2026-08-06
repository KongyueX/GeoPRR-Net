from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import field_blind_final_chain_preflight as gate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FieldBlindFinalChainPreflightTests(unittest.TestCase):
    def test_explicit_dataset_paths_bind_without_opening_data_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = (root / "never-open-manifest.jsonl").resolve()
            labels = (root / "never-open-labels.jsonl").resolve()
            identity = root / "identity.json"
            identity.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol": gate._load_dataset_identity.__globals__[
                            "DATASET_IDENTITY_PROTOCOL"
                        ],
                        "status": "owner_frozen",
                        "definition_authority": "dataset_owner",
                        "cohort_definition": {
                            "declared_images": 1201,
                            "deduplicated_before_freeze": True,
                            "declared_as_frozen_unseen_blind_test": True,
                            "source_unit": "original_full_scene_photograph",
                        },
                        "unlabeled_manifest": {
                            "path": str(manifest),
                            "sha256": "a" * 64,
                            "rows": 1201,
                            "input_role": "original_full_scene",
                            "contains_labels": False,
                            "contains_manual_or_gt_range": False,
                            "contains_manual_or_gt_crop": False,
                        },
                        "labels": {
                            "path": str(labels),
                            "sha256": "b" * 64,
                            "rows": 1201,
                        },
                        "authorization": {
                            "one_shot_image_inference": True,
                            "one_shot_scoring_after_prediction_seal": True,
                            "no_tuning_after_result": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(manifest.exists())
            self.assertFalse(labels.exists())
            result = gate.validate_explicit_dataset_bindings(
                dataset_identity_path=identity,
                manifest_path=manifest,
                labels_path=labels,
            )
            self.assertEqual(
                result["status"],
                "explicit_owner_bindings_verified_without_data_access",
            )
            self.assertFalse(result["manifest"]["opened"])
            self.assertFalse(result["manifest"]["hashed"])
            self.assertFalse(result["labels"]["opened"])
            self.assertFalse(result["labels"]["hashed"])
            self.assertFalse(manifest.exists())
            self.assertFalse(labels.exists())

            with self.assertRaisesRegex(ValueError, "explicit manifest path differs"):
                gate.validate_explicit_dataset_bindings(
                    dataset_identity_path=identity,
                    manifest_path=root / "different.jsonl",
                    labels_path=labels,
                )
            self.assertFalse(manifest.exists())
            self.assertFalse(labels.exists())

    def test_public_gate_requires_exact_sealed_roster_and_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frontend = root / "frontend.json"
            frontend.write_text("{}", encoding="utf-8")
            roster = root / "method_roster.json"
            roster.write_text(
                json.dumps(
                    {
                        "shared_frontend": {
                            "plan": {
                                "path": str(frontend.resolve()),
                                "sha256": _sha(frontend),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text("{}", encoding="utf-8")
            bundle_paths = {
                role: root / f"{role}.bundle.json" for role in gate.METHOD_ROLES
            }
            for role, path in bundle_paths.items():
                path.write_text(json.dumps({"role": role}), encoding="utf-8")
            seal = root / "seal.json"
            seal.write_text(
                json.dumps(
                    {
                        "protocol": gate.MATERIALIZATION_PROTOCOL,
                        "status": "sealed",
                        "artifacts": {
                            "summary": {
                                "path": str(summary.resolve()),
                                "sha256": _sha(summary),
                            },
                            "method_roster": {
                                "path": str(roster.resolve()),
                                "sha256": _sha(roster),
                            },
                            **{
                                f"bundle_{role}": {
                                    "path": str(path.resolve()),
                                    "sha256": _sha(path),
                                }
                                for role, path in bundle_paths.items()
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            verified = {
                "status": "verified",
                "verified": True,
                "method_roles": list(gate.METHOD_ROLES),
                "garc_shared_range_binding_sha256": "c" * 64,
                "field_manifest_opened": False,
                "field_images_opened": False,
                "field_labels_opened": False,
            }
            bundles = {role: object() for role in gate.METHOD_ROLES}
            with (
                patch.object(gate, "verify_materialization", return_value=verified),
                patch.object(
                    gate,
                    "load_method_roster",
                    return_value=(roster.resolve(), {}, bundles),
                ),
                patch.object(
                    gate,
                    "load_frontend_plan",
                    return_value=(frontend.resolve(), {}),
                ),
            ):
                result = gate.validate_public_materialization(
                    materialized_root=root,
                    method_roster_path=roster,
                    frontend_plan_path=frontend,
                )
                self.assertEqual(result["status"], "public_materialization_verified")
                self.assertEqual(result["sealed_bundle_count"], 5)
                broken = json.loads(seal.read_text(encoding="utf-8"))
                del broken["artifacts"]["bundle_vdn_shared_range"]
                seal.write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "exactly the summary"):
                    gate.validate_public_materialization(
                        materialized_root=root,
                        method_roster_path=roster,
                        frontend_plan_path=frontend,
                    )
                broken["artifacts"]["bundle_vdn_shared_range"] = {
                    "path": str(bundle_paths["vdn_shared_range"].resolve()),
                    "sha256": _sha(bundle_paths["vdn_shared_range"]),
                }
                seal.write_text(json.dumps(broken), encoding="utf-8")
                other_roster = root / "other.json"
                other_roster.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sealed materialized roster"):
                    gate.validate_public_materialization(
                        materialized_root=root,
                        method_roster_path=other_roster,
                        frontend_plan_path=frontend,
                    )

    def test_runtime_gate_instantiates_all_five_without_dataset_argument(self) -> None:
        roster = Path("C:/synthetic/roster.json")
        frontend = Path("C:/synthetic/frontend.json")
        bundles = {role: object() for role in gate.METHOD_ROLES}
        adapters = {
            role: SimpleNamespace(identity={"role": role})
            for role in gate.METHOD_ROLES
        }
        with (
            patch.object(
                gate,
                "load_method_roster",
                return_value=(roster, {"methods": {}}, bundles),
            ),
            patch.object(
                gate,
                "load_frontend_plan",
                return_value=(frontend, {"detector_checkpoint": {}}),
            ),
            patch.object(gate, "_load_shared_detector", return_value=object()),
            patch.object(gate, "_load_adapters", return_value=adapters),
            patch.object(gate, "sha256_file", return_value="d" * 64),
        ):
            result = gate.runtime_preflight(
                method_roster_path=roster,
                frontend_plan_path=frontend,
            )
        self.assertEqual(
            result["status"],
            "all_public_runtimes_instantiated_without_dataset_access",
        )
        self.assertEqual(result["method_roles"], list(gate.METHOD_ROLES))
        self.assertFalse(result["field_manifest_opened"])
        self.assertFalse(result["field_images_opened"])
        self.assertFalse(result["field_labels_opened"])


if __name__ == "__main__":
    unittest.main()
