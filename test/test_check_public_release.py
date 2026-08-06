from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.check_public_release import (
    category_suffix_violations,
    dependency_closure,
    forbidden_release_paths,
    inventory_files,
)


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseCheckerTests(unittest.TestCase):
    def test_inventory_rejects_parent_escape(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stay under"):
            inventory_files({"categories": {"bad": ["../secret.txt"]}})

    def test_dependency_closure_resolves_repository_module(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "experiments"
            package.mkdir()
            (package / "entry.py").write_text(
                "import numpy\nfrom experiments.helper import VALUE\n", encoding="utf-8"
            )
            (package / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
            closure, external, unresolved = dependency_closure(
                root, ["experiments/entry.py"], {"experiments"}
            )
        self.assertEqual(
            closure, {"experiments/entry.py", "experiments/helper.py"}
        )
        self.assertEqual(external, {"numpy"})
        self.assertEqual(unresolved, [])

    def test_forbidden_release_paths_reject_data_outputs_and_private_segments(self) -> None:
        violations = forbidden_release_paths(
            [
                "experiments/field_blind_multimethod.py",
                "field_data/private.jsonl",
                "paper/manuscript.tex",
                "experiments/model.pt",
                "experiments/private/local.json",
            ],
            prefixes=["field_data/", "paper/"],
            suffixes=[".pt", ".jsonl"],
            patterns=[r"(?:^|/)private(?:/|$)"],
        )
        self.assertEqual(
            violations,
            [
                "experiments/model.pt",
                "experiments/private/local.json",
                "field_data/private.jsonl",
                "paper/manuscript.tex",
            ],
        )

    def test_category_suffix_policy_rejects_data_in_code_only_category(self) -> None:
        inventory = {
            "categories": {"field_bundle_code_only": ["field/output.jsonl"]},
            "category_allowed_suffixes": {"field_bundle_code_only": [".py", ".ps1"]},
        }
        self.assertEqual(
            category_suffix_violations(inventory),
            ["field_bundle_code_only: field/output.jsonl"],
        )

    def test_inventory_covers_detector_bundle_and_final_release_chain(self) -> None:
        inventory = json.loads(
            (ROOT / "experiments/public_release_inventory.json").read_text(
                encoding="utf-8"
            )
        )
        declared = set(inventory_files(inventory))
        required = {
            "experiments/build_syncg_meter_detector_public.py",
            "experiments/check_syncg_meter_detector_public.py",
            "experiments/fetch_syncg_meter_detector_pretrained.py",
            "experiments/syncg_meter_detector_frontend.py",
            "experiments/train_syncg_meter_detector.py",
            "experiments/syncg_meter_detector_protocol.json",
            "experiments/run_syncg_meter_detector_training.ps1",
            "experiments/run_syncg_meter_detector_after_paper_event_driven.ps1",
            "experiments/build_field_blind_bundle_inputs.py",
            "experiments/materialize_field_blind_bundles.py",
            "experiments/run_materialize_field_blind_bundles_after_paper_event_driven.ps1",
            "experiments/run_field_bundle_materialization_after_detector_event_driven.ps1",
            "experiments/field_blind_final_chain_preflight.py",
            "experiments/run_field_blind_multimethod_after_bundles_event_driven.ps1",
            "test/test_syncg_meter_detector_public.py",
            "test/test_build_field_blind_bundle_inputs.py",
            "test/test_materialize_field_blind_bundles.py",
            "test/test_run_materialize_field_blind_bundles_after_paper_event_driven.py",
            "test/test_run_field_bundle_materialization_after_detector_event_driven.py",
            "test/test_field_blind_final_chain_preflight.py",
            "test/test_run_field_blind_multimethod_after_bundles_event_driven.py",
        }
        self.assertEqual(required.difference(declared), set())
        self.assertIn(
            "experiments/field_blind_final_chain_preflight.py",
            inventory["python_entrypoints"],
        )
        self.assertEqual(category_suffix_violations(inventory), [])


if __name__ == "__main__":
    unittest.main()
