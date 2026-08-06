from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import build_field_blind_bundle_inputs as builder
from experiments import materialize_field_blind_bundles as materializer
from experiments.field_blind_multimethod import METHOD_ROLES
from experiments.v5_unified_full_auto_adapter import (
    EXECUTION_FORMAL,
    REFERENCE_MODE_AUTO,
    FrozenComponentBinding,
    FrozenFullAutoBundle,
    sha256_file,
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    return path


class FieldBlindBundleInputBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.public = self.root / "public-authorities"
        self.public.mkdir()
        self.vdn_oof_summary = _write_json(
            self.public / "vdn-oof-summary.json", {"status": "complete"}
        )
        self.vdn_oof_patch = mock.patch.object(
            builder.vdn_factory, "OOF_SUMMARY", self.vdn_oof_summary
        )
        self.vdn_oof_patch.start()
        self.addCleanup(self.vdn_oof_patch.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _component(
        self,
        name: str,
        provider: str,
        artifact: Path,
        source: Path,
    ) -> FrozenComponentBinding:
        return FrozenComponentBinding(
            name=name,
            provider_protocol="synthetic_provider_v1",
            provider_identity={"protocol": "synthetic_provider_v1", "provider": provider},
            artifact_sha256={"checkpoint": sha256_file(artifact)},
            source_sha256={"provider": sha256_file(source)},
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )

    def _authority(self) -> builder.PreparedAuthorities:
        reference = _write(self.public / "reference.pt", b"reference")
        garc_progress_artifact = _write(
            self.public / "pepd-seed20.pt", b"pepd-seed-20"
        )
        garc_progress_source = _write(
            self.public / "pepd-provider.py", b"PEPD = True\n"
        )
        garc_range_artifact = _write(self.public / "garc-range.pt", b"garc-range")
        garc_range_source = _write(
            self.public / "garc-range.py", b"GARC_RANGE = True\n"
        )
        v5_range_artifact = _write(self.public / "v5-range.pt", b"v5-range")
        v5_range_source = _write(
            self.public / "v5-range.py", b"V5_RANGE = True\n"
        )
        progress = self._component(
            "progress", "pepd", garc_progress_artifact, garc_progress_source
        )
        pepd_progress = FrozenComponentBinding(
            name="progress",
            provider_protocol=progress.provider_protocol,
            provider_identity=progress.provider_identity,
            artifact_sha256=progress.artifact_sha256,
            source_sha256={
                **progress.source_sha256,
                "reference_detector_loader": sha256_file(
                    builder.REFERENCE_LOADER_SOURCE
                ),
            },
            frozen=True,
            verified_complete=True,
            synthetic=False,
        )
        garc_range = self._component(
            "automatic_numeric_range",
            "garc_range",
            garc_range_artifact,
            garc_range_source,
        )
        v5_range = self._component(
            "automatic_numeric_range",
            "pure_v5_range",
            v5_range_artifact,
            v5_range_source,
        )
        garc_bundle = FrozenFullAutoBundle.create(
            method_name="GARC-source+auto-ref",
            progress_binding=progress,
            range_binding=garc_range,
            factory_source_sha256=sha256_file(builder.RUNTIME_FACTORY),
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=sha256_file(reference),
            execution_mode=EXECUTION_FORMAL,
        )
        v5_bundle = FrozenFullAutoBundle.create(
            method_name="GARC-V5-calibration+auto-ref",
            progress_binding=progress,
            range_binding=v5_range,
            factory_source_sha256=sha256_file(builder.RUNTIME_FACTORY),
            reference_mode=REFERENCE_MODE_AUTO,
            reference_detector_sha256=sha256_file(reference),
            execution_mode=EXECUTION_FORMAL,
        )
        garc_bundle_path = self.public / "garc-validation.bundle.json"
        v5_bundle_path = self.public / "v5-calibration.bundle.json"
        garc_bundle.write(garc_bundle_path)
        v5_bundle.write(v5_bundle_path)

        vdn_checkpoint = _write(self.public / "vdn-seed20.pt", b"vdn")
        vdn_verification = _write_json(
            self.public / "vdn-verification.json", {"verified": True}
        )
        transformer = _write(self.public / "transformer.pt", b"transformer")
        pointer = _write(self.public / "pointer.pt", b"pointer")
        final_plan = _write_json(self.public / "garc-plan.json", {"plan": "garc"})
        base_plan = _write_json(self.public / "v5-plan.json", {"plan": "v5"})
        garc_summary = _write_json(self.public / "garc-summary.json", {"status": "complete"})
        paper_summary = _write_json(self.public / "paper-summary.json", {"status": "complete"})
        paper_seal = _write_json(self.public / "paper-seal.json", {"status": "sealed"})
        frontend_plan = _write_json(self.public / "frontend-plan.json", {"status": "frozen"})
        frontend_seal = _write_json(self.public / "frontend-seal.json", {"status": "sealed"})
        external_protocol = _write_json(
            self.public / "external-protocol.json", {"status": "frozen"}
        )
        vdn_identity = {
            "protocol": builder.PROGRESS_PROVIDER_PROTOCOL,
            "provider": "vdn_official200",
            "checkpoint_sha256": sha256_file(vdn_checkpoint),
            "verification_sha256": sha256_file(vdn_verification),
            "reference_detector_sha256": sha256_file(reference),
        }
        transformer_identity = {
            "protocol": builder.PROGRESS_PROVIDER_PROTOCOL,
            "provider": "original_transformer_native_progress",
            "component_sha256": {
                "pointer_segmentation": sha256_file(pointer),
                "original_transformer": sha256_file(transformer),
                "production_pipeline_source": sha256_file(
                    builder.PRODUCTION_PIPELINE_SOURCE
                ),
            },
            "task_config_sha256": builder.canonical_json_sha256(
                builder.asdict(builder.FrozenLegacyTaskConfig())
            ),
            "reference_detector_loaded": False,
            "reference_detector_invoked": False,
        }
        candidates = {
            reference,
            garc_progress_artifact,
            garc_progress_source,
            garc_range_artifact,
            garc_range_source,
            v5_range_artifact,
            v5_range_source,
            garc_bundle_path,
            v5_bundle_path,
            vdn_checkpoint,
            vdn_verification,
            transformer,
            pointer,
            final_plan,
            base_plan,
            garc_summary,
            paper_summary,
            paper_seal,
            frontend_plan,
            frontend_seal,
            external_protocol,
            builder.RUNTIME_FACTORY,
            builder.ADAPTER_SOURCE,
            builder.PROGRESS_WRAPPER_SOURCE,
            builder.DIRECTION_ADAPTER_SOURCE,
            builder.LEGACY_ADAPTER_SOURCE,
            builder.REFERENCE_LOADER_SOURCE,
            builder.PRODUCTION_PIPELINE_SOURCE,
            builder.vdn_factory.SOURCE,
            builder.vdn_factory.OOF_SUMMARY,
        }
        return builder.PreparedAuthorities(
            garc_summary=garc_summary,
            paper_summary=paper_summary,
            paper_seal=paper_seal,
            frontend_plan=frontend_plan,
            frontend_seal=frontend_seal,
            final_plan_path=final_plan,
            final_plan={"garc": {"device": "cuda:0"}},
            base_v5_plan_path=base_plan,
            base_v5_plan={"garc": {"device": "cuda:0"}},
            garc_bundle_path=garc_bundle_path,
            garc_bundle=garc_bundle,
            v5_bundle_path=v5_bundle_path,
            v5_bundle=v5_bundle,
            pepd_progress=pepd_progress,
            vdn_identity=vdn_identity,
            vdn_checkpoint=vdn_checkpoint,
            vdn_verification=vdn_verification,
            vdn_reference_detector=reference,
            transformer_identity=transformer_identity,
            transformer_checkpoint=transformer,
            transformer_pointer_segmentation=pointer,
            transformer_task_config=builder.asdict(builder.FrozenLegacyTaskConfig()),
            external_protocol_path=external_protocol,
            artifact_candidates=tuple(sorted(candidates, key=lambda path: str(path))),
        )

    def test_builds_five_identity_bound_inputs_and_shared_range(self) -> None:
        authority = self._authority()
        output = self.root / "prepared"
        captured: dict[str, object] = {}

        def fake_freeze_spec(**kwargs):
            captured.update(kwargs)
            builder._write_new_json(
                Path(kwargs["output"]),
                {"protocol": materializer.SPEC_PROTOCOL, "status": "synthetic-test"},
            )
            return Path(kwargs["output"])

        formal_roots = builder._allowed_roots()
        with (
            mock.patch.object(
                builder,
                "_allowed_roots",
                return_value=(*formal_roots, self.root),
            ),
            mock.patch.object(builder, "_load_authorities", return_value=authority),
            mock.patch.object(
                builder.materializer, "freeze_spec", side_effect=fake_freeze_spec
            ) as freeze,
            mock.patch.object(
                builder.materializer,
                "validate_spec",
                return_value={"sources": {role: object() for role in METHOD_ROLES}},
            ),
        ):
            result = builder.build_inputs(
                garc_summary=authority.garc_summary,
                paper_summary=authority.paper_summary,
                paper_seal=authority.paper_seal,
                frontend_plan=authority.frontend_plan,
                output_root=output,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(freeze.call_count, 1)
        sources = captured["source_bundles"]
        configs = captured["factory_configs"]
        self.assertEqual(set(sources), set(METHOD_ROLES))
        self.assertEqual(set(configs), set(METHOD_ROLES))
        bundles = {
            role: FrozenFullAutoBundle.load(Path(path))
            for role, path in sources.items()
        }
        garc_range = bundles["garc_final"].range_binding_sha256
        for role in (
            "pepd_shared_range",
            "vdn_shared_range",
            "transformer_shared_range",
        ):
            self.assertEqual(bundles[role].range_binding_sha256, garc_range)
        self.assertNotEqual(bundles["v5_complete"].range_binding_sha256, garc_range)
        self.assertEqual(
            bundles["pepd_shared_range"].progress_binding.provider_identity,
            authority.pepd_progress.provider_identity,
        )
        self.assertEqual(
            bundles["vdn_shared_range"].progress_binding.provider_identity,
            authority.vdn_identity,
        )
        self.assertEqual(
            bundles["transformer_shared_range"].progress_binding.provider_identity,
            authority.transformer_identity,
        )
        final_plan = str(authority.final_plan_path.resolve())
        for role in (
            "garc_final",
            "pepd_shared_range",
            "vdn_shared_range",
            "transformer_shared_range",
        ):
            config = json.loads(Path(configs[role]).read_text(encoding="utf-8"))
            self.assertEqual(config["source_plan"]["path"], final_plan)
        v5_config = json.loads(
            Path(configs["v5_complete"]).read_text(encoding="utf-8")
        )
        self.assertEqual(
            v5_config["source_plan"]["path"],
            str(authority.base_v5_plan_path.resolve()),
        )
        inventory = json.loads(
            (output / "source_bundle_inventory.json").read_text(encoding="utf-8")
        )
        catalog = json.loads(
            (output / "runtime_artifact_catalog.json").read_text(encoding="utf-8")
        )
        self.assertEqual(set(inventory["methods"]), set(METHOD_ROLES))
        self.assertFalse(inventory["contracts"]["ground_truth_or_manual_range_used"])
        for artifact in (inventory, catalog, result):
            self.assertEqual(artifact["audit"]["images_opened"], 0)
            self.assertEqual(artifact["audit"]["directory_enumerations"], 0)
            self.assertEqual(
                artifact["audit"]["restricted_namespace_artifacts_opened"], 0
            )
            self.assertFalse(artifact["audit"]["field_images_opened"])

    def test_schema_gap_lists_frontend_lineage_and_external_authority(self) -> None:
        gaps = builder._schema_gaps(
            {"validation": {}, "selected": {}, "recognizer_selection": {}},
            {"sources": {}},
            {"artifacts": {}},
            {"detector_checkpoint": {}, "detector_source": {}, "selection_audit": {}},
        )
        self.assertIn(
            "frontend.selection_audit.training_summary.{path,sha256}", gaps
        )
        self.assertIn(
            "frontend.selection_audit.selection_claim.{path,sha256}", gaps
        )
        self.assertIn(
            "paper.sources.external_comparison.{path,sha256}", gaps
        )
        self.assertIn("garc.selected.{plan,plan_sha256}", gaps)

    def test_external_prediction_relative_artifact_is_root_bound(self) -> None:
        prediction_root = self.public / "external-predictions"
        rows = _write(prediction_root / "predictions.jsonl", b"{}\n")
        summary = {
            "artifacts": {
                "predictions": {
                    "path": "predictions.jsonl",
                    "sha256": sha256_file(rows),
                }
            }
        }
        with mock.patch.object(
            builder,
            "_allowed_roots",
            return_value=(*builder._allowed_roots(), self.root),
        ):
            found = builder._walk_exact_bindings(
                summary,
                location="external_predictions",
                relative_root=prediction_root,
            )
        self.assertEqual(found, [rows.resolve()])

    def test_restricted_namespace_is_rejected_before_preflight_access(self) -> None:
        with self.assertRaisesRegex(Exception, "restricted external namespace"):
            builder.preflight(
                garc_summary=Path(r"C:\pointer_read\field\do-not-open.json"),
                paper_summary=self.public / "missing-paper.json",
                paper_seal=self.public / "missing-seal.json",
                frontend_plan=self.public / "missing-frontend.json",
                output_root=self.root / "unused-output",
            )

    def test_repository_data_and_unc_paths_are_outside_authority_roots(self) -> None:
        with self.assertRaisesRegex(Exception, "outside the public/model authority roots"):
            builder._guard(
                builder.PROJECT_ROOT / "data" / "public-candidate.json",
                label="repository data candidate",
                must_exist=False,
            )
        with self.assertRaisesRegex(Exception, "outside the public/model authority roots"):
            builder._guard(
                Path(r"\\untrusted-host\share\public-looking.json"),
                label="UNC candidate",
                must_exist=False,
            )

    def test_resolved_link_cannot_escape_an_allowed_root(self) -> None:
        safe = self.root / "safe-authority"
        outside = self.root / "outside-authority"
        safe.mkdir()
        outside.mkdir()
        link = safe / "linked-directory"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlink unavailable: {error}")
        with (
            mock.patch.object(builder, "_allowed_roots", return_value=(safe,)),
            self.assertRaisesRegex(Exception, "outside the public/model authority roots"),
        ):
            builder._guard_directory(link, label="linked public authority")

    def test_source_has_no_directory_enumeration_or_dataset_cli(self) -> None:
        source = Path(builder.__file__).read_text(encoding="utf-8")
        for forbidden in (".iterdir(", ".glob(", ".rglob(", "os.walk("):
            self.assertNotIn(forbidden, source)
        parser_help = builder._parser().format_help().casefold()
        for forbidden_option in (
            "--field",
            "--manifest",
            "--labels",
            "--image",
            "--crop",
            "--range",
        ):
            self.assertNotIn(forbidden_option, parser_help)


if __name__ == "__main__":
    unittest.main()
