from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import assemble_paper_results as paper


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": paper.sha256_file(path)}


def official_metrics(vector: paper.MethodVector) -> dict[str, Any]:
    metric = paper.vector_metrics(vector)
    return {
        "samples": metric["samples"],
        "groups": metric["groups"],
        "successes": metric["successful"],
        "failures": metric["samples"] - metric["successful"],
        "native_coverage": metric["coverage"],
        "full_denominator_nmae": metric["full_denominator_nmae_failure_penalty_1"],
        "full_denominator_nmae_p95": metric["full_denominator_p95_normalized_error"],
        "conditional_nmae": metric["conditional_nmae"],
        "group_macro_full_denominator_nmae": metric["macro_group_nmae"],
    }


class SyntheticEvidence:
    def __init__(self, root: Path):
        self.root = root
        self.joint_rows = [
            {"sample_id": f"s{index}", "group_id": "g0" if index < 3 else "g1"}
            for index in range(6)
        ]
        self.full_rows = [
            *self.joint_rows,
            {"sample_id": "s6", "group_id": "g2"},
            {"sample_id": "s7", "group_id": "g2"},
        ]
        self.truth = {
            row["sample_id"]: (row["group_id"], float(10 * (index + 1)), 0.0, 100.0)
            for index, row in enumerate(self.full_rows)
        }
        self.paths, self.expected = self._build()

    def _build(self) -> tuple[paper.EvidencePaths, paper.ExpectedEvidence]:
        public_source = self.root / "public" / "syncg_train.jsonl"
        write_jsonl(
            public_source,
            [
                {
                    "sample_id": sample_id,
                    "group_id": group,
                    "ground_truth": reading,
                    "scale_start": start,
                    "scale_end": end,
                }
                for sample_id, (group, reading, start, end) in self.truth.items()
            ],
        )
        public_source_protocol = self.root / "public" / "syncg_train.protocol.json"
        write_json(public_source_protocol, {"protocol": "synthetic_public_source_v1"})
        public_protocol = self.root / "public" / "protocol.json"
        write_json(
            public_protocol,
            {
                "protocol": "synthetic_automatic_range_public_v1",
                "source_bindings": {
                    "syncg_train_manifest": binding(public_source),
                    "syncg_train_manifest_protocol": binding(public_source_protocol),
                },
            },
        )
        partition = self.root / "public" / "independent_validation.label_free.jsonl"
        write_jsonl(partition, self.full_rows)

        cohort_roster = self.root / "cohort" / "joint.label_free.jsonl"
        mapping = self.root / "cohort" / "mapping.label_free.jsonl"
        cohort_summary = self.root / "cohort" / "summary.json"
        write_jsonl(cohort_roster, self.joint_rows)
        write_jsonl(
            mapping,
            [
                {
                    **row,
                    "pepd_seed": 1 + index % 3,
                    "group_unseen_by_progress_checkpoint": True,
                }
                for index, row in enumerate(self.joint_rows)
            ],
        )
        write_json(
            cohort_summary,
            {
                "protocol": "synthetic_joint_cohort_v1",
                "artifacts": {
                    "cohort": binding(cohort_roster),
                    "mapping": binding(mapping),
                },
            },
        )
        expected = paper.ExpectedEvidence(
            joint_samples=6,
            joint_groups=2,
            range_samples=8,
            range_groups=3,
            v5_samples=6,
            v5_groups=2,
            fixed_geometry_unseen_samples=2,
            fixed_geometry_unseen_groups=1,
            fixed_geometry_overlap_samples=6,
            fixed_geometry_overlap_groups=2,
            joint_summary_sha256=paper.sha256_file(cohort_summary),
            cohort_sha256=paper.sha256_file(cohort_roster),
            mapping_sha256=paper.sha256_file(mapping),
        )

        external_protocol_path = self.root / "external" / "protocol.json"
        write_json(
            external_protocol_path,
            {
                "protocol": "garc_external_progress_412_comparison_v1",
                "cohort": {
                    "summary": binding(cohort_summary),
                    "label_free_roster": binding(cohort_roster),
                    "progress_mapping": binding(mapping),
                    "samples": expected.joint_samples,
                    "physical_groups": expected.joint_groups,
                },
                "public_protocol": binding(public_protocol),
            },
        )
        preflight = self.root / "external" / "preflight.json"
        write_json(preflight, {"status": "passed_with_transformer_sensitivity_only"})
        plan = self.root / "garc" / "plan.json"
        calibration = self.root / "garc" / "calibration.json"
        write_json(plan, {"status": "frozen"})
        write_json(calibration, {"status": "frozen"})

        garc_offsets = [1, 1, 2, 2, 3, 3]
        handoff_rows: list[dict[str, Any]] = []
        for index, row in enumerate(self.joint_rows):
            target = self.truth[row["sample_id"]][1]
            handoff_rows.append(
                {
                    **row,
                    "range_accepted": True,
                    "predicted_scale_start": 0.0,
                    "predicted_scale_end": 100.0,
                    "garc_full_status": True,
                    "garc_predicted_reading": target + garc_offsets[index],
                    "garc_all_components_group_unseen": True,
                }
            )
        handoff_root = self.root / "garc" / "handoff"
        handoff_rows_path = handoff_root / "handoff.label_free.jsonl"
        write_jsonl(handoff_rows_path, handoff_rows)
        handoff_summary_path = handoff_root / "summary.json"
        handoff_summary = {
            "protocol": "garc_external_progress_412_handoff_v1",
            "status": "label_free_handoff_sealed",
            "frozen_protocol": binding(external_protocol_path),
            "preflight": binding(preflight),
            "cohort": {
                "samples": expected.joint_samples,
                "physical_groups": expected.joint_groups,
            },
            "garc": {
                "all_412_progress_geometry_head_and_backbone_group_unseen": True,
            },
            "method_roles": {
                "vdn_official200": "strict_grouped_oof_external_progress_comparator",
                "original_transformer": "fixed_checkpoint_non_oof_sensitivity_only",
            },
            "artifacts": {
                "rows": {"path": handoff_rows_path.name, "sha256": paper.sha256_file(handoff_rows_path)}
            },
            "audit": {"restricted_namespace_images_opened": 0},
        }
        write_json(handoff_summary_path, handoff_summary)
        handoff_seal_path = handoff_root / "seal.json"
        handoff_bundle = paper.canonical_sha256(
            {
                "protocol_sha256": paper.sha256_file(external_protocol_path),
                "preflight_sha256": paper.sha256_file(preflight),
                "summary_sha256": paper.sha256_file(handoff_summary_path),
                "rows_sha256": paper.sha256_file(handoff_rows_path),
            }
        )
        write_json(
            handoff_seal_path,
            {
                "protocol": "garc_external_progress_412_handoff_v1",
                "status": "sealed",
                "summary_sha256": paper.sha256_file(handoff_summary_path),
                "rows_sha256": paper.sha256_file(handoff_rows_path),
                "bundle_sha256": handoff_bundle,
            },
        )

        garc_vector = paper._vector_from_readings("garc", handoff_rows, self.truth)
        garc_metrics = paper.vector_metrics(garc_vector)
        garc_prediction_root = self.root / "garc" / "validation_predictions"
        garc_prediction_rows_path = garc_prediction_root / "predictions.label_free.jsonl"
        write_jsonl(garc_prediction_rows_path, handoff_rows)
        garc_runtime_bundle_path = garc_prediction_root / "full_auto_bundle.json"
        write_json(garc_runtime_bundle_path, {"status": "frozen"})
        garc_prediction_summary_path = garc_prediction_root / "summary.json"
        write_json(
            garc_prediction_summary_path,
            {
                "status": "predictions_sealed",
                "artifacts": {
                    "predictions": {
                        "path": garc_prediction_rows_path.name,
                        "sha256": paper.sha256_file(garc_prediction_rows_path),
                    },
                    "bundle": {
                        "path": garc_runtime_bundle_path.name,
                        "sha256": paper.sha256_file(garc_runtime_bundle_path),
                    },
                },
            },
        )
        write_json(
            garc_prediction_root / "seal.json",
            {
                "summary_sha256": paper.sha256_file(garc_prediction_summary_path),
                "predictions_sha256": paper.sha256_file(garc_prediction_rows_path),
                "bundle_sha256": paper.sha256_file(garc_runtime_bundle_path),
                "plan_sha256": paper.sha256_file(plan),
            },
        )
        garc_validation_path = self.root / "garc" / "validation.json"
        garc_validation = {
            "protocol": "garc_full_auto_public_validation_v1",
            "status": "independent_validation_complete",
            "mode": "formal",
            "claim_eligible": True,
            "parent_plan": binding(plan),
            "parent_protocol": binding(public_protocol),
            "frozen_calibration": binding(calibration),
            "validation_predictions": {
                "path": str(garc_prediction_root),
                "summary_sha256": paper.sha256_file(garc_prediction_summary_path),
                "predictions_sha256": paper.sha256_file(garc_prediction_rows_path),
            },
            "evidence_eligibility": {
                "joint_oof_end_to_end_claim": True,
                "full_1080_end_to_end_claim": False,
                "ocr_and_fixed_fold_geometry_sensitivity": True,
            },
            "overlap_audit": {
                "jointly_unseen_samples": expected.joint_samples,
                "jointly_unseen_groups": expected.joint_groups,
                "joint_cohort_complete": True,
                "joint_mapping_sha256": expected.joint_summary_sha256,
                "primary_fixed_fold_sensitivity": {
                    "geometry_unseen_samples": expected.fixed_geometry_unseen_samples,
                    "geometry_unseen_groups": expected.fixed_geometry_unseen_groups,
                    "geometry_fit_overlap_samples": expected.fixed_geometry_overlap_samples,
                    "geometry_fit_overlap_groups": expected.fixed_geometry_overlap_groups,
                    "full_1080_all_component_unseen": False,
                },
            },
            "metrics": {
                "joint_oof_end_to_end_frozen_acceptance": {
                    "samples": expected.joint_samples,
                    "groups": expected.joint_groups,
                    "reading_nmae_full_denominator_failure_penalty_1": garc_metrics[
                        "full_denominator_nmae_failure_penalty_1"
                    ],
                },
                "joint_oof_range_frozen_acceptance": {
                    "samples": expected.joint_samples,
                    "groups": expected.joint_groups,
                },
                "range_component_frozen_acceptance": {
                    "samples": expected.range_samples,
                    "groups": expected.range_groups,
                    "coverage": 0.75,
                    "pair_rounded_exact_full_denominator": 0.5,
                },
            },
            "audit": {"restricted_namespace_images_opened": 0},
        }
        write_json(garc_validation_path, garc_validation)
        garc_summary_path = self.root / "garc" / "summary.json"
        write_json(
            garc_summary_path,
            {
                "protocol": "garc_full_auto_public_event_chain_summary_v1",
                "status": "complete",
                "selected": {
                    "plan": str(plan),
                    "plan_sha256": paper.sha256_file(plan),
                    "calibration": str(calibration),
                    "calibration_sha256": paper.sha256_file(calibration),
                },
                "validation": {
                    "path": str(garc_validation_path),
                    "sha256": paper.sha256_file(garc_validation_path),
                    "paper_claim_allowed": {
                        "full_1080_all_component_unseen": False,
                        "joint_412_all_component_oof_end_to_end": True,
                    },
                },
                "external_comparison_handoff": {
                    "handoff": {
                        "root": str(handoff_root),
                        "summary_sha256": paper.sha256_file(handoff_summary_path),
                        "seal_sha256": paper.sha256_file(handoff_seal_path),
                    }
                },
                "audit": {"restricted_namespace_images_opened": 0},
            },
        )

        external_vectors: dict[str, paper.MethodVector] = {}
        external_sources: dict[str, dict[str, Any]] = {}
        external_prediction_rows: dict[str, list[dict[str, Any]]] = {}
        for method, offsets in {
            "vdn_official200": [2, 2, 3, 3, 4, 4],
            "original_transformer": [4, 4, 5, 5, 6, 6],
        }.items():
            rows = []
            for index, row in enumerate(self.joint_rows):
                target = self.truth[row["sample_id"]][1]
                rows.append(
                    {
                        **row,
                        "method": method,
                        "status": True,
                        "prediction_progress": (target + offsets[index]) / 100.0,
                        "strict_oof_eligible": method == "vdn_official200",
                    }
                )
            external_prediction_rows[method] = rows
            root = self.root / "external" / method
            rows_path = root / "predictions.label_free.jsonl"
            write_jsonl(rows_path, rows)
            prediction_summary_path = root / "summary.json"
            prediction_summary = {
                "protocol": "garc_external_progress_412_predictions_v1",
                "status": "predictions_sealed",
                "method": method,
                "strict_oof_eligible": method == "vdn_official200",
                "frozen_protocol": binding(external_protocol_path),
                "preflight": binding(preflight),
                "handoff": {"bundle_sha256": handoff_bundle},
                "artifacts": {
                    "predictions": {"path": rows_path.name, "sha256": paper.sha256_file(rows_path)}
                },
            }
            write_json(prediction_summary_path, prediction_summary)
            prediction_seal_path = root / "seal.json"
            bundle_sha = paper.canonical_sha256(
                {
                    "method": method,
                    "protocol_sha256": paper.sha256_file(external_protocol_path),
                    "handoff_bundle_sha256": handoff_bundle,
                    "summary_sha256": paper.sha256_file(prediction_summary_path),
                    "predictions_sha256": paper.sha256_file(rows_path),
                }
            )
            write_json(
                prediction_seal_path,
                {
                    "protocol": "garc_external_progress_412_predictions_v1",
                    "status": "sealed",
                    "method": method,
                    "summary_sha256": paper.sha256_file(prediction_summary_path),
                    "predictions_sha256": paper.sha256_file(rows_path),
                    "handoff_bundle_sha256": handoff_bundle,
                    "bundle_sha256": bundle_sha,
                },
            )
            external_sources[method] = {
                "path": str(root),
                "summary_sha256": paper.sha256_file(prediction_summary_path),
                "bundle_sha256": bundle_sha,
            }
            external_vectors[method] = paper._vector_from_readings(
                method, handoff_rows, self.truth, rows
            )

        external_score_path = self.root / "external" / "comparison.json"
        write_json(
            external_score_path,
            {
                "protocol": "garc_external_progress_412_score_v1",
                "status": "complete",
                "frozen_protocol": binding(external_protocol_path),
                "preflight": binding(preflight),
                "handoff": {
                    "path": str(handoff_root),
                    "summary_sha256": paper.sha256_file(handoff_summary_path),
                    "bundle_sha256": handoff_bundle,
                },
                "metrics": {
                    "garc": self._external_metric(garc_metrics),
                    **{
                        method: self._external_metric(paper.vector_metrics(vector))
                        for method, vector in external_vectors.items()
                    },
                },
                "claim_eligibility": {
                    "strict_formal_table": ["garc", "vdn_official200"],
                    "fixed_checkpoint_sensitivity_table": ["original_transformer"],
                    "original_transformer_strict_oof_claim": False,
                    "vdn_strict_progress_oof_claim": True,
                    "garc_all_component_joint_oof_claim": True,
                },
                "sources": external_sources,
                "audit": {
                    "restricted_namespace_images_opened": 0,
                    "field_samples_used": 0,
                    "test_samples_used": 0,
                    "sealed_samples_used": 0,
                    "confirmatory_samples_used": 0,
                },
            },
        )

        under_protocol = self.root / "under_pressure" / "protocol.json"
        write_json(
            under_protocol,
            {
                "protocol": "under_pressure_official_public_independent_validation_v1",
                "status": "frozen_before_formal_inference",
                "samples": expected.range_samples,
                "groups": expected.range_groups,
                "parent_protocol": binding(public_protocol),
                "partition_manifest": binding(partition),
            },
        )
        up_root = self.root / "under_pressure" / "predictions"
        up_rows: list[dict[str, Any]] = []
        for index, row in enumerate(self.full_rows):
            target = self.truth[row["sample_id"]][1]
            success = index != 5
            up_rows.append(
                {
                    **row,
                    "status": success,
                    "predicted_reading": target + 3.0 if success else None,
                }
            )
        up_rows_path = up_root / "predictions.label_free.jsonl"
        write_jsonl(up_rows_path, up_rows)
        up_prediction_summary_path = up_root / "summary.json"
        write_json(
            up_prediction_summary_path,
            {
                "protocol": "under_pressure_official_public_predictions_v1",
                "status": "predictions_sealed",
                "mode": "formal",
                "claim_eligible": True,
                "samples": expected.range_samples,
                "groups": expected.range_groups,
                "parent_protocol": binding(under_protocol),
                "artifacts": {
                    "predictions": {"path": up_rows_path.name, "sha256": paper.sha256_file(up_rows_path)}
                },
                "audit": {
                    "field_images_opened": 0,
                    "test_images_opened": 0,
                    "sealed_images_opened": 0,
                },
            },
        )
        up_prediction_seal_path = up_root / "seal.json"
        up_prediction_seal = {
            "protocol": "under_pressure_official_public_predictions_v1",
            "status": "sealed_before_labels",
            "parent_protocol_sha256": paper.sha256_file(under_protocol),
            "summary_sha256": paper.sha256_file(up_prediction_summary_path),
            "predictions_sha256": paper.sha256_file(up_rows_path),
            "samples": expected.range_samples,
            "groups": expected.range_groups,
        }
        write_json(up_prediction_seal_path, up_prediction_seal)
        full_vector = self._under_vector(up_rows, [row["sample_id"] for row in self.full_rows])
        up1080_score_path = self.root / "under_pressure" / "score.json"
        write_json(
            up1080_score_path,
            {
                "protocol": "under_pressure_official_public_score_v1",
                "status": "formal_public_independent_validation_complete",
                "claim_eligible": True,
                "prediction_bundle": {
                    "path": str(up_root),
                    "seal_sha256": paper.canonical_sha256(up_prediction_seal),
                    "predictions_sha256": paper.sha256_file(up_rows_path),
                },
                "metrics": official_metrics(full_vector),
                "audit": {"field_test_sealed_images_opened": 0},
            },
        )
        write_json(
            up1080_score_path.with_name(up1080_score_path.name + ".seal.json"),
            {
                "protocol": "under_pressure_official_public_score_v1",
                "score_sha256": paper.sha256_file(up1080_score_path),
            },
        )
        joint_vector = self._under_vector(up_rows, [row["sample_id"] for row in self.joint_rows])
        up412_score_path = self.root / "under_pressure" / "score_joint.json"
        write_json(
            up412_score_path,
            {
                "protocol": "under_pressure_official_joint_oof_score_v1",
                "status": "formal_joint_oof_score_complete",
                "claim_eligible": True,
                "selection": {
                    "sample_ids_sha256": paper.canonical_sha256(
                        [row["sample_id"] for row in self.joint_rows]
                    ),
                    "joint_summary_path": str(cohort_summary),
                    "joint_summary_sha256": expected.joint_summary_sha256,
                    "cohort_sha256": expected.cohort_sha256,
                    "mapping_sha256": expected.mapping_sha256,
                },
                "parent_prediction_bundle": {
                    "path": str(up_root),
                    "seal_sha256": paper.canonical_sha256(up_prediction_seal),
                    "predictions_sha256": paper.sha256_file(up_rows_path),
                    "full_samples": expected.range_samples,
                    "full_groups": expected.range_groups,
                },
                "metrics": official_metrics(joint_vector),
                "audit": {"field_test_sealed_images_opened": 0},
            },
        )
        write_json(
            up412_score_path.with_name(up412_score_path.name + ".seal.json"),
            {
                "protocol": "under_pressure_official_joint_oof_score_v1",
                "score_sha256": paper.sha256_file(up412_score_path),
                "prediction_seal_sha256": paper.canonical_sha256(up_prediction_seal),
                "joint_summary_sha256": expected.joint_summary_sha256,
            },
        )

        v5_root = self.root / "v5"
        v5_predictions = v5_root / "strict_oof_predictions.jsonl"
        write_jsonl(v5_predictions, [{"sample_id": f"v{index}"} for index in range(6)])
        folds = []
        for seed in (1, 2, 3):
            fold_path = v5_root / f"fold_{seed}.json"
            write_json(fold_path, {"status": "complete", "pepd_seed": seed})
            folds.append(
                {
                    "pepd_seed": seed,
                    "summary": str(fold_path),
                    "summary_sha256": paper.sha256_file(fold_path),
                }
            )
        v5_strict_path = v5_root / "strict_oof_summary.json"
        v5_strict = {
            "status": "complete",
            "overlap_and_assignment_audit": {
                "eligible_union_samples": expected.v5_samples,
                "eligible_union_groups": expected.v5_groups,
                "all_rows_jointly_unseen_by_pepd_and_head": True,
                "field_samples_read": 0,
                "public_test_samples_read": 0,
            },
            "metrics": {
                "samples": expected.v5_samples,
                "groups": expected.v5_groups,
                "coverage": 1.0,
                "full_denominator_nmae": 0.02,
                "p95_absolute_progress_error": 0.04,
            },
            "folds": folds,
            "artifacts": {
                "strict_oof_predictions_sha256": paper.sha256_file(v5_predictions)
            },
        }
        write_json(v5_strict_path, v5_strict)
        v5_summary_path = v5_root / "summary.json"
        write_json(
            v5_summary_path,
            {
                "protocol": "cagh_v5_enhanced_authoritative_pepd_oof_v1",
                "status": "complete",
                "strict_oof": v5_strict,
                "artifacts": {
                    "strict_oof_summary": str(v5_strict_path),
                    "strict_oof_summary_sha256": paper.sha256_file(v5_strict_path),
                    "strict_oof_predictions": str(v5_predictions),
                    "strict_oof_predictions_sha256": paper.sha256_file(v5_predictions),
                },
                "field_samples_read": 0,
                "public_test_samples_read": 0,
            },
        )
        return (
            paper.EvidencePaths(
                garc_summary=garc_summary_path,
                external_comparison=external_score_path,
                under_pressure_1080_score=up1080_score_path,
                under_pressure_412_score=up412_score_path,
                v5_oof_summary=v5_summary_path,
            ),
            expected,
        )

    @staticmethod
    def _external_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "samples": metric["samples"],
            "physical_groups": metric["groups"],
            "successful": metric["successful"],
            "coverage": metric["coverage"],
            "full_denominator_nmae_failure_penalty_1": metric[
                "full_denominator_nmae_failure_penalty_1"
            ],
            "full_denominator_p95_normalized_error": metric[
                "full_denominator_p95_normalized_error"
            ],
            "macro_group_nmae": metric["macro_group_nmae"],
            "conditional_nmae": metric["conditional_nmae"],
        }

    def _under_vector(self, rows: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> paper.MethodVector:
        return paper._under_pressure_vector(rows, ids, self.truth)


class PaperResultAssemblerTest(unittest.TestCase):
    def test_restricted_namespace_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "field_data" / "summary.json"
            with self.assertRaisesRegex(paper.AssemblyError, "restricted namespace"):
                paper.guard_path(forbidden, "forbidden", must_exist=False)

    def test_preflight_is_not_ready_without_emitting_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = paper.EvidencePaths(
                garc_summary=root / "garc.json",
                external_comparison=root / "external.json",
                under_pressure_1080_score=root / "up1080.json",
                under_pressure_412_score=root / "up412.json",
                v5_oof_summary=root / "v5.json",
            )
            state = paper.readiness(paths)
            self.assertEqual(state["status"], "not_ready")
            self.assertFalse(state["metrics_emitted"])
            self.assertEqual(len(state["missing"]), 5)

    def test_synthetic_full_assembly_and_claim_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvidence(Path(directory))
            output = Path(directory) / "paper_output"
            summary_path = paper.assemble(
                fixture.paths,
                output,
                expected=fixture.expected,
                bootstrap_iterations=200,
                bootstrap_seed=7,
            )
            summary = paper.strict_json(summary_path)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(
                {row["method"] for row in summary["main_table"]},
                {"garc", "vdn_official200", "under_pressure_official"},
            )
            transformer = next(
                row for row in summary["sensitivity_table"] if row["method"] == "original_transformer"
            )
            self.assertFalse(transformer["strict_main_table_eligible"])
            self.assertEqual(len(summary["paired_group_bootstrap"]), 3)
            self.assertEqual(
                summary["code"]["assembler"]["sha256"],
                paper.sha256_file(paper.__file__),
            )
            self.assertTrue((output / "seal.json").is_file())

    def test_tampered_prediction_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvidence(Path(directory))
            row_path = Path(directory) / "external" / "vdn_official200" / "predictions.label_free.jsonl"
            row_path.write_text(row_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(paper.AssemblyError, "row hash drift"):
                paper.assemble(
                    fixture.paths,
                    Path(directory) / "paper_output",
                    expected=fixture.expected,
                    bootstrap_iterations=100,
                )

    def test_transformer_cannot_enter_strict_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = SyntheticEvidence(Path(directory))
            score = paper.strict_json(fixture.paths.external_comparison)
            score["claim_eligibility"]["strict_formal_table"].append("original_transformer")
            write_json(fixture.paths.external_comparison, score)
            with self.assertRaisesRegex(paper.AssemblyError, "strict external table role drift"):
                paper.assemble(
                    fixture.paths,
                    Path(directory) / "paper_output",
                    expected=fixture.expected,
                    bootstrap_iterations=100,
                )


if __name__ == "__main__":
    unittest.main()
