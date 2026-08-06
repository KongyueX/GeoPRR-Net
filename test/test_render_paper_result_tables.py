from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.render_paper_result_tables import (
    OUTPUT_FILES,
    SOURCE_FILES,
    SOURCE_PROTOCOL,
    TableRenderError,
    canonical_sha256,
    render,
    sha256_file,
    validate_source,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RenderPaperResultTablesTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        source = root / "paper_final_results_v1"
        source.mkdir()
        main = [
            {
                "table": "strict_joint_412_main",
                "method": method,
                "samples": 412,
                "groups": 19,
                "coverage": 0.99,
                "full_denominator_nmae_failure_penalty_1": 0.1 + index / 100,
                "full_denominator_p95_normalized_error": 0.2,
                "conditional_nmae": 0.09,
                "macro_group_nmae": 0.11,
                "claim_eligible": True,
                "note": "fixture",
            }
            for index, method in enumerate(
                ("garc", "vdn_official200", "under_pressure_official")
            )
        ]
        sensitivity = [
            {
                "table": "sensitivity_only",
                "family": "public_1080_external_full_auto",
                "method": "under_pressure_official",
                "samples": 1080,
                "groups": 50,
                "coverage": 0.25,
                "metric_name": "full_denominator_nmae_failure_penalty_1",
                "metric_value": 0.8,
                "strict_main_table_eligible": False,
                "reason": "fixture",
            }
        ]
        component = [
            {
                "table": "component_oof",
                "method": "enhanced_v5_geometry_head",
                "samples": 4380,
                "groups": 197,
                "coverage": 0.998,
                "full_denominator_nmae": 0.014,
                "p95_absolute_progress_error": 0.033,
                "claim_eligible": True,
                "note": "fixture",
            }
        ]
        bootstrap = [
            {
                "evidence_role": "strict_joint_412_main",
                "reference": "garc",
                "comparator": "vdn_official200",
                "nmae_difference_definition": "comparator_minus_garc",
                "positive_nmae_difference_favors": "garc",
                "nmae_difference": 0.05,
                "nmae_difference_group_bootstrap_95ci": [0.01, 0.09],
                "garc_relative_nmae_reduction_percent": 33.3,
                "coverage_difference_definition": "garc_minus_comparator",
                "coverage_difference": 0.01,
                "coverage_difference_group_bootstrap_95ci": [-0.01, 0.03],
                "physical_groups": 19,
                "iterations": 10000,
                "seed": 20260805,
            }
        ]
        summary = {
            "schema_version": 1,
            "protocol": SOURCE_PROTOCOL,
            "status": "complete",
            "cohort": {"samples": 412, "groups": 19},
            "main_table": main,
            "sensitivity_table": sensitivity,
            "component_table": component,
            "paired_group_bootstrap": bootstrap,
        }
        (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        _write_csv(source / "main_table.csv", main)
        _write_csv(source / "sensitivity_table.csv", sensitivity)
        _write_csv(source / "component_table.csv", component)
        flattened = []
        for row in bootstrap:
            item = {key: value for key, value in row.items() if not isinstance(value, list)}
            item.update(
                {
                    "nmae_difference_ci95_low": row["nmae_difference_group_bootstrap_95ci"][0],
                    "nmae_difference_ci95_high": row["nmae_difference_group_bootstrap_95ci"][1],
                    "coverage_difference_ci95_low": row["coverage_difference_group_bootstrap_95ci"][0],
                    "coverage_difference_ci95_high": row["coverage_difference_group_bootstrap_95ci"][1],
                }
            )
            flattened.append(item)
        _write_csv(source / "paired_group_bootstrap.csv", flattened)
        artifacts = {
            name: {
                "sha256": sha256_file(source / filename),
                "bytes": (source / filename).stat().st_size,
            }
            for name, filename in SOURCE_FILES.items()
        }
        seal = {
            "schema_version": 1,
            "protocol": SOURCE_PROTOCOL,
            "status": "sealed",
            "artifacts": artifacts,
            "bundle_sha256": canonical_sha256(artifacts),
        }
        (source / "seal.json").write_text(json.dumps(seal), encoding="utf-8")
        return source

    def test_renders_four_authenticated_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._fixture(root)
            output = root / "generated_tables"
            manifest_path = render(source, output)
            self.assertTrue(manifest_path.is_file())
            for filename in OUTPUT_FILES.values():
                self.assertTrue((output / filename).is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["audit"]["source_seal_verified"])
            self.assertIn("GARC (ours)", (output / "strict-main-table.tex").read_text(encoding="utf-8"))
            self.assertIn("[0.010000, 0.090000]", (output / "paired-bootstrap-table.tex").read_text(encoding="utf-8"))
            with self.assertRaisesRegex(TableRenderError, "refusing to overwrite"):
                render(source, output)

    def test_missing_or_tampered_source_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._fixture(root)
            (source / "main_table.csv").unlink()
            with self.assertRaisesRegex(TableRenderError, "absent"):
                validate_source(source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._fixture(root)
            with (source / "component_table.csv").open("a", encoding="utf-8") as handle:
                handle.write("tamper\n")
            with self.assertRaisesRegex(TableRenderError, "hash drift"):
                validate_source(source)


if __name__ == "__main__":
    unittest.main()
