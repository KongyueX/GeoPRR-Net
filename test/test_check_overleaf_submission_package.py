from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.check_overleaf_submission_package import (
    GENERATED_FILES,
    GENERATED_PROTOCOL,
    canonical_sha256,
    inspect_package,
    sha256_file,
)


class CheckOverleafSubmissionPackageTest(unittest.TestCase):
    def _fixture(self, root: Path, *, missing_bib: bool = False, todo: bool = False) -> Path:
        package = root / "official"
        (package / "Definitions").mkdir(parents=True)
        (package / "figures").mkdir()
        generated = package / "generated_tables"
        generated.mkdir()
        (package / "Definitions/mdpi.cls").write_text("fixture", encoding="utf-8")
        (package / "Definitions/journalnames.tex").write_text("fixture", encoding="utf-8")
        (package / "figures/overview.png").write_bytes(b"png-fixture")
        for filename in GENERATED_FILES:
            (generated / filename).write_text("% generated\n", encoding="utf-8")
        artifacts = {
            name: {
                "path": filename,
                "sha256": sha256_file(generated / filename),
                "bytes": (generated / filename).stat().st_size,
            }
            for name, filename in zip(
                ("main_table", "paired_group_bootstrap", "component_table", "sensitivity_table"),
                GENERATED_FILES,
            )
        }
        manifest = {
            "protocol": GENERATED_PROTOCOL,
            "status": "complete",
            "artifacts": artifacts,
        }
        (generated / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        sealed = {
            **artifacts,
            "manifest": {
                "path": "manifest.json",
                "sha256": sha256_file(generated / "manifest.json"),
                "bytes": (generated / "manifest.json").stat().st_size,
            },
        }
        (generated / "seal.json").write_text(
            json.dumps(
                {
                    "protocol": GENERATED_PROTOCOL,
                    "status": "sealed",
                    "artifacts": sealed,
                    "bundle_sha256": canonical_sha256(sealed),
                }
            ),
            encoding="utf-8",
        )
        inputs = "\n".join(
            rf"\input{{generated_tables/{Path(filename).stem}}}" for filename in GENERATED_FILES
        )
        todo_text = r"\todo{replace me}" if todo else ""
        manuscript = rf"""\documentclass[electronics,article,submit,moreauthors]{{Definitions/mdpi}}
\newcommand{{\todo}}[1]{{#1}}
\abstract{{A compact valid abstract.}}
\keyword{{meter}}
\begin{{document}}
\cite{{known}}
\includegraphics{{figures/overview.png}}
{inputs}
{todo_text}
\end{{document}}
"""
        (package / "manuscript.tex").write_text(manuscript, encoding="utf-8")
        bib_key = "different" if missing_bib else "known"
        (package / "references.bib").write_text(
            f"@article{{{bib_key}, title={{Fixture}}, author={{A}}, year={{2026}}}}\n",
            encoding="utf-8",
        )
        return package

    def test_ready_inventory_does_not_build_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._fixture(Path(temporary))
            result = inspect_package(package)
            self.assertTrue(result["ready"])
            self.assertEqual(result["status"], "ready")
            self.assertFalse(result["archive_created"])
            self.assertFalse(result["manuscript_modified"])
            self.assertGreater(len(result["inventory"]), 8)

    def test_reports_todo_missing_bib_and_generated_table_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._fixture(Path(temporary), missing_bib=True, todo=True)
            (package / "generated_tables/sensitivity-table.tex").unlink()
            result = inspect_package(package)
            self.assertFalse(result["ready"])
            joined = "\n".join(result["issues"])
            self.assertIn("missing generated table artifacts", joined)
            self.assertIn("unresolved TODO", joined)
            self.assertIn("missing BibTeX entries: known", joined)

    def test_accepts_exact_v2_result_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = self._fixture(root)
            results = root / "paper_final_results_v2"
            results.mkdir()
            (results / "summary.json").write_text('{"status":"complete"}\n', encoding="utf-8")
            (results / "seal.json").write_text('{"status":"sealed"}\n', encoding="utf-8")

            generated = package / "generated_tables"
            manifest_path = generated / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source"] = {
                "path": str(results.resolve()),
                "summary_sha256": sha256_file(results / "summary.json"),
                "seal_sha256": sha256_file(results / "seal.json"),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            seal_path = generated / "seal.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            seal["artifacts"]["manifest"] = {
                "path": "manifest.json",
                "sha256": sha256_file(manifest_path),
                "bytes": manifest_path.stat().st_size,
            }
            seal["bundle_sha256"] = canonical_sha256(seal["artifacts"])
            seal_path.write_text(json.dumps(seal), encoding="utf-8")

            result = inspect_package(package, results)
            self.assertTrue(result["ready"], result["issues"])
            self.assertEqual(
                Path(result["expected_paper_results_root"]), results.resolve()
            )

            wrong = root / "different_results_v2"
            mismatch = inspect_package(package, wrong)
            self.assertFalse(mismatch["ready"])
            self.assertIn(
                "generated tables are not bound to the expected v2 result root",
                "\n".join(mismatch["issues"]),
            )


if __name__ == "__main__":
    unittest.main()
